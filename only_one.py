import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import timm
import numpy as np
from sklearn.utils import class_weight
import os
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- 1. 멀티태스크를 위한 커스텀 데이터셋 ---
class FabricMultiTaskDataset(Dataset):
    def __init__(self, root, transform=None):
        # Use an internal ImageFolder instance to find classes and samples
        self.image_folder = datasets.ImageFolder(root)
        self.samples = self.image_folder.samples
        self.loader = self.image_folder.loader
        self.transform = transform
        
        self.fabric_classes = self.image_folder.classes
        self.fabric_class_to_idx = self.image_folder.class_to_idx
        
        # 세탁법 그룹 정의
        self.washing_classes = ['machine_wash', 'delicate', 'dry_clean']
        self.fabric_to_washing_idx = {
            'Blended': 0, 'Cotton': 0, 'Denim': 0, 'Fleece': 0, 
            'Nylon': 0, 'Polyester': 0, 'Silk': 2, 'Terrycloth': 0, 
            'Viscose': 1, 'Wool': 2
        }
        
        # 원단 인덱스 -> 세탁법 인덱스 매핑 생성
        self.fabric_idx_to_washing_idx = {
            self.fabric_class_to_idx[fabric]: wash_idx 
            for fabric, wash_idx in self.fabric_to_washing_idx.items()
            if fabric in self.fabric_class_to_idx
        }

    def __getitem__(self, index):
        img_path, fabric_idx = self.samples[index]
        img = self.loader(img_path)
        
        washing_idx = self.fabric_idx_to_washing_idx[fabric_idx]
        
        if self.transform:
            img = self.transform(img)
            
        return img, fabric_idx, washing_idx

    def __len__(self):
        return len(self.samples)

# --- 2. 멀티태스크 모델 정의 ---
class MultiTaskConvNeXt(nn.Module):
    def __init__(self, num_fabric_classes, num_washing_classes, task_type='multitask'):
        super().__init__()
        self.task_type = task_type
        # ConvNeXt V2
        self.backbone = timm.create_model('convnextv2_tiny', pretrained=True, features_only=True)
        
        # 모델의 특징 추출 결과 차원 확인
        feature_dim = self.backbone.feature_info[-1]['num_chs']
        
        # 몸통 부분 동결
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 태스크 1: 원단 분류 헤드
        if self.task_type in ['multitask', 'fabric']:
            self.fabric_head = nn.Linear(feature_dim, num_fabric_classes)
        
        # 태스크 2: 세탁법 분류 헤드
        if self.task_type in ['multitask', 'washing']:
            self.washing_head = nn.Linear(feature_dim, num_washing_classes)

    def forward(self, x):
        features = self.backbone(x)
        # Global Average Pooling을 직접 적용
        pooled_features = features[-1].mean(dim=(-1, -2))
        
        fabric_output = None
        if self.task_type in ['multitask', 'fabric']:
            fabric_output = self.fabric_head(pooled_features)

        washing_output = None
        if self.task_type in ['multitask', 'washing']:
            washing_output = self.washing_head(pooled_features)
        
        return fabric_output, washing_output

def main():
    # --- GPU/CPU 장치 설정 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"\n--- {torch.cuda.device_count()}개의 GPU가 감지되었습니다 ---")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print("---------------------------------\n")
    else:
        print("\n--- GPU가 감지되지 않았습니다. CPU로 학습을 진행합니다. ---\n")

    # --- 설정 변수 정의 ---
    TASK_TYPE = 'multitask' # 'multitask', 'fabric', 'washing' 중 선택
    CROP_SIZE = (224, 224)
    BATCH_SIZE = 32
    DATA_DIR = 'fabric_dataset'
    VALIDATION_SPLIT = 0.2
    EPOCHS = 100
    LEARNING_RATE = 1e-3
    FABRIC_LOSS_WEIGHT = 0.2
    WASHING_LOSS_WEIGHT = 0.8
    USE_CLASS_WEIGHTS = True

    print(f"--- 현재 작업 모드: {TASK_TYPE} ---\n")

    # --- 데이터 증강 및 전처리 파이프라인 구축 ---
    train_transform = transforms.Compose([
        transforms.RandomCrop(CROP_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- 데이터셋 로드 및 분할 ---
    print("멀티태스크 데이터셋 로드를 시작합니다...")
    
    full_train_dataset = FabricMultiTaskDataset(root=DATA_DIR, transform=train_transform)
    full_val_dataset = FabricMultiTaskDataset(root=DATA_DIR, transform=val_transform)

    train_size = int((1 - VALIDATION_SPLIT) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    indices = list(range(len(full_train_dataset)))
    train_indices, val_indices = random_split(indices, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_val_dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    fabric_class_names = full_train_dataset.fabric_classes
    washing_class_names = full_train_dataset.washing_classes
    num_fabric_classes = len(fabric_class_names)
    num_washing_classes = len(washing_class_names)
    
    print(f"원단 클래스 종류: {fabric_class_names}")
    print(f"세탁법 클래스 종류: {washing_class_names}\n")

    # --- 클래스 가중치 계산 ---
    fabric_class_weights = None
    washing_class_weights = None
    if USE_CLASS_WEIGHTS:
        print("클래스 가중치를 계산합니다...")
        if TASK_TYPE in ['multitask', 'fabric']:
            train_fabric_labels = [full_train_dataset.samples[i][1] for i in train_indices.indices]
            fabric_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_fabric_labels), y=train_fabric_labels)
            fabric_class_weights = torch.tensor(fabric_weights, dtype=torch.float).to(device)
            print("계산된 원단 클래스 가중치:", fabric_class_weights)
        
        if TASK_TYPE in ['multitask', 'washing']:
            train_washing_labels = [full_train_dataset.fabric_idx_to_washing_idx[full_train_dataset.samples[i][1]] for i in train_indices.indices]
            washing_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_washing_labels), y=train_washing_labels)
            washing_class_weights = torch.tensor(washing_weights, dtype=torch.float).to(device)
            print("계산된 세탁법 클래스 가중치:", washing_class_weights)
        print("")
    else:
        print("클래스 가중치를 사용하지 않습니다.\n")

    # --- 모델 구축 ---
    print(f"{TASK_TYPE} 모드로 모델을 구축합니다...")
    model = MultiTaskConvNeXt(num_fabric_classes, num_washing_classes, task_type=TASK_TYPE)
    model = model.to(device)

    # --- 손실 함수 및 옵티마이저 정의 ---
    criterion_fabric = nn.CrossEntropyLoss(weight=fabric_class_weights) if TASK_TYPE in ['multitask', 'fabric'] else None
    criterion_washing = nn.CrossEntropyLoss(weight=washing_class_weights) if TASK_TYPE in ['multitask', 'washing'] else None
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    # --- 모델 학습 ---
    writer = SummaryWriter(f"logs/fit/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{TASK_TYPE}")
    print(f"TensorBoard 로그는 다음 명령어로 확인할 수 있습니다:\n tensorboard --logdir logs\n")
    
    history = {
        'train_loss': [], 'train_fabric_acc': [], 'train_washing_acc': [],
        'val_loss': [], 'val_fabric_acc': [], 'val_washing_acc': []
    }

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        # --- 훈련 단계 ---
        model.train()
        running_loss = 0.0
        running_fabric_corrects = 0
        running_washing_corrects = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for inputs, fabric_labels, washing_labels in train_pbar:
            inputs, fabric_labels, washing_labels = inputs.to(device), fabric_labels.to(device), washing_labels.to(device)
            
            optimizer.zero_grad()
            
            fabric_outputs, washing_outputs = model(inputs)
            
            loss_fabric, loss_washing = 0, 0
            if TASK_TYPE in ['multitask', 'fabric']:
                loss_fabric = criterion_fabric(fabric_outputs, fabric_labels)
            if TASK_TYPE in ['multitask', 'washing']:
                loss_washing = criterion_washing(washing_outputs, washing_labels)

            if TASK_TYPE == 'multitask':
                total_loss = (loss_fabric * FABRIC_LOSS_WEIGHT) + (loss_washing * WASHING_LOSS_WEIGHT)
            elif TASK_TYPE == 'fabric':
                total_loss = loss_fabric
            else: # 'washing'
                total_loss = loss_washing
            
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item() * inputs.size(0)
            if TASK_TYPE in ['multitask', 'fabric']:
                _, fabric_preds = torch.max(fabric_outputs, 1)
                running_fabric_corrects += torch.sum(fabric_preds == fabric_labels.data)
            if TASK_TYPE in ['multitask', 'washing']:
                _, washing_preds = torch.max(washing_outputs, 1)
                running_washing_corrects += torch.sum(washing_preds == washing_labels.data)
            
            train_pbar.set_postfix({'loss': f'{total_loss.item():.4f}'})

        epoch_loss = running_loss / len(train_dataset)
        history['train_loss'].append(epoch_loss)
        writer.add_scalar('Loss/train', epoch_loss, epoch)

        if TASK_TYPE in ['multitask', 'fabric']:
            epoch_fabric_acc = running_fabric_corrects.double() / len(train_dataset)
            history['train_fabric_acc'].append(epoch_fabric_acc.item())
            writer.add_scalar('Accuracy/train_fabric', epoch_fabric_acc, epoch)
        if TASK_TYPE in ['multitask', 'washing']:
            epoch_washing_acc = running_washing_corrects.double() / len(train_dataset)
            history['train_washing_acc'].append(epoch_washing_acc.item())
            writer.add_scalar('Accuracy/train_washing', epoch_washing_acc, epoch)

        # --- 검증 단계 ---
        model.eval()
        val_loss = 0.0
        val_fabric_corrects = 0
        val_washing_corrects = 0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
        with torch.no_grad():
            for inputs, fabric_labels, washing_labels in val_pbar:
                inputs, fabric_labels, washing_labels = inputs.to(device), fabric_labels.to(device), washing_labels.to(device)

                fabric_outputs, washing_outputs = model(inputs)
                
                loss_fabric, loss_washing = 0, 0
                if TASK_TYPE in ['multitask', 'fabric']:
                    loss_fabric = criterion_fabric(fabric_outputs, fabric_labels)
                if TASK_TYPE in ['multitask', 'washing']:
                    loss_washing = criterion_washing(washing_outputs, washing_labels)

                if TASK_TYPE == 'multitask':
                    total_loss = (loss_fabric * FABRIC_LOSS_WEIGHT) + (loss_washing * WASHING_LOSS_WEIGHT)
                elif TASK_TYPE == 'fabric':
                    total_loss = loss_fabric
                else: # 'washing'
                    total_loss = loss_washing
                
                val_loss += total_loss.item() * inputs.size(0)
                if TASK_TYPE in ['multitask', 'fabric']:
                    _, fabric_preds = torch.max(fabric_outputs, 1)
                    val_fabric_corrects += torch.sum(fabric_preds == fabric_labels.data)
                if TASK_TYPE in ['multitask', 'washing']:
                    _, washing_preds = torch.max(washing_outputs, 1)
                    val_washing_corrects += torch.sum(washing_preds == washing_labels.data)
                
                val_pbar.set_postfix({'loss': f'{total_loss.item():.4f}'})

        val_epoch_loss = val_loss / len(val_dataset)
        history['val_loss'].append(val_epoch_loss)
        writer.add_scalar('Loss/val', val_epoch_loss, epoch)

        if TASK_TYPE in ['multitask', 'fabric']:
            val_epoch_fabric_acc = val_fabric_corrects.double() / len(val_dataset)
            history['val_fabric_acc'].append(val_epoch_fabric_acc.item())
            writer.add_scalar('Accuracy/val_fabric', val_epoch_fabric_acc, epoch)
        if TASK_TYPE in ['multitask', 'washing']:
            val_epoch_washing_acc = val_washing_corrects.double() / len(val_dataset)
            history['val_washing_acc'].append(val_epoch_washing_acc.item())
            writer.add_scalar('Accuracy/val_washing', val_epoch_washing_acc, epoch)
        
        print(f"Epoch {epoch+1}/{EPOCHS} -> Train Loss: {epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f}")
        if TASK_TYPE in ['multitask', 'fabric']:
            print(f"  [Fabric]   Train Acc: {epoch_fabric_acc:.4f} | Val Acc: {val_epoch_fabric_acc:.4f}")
        if TASK_TYPE in ['multitask', 'washing']:
            print(f"  [Washing]  Train Acc: {epoch_washing_acc:.4f} | Val Acc: {val_epoch_washing_acc:.4f}")

        # --- 최고 성능 모델 저장 ---
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            save_dir = 'model_sav'
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f'fabric_classifier_model_{TASK_TYPE}_{epoch+1}.pth')
            torch.save(model.state_dict(), model_path)
            print(f"  ** 최고 성능 모델 저장됨 (Epoch {epoch+1}, Val Loss: {best_val_loss:.4f}) -> {model_path} **")

    writer.close()
    print("\n모델 학습이 완료되었습니다.")

    # --- 결과 시각화 ---
    epochs_range = range(EPOCHS)
    num_plots = 1 + (1 if TASK_TYPE in ['multitask', 'fabric'] else 0) + (1 if TASK_TYPE in ['multitask', 'washing'] else 0)
    plt.figure(figsize=(6 * num_plots, 5))

    plot_idx = 1
    plt.subplot(1, num_plots, plot_idx)
    plt.plot(epochs_range, history['train_loss'], label='Training Loss')
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Total Loss')
    plot_idx += 1

    if TASK_TYPE in ['multitask', 'fabric']:
        plt.subplot(1, num_plots, plot_idx)
        plt.plot(epochs_range, history['train_fabric_acc'], label='Train Fabric Acc')
        plt.plot(epochs_range, history['val_fabric_acc'], label='Val Fabric Acc')
        plt.legend(loc='lower right')
        plt.title('Fabric Classification Accuracy')
        plot_idx += 1

    if TASK_TYPE in ['multitask', 'washing']:
        plt.subplot(1, num_plots, plot_idx)
        plt.plot(epochs_range, history['train_washing_acc'], label='Train Washing Acc')
        plt.plot(epochs_range, history['val_washing_acc'], label='Val Washing Acc')
        plt.legend(loc='lower right')
        plt.title('Washing Method Classification Accuracy')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
