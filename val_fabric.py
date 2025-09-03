import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset
import timm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report
import os
from tqdm import tqdm

# --- only_one.py 와 호환되는 클래스 정의 ---

# 1. 멀티태스크를 위한 커스텀 데이터셋
class FabricMultiTaskDataset(Dataset):
    """
    main.py와 동일한 멀티태스크 데이터셋 클래스.
    이미지, 원단 라벨, 세탁법 라벨을 반환합니다.
    """
    def __init__(self, root, transform=None):
        # ImageFolder를 내부적으로 사용하여 파일 목록과 클래스를 가져옵니다.
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

# 2. 멀티태스크 모델 정의
class MultiTaskConvNeXt(nn.Module):
    """
    only_one.py와 호환되는 멀티태스크 모델 클래스.
    """
    def __init__(self, num_fabric_classes, num_washing_classes, task_type='multitask'):
        super().__init__()
        self.task_type = task_type
        # 모델 구조만 정의하기 위해 pretrained=False로 설정합니다.
        # 실제 가중치는 load_state_dict를 통해 불러옵니다.
        self.backbone = timm.create_model('convnextv2_tiny', pretrained=False, features_only=True)
        feature_dim = self.backbone.feature_info[-1]['num_chs']
        
        if self.task_type in ['multitask', 'fabric']:
            self.fabric_head = nn.Linear(feature_dim, num_fabric_classes)
        
        if self.task_type in ['multitask', 'washing']:
            self.washing_head = nn.Linear(feature_dim, num_washing_classes)

    def forward(self, x):
        features = self.backbone(x)
        # Global Average Pooling
        pooled_features = features[-1].mean(dim=(-1, -2))
        
        fabric_output = None
        if self.task_type in ['multitask', 'fabric']:
            fabric_output = self.fabric_head(pooled_features)

        washing_output = None
        if self.task_type in ['multitask', 'washing']:
            washing_output = self.washing_head(pooled_features)
        
        return fabric_output, washing_output

def evaluate_model():
    # --- 1. 설정 변수 ---
    TASK_TYPE = 'multitask' # 'multitask', 'fabric', 'washing' 중 선택
    MODEL_PATH = 'model_sav/fabric_classifier_model_multitask_v0.7_143.pth' # 예: 'model_sav/fabric_classifier_model_multitask_85.pth'
    
    CROP_SIZE = (224, 224)
    BATCH_SIZE = 32
    DATA_DIR = 'fabric_dataset'
    VALIDATION_SPLIT = 0.2
    
    # --- GPU/CPU 장치 설정 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"평가를 위해 {device} 장치를 사용합니다.")
    print(f"--- 현재 평가 모드: {TASK_TYPE} ---")

    # --- 2. 데이터셋 및 DataLoader 준비 ---
    print("검증 데이터셋을 준비합니다...")
    val_transform = transforms.Compose([
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_val_dataset = FabricMultiTaskDataset(root=DATA_DIR, transform=val_transform)

    train_size = int((1 - VALIDATION_SPLIT) * len(full_val_dataset))
    val_size = len(full_val_dataset) - train_size
    indices = list(range(len(full_val_dataset)))
    _, val_indices = random_split(indices, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    val_dataset = torch.utils.data.Subset(full_val_dataset, val_indices)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    fabric_class_names = full_val_dataset.fabric_classes
    washing_class_names = full_val_dataset.washing_classes
    num_fabric_classes = len(fabric_class_names)
    num_washing_classes = len(washing_class_names)
    print("데이터셋 준비 완료.")

    # --- 3. 학습된 모델 로드 ---
    if not os.path.exists(MODEL_PATH):
        print(f"오류: 모델 파일 '{MODEL_PATH}'을 찾을 수 없습니다.")
        return

    print(f"학습된 모델을 로드합니다: {MODEL_PATH}")
    model = MultiTaskConvNeXt(
        num_fabric_classes=num_fabric_classes, 
        num_washing_classes=num_washing_classes,
        task_type=TASK_TYPE
    )
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except RuntimeError as e:
        print(f"[오류] 모델 가중치를 불러오는 데 실패했습니다. TASK_TYPE과 모델이 일치하는지 확인하세요.")
        print(f"  - 모델: {MODEL_PATH}")
        print(f"  - 설정된 TASK_TYPE: '{TASK_TYPE}'")
        print(f"  - 에러 메시지: {e}")
        return
        
    model = model.to(device)
    model.eval()
    print("모델 로드 완료.")

    # --- 4. 모델 평가 및 예측 수집 ---
    all_fabric_labels, all_fabric_preds = [], []
    all_washing_labels, all_washing_preds = [], []

    print("검증 데이터셋으로 예측을 시작합니다...")
    with torch.no_grad():
        for inputs, fabric_labels, washing_labels in tqdm(val_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            
            fabric_outputs, washing_outputs = model(inputs)
            
            if TASK_TYPE in ['multitask', 'fabric']:
                _, fabric_preds = torch.max(fabric_outputs, 1)
                all_fabric_labels.extend(fabric_labels.cpu().numpy())
                all_fabric_preds.extend(fabric_preds.cpu().numpy())

            if TASK_TYPE in ['multitask', 'washing']:
                _, washing_preds = torch.max(washing_outputs, 1)
                all_washing_labels.extend(washing_labels.cpu().numpy())
                all_washing_preds.extend(washing_preds.cpu().numpy())
    print("예측 완료.")

    # --- 5. 결과 분석 및 출력 ---
    
    # --- 태스크 1: 원단 분류 결과 ---
    if TASK_TYPE in ['multitask', 'fabric']:
        print("="*35)
        print("   Task 1: Fabric Classification   ")
        print("="*35)
        
        print("\n--- 분류 리포트 (원단) ---")
        print(classification_report(all_fabric_labels, all_fabric_preds, target_names=fabric_class_names, digits=4))

        cm_fabric = confusion_matrix(all_fabric_labels, all_fabric_preds, normalize='true')
        class_accuracies_fabric = cm_fabric.diagonal()
        sorted_accuracies_fabric = sorted(zip(fabric_class_names, class_accuracies_fabric), key=lambda x: x[1])

        print("\n--- 클래스별 정확도 (원단, 낮은 순) ---")
        for class_name, acc in sorted_accuracies_fabric:
            print(f"{class_name:<20}: {acc:.4f}")

        plt.figure(figsize=(15, 12))
        sns.heatmap(cm_fabric, annot=True, fmt='.2f', cmap='Blues', xticklabels=fabric_class_names, yticklabels=fabric_class_names)
        plt.title('Normalized Confusion Matrix (Fabric Classification)', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.show()

    # --- 태스크 2: 세탁법 분류 결과 ---
    if TASK_TYPE in ['multitask', 'washing']:
        print("\n\n" + "="*41)
        print("  Task 2: Washing Method Classification  ")
        print("="*41)

        print("\n--- 분류 리포트 (세탁법) ---")
        print(classification_report(all_washing_labels, all_washing_preds, target_names=washing_class_names, digits=4))

        cm_washing = confusion_matrix(all_washing_labels, all_washing_preds, normalize='true')
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_washing, annot=True, fmt='.2f', cmap='Greens', xticklabels=washing_class_names, yticklabels=washing_class_names)
        plt.title('Normalized Confusion Matrix (Washing Method Classification)', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    evaluate_model()
