import os
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.models import convnext_tiny
from sklearn.metrics import classification_report
from torchvision.datasets import ImageFolder
from tensorboardX import SummaryWriter

#########################################################################
# 모델 저장 함수
#########################################################################
def save_model(model, model_name, epoch, save_dir="model/"):
    os.makedirs(save_dir, exist_ok=True)
    save_path_weight = os.path.join(save_dir, f"{model_name}_weight_epoch{epoch}.pth")
    save_path_full = os.path.join(save_dir, f"{model_name}_full_epoch{epoch}.pth")
    torch.save(model.state_dict(), save_path_weight)
    torch.save(model, save_path_full)
    print(f"model saved at epoch {epoch} to: {save_path_weight}")


#########################################################################
# 모델 평가 함수
#########################################################################
def eval_model(model, epoch, test_loader):
    model.eval()

    all_preds = []
    all_labels = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad() :
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 모델 추론
            outputs = model(images)                 # (batch_size, num_classes)
            preds = torch.argmax(outputs, dim=1)    # 가장 확률 높은 클래스로 예측

            # 결과 수집 (CPU로 옮기기)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 병합된 라벨 기준으로 리포트 생성
    report = classification_report(all_labels, all_preds)
    print("="*50)
    print(f"epoch : {epoch}번째 모델 평가")
    print(report)


def main():
    CROP_SIZE = (224, 224)
    BATCH_SIZE = 32
    OUT_FEATURES = 24
    EPOCHS = 30

    ##################################
    # 데이터 불러오기 & 전처리
    ##################################
    transforms_train = transforms.Compose([
        transforms.RandomCrop(CROP_SIZE),       # Crop 사용
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transforms_test = transforms.Compose([
        transforms.CenterCrop(CROP_SIZE),       # Crop 사용
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_raw = ImageFolder("./dataset/train", transform=transforms_train)
    test_raw = ImageFolder("./dataset/test", transform=transforms_test)

    train_dataloader = DataLoader(train_raw, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_raw, batch_size=BATCH_SIZE, shuffle=False)


    ##################################
    # 모델 불러오기
    ##################################
    model = convnext_tiny(pretrained=True)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, OUT_FEATURES, bias=True)


    ##################################
    # 모델 학습
    ##################################
    writer = SummaryWriter()
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    step = 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    for epoch in range(EPOCHS) :
        for data, labels in tqdm.tqdm(train_dataloader) :
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
            
            writer.add_scalar("Loss/train", loss.item(), step)
            step += 1
        
        print(f"{epoch+1} loss ::: {loss.item()}")

        if (epoch + 1) in [10, 20, 30]:
            save_model(model, "convNeXtTiny_re_crop", epoch+1)
            eval_model(model, epoch+1, test_dataloader)



if __name__ == '__main__':
    main()