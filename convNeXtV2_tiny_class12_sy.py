import timm
import os
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from sklearn.metrics import classification_report, accuracy_score, f1_score
from torchvision.datasets import ImageFolder
from tensorboardX import SummaryWriter
from PIL import Image

AUGMENTATIONS = {
    "random_crop": transforms.RandomCrop(224),
    "horizontal_flip": transforms.RandomHorizontalFlip(p=0.5),
    "rotation": transforms.RandomRotation(10)
}

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    all_preds = []
    all_labels = []

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

    # 평가 지표 계산
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    report = classification_report(all_labels, all_preds)

    metrics = {
        "epoch": epoch,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "classification_report": report
    }

    return metrics


#########################################################################
# 모델 추론 함수 (이미지 1장)
#########################################################################
def infer_model(model, transforms_test, img_path, class_names, device="cpu"):
    model.eval()
    model.to(device)

    try:
        img = Image.open(img_path).convert("RGB")

    except Exception as e:
        print(f"이미지 열기 실패: {e}")

        return None, None
    
    input_tensor = transforms_test(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(input_tensor)              # 점수 출력 [1, 24]
        prob = torch.softmax(pred, dim=1)[0]    # 점수를 확률로 변환 [0.1, 0.97, 0.02, ...]
        pred_class = pred.argmax(dim=1).item()  # 가장 높은 점수의 클랙스 인덱스 반환 -> pred_class=1
        confidence = float(prob[pred_class])    # 확률(신뢰도) 추출
        name = class_names[pred_class]          # 해당 클래스 이름 가져오기

    return (img_path, name, confidence)



#########################################################################
# 모델 추론 함수 (이미지 여러개)
#########################################################################
def infer_batch(model, transforms_test, img_paths, class_names, device="cpu"):
    model.eval()
    model.to(device)

    results = []
    for path in img_paths:
        try:
            img = Image.open(path).convert("RGB")
            batch_tensor = transforms_test(img).unsqueeze(0).to(device)
            
        except Exception as e:
            print(f"이미지 열기 실패 ({path}): {e}")

        with torch.no_grad():
            pred = model(batch_tensor)
            prob = torch.softmax(pred, dim=1)[0]
            pred_class = pred.argmax(dim=1).item()
            confidence = float(prob[pred_class])
            name = class_names[pred_class]
        
        results.append((path, name, confidence))

    return results



#########################################################################
# 추론 함수
#########################################################################
def infer(model, transforms_test, img_input, class_names, device="cpu"):
    if not isinstance(img_input, list):
        img_input = [img_input]

    if len(img_input) > 1:
        return infer_batch(model, transforms_test, img_input, class_names, device)
    else:
        return [infer_model(model, transforms_test, img_input[0], class_names, device)]



#########################################################################
# 원본데이터 그대로 학습
#########################################################################
def baseline_training(img_size=(224, 224)):
    transforms_train = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transforms_test = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return transforms_train, transforms_test



#########################################################################
# 증강
#########################################################################
def get_transforms(img_size=224, mean=None, std=None, grayscale=False, augmentations=None):
    DEFAULT_MEAN = [0.485, 0.456, 0.406]
    DEFAULT_STD = [0.229, 0.224, 0.225]

    mean = mean if mean is not None else DEFAULT_MEAN
    std = std if std is not None else DEFAULT_STD

    train_transforms = []
    test_transforms = []

    # train
    if grayscale:
        train_transforms.append(transforms.Grayscale(num_output_channels=3))
        test_transforms.append(transforms.Grayscale(num_output_channels=3))

    has_random_crop = augmentations and "random_crop" in augmentations

    if augmentations:
        # print("augmentations ::: ", augmentations)
        for aug_name in augmentations:
            if aug_name in AUGMENTATIONS:
                train_transforms.append(AUGMENTATIONS[aug_name])
            else:
                raise ValueError(f"Augmentation '{aug_name}' is None")

    if not has_random_crop:
        train_transforms.append(transforms.Resize((img_size, img_size)))

    train_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    transforms_train = transforms.Compose(train_transforms)

    # test
    resize_size = int(img_size * 256 / 224)  # 224 → 256

    test_transforms.extend([
        transforms.Resize(resize_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    transforms_test = transforms.Compose(test_transforms)

    return transforms_train, transforms_test




#############################################################################################################
def main():
    OUT_FEATURES = 12
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 30
    SAVE_NAME = "convNeXtTiny_V2_class12"
    IMG_PATH = "data/denim.jpg"       # 이미지 1개 or 이미지 리스트["data/t_shirt.jpg", "data/sweater_test.jpg"] 상관없음

    ###############
    # 데이터 가져오기
    ###############
    # transforms_train, transforms_test = baseline_training()
    transforms_train, transforms_test = get_transforms()

    train_raw = ImageFolder("./dataset12/train", transform=transforms_train)
    test_raw = ImageFolder("./dataset12/test", transform=transforms_test)

    train_dataloader = DataLoader(train_raw, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_raw, batch_size=BATCH_SIZE, shuffle=False)

    class_names = train_raw.classes

    ###############
    # 모델 불러오기
    ###############
    model = timm.create_model("convnextv2_tiny", pretrained=True)
    model.head.fc = nn.Linear(model.head.fc.in_features, OUT_FEATURES, bias=True)

    # 전체 파라미터 동결
    for param in model.parameters():
        param.requires_grad = False

    # 분류기 파라미터만 학습 가능하게 설정
    for param in model.head.fc.parameters():
        param.requires_grad = True

    ###############
    # 모델 학습
    ###############
    writer = SummaryWriter()
    optimizer = optim.Adam(model.head.fc.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    step = 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    for epoch in range(EPOCHS) :
        model.train()

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

        if (epoch + 1) in range(10, EPOCHS + 1, 10):
            save_model(model, SAVE_NAME, epoch+1)
            metrics = eval_model(model, epoch+1, test_dataloader)

            print("="*50)
            print(f"Epoch {metrics['epoch']} 평가 결과")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Macro F1: {metrics['f1_macro']:.4f}")
            print(metrics["classification_report"])

    ###############
    # 모델 추론
    ###############
    results = infer(model, transforms_test, IMG_PATH, class_names)
    # print(f"{path} → {name} ({conf:.4f})")

    for path, name, conf in results:
        print(f"{path} → {name} ({conf:.4f})")


if __name__ == '__main__':
    main()