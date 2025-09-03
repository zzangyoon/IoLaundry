import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import torch
from torchvision import datasets, transforms
import timm
from torch.utils.data import DataLoader, Dataset
import platform

# --- 한글 폰트 설정 ---
# 운영체제에 맞는 폰트 설정을 자동으로 적용합니다.
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin': # macOS
    plt.rc('font', family='AppleGothic')
else: # Linux
    # Linux 사용자는 나눔고딕과 같은 한글 폰트를 미리 설치해야 합니다.
    # 예: sudo apt-get install fonts-nanum*
    # 나눔고딕이 설치되어 있다고 가정하고 설정합니다.
    try:
        plt.rc('font', family='NanumGothic')
    except:
        print("경고: 나눔고딕 폰트가 설치되어 있지 않아 한글이 깨질 수 있습니다.")

# 마이너스 부호가 깨지는 현상 방지
plt.rcParams['axes.unicode_minus'] = False


# --- 1. 설정 변수 ---
DATA_DIR = 'fabric'
# t-SNE는 계산량이 많으므로, 전체 데이터셋 대신 샘플링하여 분석할 수 있습니다.
# None으로 설정하면 전체 데이터셋을 사용합니다.
SAMPLE_SIZE = None 
IMAGE_SIZE = 224

# --- 2. 데이터 로드 및 전처리를 위한 커스텀 데이터셋 ---
class ImagePathDataset(Dataset):
    """이미지 경로 리스트를 받아 이미지를 로드하는 커스텀 데이터셋"""
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# --- 3. 픽셀 값 분포 분석 함수 ---
def analyze_pixel_distribution(image_paths):
    """
    데이터셋의 모든 이미지에 대해 RGB 채널별 픽셀 값 평균을 계산하고
    히스토그램으로 시각화합니다.
    """
    print("\n--- 1. 픽셀 값 분포 분석 시작 ---")
    
    if SAMPLE_SIZE and len(image_paths) > SAMPLE_SIZE:
        print(f"전체 {len(image_paths)}개 이미지 중 {SAMPLE_SIZE}개를 샘플링하여 분석합니다.")
        indices = np.random.choice(len(image_paths), SAMPLE_SIZE, replace=False)
        sampled_paths = [image_paths[i] for i in indices]
    else:
        print(f"전체 {len(image_paths)}개 이미지를 분석합니다.")
        sampled_paths = image_paths

    means_r, means_g, means_b = [], [], []
    
    for path in tqdm(sampled_paths, desc="픽셀 값 분석 중"):
        try:
            with Image.open(path) as img:
                img_rgb = img.convert('RGB')
                img_array = np.array(img_rgb) / 255.0 # 0-1 정규화
                means_r.append(img_array[:, :, 0].mean())
                means_g.append(img_array[:, :, 1].mean())
                means_b.append(img_array[:, :, 2].mean())
        except Exception as e:
            print(f"경고: {path} 파일을 처리하는 중 오류 발생 - {e}")

    print("픽셀 값 분석 완료. 결과 시각화를 시작합니다.")
    
    plt.figure(figsize=(15, 5))
    plt.suptitle('RGB 채널별 픽셀 값 평균 분포', fontsize=16)
    
    plt.subplot(1, 3, 1)
    sns.histplot(means_r, color='red', kde=True)
    plt.title('Red 채널')
    plt.xlabel('평균 픽셀 값')
    plt.ylabel('이미지 수')

    plt.subplot(1, 3, 2)
    sns.histplot(means_g, color='green', kde=True)
    plt.title('Green 채널')
    plt.xlabel('평균 픽셀 값')

    plt.subplot(1, 3, 3)
    sns.histplot(means_b, color='blue', kde=True)
    plt.title('Blue 채널')
    plt.xlabel('평균 픽셀 값')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    print("--- 픽셀 값 분포 분석 종료 ---\n")


# --- 4. t-SNE를 위한 특징 추출 및 시각화 함수 ---
def visualize_tsne_with_features(image_paths, labels, class_names):
    """
    사전 학습된 모델로 이미지 특징을 추출하고 t-SNE로 시각화합니다.
    """
    print("--- 2. t-SNE 차원 축소 시각화 시작 ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"특징 추출을 위해 {device} 장치를 사용합니다.")

    # 사전 학습된 ConvNeXt 모델 로드 (특징 추출기)
    print("사전 학습된 ConvNeXt 모델을 로드합니다...")
    model = timm.create_model('convnextv2_tiny', pretrained=True, num_classes=0)
    model = model.to(device)
    model.eval()
    print("모델 로드 완료.")

    # 이미지 전처리 정의
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if SAMPLE_SIZE and len(image_paths) > SAMPLE_SIZE:
        print(f"전체 {len(image_paths)}개 이미지 중 {SAMPLE_SIZE}개를 샘플링하여 특징을 추출합니다.")
        indices = np.random.choice(len(image_paths), SAMPLE_SIZE, replace=False)
        sampled_paths = [image_paths[i] for i in indices]
        sampled_labels = [labels[i] for i in indices]
    else:
        print(f"전체 {len(image_paths)}개 이미지의 특징을 추출합니다. (시간이 소요될 수 있습니다)")
        sampled_paths = image_paths
        sampled_labels = labels

    # 데이터셋 및 데이터로더 생성
    dataset = ImagePathDataset(sampled_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    # 특징 추출
    all_features = []
    with torch.no_grad():
        for images in tqdm(dataloader, desc="이미지 특징 추출 중"):
            images = images.to(device)
            features = model(images)
            all_features.append(features.cpu().numpy())
    
    features_array = np.concatenate(all_features)
    print("특징 추출 완료.")

    # t-SNE 실행
    print("t-SNE 알고리즘을 실행합니다. (계산에 시간이 걸릴 수 있습니다)")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    tsne_results = tsne.fit_transform(features_array)
    print("t-SNE 계산 완료. 결과 시각화를 시작합니다.")

    # 시각화
    plt.figure(figsize=(10, 10))
    sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=sampled_labels,
        palette=sns.color_palette("hsv", len(class_names)),
        s=50,
        alpha=0.7
    )
    plt.title('t-SNE를 이용한 원단 데이터셋 시각화', fontsize=20)
    plt.xlabel('t-SNE 차원 1', fontsize=12)
    plt.ylabel('t-SNE 차원 2', fontsize=12)
    plt.legend(title='원단 종류', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print("--- t-SNE 시각화 종료 ---")


# --- 5. 밝기 및 대비 분석 함수 ---
def analyze_brightness_contrast(image_paths):
    """
    데이터셋의 모든 이미지에 대해 밝기와 대비를 계산하고
    히스토그램으로 시각화합니다.
    """
    print("\n--- 3. 밝기 및 대비 분석 시작 ---")

    if SAMPLE_SIZE and len(image_paths) > SAMPLE_SIZE:
        print(f"전체 {len(image_paths)}개 이미지 중 {SAMPLE_SIZE}개를 샘플링하여 분석합니다.")
        indices = np.random.choice(len(image_paths), SAMPLE_SIZE, replace=False)
        sampled_paths = [image_paths[i] for i in indices]
    else:
        print(f"전체 {len(image_paths)}개 이미지를 분석합니다.")
        sampled_paths = image_paths

    brightness_values = []
    contrast_values = []

    for path in tqdm(sampled_paths, desc="밝기/대비 분석 중"):
        try:
            with Image.open(path) as img:
                # 이미지를 그레이스케일로 변환
                img_gray = img.convert('L')
                img_array = np.array(img_gray)
                
                # 밝기 (픽셀 값의 평균)
                brightness_values.append(img_array.mean())
                
                # 대비 (픽셀 값의 표준 편차)
                contrast_values.append(img_array.std())
        except Exception as e:
            print(f"경고: {path} 파일을 처리하는 중 오류 발생 - {e}")

    print("밝기/대비 분석 완료. 결과 시각화를 시작합니다.")

    plt.figure(figsize=(12, 6))
    plt.suptitle('이미지 밝기 및 대비 분포', fontsize=16)

    plt.subplot(1, 2, 1)
    sns.histplot(brightness_values, color='orange', kde=True)
    plt.title('밝기(Brightness) 분포')
    plt.xlabel('평균 픽셀 값 (0-255)')
    plt.ylabel('이미지 수')

    plt.subplot(1, 2, 2)
    sns.histplot(contrast_values, color='purple', kde=True)
    plt.title('대비(Contrast) 분포')
    plt.xlabel('픽셀 값 표준 편차')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    print("--- 밝기 및 대비 분석 종료 ---\n")


def main():
    """메인 실행 함수"""
    from collections import Counter

    if not os.path.isdir(DATA_DIR):
        print(f"오류: '{DATA_DIR}' 폴더를 찾을 수 없습니다. 스크립트가 프로젝트 루트에서 실행되고 있는지 확인하세요.")
        return

    # 데이터셋에서 모든 이미지 경로와 라벨 로드
    dataset = datasets.ImageFolder(root=DATA_DIR)
    all_image_paths = [path for path, _ in dataset.samples]
    all_labels = [dataset.classes[label_idx] for _, label_idx in dataset.samples]
    
    print(f"'{DATA_DIR}' 폴더에서 총 {len(all_image_paths)}개의 이미지를 찾았습니다.")
    print(f"전체 클래스 종류 ({len(dataset.classes)}개): {dataset.classes}\n")

    # 클래스별 샘플 수 계산 및 필터링할 클래스 결정
    class_counts = Counter(all_labels)
    MIN_SAMPLES = 200
    
    filtered_class_names = sorted([
        name for name, count in class_counts.items() 
        if count >= MIN_SAMPLES and name.lower() != 'unclassified'
    ])
    
    print(f"샘플 수가 {MIN_SAMPLES}개 미만인 클래스와 'Unclassified' 클래스는 분석에서 제외합니다.")
    print(f"분석 대상 클래스 ({len(filtered_class_names)}개): {filtered_class_names}\n")

    # 필터링된 데이터만 포함하는 새로운 리스트 생성
    filtered_image_paths = []
    filtered_labels = []
    for path, label in zip(all_image_paths, all_labels):
        if label in filtered_class_names:
            filtered_image_paths.append(path)
            filtered_labels.append(label)
            
    print(f"필터링 후 총 {len(filtered_image_paths)}개의 이미지로 t-SNE 분석을 진행합니다.\n")

    # 1. 픽셀 값 분포 분석 실행
    # analyze_pixel_distribution(image_paths)
    
    # 2. t-SNE 시각화 실행
    visualize_tsne_with_features(filtered_image_paths, filtered_labels, filtered_class_names)

    # 3. 밝기 및 대비 분석 실행
    # analyze_brightness_contrast(image_paths)

if __name__ == '__main__':
    # Windows에서 multiprocessing 사용 시 필요할 수 있는 구문
    # torch.multiprocessing.freeze_support() 
    main()
