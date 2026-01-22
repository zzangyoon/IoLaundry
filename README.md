# 🧺 IoLaundry - AI 기반 스마트 세탁 시스템

ConvNeXtV2 기반 Multi-Task Learning으로 의류 원단을 자동 분류하고 최적의 세탁 방법을 추천하는 딥러닝 프로젝트

## 1. 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **문제 정의** | 일반 사용자가 의류 원단 종류를 구분하기 어려워 부적절한 세탁으로 옷 손상 발생 |
| **해결 방안** | 딥러닝 기반 이미지 분류로 원단 자동 인식 + 적합한 세탁 방법 추천 |
| **핵심 기술** | Multi-Task Learning, Transfer Learning, Grad-CAM |

## 2. 주요 기능

- **원단 분류**: 10종 원단 자동 인식 (Cotton, Denim, Silk, Wool, Polyester 등)
- **세탁 방법 추천**: 3가지 세탁 방법 추천 (machine_wash, delicate, dry_clean)
- **실시간 추론**: 이미지 업로드 또는 카메라 촬영 즉시 분석 결과 제공
- **모델 해석성**: Grad-CAM으로 모델 판단 근거 시각화

## 3. 시스템 아키텍처

```
Input Image (224×224×3)
        │
        ▼
┌───────────────────────┐
│   ConvNeXtV2-Tiny     │
│     (Backbone)        │
│   pretrained=True     │
└───────────────────────┘
        │
        ▼ Global Average Pooling
        │
   Feature Vector (768-dim)
        │
   ┌────┴──────┐
   ▼           ▼
┌────────┐  ┌────────┐
│Fabric  │  │Wash    │
│ Head   │  │ Head   │
│(768→10)│  │(768→3) │
└────────┘  └────────┘
    │           │
    ▼           ▼
 원단 분류    세탁 방법
```

## 4. 핵심 기술 포인트

### 1. Multi-Task Learning
- 단일 backbone으로 2가지 태스크 동시 수행
- 파라미터 효율성 및 공유 representation 학습

### 2. Transfer Learning
- ImageNet pretrained ConvNeXtV2 활용
- Backbone freeze, Head만 fine-tuning
- 제한된 데이터셋으로도 높은 성능 달성

### 3. Grad-CAM 해석성
- 모델 판단 근거 시각화
- 사용자 신뢰도 향상 및 디버깅 활용

## 5. 기술 스택

| 분류 | 기술 |
|------|------|
| **Deep Learning** | PyTorch, timm (ConvNeXtV2-Tiny) |
| **Web Framework** | Streamlit |
| **Visualization** | TensorBoard, Grad-CAM, Matplotlib, Seaborn |
| **ML Tools** | scikit-learn, torchvision |
| **Package Manager** | uv |

## 6. 데이터셋

### 원단 클래스 (10종)

| 클래스 | 설명 |
|--------|------|
| Blended | 혼방 |
| Cotton | 면 |
| Denim | 데님 |
| Fleece | 플리스 |
| Nylon | 나일론 |
| Polyester | 폴리에스터 |
| Silk | 실크 |
| Terrycloth | 테리 |
| Viscose | 비스코스 |
| Wool | 울 |

### 세탁 방법 클래스 (3종)

| 클래스 | 설명 |
|--------|------|
| machine_wash | 일반 세탁기 세탁 |
| delicate | 손세탁/약한 세탁 |
| dry_clean | 드라이클리닝 |

## 📁 프로젝트 구조

```
IoLaundry/
├── app_sy.py                    # Streamlit 웹 애플리케이션
├── convNeXtV2_tiny_sy.py        # 모델 학습 스크립트
├── grad_cam_sy.py               # Grad-CAM 시각화
├── common/                      # 공통 유틸리티
│   ├── model_loader.py          # 모델 로드 함수
│   ├── transforms.py            # 이미지 전처리
│   └── metrics.py               # 평가 지표
├── model_sav/                   # 학습된 모델 가중치 (.pth)
├── fabric_dataset/              # 원단 이미지 데이터셋
├── runs/                        # TensorBoard 로그
└── pyproject.toml               # 프로젝트 의존성 (uv)
```

## 🚀 설치 및 실행

### 환경 설정

```bash
# uv로 의존성 설치
uv sync

# 가상환경 활성화
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 웹 애플리케이션 실행

```bash
streamlit run app_sy.py
```

### 모델 학습

```bash
python convNeXtV2_tiny_sy.py
```

### TensorBoard 실행

```bash
tensorboard --logdir=runs
```

## 📸 스크린샷
<img 
    src="https://github.com/user-attachments/assets/473fc205-e70f-49e6-9d4b-63da14f57c21" 
    alt="스크린샷 2026-01-22 161719" 
    width="600" 
    style="border: 3px solid #4CAF50; border-radius: 10px; box-shadow: 3px 3px 8px rgba(0,0,0,0.3);" 
/>


