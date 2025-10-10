import streamlit as st
import timm
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from PIL import Image

DATA_DIR = "fabric_dataset"
MODEL_PATH = "model_sav/fabric_classifier_model_multitask_v0.6_64.pth"


# --- 브라우저 탭 ---
st.set_page_config(
    page_title="IoLaundry",
    page_icon="👕",
    layout="wide"
)


# --- 클래스 이름 정의 ---
try:
    # 데이터셋 폴더 구조에서 클래스 이름을 자동으로 가져오기
    temp_dataset = datasets.ImageFolder(root=DATA_DIR)
    FABRIC_CLASSES = temp_dataset.classes
    WASHING_CLASSES = ['machine_wash', 'delicate', 'dry_clean']
except FileNotFoundError:
    st.error(f"'{DATA_DIR}' 폴더를 찾을 수 없습니다. 앱이 정상적으로 동작하려면 프로젝트 루트에 데이터셋 폴더가 필요합니다.")
    # 폴더가 없을 경우를 대비한 기본값
    FABRIC_CLASSES = ['Blended', 'Cotton', 'Denim', 'Fleece', 'Nylon', 'Polyester', 'Silk', 'Terrycloth', 'Viscose', 'Wool']
    WASHING_CLASSES = ['machine_wash', 'delicate', 'dry_clean']


# --- 모델 정의 ---
class MultiTaskConvNeXt(nn.Module):
    def __init__(self, num_fabric_classes, num_washing_classes, task_type='multitask'):
        super().__init__()
        self.task_type = task_type
        # 모델 구조만 정의, 가중치는 파일에서 로드하므로 pretrained=False
        self.backbone = timm.create_model('convnextv2_tiny', pretrained=False, features_only=True)
        feature_dim = self.backbone.feature_info[-1]['num_chs']     # num_chs : feature_map 채널수

        self.fabric_head = nn.Linear(feature_dim, num_fabric_classes)
        self.washing_head = nn.Linear(feature_dim, num_washing_classes)

    def forward(self, x):
        features = self.backbone(x)
        pooled_features = features[-1].mean(dim=(-1, -2))

        fabric_output = self.fabric_head(pooled_features)
        washing_output = self.washing_head(pooled_features)
        
        return fabric_output, washing_output


# --- 이미지 전처리 ---
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # 채널수 문제 방지 (RGB 변환)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return transform(image).unsqueeze(0)


# --- 모델 로드 ---
@st.cache_resource
def load_model(model_path, task_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_fabric_classes = len(FABRIC_CLASSES)
    num_washing_classes = len(WASHING_CLASSES)
    
    model = MultiTaskConvNeXt(num_fabric_classes, num_washing_classes, task_type)
    
    try:
        # 모델 가중치 로드 (strict=False로 설정하여 불일치하는 키 무시)
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        model.to(device)
        model.eval() # 평가 모드로 설정
        return model, None
    except Exception as e:
        return None, str(e)


model = None
task_type = 'multitask'
with st.spinner(f"모델을 로드하는 중..."):
    model, error_msg = load_model(MODEL_PATH, task_type)
if error_msg:
    st.error(f"모델 로드 실패: {error_msg}")
else :
    st.toast(f"**모델 로드 완료**")


# --- 메인화면 UI ---
st.title("👕 IoLaundry")
st.write("")
st.markdown("업로드된 사진 또는 카메라로 촬영된 사진을 분석하여 **원단 종류**와 **적절한 세탁 방법**을 예측하는 서비스")
st.markdown("---")

st.write("")
st.write("")
st.write("")
st.subheader("📸 이미지 입력")

tab1, tab2 = st.tabs(["🖼️ 파일 업로드", "📷 카메라 촬영"])
uploaded_image = None
with tab1 :
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        uploaded_image = Image.open(uploaded_file)

with tab2 :
    camera_image = st.camera_input("")
    if camera_image:
        uploaded_image = Image.open(camera_image)

# --- 분석 및 결과 출력 ---
if uploaded_image is not None:
    st.write("")
    st.write("")
    st.write("")
    st.header("🔍 분석 결과")
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("입력 사진")
        st.image(uploaded_image, caption="분석할 사진", use_container_width =True)

    with col2:
        if model:
            placeholder = st.empty()  
            placeholder.subheader("분석 진행 중...")
            with st.spinner('모델이 이미지를 분석하고 있습니다...'):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                # 이미지 전처리 및 예측
                image_tensor = preprocess_image(uploaded_image).to(device)
                fabric_output, washing_output = model(image_tensor)
                
                # 예측 결과 저장
                predicted_fabric = "N/A"
                predicted_washing = "N/A"

                fabric_idx = torch.argmax(fabric_output, dim=1).item()
                washing_idx = torch.argmax(washing_output, dim=1).item()
                predicted_fabric = FABRIC_CLASSES[fabric_idx]
                predicted_washing = WASHING_CLASSES[washing_idx]

            placeholder.subheader("예측 결과")
            st.success("분석 완료!")

            st.metric(label="🧵 **예상 원단 종류**", value=predicted_fabric)
            st.metric(label="🛁 **권장 세탁 방법**", value=predicted_washing)

        else:
            st.error("모델이 로드되지 않아 분석을 진행할 수 없습니다.")
else:
    st.info("⬆️ 위에서 사진을 업로드하거나 카메라로 촬영하여 분석을 시작하세요.")

st.markdown("---")
st.caption("<p style='text-align: right;'>© 2025 IoLaundry</p>", unsafe_allow_html=True)

