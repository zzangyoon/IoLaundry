from pytorch_grad_cam import GradCAM, AblationCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.datasets import ImageFolder
from torchvision import transforms
from IPython.display import display
import numpy as np
import torch
import torch.nn as nn
import os
import timm
from PIL import Image

def main():
    OUT_FEATURES = 10
    MODEL_PATH = "model_sav/fabric_classifier_model_multitask_v0.7_143.pth"
    # MODEL_PATH = "model_sav/fabric_classifier_model_multitask_v0.6_64.pth"
    IMG_PATH = "data/silk_background.jpg"
    # IMG_PATH = "data/dark_denim.jpg"
    # IMG_PATH = "data/wool_wrinkle.jpg"


    ###############
    # 모델 불러오기
    ###############
    model = timm.create_model("convnextv2_tiny", pretrained=False)
    model.head.fc = nn.Linear(model.head.fc.in_features, OUT_FEATURES, bias=True)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"), strict=False)
    # print(model)

    image = Image.open(IMG_PATH)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_img = transform(image).unsqueeze(0)

    # 마지막 conv_layer
    layer = [model.stages[3].blocks[2].conv_dw]
    targets = [ClassifierOutputTarget(1)]

    cam = GradCAM(model=model, target_layers=layer)
    cam.batch_size = 1
    grayscale_cam = cam(input_tensor=input_img, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    print("grayscale_cam.max() ::: ", grayscale_cam.max())

    # 시각화
    image = image.resize((224, 224))
    rgb_img = np.float32(image) / 255
    result_grad_cam = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    display(Image.fromarray(result_grad_cam))


if __name__ == '__main__':
    main()