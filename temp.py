import os
from collections import defaultdict
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import platform

def plot_tfevents(tfevents_path):
    """
    Reads a TensorBoard event file and visualizes the scalar data
    (loss, accuracy) using matplotlib.
    """
    # --- 한글 폰트 설정 ---
    if platform.system() == 'Windows':
        plt.rc('font', family='Malgun Gothic')
    elif platform.system() == 'Darwin': # macOS
        plt.rc('font', family='AppleGothic')
    else: # Linux
        try:
            plt.rc('font', family='NanumGothic')
        except:
            print("경고: 나눔고딕 폰트가 설치되어 있지 않아 한글이 깨질 수 있습니다.")
    plt.rcParams['axes.unicode_minus'] = False

    # --- 1. 이벤트 파일 로드 ---
    if not os.path.exists(tfevents_path):
        print(f"오류: 이벤트 파일 '{tfevents_path}'을(를) 찾을 수 없습니다.")
        return

    print(f"'{tfevents_path}' 파일에서 데이터를 읽어옵니다...")
    # EventAccumulator를 사용하여 이벤트 파일 로드
    # size_guidance는 로드할 데이터의 대략적인 크기를 지정합니다. 0으로 설정하면 모든 데이터를 로드합니다.
    accumulator = event_accumulator.EventAccumulator(
        tfevents_path,
        size_guidance={event_accumulator.SCALARS: 0}
    )
    accumulator.Reload() # 파일 내용 로드

    # --- 2. 스칼라 데이터 추출 ---
    # 사용 가능한 모든 스칼라 태그(e.g., 'Loss/train')를 가져옵니다.
    tags = accumulator.Tags()['scalars']
    
    data = defaultdict(list)
    steps = defaultdict(list)

    for tag in tags:
        events = accumulator.Scalars(tag)
        for event in events:
            data[tag].append(event.value)
            steps[tag].append(event.step)
    
    print("데이터 추출 완료. 시각화를 시작합니다...")

    # --- 3. Matplotlib으로 시각화 ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'TensorBoard 로그 시각화\n({os.path.basename(tfevents_path)})', fontsize=16)

    # 서브플롯 1: 손실 (Loss)
    ax1 = axes[0]
    if 'Loss/train' in data and 'Loss/val' in data:
        ax1.plot(steps['Loss/train'], data['Loss/train'], label='Training Loss')
        ax1.plot(steps['Loss/val'], data['Loss/val'], label='Validation Loss')
        ax1.set_title('학습 및 검증 손실 (Loss)')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
    else:
        ax1.set_title('손실 데이터 없음')

    # 서브플롯 2: 원단 정확도 (Fabric Accuracy)
    ax2 = axes[1]
    if 'Accuracy/train_fabric' in data and 'Accuracy/val_fabric' in data:
        ax2.plot(steps['Accuracy/train_fabric'], data['Accuracy/train_fabric'], label='Train Fabric Acc')
        ax2.plot(steps['Accuracy/val_fabric'], data['Accuracy/val_fabric'], label='Val Fabric Acc')
        ax2.set_title('원단 분류 정확도 (Fabric Accuracy)')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
    else:
        ax2.set_title('원단 정확도 데이터 없음')

    # 서브플롯 3: 세탁법 정확도 (Washing Accuracy)
    ax3 = axes[2]
    if 'Accuracy/train_washing' in data and 'Accuracy/val_washing' in data:
        ax3.plot(steps['Accuracy/train_washing'], data['Accuracy/train_washing'], label='Train Washing Acc')
        ax3.plot(steps['Accuracy/val_washing'], data['Accuracy/val_washing'], label='Val Washing Acc')
        ax3.set_title('세탁법 분류 정확도 (Washing Accuracy)')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Accuracy')
        ax3.legend()
        ax3.grid(True)
    else:
        ax3.set_title('세탁법 정확도 데이터 없음')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    # 시각화할 TensorBoard 이벤트 파일 경로
    # Windows에서는 경로 구분자로 '\' 대신 '/'를 사용하거나, r'...' 형태의 raw string을 사용하는 것이 좋습니다.
    event_file_path = r'logs\fit\20250829-114851\events.out.tfevents.1756435731.BOOK-E968UM43R7.11872.0'
    
    plot_tfevents(event_file_path)