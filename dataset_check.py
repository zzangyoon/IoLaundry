# fabric_dataset 특징 분석 및 폴더별 이미지 개수 확인
import os

# fabric_dataset 경로 지정
fabric_root = 'fabric_dataset'

def analyze_fabric_dataset(root_dir):
	print(f'원단 데이터셋 최상위 폴더: {root_dir}')
	subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
	# Unclassified 폴더 삭제
	unclassified_path = os.path.join(root_dir, 'Unclassified')
	if 'Unclassified' in subdirs and os.path.exists(unclassified_path):
		import shutil
		shutil.rmtree(unclassified_path)
		print('Unclassified 폴더를 삭제했습니다.')
	subdirs = [d for d in subdirs if d != 'Unclassified']
	print(f'Unclassified 폴더 삭제 후 원단(폴더) 종류: {len(subdirs)}')
	folder_image_count = {}
	for subdir in subdirs:
		subdir_path = os.path.join(root_dir, subdir)
		image_files = [f for f in os.listdir(subdir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
		folder_image_count[subdir] = len(image_files)
	# 이미지 개수 기준 내림차순 정렬
	sorted_folders = sorted(folder_image_count.items(), key = lambda x: x[1], reverse = True)
	# 상위 10개 폴더만 남기고 나머지 폴더 삭제
	top_10_folders = sorted_folders[:10]
	folders_to_delete = [folder for folder, _ in sorted_folders[10:]]
	for folder in folders_to_delete:
		folder_path = os.path.join(root_dir, folder)
		if os.path.exists(folder_path):
			import shutil
			shutil.rmtree(folder_path)
			print(f'{folder} 폴더를 삭제했습니다.')
	print('\n[상위 10개 폴더와 이미지 개수 내림차순 출력]')
	for folder, count in top_10_folders:
		print(f'{folder}: {count}장')

# 특징 분석 주석
# - fabric_dataset은 다양한 원단 종류별로 폴더가 나뉘어 있음
# - 각 폴더에는 해당 원단의 이미지(.jpg, .jpeg, .png)가 다수 포함됨
# - 폴더명은 원단의 종류를 의미하며, 예시: Wool, Cotton, Denim 등
# - Unclassified, Utilities 등 특수 폴더도 존재

if __name__ == "__main__":
	analyze_fabric_dataset(fabric_root)
