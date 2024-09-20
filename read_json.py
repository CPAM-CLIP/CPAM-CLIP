import os
import json

# 기준 디렉토리 설정 (예시로 ./hyperparameter를 사용)
base_dir = './hyper-coco'

# mIoU 값을 저장할 리스트
miou_values = []

# 모든 하위 폴더와 파일을 탐색하는 함수
# for root, dirs, files in os.walk(base_dir):
for root, dirs, files in sorted(os.walk(base_dir), key=lambda x: x[0]):    
    for file in files:
        if file.endswith('.json'):  # JSON 파일만 처리
            file_path = os.path.join(root, file)
            
            # JSON 파일을 열고 내용 읽기
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
                
                # mIoU 값을 추출하고 리스트에 추가
                # if 'mIoU' in data:
                #     miou_values.append(data['mIoU'])
                # if data['mIoU'] > 59: 
                #     print(json_file)
                print(data['mIoU'])

# 결과 출력
# print("mIoU values:", miou_values)
