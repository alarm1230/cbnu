import cv2
from cv2 import dnn_superres
import numpy as np
import time
from time import sleep

start = time.time()

# ------------------ 설정 ------------------
MODEL_PATH    = 'FSRCNN_x4.pb'    # 다운로드해 둔 FSRCNN_x4.pb 경로
INPUT_IMAGE   = 'face.jpg'        # 원본 이미지 경로
OUTPUT_IMAGE  = 'face_comparison.jpg'  # 이어붙인 결과 저장 경로
SCALE_FACTOR  = 4                 # 업스케일 배율
scale = SCALE_FACTOR

# ------------------ Super-Resolution 초기화 ------------------
sr = dnn_superres.DnnSuperResImpl_create()
sr.readModel(MODEL_PATH)
sr.setModel('fsrcnn', SCALE_FACTOR)

# ------------------ 이미지 로드 ------------------
img = cv2.imread(INPUT_IMAGE)
if img is None:
    raise FileNotFoundError(f"입력 이미지를 찾을 수 없습니다: {INPUT_IMAGE}")

# ------------------ 업스케일 ------------------
upscaled = sr.upsample(img)
# upscaled.shape == (orig_h * 4, orig_w * 4, 3)

# ------------------ 원본을 업스케일 결과 크기로 리사이즈 ------------------
up_h, up_w = upscaled.shape[:2]
orig_resized = img.repeat(scale, axis=0).repeat(scale, axis=1)

# ------------------ 좌우 이어붙이기 ------------------
# (왼쪽: 리사이즈된 원본, 오른쪽: SR 결과)
comparison = np.hstack((orig_resized, upscaled))

# ------------------ 결과 저장 ------------------
cv2.imwrite(OUTPUT_IMAGE, comparison)
print(f"▶ 원본↔SR 비교 이미지가 저장되었습니다: {OUTPUT_IMAGE}")

end = time.time()

print(f"실행 시간 : {end - start}")
