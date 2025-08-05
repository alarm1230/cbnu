import os
import cv2
import numpy as np
from mmseg.apis import MMSegInferencer
from scipy.ndimage import binary_fill_holes

# ------------------ 설정 ------------------
CONFIG        = 'mmsegmentation/configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py'
CHECKPOINT    = 'mmsegmentation/checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'
DEVICE        = 'cuda:0'
VIDEO_PATH    = '2.mp4'
ROAD_CLASS_ID = 0
SCALE         = 0.5
OUTPUT_NAME   = 'output_segmentation.mp4'

# ------------------ Inferencer 초기화 ------------------
inferencer = MMSegInferencer(
    model   = CONFIG,
    weights = CHECKPOINT,
    device  = DEVICE,
    palette = None
)

# ------------------ 비디오 열기 ------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"비디오 열기 실패: {VIDEO_PATH}")

fps    = cap.get(cv2.CAP_PROP_FPS)
ret, frame0 = cap.read()
if not ret:
    raise RuntimeError("첫 프레임을 읽을 수 없습니다.")

h0, w0 = frame0.shape[:2]
h, w   = (int(h0*SCALE), int(w0*SCALE)) if SCALE!=1.0 else (h0, w0)
out_size = (w*2, h)

fourcc     = cv2.VideoWriter_fourcc(*'mp4v')
out_path   = os.path.join(os.getcwd(), OUTPUT_NAME)
writer     = cv2.VideoWriter(out_path, fourcc, fps, out_size)

# 프레임 루프 전 이전 마스크 초기화
prev_mask_closed = None

# 첫 프레임 리셋
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1) 스케일링
    in_frame = cv2.resize(frame, (w, h)) if SCALE!=1.0 else frame.copy()

    # 2) 세그멘테이션 추론
    result = inferencer(in_frame)
    if isinstance(result, list):
        result = result[0]
    seg_map = result['predictions']

    # 3) 도로 마스크 생성 + 최대 컴포넌트 필터링
    mask_small = (seg_map == ROAD_CLASS_ID)
    bin_mask   = mask_small.astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    if labels.max() > 0:
        areas  = stats[1:, cv2.CC_STAT_AREA]
        largest= 1 + np.argmax(areas)
        mask_filtered = (labels==largest)
    else:
        mask_filtered = mask_small

    # 4) Closing 으로 빈틈 메우기
    kernel       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51,51))
    mask_closed  = cv2.morphologyEx(mask_filtered.astype(np.uint8),
                                    cv2.MORPH_CLOSE, kernel).astype(bool)

    # 5) 이전 프레임 마스크와 OR
    if prev_mask_closed is None:
        disp_mask = mask_closed
    else:
        disp_mask = prev_mask_closed | mask_closed

    # 6) overlay 생성
    overlay = in_frame.copy()
    overlay[disp_mask] = (0,200,0)

    # 7) 좌우 병합
    combined = np.hstack((in_frame, overlay))

    # 8) 화면에 띄우기
    cv2.imshow('Road Seg (prev | curr)', combined)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("q 눌러서 중간 종료: 지금까지 처리된 프레임까지만 저장합니다.")
        break

    # 9) 파일에 쓰기
    writer.write(combined)

    # 10) prev 업데이트
    prev_mask_closed = mask_closed.copy()

# 자원 해제
cap.release()
writer.release()
cv2.destroyAllWindows()

print(f"▶ inference 비디오가 저장되었습니다: {out_path}")
