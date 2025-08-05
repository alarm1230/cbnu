import cv2
import time
import torch
import numpy as np
from ultralytics import YOLO
from cv2 import dnn_superres

VIDEO_SRC = "test.mp4"

# ------------------ 설정 ------------------
POSE_MODEL_PATH    = 'yolo11s-pose.pt'
DET_MODEL_PATH     = 'yolo11s.pt'
SMOKING_MODEL_PATH = 'yolov5_smoking.pt'
SR_MODEL_PATH      = 'FSRCNN_x4.pb'
DIST_THRESHOLD     = 50
HOLD_SECONDS       = 3
CONF_KP            = 0.30
CONF_BOX           = 0.50
SR_SCALE           = 4
DISPLAY_SCALE      = 0.6

SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (5,11),(6,12)
]
FACE_IDXS = [0,1,2,3,4]  # 코, 눈, 귀

def get_face_center(pts, confs, box, conf_thr=CONF_KP):
    x1,y1,x2,y2 = box
    mask_conf = confs > conf_thr
    mask_box  = (
        (pts[:,0]>=x1)&(pts[:,0]<=x2)&
        (pts[:,1]>=y1)&(pts[:,1]<=y2)
    )
    valid = [i for i in FACE_IDXS if mask_conf[i] and mask_box[i]]
    if not valid: return None
    sel = pts[valid]
    return int(sel[:,0].mean()), int(sel[:,1].mean())

# 모델 로드
pose_model = YOLO(POSE_MODEL_PATH)
det_model  = YOLO(DET_MODEL_PATH)
smo_model  = torch.hub.load("ultralytics/yolov5", "custom", path=SMOKING_MODEL_PATH)

# Super-Resolution 초기화
sr = dnn_superres.DnnSuperResImpl_create()
sr.readModel(SR_MODEL_PATH)
sr.setModel('fsrcnn', SR_SCALE)

# 카메라 열기
cap = cv2.VideoCapture(VIDEO_SRC)
if not cap.isOpened():
    raise RuntimeError("카메라 열기 실패")

# 창 생성
MAIN_WIN = "YOLO + Pose Alert"
CROP_WIN = "Crop & SR (alert only)"
cv2.namedWindow(MAIN_WIN)
cv2.namedWindow(CROP_WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(CROP_WIN, 600, 300)  # 고정 크기

cond_start_hand = None   # 손-ROI 경보 시작 시간

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    annotated = frame.copy()
    crop_roi  = None
    hand_in_roi = False

    # 1) DET + POSE
    det_res  = det_model.predict(frame, verbose=False, classes=[0])[0]
    pose_res = pose_model(frame, verbose=False)[0]
    persons  = pose_res.keypoints.data if pose_res.keypoints is not None else []

    # 2) 박스별 처리
    if det_res.boxes is not None:
        for box in det_res.boxes:
            conf = float(box.conf)
            if conf < CONF_BOX:
                continue

            x1,y1,x2,y2 = map(int, box.xyxy[0])
            # 사람 박스: 파란색
            cv2.rectangle(annotated, (x1,y1),(x2,y2), (255,0,0), 2)
            cv2.putText(annotated, f"person {conf:.2f}", (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)

            # 박스 안에 있는 포즈 찾기
            matched = None
            for person in persons:
                pts   = person[:,:2].cpu().numpy()
                confs = person[:,2].cpu().numpy()
                # 코가 박스 안에 있으면 매치
                if confs[0]>CONF_KP and x1<=pts[0,0]<=x2 and y1<=pts[0,1]<=y2:
                    matched = (pts, confs)
                    break
            if not matched:
                continue

            pts, confs = matched

            # Skeleton 그리기
            valid = (confs>CONF_KP)&\
                    (pts[:,0]>=x1)&(pts[:,0]<=x2)&\
                    (pts[:,1]>=y1)&(pts[:,1]<=y2)
            for i in np.where(valid)[0]:
                xi, yi = pts[i].astype(int)
                cv2.circle(annotated,(xi,yi),3,(0,255,0),-1)
            for a,b in SKELETON:
                if valid[a] and valid[b]:
                    xa,ya = pts[a].astype(int)
                    xb,yb = pts[b].astype(int)
                    cv2.line(annotated,(xa,ya),(xb,yb),(0,255,255),2)

            # 얼굴 ROI 계산 & 표시 (빨간색)
            fc = get_face_center(pts, confs, (x1,y1,x2,y2))
            if fc:
                cx, cy = fc
                if confs[5]>CONF_KP and confs[6]>CONF_KP:
                    ys   = int((pts[5,1]+pts[6,1])/2)
                    side = max(abs(cy-ys),1)
                    half = side
                    xl, xr = max(cx-half,0), min(cx+half,w)
                    yt, yb = max(cy-half,0), min(cy+half,h)
                    crop_roi = frame[yt:yb, xl:xr].copy()
                    cv2.rectangle(annotated, (xl,yt),(xr,yb), (0,0,255),2)

                    # 손이 ROI 안에 있는지 체크 (wrists idx 9,10)
                    for idx in (9,10):
                        if confs[idx] > CONF_KP:
                            hx, hy = pts[idx]
                            if xl <= hx <= xr and yt <= hy <= yb:
                                hand_in_roi = True

    # 3) 손이 ROI 안에 3초 이상 있으면 알람
    now = time.time()
    if hand_in_roi:
        if cond_start_hand is None:
            cond_start_hand = now
        elif now - cond_start_hand >= HOLD_SECONDS:
            cv2.putText(annotated, "!!! ALERT !!!", (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,0,255), 5)
    else:
        cond_start_hand = None

    # 4) 메인 창 표시
    disp = cv2.resize(annotated,
                      (int(w*DISPLAY_SCALE), int(h*DISPLAY_SCALE)))
    cv2.imshow(MAIN_WIN, disp)

    # 5) Crop & SR 창 (경보 중에만)
    if cond_start_hand is not None and crop_roi is not None and crop_roi.size>0:
        sr_out = sr.upsample(crop_roi)
        face_up = np.repeat(
            np.repeat(crop_roi, SR_SCALE, axis=0),
            SR_SCALE, axis=1
        )
        combo = np.hstack((face_up, sr_out))
        cv2.imshow(CROP_WIN, combo)
    else:
        cv2.imshow(CROP_WIN, np.zeros((300,300,3), dtype=np.uint8))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
