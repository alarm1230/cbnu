# Ultralytics 라이브러리의 데이터셋 불러오는 부분에 아래 코드를 삽입하면 데이터셋 로딩 시 5채널로 변환되어 입력됨

self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
self.midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large").to(self.device).eval()
transforms_module = torch.hub.load("intel-isl/MiDaS", "transforms")
self.midas_transform = transforms_module.dpt_transform


def five_channel_transform(self, im: np.ndarray) -> np.ndarray:
    # 1) BGR→RGB
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    # 2) MiDaS 전처리 → depth 예측 (float32 [H0,W0])
    inp = self.midas_transform(rgb).to(self.device)
    with torch.no_grad():
        pred = self.midas(inp).unsqueeze(0)
    depth = torch.nn.functional.interpolate(
        pred, size=(h, w), mode="bicubic", align_corners=False
    )
    depth = depth.cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)

    # 3) Canny edge
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), sigmaX=1.2)
    edges = cv2.Canny(gray, 70, 200)

    # 4) 채널 합치기 (BGR→B,G,R 순서 유지)
    depth_c = (depth * 255).astype(np.uint8)[..., None]
    edge_c = (edges > 0).astype(np.uint8)[..., None]

    im5 = np.concatenate([im, depth_c, edge_c], axis=2)  # H×W×5

    return im5
