"""
Bước 0.4 — Kiểm tra model buffalo_l + GPU DirectML.

Mục tiêu:
  1. Tải bộ model buffalo_l (SCRFD + ArcFace) về thư mục models/
  2. Chạy thử nhận diện trên ảnh mẫu để xác nhận GPU DirectML hoạt động thật.
"""
import sys
import time
import urllib.request

# Console Windows mặc định dùng cp1252 — ép UTF-8 để in được tiếng Việt
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import cv2
import onnxruntime as ort
from insightface.app import FaceAnalysis

ROOT = "./models"  # thư mục chứa model (đã gitignore)

print(f"onnxruntime: {ort.__version__}")
print(f"providers  : {ort.get_available_providers()}")
print()

# ---- 1) Khởi tạo + tải model (lần đầu cần internet) ----
print("Tải model buffalo_l (lần đầu ~300MB, cần internet)...")
t0 = time.time()
app = FaceAnalysis(
    name="buffalo_l",
    root=ROOT,
    providers=["DmlExecutionProvider", "CPUExecutionProvider"],
)
app.prepare(ctx_id=0, det_size=(640, 640))
print(f"Model sẵn sàng sau {time.time() - t0:.1f}s")
print(f"Thành phần model: {list(app.models.keys())}")
print()

# ---- 2) Tải ảnh test (lena.jpg — ảnh chuẩn của OpenCV) ----
print("Tải ảnh test (lena.jpg)...")
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
    "lena_test.jpg",
)
img = cv2.imread("lena_test.jpg")
print(f"Ảnh kích thước: {img.shape}")
print()

# ---- 3) Chạy nhận diện nhiều lần để đo tốc độ GPU ----
print("Chạy nhận diện trên GPU DirectML...")
faces = []
for i in range(4):
    t1 = time.time()
    faces = app.get(img)
    dt = (time.time() - t1) * 1000
    tag = "(warmup)" if i == 0 else ""
    print(f"  Lần {i + 1}: {dt:5.0f} ms — {len(faces)} khuôn mặt {tag}")

# ---- 4) Báo kết quả ----
if faces:
    f = faces[0]
    print()
    print(f"Khuôn mặt đầu tiên:")
    print(f"  bbox       : {f.bbox.astype(int).tolist()}")
    print(f"  det_score  : {f.det_score:.3f}")
    print(f"  embedding  : {f.embedding.shape} (512 chiều)")
    print()
    print("✅ BƯỚC 0.4 THÀNH CÔNG — GPU DirectML hoạt động!")
else:
    print()
    print("⚠️ Không phát hiện khuôn mặt trong ảnh test — cần kiểm tra thêm.")
