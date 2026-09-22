"""Bước 4 — Kiểm tra FaceDetector bằng nhận diện THẬT trên GPU.

Chạy:  .venv\\Scripts\\python.exe scripts\\test_detector.py
Test:  ảnh lena (1 mặt) → phải phát hiện được; ảnh trắng → 0 mặt.
"""
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.core.detector import FaceDetector  # noqa: E402

# 1) Tải ảnh test (lena — 1 khuôn mặt chuẩn của OpenCV)
print("Tải ảnh test (lena.jpg)...")
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
    "lena_test.jpg",
)
img = cv2.imread("lena_test.jpg")
assert img is not None, "Không đọc được ảnh test"
print(f"Ảnh: {img.shape}")

# 2) Tạo detector (lần đầu nạp model + khởi tạo GPU mất vài giây)
print("Tạo FaceDetector (DirectML)...")
t0 = time.time()
detector = FaceDetector()
print(f"  Detector sẵn sàng sau {time.time() - t0:.1f}s")

# 3) Detect nhiều lần (lần đầu warmup)
for i in range(3):
    t1 = time.time()
    faces = detector.detect(img)
    dt = (time.time() - t1) * 1000
    print(f"  Lần {i + 1}: {dt:5.0f} ms — {len(faces)} khuôn mặt")

# 4) Ảnh lena phải có ≥1 khuôn mặt + bbox hợp lệ
assert len(faces) >= 1, "Ảnh lena phải phát hiện được khuôn mặt"
face = faces[0]
x1, y1, x2, y2 = face.bbox.astype(int)
assert x1 < x2 and y1 < y2, f"Bbox không hợp lệ: {[x1, y1, x2, y2]}"
assert 0 <= face.det_score <= 1, f"Điểm tin cậy lạ: {face.det_score}"
print(f"[1] Phát hiện khuôn mặt lena: bbox={[x1, y1, x2, y2]} · score={face.det_score:.3f}: OK")

# 5) Ảnh trắng → 0 khuôn mặt
blank = np.full((480, 640, 3), 255, dtype=np.uint8)
faces_blank = detector.detect(blank)
assert len(faces_blank) == 0, "Ảnh trắng không được phát hiện khuôn mặt"
print("[2] Ảnh trắng → 0 khuôn mặt: OK")

Path("lena_test.jpg").unlink(missing_ok=True)
print("\n=== TEST DETECTOR QUA ===")
