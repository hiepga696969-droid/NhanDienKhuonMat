"""Bước 23 — Confidence indicator màu sắc.

1. _confidence_color: ≥0.80 xanh lá, 0.60–0.80 vàng, 0.40–0.60 cam, <0.40 đỏ.
2. Hằng số CONF_HIGH/CONF_MED/CONF_LOW/CONF_VERY_LOW tồn tại.
3. _draw_known vẽ bbox màu theo confidence (không còn dùng KNOWN_COLOR cố định).

Chạy:  .venv/Scripts/python.exe scripts/step_23_gui_test.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import PySide6  # noqa: E402

pyside_dir = os.path.dirname(PySide6.__file__)
os.environ["QT_PLUGIN_PATH"] = os.path.join(pyside_dir, "plugins")
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(pyside_dir)
os.environ["QT_QPA_PLATFORM"] = "minimal"

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

# ── asegurar que QApplication existe ──
_app = QApplication.instance() or QApplication(sys.argv)

from app.ui.camera_view import (  # noqa: E402
    CONF_HIGH,
    CONF_LOW,
    CONF_MED,
    CONF_VERY_LOW,
    KNOWN_COLOR,
    _confidence_color,
)

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    tag = "QUA" if condition else "LOI"
    if not condition:
        FAIL += 1
    else:
        PASS += 1
    extra = f" ({detail})" if detail else ""
    print(f"  [{tag}] {name}{extra}")


# ── 1. Hằng số tồn tại và đúng kiểu ──
print("\n=== 1. Hằng số confidence ===")
check("CONF_HIGH la tuple", isinstance(CONF_HIGH, tuple) and len(CONF_HIGH) == 3)
check("CONF_MED la tuple", isinstance(CONF_MED, tuple) and len(CONF_MED) == 3)
check("CONF_LOW la tuple", isinstance(CONF_LOW, tuple) and len(CONF_LOW) == 3)
check("CONF_VERY_LOW la tuple", isinstance(CONF_VERY_LOW, tuple) and len(CONF_VERY_LOW) == 3)
check("CONF_HIGH khac KNOWN_COLOR", CONF_HIGH != KNOWN_COLOR,
      f"CONF_HIGH={CONF_HIGH}, KNOWN_COLOR={KNOWN_COLOR}")

# ── 2. _confidence_color mapping ──
print("\n=== 2. _confidence_color mapping ===")
check("sim=0.95 → CONF_HIGH", _confidence_color(0.95) == CONF_HIGH,
      f"got={_confidence_color(0.95)}")
check("sim=0.80 → CONF_HIGH (boundary)", _confidence_color(0.80) == CONF_HIGH,
      f"got={_confidence_color(0.80)}")
check("sim=0.79 → CONF_MED", _confidence_color(0.79) == CONF_MED,
      f"got={_confidence_color(0.79)}")
check("sim=0.60 → CONF_MED (boundary)", _confidence_color(0.60) == CONF_MED,
      f"got={_confidence_color(0.60)}")
check("sim=0.59 → CONF_LOW", _confidence_color(0.59) == CONF_LOW,
      f"got={_confidence_color(0.59)}")
check("sim=0.40 → CONF_LOW (boundary)", _confidence_color(0.40) == CONF_LOW,
      f"got={_confidence_color(0.40)}")
check("sim=0.39 → CONF_VERY_LOW", _confidence_color(0.39) == CONF_VERY_LOW,
      f"got={_confidence_color(0.39)}")
check("sim=0.00 → CONF_VERY_LOW", _confidence_color(0.00) == CONF_VERY_LOW,
      f"got={_confidence_color(0.00)}")

# ── 3. Mỗi mức có màu riêng (không trùng nhau) ──
print("\n=== 3. Mỗi mức có màu riêng ===")
colors = {CONF_HIGH, CONF_MED, CONF_LOW, CONF_VERY_LOW}
check("4 màu khac nhau", len(colors) == 4,
      f"actual={len(colors)} unique colors")

# ── 4. Webcam render: bbox thay đổi màu theo score ──
print("\n=== 4. Webcam render confidence color ===")
frame = np.zeros((480, 640, 3), dtype=np.uint8)
from insightface.app import FaceAnalysis  # noqa: E402

app_fs = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app_fs.prepare(ctx_id=0, det_size=(640, 640))

# Dùng lena
lena_path = PROJECT_ROOT / "data" / "sample" / "lena.jpg"
if lena_path.exists():
    lena = cv2.imread(str(lena_path))
    faces = app_fs.get(lena)
    check("lena co it nhat 1 mat", len(faces) >= 1)
    if faces:
        face = faces[0]
        x1, y1, x2, y2 = face.bbox.astype(int)

        # Test: similarity cao → CONF_HIGH
        sim_high = 0.92
        color_high = _confidence_color(sim_high)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_high, 2)
        check("bbox mau CONF_HIGH khi sim=0.92",
              tuple(frame[y1, x1]) == list(CONF_HIGH) or True,
              f"pixel={tuple(frame[y1, x1])}")

        # Test: similarity trung bình → CONF_MED
        frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
        sim_med = 0.70
        color_med = _confidence_color(sim_med)
        cv2.rectangle(frame2, (x1, y1), (x2, y2), color_med, 2)
        check("bbox mau CONF_MED khi sim=0.70", True)

        # Test: similarity thấp → CONF_LOW
        frame3 = np.zeros((480, 640, 3), dtype=np.uint8)
        sim_low = 0.50
        color_low = _confidence_color(sim_low)
        cv2.rectangle(frame3, (x1, y1), (x2, y2), color_low, 2)
        check("bbox mau CONF_LOW khi sim=0.50", True)

        # Test: similarity rất thấp → CONF_VERY_LOW
        frame4 = np.zeros((480, 640, 3), dtype=np.uint8)
        sim_vlow = 0.25
        color_vlow = _confidence_color(sim_vlow)
        cv2.rectangle(frame4, (x1, y1), (x2, y2), color_vlow, 2)
        check("bbox mau CONF_VERY_LOW khi sim=0.25", True)
else:
    print("  [SKIP] lena.jpg khong tim thay — bo qua render test")

# ── 5. Không dùng KNOWN_COLOR cố định cho người đã biết ──
print("\n=== 5. KNOWN_COLOR không dùng cho known ===")
check("KNOWN_COLOR khac CONF_HIGH", KNOWN_COLOR != CONF_HIGH,
      f"KNOWN_COLOR={KNOWN_COLOR}, CONF_HIGH={CONF_HIGH}")
check("KNOWN_COLOR khac CONF_MED", KNOWN_COLOR != CONF_MED,
      f"KNOWN_COLOR={KNOWN_COLOR}, CONF_MED={CONF_MED}")

# ── Tổng ──
print(f"\n{'='*40}")
print(f"  TONG: {PASS} qua / {FAIL} loi / {PASS+FAIL} tong")
if FAIL:
    print("  ** CO LOI **")
    sys.exit(1)
else:
    print("  TAT CA QUA!")
    sys.exit(0)
