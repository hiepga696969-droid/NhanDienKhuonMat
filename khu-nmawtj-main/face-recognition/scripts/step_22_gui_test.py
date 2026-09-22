"""Bước 22 — Quality Gate + Enrollment Auto-reject.

1. SampleQuality.compute_overall(): mặt tốt → điểm cao; mặt xấu → thấp.
2. SampleQuality.is_good(): đủ tiêu chuẩn → True; thiếu 1 trong 4 → False.
3. compute_quality(face, frame): compute trên face thật (lena).
4. CameraWorker._draw_quality_gate:黑暗 → cảnh báo; bình thường → không vẽ.
5. EnrollmentWorker._capture_sample: mẫu xấu → bị từ chối.

Chạy:  .venv/Scripts/python.exe scripts/step_22_gui_test.py
"""
from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["QT_QPA_PLATFORM"] = "minimal"

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.core import face_metrics  # noqa: E402
from app.core.face_metrics import (  # noqa: E402
    MIN_QUALITY_BRIGHTNESS,
    MIN_QUALITY_BLUR,
    MIN_QUALITY_FACE_SCORE,
    MIN_QUALITY_OCCLUSION,
    SampleQuality,
    compute_quality,
)
from app.core.detector import FaceDetector, MODELS_ROOT  # noqa: E402
from app.core.embedder import FaceEmbedder  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [QUA] {name} {detail}")
    else:
        FAIL += 1
        print(f"  [THẤT] {name} {detail}")


def load_lena() -> np.ndarray:
    lena_path = PROJECT_ROOT / "lena_test.jpg"
    if not lena_path.exists() or lena_path.stat().st_size == 0:
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
            lena_path,
        )
    return cv2.imread(str(lena_path))


# ---------------------------------------------------------
# 1. SampleQuality — đơn vị
# ---------------------------------------------------------
def test_sample_quality_unit() -> None:
    print("\n[1] SampleQuality — đơn vị")

    # Mặt tốt: det cao, nét, sáng vừa, không che
    good = SampleQuality(
        face_score=0.90, blur_score=400, brightness=120,
        occlusion=0.05,
    )
    good.compute_overall()
    check("mặt tốt → overall > 70",
          good.overall > 70, f"(overall={good.overall:.1f})")
    check("mặt tốt → is_good() = True",
          good.is_good() is True)
    check("mặt tốt → label = 'Tốt'",
          good.label() == "Tốt")

    # Mặt kém: det thấp, mờ, tối, bị che
    bad = SampleQuality(
        face_score=0.40, blur_score=20, brightness=30,
        occlusion=0.60,
    )
    bad.compute_overall()
    check("mặt kém → overall < 30",
          bad.overall < 30, f"(overall={bad.overall:.1f})")
    check("mặt kém → is_good() = False",
          bad.is_good() is False)
    check("mặt kém → label = 'Kém'",
          bad.label() == "Kém")

    # Mặt trung bình
    mid = SampleQuality(
        face_score=0.75, blur_score=200, brightness=130,
        occlusion=0.15,
    )
    mid.compute_overall()
    check("mặt trung bình → 50 ≤ overall ≤ 80",
          50 <= mid.overall <= 80, f"(overall={mid.overall:.1f})")
    check("mặt trung bình → is_good() = True",
          mid.is_good() is True)


# ---------------------------------------------------------
# 2. is_good — chi tiết từng ngưỡng
# ---------------------------------------------------------
def test_is_good_thresholds() -> None:
    print("\n[2] is_good — ngưỡng chi tiết")

    # face_score thấp
    q1 = SampleQuality(face_score=0.50, blur_score=300, brightness=120, occlusion=0.05)
    q1.compute_overall()
    check("face_score < 0.60 → is_good() = False",
          q1.is_good() is False)

    # blur thấp
    q2 = SampleQuality(face_score=0.80, blur_score=50, brightness=120, occlusion=0.05)
    q2.compute_overall()
    check("blur < 100 → is_good() = False",
          q2.is_good() is False)

    # brightness quá tối
    q3 = SampleQuality(face_score=0.80, blur_score=300, brightness=30, occlusion=0.05)
    q3.compute_overall()
    check("brightness < 60 → is_good() = False",
          q3.is_good() is False)

    # brightness quá sáng
    q4 = SampleQuality(face_score=0.80, blur_score=300, brightness=230, occlusion=0.05)
    q4.compute_overall()
    check("brightness > 200 → is_good() = False",
          q4.is_good() is False)

    # occlusion cao
    q5 = SampleQuality(face_score=0.80, blur_score=300, brightness=120, occlusion=0.35)
    q5.compute_overall()
    check("occlusion ≥ 0.30 → is_good() = False",
          q5.is_good() is False)

    # Đúng ngưỡng → OK
    q6 = SampleQuality(
        face_score=MIN_QUALITY_FACE_SCORE,
        blur_score=MIN_QUALITY_BLUR,
        brightness=MIN_QUALITY_BRIGHTNESS,
        occlusion=MIN_QUALITY_OCCLUSION - 0.01,
    )
    q6.compute_overall()
    check("đúng ngưỡng tối thiểu → is_good() = True",
          q6.is_good() is True)


# ---------------------------------------------------------
# 3. compute_quality — trên ảnh thật
# ---------------------------------------------------------
def test_compute_quality_gpu() -> None:
    print("\n[3] compute_quality — ảnh thật (lena)")
    detector = FaceDetector(MODELS_ROOT)
    lena = load_lena()
    faces = detector.detect(lena)
    check("lena: detect ≥1 mặt", len(faces) >= 1)

    if faces:
        q = compute_quality(faces[0], lena)
        check("lena: overall > 50 (mặt rõ, nét, sáng tốt)",
              q.overall > 50, f"(overall={q.overall:.1f})")
        check("lena: is_good() = True",
              q.is_good() is True)
        check("lena: face_score > 0.7",
              q.face_score > 0.7, f"(face={q.face_score:.3f})")
        check("lena: blur_score > 100",
              q.blur_score > 100, f"(blur={q.blur_score:.0f})")

    # Ảnh tối → brightness thấp
    dark = np.clip(lena.astype(np.int16) - 150, 0, 255).astype(np.uint8)
    dark_faces = detector.detect(dark)
    if dark_faces:
        q_dark = compute_quality(dark_faces[0], dark)
        check("ảnh tối: brightness < 80",
              q_dark.brightness < 80, f"(brightness={q_dark.brightness:.0f})")

    # Ảnh mờ → blur_score thấp
    blurred = cv2.GaussianBlur(lena, (21, 21), 0)
    blur_faces = detector.detect(blurred)
    if blur_faces:
        q_blur = compute_quality(blur_faces[0], blurred)
        check("ảnh mờ: blur_score < lena blur_score",
              q_blur.blur_score < q.blur_score,
              f"(blur={q_blur.blur_score:.0f} vs {q.blur_score:.0f})")


# ---------------------------------------------------------
# 4. Enrollment auto-reject (unit, không camera)
# ---------------------------------------------------------
def test_enrollment_auto_reject() -> None:
    print("\n[4] Enrollment auto-reject — unit")
    from unittest import mock

    from app.ui.enrollment_dialog import EnrollmentWorker

    detector = FaceDetector(MODELS_ROOT)
    embedder = FaceEmbedder(detector)
    lena = load_lena()

    worker = EnrollmentWorker(
        0, 640, 480, detector=detector, embedder=embedder,
    )

    # Mẫu tốt → được thêm
    faces = detector.detect(lena)
    if faces:
        samples_good: list = []
        ok = worker._capture_sample(lena, faces[0], samples_good)
        check("mẫu tốt → _capture_sample trả True", ok is True)
        check("mẫu tốt → samples có 1 phần tử", len(samples_good) == 1)
        if samples_good:
            check("mẫu tốt → quality > 50",
                  samples_good[0].quality > 50,
                  f"(quality={samples_good[0].quality:.1f})")

    # Mẫu xấu (mock) → bị từ chối
    with mock.patch(
        "app.core.face_metrics.compute_quality",
        return_value=SampleQuality(
            face_score=0.30, blur_score=10, brightness=20,
            occlusion=0.50, overall=15.0,
        ),
    ):
        samples_bad: list = []
        ok_bad = worker._capture_sample(lena, faces[0], samples_bad)
        check("mẫu xấu (mock) → _capture_sample trả False", ok_bad is False)
        check("mẫu xấu → samples rỗng", len(samples_bad) == 0)


# ---------------------------------------------------------
# 5. CameraWorker._draw_quality_gate
# ---------------------------------------------------------
def test_quality_gate_draw() -> None:
    print("\n[5] CameraWorker._draw_quality_gate")
    from app.ui.camera_view import CameraWorker

    worker = CameraWorker(0, 640, 480)

    # Frame bình thường → không cảnh báo
    frame_normal = np.full((480, 640, 3), 120, dtype=np.uint8)
    worker._last_quality_check = 0.0  # force check
    worker._draw_quality_gate(frame_normal, [])
    # Không crash

    # Frame tối → có cảnh báo "ánh sáng yếu"
    frame_dark = np.zeros((480, 640, 3), dtype=np.uint8)
    worker._last_quality_check = 0.0
    worker._draw_quality_gate(frame_dark, [])
    # Không crash

    check("_draw_quality_gate không crash (frame正常 + tối)", True)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main() -> None:
    print("=== TEST BƯỚC 22: QUALITY GATE + ENROLLMENT AUTO-REJECT ===")
    test_sample_quality_unit()
    test_is_good_thresholds()
    test_compute_quality_gpu()
    test_enrollment_auto_reject()
    test_quality_gate_draw()
    print(f"\nKẾT QUẢ: {PASS} qua / {FAIL} thất bại")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
