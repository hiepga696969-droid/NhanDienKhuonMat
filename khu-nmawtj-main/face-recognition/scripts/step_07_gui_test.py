"""Test thực tế Bước 7+24 — EnrollmentDialog 4 bước + Face verification.

1. EnrollmentService: lưu người + mẫu embedding + thumbnail vào DB tạm.
2. face_metrics: EAR, nose_shift (xoay đầu) với dữ liệu THẬT từ lena.
3. Chuỗi GUIDANCE_STEPS (4 bước) + Face verification (test bằng embedding thật).
4. GUI: mở EnrollmentDialog với camera KHÔNG tồn tại → lỗi mềm, không crash.
5. Bước 24: kiểm tra 4 bước, ngưỡng mới, MAX_AUTO_RETRIES.

Chạy:  .venv/Scripts/python.exe scripts/step_07_gui_test.py
"""
from __future__ import annotations

import os
import sys
import urllib.request
from collections import deque
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import PySide6  # noqa: E402

# Khắc phục đường dẫn plugins + chế độ ảo (giống step_03)
pyside_dir = os.path.dirname(PySide6.__file__)
os.environ["QT_PLUGIN_PATH"] = os.path.join(pyside_dir, "plugins")
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(pyside_dir)
os.environ["QT_QPA_PLATFORM"] = "minimal"

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.config import Config  # noqa: E402
from app.core import face_metrics  # noqa: E402
from app.core.detector import FaceDetector, MODELS_ROOT  # noqa: E402
from app.core.embedder import FaceEmbedder  # noqa: E402
from app.infrastructure.db import DATA_DIR, Database  # noqa: E402
from app.infrastructure.repositories import (  # noqa: E402
    EMBEDDING_DIM,
    FaceSampleRepository,
    PersonRepository,
)
from app.services.enrollment import CapturedSample, EnrollmentService  # noqa: E402
from app.ui.enrollment_dialog import (  # noqa: E402
    GUIDANCE_STEPS,
    MAX_AUTO_RETRIES,
    MIN_FACE_SCORE,
    MIN_SHARPNESS,
    VERIFY_THRESHOLD,
    _check_frontal,
    _check_turn_left,
    _check_turn_right,
    _pick_sharpest,
    EnrollmentWorker,
)
from app.ui.enrollment_dialog import EnrollmentDialog  # noqa: E402

TEMP_DB = PROJECT_ROOT / "data" / "test_enrollment.db"
TEMP_THUMBS = PROJECT_ROOT / "data" / "thumbs_test"

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


# ---------------------------------------------------------
# 1. EnrollmentService trên DB tạm
# ---------------------------------------------------------
def test_service() -> None:
    print("\n[1] EnrollmentService — lưu người + mẫu + thumbnail")
    db = Database(TEMP_DB)
    service = EnrollmentService(db, thumbs_dir=TEMP_THUMBS)

    rng = np.random.default_rng(11)
    samples: list[CapturedSample] = []
    for i in range(4):  # Bước 24: 4 mẫu (thay vì 6)
        emb = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        crop = np.full((120, 120, 3), 30 + 20 * i, dtype=np.uint8)
        samples.append(CapturedSample(embedding=emb, quality=0.9, face_crop=crop))

    person = service.save_person("Nguyễn Văn A", samples)
    check("trả Person có id", len(person.id) == 32)
    check("tên lưu đúng", person.name == "Nguyễn Văn A")

    people = PersonRepository(db)
    face_samples = FaceSampleRepository(db)
    check("DB có 1 người", people.count() == 1)
    check("DB có đủ 4 mẫu", face_samples.count_by_person(person.id) == 4)
    check("thumbnail_path đã cập nhật", person.thumbnail_path.endswith(".jpg"))
    thumb_abs = DATA_DIR / person.thumbnail_path
    check("file thumbnail tồn tại trên đĩa", thumb_abs.exists())
    img = cv2.imread(str(thumb_abs))
    check("thumbnail đọc được (ảnh hợp lệ)", img is not None and img.size > 0)

    try:
        service.save_person("   ", samples)
        check("tên trống phải báo lỗi", False)
    except ValueError:
        check("tên trống bị từ chối", True)
    try:
        service.save_person("B", [])
        check("không mẫu phải báo lỗi", False)
    except ValueError:
        check("không có mẫu bị từ chối", True)

    db.close()
    for suffix in ("", "-wal", "-shm"):
        Path(str(TEMP_DB) + suffix).unlink(missing_ok=True)
    for f in TEMP_THUMBS.glob("*.jpg"):
        f.unlink(missing_ok=True)
    try:
        TEMP_THUMBS.rmdir()
    except OSError:
        pass


# ---------------------------------------------------------
# 2. face_metrics — đo trên dữ liệu THẬT (GPU)
# ---------------------------------------------------------
def test_face_metrics() -> None:
    print("\n[2] face_metrics — EAR / nose_shift / mouth (dữ liệu thật)")
    detector = FaceDetector(MODELS_ROOT)

    lena_path = PROJECT_ROOT / "lena_test.jpg"
    if not lena_path.exists() or lena_path.stat().st_size == 0:
        print("  Tải ảnh test lena.jpg...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
            lena_path,
        )
    lena = cv2.imread(str(lena_path))
    faces = detector.detect(lena)
    check("lena: 1 mặt", len(faces) == 1)
    if not faces:
        return
    f = faces[0]

    ear = face_metrics.eye_aspect_ratio(f)
    check("EAR lena (mắt mở) > 0.25", ear > face_metrics.EAR_BLINK_THRESHOLD,
          f"(EAR={ear:.3f})")
    check("lena nhìn thẳng (is_frontal)", face_metrics.is_frontal(f))
    check("lena nhìn thẳng CHẶT (is_frontal_strict)", face_metrics.is_frontal_strict(f))

    # Ảnh nhóm: các người quay nhiều hướng
    group = cv2.imread(sys.prefix + "/Lib/site-packages/insightface/data/images/t1.jpg")
    group_faces = detector.detect(group)
    check("ảnh nhóm có ≥ 2 mặt", len(group_faces) >= 2,
          f"(tìm thấy {len(group_faces)})")
    turns = [face_metrics.nose_shift(g) for g in group_faces]
    check("có người quay trái (nose_shift < -0.25)", min(turns) < -face_metrics.NOSE_SHIFT_THRESHOLD,
          f"(min={min(turns):.3f})")
    check("có người quay phải (nose_shift > 0.25)", max(turns) > face_metrics.NOSE_SHIFT_THRESHOLD,
          f"(max={max(turns):.3f})")

    # Face giả: kiểm tra is_blinking / is_mouth_open với landmark nhân tạo
    fake = type("FakeFace", (), {})()
    lm = np.zeros((68, 3), dtype=np.float32)
    # CẢ HAI mắt nhắm (EAR ~0.1) — vì EAR là trung bình 2 mắt
    lm[36] = [0, 0, 0]; lm[39] = [10, 0, 0]
    lm[37] = [2.5, 1, 0]; lm[41] = [2.5, -1, 0]
    lm[38] = [5, 1, 0]; lm[40] = [5, -1, 0]
    lm[42] = [20, 0, 0]; lm[45] = [30, 0, 0]
    lm[43] = [22, 1, 0]; lm[47] = [22, -1, 0]
    lm[44] = [25, 1, 0]; lm[46] = [25, -1, 0]
    fake.landmark_3d_68 = lm
    fake.kps = np.array([[30, 0], [0, 0], [15, 5], [15, 10], [15, -10]], dtype=np.float32)
    check("EAR mắt nhắm < 0.25", face_metrics.eye_aspect_ratio(fake) < face_metrics.EAR_BLINK_THRESHOLD,
          f"(EAR={face_metrics.eye_aspect_ratio(fake):.3f})")
    check("is_blinking với mắt nhắm", face_metrics.is_blinking(fake))

    # Fake 2: mở miệng to
    fake2 = type("FakeFace", (), {})()
    lm2 = np.zeros((68, 3), dtype=np.float32)
    for i in range(68):
        lm2[i] = [i, 0, 0]  # placeholder — miệng sẽ tự tính
    # Môi trong 60-67: khép trái/phải 0..10, dọc to 8
    lm2[60] = [0, 0, 0]; lm2[64] = [10, 0, 0]
    lm2[63] = [5, 8, 0]; lm2[67] = [5, 0, 0]  # dọc 8 / ngang 10 = 0.8
    fake2.landmark_3d_68 = lm2
    fake2.kps = np.zeros((5, 2), dtype=np.float32)
    mr = face_metrics.mouth_ratio(fake2)
    check("miệng mở to → is_mouth_open", face_metrics.is_mouth_open(fake2),
          f"(mouth_ratio={mr:.3f})")

    # Fake 3: kps mũi lệch trái/phải → nose_shift
    # kps: [mắt phải, mắt trái, mũi, ...] — 2 mắt cách nhau 20px, giữa = x=10
    fake3 = type("FakeFace", (), {})()
    fake3.kps = np.array([[20, 0], [0, 0], [10, 5], [0, 0], [0, 0]], dtype=np.float32)
    check("nose_shift mũi giữa ≈ 0", abs(face_metrics.nose_shift(fake3)) < 0.1,
          f"(={face_metrics.nose_shift(fake3):.3f})")
    fake3.kps[2] = [30, 5]  # mũi lệch phải 20px / 20px = 1.0
    check("nose_shift dương (quay trái)", face_metrics.nose_shift(fake3) > 0.25,
          f"(={face_metrics.nose_shift(fake3):.3f})")

    # ---- Bước 24: GUIDANCE_STEPS 4 bước ----
    check("có 4 bước đăng ký", len(GUIDANCE_STEPS) == 4,
          f"(có {len(GUIDANCE_STEPS)} bước)")
    step_names = [s.name for s in GUIDANCE_STEPS]
    check("bước 1 = Nhìn thẳng", step_names[0] == "Nhìn thẳng")
    check("bước 2 = Quay trái", step_names[1] == "Quay trái")
    check("bước 3 = Quay phải", step_names[2] == "Quay phải")
    check("bước 4 = Nhìn thẳng (xác minh)", step_names[3] == "Nhìn thẳng")

    # Bước nhìn thẳng dùng is_frontal_strict + giữ yên 2s
    straight_step = GUIDANCE_STEPS[0]
    check("bước nhìn thẳng giữ yên ≥ 2s",
          straight_step.min_hold_seconds >= 2.0)
    # Bước quay đầu giữ yên 1.0s (Bước 25)
    check("bước quay trái giữ yên 1.0s",
          GUIDANCE_STEPS[1].min_hold_seconds == 1.0)
    check("bước quay phải giữ yên 1.0s",
          GUIDANCE_STEPS[2].min_hold_seconds == 1.0)
    # Kiểm tra MAX_AUTO_RETRIES
    check("MAX_AUTO_RETRIES = 3", MAX_AUTO_RETRIES == 3)
    # Kiểm tra ngưỡng mới
    check("MIN_SHARPNESS = 25 (webcam laptop)", MIN_SHARPNESS == 25)
    check("MIN_FACE_SCORE = 0.60 (webcam laptop)", MIN_FACE_SCORE == 0.60)

    # ---- is_frontal_strict: NGHIÊNG ĐẦU (roll) → từ chối (mẫu nhìn thẳng
    # phải thực sự đối diện camera, không nghiêng — cải thiện nhận diện) ----
    fake_tilt = type("FakeFace", (), {})()
    lm_t = np.zeros((68, 3), dtype=np.float32)
    lm_t[36] = [0, 0, 0]; lm_t[39] = [10, 0, 0]
    lm_t[37] = [2.5, 4, 0]; lm_t[41] = [2.5, -4, 0]
    lm_t[38] = [5, 4, 0]; lm_t[40] = [5, -4, 0]
    lm_t[42] = [20, 0, 0]; lm_t[45] = [30, 0, 0]
    lm_t[43] = [22, 4, 0]; lm_t[47] = [22, -4, 0]
    lm_t[44] = [25, 4, 0]; lm_t[46] = [25, -4, 0]
    fake_tilt.landmark_3d_68 = lm_t
    fake_tilt.kps = np.array(
        [[30, 0], [0, 0], [15, 5], [15, 10], [15, -10]], dtype=np.float32
    )
    fake_tilt.pose = np.array([0.0, 0.0, 30.0])  # nghiêng đầu 30°
    check("nhìn thẳng nhưng NGHIÊNG ĐẦU → strict TỪ CHỐI",
          face_metrics.is_frontal_strict(fake_tilt) is False)
    fake_straight = type("FakeFace", (), {})()
    fake_straight.landmark_3d_68 = lm_t.copy()
    fake_straight.kps = fake_tilt.kps.copy()
    fake_straight.pose = np.array([0.0, 0.0, 2.0])  # không nghiêng
    check("không nghiêng + mũi giữa + mắt mở → strict CHẤP NHẬN",
          face_metrics.is_frontal_strict(fake_straight) is True)

    # ---- Độ nét khi đăng ký: lọc ảnh MỜ + chọn khung nét nhất ----
    embedder7 = FaceEmbedder(detector)
    worker7 = EnrollmentWorker(0, 640, 480, detector=detector, embedder=embedder7)
    sharp = EnrollmentWorker._sharpness(lena, f)
    blurred_face = cv2.GaussianBlur(lena, (9, 9), 0)
    fb2 = detector.detect(blurred_face)[0]
    blurred_sharp = EnrollmentWorker._sharpness(blurred_face, fb2)
    check("_sharpness: ảnh nét cao hơn ảnh mờ nhiều lần",
          sharp > blurred_sharp * 5,
          f"(nét={sharp:.0f}, mờ={blurred_sharp:.0f})")
    check("_pick_best_face: mặt NÉT → chọn được",
          worker7._pick_best_face(lena) is not None)
    check("_pick_best_face: mặt MỜ → từ chối (None)",
          worker7._pick_best_face(blurred_face) is None,
          f"(độ nét mờ={blurred_sharp:.0f} < MIN_SHARPNESS={MIN_SHARPNESS})")

    # _pick_sharpest: đệm khung → chọn khung có độ nét CAO NHẤT khi chụp
    frames = [np.full((10, 10, 3), i, dtype=np.uint8) for i in range(3)]
    faces_fake = [
        type("F", (), {"bbox": np.array([0, 0, 5, 5])})() for _ in range(3)
    ]
    recent = deque(maxlen=5)
    recent.append((10.0, frames[0], faces_fake[0]))
    recent.append((80.0, frames[1], faces_fake[1]))
    recent.append((30.0, frames[2], faces_fake[2]))
    best_frame, best_face = _pick_sharpest(recent, frames[0], faces_fake[0])
    check("_pick_sharpest: chọn khung NÉT NHẤT trong đệm",
          best_frame is frames[1] and best_face is faces_fake[1])

    lena_path.unlink(missing_ok=True)


# ---------------------------------------------------------
# 3. GUIDANCE_STEPS + Face verification (GPU thật)
# ---------------------------------------------------------
def test_verification() -> None:
    print("\n[3] Face verification — cùng người cao, khác người thấp")
    detector = FaceDetector(MODELS_ROOT)
    embedder = FaceEmbedder(detector)

    lena_path = PROJECT_ROOT / "lena_test.jpg"
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
        lena_path,
    )
    lena = cv2.imread(str(lena_path))
    group = cv2.imread(sys.prefix + "/Lib/site-packages/insightface/data/images/t1.jpg")

    emb_lena = embedder.embed_face(detector.detect(lena)[0])
    faces = sorted(detector.detect(group), key=lambda x: -(x.bbox[2] - x.bbox[0]))
    emb_other = embedder.embed_face(faces[1])

    sim_same = FaceEmbedder.cosine_similarity(emb_lena, emb_lena)
    sim_diff = FaceEmbedder.cosine_similarity(emb_lena, emb_other)
    check("cùng người (lena vs chính nó) ≥ ngưỡng", sim_same >= VERIFY_THRESHOLD,
          f"(sim={sim_same:.3f})")
    check("khác người < ngưỡng verification", sim_diff < VERIFY_THRESHOLD,
          f"(sim={sim_diff:.3f})")
    check("4 bước hướng dẫn đúng thứ tự",
          [s.name for s in GUIDANCE_STEPS] ==
          ["Nhìn thẳng", "Quay trái", "Quay phải", "Nhìn thẳng"])

    lena_path.unlink(missing_ok=True)


# ---------------------------------------------------------
# 4. GUI: dialog với camera không tồn tại → lỗi mềm
# ---------------------------------------------------------
def test_gui_no_camera() -> None:
    print("\n[4] GUI — EnrollmentDialog với camera không tồn tại")
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QMessageBox

    app = QApplication.instance() or QApplication(sys.argv)

    config = Config(camera_index=99, camera_width=320, camera_height=240)
    db = Database(TEMP_DB)
    dialog = EnrollmentDialog(config, db, detector=None, parent=None)

    def fake_critical(parent, title, text):
        print(f"    (QMessageBox.critical giả: {title})")

    QMessageBox.critical = staticmethod(fake_critical)
    dialog._name_edit.setText("Người test")
    dialog._start_capture()

    def finish():
        dialog.close()

    QTimer.singleShot(2500, finish)
    dialog.exec()

    check("dialog không crash với camera hỏng", True)
    check("nút Bắt đầu thu được bật lại", dialog._start_btn.isEnabled())
    check("steps preview có đủ 4 bước chưa tới", dialog._steps_label.text().count("○") == 4,
          f"(text={dialog._steps_label.text()!r})")

    # Bước 25: Progress bar tồn tại + ban đầu = 0
    check("progress bar tồn tại", hasattr(dialog, "_progress_bar"))
    check("progress bar ban đầu = 0", dialog._progress_bar.value() == 0)

    # Bước 25: Hint label tồn tại
    check("hint label tồn tại", hasattr(dialog, "_hint_label"))
    check("hint label ban đầu trống", dialog._hint_label.text() == "")

    # Bước 25: _update_hint mapping
    dialog._hint_label.setText("")
    dialog._update_hint("Chưa thấy khuôn mặt")
    check("hint: 'Chưa thấy khuôn mặt' → 👋",
          "👋" in dialog._hint_label.text())
    dialog._update_hint("Khuôn mặt quá nhỏ — tiến lại gần")
    check("hint: 'quá nhỏ' → 📏",
          "📏" in dialog._hint_label.text())
    dialog._update_hint("Bị chói — tránh ngược sáng")
    check("hint: 'chói' → 💡",
          "💡" in dialog._hint_label.text())
    dialog._update_hint("Ánh sáng quá tối")
    check("hint: 'tối' → 💡",
          "💡" in dialog._hint_label.text())
    dialog._update_hint("⚠ Mẫu chưa đạt: ảnh mờ — đang tự chỉnh...")
    hint_after_retry = dialog._hint_label.text()
    check("hint: 'mẫu chưa đạt' → giữ nguyên hint trước",
          hint_after_retry != "")  # không xóa hint khi retry
    dialog._update_hint("Đã thu xong")
    check("hint: tin nhắn khác → xóa hint", dialog._hint_label.text() == "")

    # Bước 25: _on_sample_captured cập nhật progress bar
    dialog._on_sample_captured(2, 4, "82/100 (Tốt)")
    check("_on_sample_captured(2/4) → progress bar = 50%",
          dialog._progress_bar.value() == 50)
    check("_on_sample_captured(2/4) → hint xóa",
          dialog._hint_label.text() == "")
    dialog._on_sample_captured(4, 4, "90/100 (Tốt)")
    check("_on_sample_captured(4/4) → progress bar = 100%",
          dialog._progress_bar.value() == 100)

    for suffix in ("", "-wal", "-shm"):
        Path(str(TEMP_DB) + suffix).unlink(missing_ok=True)


def main() -> None:
    print("=== TEST BƯỚC 7+24+25: ĐĂNG KÝ KHUÔN MẶT ===")
    test_service()
    test_face_metrics()
    test_verification()
    test_gui_no_camera()

    print(f"\nKẾT QUẢ: {PASS} qua / {FAIL} thất bại")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
