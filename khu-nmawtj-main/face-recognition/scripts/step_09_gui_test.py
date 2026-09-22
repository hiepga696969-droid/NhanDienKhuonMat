"""Bước 9 — Kiểm tra nhận diện thời gian thực (chế độ minimal, tự thoát).

1. RecognitionService: nạp embedding từ DB → match đúng người (lena) /
   từ chối người khác (nhóm 6 người); ghi sự kiện + snapshot + count_today.
2. CameraWorker._process_frame (dữ liệu THẬT, GPU): người đã biết → không
   báo người lạ + phát recognition_hit; chưa có dữ liệu → NGƯỜI LẠ + làm mờ
   (giảm độ tương phản vùng mặt) + phát unknown_crop.
3. GUI: CameraView với camera 99 → lỗi mềm không crash, nút [Đăng ký ngay]
   ẩn, panel trạng thái hiển thị; EnrollmentDialog với seed_sample hiện đúng UI.

Chạy:  .venv\\Scripts\\python.exe scripts\\step_09_gui_test.py
"""
from __future__ import annotations

import os
import sys
import urllib.request
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
from app.core.detector import FaceDetector, MODELS_ROOT  # noqa: E402
from app.core.embedder import FaceEmbedder  # noqa: E402
from app.infrastructure.db import DATA_DIR, Database  # noqa: E402
from app.infrastructure.repositories import RecognitionEventRepository  # noqa: E402
from app.services.enrollment import CapturedSample, EnrollmentService  # noqa: E402
from app.services.recognition import RecognitionService  # noqa: E402

TEMP_DB = PROJECT_ROOT / "data" / "test_recognition.db"
TEMP_THUMBS = PROJECT_ROOT / "data" / "thumbs_test9"

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


def cleanup() -> None:
    for suffix in ("", "-wal", "-shm"):
        Path(str(TEMP_DB) + suffix).unlink(missing_ok=True)
    for f in TEMP_THUMBS.glob("*.jpg"):
        f.unlink(missing_ok=True)
    try:
        TEMP_THUMBS.rmdir()
    except OSError:
        pass
    snapshots = DATA_DIR / "snapshots"
    if snapshots.exists():
        for f in snapshots.glob("*.jpg"):
            f.unlink(missing_ok=True)


def load_lena() -> np.ndarray:
    """Tải ảnh lena về project (bỏ qua nếu đã có và không rỗng)."""
    lena_path = PROJECT_ROOT / "lena_test.jpg"
    if not lena_path.exists() or lena_path.stat().st_size == 0:
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
            lena_path,
        )
    return cv2.imread(str(lena_path))


# ---------------------------------------------------------
# 1. RecognitionService — match + ghi sự kiện (GPU)
# ---------------------------------------------------------
def test_service() -> None:
    print("\n[1] RecognitionService — match + sự kiện + snapshot")
    db = Database(TEMP_DB)
    svc = RecognitionService(db)
    events = RecognitionEventRepository(db)

    detector = FaceDetector(MODELS_ROOT)
    embedder = FaceEmbedder(detector)

    lena = load_lena()
    lena_face = detector.detect(lena)[0]
    emb_lena = embedder.embed_face(lena_face)

    group = cv2.imread(sys.prefix + "/Lib/site-packages/insightface/data/images/t1.jpg")
    faces = sorted(detector.detect(group), key=lambda x: -(x.bbox[2] - x.bbox[0]))
    emb_other = embedder.embed_face(faces[1])
    check("lena ≠ người thứ 2 trong ảnh nhóm",
          FaceEmbedder.cosine_similarity(emb_lena, emb_other) < 0.4,
          f"(cosine={FaceEmbedder.cosine_similarity(emb_lena, emb_other):.3f})")

    # Đăng ký lena thành người 'Nguyễn Văn A' (DB tạm)
    enrollment = EnrollmentService(db, thumbs_dir=TEMP_THUMBS)
    person = enrollment.save_person(
        "Nguyễn Văn A",
        [CapturedSample(embedding=emb_lena, quality=0.9, face_crop=lena)],
    )

    # Chưa reload → không có dữ liệu để match
    check("trước reload: chưa có mẫu", svc.size == 0)
    svc.reload()
    check("sau reload: có 1 mẫu", svc.size == 1)

    result = svc.match(emb_lena, threshold=0.40)
    check("lena → đúng người A", result is not None
          and result.person_id == person.id
          and result.similarity > 0.8,
          f"(sim={result.similarity:.3f})" if result else "")
    check("label_of đúng tên", svc.label_of(person.id) == "Nguyễn Văn A")

    check("người khác → None (dưới ngưỡng)", svc.match(emb_other, threshold=0.40) is None)

    # Ghi sự kiện: người đã biết
    crop = lena.copy()
    event_id = svc.save_event(person_id=person.id, similarity=0.9, face_crop=crop)
    check("save_event trả id", bool(event_id))
    snapshots = DATA_DIR / "snapshots"
    check("snapshot tồn tại trên đĩa", any(snapshots.glob("*.jpg")))
    check("count_today ≥ 1", events.count_today() >= 1,
          f"(={events.count_today()})")

    # Ghi sự kiện: người lạ
    svc.save_event(person_id=None, similarity=None, face_crop=crop, is_unknown=True)
    with db.session() as conn:
        row = conn.execute(
            "SELECT label, is_unknown, person_id FROM recognition_events"
            " WHERE id = ?", (event_id,),
        ).fetchone()
    check("sự kiện đã biết: label = tên người", row["label"] == "Nguyễn Văn A")
    check("sự kiện đã biết: is_unknown = 0", row["is_unknown"] == 0)

    lena_path = PROJECT_ROOT / "lena_test.jpg"
    lena_path.unlink(missing_ok=True)
    db.close()
    cleanup()


# ---------------------------------------------------------
# 2. CameraWorker._process_frame — nhận diện thật + làm mờ (GPU)
# ---------------------------------------------------------
def test_worker_frames() -> None:
    print("\n[2] CameraWorker._process_frame — người lạ bị làm mờ")
    from app.ui.camera_view import CameraWorker

    detector = FaceDetector(MODELS_ROOT)
    embedder = FaceEmbedder(detector)

    # --- (a) Có dữ liệu: lena là người ĐÃ BIẾT ---
    db = Database(TEMP_DB)
    svc = RecognitionService(db)
    enrollment = EnrollmentService(db, thumbs_dir=TEMP_THUMBS)
    lena = load_lena()
    lena_face = detector.detect(lena)[0]
    emb_lena = embedder.embed_face(lena_face)
    person = enrollment.save_person(
        "Nguyễn Văn A",
        [CapturedSample(embedding=emb_lena, quality=0.9, face_crop=lena)],
    )
    svc.reload()

    hits: list = []
    unknowns: list = []
    # Bước 9 chưa có anti-spoofing (Bước 17) — tắt để test đúng hành vi cũ:
    # ảnh lena TĨNH không chớp mắt, nếu bật sẽ bị coi là giả mạo.
    worker = CameraWorker(0, 640, 480, detector=detector, embedder=embedder,
                          service=svc, threshold=0.40,
                          anti_spoofing_enabled=False)
    worker.recognition_hit.connect(lambda pid, sim, crop: hits.append((pid, sim)))
    worker.unknown_face.connect(lambda has: unknowns.append(has))

    has_unknown = worker._process_frame(lena.copy())
    check("lena đã biết → không báo người lạ", has_unknown is False)
    check("phát recognition_hit đúng người", len(hits) == 1 and hits[0][0] == person.id,
          f"(hits={hits})")
    check("không phát unknown_face", len(unknowns) == 0)

    # --- (b) KHÔNG có dữ liệu: mọi khuôn mặt là NGƯỜI LẠ + làm mờ ---
    db2 = Database(TEMP_DB)  # DB trống (đã cleanup sau phần a? chưa — xóa người)
    from app.services.person import PersonService
    PersonService(db2).delete(person.id)
    svc2 = RecognitionService(db2)
    svc2.reload()
    check("service rỗng sau xóa", svc2.size == 0)

    worker2 = CameraWorker(0, 640, 480, detector=detector, embedder=embedder,
                           service=svc2, threshold=0.40,
                           anti_spoofing_enabled=False)
    crops: list = []
    worker2.recognition_unknown.connect(lambda crop: crops.append(crop))

    frame = lena.copy()
    x1, y1, x2, y2 = detector.detect(frame)[0].bbox.astype(int)
    # Đo vùng BÊN TRONG (bỏ viền 4px — khung đỏ vẽ trên viền làm nhiễu)
    inner = (slice(y1 + 4, y2 - 4), slice(x1 + 4, x2 - 4))
    region_before = frame[inner].copy()
    has_unknown = worker2._process_frame(frame)
    region_after = frame[inner]
    check("không dữ liệu → báo người lạ", has_unknown is True)
    # Đo độ nét bằng Laplacian variance (metric chuẩn đo độ mờ — Gaussian
    # giữ cấu trúc tổng nên std không giảm mạnh, còn độ nét cạnh giảm mạnh)
    sharp_before = float(cv2.Laplacian(region_before, cv2.CV_64F).var())
    sharp_after = float(cv2.Laplacian(region_after, cv2.CV_64F).var())
    check("vùng mặt bị làm mờ (độ nét giảm mạnh)",
          sharp_after < sharp_before * 0.5,
          f"(laplacian {sharp_before:.0f}→{sharp_after:.0f})")
    check("phát unknown_crop (mẫu cho Đăng ký ngay)", len(crops) == 1
          and crops[0] is not None and crops[0].size > 0)

    lena_path = PROJECT_ROOT / "lena_test.jpg"
    lena_path.unlink(missing_ok=True)
    db.close()
    db2.close()
    cleanup()


# ---------------------------------------------------------
# 3. GUI — CameraView lỗi camera mềm + EnrollmentDialog seed
# ---------------------------------------------------------
def test_gui() -> None:
    print("\n[3] GUI — CameraView (camera 99) + EnrollmentDialog (seed)")
    from PySide6.QtWidgets import QApplication, QMessageBox
    from app.services.auth import AuthService
    from app.services.enrollment import CapturedSample
    from app.ui.camera_view import CameraView
    from app.ui.enrollment_dialog import EnrollmentDialog

    app = QApplication.instance() or QApplication(sys.argv)
    config = Config(camera_index=99, camera_width=320, camera_height=240)
    auth = AuthService(config)

    from PySide6.QtCore import QEventLoop, QTimer

    db = Database(TEMP_DB)
    view = CameraView(config, db=db)
    check("nút [Đăng ký ngay] ẩn khi chưa có người lạ", not view._enroll_now_btn.isVisible())
    check("panel trạng thái hiển thị mặc định",
          "chưa mở" in view._status_label.text().lower())

    # Mở camera 99 → lỗi mềm, không crash.
    # Tín hiệu error từ worker thread là QUEUED → cần chạy event loop để xử lý
    QMessageBox.critical = staticmethod(lambda *a, **k: None)
    view.start_camera()
    loop = QEventLoop()
    poll = QTimer()
    poll.timeout.connect(
        lambda: loop.quit() if view._start_btn.isEnabled() else None
    )
    poll.start(100)
    QTimer.singleShot(20000, loop.quit)  # an toàn: chờ tối đa 20s (nạp model)
    loop.exec()
    check("lỗi camera được xử lý mềm (không crash)", True)
    check("nút Mở webcam được bật lại sau lỗi", view._start_btn.isEnabled())

    # EnrollmentDialog với seed_sample: UI báo mẫu 1 từ webcam
    seed = CapturedSample(
        embedding=np.zeros(512, dtype=np.float32),
        quality=0.9,
        face_crop=np.full((100, 100, 3), 120, dtype=np.uint8),
    )
    dialog = EnrollmentDialog(config, db, detector=None, seed_sample=seed)
    check("dialog seed: video label nhắc mẫu 1 từ webcam",
          "Mẫu 1" in dialog._video_label.text())
    check("dialog seed: bước 1 (Nhìn thẳng) đánh dấu xong",
          "✓ Nhìn thẳng" in dialog._steps_label.text())

    view.close()
    db.close()
    cleanup()


def main() -> None:
    print("=== TEST BƯỚC 9: NHẬN DIỆN THỜI GIAN THỰC HOÀN CHỈNH ===")
    test_service()
    test_worker_frames()
    test_gui()

    print(f"\nKẾT QUẢ: {PASS} qua / {FAIL} thất bại")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
