"""Bước 10 — Kiểm tra PhotoView (nhận diện ảnh tĩnh, chế độ minimal, tự thoát).

1. PhotoWorker (GPU THẬT): ảnh lena (đã đăng ký) → 1 khuôn mặt ĐÃ BIẾT;
   ảnh nhóm 6 người → 6 khuôn mặt NGƯỜI LẠ (mỗi người lạ có embedding+crop).
2. Sự kiện source='photo': save_event ghi đúng cột source.
3. GUI (stub detector — không GPU): mở ảnh → [Nhận diện] → bảng kết quả
   + nút [Đăng ký ngay] cho người lạ; bấm đăng ký → dialog nhận seed + chạy lại.

Chạy:  .venv\\Scripts\\python.exe scripts\\step_10_gui_test.py
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
from app.services.enrollment import CapturedSample, EnrollmentService  # noqa: E402
from app.services.recognition import RecognitionService  # noqa: E402

TEMP_DB = PROJECT_ROOT / "data" / "test_photo.db"
TEMP_THUMBS = PROJECT_ROOT / "data" / "thumbs_test10"
TEMP_IMG = PROJECT_ROOT / "data" / "test_photo_input.jpg"

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
    Path(str(TEMP_IMG)).unlink(missing_ok=True)


def load_lena() -> np.ndarray:
    lena_path = PROJECT_ROOT / "lena_test.jpg"
    if not lena_path.exists() or lena_path.stat().st_size == 0:
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
            lena_path,
        )
    return cv2.imread(str(lena_path))


# ---------------------------------------------------------
# 1. PhotoWorker — nhận diện ảnh thật (GPU)
# ---------------------------------------------------------
def test_worker() -> None:
    print("\n[1] PhotoWorker — ảnh thật (lena + ảnh nhóm)")
    from app.ui.photo_view import PhotoWorker

    db = Database(TEMP_DB)
    svc = RecognitionService(db)
    detector = FaceDetector(MODELS_ROOT)
    embedder = FaceEmbedder(detector)

    lena = load_lena()
    lena_face = detector.detect(lena)[0]
    emb_lena = embedder.embed_face(lena_face)
    EnrollmentService(db, thumbs_dir=TEMP_THUMBS).save_person(
        "Nguyễn Văn A",
        [CapturedSample(embedding=emb_lena, quality=0.9, face_crop=lena)],
    )
    svc.reload()

    # (a) ảnh lena → 1 mặt ĐÃ BIẾT
    worker = PhotoWorker(lena, detector, embedder, svc, threshold=0.40)
    out = {}
    worker.finished.connect(lambda frame, results: out.update(frame=frame, results=results))
    worker.run()  # chạy đồng bộ (test) — thực tế chạy trong QThread
    results = out.get("results", [])
    check("lena: có 1 khuôn mặt", len(results) == 1, f"(={len(results)})")
    check("lena: là người đã biết", results and not results[0].is_unknown
          and results[0].similarity > 0.8,
          f"(sim={results[0].similarity:.3f})" if results else "")
    check("lena: ảnh đã vẽ khung (frame thay đổi)",
          "frame" in out and not np.array_equal(out["frame"], lena))

    # (b) ảnh nhóm 6 người → tất cả NGƯỜI LẠ (chỉ đăng ký lena)
    group = cv2.imread(sys.prefix + "/Lib/site-packages/insightface/data/images/t1.jpg")
    worker2 = PhotoWorker(group, detector, embedder, svc, threshold=0.40)
    out2 = {}
    worker2.finished.connect(lambda frame, results: out2.update(frame=frame, results=results))
    worker2.run()
    results2 = out2.get("results", [])
    check("ảnh nhóm: ≥ 2 khuôn mặt", len(results2) >= 2, f"(={len(results2)})")
    check("ảnh nhóm: toàn người lạ (chưa đăng ký)",
          all(r.is_unknown for r in results2))
    check("mỗi người lạ có embedding + crop (cho Đăng ký ngay)",
          all(r.embedding is not None and r.crop is not None for r in results2))

    lena_path = PROJECT_ROOT / "lena_test.jpg"
    lena_path.unlink(missing_ok=True)
    db.close()
    cleanup()


# ---------------------------------------------------------
# 2. Sự kiện source='photo'
# ---------------------------------------------------------
def test_event_source() -> None:
    print("\n[2] Sự kiện nhận diện ảnh — source='photo'")
    db = Database(TEMP_DB)
    svc = RecognitionService(db)
    crop = np.full((80, 80, 3), 100, dtype=np.uint8)

    svc.save_event(person_id=None, similarity=None, face_crop=crop,
                   is_unknown=True, source="photo")
    with db.session() as conn:
        row = conn.execute(
            "SELECT source, is_unknown, label FROM recognition_events"
            " WHERE source = 'photo'"
        ).fetchone()
    check("sự kiện lưu với source='photo'", row is not None)
    check("người lạ ảnh: is_unknown=1, label đúng",
          row["is_unknown"] == 1 and row["label"] == "Người lạ")

    db.close()
    cleanup()


# ---------------------------------------------------------
# 3. GUI — PhotoView với stub detector (không GPU)
# ---------------------------------------------------------
class FakeFace:
    def __init__(self, x: int, y: int, size: int) -> None:
        self.bbox = np.array([x, y, x + size, y + size], dtype=np.float32)
        self.det_score = 0.9
        emb = np.random.default_rng(0).standard_normal(512).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        self.normed_embedding = emb


class StubDetector:
    """Detector giả: luôn trả N khuôn mặt (đếm số lần detect).

    Có thuộc tính ``app`` để FaceEmbedder khởi tạo được (embed_face chỉ đọc
    face.normed_embedding nên không cần GPU thật).
    """

    def __init__(self, n_faces: int = 2) -> None:
        self.n_faces = n_faces
        self.calls = 0
        self.app = None  # FaceEmbedder chỉ lưu lại, không dùng

    def detect(self, img):
        self.calls += 1
        return [FakeFace(10 + i * 80, 10, 60) for i in range(self.n_faces)]


def test_gui() -> None:
    print("\n[3] GUI — PhotoView (mở ảnh → nhận diện → đăng ký ngay)")
    from PySide6.QtCore import QEventLoop, QTimer
    from PySide6.QtWidgets import QApplication, QDialog, QPushButton

    import app.ui.photo_view as pv
    from app.ui.photo_view import PhotoView

    app = QApplication.instance() or QApplication(sys.argv)
    config = Config(camera_index=99, camera_width=320, camera_height=240)

    # Tạo ảnh test trên đĩa
    cv2.imwrite(str(TEMP_IMG), np.full((200, 300, 3), 128, dtype=np.uint8))
    lena_path = PROJECT_ROOT / "lena_test.jpg"
    if not lena_path.exists() or lena_path.stat().st_size == 0:
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
            lena_path,
        )

    db = Database(TEMP_DB)
    stub = StubDetector(n_faces=2)
    view = PhotoView(config, db, detector=stub)

    # Mở ảnh qua QFileDialog giả
    orig_getfile = pv.QFileDialog.getOpenFileName
    pv.QFileDialog.getOpenFileName = staticmethod(
        lambda *a, **k: (str(lena_path), "")
    )
    view._open_image()
    pv.QFileDialog.getOpenFileName = orig_getfile
    check("mở ảnh: ảnh đã tải + nút Nhận diện bật",
          view._image is not None and view._analyze_btn.isEnabled())
    check("mở ảnh: hiển thị tên file", "lena" in view._file_label.text())

    # Nhận diện → chờ worker xong (queued signal cần event loop)
    view._analyze()
    loop = QEventLoop()
    poll = QTimer()
    poll.timeout.connect(
        lambda: loop.quit() if "Đã lưu" in view._status_label.text() else None
    )
    poll.start(100)
    QTimer.singleShot(15000, loop.quit)
    loop.exec()

    check("nhận diện xong: 2 khuôn mặt trong bảng kết quả",
          view._results_list.count() == 2)
    check("nhận diện xong: thông báo đã lưu sự kiện",
          "Đã lưu 2 sự kiện" in view._status_label.text(),
          f"(text={view._status_label.text()!r})")

    # Có nút [Đăng ký ngay] trên từng người lạ
    enroll_buttons = 0
    for i in range(view._results_list.count()):
        row = view._results_list.itemWidget(view._results_list.item(i))
        buttons = row.findChildren(QPushButton)
        enroll_buttons += sum(1 for b in buttons if "Đăng ký" in b.text())
    check("có nút [Đăng ký ngay] cho từng người lạ", enroll_buttons == 2,
          f"(={enroll_buttons})")

    # Bấm [Đăng ký ngay] → EnrollmentDialog nhận seed + chạy lại nhận diện
    captured: dict = {}
    class FakeDialog:
        def __init__(self, config, db, detector=None, seed_sample=None, parent=None):
            captured["seed"] = seed_sample
        def exec(self):
            return QDialog.DialogCode.Accepted

    orig_dialog = pv.EnrollmentDialog
    pv.EnrollmentDialog = FakeDialog
    analyze_calls = []
    orig_analyze = view._analyze
    view._analyze = lambda: analyze_calls.append(1)

    first_result = view._results[0]
    view._on_enroll(first_result)
    check("đăng ký ngay: dialog nhận seed (ảnh crop + embedding)",
          captured.get("seed") is not None
          and captured["seed"].embedding is not None
          and captured["seed"].face_crop is not None)
    check("đăng ký ngay: chạy lại nhận diện", len(analyze_calls) == 1)

    pv.EnrollmentDialog = orig_dialog
    view._analyze = orig_analyze

    lena_path.unlink(missing_ok=True)
    db.close()
    cleanup()


def main() -> None:
    print("=== TEST BƯỚC 10: NHẬN DIỆN ẢNH TĨNH (PHOTOVIEW) ===")
    test_worker()
    test_event_source()
    test_gui()

    print(f"\nKẾT QUẢ: {PASS} qua / {FAIL} thất bại")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
