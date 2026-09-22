"""Bước 17 — Kiểm tra chống giả mạo (anti-spoofing, FR-8).

1. LivenessTracker (unit, KHÔNG cần GPU): phát hiện chu kỳ chớp mắt
   mở→nhắm→mở; cửa sổ trượt 10 giây + KHOAN DUNG 6 giây (mới theo dõi
   chưa báo vội); reset khi người biến mất >5s; nhắm quá lâu (>1s) không đếm.
2. CameraWorker (GPU thật, dùng ảnh lena TĨNH):
   - Bật chống giả mạo + ảnh tĩnh không chớp mắt → KHÔNG phát recognition_hit,
     phát spoof_suspected + khuôn mặt bị làm mờ (chặn, coi như người lạ).
   - Tracker đã có chớp mắt gần đây → nhận diện bình thường (hit).
   - Tắt chống giả mạo → hành vi cũ (hit) — hồi quy.
3. GUI: SettingsView có checkbox "Chống giả mạo" (mặc định TẮT — người
   dùng thật không muốn bị làm phiền), bật + Lưu → config.anti_spoofing_enabled
   = True; Khôi phục mặc định → False.
4. RecognitionService.save_event(is_spoof=True) → label "Giả mạo: <tên>",
   giữ person_id, is_unknown = 0.

Chạy:  .venv\\Scripts\\python.exe scripts\\step_17_gui_test.py
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

from unittest import mock  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import app.config as cfg_mod  # noqa: E402
import app.ui.settings_view as sv  # noqa: E402
from app.config import Config  # noqa: E402
from app.core.detector import FaceDetector, MODELS_ROOT  # noqa: E402
from app.core.embedder import FaceEmbedder  # noqa: E402
from app.core.liveness import LivenessTracker  # noqa: E402
from app.infrastructure.db import DATA_DIR, Database  # noqa: E402
from app.infrastructure.repositories import RecognitionEventRepository  # noqa: E402
from app.services.enrollment import CapturedSample, EnrollmentService  # noqa: E402
from app.services.recognition import RecognitionService  # noqa: E402

TEMP_DB = PROJECT_ROOT / "data" / "test_anti_spoof.db"
TEMP_THUMBS = PROJECT_ROOT / "data" / "thumbs_test17"
TEMP_CONFIG = PROJECT_ROOT / "data" / "test_config_17.json"

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
    TEMP_CONFIG.unlink(missing_ok=True)


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
# 1. LivenessTracker — unit test (không GPU)
# ---------------------------------------------------------
def test_liveness_tracker() -> None:
    """Phát hiện chu kỳ chớp mắt + cửa sổ trượt (mô phỏng EAR bằng mock)."""
    print("\n[1] LivenessTracker — chu kỳ chớp mắt + cửa sổ trượt")

    class FakeClock:
        """Đồng hồ giả — test cửa sổ 3s mà không phải chờ thật."""

        def __init__(self) -> None:
            self.now = 0.0

        def __call__(self) -> float:
            return self.now

    clock = FakeClock()

    # (a) Mắt MỞ liên tục (không bao giờ nhắm): mới theo dõi → còn KHOAN
    # DUNG (tạm coi là sống — người thật vừa vào khung chưa kịp chớp);
    # hết 6s khoan dung mà vẫn không chớp → KHÔNG sống (nghi giả mạo)
    tracker = LivenessTracker(now_fn=clock)
    with mock.patch(
        "app.core.face_metrics.eye_aspect_ratio", return_value=0.40
    ):
        for _ in range(5):
            clock.now += 0.1
            tracker.update(object())
    check("mở liên tục nhưng còn trong khoan dung → tạm sống", tracker.is_live())
    clock.now = 10.0  # đã theo dõi > 6s (hết khoan dung) mà không chớp
    check("mở liên tục, hết khoan dung → không sống", not tracker.is_live())

    # (b) Chu kỳ MỞ → NHẮM → MỞ = 1 lần chớp → SỐNG
    tracker = LivenessTracker(now_fn=clock)
    clock.now = 0.0
    with mock.patch(
        "app.core.face_metrics.eye_aspect_ratio",
        side_effect=[0.40, 0.10, 0.40],  # mở → nhắm → mở
    ):
        clock.now = 0.1
        tracker.update(object())
        clock.now = 0.2
        tracker.update(object())
        clock.now = 0.3
        tracker.update(object())
    check("chu kỳ mở→nhắm→mở → sống", tracker.is_live(),
          f"(blink={tracker.is_live()})")

    # (c) Nhắm QUÁ LÂU (>1s) → không phải chớp → không đếm
    tracker = LivenessTracker(now_fn=clock)
    clock.now = 0.0
    with mock.patch(
        "app.core.face_metrics.eye_aspect_ratio",
        side_effect=[0.40, 0.10, 0.10, 0.40],  # mở → nhắm (2s) → mở
    ):
        clock.now = 0.1
        tracker.update(object())
        clock.now = 0.2
        tracker.update(object())
        clock.now = 2.2   # nhắm 2 giây — quá lâu
        tracker.update(object())
        clock.now = 2.3
        tracker.update(object())
    clock.now = 7.0  # hết khoan dung 6s — xác minh không có chớp nào được đếm
    check("nhắm >1s → không tính là chớp", not tracker.is_live())

    # (d) Cửa sổ trượt: chớp xong rồi HẾT 3 giây → hết sống
    tracker = LivenessTracker(now_fn=clock)
    clock.now = 0.0
    with mock.patch(
        "app.core.face_metrics.eye_aspect_ratio",
        side_effect=[0.40, 0.10, 0.40],  # 1 lần chớp tại t≈0.3
    ):
        clock.now = 0.1
        tracker.update(object())
        clock.now = 0.2
        tracker.update(object())
        clock.now = 0.3
        tracker.update(object())
    check("vừa chớp xong → sống", tracker.is_live())
    clock.now = 11.0  # hơn 10 giây kể từ lần chớp cuối (0.3) + hết khoan dung
    check("hết cửa sổ 10s → không còn sống", not tracker.is_live())

    # (e) EAR = 0 (không đo được landmark) → bỏ qua, không crash, không sống
    tracker = LivenessTracker(now_fn=clock)
    clock.now = 0.0
    with mock.patch("app.core.face_metrics.eye_aspect_ratio", return_value=0.0):
        tracker.update(object())
    check("EAR=0 (thiếu landmark) → bỏ qua, không sống", not tracker.is_live())

    # (f) Mắt NHẮM NGAY TỪ FRAME ĐẦU (chưa từng thấy mở) → nhắm NHIỀU frame
    #     liên tục KHÔNG crash (regression: bug "now - None" làm chết luồng
    #     camera) và khi mở lại KHÔNG tính là chớp (chưa có bằng chứng mở)
    tracker = LivenessTracker(now_fn=clock)
    clock.now = 0.0
    with mock.patch(
        "app.core.face_metrics.eye_aspect_ratio",
        side_effect=[0.10, 0.10, 0.10, 0.40],  # nhắm (3 frame) → mở
    ):
        clock.now = 0.1
        tracker.update(object())   # frame đầu: unknown → closed
        clock.now = 1.5            # vẫn nhắm — trước đây crash tại đây
        tracker.update(object())
        clock.now = 3.0            # nhắm đã vượt MAX_BLINK_DURATION (1s)
        tracker.update(object())
        clock.now = 3.1
        tracker.update(object())   # mở lại → không đếm chớp
    clock.now = 7.0  # hết khoan dung 6s — xác minh không crash + không đếm
    check("nhắm ngay từ frame đầu, nhắm lâu → không crash, không sống",
          not tracker.is_live())

    # (g) KHOAN DUNG: theo dõi 3s chưa thấy chớp → VẪN sống (không báo vội)
    tracker = LivenessTracker(now_fn=clock)
    clock.now = 0.0
    with mock.patch(
        "app.core.face_metrics.eye_aspect_ratio", return_value=0.40
    ):
        for _ in range(3):
            clock.now += 1.0
            tracker.update(object())
    check("theo dõi 3s chưa chớp → còn khoan dung (chưa báo giả mạo)",
          tracker.is_live())

    # (h) Người BIẾN MẤT > 5s rồi quay lại → quan sát MỚI (khoan dung lại)
    tracker = LivenessTracker(now_fn=clock)
    clock.now = 0.0
    with mock.patch("app.core.face_metrics.eye_aspect_ratio", return_value=0.40):
        clock.now = 1.0
        tracker.update(object())
        clock.now = 2.0
        tracker.update(object())
    clock.now = 8.0  # biến mất 6s (> RESET_SECONDS = 5s)
    with mock.patch("app.core.face_metrics.eye_aspect_ratio", return_value=0.40):
        tracker.update(object())
    check("biến mất >5s rồi quay lại → khoan dung tính lại (chưa báo vội)",
          tracker.is_live())


# ---------------------------------------------------------
# 2. CameraWorker — ảnh tĩnh bị chặn / có chớp mắt thì nhận diện (GPU)
# ---------------------------------------------------------
def test_worker_anti_spoof() -> None:
    """Bật chống giả mạo: ảnh TĨNH → spoof; tracker có chớp → hit."""
    print("\n[2] CameraWorker — ảnh tĩnh bị chặn (giả mạo) / có chớp → nhận diện")
    from app.ui.camera_view import CameraWorker

    detector = FaceDetector(MODELS_ROOT)
    embedder = FaceEmbedder(detector)

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

    # --- (a) Bật chống giả mạo + ảnh TĨNH (không chớp mắt) → SPOOF ---
    hits: list = []
    spoofs: list = []
    worker = CameraWorker(
        0, 640, 480,
        detector=detector, embedder=embedder, service=svc, threshold=0.40,
        anti_spoofing_enabled=True,
    )
    worker.recognition_hit.connect(
        lambda pid, sim, crop: hits.append((pid, sim))
    )
    worker.spoof_suspected.connect(
        lambda pid, sim, crop: spoofs.append((pid, sim))
    )

    frame = lena.copy()
    x1, y1, x2, y2 = detector.detect(frame)[0].bbox.astype(int)
    inner = (slice(y1 + 4, y2 - 4), slice(x1 + 4, x2 - 4))
    sharp_before = float(cv2.Laplacian(frame[inner], cv2.CV_64F).var())

    # Mô phỏng: khuôn mặt này đã được theo dõi LIÊN TỤC > 6s mà KHÔNG có
    # lần chớp mắt nào (như ảnh tĩnh giơ trước camera) → hết khoan dung,
    # đủ căn cứ nghi giả mạo ngay từ frame xử lý đầu tiên.
    clock = [0.0]

    def fake_now() -> float:
        return clock[0]

    stale_tracker = LivenessTracker(now_fn=fake_now)
    with mock.patch(
        "app.core.face_metrics.eye_aspect_ratio", return_value=0.40
    ):
        for t in range(0, 11):  # 0..10 giây mắt mở liên tục, không chớp
            clock[0] = float(t)
            stale_tracker.update(object())
    check("theo dõi 10s không chớp → hết khoan dung, không sống",
          not stale_tracker.is_live())
    worker._liveness[person.id] = stale_tracker

    has_unknown = worker._process_frame(frame)
    sharp_after = float(cv2.Laplacian(frame[inner], cv2.CV_64F).var())

    check("ảnh tĩnh + chống giả mạo → KHÔNG phát hit", len(hits) == 0,
          f"(hits={hits})")
    check("ảnh tĩnh + chống giả mạo → phát spoof_suspected",
          len(spoofs) == 1 and spoofs[0][0] == person.id,
          f"(spoofs={spoofs})")
    check("khuôn mặt nghi giả mạo bị LÀM MỜ (chặn hiển thị)",
          sharp_after < sharp_before * 0.5,
          f"(laplacian {sharp_before:.0f}→{sharp_after:.0f})")
    check("nghi giả mạo KHÔNG báo 'người lạ' (không mở nút Đăng ký ngay)",
          has_unknown is False)

    # --- (b) Tracker đã có CHỚP MẮT gần đây → nhận diện bình thường ---
    clock_times: list[float] = [0.0]

    def fake_now() -> float:
        return clock_times[0]

    live_tracker = LivenessTracker(now_fn=fake_now)
    with mock.patch(
        "app.core.face_metrics.eye_aspect_ratio",
        side_effect=[0.40, 0.10, 0.40],  # giả lập 1 lần chớp
    ):
        for t in (0.1, 0.2, 0.3):
            clock_times[0] = t
            live_tracker.update(object())
    check("tracker giả lập đã có chớp → sống", live_tracker.is_live())

    hits2: list = []
    worker2 = CameraWorker(
        0, 640, 480,
        detector=detector, embedder=embedder, service=svc, threshold=0.40,
        anti_spoofing_enabled=True,
    )
    worker2.recognition_hit.connect(
        lambda pid, sim, crop: hits2.append((pid, sim))
    )
    # Gán tracker ĐÃ SỐNG cho người này — như người thật vừa chớp mắt
    worker2._liveness[person.id] = live_tracker
    worker2._process_frame(lena.copy())
    check("đã chớp mắt gần đây → nhận diện bình thường (hit)",
          len(hits2) == 1 and hits2[0][0] == person.id,
          f"(hits2={hits2})")

    # --- (c) Tắt chống giả mạo → hành vi cũ (hit) — hồi quy ---
    hits3: list = []
    worker3 = CameraWorker(
        0, 640, 480,
        detector=detector, embedder=embedder, service=svc, threshold=0.40,
        anti_spoofing_enabled=False,
    )
    worker3.recognition_hit.connect(
        lambda pid, sim, crop: hits3.append((pid, sim))
    )
    worker3._process_frame(lena.copy())
    check("tắt chống giả mạo → hit như cũ (hồi quy)",
          len(hits3) == 1 and hits3[0][0] == person.id,
          f"(hits3={hits3})")

    lena_path = PROJECT_ROOT / "lena_test.jpg"
    lena_path.unlink(missing_ok=True)
    db.close()
    cleanup()


# ---------------------------------------------------------
# 3. GUI — SettingsView checkbox Chống giả mạo
# ---------------------------------------------------------
def test_gui_checkbox() -> None:
    """Checkbox 'Chống giả mạo' trong Cài đặt: mặc định BẬT, lưu, reset."""
    print("\n[3] GUI — SettingsView checkbox Chống giả mạo")
    from PySide6.QtWidgets import QApplication
    from app.services.auth import AuthService
    from app.services.sync import SyncService

    app = QApplication.instance() or QApplication(sys.argv)
    cfg_mod.CONFIG_PATH = TEMP_CONFIG

    config = Config()
    auth = AuthService(config)
    sync = SyncService(Database(TEMP_DB), config)
    view = sv.SettingsView(config, auth, sync)

    check("checkbox tồn tại trong giao diện",
          hasattr(view, "_spoof_check"))
    check("mặc định TẮT (config mặc định False)",
          view._spoof_check.isChecked() is False)

    # Bật + Lưu → config.anti_spoofing_enabled = True (file tạm)
    view._spoof_check.setChecked(True)
    orig_info = sv.QMessageBox.information
    sv.QMessageBox.information = staticmethod(lambda *a, **k: None)
    try:
        view._on_save()
    finally:
        sv.QMessageBox.information = orig_info
    check("lưu: config.anti_spoofing_enabled = True",
          config.anti_spoofing_enabled is True)
    loaded = Config.load()
    check("đọc lại từ file tạm: True", loaded.anti_spoofing_enabled is True)

    # Khôi phục mặc định → checkbox về TẮT (chưa lưu)
    view._on_reset_defaults()
    check("khôi phục mặc định → checkbox TẮT lại",
          view._spoof_check.isChecked() is False)

    view.close()
    cleanup()


# ---------------------------------------------------------
# 4. RecognitionService — save_event(is_spoof=True)
# ---------------------------------------------------------
def test_save_event_spoof() -> None:
    """Sự kiện giả mạo: label 'Giả mạo: <tên>', giữ person_id, is_unknown=0."""
    print("\n[4] save_event(is_spoof=True) — ghi sự kiện giả mạo vào lịch sử")
    from app.infrastructure.repositories import PersonRepository

    db = Database(TEMP_DB)
    svc = RecognitionService(db)
    people = PersonRepository(db)

    person = people.add("Nguyễn Văn A", thumbnail_path="")
    crop = np.full((100, 100, 3), 120, dtype=np.uint8)

    event_id = svc.save_event(
        person_id=person.id,
        similarity=0.92,
        face_crop=crop,
        is_spoof=True,
    )
    check("save_event(is_spoof=True) trả id", bool(event_id))

    with db.session() as conn:
        row = conn.execute(
            "SELECT label, person_id, is_unknown, similarity FROM recognition_events"
            " WHERE id = ?", (event_id,),
        ).fetchone()
    check("label = 'Giả mạo: Nguyễn Văn A'",
          row["label"] == "Giả mạo: Nguyễn Văn A",
          f"(label={row['label']!r})")
    check("giữ person_id (lịch sử biết ai bị nghi ngờ)",
          row["person_id"] == person.id)
    check("is_unknown = 0 (vẫn là người đã đăng ký)",
          row["is_unknown"] == 0)
    check("giữ similarity", abs(row["similarity"] - 0.92) < 1e-9)

    events = RecognitionEventRepository(db)
    latest = events.list_events(limit=1)
    check("sự kiện xuất hiện trong Lịch sử (list_events)",
          len(latest) == 1 and latest[0].label.startswith("Giả mạo:"))

    db.close()
    cleanup()


def main() -> None:
    print("=== TEST BƯỚC 17: CHỐNG GIẢ MẠO (ANTI-SPOOFING) ===")
    cleanup()
    test_liveness_tracker()
    test_worker_anti_spoof()
    test_gui_checkbox()
    test_save_event_spoof()
    cleanup()

    print(f"\nKẾT QUẢ: {PASS} qua / {FAIL} thất bại")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
