"""Bước 19 — Temporal Smoothing + Face Tracking (tên không nhấp nháy).

1. TemporalBuffer: 5 khung → 60% vote → hiển thị tên; <60% → None;
   persist 15 khung sau khi mất tín hiệu.
2. FaceTracker: centroid tracking gán track_id ổn định; bbox mới quá xa
   → track mới; biến mất quá lâu → xóa.
3. CameraWorker: dùng TemporalBuffer + FaceTracker → tên ổn định hơn;
   reset tracker/buffers khi stop.
4. SettingsView: slider "Độ ổn định tên" 0–15 khung, hiển thị giá trị.
5. Config: smoothing_window lưu/đọc đúng.

Chạy:  .venv/Scripts/python.exe scripts/step_19_gui_test.py
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

from unittest import mock  # noqa: E402

import numpy as np  # noqa: E402

from app.config import Config  # noqa: E402
from app.core.matcher import MatchResult  # noqa: E402
from app.core.temporal import (  # noqa: E402
    DEFAULT_WINDOW_SIZE,
    FaceTracker,
    PERSIST_FRAMES,
    TemporalBuffer,
)

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
# 1. TemporalBuffer — bỏ phiếu + persist
# ---------------------------------------------------------
def test_temporal_buffer() -> None:
    print("\n[1] TemporalBuffer — bỏ phiếu 60% + persist")
    buf = TemporalBuffer(window_size=5)

    # 5 khung liên tiếp cùng 1 người → hiển thị tên
    for _ in range(5):
        r = buf.add(MatchResult("person_A", 0.85))
    check("5 khung liên tiếp → hiển thị tên A",
          r == "person_A", f"(result={r})")

    # 5 khung liên tiếp khác 1 người → hiển thị tên B
    buf2 = TemporalBuffer(window_size=5)
    for _ in range(5):
        r2 = buf2.add(MatchResult("person_B", 0.80))
    check("5 khung liên tiếp tên B → hiển thị B",
          r2 == "person_B", f"(result={r2})")

    # 3/5 khung tên A, 2/5 None → 60% vote A → hiển thị A
    buf3 = TemporalBuffer(window_size=5)
    results = []
    for mid in [MatchResult("A", 0.8), MatchResult("A", 0.7),
                MatchResult("A", 0.75), None, None]:
        results.append(buf3.add(mid))
    check("3/5 khung A (60%) → hiển thị A",
          any(r == "A" for r in results), f"(results={results})")

    # Persist: mất tín hiệu → giữ tên thêm PERSIST_FRAMES khung
    # Với window_size=3: 3 khung đầu votes vẫn đủ (2/3=66.7% > 60%),
    # persist chỉ bắt đầu khi votes < 60% (sau 2 None: window=[X,None,None])
    buf4 = TemporalBuffer(window_size=3)
    for _ in range(3):
        buf4.add(MatchResult("X", 0.9))
    persist_results = []
    # 2 None đầu: window=[X,X,None]→66.7% và [X,None,None]→33.3% (persist bắt đầu)
    for _ in range(PERSIST_FRAMES + 3):
        persist_results.append(buf4.add(None))
    # Tổng: 3 khung đầu X + 2 None votes đủ + PERSIST_FRAMES persist = 3+2+15=20
    x_count = sum(1 for r in persist_results if r == "X")
    check(f"Persist giữ tên X tổng cộng ~{PERSIST_FRAMES + 2} khung",
          x_count >= PERSIST_FRAMES,
          f"(got {x_count} khung X)")
    # Sau persist hết → trả None
    for _ in range(5):  # thêm vài frame nữa để đảm bảo persist hết
        post = buf4.add(None)
    check("Sau persist hết → trả None", post is None, f"(result={post})")

    # Reset
    buf5 = TemporalBuffer(window_size=3)
    buf5.add(MatchResult("Y", 0.8))
    buf5.add(MatchResult("Y", 0.8))
    buf5.reset()
    r5 = buf5.add(None)
    check("Reset xóa bộ nhớ → không giữ tên cũ", r5 is None, f"(result={r5})")


# ---------------------------------------------------------
# 2. FaceTracker — centroid tracking
# ---------------------------------------------------------
def test_face_tracker() -> None:
    print("\n[2] FaceTracker — centroid tracking")
    tracker = FaceTracker()

    # Frame 1: 2 bbox → 2 track mới
    b1 = [np.array([100, 100, 200, 200]), np.array([400, 100, 500, 200])]
    m1 = tracker.update(b1)
    check("Frame 1: 2 bbox → 2 track mới",
          len(m1) == 2 and len(set(m1.values())) == 2,
          f"(map={m1})")

    # Frame 2: 2 bbox hơi di chuyển → giữ cùng track_id
    b2 = [np.array([105, 105, 205, 205]), np.array([410, 100, 510, 200])]
    m2 = tracker.update(b2)
    check("Frame 2: bbox di chuyển nhẹ → giữ track_id",
          set(m1.values()) == set(m2.values()),
          f"(m1={m1}, m2={m2})")

    # Frame 3: 1 bbox (người thứ 2 ra ngoài) → 1 track biến mất
    b3 = [np.array([110, 110, 210, 210])]
    m3 = tracker.update(b3)
    check("Frame 3: 1 bbox → giữ 1 track, track kia biến mất",
          len(m3) == 1, f"(map={m3})")

    # Reset
    tracker.reset()
    m4 = tracker.update([np.array([100, 100, 200, 200])])
    check("Reset → track mới (id=0)",
          0 in m4.values(), f"(map={m4})")


# ---------------------------------------------------------
# 3. Config smoothing_window
# ---------------------------------------------------------
def test_config() -> None:
    print("\n[3] Config — smoothing_window")
    cfg = Config()
    check("smoothing_window mặc định = 5",
          cfg.smoothing_window == DEFAULT_WINDOW_SIZE == 5,
          f"(value={cfg.smoothing_window})")


# ---------------------------------------------------------
# 4. CameraWorker — temporal smoothing tích hợp
# ---------------------------------------------------------
def test_camera_worker_temporal() -> None:
    print("\n[4] CameraWorker — temporal smoothing tích hợp")
    from app.ui.camera_view import CameraWorker

    # Tạo fake detector + embedder + service
    detector = mock.MagicMock()
    embedder = mock.MagicMock()
    service = mock.MagicMock()

    fake_bbox = np.array([100, 100, 300, 300])
    fake_landmark = np.zeros((68, 3), dtype=np.float32)
    fake_face = mock.MagicMock()
    fake_face.bbox = fake_bbox
    fake_face.landmark_3d_68 = fake_landmark
    fake_face.det_score = 0.9

    detector.detect.return_value = [fake_face]

    # Mock match trả kết quả ổn định
    stable_result = MatchResult("user_1", 0.82)
    service.match.return_value = stable_result
    service.label_of.return_value = "Nguyễn Văn A"

    # Tạo worker với smoothing_window=3
    w = CameraWorker(
        0, 640, 480,
        detector=detector, embedder=embedder, service=service,
        threshold=0.40, smoothing_window=3,
        anti_spoofing_enabled=False,
    )

    hits = []
    w.recognition_hit.connect(lambda pid, sim, crop: hits.append(pid))

    # Chạy 5 frame liên tiếp → smoothing cần ≥3/5 vote
    for _ in range(5):
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        w._process_frame(fake_frame)

    check("5 frame match ổn định → ít nhất 1 hit",
          len(hits) >= 1, f"(hits={len(hits)})")

    # Kiểm tra tracker hoạt động
    check("FaceTracker có track",
          len(w._tracker._tracks) > 0, f"(tracks={len(w._tracker._tracks)})")

    # Stop → reset tracker + buffers
    w.stop()
    check("Stop → tracker reset", len(w._tracker._tracks) == 0)
    check("Stop → buffers clear", len(w._buffers) == 0)


# ---------------------------------------------------------
# 5. SettingsView — slider độ ổn định
# ---------------------------------------------------------
def test_settings_slider() -> None:
    print("\n[5] SettingsView — slider độ ổn định tên")
    from PySide6.QtWidgets import QApplication

    from app.services.auth import AuthService
    from app.services.sync import SyncService
    from app.infrastructure.db import Database

    app = QApplication.instance() or QApplication(sys.argv)
    import app.config as cfg_mod
    from pathlib import Path
    temp_config = PROJECT_ROOT / "data" / "test_config_19.json"
    cfg_mod.CONFIG_PATH = temp_config

    db = Database(PROJECT_ROOT / "data" / "test_19.db")
    config = Config()
    from app.ui.settings_view import SettingsView
    view = SettingsView(config, AuthService(config), SyncService(db, config))

    check("SettingsView có slider smoothing_window",
          hasattr(view, "_smoothing_slider"))
    check("Slider range 0–15",
          view._smoothing_slider.minimum() == 0 and
          view._smoothing_slider.maximum() == 15)

    # Đổi slider → label cập nhật
    view._smoothing_slider.setValue(0)
    view._update_smoothing_label()
    check("Slider=0 → label 'Tắt'",
          view._smoothing_label.text() == "Tắt")

    view._smoothing_slider.setValue(7)
    view._update_smoothing_label()
    check("Slider=7 → label '7 khung'",
          view._smoothing_label.text() == "7 khung")

    view.close()
    db.close()
    temp_config.unlink(missing_ok=True)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main() -> None:
    print("=== TEST BƯỚC 19: TEMPORAL SMOOTHING + FACE TRACKING ===")
    test_temporal_buffer()
    test_face_tracker()
    test_config()
    test_camera_worker_temporal()
    test_settings_slider()
    print(f"\nKẾT QUẢ: {PASS} qua / {FAIL} thất bại")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
