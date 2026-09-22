"""Bước 18 — UI/UX + Bảo mật nhận diện (mặt bị che / đeo kính).

1. face_metrics.occlusion_score (unit, không GPU): det_score thấp / landmark
   ngoài khung → điểm cao; mặt rõ → điểm thấp; ngưỡng WARN < BLOCK.
2. GPU (lena thật): mặt rõ < WARN; che 1 bên ≥ WARN; khẩu trang (che nửa
   dưới) → detector KHÔNG tìm thấy mặt (tự chặn khi đăng ký).
3. Enrollment: _pick_best_face TỪ CHỐI mặt bị che (occlusion cao → None),
   chấp nhận mặt rõ; dialog có tip đeo kính.
4. CameraWorker: occlusion cao → KHÔNG hit (chặn, không nhận diện sai);
   occlusion vừa → hit kèm cảnh báo; mặt rõ → hit bình thường.
5. PhotoWorker: kết quả đánh dấu occluded + label '⚠ bị che'.
6. UI: các tiêu đề sectionTitle + theme có style sectionTitle.

Chạy:  .venv\\Scripts\\python.exe scripts\\step_18_uiux_test.py
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
from app.core import face_metrics  # noqa: E402
from app.core.detector import FaceDetector, MODELS_ROOT  # noqa: E402
from app.core.embedder import FaceEmbedder  # noqa: E402
from app.infrastructure.db import DATA_DIR, Database  # noqa: E402
from app.services.enrollment import CapturedSample, EnrollmentService  # noqa: E402
from app.services.recognition import RecognitionService  # noqa: E402
from app.ui.enrollment_dialog import EnrollmentDialog, EnrollmentWorker  # noqa: E402
from app.ui.photo_view import PhotoWorker  # noqa: E402

TEMP_DB = PROJECT_ROOT / "data" / "test_uiux.db"
TEMP_THUMBS = PROJECT_ROOT / "data" / "thumbs_test18"
TEMP_CONFIG = PROJECT_ROOT / "data" / "test_config_18.json"

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


def make_fake_face(
    det_score: float,
    outside_count: int = 0,
    warp_eye: str | None = None,
) -> object:
    """Fake Face: landmark 68 điểm bên trong bbox; đẩy N điểm ra ngoài.

    QUAN TRỌNG: vùng MẮT (36-47) đặt TƯỜNG MINH với EAR HỢP LÝ (~0.4)
    — mắt thật mở: EAR ≈ 0.3-0.6.

    warp_eye='left' | 'right': bóp méo 1 bên mắt (EAR vọt > 0.7) để
    kích hoạt eye_anomaly — mô phỏng vật cản che 1 bên mặt.
    """
    bbox = np.array([100, 100, 300, 300])
    lm = np.zeros((68, 3), dtype=np.float32)
    # Lưới cơ sở: y theo hàng (mỗi hàng 17 điểm, cách 20px), x tăng đều
    for i in range(68):
        lm[i] = [140 + (i % 17) * 6, 120 + (i // 17) * 22, 0]
    # MẮT TRÁI (36-41): khóe 36=(160,170), 39=(180,170) → rộng 20
    #                     trên/dưới: 37,41 và 38,40 lệch ±4px dọc
    lm[36] = [160, 170, 0]
    lm[37] = [164, 166, 0]
    lm[38] = [172, 166, 0]
    lm[39] = [180, 170, 0]
    lm[40] = [172, 174, 0]
    lm[41] = [164, 174, 0]
    # MẮT PHẢI (42-47): khóe 42=(190,170), 45=(210,170)
    lm[42] = [190, 170, 0]
    lm[43] = [194, 166, 0]
    lm[44] = [202, 166, 0]
    lm[45] = [210, 170, 0]
    lm[46] = [202, 174, 0]
    lm[47] = [194, 174, 0]
    # Bóp méo 1 bên mắt (EAR vọt > 0.7 — mô phỏng landmark bị kéo méo
    # khi vật cản che 1 bên mặt)
    if warp_eye == "left":
        lm[37] = [164, 160, 0]  # kéo lên cao → EAR tăng mạnh
        lm[41] = [164, 180, 0]  # kéo xuống thấp
        lm[38] = [172, 160, 0]
        lm[40] = [172, 180, 0]
    elif warp_eye == "right":
        lm[43] = [194, 160, 0]
        lm[47] = [194, 180, 0]
        lm[44] = [202, 160, 0]
        lm[46] = [202, 180, 0]
    for i in range(min(outside_count, 68)):
        lm[i][0] = bbox[0] - 30.0  # đẩy ra ngoài bên trái khung
    face = type("FakeFace", (), {})()
    face.bbox = bbox
    face.det_score = det_score
    face.landmark_3d_68 = lm
    return face


# ---------------------------------------------------------
# 1. occlusion_score — unit (không GPU)
# ---------------------------------------------------------
def test_occlusion_unit() -> None:
    print("\n[1] occlusion_score — đơn vị (fake face)")
    clean = make_fake_face(det_score=0.85)
    check("mặt RÕ (det cao, landmark trong khung) → score < WARN",
          face_metrics.occlusion_score(clean) < face_metrics.OCCLUSION_WARN,
          f"(score={face_metrics.occlusion_score(clean):.3f})")
    # det 0.60 + outside 4 → det_comp=0.88, out_comp=0.29
    # total = 0.35*0.88 + 0.05*0.29 = 0.323 → WARN
    warn = make_fake_face(det_score=0.60, outside_count=4)
    check("mặt HƠI CHE (det vừa + landmark lệch) → WARN ≤ score < BLOCK",
          face_metrics.OCCLUSION_WARN <= face_metrics.occlusion_score(warn)
          < face_metrics.OCCLUSION_BLOCK,
          f"(score={face_metrics.occlusion_score(warn):.3f})")
    # det 0.50 + outside 20 + eye warp → det=1.0, out=1.0, eye=1.0
    # total = 0.35 + 0.05 + 0.20 = 0.60 → BLOCK
    blocked = make_fake_face(
        det_score=0.50, outside_count=20, warp_eye="left",
    )
    check("mặt CHE NHIỀU (det thấp + landmark lệch + mắt méo) → score ≥ BLOCK",
          face_metrics.occlusion_score(blocked) >= face_metrics.OCCLUSION_BLOCK,
          f"(score={face_metrics.occlusion_score(blocked):.3f})")
    check("ngưỡng: WARN < BLOCK",
          face_metrics.OCCLUSION_WARN < face_metrics.OCCLUSION_BLOCK)


# ---------------------------------------------------------
# 2. GPU — lena thật + vật cản giả
# ---------------------------------------------------------
def test_occlusion_gpu() -> None:
    print("\n[2] occlusion GPU — lena thật + vật cản giả")
    detector = FaceDetector(MODELS_ROOT)
    lena = load_lena()
    f = detector.detect(lena)[0]
    score = face_metrics.occlusion_score(f)
    check("lena mặt rõ → score < WARN",
          score < face_metrics.OCCLUSION_WARN, f"(score={score:.3f})")

    x1, y1, x2, y2 = f.bbox.astype(int)
    w, h = x2 - x1, y2 - y1
    # Che 1 bên (tóc/tay) — detector vẫn tìm thấy mặt, điểm occlusion tăng
    side = lena.copy()
    cv2.rectangle(side, (x1 - 10, y1 - 10), (x1 + int(w * 0.45), y2 + 10), 0, -1)
    side_faces = detector.detect(side)
    check("che 1 bên: detector VẪN thấy mặt", len(side_faces) >= 1)
    if side_faces:
        side_score = face_metrics.occlusion_score(
            sorted(side_faces, key=lambda x: -(x.bbox[2] - x.bbox[0]))[0]
        )
        check("che 1 bên → score ≥ WARN",
              side_score >= face_metrics.OCCLUSION_WARN,
              f"(score={side_score:.3f})")

    # Khẩu trang (che nửa dưới) — detector KHÔNG thấy mặt → đăng ký tự chặn
    mask = lena.copy()
    cv2.rectangle(mask, (x1 - 10, y1 + int(h * 0.55)), (x2 + 10, y2 + 10), 0, -1)
    check("khẩu trang che nửa dưới → KHÔNG thấy mặt (tự chặn)",
          len(detector.detect(mask)) == 0)


# ---------------------------------------------------------
# 3. Enrollment — từ chối mặt bị che + tip đeo kính
# ---------------------------------------------------------
def test_enrollment_occlusion() -> None:
    print("\n[3] Enrollment — từ chối mặt bị che + tip đeo kính")
    detector = FaceDetector(MODELS_ROOT)
    embedder = FaceEmbedder(detector)
    lena = load_lena()
    worker = EnrollmentWorker(0, 640, 480, detector=detector, embedder=embedder)

    # Patch occlusion_score → cao: _pick_best_face TỪ CHỐI (None)
    with mock.patch(
        "app.core.face_metrics.occlusion_score", return_value=0.80
    ):
        check("mặt bị che nhiều → _pick_best_face trả None (chặn mẫu xấu)",
              worker._pick_best_face(lena) is None)
    # Patch occlusion_score → thấp: chấp nhận
    with mock.patch(
        "app.core.face_metrics.occlusion_score", return_value=0.20
    ):
        check("mặt rõ → _pick_best_face chọn được",
              worker._pick_best_face(lena) is not None)

    # GUI: dialog có tip đeo kính
    from PySide6.QtWidgets import QApplication, QLabel

    app = QApplication.instance() or QApplication(sys.argv)
    db = Database(TEMP_DB)
    dialog = EnrollmentDialog(Config(camera_index=99), db, detector=None)
    tips = [l.text() for l in dialog.findChildren(QLabel) if "Đeo kính" in l.text()]
    check("dialog đăng ký có tip đeo kính", len(tips) == 1, f"(tips={tips})")
    dialog.close()
    db.close()


# ---------------------------------------------------------
# 4. CameraWorker — phân tầng cảnh báo che khuất
# ---------------------------------------------------------
def test_camera_tiers() -> None:
    print("\n[4] CameraWorker — occlusion cao chặn / vừa cảnh báo / rõ bình thường")
    from app.ui.camera_view import CameraWorker

    detector = FaceDetector(MODELS_ROOT)
    embedder = FaceEmbedder(detector)
    lena = load_lena()

    db = Database(TEMP_DB)
    svc = RecognitionService(db)
    enrollment = EnrollmentService(db, thumbs_dir=TEMP_THUMBS)
    emb = embedder.embed_face(detector.detect(lena)[0])
    person = enrollment.save_person(
        "Nguyễn Văn A",
        [CapturedSample(embedding=emb, quality=0.9, face_crop=lena)],
    )
    svc.reload()

    def make_worker(occ_value: float):
        hits: list = []
        spoofs: list = []
        occlusions: list = []
        w = CameraWorker(
            0, 640, 480,
            detector=detector, embedder=embedder, service=svc, threshold=0.40,
            anti_spoofing_enabled=False,  # cô lập hành vi occlusion
        )
        w.recognition_hit.connect(lambda pid, sim, crop: hits.append((pid, sim)))
        w.spoof_suspected.connect(lambda pid, sim, crop: spoofs.append((pid, sim)))
        w.occlusion_blocked.connect(
            lambda pid, sim, crop: occlusions.append((pid, sim))
        )
        return w, hits, spoofs, occlusions

    # (a) Occlusion CAO (0.80) → KHÔNG hit/spoof, NHƯNG phát occlusion_blocked
    w1, hits1, spoofs1, occ1 = make_worker(0.80)
    with mock.patch(
        "app.core.face_metrics.occlusion_score", return_value=0.80
    ):
        unknown1 = w1._process_frame(lena.copy())
    check("mặt bị che nhiều → KHÔNG phát hit", len(hits1) == 0, f"(hits={hits1})")
    check("mặt bị che nhiều → KHÔNG phát spoof", len(spoofs1) == 0)
    check("mặt bị che nhiều → KHÔNG báo người lạ", unknown1 is False)
    check("mặt bị che nhiều → PHÁT occlusion_blocked (đúng person)",
          len(occ1) == 1 and occ1[0][0] == person.id, f"(occ={occ1})")

    # (b) Occlusion VỪA (0.40 — WARN nhưng dưới BLOCK 0.50) → vẫn nhận diện (hit)
    w2, hits2, _, occ2 = make_worker(0.40)  # noqa: E501
    with mock.patch(
        "app.core.face_metrics.occlusion_score", return_value=0.40
    ):
        w2._process_frame(lena.copy())
    check("mặt hơi bị che → vẫn nhận diện (hit kèm ⚠)",
          len(hits2) == 1 and hits2[0][0] == person.id, f"(hits={hits2})")
    check("mặt hơi bị che → KHÔNG phát occlusion_blocked", len(occ2) == 0)

    # (c) Mặt RÕ (0.20) → nhận diện bình thường
    w3, hits3, _, occ3 = make_worker(0.20)  # noqa: E501
    with mock.patch(
        "app.core.face_metrics.occlusion_score", return_value=0.20
    ):
        w3._process_frame(lena.copy())
    check("mặt rõ → nhận diện bình thường (hit)",
          len(hits3) == 1 and hits3[0][0] == person.id, f"(hits={hits3})")
    check("mặt rõ → KHÔNG phát occlusion_blocked", len(occ3) == 0)

    db.close()
    cleanup()


# ---------------------------------------------------------
# 5. Ghi sự kiện 'Mặt bị che' vào Lịch sử (audit trail)
# ---------------------------------------------------------
def test_occlusion_event() -> None:
    print("\n[5] Ghi sự kiện 'Mặt bị che' vào lịch sử")
    from PySide6.QtWidgets import QApplication

    from app.services.enrollment import CapturedSample, EnrollmentService
    from app.services.recognition import RecognitionService

    app = QApplication.instance() or QApplication(sys.argv)
    detector = FaceDetector(MODELS_ROOT)
    embedder = FaceEmbedder(detector)
    lena = load_lena()

    db = Database(TEMP_DB)
    svc = RecognitionService(db)
    enrollment = EnrollmentService(db, thumbs_dir=TEMP_THUMBS)
    emb = embedder.embed_face(detector.detect(lena)[0])
    person = enrollment.save_person(
        "Nguyễn Văn A",
        [CapturedSample(embedding=emb, quality=0.9, face_crop=lena)],
    )

    # (a) save_event(is_occluded=True) → label 'Mặt bị che: <tên>'
    event_id = svc.save_event(
        person_id=person.id,
        similarity=0.85,
        face_crop=lena,
        is_occluded=True,
    )
    check("save_event(is_occluded=True) trả id", bool(event_id))
    with db.session() as conn:
        ev = conn.execute(
            "SELECT label, person_id, is_unknown FROM recognition_events"
            " WHERE id = ?",
            (event_id,),
        ).fetchone()
    check("label = 'Mặt bị che: Nguyễn Văn A'",
          ev and ev["label"] == "Mặt bị che: Nguyễn Văn A",
          f"(label={ev and ev['label']!r})")
    check("giữ person_id (biết là ai)", ev and ev["person_id"] == person.id)
    check("is_unknown = 0", ev and int(ev["is_unknown"]) == 0)

    # (b) HistoryView hiển thị status '⚠ Bị chặn' cho sự kiện bị che
    from app.services.auth import AuthService
    from app.ui.history_view import HistoryView

    config = Config()
    view = HistoryView(db, AuthService(config))
    view._refresh()
    statuses = [
        view._table.item(r, 4).text()
        for r in range(view._table.rowCount())
    ]
    check("HistoryView có cột trạng thái '⚠ Bị chặn'",
          "⚠ Bị chặn" in statuses, f"(statuses={statuses})")
    # (c) sự kiện bình thường vẫn '✓ Xác nhận' — không bị phá
    normal_id = svc.save_event(
        person_id=person.id, similarity=0.90, face_crop=lena
    )
    view._refresh()
    statuses = [view._table.item(r, 4).text()
                for r in range(view._table.rowCount())]
    check("sự kiện bình thường vẫn '✓ Xác nhận'",
          "✓ Xác nhận" in statuses, f"(statuses={statuses})")
    check("có 1 dòng '⚠ Bị chặn' (chỉ sự kiện bị che)",
          statuses.count("⚠ Bị chặn") == 1, f"(statuses={statuses})")

    view.close()
    db.close()
    cleanup()


# ---------------------------------------------------------
# 6. PhotoWorker — kết quả đánh dấu occluded
# ---------------------------------------------------------
def test_photo_occlusion() -> None:
    print("\n[6] PhotoWorker — đánh dấu mặt bị che trong ảnh tĩnh")
    detector = FaceDetector(MODELS_ROOT)
    embedder = FaceEmbedder(detector)
    lena = load_lena()

    db = Database(TEMP_DB)
    svc = RecognitionService(db)
    enrollment = EnrollmentService(db, thumbs_dir=TEMP_THUMBS)
    emb = embedder.embed_face(detector.detect(lena)[0])
    enrollment.save_person(
        "Nguyễn Văn A",
        [CapturedSample(embedding=emb, quality=0.9, face_crop=lena)],
    )
    svc.reload()

    def run_photo(occ_value: float) -> list:
        """Chạy PhotoWorker 1 lần, bắt kết quả qua signal finished."""
        captured: list = []
        w = PhotoWorker(
            lena.copy(), detector=detector, embedder=embedder,
            service=svc, threshold=0.40,
        )
        w.finished.connect(lambda frame, results: captured.append(results))
        with mock.patch(
            "app.core.face_metrics.occlusion_score", return_value=occ_value
        ):
            w.run()
        return captured[0] if captured else []

    results = run_photo(0.80)
    check("PhotoWorker có kết quả", len(results) >= 1)
    if results:
        check("mặt bị che → occluded=True", results[0].occluded is True)
        check("label có '⚠ bị che'", "bị che" in results[0].label,
              f"(label={results[0].label!r})")

    results2 = run_photo(0.20)
    if results2:
        check("mặt rõ → occluded=False", results2[0].occluded is False)

    db.close()
    cleanup()


# ---------------------------------------------------------
# 7. UI — sectionTitle + theme
# ---------------------------------------------------------
def test_ui_polish() -> None:
    print("\n[7] UI — sectionTitle + theme")
    from PySide6.QtWidgets import QApplication, QLabel

    from app.services.auth import AuthService
    from app.services.sync import SyncService
    from app.ui.theme import DARK_QSS, LIGHT_QSS

    app = QApplication.instance() or QApplication(sys.argv)
    cfg_mod.CONFIG_PATH = TEMP_CONFIG

    check("theme dark có style sectionTitle", "#sectionTitle" in DARK_QSS)
    check("theme light có style sectionTitle", "#sectionTitle" in LIGHT_QSS)
    check("theme dark có QToolTip", "QToolTip" in DARK_QSS)
    check("theme dark có QCheckBox indicator", "QCheckBox::indicator" in DARK_QSS)

    db = Database(TEMP_DB)
    config = Config()
    view = sv.SettingsView(config, AuthService(config), SyncService(db, config))
    count = len([l for l in view.findChildren(QLabel)
                 if l.objectName() == "sectionTitle"])
    check("SettingsView có ≥ 4 tiêu đề sectionTitle", count >= 4,
          f"(count={count})")
    view.close()
    db.close()
    cleanup()


def main() -> None:
    print("=== TEST BƯỚC 18: UI/UX + BẢO MẬT NHẬN DIỆN (MẶT BỊ CHE) ===")
    cleanup()
    test_occlusion_unit()
    test_occlusion_gpu()
    test_enrollment_occlusion()
    test_camera_tiers()
    test_occlusion_event()
    test_photo_occlusion()
    test_ui_polish()
    cleanup()

    print(f"\nKẾT QUẢ: {PASS} qua / {FAIL} thất bại")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
