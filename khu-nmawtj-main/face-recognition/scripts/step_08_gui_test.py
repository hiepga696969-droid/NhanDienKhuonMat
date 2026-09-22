"""Bước 8 — Kiểm tra PersonListView + PersonService (chế độ minimal, tự thoát).

1. PersonService: thống kê (số mẫu + lần nhận diện gần nhất), đổi tên
   (chặn tên trống), xóa (xóa luôn file thumbnail; events giữ person_id=NULL).
2. GUI: danh sách hiển thị đúng, tìm kiếm lọc, Sửa/Xóa qua PasswordDialog
   (giả lập đúng mật khẩu), nút [＋ Đăng ký mới] gọi callback + làm mới.

Chạy:  .venv\\Scripts\\python.exe scripts\\step_08_gui_test.py
"""
from __future__ import annotations

import os
import sys
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

import numpy as np  # noqa: E402

from app.config import Config  # noqa: E402
from app.infrastructure.db import DATA_DIR, Database  # noqa: E402
from app.infrastructure.repositories import (  # noqa: E402
    EMBEDDING_DIM,
    PersonRepository,
)
from app.services.auth import AuthService  # noqa: E402
from app.services.enrollment import CapturedSample, EnrollmentService  # noqa: E402
from app.services.person import PersonService  # noqa: E402

TEMP_DB = PROJECT_ROOT / "data" / "test_persons.db"
TEMP_THUMBS = PROJECT_ROOT / "data" / "thumbs_test8"

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


def cleanup_db() -> None:
    for suffix in ("", "-wal", "-shm"):
        Path(str(TEMP_DB) + suffix).unlink(missing_ok=True)
    for f in TEMP_THUMBS.glob("*.jpg"):
        f.unlink(missing_ok=True)
    try:
        TEMP_THUMBS.rmdir()
    except OSError:
        pass


def seed_person(db: Database, name: str, seed: int):
    """Tạo người thật qua EnrollmentService (có thumbnail trên đĩa)."""
    enrollment = EnrollmentService(db, thumbs_dir=TEMP_THUMBS)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(5):
        emb = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        crop = np.full((96, 96, 3), 120, dtype=np.uint8)
        samples.append(CapturedSample(embedding=emb, quality=0.9, face_crop=crop))
    return enrollment.save_person(name, samples)


# ---------------------------------------------------------
# 1. PersonService trên DB tạm
# ---------------------------------------------------------
def test_service() -> None:
    print("\n[1] PersonService — thống kê / đổi tên / xóa")
    db = Database(TEMP_DB)
    svc = PersonService(db)
    a = seed_person(db, "Nguyễn Văn A", seed=1)
    b = seed_person(db, "Trần Thị B", seed=2)

    # Ghi 1 sự kiện nhận diện cho A (giả lập dữ liệu Bước 9)
    with db.session() as conn:
        conn.execute(
            "INSERT INTO recognition_events (id, person_id, label, source)"
            " VALUES (?, ?, ?, ?)",
            ("evt-1", a.id, a.name, "webcam"),
        )

    stats = svc.list_with_stats()
    check("danh sách có 2 người", len(stats) == 2, f"(={len(stats)})")
    by_name = {s.person.name: s for s in stats}
    check("A có đủ 5 mẫu", by_name["Nguyễn Văn A"].sample_count == 5)
    check("A có lần nhận diện gần nhất",
          by_name["Nguyễn Văn A"].last_detected_at is not None)
    check("B chưa từng được nhận diện",
          by_name["Trần Thị B"].last_detected_at is None)

    check("đổi tên hợp lệ", svc.rename(a.id, "Nguyễn Văn A2"))
    check("đổi tên trống bị chặn", svc.rename(a.id, "   ") is False)
    check("tên đã cập nhật trong DB",
          PersonRepository(db).get(a.id).name == "Nguyễn Văn A2")

    thumb = DATA_DIR / a.thumbnail_path
    check("file thumbnail A tồn tại trên đĩa", thumb.exists())
    check("xóa A thành công", svc.delete(a.id))
    check("file thumbnail A đã bị xóa", not thumb.exists())
    check("số người còn 1", svc.count() == 1)

    with db.session() as conn:
        row = conn.execute(
            "SELECT person_id FROM recognition_events WHERE id = 'evt-1'"
        ).fetchone()
    check("event giữ lại với person_id = NULL",
          row is not None and row["person_id"] is None)
    check("xóa người không tồn tại → False", svc.delete("khong-co") is False)

    db.close()
    cleanup_db()


# ---------------------------------------------------------
# 2. GUI: PersonListView
# ---------------------------------------------------------
def test_gui() -> None:
    print("\n[2] GUI — PersonListView (tìm kiếm / Sửa / Xóa / Đăng ký)")
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    from argon2 import PasswordHasher

    import app.ui.person_list_view as plv
    from app.ui.person_list_view import PersonListView

    app = QApplication.instance() or QApplication(sys.argv)

    config = Config(camera_index=99, camera_width=320, camera_height=240)
    auth = AuthService(config)
    # Đặt hash mật khẩu trực tiếp — KHÔNG ghi config.json thật
    auth.config.password_hash = PasswordHasher().hash("mat-khau-123")

    db = Database(TEMP_DB)
    p1 = seed_person(db, "Nguyễn Văn A", seed=3)
    p2 = seed_person(db, "Trần Thị B", seed=4)

    enroll_calls: list[int] = []
    view = PersonListView(
        db, auth,
        enroll_callback=lambda: enroll_calls.append(1),
    )
    view.show()

    check("danh sách hiển thị 2 người", view._list.count() == 2,
          f"(={view._list.count()})")
    check("footer hiện tổng số", "2 người" in view._shown_label.text(),
          f"(text={view._shown_label.text()!r})")

    # Tìm kiếm lọc theo tên (không phân biệt hoa thường)
    view._search.setText("TRẦN")
    check("lọc theo tên còn 1 người", view._list.count() == 1)
    view._search.setText("không-có-tên-này")
    check("tìm không khớp → rỗng + thông báo", view._list.count() == 0
          and view._empty_label.isVisible())
    view._search.setText("")
    check("xóa từ khóa → hiện lại 2", view._list.count() == 2)

    # Sửa tên: giả lập PasswordDialog OK + QInputDialog trả tên mới
    orig_require = plv.PasswordDialog.require
    orig_gettext = plv.QInputDialog.getText
    orig_msgbox_question = plv.QMessageBox.question
    orig_msgbox_info = plv.QMessageBox.information
    plv.PasswordDialog.require = staticmethod(lambda *a, **k: True)
    plv.QInputDialog.getText = staticmethod(lambda *a, **k: ("Nguyễn Văn Đã Sửa", True))
    # Giả lập QMessageBox.question = Yes (xác nhận xóa)
    from PySide6.QtWidgets import QMessageBox
    plv.QMessageBox.question = staticmethod(
        lambda *a, **k: QMessageBox.StandardButton.Yes
    )
    plv.QMessageBox.information = staticmethod(lambda *a, **k: None)

    first_id = view._list.item(0).data(Qt.ItemDataRole.UserRole)
    view._on_edit(first_id, "Nguyễn Văn A")
    check("đổi tên qua GUI cập nhật DB",
          PersonRepository(db).get(first_id).name == "Nguyễn Văn Đã Sửa")

    # Xóa: giả lập xác nhận Yes + PasswordDialog OK
    view._on_delete(first_id, "Nguyễn Văn Đã Sửa")
    check("xóa qua GUI còn 1 người", view._list.count() == 1)

    # Nút Đăng ký mới: gọi callback (MainWindow mở EnrollmentDialog) + làm mới
    view._on_enroll()
    check("callback đăng ký được gọi", len(enroll_calls) == 1)
    check("danh sách vẫn còn 1 người", view._list.count() == 1)

    # Xóa nốt người còn lại → trạng thái rỗng
    second_id = view._list.item(0).data(Qt.ItemDataRole.UserRole)
    view._on_delete(second_id, "Trần Thị B")
    check("xóa hết → danh sách rỗng", view._list.count() == 0)
    check("hiện thông báo trống", view._empty_label.isVisible())

    # Xác nhận HỦY: bấm No → không xóa
    p3 = seed_person(db, "NgườiWillCancel", seed=5)
    view._refresh()
    cancel_count = view._list.count()
    plv.QMessageBox.question = staticmethod(
        lambda *a, **k: QMessageBox.StandardButton.No
    )
    view._on_delete(p3.id, "NgườiWillCancel")
    check("hủy xác nhận → KHÔNG xóa", view._list.count() == cancel_count)
    plv.QMessageBox.question = staticmethod(
        lambda *a, **k: QMessageBox.StandardButton.Yes
    )
    # Xóa người vừa thêm để khôi phục trạng thái
    view._on_delete(p3.id, "NgườiWillCancel")

    # person_changed signal phát ra khi xóa
    signal_fired = []
    view.person_changed.connect(lambda: signal_fired.append(1))
    p_sig = seed_person(db, "Signal Test", seed=6)
    view._refresh()
    view._on_delete(p_sig.id, "Signal Test")
    check("person_changed signal phát ra khi xóa", len(signal_fired) == 1)

    # Khôi phục hàm giả
    plv.PasswordDialog.require = orig_require
    plv.QInputDialog.getText = orig_gettext
    plv.QMessageBox.question = orig_msgbox_question
    plv.QMessageBox.information = orig_msgbox_info

    view.close()
    db.close()
    cleanup_db()


def main() -> None:
    print("=== TEST BƯỚC 8: DANH SÁCH NGƯỜI DÙNG (PERSONLISTVIEW) ===")
    test_service()
    test_gui()

    print(f"\nKẾT QUẢ: {PASS} qua / {FAIL} thất bại")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
