"""Bước 11 — Kiểm tra HistoryView + HistoryService (chế độ minimal, tự thoát).

1. HistoryService: liệt kê mới nhất trước, lọc theo tên/nguồn/ngày, đếm
   cho phân trang, xóa sự kiện kèm file snapshot trên đĩa.
2. GUI: bảng hiển thị đúng, bộ lọc hoạt động, panel chi tiết (snapshot
   người lạ bị làm mờ), Xóa qua PasswordDialog, phân trang 50/trang.

Chạy:  .venv\\Scripts\\python.exe scripts\\step_11_gui_test.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import PySide6  # noqa: E402

# Khắc phục đường dẫn plugins + chế độ ảo (giống step_03/08)
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
from app.services.history import HistoryService  # noqa: E402
from app.ui.history_view import HistoryView  # noqa: E402

TEMP_DB = PROJECT_ROOT / "data" / "test_history.db"
TEMP_THUMBS = PROJECT_ROOT / "data" / "thumbs_test11"
TEMP_SNAPS = PROJECT_ROOT / "data" / "snapshots_test11"

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
    for d in (TEMP_THUMBS, TEMP_SNAPS):
        for f in d.glob("*"):
            f.unlink(missing_ok=True)
        try:
            d.rmdir()
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


def make_snapshot(name: str) -> str:
    """Tạo 1 file ảnh JPEG nhỏ trong thư mục snapshot tạm; trả đường dẫn tương đối."""
    import cv2

    TEMP_SNAPS.mkdir(parents=True, exist_ok=True)
    img = np.full((80, 80, 3), 150, dtype=np.uint8)
    rel = f"snapshots_test11/{name}.jpg"
    cv2.imwrite(str(DATA_DIR / rel), img)
    return rel


def add_event(
    db: Database,
    event_id: str,
    label: str,
    source: str,
    detected_at: str,
    person_id: str | None = None,
    similarity: float | None = None,
    snapshot_path: str = "",
    is_unknown: bool = False,
) -> None:
    """Ghi trực tiếp 1 sự kiện vào DB tạm (kiểm soát detected_at)."""
    with db.session() as conn:
        conn.execute(
            "INSERT INTO recognition_events"
            " (id, person_id, label, source, detected_at, similarity, snapshot_path, is_unknown)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (event_id, person_id, label, source, detected_at,
             similarity, snapshot_path, 1 if is_unknown else 0),
        )


def now_utc() -> str:
    """detected_at của thời điểm hiện tại (UTC, đúng định dạng của schema)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


# ---------------------------------------------------------
# 1. HistoryService trên DB tạm
# ---------------------------------------------------------
def test_service() -> None:
    print("\n[1] HistoryService — liệt kê / lọc / đếm / xóa")
    db = Database(TEMP_DB)
    svc = HistoryService(db)
    a = seed_person(db, "Nguyễn Văn A", seed=1)
    b = seed_person(db, "Trần Thị B", seed=2)

    snap_known = make_snapshot("ev_known")
    snap_unknown = make_snapshot("ev_unknown")
    snap_old = make_snapshot("ev_old")

    add_event(db, "ev-1", a.name, "webcam", now_utc(),
              person_id=a.id, similarity=0.91, snapshot_path=snap_known)
    add_event(db, "ev-2", "Người lạ", "photo", now_utc(),
              similarity=None, snapshot_path=snap_unknown, is_unknown=True)
    add_event(db, "ev-3", b.name, "webcam", "2020-01-01T00:00:00.000Z",
              person_id=b.id, similarity=0.80, snapshot_path=snap_old)

    # Liệt kê mặc định: mới nhất trước, đủ 3 sự kiện
    events = svc.list_events()
    check("liệt kê đủ 3 sự kiện", len(events) == 3, f"(={len(events)})")
    check("sắp mới nhất trước", events[0].id == "ev-1"
          or events[0].id == "ev-2", f"(đầu={events[0].id})")
    check("sự kiện cũ 2020 xếp cuối", events[-1].id == "ev-3")

    # Lọc theo nguồn
    check("lọc nguồn photo còn 1", len(svc.list_events(source="photo")) == 1)
    check("lọc nguồn không tồn tại → rỗng",
          len(svc.list_events(source="mobile")) == 0)

    # Lọc theo tên (không phân biệt hoa thường)
    check("lọc theo tên 'trần' còn 1", len(svc.list_events(query="trần")) == 1)
    check("lọc theo tên không khớp → rỗng",
          len(svc.list_events(query="không-có")) == 0)

    # Lọc theo ngày (mốc 00:00 hôm nay → chỉ còn sự kiện hôm nay)
    check("ngày 'Hôm nay' còn 2 (loại bỏ 2020)",
          svc.count_events(date_range="Hôm nay") == 2)
    check("ngày 'Tất cả' đếm đủ 3", svc.count_events(date_range="Tất cả") == 3)

    # Phân trang
    check("limit 2 → trả 2", len(svc.list_events(limit=2)) == 2)
    check("offset 2 → trả 1 (trang 2)", len(svc.list_events(limit=2, offset=2)) == 1)

    # Xóa: dòng + file snapshot trên đĩa biến mất
    snap_path = (DATA_DIR / snap_known)
    check("file snapshot ev-1 tồn tại", snap_path.exists())
    check("xóa ev-1 thành công", svc.delete_event("ev-1"))
    check("dòng ev-1 đã xóa khỏi DB", svc.get_event("ev-1") is None)
    check("file snapshot ev-1 đã xóa trên đĩa", not snap_path.exists())
    check("xóa sự kiện không tồn tại → False", svc.delete_event("khong-co") is False)

    # Sau khi xóa ev-1 còn 2
    check("còn 2 sự kiện sau khi xóa", svc.count_events() == 2)

    db.close()
    cleanup_db()


# ---------------------------------------------------------
# 2. GUI: HistoryView
# ---------------------------------------------------------
def test_gui() -> None:
    print("\n[2] GUI — HistoryView (bảng / lọc / chi tiết / xóa / phân trang)")
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    from argon2 import PasswordHasher

    import app.ui.history_view as hv
    from app.ui.history_view import HistoryView

    app = QApplication.instance() or QApplication(sys.argv)

    config = Config(camera_index=99, camera_width=320, camera_height=240)
    auth = AuthService(config)
    auth.config.password_hash = PasswordHasher().hash("mat-khau-123")

    db = Database(TEMP_DB)
    svc = HistoryService(db)
    a = seed_person(db, "Nguyễn Văn A", seed=3)

    add_event(db, "g1", a.name, "webcam", now_utc(),
              person_id=a.id, similarity=0.91, snapshot_path=make_snapshot("g_known"))
    add_event(db, "g2", "Người lạ", "photo", now_utc(),
              snapshot_path=make_snapshot("g_unknown"), is_unknown=True)
    add_event(db, "g3", a.name, "webcam", "2020-01-01T00:00:00.000Z",
              person_id=a.id, similarity=0.80, snapshot_path=make_snapshot("g_old"))

    view = HistoryView(db, auth)
    view.show()

    check("bảng hiển thị 3 sự kiện", view._table.rowCount() == 3,
          f"(={view._table.rowCount()})")
    check("footer tổng số 3", "3 sự kiện" in view._count_label.text(),
          f"(text={view._count_label.text()!r})")
    check("cột trạng thái có ✓ Xác nhận",
          any(view._table.item(r, 4).text() == "✓ Xác nhận"
              for r in range(view._table.rowCount())))
    check("cột trạng thái có ⚠ Chưa ĐK",
          any(view._table.item(r, 4).text() == "⚠ Chưa ĐK"
              for r in range(view._table.rowCount())))

    # Tự chọn dòng đầu → nút Xóa bật
    view._table.selectRow(0)
    check("nút Xóa bật khi có dòng chọn", view._delete_btn.isEnabled())

    # Chọn dòng NGƯỜI ĐÃ BIẾT → hiện điểm tương đồng (0.91 hoặc 0.80)
    for r in range(view._table.rowCount()):
        if view._table.item(r, 4).text() == "✓ Xác nhận":
            view._table.selectRow(r)
            break
    check("chi tiết hiển thị điểm tương đồng",
          "Điểm tương đồng: 0." in view._detail_sim.text(),
          f"(text={view._detail_sim.text()!r})")

    # Chọn dòng NGƯỜI LẠ → snapshot bị làm mờ (graphicsEffect != None)
    for r in range(view._table.rowCount()):
        if view._table.item(r, 4).text() == "⚠ Chưa ĐK":
            view._table.selectRow(r)
            break
    check("snapshot người lạ bị LÀM MỜ",
          view._snapshot_label.graphicsEffect() is not None)
    check("chi tiết người lạ: điểm —",
          "Điểm tương đồng: —" in view._detail_sim.text())
    # Chọn dòng người đã biết → hết mờ
    for r in range(view._table.rowCount()):
        if view._table.item(r, 4).text() == "✓ Xác nhận":
            view._table.selectRow(r)
            break
    check("snapshot người đã biết KHÔNG mờ",
          view._snapshot_label.graphicsEffect() is None)
    check("snapshot có pixmap (ảnh thật)",
          not view._snapshot_label.pixmap().isNull())

    # Bộ lọc nguồn: chọn 'photo' → còn 1 dòng (người lạ)
    idx = view._source_combo.findData("photo")
    view._source_combo.setCurrentIndex(idx)
    check("lọc nguồn photo còn 1 dòng", view._table.rowCount() == 1,
          f"(={view._table.rowCount()})")
    view._source_combo.setCurrentIndex(0)  # về Tất cả

    # Tìm kiếm theo tên
    view._search.setText("TRẦN")
    check("tìm 'TRẦN' → rỗng + thông báo",
          view._table.rowCount() == 0 and view._empty_label.isVisible())
    view._search.setText("Nguyễn")
    check("tìm 'Nguyễn' còn 2", view._table.rowCount() == 2)
    view._search.setText("")
    check("xóa từ khóa → lại 3", view._table.rowCount() == 3)

    # Xóa qua PasswordDialog (giả lập đúng mật khẩu)
    orig_require = hv.PasswordDialog.require
    hv.PasswordDialog.require = staticmethod(lambda *a, **k: True)
    view._table.selectRow(0)
    before = view._table.rowCount()
    view._on_delete()
    check("xóa sự kiện → bảng còn ít hơn 1",
          view._table.rowCount() == before - 1,
          f"({before} → {view._table.rowCount()})")
    hv.PasswordDialog.require = orig_require

    # Phân trang: thêm đủ sự kiện để có 2 trang (50/trang)
    for i in range(55):
        add_event(db, f"bulk-{i:03d}", a.name, "webcam", now_utc(),
                  person_id=a.id, similarity=0.5 + (i % 40) / 100.0,
                  snapshot_path=make_snapshot(f"bulk_{i}"))
    view._refresh()
    check("trang 1 hiển thị đủ 50 dòng", view._table.rowCount() == 50,
          f"(={view._table.rowCount()})")
    check("tổng 2 trang", view._total_pages == 2, f"(={view._total_pages})")
    check("nhãn trang 1/2", view._page_label.text() == "Trang 1/2",
          f"(text={view._page_label.text()!r})")
    check("nút Trước tắt ở trang 1", not view._prev_btn.isEnabled())
    check("nút Sau bật ở trang 1", view._next_btn.isEnabled())

    view._go_next()
    check("trang 2 còn 7 dòng (57 - 50)", view._table.rowCount() == 7,
          f"(={view._table.rowCount()})")
    check("nhãn trang 2/2", view._page_label.text() == "Trang 2/2",
          f"(text={view._page_label.text()!r})")
    check("nút Sau tắt ở trang cuối", not view._next_btn.isEnabled())
    view._go_prev()
    check("quay về trang 1", view._page_label.text() == "Trang 1/2")

    # Lọc 'Hôm nay' → loại bỏ CÁC sự kiện 2020 (g3 còn lại + g-old vừa thêm)
    add_event(db, "g-old", a.name, "webcam", "2020-01-01T00:00:00.000Z",
              person_id=a.id, similarity=0.5, snapshot_path="")
    total_all = svc.count_events(date_range="Tất cả")
    total_today = svc.count_events(date_range="Hôm nay")
    check("lọc Hôm nay loại bỏ 2 sự kiện 2020",
          total_all == total_today + 2,
          f"(Tất cả={total_all}, Hôm nay={total_today})")
    view._date_combo.setCurrentIndex(0)

    view.close()
    db.close()
    cleanup_db()


# ---------------------------------------------------------
# 3. Lọc theo TRẠNG THÁI (Bước 18 mở rộng) — DB tạm RIÊNG để không ảnh
#    hưởng các test đếm sự kiện ở trên
# ---------------------------------------------------------
def test_status_filter() -> None:
    print("\n[3] Lọc theo trạng thái — '⚠ Bị chặn' (Giả mạo / Mặt bị che)")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    config = Config(camera_index=99, camera_width=320, camera_height=240)
    auth = AuthService(config)
    auth.config.password_hash = "x"  # không cần mật khẩu thật cho test này

    db = Database(TEMP_DB)
    svc = HistoryService(db)
    a = seed_person(db, "Nguyễn Văn A", seed=4)

    # 2 bình thường + 1 người lạ + 1 Giả mạo + 1 Mặt bị che
    add_event(db, "s-1", a.name, "webcam", now_utc(),
              person_id=a.id, similarity=0.91, snapshot_path="")
    add_event(db, "s-2", a.name, "photo", now_utc(),
              person_id=a.id, similarity=0.80, snapshot_path="")
    add_event(db, "s-3", "Người lạ", "webcam", now_utc(),
              similarity=None, snapshot_path="", is_unknown=True)
    add_event(db, "s-4", "Giả mạo: Nguyễn Văn A", "webcam", now_utc(),
              person_id=a.id, similarity=0.9, snapshot_path="")
    add_event(db, "s-5", "Mặt bị che: Nguyễn Văn A", "webcam", now_utc(),
              person_id=a.id, similarity=0.85, snapshot_path="")

    # Service
    check("status=blocked đếm đủ 2 (Giả mạo + Mặt bị che)",
          svc.count_events(status="blocked") == 2)
    check("status=blocked chỉ trả sự kiện bị chặn",
          {e.label for e in svc.list_events(status="blocked")}
          == {"Giả mạo: Nguyễn Văn A", "Mặt bị che: Nguyễn Văn A"})
    check("status=unknown chỉ còn người lạ (1)",
          svc.count_events(status="unknown") == 1)
    check("status=confirmed loại bỏ bị chặn + người lạ",
          svc.count_events(status="confirmed") == 2)
    check("status=confirmed không chứa 'Giả mạo'/'Mặt bị che'",
          all(not e.label.startswith(("Giả mạo:", "Mặt bị che:"))
              for e in svc.list_events(status="confirmed")))
    check("status rỗng = tất cả (5 sự kiện)", svc.count_events() == 5)
    check("kết hợp status + tên (blocked + 'Nguyễn') = 2",
          svc.count_events(query="Nguyễn", status="blocked") == 2)

    # GUI
    view = HistoryView(db, auth)
    view.show()
    idx = view._status_combo.findData("blocked")
    view._status_combo.setCurrentIndex(idx)
    check("GUI lọc '⚠ Bị chặn' còn 2 dòng", view._table.rowCount() == 2,
          f"(={view._table.rowCount()})")
    labels = [view._table.item(r, 1).text() for r in range(view._table.rowCount())]
    check("GUI chỉ hiện Giả mạo / Mặt bị che",
          all(lbl.startswith(("Giả mạo:", "Mặt bị che:")) for lbl in labels),
          f"(labels={labels})")
    check("GUI cột trạng thái đều '⚠ Bị chặn'",
          all(view._table.item(r, 4).text() == "⚠ Bị chặn"
              for r in range(view._table.rowCount())))
    check("GUI footer tổng 2 sự kiện", "2 sự kiện" in view._count_label.text(),
          f"(text={view._count_label.text()!r})")
    view._status_combo.setCurrentIndex(0)  # về Tất cả
    check("GUI về Tất cả → đủ 5 dòng", view._table.rowCount() == 5,
          f"(={view._table.rowCount()})")

    view.close()
    db.close()
    cleanup_db()


def main() -> None:
    print("=== TEST BƯỚC 11: LỊCH SỬ NHẬN DIỆN (HISTORYVIEW) ===")
    cleanup_db()  # DB tạm sạch từ đầu (chạy lại an toàn)
    test_service()
    test_gui()
    test_status_filter()

    print(f"\nKẾT QUẢ: {PASS} qua / {FAIL} thất bại")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
