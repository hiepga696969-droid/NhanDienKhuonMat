"""Smoke test — kiểm tra toàn bộ app hoạt động với SQL Server FaceRecognitionDB.

Chạy:  .venv\\Scripts\\python.exe scripts\\test_sqlserver_migration.py

Kiểm tra TẤT CẢ các thao tác SQL của app trên database SQL Server thật
(SQLite đã bị thay thế): CRUD persons / face_samples / recognition_events,
điểm danh attendance, outbox đồng bộ, bảng settings, phân trang
(OFFSET/FETCH), bộ lọc trạng thái, datetime đọc/ghi.

Script CHỈ tạo dữ liệu TEST có id riêng rồi TỰ XÓA đúng số dòng đã tạo —
không đụng dữ liệu thật, không tạo/xóa/sửa cấu trúc bảng nào.
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from app.infrastructure.db import Database  # noqa: E402
from app.infrastructure.repositories import (  # noqa: E402
    EMBEDDING_DIM,
    FaceSampleRepository,
    PersonRepository,
    RecognitionEventRepository,
    SyncOutboxRepository,
)
from app.infrastructure.sqlserver_db import utc_now_iso  # noqa: E402
from app.services.attendance import AttendanceService  # noqa: E402
from app.services.history import HistoryService  # noqa: E402
from app.services.sync import _get_setting, _set_setting  # noqa: E402

PASS = "✅"
FAIL = "❌"


def _random_embedding() -> np.ndarray:
    vec = np.random.default_rng(seed=123).standard_normal(EMBEDDING_DIM).astype(np.float32)
    return vec / np.linalg.norm(vec)


def main() -> int:
    print("=" * 60)
    print("SMOKE TEST — Face Recognition trên SQL Server")
    print("=" * 60)

    failures: list[str] = []

    def check(name: str, cond: bool, extra: str = "") -> None:
        print(f"{PASS} {name}" + (f" → {extra}" if extra else ""))
        if not cond:
            failures.append(name)
            print(f"{FAIL} {name} — THẤT BẠI")

    db = Database()
    people = PersonRepository(db)
    samples = FaceSampleRepository(db)
    events = RecognitionEventRepository(db)
    outbox = SyncOutboxRepository(db)
    attendance = AttendanceService(db)
    history = HistoryService(db)

    # ── 0) Kết nối + schema ──────────────────────────────────────
    try:
        db.connect()
        print(f"{PASS} Kết nối SQL Server OK (FaceRecognitionDB)")
    except Exception as exc:  # noqa: BLE001
        print(f"{FAIL} Kết nối SQL Server thất bại: {exc}")
        return 1

    test_marker = "SMOKE" + uuid.uuid4().hex[:8]
    sample_ids: list[str] = []
    event_ids: list[str] = []
    outbox_ids: list[str] = []
    person_id = ""
    try:
        # ── 1) persons: thêm / đọc / đổi tên / đếm ───────────────
        person = people.add(
            name=f"Người test {test_marker}", thumbnail_path="", record_outbox=False
        )
        person_id = person.id
        created_at = person.created_at
        check("persons.add + tự sinh created_at (UTC ISO)", bool(created_at) and created_at.endswith("Z"),
              f"created_at={created_at}")
        fetched = people.get(person_id)
        check("persons.get", fetched is not None and fetched.name.startswith("Người test"))
        people.update_info(person_id, "Tên đã đổi " + test_marker, department="IT", position="Tester")
        fetched = people.get(person_id)
        check("persons.update_info", fetched is not None and fetched.department == "IT")
        check("persons.count", people.count() >= 1)

        # ── 2) face_samples: embedding BLOB ──────────────────────
        emb = _random_embedding()
        sample = samples.add(person_id=person_id, embedding=emb, quality=0.9, record_outbox=False)
        sample_ids.append(sample.id)
        got = samples.list_by_person(person_id)
        check("face_samples.add + đọc BLOB", len(got) == 1 and np.allclose(got[0].embedding, emb, atol=1e-6),
              f"shape={got[0].embedding.shape}")
        check("face_samples.captured_at ISO", got[0].captured_at.endswith("Z"),
              f"captured_at={got[0].captured_at}")

        # ── 3) recognition_events: lọc + phân trang ──────────────
        ev1 = events.add(person_id=person_id, label="Tên đã đổi " + test_marker, source="webcam",
                         similarity=0.85, snapshot_path="", is_unknown=False,
                         mode="checkin", record_outbox=False)
        event_ids.append(ev1)
        ev2 = events.add(person_id=None, label="Người lạ", source="webcam",
                         similarity=0.2, snapshot_path="", is_unknown=True,
                         mode="", record_outbox=False)
        event_ids.append(ev2)
        ev3 = events.add(person_id=person_id, label="Giả mạo: " + test_marker, source="photo",
                         similarity=0.5, snapshot_path="", is_unknown=False,
                         mode="", record_outbox=False)
        event_ids.append(ev3)

        page = events.list_events(query=test_marker, limit=10, offset=0)
        check("events.list_events (OFFSET/FETCH + lọc theo tên)", len(page) >= 2,
              f"query={test_marker} → {len(page)} dòng")
        blocked = events.list_events(status="blocked", limit=10, offset=0)
        check("events.list_events (lọc 'blocked' — LIKE tham số)", any(e.id == ev3 for e in blocked))
        confirmed = events.list_events(status="confirmed", limit=10, offset=0)
        check("events.list_events (lọc 'confirmed')", any(e.id == ev1 for e in confirmed)
              and not any(e.id == ev2 for e in confirmed))
        checkin_page = events.list_events(mode="checkin", limit=10, offset=0)
        check("events.list_events (lọc mode='checkin')", any(e.id == ev1 for e in checkin_page))
        check("events.count_events", events.count_events(query=test_marker) >= 2)
        check("events.last_detected_at (TOP 1)",
              events.last_detected_at(person_id) is not None)
        one = history.get_event(ev1)
        check("HistoryService.get_event", one is not None and one.detected_at.endswith("Z"),
              f"detected_at={one.detected_at if one else '?'}")

        # ── 4) attendance: vào/ra ca + thống kê ──────────────────
        r_in = attendance.record_check_in(person_id, note="test vào ca")
        check("attendance.record_check_in", r_in["ok"] and not r_in["already_done"],
              r_in["message"])
        r_in2 = attendance.record_check_in(person_id)
        check("attendance.check_in không ghi đè", r_in2["ok"] and r_in2["already_done"])
        r_out = attendance.record_check_out(person_id, note="test ra ca")
        check("attendance.record_check_out", r_out["ok"] and not r_out["already_done"],
              r_out["message"])
        today = attendance.list_today()
        check("attendance.list_today (JOIN persons)", any(r.person_id == person_id for r in today))
        stats = attendance.stats_today()
        check("attendance.stats_today", stats["checked_in"] >= 1,
              f"checked_in={stats['checked_in']}")

        # ── 5) sync_outbox + settings ────────────────────────────
        outbox.add("person", person_id, "upsert")
        pending = outbox.list_pending()
        row = next((p for p in pending if p[1] == "person" and p[2] == person_id), None)
        check("outbox.add + list_pending", row is not None)
        outbox_ids.append(row[0])
        outbox.mark_synced(outbox_ids, utc_now_iso())
        check("outbox.mark_synced", outbox.count_pending() >= 0 and
              all(p[0] != row[0] for p in outbox.list_pending()))

        _set_setting(db, f"test_{test_marker}", "1")
        check("settings UPDATE→INSERT (_set_setting)", _get_setting(db, f"test_{test_marker}") == "1")
        _set_setting(db, f"test_{test_marker}", "2")
        check("settings UPDATE giá trị hiện có", _get_setting(db, f"test_{test_marker}") == "2")

        # ── 6) Dọn dữ liệu test (chỉ đúng các dòng script đã tạo) ─
        for eid in event_ids:
            events.delete(eid, record_outbox=False)
        with db.session() as conn:
            cur_s = conn.execute(
                "DELETE FROM face_samples WHERE person_id = ?", (person_id,)
            )
            cur = conn.execute(
                "DELETE FROM attendance WHERE person_id = ?", (person_id,)
            )
            cur2 = conn.execute(
                "DELETE FROM settings WHERE key = ?", (f"test_{test_marker}",)
            )
            conn.execute(
                "DELETE FROM sync_outbox WHERE entity_id = ? AND entity = 'person'",
                (person_id,),
            )
        deleted_person = people.delete(person_id, record_outbox=False)
        check("Dọn dữ liệu test (face_samples/attendance/settings/outbox/person)",
              deleted_person and cur.rowcount == 1 and cur2.rowcount == 1
              and cur_s.rowcount == 1)

    except Exception as exc:  # noqa: BLE001
        failures.append("LỖI NGOẠI LỆ")
        print(f"{FAIL} Lỗi bất ngờ: {exc!r}")
        import traceback

        traceback.print_exc()
        # Cố gắng dọn sạch nếu lỗi giữa chừng
        try:
            for eid in event_ids:
                events.delete(eid, record_outbox=False)
            if person_id:
                with db.session() as conn:
                    conn.execute("DELETE FROM face_samples WHERE person_id = ?", (person_id,))
                    conn.execute("DELETE FROM attendance WHERE person_id = ?", (person_id,))
                    conn.execute(
                        "DELETE FROM settings WHERE key = ?", (f"test_{test_marker}",)
                    )
                    conn.execute(
                        "DELETE FROM sync_outbox WHERE entity_id = ? AND entity = 'person'",
                        (person_id,),
                    )
                people.delete(person_id, record_outbox=False)
        except Exception:  # noqa: BLE001
            print("⚠ Không tự dọn được dữ liệu test — xóa tay theo id:", person_id)

    db.close()
    print("=" * 60)
    if failures:
        print(f"KẾT QUẢ: {len(failures)} mục THẤT BẠI → {failures}")
        return 1
    print("KẾT QUẢ: TẤT CẢ ĐÃ ĐẠT — app hoạt động trọn vẹn với SQL Server")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
