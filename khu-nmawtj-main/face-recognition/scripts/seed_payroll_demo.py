"""Tạo dữ liệu DEMO cho trang Kỳ công — 1 nhân viên, công ~12 ngày gần nhất.

Chạy:  .venv\\Scripts\\python.exe scripts\\seed_payroll_demo.py
Dọn:   .venv\\Scripts\\python.exe scripts\\seed_payroll_demo.py --clean

Tạo 1 nhân viên 'DEMO Ky Cong' + attendance của các ngày LÀM VIỆC gần nhất
(trừ CN, bỏ qua HÔM NAY để bạn tự quét thật):
  - Đa số ngày: vào 08:00 → ra 17:10 (đủ công)
  - 1 số ngày:  vào 09:45 (đi muộn khi bật 'Áp dụng giờ ca 08:00')
  - 1 ngày:     chưa RA CA (check_out NULL — được cộng giờ credit)

KHÔNG tạo/xóa/sửa cấu trúc bảng; bỏ qua ngày đã có attendance (chạy lại an toàn).
"""
from __future__ import annotations

import sys
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.infrastructure.db import Database  # noqa: E402
from app.infrastructure.repositories import PersonRepository  # noqa: E402

DEMO_NAME = "DEMO Ky Cong"
DAYS_BACK = 12  # tạo công cho 12 ngày gần nhất (bỏ CN và hôm nay)


def _iso_utc_at(wd: date, hh: int, mm: int) -> str:
    """Giờ ĐỊA PHƯƠNG (wd, hh:mm) → ISO UTC có 'Z' (định dạng app ghi)."""
    local = datetime(wd.year, wd.month, wd.day, hh, mm).astimezone()
    return local.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def clean(db: Database) -> None:
    """Xóa toàn bộ dữ liệu DEMO (attendance + persons theo tên marker)."""
    with db.session() as conn:
        rows = conn.execute(
            "SELECT id FROM persons WHERE name = ?", (DEMO_NAME,)
        ).fetchall()
        ids = [str(r["id"]) for r in rows]
        for pid in ids:
            conn.execute("DELETE FROM attendance WHERE person_id = ?", (pid,))
            conn.execute("DELETE FROM persons WHERE id = ?", (pid,))
    print(f"Đã dọn {len(ids)} nhân viên DEMO (+ attendance của họ).")


def seed(db: Database) -> None:
    people = PersonRepository(db)
    # Tìm nhân viên DEMO đã có (chạy lại không tạo trùng)
    existing = next(
        (p for p in people.list_all() if p.name == DEMO_NAME), None
    )
    if existing is None:
        person = people.add(name=DEMO_NAME, thumbnail_path="", record_outbox=False)
        print(f"Đã tạo nhân viên '{DEMO_NAME}' (id={person.id})")
    else:
        person = existing
        print(f"Dùng lại nhân viên DEMO có sẵn (id={person.id})")
    pid = person.id

    today = date.today()
    added = skipped = 0
    for i in range(1, DAYS_BACK + 1):
        wd = today - timedelta(days=i)
        if wd.weekday() == 6:  # Chủ nhật — nghỉ
            continue
        # Mẫu dữ liệu: đi muộn khi (i % 4 == 1); chưa RA CA khi (i % 5 == 2)
        late = (i % 4 == 1)
        no_out = (i % 5 == 2)
        t_in = (9, 45) if late else (8, 0)
        t_out = None if no_out else (17, 10)

        with db.session() as conn:
            exists = conn.execute(
                "SELECT 1 FROM attendance WHERE person_id = ? AND work_date = ?",
                (pid, wd.isoformat()),
            ).fetchone()
            if exists:
                skipped += 1
                continue
            ts_in = db.ts("attendance", "check_in", _iso_utc_at(wd, *t_in))
            ts_out = (
                db.ts("attendance", "check_out", _iso_utc_at(wd, *t_out))
                if t_out else None
            )
            conn.execute(
                "INSERT INTO attendance"
                " (id, person_id, work_date, check_in, check_out,"
                "  checked_out, note_in, note_out, last_seen, seen_count)"
                " VALUES (?, ?, ?, ?, ?, ?, '', '', ?, 1)",
                (
                    uuid.uuid4().hex,
                    pid,
                    wd.isoformat(),
                    ts_in,
                    ts_out,
                    0 if no_out else 1,
                    ts_in,
                ),
            )
        added += 1

    print(
        f"Xong — thêm {added} ngày công (bỏ qua {skipped} ngày đã có).\n"
        f"→ Mở app → trang 'Kỳ công' → chọn kỳ chứa các ngày này để xem.\n"
        f"→ Bật 'Áp dụng giờ ca' 08:00–17:00 để thấy ngày vào 09:45 bị tính ĐI MUỘN.\n"
        f"→ Hôm nay chưa seed — hãy quét VÀO CA/RA CA thật để thấy dòng mới."
    )


def main() -> int:
    db = Database()
    try:
        db.connect()
    except Exception as exc:  # noqa: BLE001
        print(f"Không kết nối được SQL Server: {exc}")
        return 1
    try:
        if "--clean" in sys.argv:
            clean(db)
        else:
            seed(db)
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
