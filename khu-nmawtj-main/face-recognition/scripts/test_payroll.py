"""Kiểm tra PayrollService — kỳ công / chốt công trên SQL Server thật.

Chạy:  .venv\\Scripts\\python.exe scripts\\test_payroll.py

Tạo 1 nhân viên TEST + 3 dòng attendance giả lập trong tuần hiện tại
(đủ công / đi muộn / chưa ra ca) → chạy summarize + xuất CSV → đối chiếu
kết quả → TỰ XÓA đúng các dòng đã tạo (không đụng dữ liệu thật, không
tạo/xóa/sửa cấu trúc bảng nào).
"""
from __future__ import annotations

import os
import sys
import tempfile
import uuid
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.infrastructure.db import Database  # noqa: E402
from app.infrastructure.repositories import PersonRepository  # noqa: E402
from app.services.payroll import (  # noqa: E402
    CREDIT_MONTHLY,
    PayPeriod,
    PayrollService,
    code_matches,
    current_period,
    employee_code,
    summarize_lookup,
)

PASS = "✅"
FAIL = "❌"


def _iso_utc_at(wd: date, hh: int, mm: int) -> str:
    """Mốc giờ ĐỊA PHƯƠNG (wd, hh:mm) → ISO UTC '...Z' (định dạng app ghi)."""
    local = datetime(wd.year, wd.month, wd.day, hh, mm).astimezone()
    utc = local.astimezone(timezone.utc)
    return utc.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def main() -> int:
    print("=" * 60)
    print("TEST PAYROLL — kỳ công trên SQL Server")
    print("=" * 60)
    failures: list[str] = []

    def check(name: str, cond: bool, extra: str = "") -> None:
        print(f"{PASS} {name}" + (f" → {extra}" if extra else ""))
        if not cond:
            failures.append(name)
            print(f"{FAIL} {name} — THẤT BẠI")

    db = Database()
    try:
        db.connect()
        print(f"{PASS} Kết nối SQL Server OK")
    except Exception as exc:  # noqa: BLE001
        print(f"{FAIL} Không kết nối được SQL Server: {exc}")
        return 1

    people = PersonRepository(db)
    payroll = PayrollService(db)
    person_id = ""
    attendance_ids: list[str] = []
    try:
        # ── 0) current_period: tính mốc kỳ hợp lý ─────────────────
        weekly = current_period("weekly")
        check("current_period(weekly) bắt đầu thứ Hai", weekly.start.weekday() == 0,
              weekly.describe())
        monthly = current_period("monthly")
        check("current_period(monthly) 26 → 25",
              monthly.start.day == 26 and monthly.end.day == 25,
              monthly.describe())

        # ── 1) Tạo nhân viên TEST + attendance giả lập ────────────
        person = people.add(
            name=f"Kỳ công TEST {uuid.uuid4().hex[:6]}",
            thumbnail_path="",
            record_outbox=False,
        )
        person_id = person.id
        today = date.today()
        monday = today - timedelta(days=today.weekday())
        period = PayPeriod("weekly", monday, monday + timedelta(days=6), "Kỳ test")

        # (offset ngày từ thứ Hai, giờ vào, giờ ra) — giờ ĐỊA PHƯƠNG
        plan = [
            (0, (8, 0), (17, 0)),   # đủ công 9h
            (1, (9, 30), (17, 0)),  # đi muộn 1 lần, công 7.5h
            (2, (8, 0), None),      # chưa RA CA → credit 8h
        ]
        for off, t_in, t_out in plan:
            wd = monday + timedelta(days=off)
            rec_id = uuid.uuid4().hex
            ts_in = db.ts("attendance", "check_in", _iso_utc_at(wd, *t_in))
            ts_out = (
                db.ts("attendance", "check_out", _iso_utc_at(wd, *t_out))
                if t_out else None
            )
            with db.session() as conn:
                conn.execute(
                    "INSERT INTO attendance"
                    " (id, person_id, work_date, check_in, check_out,"
                    "  checked_out, note_in, note_out, last_seen, seen_count)"
                    " VALUES (?, ?, ?, ?, ?, 1, '', '', ?, 1)",
                    (rec_id, person_id, wd.isoformat(), ts_in, ts_out, ts_in),
                )
            attendance_ids.append(rec_id)
        check("Tạo 3 dòng attendance test", len(attendance_ids) == 3)

        # ── 2) summarize: tổng hợp công ───────────────────────────
        rows = payroll.summarize(
            period,
            shift_start=time(8, 0),
            shift_end=time(17, 0),
            late_grace_minutes=0,
        )
        mine = next((r for r in rows if r.person_id == person_id), None)
        check("summarize trả đúng nhân viên test", mine is not None)
        if mine is None:
            raise RuntimeError("Không tìm thấy nhân viên test trong kết quả")
        check("work_days = 3", mine.work_days == 3, f"got {mine.work_days}")
        check("late_count = 1 (vào 09:30)", mine.late_count == 1,
              f"got {mine.late_count}")
        check("missing_checkout = 1", mine.missing_checkout == 1,
              f"got {mine.missing_checkout}")
        expected_hours = 9.0 + 7.5 + CREDIT_MONTHLY  # 24.5
        check(f"total_hours = {expected_hours}", abs(mine.total_hours - expected_hours) < 0.01,
              f"got {mine.total_hours:.1f}")
        # Tuần làm việc (trừ CN) = 6 ngày → nghỉ 3
        check("absent_days = 3 (kỳ 6 ngày làm việc)", mine.absent_days == 3,
              f"got {mine.absent_days}")
        check("Chi tiết 3 ngày đầy đủ", len(mine.days) == 3)

        # ── 3) lookup: tra cứu theo Mã NV / tên / tháng ───────────
        code = employee_code(person_id)
        check("employee_code ổn định 'NV'+6 hex",
              code.startswith("NV") and len(code) == 8, code)
        check("code_matches('NV...') đúng", code_matches(person_id, code))
        check("code_matches lowercase", code_matches(person_id, code.lower()))
        check("code_matches sai → False", not code_matches(person_id, "NVzzzzzz"))

        by_code = payroll.lookup(query=code)
        check("lookup theo MÃ NV", any(r["person_id"] == person_id for r in by_code),
              f"{len(by_code)} dòng")
        by_name = payroll.lookup(query=mine.name.split()[0])  # 1 phần tên
        check("lookup theo 1 PHẦN tên", any(r["person_id"] == person_id for r in by_name),
              f"query='{mine.name.split()[0]}'")
        # Tuần hiện tại có thể vắt qua 2 tháng → tính kỳ vọng theo ngày
        # (các ngày wd thuộc tháng hiện tại mới xuất hiện khi lọc tháng)
        exp_days_in_month = 0
        exp_hours_in_month = 0.0
        for off, t_in, t_out in plan:
            wd = monday + timedelta(days=off)
            if wd.month != date.today().month:
                continue
            exp_days_in_month += 1
            if t_out is None:
                exp_hours_in_month += CREDIT_MONTHLY
            else:
                exp_hours_in_month += (
                    (t_out[0] * 60 + t_out[1]) - (t_in[0] * 60 + t_in[1])
                ) / 60.0
        this_month = payroll.lookup(
            query=code, month=date.today().month, year=date.today().year,
        )
        check("lookup theo THÁNG hiện tại", len(this_month) == exp_days_in_month,
              f"{len(this_month)}/{exp_days_in_month} dòng")
        other = payroll.lookup(
            query=code, month=(date.today().month % 12) + 1, year=date.today().year
        )
        check("lookup tháng KHÁC → rỗng", all(r["person_id"] != person_id for r in other))
        combined = payroll.lookup(query=code, month=date.today().month,
                                  year=date.today().year)
        check("lookup KẾT HỢP (mã + tháng)", len(combined) == exp_days_in_month)
        hours_sum = sum(r["hours"] for r in combined)
        check("lookup giờ công cộng đúng", abs(hours_sum - exp_hours_in_month) < 0.01,
              f"got {hours_sum:.2f} / exp {exp_hours_in_month:.2f}")

        stats = summarize_lookup(combined)
        s = stats.get(person_id)
        check("summarize_lookup: ngày công đúng",
              (s["work_days"] if s else 0) == exp_days_in_month,
              f"got {s['work_days'] if s else 0}/{exp_days_in_month}")
        salary = 300_000
        money = (s["work_days"] if s else 0) * salary
        check("tiền công = ngày công × lương/ngày",
              money == exp_days_in_month * salary, f"{money:,} đ")

        # ── 4) Xuất CSV ─────────────────────────────────────────
        tmp = os.path.join(tempfile.gettempdir(), f"payroll_test_{uuid.uuid4().hex[:6]}.csv")
        n = payroll.export_csv(tmp, [mine], period)
        check("export_csv ghi 1 dòng", n == 1, tmp)
        check("CSV tồn tại", os.path.exists(tmp) and os.path.getsize(tmp) > 0)
        os.remove(tmp)

    except Exception as exc:  # noqa: BLE001
        failures.append("LỖI NGOẠI LỆ")
        print(f"{FAIL} Lỗi bất ngờ: {exc!r}")
        import traceback
        traceback.print_exc()
    finally:
        # ── Dọn dữ liệu test (CHỈ đúng các dòng script đã tạo) ────
        try:
            with db.session() as conn:
                for rid in attendance_ids:
                    conn.execute("DELETE FROM attendance WHERE id = ?", (rid,))
            if person_id:
                people.delete(person_id, record_outbox=False)
            print(f"{PASS} Dọn dữ liệu test xong ({len(attendance_ids)} dòng attendance)")
        except Exception:  # noqa: BLE001
            print("⚠ Không tự dọn được — xóa tay attendance ids:", attendance_ids)

    db.close()
    print("=" * 60)
    if failures:
        print(f"KẾT QUẢ: {len(failures)} mục THẤT BẠI → {failures}")
        return 1
    print("KẾT QUẢ: TẤT CẢ ĐẠT — Payroll hoạt động đúng trên SQL Server")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
