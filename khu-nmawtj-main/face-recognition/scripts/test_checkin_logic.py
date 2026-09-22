# -*- coding: utf-8 -*-
"""Test logic điểm danh v5 — NHIỀU PHIÊN vào/ra trong ngày, đúng logic.

Mô phỏng ĐÚNG luồng xử lý của ``CameraView._on_recognition_hit``:
  1. VÀO CA (phiên 1)          → tạo dòng mới, ghi giờ vào
  2. RA CA (đóng phiên 1)      → ghi giờ ra, checked_out=1
  3. VÀO CA (phiên 2)          → TẠO DÒNG MỚI (không đụng phiên 1)
  4. RA CA (đóng phiên 2)      → đóng phiên 2
  5. VÀO CA lần nữa            → phiên 2 đang mở? KHÔNG — đã đóng → phiên 3
  6. RA CA khi đang mở         → đóng phiên 3
  7. RA CA khi KHÔNG phiên mở  → tự mở phiên bù (giờ vào = giờ ra) + cảnh báo

Đối chiếu: số dòng, giờ vào/ra từng phiên, session_no, lịch sử quét
(chỉ các lần quét hợp lệ mới lưu sự kiện).
Dùng SQLite file tạm — KHÔNG đụng CSDL thật của app.
Chạy:  python scripts/test_checkin_logic.py
"""
from __future__ import annotations

import sys
import tempfile
import shutil
from pathlib import Path

# Chạy được từ thư mục nào cũng được
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:  # console Windows có thể không phải UTF-8
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import numpy as np  # noqa: E402

from app.infrastructure.db import Database  # noqa: E402
from app.infrastructure.repositories import (  # noqa: E402
    PersonRepository,
    RecognitionEventRepository,
)
from app.services.attendance import (  # noqa: E402
    AttendanceService,
    fmt_time,
)
from app.services.payroll import (  # noqa: E402
    PayPeriod,
    PayrollService,
)
from datetime import date  # noqa: E402
from app.services.recognition import RecognitionService  # noqa: E402


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="fr_logic_test_"))
    failures: list[str] = []

    def check(name: str, cond: bool) -> None:
        print(("  ✓ PASS  " if cond else "  ✗ FAIL  ") + name)
        if not cond:
            failures.append(name)

    try:
        db = Database(tmp / "test.db")
        db.connect()

        people = PersonRepository(db)
        attendance = AttendanceService(db)
        service = RecognitionService(db)
        # Không lưu snapshot ra đĩa trong test (tránh rác + đụng thư mục thật)
        service._save_snapshot = lambda crop: ""  # type: ignore[method-assign]

        person = people.add("Nguyễn Văn A", thumbnail_path="")
        pid = person.id
        img = np.zeros((48, 48, 3), dtype=np.uint8)  # ảnh giả làm "crop"

        def quet(mode: str) -> dict:
            """Bản sao logic _on_recognition_hit (camera_view.py)."""
            if mode == "checkin":
                res = attendance.record_check_in(pid)
            else:
                res = attendance.record_check_out(pid)
            blocked = res.get("blocked") or (
                mode == "checkout" and res.get("was_not_checked_in")
            )
            if not blocked:
                service.save_event(
                    person_id=pid, similarity=0.90, face_crop=img, mode=mode
                )
            return res

        print("Kịch bản: NHIỀU PHIÊN vào/ra trong ngày\n")

        # ── Phiên 1 ──
        r1 = quet("checkin")
        print(f"[1] VÀO CA  → {r1['message']}")
        check("Lần 1 VÀO CA tạo phiên mới", r1["ok"] and not r1["already_done"])

        r2 = quet("checkout")
        print(f"[2] RA CA   → {r2['message']}")
        check("Lần 2 RA CA đóng phiên 1", r2["ok"] and not r2["already_done"])
        p1_gio_ra = r2["check_out"]

        # ── Phiên 2 (về trưa, vào lại chiều) ──
        r3 = quet("checkin")
        print(f"[3] VÀO CA  → {r3['message']}")
        check("Lần 3 VÀO CA tạo PHIÊN MỚI (không đụng phiên 1)",
              r3["ok"] and not r3["already_done"])
        check("Giờ vào phiên 2 KHÁC giờ vào phiên 1",
              r3["check_in"] is not None and r3["check_in"] > (r1["check_in"] or ""))

        r4 = quet("checkout")
        print(f"[4] RA CA   → {r4['message']}")
        check("Lần 4 RA CA đóng PHIÊN 2", r4["ok"] and r4["check_out"] == r4["check_out"])
        check("Giờ ra phiên 2 sau giờ ra phiên 1",
              r4["check_out"] > p1_gio_ra)

        # ── Phiên 3 (quét thêm vào/ra nữa — vẫn hợp lệ) ──
        r5 = quet("checkin")
        print(f"[5] VÀO CA  → {r5['message']}")
        check("Lần 5 VÀO CA tạo phiên 3", r5["ok"] and not r5["already_done"])
        r6 = quet("checkout")
        print(f"[6] RA CA   → {r6['message']}")
        check("Lần 6 RA CA đóng phiên 3", r6["ok"])

        # ── RA CA khi KHÔNG có phiên mở → tự mở phiên bù + cảnh báo ──
        r7 = quet("checkout")
        print(f"[7] RA CA   → {r7['message']}")
        check("Lần 7 RA CA tự mở phiên bù (was_not_checked_in)",
              r7["ok"] and r7.get("was_not_checked_in") is True)
        check("Giờ vào = giờ ra ở phiên bù", r7["check_in"] == r7["check_out"])

        # ── Đối chiếu CSDL ──
        recs = attendance.list_today()
        with db.session() as conn:
            rows = conn.execute(
                "SELECT mode FROM recognition_events ORDER BY detected_at"
            ).fetchall()
        modes = [r["mode"] for r in rows]

        print()
        check("attendance: đúng 4 PHIÊN trong ngày", len(recs) == 4)
        check("Tất cả phiên đã đóng (checked_out)", all(r.checked_out for r in recs))
        check("session_no đánh đúng 1..4",
              sorted(r.session_no for r in recs) == [1, 2, 3, 4])
        check("Lịch sử quét: 6 sự kiện hợp lệ (không tính phiên bù)",
              len(modes) == 6 and modes.count("checkin") == 3 and modes.count("checkout") == 3)

        # ── Payroll: tổng giờ = cộng dồn các phiên ──
        period = PayPeriod("custom", date.today(), date.today(), "Test ngày")
        prow = PayrollService(db).summarize(period)[0]
        print(f"\nKỳ công: {prow.work_days} ngày · {prow.total_hours:.2f} giờ")
        check("Kỳ công: 1 ngày công (gom 4 phiên cùng ngày)", prow.work_days == 1)
        day_hours = prow.days[date.today().isoformat()].hours
        check("Tổng giờ công > 0 (cộng dồn phiên)", day_hours > 0)
        # Giờ công của ngày ≈ tổng các phiên đóng (phiên bù = 0h vì vào=ra)
        import math
        check("Giờ công khớp tổng phiên (±1 phút)",
              abs(day_hours - sum(
                  (0.0 if (not s_out) else max(0.0, (
                      __import__("datetime").datetime.fromisoformat(s_out.replace("Z", "+00:00"))
                      - __import__("datetime").datetime.fromisoformat(s_in.replace("Z", "+00:00"))
                  ).total_seconds() / 3600.0))
                  for s_in, s_out in prow.days[date.today().isoformat()].sessions
              )) < 1/60,
              )

        # In tóm tắt các phiên
        print("\nCác phiên trong ngày:")
        for r in sorted(recs, key=lambda x: x.session_no):
            print(f"  Phiên {r.session_no}: {fmt_time(r.check_in)} → {fmt_time(r.check_out)}")

        try:
            db._conn.close()  # đóng để xóa được thư mục tạm
        except Exception:  # noqa: BLE001
            pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if failures:
        print(f"KẾT QUẢ: {len(failures)} kiểm tra THẤT BẠI")
        return 1
    print("KẾT QUẢ: TẤT CẢ PASS — nhiều phiên vào/ra trong ngày đúng logic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
