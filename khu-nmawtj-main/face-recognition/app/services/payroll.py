"""PayrollService — kỳ công / chốt công nhận lương.

Kỳ công (pay period): khoảng thời gian chốt công lặp lại (vd kỳ 1 tháng:
26 tháng trước → 25 tháng này). Người dùng chọn 2 mốc ở trang Kỳ công →
tính công cho từng nhân viên trên toàn bộ attendance trong khoảng.

Nguyên tắc tính (đơn giản, minh bạch — không cấu hình ca phức tạp):
  - Ngày công      = số ngày CÓ điểm danh VÀO CA trong kỳ.
  - Giờ công       = tổng (check_out − check_in) các ngày có giờ ra; ngày
                     chưa ra ca (check_out = NULL) được cộng thêm GIỜ CÔNG
                     CREDIT MẶC ĐỊNH theo loại kỳ (lương tháng 8h/ngày, lương giờ
                     4h/ngày) để không làm thiệt người quét quên RA CA.
  - Đi muộn/về sớm = số lần check_in/check_out vượt mốc ca (chỉ khi khai
                     ca giờ; không khai ca → 0).
  - Nghỉ có/lương  = số ngày làm việc của kỳ trừ ngày công (trừ CN nếu
                     bật "trừ CN").

Tra cứu (lookup): lọc dòng chấm công THÔ theo Mã NV/ID · tên · tháng —
dùng cho bảng tra cứu + thống kê tiền công (Tổng tiền công = ngày công ×
lương/ngày, mức lương nhập ở UI — project chưa lưu lương trong DB).

Định dạng giờ: attendance lưu ISO UTC; hiển thị quy đổi về giờ ĐỊA PHƯƠNG
(giống fmt_time của attendance.py). Người dùng hiểu giờ trên bảng là giờ
đã làm việc thực tế.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone

from app.infrastructure.db import Database
from app.services.attendance import AttendanceRecord, AttendanceService

logger = logging.getLogger(__name__)

# Credit giờ công mặc định cho ngày CHƯA RA CA (tùy loại kỳ):
CREDIT_MONTHLY = 8.0  # lương tháng: công một ngày mặc định tính 8 giờ
CREDIT_HOURLY = 4.0   # lương giờ: quét vào được credit 4 giờ/ngày

# Nhãn loại kỳ (hiển thị ở UI)
PERIOD_TYPE_LABELS: dict[str, str] = {
    "monthly": "Kỳ theo tháng (chốt ngày 25)",
    "twice_monthly": "Kỳ 2 lần/tháng (1–15, 16–cuối)",
    "weekly": "Kỳ theo tuần",
    "custom": "Tùy chọn ngày",
}


def _parse_iso_utc(iso: str) -> datetime | None:
    """ISO (có/không 'Z') → datetime có tz; lỗi → None."""
    try:
        dt = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _local_time_of(iso: str) -> time | None:
    """Giờ:phút (giờ địa phương) của mốc ISO UTC."""
    dt = _parse_iso_utc(iso)
    return dt.astimezone().time() if dt else None


@dataclass
class PayrollRow:
    """Một dòng tổng hợp công của 1 nhân viên trong kỳ."""

    person_id: str
    name: str
    department: str = ""
    position: str = ""
    work_days: int = 0            # số ngày có điểm danh VÀO CA
    late_count: int = 0           # số lần đi muộn (khai ca giờ)
    early_leave_count: int = 0    # số lần về sớm (khai ca giờ)
    missing_checkout: int = 0     # số ngày chưa RA CA (được credit)
    total_hours: float = 0.0      # tổng giờ công (bao gồm credit)
    absent_days: int = 0          # ngày làm việc của kỳ − ngày công
    # Chi tiết từng ngày trong kỳ (bật "Xem chi tiết"):
    days: dict[str, DayDetail] = field(default_factory=dict)


@dataclass
class DayDetail:
    """Công 1 ngày của 1 nhân viên (bảng chi tiết).

    v5 nhiều phiên/ngày: ``sessions`` chứa TỪNG phiên (check_in/check_out
    riêng); ``check_in``/``check_out`` = giờ vào/ra ĐẦU TIÊN-CUỐI CÙNG của
    ngày (hiển thị). ``hours`` = TỔNG giờ công của TẤT CẢ phiên trong ngày.
    """

    work_date: str        # 'YYYY-MM-DD'
    check_in: str = ""    # ISO UTC (giờ vào ĐẦU TIÊN trong ngày)
    check_out: str = ""   # ISO UTC (giờ ra CUỐI CÙNG; rỗng = chưa ra ca)
    note_in: str = ""
    note_out: str = ""
    late: bool = False
    early_leave: bool = False
    missing_checkout: bool = False
    credit_hours: float = CREDIT_HOURLY  # giờ credit khi CHƯA RA CA
    sessions: list = field(default_factory=list)  # v5: [(in, out), ...] từng phiên

    @property
    def hours(self) -> float:
        """TỔNG giờ công của ngày = cộng DỒN từng phiên (v5 nhiều phiên).

        - ``sessions`` RỖNG (dòng thô tra cứu — 1 phiên/dòng) → tính như
          cũ: (ra − vào); chưa ra ca → credit (không vỡ hành vi cũ).
        - Có N phiên đóng → tổng (out − in) từng phiên.
        - Còn phiên CHƯA RA CA → cộng thêm NỬA credit cho phần đang làm
          (chỉ có phiên mở, chưa đóng gì → credit nguyên ngày như cũ —
          không thiệt người quét quên RA CA).
        """
        if not self.sessions:
            # Fallback dòng thô (lookup) — GIỮ NGUYÊN logic cũ
            if not self.check_in:
                return 0.0
            if not self.check_out:
                return self.credit_hours
            d_in = _parse_iso_utc(self.check_in)
            d_out = _parse_iso_utc(self.check_out)
            if d_in is None or d_out is None:
                return 0.0
            return max(0.0, (d_out - d_in).total_seconds() / 3600.0)
        total = 0.0
        has_open = False
        for s_in, s_out in self.sessions:
            if not s_out:
                has_open = True
                continue
            d_in = _parse_iso_utc(s_in)
            d_out = _parse_iso_utc(s_out)
            if d_in is None or d_out is None:
                continue
            total += max(0.0, (d_out - d_in).total_seconds() / 3600.0)
        if has_open:
            if total == 0.0:
                # Chỉ có phiên chưa ra → credit cho cả ngày (hành vi cũ)
                return self.credit_hours
            # Có công đóng + phần đang làm mở → cộng thêm nửa credit
            total += self.credit_hours / 2.0
        return total


@dataclass
class PayPeriod:
    """Định nghĩa 1 kỳ công (mốc đầu/cuối theo NGÀY, tính đầu-cuối)."""

    kind: str            # 'monthly' / 'twice_monthly' / 'weekly' / 'custom'
    start: date          # ngày đầu kỳ
    end: date            # ngày cuối kỳ
    label: str = ""      # nhãn hiển thị

    def describe(self) -> str:
        fmt = "%d/%m/%Y"
        base = f"{self.start.strftime(fmt)} → {self.end.strftime(fmt)}"
        return f"{self.label}: {base}" if self.label else base


def current_period(kind: str, ref: date | None = None) -> PayPeriod:
    """Kỳ công CHỨA ngày ``ref`` (mặc định hôm nay) theo loại kỳ đã chọn.

    - monthly       : 26 tháng trước → 25 tháng này (lương tháng, chốt 25).
    - twice_monthly : 1–15 và 16–cuối tháng.
    - weekly        : thứ Hai → Chủ nhật.
    - custom        : cả tháng hiện tại (người dùng tự chỉnh mốc sau).
    """
    ref = ref or date.today()
    if kind == "monthly":
        if ref.day >= 26:
            start = ref.replace(day=26)
        else:
            prev_last = ref.replace(day=1) - timedelta(days=1)
            start = prev_last.replace(day=26)
        end = _add_months(start, 1).replace(day=25)
        return PayPeriod(kind, start, end, PERIOD_TYPE_LABELS[kind])
    if kind == "twice_monthly":
        if ref.day <= 15:
            return PayPeriod(kind, ref.replace(day=1), ref.replace(day=15),
                             PERIOD_TYPE_LABELS[kind])
        last = _month_end(ref)
        return PayPeriod(kind, ref.replace(day=16), last, PERIOD_TYPE_LABELS[kind])
    if kind == "weekly":
        start = ref - timedelta(days=ref.weekday())  # thứ Hai
        return PayPeriod(kind, start, start + timedelta(days=6),
                         PERIOD_TYPE_LABELS[kind])
    # custom / loại lạ → cả tháng hiện tại
    return PayPeriod("custom", ref.replace(day=1), _month_end(ref), "Tùy chọn ngày")


def _add_months(d: date, months: int) -> date:
    """Cộng/trừ N tháng (kẹp ngày 31 về ngày cuối tháng nếu tràn)."""
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    day = min(d.day, _month_end(date(y, m, 1)).day)
    return date(y, m, day)


def _month_end(d: date) -> date:
    """Ngày cuối cùng của tháng chứa ``d``."""
    nxt = (d.replace(day=1) + timedelta(days=32)).replace(day=1)
    return nxt - timedelta(days=1)


def working_days(start: date, end: date, exclude_sundays: bool = True) -> int:
    """Số ngày làm việc của kỳ (mặc định trừ CN; T7 vẫn tính)."""
    days = 0
    cur = start
    while cur <= end:
        if not (exclude_sundays and cur.weekday() == 6):
            days += 1
        cur += timedelta(days=1)
    return days


# ---------------------------------------------------------------------------
# Mã nhân viên hiển thị (NV + 6 hex đầu của UUID) — không cần cột mới trong DB
# ---------------------------------------------------------------------------

def employee_code(person_id: str) -> str:
    """Mã nhân viên ngắn gọn từ id UUID: 'NV001' ↔ 6 ký tự hex đầu của id.

    Không đổi database: mã suy ra DUY NHẤT từ id nên luôn ổn định (cùng 1
    id luôn ra cùng 1 mã). Người dùng gõ 'NV001' (không phân biệt hoa/
    thường) hoặc đoạn đầu id đều lọc được.
    """
    try:
        return "NV" + str(person_id)[:6].upper()
    except (TypeError, ValueError):
        return "NV??????"


def code_matches(person_id: str, query: str) -> bool:
    """True nếu ``query`` khớp mã NV (đầy đủ hoặc phần đầu) của id.

    Chấp nhận: 'NV001', 'nv001', '001' (không tiền tố — khớp hex đầu).
    """
    q = (query or "").strip().lower()
    if not q:
        return True
    if q.startswith("nv") and len(q) > 2:
        hexq = q[2:]
    elif q.isdigit():
        hexq = q
    else:
        return False
    return str(person_id).lower().startswith(hexq)


class PayrollService:
    """Tổng hợp công theo kỳ công — UI không viết SQL trực tiếp.

    Tên/phòng ban lấy sẵn từ JOIN persons của AttendanceService._query,
    không cần truy vấn thêm bảng persons ở đây.
    """

    def __init__(self, db: Database) -> None:
        self._attendance = AttendanceService(db)

    # ---------------------------------------------------------
    # Tổng hợp kỳ
    # ---------------------------------------------------------
    def summarize(
        self,
        period: PayPeriod,
        exclude_sundays: bool = True,
        shift_start: time | None = None,
        shift_end: time | None = None,
        late_grace_minutes: int = 0,
        hourly: bool = False,
    ) -> list[PayrollRow]:
        """Tính công từng nhân viên trong kỳ.

        ``shift_start``/``shift_end``: giờ vào/ra chuẩn (khai ở UI, tùy
        chọn — None = không kiểm tra muộn/sớm).
        ``late_grace_minutes``: số phút dung sai cho đi muộn.
        ``hourly``: True = tính theo lương giờ (credit 4h/ngày chưa ra ca),
        False = lương tháng (credit 8h/ngày).
        """
        # AttendanceService.list_between lọc theo work_date (ngày ĐỊA
        # PHƯƠNG) — truyền 2 mốc ngày của kỳ là đủ.
        try:
            records = self._attendance.list_between(period.start, period.end)
        except Exception:  # noqa: BLE001 — DB lỗi → trả rỗng, không sập trang
            logger.exception("Lỗi tải điểm danh cho kỳ công")
            return []

        credit = CREDIT_HOURLY if hourly else CREDIT_MONTHLY
        total_working = working_days(period.start, period.end, exclude_sundays)
        grace = timedelta(minutes=late_grace_minutes)

        # Gom điểm danh theo nhân viên
        grouped: dict[str, list[AttendanceRecord]] = {}
        for rec in records:
            grouped.setdefault(rec.person_id, []).append(rec)

        rows: list[PayrollRow] = []
        for person_id, recs in grouped.items():
            recs.sort(key=lambda r: r.work_date)
            row = PayrollRow(
                person_id=person_id,
                name=recs[0].name,
                department=recs[0].department,
                position=recs[0].position,
            )
            # v5 NHIỀU PHIÊN/NGÀY: gom các dòng CÙNG NGÀY thành 1 DayDetail
            # — tổng giờ = cộng dồn (ra − vào) từng phiên.
            by_day: dict[str, list[AttendanceRecord]] = {}
            for rec in recs:
                by_day.setdefault(rec.work_date, []).append(rec)
            for wd_str, day_recs in by_day.items():
                day_recs.sort(key=lambda r: r.check_in)
                first = day_recs[0]
                wd = date.fromisoformat(wd_str)
                detail = DayDetail(
                    work_date=wd_str,
                    check_in=first.check_in or "",
                    # Giờ ra CUỐI CÙNG trong ngày (nếu có phiên nào đã ra)
                    check_out=next(
                        (r.check_out for r in reversed(day_recs) if r.check_out),
                        "",
                    ),
                    note_in=first.note_in,
                    note_out=next(
                        (r.note_out for r in reversed(day_recs) if r.note_out),
                        "",
                    ),
                    credit_hours=credit,
                )
                for rec in day_recs:
                    detail.sessions.append((rec.check_in or "", rec.check_out or ""))
                # Đi muộn: giờ vào ĐẦU TIÊN (địa phương) muộn hơn ca + dung sai
                if shift_start is not None and detail.check_in:
                    t_in = _local_time_of(detail.check_in)
                    if t_in is not None:
                        limit = (
                            datetime.combine(wd, shift_start) + grace
                        ).time()
                        detail.late = t_in > limit
                        if detail.late:
                            row.late_count += 1
                # Về sớm: giờ ra CUỐI CÙNG (địa phương) sớm hơn ca
                if shift_end is not None and detail.check_out:
                    t_out = _local_time_of(detail.check_out)
                    if t_out is not None:
                        detail.early_leave = t_out < shift_end
                        if detail.early_leave:
                            row.early_leave_count += 1
                # Chưa RA CA: CÓ phiên nào chưa đóng trong ngày
                if any(s_in and not s_out for s_in, s_out in detail.sessions):
                    detail.missing_checkout = True
                    row.missing_checkout += 1
                # hours = cộng dồn TẤT CẢ phiên của ngày (xem DayDetail.hours)
                row.total_hours += detail.hours
                row.work_days += 1
                row.days[wd_str] = detail
            row.absent_days = max(0, total_working - row.work_days)
            rows.append(row)

        rows.sort(key=lambda r: (r.department, r.name))
        return rows

    # ---------------------------------------------------------
    # Tra cứu chấm công theo bộ lọc (Mã NV/ID · tên · tháng)
    # ---------------------------------------------------------
    def lookup(
        self,
        query: str = "",
        month: int | None = None,
        year: int | None = None,
    ) -> list[dict]:
        """Tra cứu dòng chấm công THÔ theo bộ lọc (bảng tra cứu Kỳ công).

        - ``query``: khớp TÊN (chứa, không phân biệt hoa/thường, vd 'test
          ky cong') HOẶC MÃ NV ('NV001' / '001') HOẶC đoạn đầu id UUID.
        - ``month``/``year``: lọc theo tháng chấm công (work_date giờ ĐỊA
          PHƯƠNG — cùng cách khóa khi quét VÀO/RA CA). Phải có cả hai mới
          lọc; bỏ trống → toàn bộ dữ liệu.

        Trả danh sách dict: {person_id, code, name, department, position,
        work_date, check_in, check_out, hours} — sắp theo tên rồi ngày.
        Giờ công/ngày giữ nguyên logic có sẵn: (giờ ra − giờ vào) thực tế;
        chưa RA CA → cộng giờ credit (8h lương tháng / 4h lương giờ).
        """
        if month is not None and year is not None:
            records = self._attendance.list_between(
                date(year, month, 1), _month_end(date(year, month, 1))
            )
        else:
            records = self._attendance.list_all()

        q = (query or "").strip().lower()
        out: list[dict] = []
        for rec in records:
            if q and not (
                q in rec.name.lower()
                or code_matches(rec.person_id, q)
                or str(rec.person_id).lower().startswith(q)
            ):
                continue
            detail = DayDetail(
                work_date=rec.work_date,
                check_in=rec.check_in or "",
                check_out=rec.check_out or "",
                credit_hours=CREDIT_MONTHLY,
            )
            out.append(
                {
                    "person_id": rec.person_id,
                    "code": employee_code(rec.person_id),
                    "name": rec.name,
                    "department": rec.department,
                    "position": rec.position,
                    "work_date": rec.work_date,
                    "check_in": rec.check_in or "",
                    "check_out": rec.check_out or "",
                    "hours": detail.hours,
                }
            )
        out.sort(key=lambda d: (d["name"], d["work_date"]))
        return out

    # ---------------------------------------------------------
    # Xuất CSV chấm công (Excel mở được — UTF-8 BOM)
    # ---------------------------------------------------------
    def export_csv(self, path, rows: list[PayrollRow], period: PayPeriod) -> int:
        """Xuất bảng tổng hợp công kỳ ra CSV; trả số dòng."""
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "BÁO CÁO CHẤM CÔNG",
                    period.describe(),
                ]
            )
            writer.writerow(
                [
                    "Nhân viên",
                    "Phòng ban",
                    "Chức vụ",
                    "Ngày công",
                    "Nghỉ (ngày)",
                    "Đi muộn",
                    "Về sớm",
                    "Chưa ra ca",
                    "Tổng giờ công",
                ]
            )
            for r in rows:
                writer.writerow(
                    [
                        r.name,
                        r.department,
                        r.position,
                        r.work_days,
                        r.absent_days,
                        r.late_count,
                        r.early_leave_count,
                        r.missing_checkout,
                        f"{r.total_hours:.1f}",
                    ]
                )
        logger.info("Đã xuất chấm công %d nhân viên → %s", len(rows), path)
        return len(rows)

    def export_lookup_csv(self, path, rows: list[dict]) -> int:
        """Xuất kết quả TRA CỨU (bảng dòng thô) ra CSV; trả số dòng."""
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["Mã NV", "Tên nhân viên", "Phòng ban", "Chức vụ",
                 "Ngày", "Giờ vào", "Giờ ra", "Số giờ làm"]
            )
            for r in rows:
                writer.writerow(
                    [
                        r["code"],
                        r["name"],
                        r["department"],
                        r["position"],
                        r["work_date"],
                        fmt_time_hhmm(r["check_in"]),
                        fmt_time_hhmm(r["check_out"]),
                        f"{r['hours']:.2f}",
                    ]
                )
        logger.info("Đã xuất tra cứu chấm công %d dòng → %s", len(rows), path)
        return len(rows)


def fmt_time_hhmm(iso: str) -> str:
    """ISO UTC → 'HH:MM' giờ địa phương; rỗng → '—' (dùng cho bảng tra cứu)."""
    if not iso:
        return "—"
    dt = _parse_iso_utc(iso)
    return dt.astimezone().strftime("%H:%M") if dt else str(iso)


def summarize_lookup(rows: list[dict]) -> dict[str, dict]:
    """Gom các dòng tra cứu theo nhân viên → số liệu thống kê từng người.

    Trả dict theo person_id: {code, name, work_days, total_hours} — UI
    nhân với LƯƠNG/NGÀY (nhập ở UI) để ra TỔNG TIỀN CÔNG:
        Tổng tiền công = số ngày công × lương/ngày.
    """
    stats: dict[str, dict] = {}
    for r in rows:
        s = stats.setdefault(
            r["person_id"],
            {
                "person_id": r["person_id"],
                "code": r["code"],
                "name": r["name"],
                "work_days": 0,
                "total_hours": 0.0,
            },
        )
        s["work_days"] += 1
        s["total_hours"] += float(r["hours"])
    return stats
