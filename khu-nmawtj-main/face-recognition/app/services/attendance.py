"""AttendanceService — điểm danh nhân viên TỰ ĐỘNG từ luồng nhận diện (webcam).

Nguyên tắc (v5 — HỖ TRỢ NHIỀU PHIÊN VÀO/RA TRONG NGÀY):
  - Quét VÀO CA khi KHÔNG có phiên đang mở → tạo 1 dòng attendance MỚI
    (check_out NULL) — quét RA CA kế tiếp sẽ đóng chính phiên này.
  - Quét RA CA → đóng phiên CHƯA RA GẦN NHẤT (check_out + checked_out=1).
  - Về trưa (RA CA) rồi quay lại chiều (VÀO CA) → 2 dòng riêng trong cùng
    ngày — đều được tính công: tổng giờ = tổng (ra − vào) từng phiên.
  - 0..N dòng / người / ngày (KHÔNG còn UNIQUE person_id + work_date —
    schema v5). ``session_no`` đánh số phiên theo thứ tự giờ vào.

Múi giờ: ``check_in``/``check_out``/``last_seen`` lưu ISO UTC (giống
``detected_at`` của recognition_events), còn ``work_date`` là ngày HÔM NAY
giờ ĐỊA PHƯƠNG ('YYYY-MM-DD') — nhân viên điểm danh 23h–01h vẫn tính
đúng ngày làm việc của họ, không lệch nửa ngày.

Trạng thái "ĐANG LÀM" = có phiên CHƯA RA CA được thấy trong cửa sổ
``ACTIVE_WINDOW_SECONDS`` giây trước hiện tại; quá cửa sổ coi như "ĐÃ RA
VỀ" với giờ ra = lần thấy sau cùng. Trạng thái tính lúc TRUY VẤN nên
dashboard tự cập nhật khi mở lại, không cần timer.

Ghi chú phạm vi: bảng attendance chỉ ghi local (không nằm trong outbox
đồng bộ cloud D1 — lược đồ D1 giữ nguyên như spec 5.4).
"""
from __future__ import annotations

import csv
import logging
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from app.infrastructure.db import Database
from app.infrastructure.repositories import PersonRepository, to_iso_str

logger = logging.getLogger(__name__)


@dataclass
class AttendanceRecord:
    """Một PHIÊN điểm danh của 1 nhân viên trong 1 ngày (v5: 0..N phiên/ngày)."""

    person_id: str
    name: str
    work_date: str          # 'YYYY-MM-DD' giờ địa phương
    check_in: str           # ISO UTC
    check_out: str | None   # ISO UTC (None = phiên chưa RA CA)
    last_seen: str          # ISO UTC
    seen_count: int
    department: str = ""    # phòng/ban (rỗng = chưa khai)
    position: str = ""      # chức vụ (rỗng = chưa khai)
    checked_out: bool = False  # True = phiên đã xác nhận RA CA
    note_in: str = ""       # ghi chú khi VÀO CA (rỗng = không có)
    note_out: str = ""      # ghi chú khi RA CA (rỗng = không có)
    session_no: int = 1     # số thứ tự phiên trong ngày (1, 2, ...)


def _utc_now_iso() -> str:
    """Thời điểm hiện tại dạng ISO UTC (khớp định dạng detected_at)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")


def today_local() -> str:
    """Ngày hôm nay giờ ĐỊA PHƯƠNG dạng 'YYYY-MM-DD' (khóa work_date)."""
    return datetime.now().astimezone().strftime("%Y-%m-%d")


# Cửa sổ "đang ở chỗ camera": seen trong N giây trước = đang làm việc.
# Webcam quét liên tục (debounce 5s/lần) nên người đang đứng trước camera
# luôn nằm trong cửa sổ này; đi khỏi quá 5 phút → coi như đã ra về.
ACTIVE_WINDOW_SECONDS = 300


def _parse_iso_utc(iso: str) -> datetime | None:
    """ISO (có/không 'Z') → datetime có tz; lỗi định dạng → None."""
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def is_still_working(rec: "AttendanceRecord", now: datetime | None = None) -> bool:
    """True nếu ``last_seen`` của PHIÊN CHƯA RA nằm trong cửa sổ ACTIVE_WINDOW."""
    if rec.checked_out:
        return False  # phiên đã đóng — không tính "đang làm"
    last = _parse_iso_utc(rec.last_seen)
    if last is None:
        return False
    now = now or datetime.now(timezone.utc)
    return (now - last).total_seconds() <= ACTIVE_WINDOW_SECONDS


def fmt_time(iso: str | None) -> str:
    """ISO UTC → 'HH:MM' giờ địa phương; None/rỗng → '—' (chưa ra về)."""
    if not iso:
        return "—"
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return iso
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone().strftime("%H:%M")


def fmt_datetime(iso: str | None) -> str:
    """ISO UTC → 'DD/MM HH:MM' giờ địa phương (bảng 7 ngày)."""
    if not iso:
        return "—"
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return iso
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone().strftime("%d/%m %H:%M")


class AttendanceService:
    """Ghi + đọc bảng điểm danh (attendance) — view KHÔNG viết SQL trực tiếp."""

    def __init__(self, db: Database) -> None:
        self._db = db
        self._people = PersonRepository(db)

    # ---------------------------------------------------------
    # Ghi (gọi từ luồng UI khi nhận diện người đã biết)
    # ---------------------------------------------------------
    def record_check_in(self, person_id: str, note: str = "") -> dict:
        """ĐIỂM DANH VÀO CA (hỗ trợ NHIỀU PHIÊN trong ngày):

        - CÓ phiên CHƯA RA CA → phiên đang mở giữ nguyên: KHÔNG tạo dòng
          mới, KHÔNG đè giờ vào gốc; có ``note`` mới thì cập nhật ghi chú
          giờ vào. ``already_done=True`` (UI không lưu sự kiện trùng).
        - KHÔNG có phiên mở (chưa vào / đã RA CA phiên trước) → tạo DÒNG
          MỚI với check_in = bây giờ, check_out NULL → quét RA CA kế tiếp
          đóng phiên NÀY (về trưa vào lại chiều = 2 dòng, đều tính công).
        - ``blocked=True`` khi người không tồn tại (bị xóa giữa 2 luồng).

        Trả về dict kết quả (xem ``_record``) để UI hiện thông báo chính xác.
        """
        return self._record(person_id, check_in=True, note=note)

    def record_check_out(self, person_id: str, note: str = "") -> dict:
        """ĐIỂM DANH RA CA (hỗ trợ NHIỀU PHIÊN trong ngày):

        - CÓ phiên CHƯA RA CA → đóng phiên GẦN NHẤT: check_out = bây giờ,
          checked_out = 1 (cố định); có ``note`` mới thì cập nhật note_out.
        - KHÔNG có phiên mở (chưa VÀO CA / phiên trước đã đóng) → TỰ MỞ
          PHIÊN BẮT BUỘC (giờ vào = giờ ra = bây giờ,
          ``was_not_checked_in=True``) để KHÔNG MẤT công của người quét RA
          CA trước (v5 — trước đây trả ``blocked=True``, mất dòng công);
          UI hiển thị cảnh báo nhưng vẫn ghi.

        Trả về dict kết quả — ``already_done=True`` khi tự mở phiên bù.
        """
        return self._record(person_id, check_in=False, note=note)

    def record_seen(self, person_id: str) -> bool:
        """Ghi 1 lần nhận diện TỰ ĐỘNG (cơ chế cũ — không phân biệt vào/ra):
        đóng phiên chưa ra gần nhất; không có phiên mở thì cập nhật phiên
        mới nhất trong ngày (chưa có dòng nào → tạo mới).

        Giữ cho tương thích (trả bool như cũ); UI mới dùng
        record_check_in/record_check_out (trả dict chi tiết).
        """
        return bool(self._record(person_id, check_in=None, note="").get("ok"))

    def _record(self, person_id: str, check_in: bool | None, note: str = "") -> dict:
        """Ghi điểm danh vào bảng attendance (mỗi PHIÊN một dòng — v5).

        ``check_in``:
          - True  → VÀO CA: phiên đang mở giữ nguyên; không có phiên mở →
                    tạo phiên MỚI (check_out NULL).
          - False → RA CA: đóng phiên chưa ra GẦN NHẤT; không có phiên mở
                    → tự mở phiên bù (vào = ra = bây giờ) + cảnh báo.
          - None  → cơ chế cũ (record_seen): đóng phiên chưa ra gần nhất.
        ``note``: ghi chú của tính năng đang chạy — lưu vào note_in khi
        VÀO CA, note_out khi RA CA (2 tính năng CÓ GHI CHÚ RIÊNG).

        Trả về dict kết quả chi tiết cho UI:
          {
            "ok": bool,               # ghi DB thành công không
            "already_done": bool,     # VÀO CA: phiên đang mở / RA CA: tự mở phiên bù
            "was_not_checked_in": bool,  # RA CA nhưng không có phiên mở (tự mở bù)
            "blocked": bool,          # True = quét KHÔNG hợp lệ → UI bỏ qua
            "check_in": str | None,   # giờ vào hiện tại (ISO)
            "check_out": str | None,  # giờ ra hiện tại (ISO)
            "message": str,           # thông báo tiếng Việt cho người dùng
          }
        """
        result: dict = {
            "ok": False,
            "already_done": False,
            "was_not_checked_in": False,
            "blocked": False,     # True = quét KHÔNG hợp lệ → bỏ qua, không ghi gì
            "check_in": None,
            "check_out": None,
            "message": "",
        }
        now_iso = _utc_now_iso()
        work_date = today_local()
        note = (note or "").strip()  # rỗng = không đổi ghi chú
        # Pre-check người tồn tại (người từng bị xóa giữa 2 luồng) — tránh
        # traceback FOREIGN KEY mỗi frame quét nhầm id cũ.
        if self._people.get(person_id) is None:
            logger.debug("Bỏ qua điểm danh: person_id không tồn tại (%s)", person_id)
            result["blocked"] = True
            result["message"] = "Nhân viên không còn trong hệ thống (đã bị xóa?)"
            return result
        # Chuẩn hóa timestamp theo KIỂU CỘT thực tế (db.ts — bỏ hậu tố 'Z'
        # khi cột attendance là DATETIME trên SQL Server, tránh lỗi cast;
        # cột NVARCHAR giữ nguyên ISO 'Z' như SQLite trước đây).
        # TÍNH SAU pre-check: lần gọi DB đầu tiên của app phải mở kết nối
        # trước (đọc INFORMATION_SCHEMA) thì db.ts mới biết kiểu cột.
        ts_now = self._db.ts("attendance", "last_seen", now_iso)
        try:
            with self._db.session() as conn:
                # PHIÊN ĐANG MỞ = dòng CHƯA RA CA gần nhất trong ngày
                # (v5: mỗi phiên 1 dòng — sắp theo giờ vào giảm dần).
                open_row = conn.execute(
                    "SELECT id, check_in, check_out, checked_out FROM attendance"
                    " WHERE person_id = ? AND work_date = ? AND checked_out = 0"
                    " ORDER BY check_in DESC",
                    (person_id, work_date),
                ).fetchone()
                if check_in is True:
                    if open_row is not None:
                        # Phiên đang mở → giữ nguyên giờ vào gốc. Có note
                        # mới → cập nhật ghi chú giờ vào của phiên đó.
                        self._touch(conn, open_row["id"], ts_now, note_in=note)
                        result.update(ok=True, already_done=True)
                        result["check_in"] = to_iso_str(open_row["check_in"])
                        result["check_out"] = (
                            to_iso_str(open_row["check_out"])
                            if open_row["check_out"] else None
                        )
                        result["message"] = (
                            "Phiên đang mở từ "
                            + fmt_time(to_iso_str(open_row["check_in"]))
                            + " — không ghi lại giờ vào"
                        )
                        logger.debug(
                            "Vào ca (phiên đang mở): %s (%s)", person_id, work_date
                        )
                    else:
                        # Không có phiên mở (chưa vào / đã RA CA phiên trước)
                        # → PHIÊN MỚI: quét RA CA kế tiếp sẽ đóng phiên này.
                        session_no = self._next_session_no(conn, person_id, work_date)
                        conn.execute(
                            "INSERT INTO attendance"
                            " (id, person_id, work_date, session_no, check_in,"
                            "  check_out, checked_out, note_in, note_out,"
                            "  last_seen, seen_count)"
                            " VALUES (?, ?, ?, ?, ?, NULL, 0, ?, NULL, ?, 1)",
                            (
                                uuid.uuid4().hex,
                                person_id,
                                work_date,
                                session_no,
                                ts_now,
                                note or None,
                                ts_now,
                            ),
                        )
                        result.update(ok=True, check_in=now_iso)
                        result["message"] = "VÀO CA thành công — " + fmt_time(now_iso)
                        logger.info(
                            "Điểm danh %s: %s (%s)",
                            result["message"], person_id, work_date,
                        )
                elif check_in is False:
                    if open_row is not None:
                        # Đóng PHIÊN CHƯA RA GẦN NHẤT — giờ ra CỐ ĐỊNH.
                        if note:
                            conn.execute(
                                "UPDATE attendance SET check_out = ?,"
                                " checked_out = 1, last_seen = ?,"
                                " seen_count = seen_count + 1, note_out = ?"
                                " WHERE id = ?",
                                (ts_now, ts_now, note, open_row["id"]),
                            )
                        else:
                            conn.execute(
                                "UPDATE attendance SET check_out = ?,"
                                " checked_out = 1, last_seen = ?,"
                                " seen_count = seen_count + 1"
                                " WHERE id = ?",
                                (ts_now, ts_now, open_row["id"]),
                            )
                        result.update(
                            ok=True,
                            check_in=to_iso_str(open_row["check_in"]),
                            check_out=now_iso,
                        )
                        result["message"] = "RA CA thành công — " + fmt_time(now_iso)
                        logger.info(
                            "Điểm danh %s: %s (%s)",
                            result["message"], person_id, work_date,
                        )
                    else:
                        # KHÔNG có phiên mở → TỰ MỞ PHIÊN BẮT BUỘC (giờ vào =
                        # giờ ra = bây giờ) — không mất công của người quét
                        # RA CA trước. UI hiển thị cảnh báo (was_not_checked_in).
                        session_no = self._next_session_no(conn, person_id, work_date)
                        conn.execute(
                            "INSERT INTO attendance"
                            " (id, person_id, work_date, session_no, check_in,"
                            "  check_out, checked_out, note_in, note_out,"
                            "  last_seen, seen_count)"
                            " VALUES (?, ?, ?, ?, ?, ?, 1, NULL, ?, ?, 1)",
                            (
                                uuid.uuid4().hex,
                                person_id,
                                work_date,
                                session_no,
                                ts_now,
                                ts_now,
                                note or None,
                                ts_now,
                            ),
                        )
                        result.update(
                            ok=True,
                            already_done=True,
                            was_not_checked_in=True,
                            check_in=now_iso,
                            check_out=now_iso,
                        )
                        result["message"] = (
                            "RA CA thành công — LƯU Ý: không có phiên đang mở, "
                            "giờ vào = giờ ra = " + fmt_time(now_iso)
                        )
                        logger.info(
                            "Điểm danh %s: %s (%s)",
                            result["message"], person_id, work_date,
                        )
                else:
                    # Cơ chế cũ (record_seen): đóng phiên chưa ra gần nhất;
                    # không có phiên mở → cập nhật phiên mới nhất trong ngày.
                    if open_row is not None:
                        conn.execute(
                            "UPDATE attendance SET check_out = ?, checked_out = 1,"
                            " last_seen = ?, seen_count = seen_count + 1"
                            " WHERE id = ?",
                            (ts_now, ts_now, open_row["id"]),
                        )
                        result.update(
                            ok=True,
                            check_in=to_iso_str(open_row["check_in"]),
                            check_out=now_iso,
                        )
                    else:
                        latest = conn.execute(
                            "SELECT id, check_in, check_out FROM attendance"
                            " WHERE person_id = ? AND work_date = ?"
                            " ORDER BY check_in DESC",
                            (person_id, work_date),
                        ).fetchone()
                        if latest is not None:
                            self._touch(conn, latest["id"], ts_now)
                            result.update(
                                ok=True,
                                check_in=to_iso_str(latest["check_in"]),
                                check_out=(
                                    to_iso_str(latest["check_out"])
                                    if latest["check_out"] else None
                                ),
                            )
                        else:
                            session_no = self._next_session_no(
                                conn, person_id, work_date
                            )
                            conn.execute(
                                "INSERT INTO attendance"
                                " (id, person_id, work_date, session_no, check_in,"
                                "  check_out, checked_out, note_in, note_out,"
                                "  last_seen, seen_count)"
                                " VALUES (?, ?, ?, ?, ?, NULL, 0, NULL, NULL, ?, 1)",
                                (
                                    uuid.uuid4().hex,
                                    person_id,
                                    work_date,
                                    session_no,
                                    ts_now,
                                    ts_now,
                                ),
                            )
                            result.update(ok=True, check_in=now_iso)
                    result["message"] = "Đã cập nhật"
                    logger.debug("Tự động: %s (%s)", person_id, work_date)
            return result
        except Exception:  # noqa: BLE001 — lỗi điểm danh KHÔNG được làm chết camera
            logger.exception("Lỗi ghi điểm danh cho %s", person_id)
            result["message"] = "Lỗi ghi điểm danh — xem logs/app.log để biết chi tiết"
            return result

    # ---------------------------------------------------------
    # Nội bộ (ghi)
    # ---------------------------------------------------------
    @staticmethod
    def _next_session_no(conn, person_id: str, work_date: str) -> int:
        """Số thứ tự PHIÊN kế tiếp của người trong ngày (1, 2, ...)."""
        row = conn.execute(
            "SELECT COUNT(*) FROM attendance"
            " WHERE person_id = ? AND work_date = ?",
            (person_id, work_date),
        ).fetchone()
        return int(row[0]) + 1 if row is not None else 1

    @staticmethod
    def _touch(conn, row_id: str, ts_now: str, note_in: str | None = None) -> None:
        """Cập nhật last_seen (và ghi chú giờ vào nếu có) cho phiên ``row_id``.

        KHÔNG đụng giờ vào/ra gốc — quét VÀO CA lặp chỉ đánh dấu "vẫn thấy".
        """
        if note_in:
            conn.execute(
                "UPDATE attendance SET last_seen = ?,"
                " seen_count = seen_count + 1, note_in = ? WHERE id = ?",
                (ts_now, note_in, row_id),
            )
        else:
            conn.execute(
                "UPDATE attendance SET last_seen = ?,"
                " seen_count = seen_count + 1 WHERE id = ?",
                (ts_now, row_id),
            )

    # ---------------------------------------------------------
    # Đọc (dashboard)
    # ---------------------------------------------------------
    def list_today(self) -> list[AttendanceRecord]:
        """Điểm danh HÔM NAY (từng phiên), phiên vào sớm đứng trước."""
        rows = self._query(
            "WHERE a.work_date = ? ORDER BY a.check_in ASC", (today_local(),)
        )
        return rows

    def list_all(self) -> list[AttendanceRecord]:
        """Toàn bộ điểm danh mọi ngày (tra cứu/kỳ công) — theo ngày rồi giờ vào."""
        return self._query(
            "ORDER BY a.work_date ASC, a.check_in ASC",
            (),
        )

    def list_between(self, start: "date", end: "date") -> list[AttendanceRecord]:
        """Điểm danh trong khoảng ngày (đầu-cuối, theo work_date ĐỊA PHƯƠNG).

        Dùng cho kỳ công/chốt công (payroll): truyền 2 mốc ngày của kỳ —
        sắp xếp theo ngày rồi giờ vào để tính công theo đúng thứ tự.
        """
        return self._query(
            "WHERE a.work_date >= ? AND a.work_date <= ?"
            " ORDER BY a.work_date ASC, a.check_in ASC",
            (start.isoformat(), end.isoformat()),
        )

    def list_range(self, days: int = 7) -> list[AttendanceRecord]:
        """Điểm danh ``days`` ngày gần nhất (mới nhất trước — bảng tổng hợp)."""
        start = (
            datetime.now().astimezone().replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            - timedelta(days=days - 1)
        ).strftime("%Y-%m-%d")
        return self._query(
            "WHERE a.work_date >= ? ORDER BY a.work_date DESC, a.check_in DESC",
            (start,),
        )

    def stats_today(self, total_employees: int | None = None) -> dict[str, int]:
        """Số liệu tổng quan hôm nay cho các thẻ thống kê trên dashboard.

        v5 nhiều phiên/ngày — thống kê theo NGƯỜI (không đếm phiên trùng):
          - "checked_in"    = số NGƯỜI có ít nhất 1 phiên trong ngày.
          - "checked_out"   = số NGƯỜI đã đóng HẾT phiên (không còn phiên mở).
          - "still_working" = số NGƯỜI có phiên CHƯA RA CA còn trong cửa
            sổ ACTIVE_WINDOW (đang ở chỗ camera).
        """
        if total_employees is None:
            total_employees = self._people.count()
        records = self.list_today()
        # Gom phiên theo người — thống kê theo người, không theo phiên
        by_person: dict[str, list[AttendanceRecord]] = {}
        for r in records:
            by_person.setdefault(r.person_id, []).append(r)
        checked_out = sum(
            1 for recs in by_person.values()
            if recs and all(r.checked_out for r in recs)
        )
        present = sum(
            1 for recs in by_person.values()
            if any(is_still_working(r) for r in recs)
        )
        return {
            "total_employees": total_employees,
            "checked_in": len(by_person),        # số người đã điểm danh trong ngày
            "still_working": present,            # có phiên chưa ra + còn hoạt động
            "checked_out": checked_out,          # đã đóng HẾT phiên trong ngày
            "not_yet": max(0, total_employees - len(by_person)),
        }

    def export_csv(self, path, days: int = 30) -> int:
        """Xuất điểm danh ``days`` ngày ra file CSV (Excel mở được, UTF-8 BOM).

        Mỗi PHIÊN một dòng (v5) — cột 'Phiên' đánh số thứ tự trong ngày.
        Trả về số dòng đã ghi. Ghi đè file nếu đã tồn tại.
        """
        records = self.list_range(days)
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["Ngày", "Nhân viên", "Phòng ban", "Chức vụ", "Phiên",
                 "Vào", "Ra", "Ghi chú vào", "Ghi chú ra", "Số lần quét"]
            )
            for r in records:
                writer.writerow(
                    [
                        r.work_date,
                        r.name,
                        r.department,
                        r.position,
                        r.session_no,
                        fmt_time(r.check_in),
                        fmt_time(r.check_out),
                        r.note_in,
                        r.note_out,
                        r.seen_count,
                    ]
                )
        logger.info("Đã xuất điểm danh %d dòng → %s", len(records), path)
        return len(records)

    # ---------------------------------------------------------
    # Nội bộ (đọc)
    # ---------------------------------------------------------
    def _query(self, where: str, params: tuple) -> list[AttendanceRecord]:
        with self._db.session() as conn:
            rows = conn.execute(
                "SELECT a.person_id, a.work_date, a.session_no, a.check_in,"
                " a.check_out, a.checked_out, a.note_in, a.note_out,"
                " a.last_seen, a.seen_count, p.name, p.department, p.position"
                " FROM attendance a JOIN persons p ON p.id = a.person_id"
                f" {where}",
                params,
            ).fetchall()
        return [
            AttendanceRecord(
                person_id=row["person_id"],
                name=row["name"],
                # Cột DATETIME/DATE trên SQL Server trả datetime/date →
                # chuẩn hóa về chuỗi ISO như sqlite3.Row trả trước đây.
                work_date=to_iso_str(row["work_date"]),
                session_no=int(row["session_no"]) if "session_no" in row.keys() else 1,
                check_in=to_iso_str(row["check_in"]),
                check_out=to_iso_str(row["check_out"]) if row["check_out"] else None,
                last_seen=to_iso_str(row["last_seen"]),
                seen_count=int(row["seen_count"]),
                department=row["department"] if "department" in row.keys() else "",
                position=row["position"] if "position" in row.keys() else "",
                checked_out=bool(row["checked_out"]) if "checked_out" in row.keys() else False,
                note_in=row["note_in"] if "note_in" in row.keys() and row["note_in"] else "",
                note_out=row["note_out"] if "note_out" in row.keys() and row["note_out"] else "",
            )
            for row in rows
        ]
