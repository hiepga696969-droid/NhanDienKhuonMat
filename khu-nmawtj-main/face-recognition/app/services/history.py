"""HistoryService — truy vấn + xóa lịch sử nhận diện (Bước 11, wireframe 5.5.4).

View KHÔNG đụng SQL trực tiếp (giống PersonService ở Bước 8): mọi thao tác
lịch sử (liệt kê, lọc theo tên/nguồn/ngày, đếm cho phân trang, xóa kèm
file snapshot trên đĩa) đều đi qua service này — dễ đổi nguồn dữ liệu
(ví dụ D1 cloud ở Bước 15) và dễ kiểm thử.

Lưu ý thời gian: ``detected_at`` lưu theo UTC (hậu tố 'Z'). Bộ lọc ngày
đổi mốc 00:00 giờ ĐỊA PHƯƠNG sang UTC rồi so chuỗi ISO (cùng định dạng
nên so sánh chuỗi được — giống count_today ở Bước 9).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from app.infrastructure.db import DATA_DIR, Database
from app.infrastructure.repositories import (
    RecognitionEvent,
    RecognitionEventRepository,
)

logger = logging.getLogger(__name__)

# Nhãn hiển thị theo nguồn sự kiện (spec: webcam / photo / mobile)
SOURCE_LABELS: dict[str, str] = {
    "webcam": "Webcam",
    "photo": "Ảnh",
    "mobile": "Mobile",
}

# Nhãn + bộ lọc loại quét VÀO/RA CA (v4) — mode lưu trong recognition_events
MODE_LABELS: dict[str, str] = {
    "checkin": "🟢 Vào ca",
    "checkout": "🔴 Ra ca",
}

# Bộ lọc ngày: tên hiển thị → số ngày lùi về mốc 00:00 (None = tất cả)
DATE_RANGES: dict[str, int | None] = {
    "Tất cả": None,
    "Hôm nay": 0,
    "7 ngày qua": 6,
    "30 ngày qua": 29,
}


class HistoryService:
    """Liệt kê / lọc / đếm / xóa sự kiện nhận diện (audit trail)."""

    def __init__(self, db: Database) -> None:
        self._events = RecognitionEventRepository(db)

    # ---------------------------------------------------------
    # Đọc
    # ---------------------------------------------------------
    def source_label(self, source: str) -> str:
        """'webcam' → 'Webcam', 'photo' → 'Ảnh', 'mobile' → 'Mobile'."""
        return SOURCE_LABELS.get(source, source)

    def mode_label(self, mode: str) -> str:
        """'checkin' → '🟢 Vào ca', 'checkout' → '🔴 Ra ca'; rỗng → '—'."""
        if not mode:
            return "—"
        return MODE_LABELS.get(mode, mode)

    def list_events(
        self,
        query: str = "",
        source: str = "",
        status: str = "",
        date_range: str = "Tất cả",
        limit: int = 50,
        offset: int = 0,
        mode: str = "",
    ) -> list[RecognitionEvent]:
        """Sự kiện mới nhất trước, theo bộ lọc (tên / nguồn / loại / ngày)."""
        since = self._since_iso(DATE_RANGES.get(date_range))
        return self._events.list_events(
            query=query,
            source=source,
            status=status,
            since=since,
            limit=limit,
            offset=offset,
            mode=mode,
        )

    def count_events(
        self,
        query: str = "",
        source: str = "",
        status: str = "",
        date_range: str = "Tất cả",
        mode: str = "",
    ) -> int:
        """Tổng sự kiện theo cùng bộ lọc — dùng để tính số trang."""
        since = self._since_iso(DATE_RANGES.get(date_range))
        return self._events.count_events(
            query=query, source=source, status=status, since=since, mode=mode
        )

    def get_event(self, event_id: str) -> RecognitionEvent | None:
        """Một sự kiện theo id; None nếu không tồn tại."""
        return self._events.get(event_id)

    # ---------------------------------------------------------
    # Xóa
    # ---------------------------------------------------------
    def delete_event(self, event_id: str) -> bool:
        """Xóa sự kiện + file snapshot trên đĩa; True nếu xóa được.

        Xóa file NGAY CẢ khi file bị lỗi (OSError) — dòng DB đã xóa,
        không để file mồ côi nằm lại thư mục snapshots.
        """
        event = self._events.get(event_id)
        if event is None:
            return False
        self._events.delete(event_id)
        if event.snapshot_path:
            self._unlink_snapshot(event.snapshot_path)
        logger.info("Đã xóa sự kiện nhận diện %s (%s)", event.id, event.label)
        return True

    def delete_events(self, event_ids: list[str]) -> int:
        """Xóa NHIỀU sự kiện cùng lúc; trả về số sự kiện đã xóa thành công.

        Duyệt từng cái qua ``delete_event()`` để dùng chung logic (ghi
        outbox đồng bộ + dọn file snapshot trên đĩa). Id không tồn tại
        (đã bị xóa ở nơi khác) được bỏ qua.
        """
        deleted = 0
        for event_id in event_ids:
            if self.delete_event(event_id):
                deleted += 1
        return deleted

    # ---------------------------------------------------------
    # Tiện ích
    # ---------------------------------------------------------
    def _unlink_snapshot(self, snapshot_path: str) -> None:
        """Xóa file snapshot — CHỈ khi đường dẫn nằm TRONG thư mục dữ liệu.

        Chống path traversal (\"../...\" thoát ra ngoài) — dù đường dẫn do
        chính app ghi nên rất hiếm gặp, nhưng luôn kiểm tra cho an toàn.
        """
        if not snapshot_path:
            return
        path = (DATA_DIR / snapshot_path).resolve()
        if not path.is_relative_to(DATA_DIR.resolve()):
            logger.warning("Bỏ qua xóa snapshot ngoài thư mục dữ liệu: %s", snapshot_path)
            return
        try:
            path.unlink(missing_ok=True)
        except OSError:  # noqa: BLE001 — file đang bị khóa, vẫn xóa được dòng DB
            logger.warning("Không xóa được file snapshot %s", path)

    @staticmethod
    def _since_iso(days_ago: int | None) -> str | None:
        """Mốc 00:00 của ngày cách đây ``days_ago`` ngày, đổi sang UTC (ISO).

        None → không lọc theo ngày. detected_at lưu UTC → so chuỗi ISO
        cùng định dạng là so được (giống count_today của Bước 9).
        """
        if days_ago is None:
            return None
        midnight_local = (
            datetime.now()
            .replace(hour=0, minute=0, second=0, microsecond=0)
            - timedelta(days=days_ago)
        ).astimezone()
        return midnight_local.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")
