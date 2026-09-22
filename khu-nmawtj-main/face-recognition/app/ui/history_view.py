"""HistoryView — Lịch sử nhận diện / Audit trail (Bước 11, wireframe 5.5.4).

Bảng liệt kê sự kiện (Thời gian · Người · Nguồn · Điểm · Trạng thái) với:
  - Bộ lọc kết hợp: tìm kiếm theo tên · nguồn (Webcam/Ảnh/Mobile) · ngày
    (Hôm nay / 7 ngày qua / 30 ngày qua).
  - Click dòng → panel chi tiết: ảnh snapshot + thông tin sự kiện.
    Snapshot của NGƯỜI LẠ bị làm mờ (QGraphicsBlurEffect) — nhất quán với
    quyền riêng tư ở CameraView.
  - [ Xóa sự kiện… ] → PasswordDialog (thao tác nhạy cảm, FR-11) → xóa
    dòng + file snapshot trên đĩa. Chọn NHIỀU dòng (Ctrl/Shift-click) rồi
    bấm nút để xóa hàng loạt trong một lần.
  - Phân trang 50 sự kiện/trang — lịch sử có thể rất dài theo thời gian.

Sử dụng QTableWidget (con của QTableView) — đủ tính năng cho dự án này
mà không cần viết model tùy biến (spec cho phép điều chỉnh bố cục nhẹ).
"""
from __future__ import annotations

import logging
from datetime import datetime

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QGraphicsBlurEffect,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.infrastructure.db import DATA_DIR, Database
from app.infrastructure.repositories import RecognitionEvent
from app.services.auth import AuthService
from app.services.history import DATE_RANGES, HistoryService
from app.ui.password_dialog import PasswordDialog

logger = logging.getLogger(__name__)

PAGE_SIZE = 50          # số sự kiện mỗi trang
SNAPSHOT_WIDTH = 200    # chiều rộng ảnh snapshot trong panel chi tiết
SNAPSHOT_HEIGHT = 150   # chiều cao ảnh snapshot

# Cột bảng (v4: thêm "Loại" — VÀO CA / RA CA)
COLUMNS = ["Thời gian", "Người", "Nguồn", "Loại", "Điểm", "Trạng thái"]

# Bộ lọc loại quét VÀO/RA CA (v4): (nhãn hiển thị, giá trị mode trong DB)
MODE_FILTERS: list[tuple[str, str]] = [
    ("Tất cả", ""),
    ("🟢 Vào ca", "checkin"),
    ("🔴 Ra ca", "checkout"),
]

# Các nguồn cho combo lọc: (nhãn hiển thị, giá trị lưu trong DB)
SOURCE_FILTERS: list[tuple[str, str]] = [
    ("Tất cả", ""),
    ("Webcam", "webcam"),
    ("Ảnh", "photo"),
    ("Mobile", "mobile"),
]

# Trạng thái sự kiện cho combo lọc: (nhãn hiển thị, giá trị truyền xuống
# repository). 'blocked' = bị CHẶN nhận diện (Giả mạo B17 / Mặt bị che B18)
# — xem nhanh các lần chặn trong Lịch sử.
STATUS_FILTERS: list[tuple[str, str]] = [
    ("Tất cả", ""),
    ("⚠ Bị chặn", "blocked"),
    ("✓ Xác nhận", "confirmed"),
    ("⚠ Chưa ĐK", "unknown"),
]


class HistoryView(QWidget):
    """Trang lịch sử nhận diện (FR-10 — audit trail)."""

    def __init__(
        self,
        db: Database,
        auth: AuthService,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._auth = auth
        self._service = HistoryService(db)
        self._page = 1
        self._total_pages = 1
        self._events: list[RecognitionEvent] = []  # sự kiện của trang hiện tại
        self._selected: RecognitionEvent | None = None
        self._build_ui()

    # ---------------------------------------------------------
    # Giao diện (wireframe 5.5.4)
    # ---------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        # Tiêu đề + tổng số
        header = QHBoxLayout()
        title = QLabel("LỊCH SỬ NHẬN DIỆN")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        header.addWidget(title)
        header.addStretch(1)
        self._count_label = QLabel()
        self._count_label.setStyleSheet("font-size: 13px;")
        header.addWidget(self._count_label)
        layout.addLayout(header)

        # Thanh bộ lọc: tìm kiếm · nguồn · ngày
        filters = QHBoxLayout()
        self._search = QLineEdit()
        self._search.setPlaceholderText("🔍 Tìm kiếm theo tên...")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._on_filter_changed)
        filters.addWidget(self._search, stretch=1)

        filters.addWidget(QLabel("Nguồn:"))
        self._source_combo = QComboBox()
        for label, value in SOURCE_FILTERS:
            self._source_combo.addItem(label, value)
        self._source_combo.currentIndexChanged.connect(self._on_filter_changed)
        filters.addWidget(self._source_combo)

        filters.addWidget(QLabel("Trạng thái:"))
        self._status_combo = QComboBox()
        for label, value in STATUS_FILTERS:
            self._status_combo.addItem(label, value)
        self._status_combo.currentIndexChanged.connect(self._on_filter_changed)
        filters.addWidget(self._status_combo)

        filters.addWidget(QLabel("Loại:"))
        self._mode_combo = QComboBox()
        for label, value in MODE_FILTERS:
            self._mode_combo.addItem(label, value)
        self._mode_combo.currentIndexChanged.connect(self._on_filter_changed)
        filters.addWidget(self._mode_combo)

        filters.addWidget(QLabel("Ngày:"))
        self._date_combo = QComboBox()
        self._date_combo.addItems(DATE_RANGES.keys())
        self._date_combo.currentIndexChanged.connect(self._on_filter_changed)
        filters.addWidget(self._date_combo)
        layout.addLayout(filters)

        # Bảng sự kiện
        self._table = QTableWidget(0, len(COLUMNS))
        self._table.setHorizontalHeaderLabels(COLUMNS)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        header_view = self._table.horizontalHeader()
        header_view.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)  # cột Người
        self._table.setColumnWidth(0, 175)
        self._table.setColumnWidth(2, 90)
        self._table.setColumnWidth(3, 90)
        self._table.setColumnWidth(4, 70)
        self._table.setColumnWidth(5, 110)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self._table, stretch=3)

        # Trạng thái rỗng
        self._empty_label = QLabel(
            "Chưa có sự kiện nào — mở webcam để bắt đầu nhận diện."
        )
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet("font-size: 14px; padding: 20px;")
        self._empty_label.hide()
        layout.addWidget(self._empty_label)

        # Panel chi tiết: snapshot + thông tin sự kiện
        detail = QHBoxLayout()
        detail.setSpacing(16)

        self._snapshot_label = QLabel("Chọn một dòng để xem chi tiết")
        self._snapshot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._snapshot_label.setFixedSize(SNAPSHOT_WIDTH + 20, SNAPSHOT_HEIGHT + 20)
        self._snapshot_label.setObjectName("card")
        self._snapshot_label.setStyleSheet("border-radius: 8px; font-size: 12px;")
        detail.addWidget(self._snapshot_label)

        info = QVBoxLayout()
        info.setSpacing(4)
        self._detail_sim = QLabel("Điểm tương đồng: —")
        self._detail_source = QLabel("Nguồn: —")
        self._detail_mode = QLabel("Loại quét: —")
        self._detail_event = QLabel("Sự kiện: —")
        self._detail_sync = QLabel("Đồng bộ cloud: —")
        for label in (self._detail_sim, self._detail_source, self._detail_mode,
                      self._detail_event, self._detail_sync):
            label.setStyleSheet("font-size: 13px;")
            info.addWidget(label)
        info.addStretch(1)
        detail.addLayout(info, stretch=1)
        layout.addLayout(detail)

        # Chân trang: phân trang + nút xóa
        footer = QHBoxLayout()
        self._prev_btn = QPushButton("‹ Trước")
        self._prev_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._prev_btn.clicked.connect(self._go_prev)
        footer.addWidget(self._prev_btn)

        self._page_label = QLabel("Trang 1/1")
        self._page_label.setStyleSheet("font-size: 13px;")
        footer.addWidget(self._page_label)

        self._next_btn = QPushButton("Sau ›")
        self._next_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._next_btn.clicked.connect(self._go_next)
        footer.addWidget(self._next_btn)
        footer.addStretch(1)

        self._delete_btn = QPushButton("🗑 Xóa sự kiện…")
        self._delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._delete_btn.setObjectName("dangerBtn")
        self._delete_btn.setEnabled(False)
        self._delete_btn.setStyleSheet("border-radius: 5px; padding: 6px 16px; font-size: 12px;")
        self._delete_btn.clicked.connect(self._on_delete)
        footer.addWidget(self._delete_btn)
        layout.addLayout(footer)

    # ---------------------------------------------------------
    # Dữ liệu & hiển thị
    # ---------------------------------------------------------
    def _refresh(self) -> None:
        """Đọc lại dữ liệu từ DB theo bộ lọc hiện tại rồi vẽ lại bảng."""
        query = self._search.text().strip()
        source = self._source_combo.currentData() or ""
        status = self._status_combo.currentData() or ""
        mode = self._mode_combo.currentData() or ""
        date_range = self._date_combo.currentText()

        total = self._service.count_events(
            query=query, source=source, status=status, date_range=date_range,
            mode=mode,
        )
        self._total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
        if self._page > self._total_pages:
            self._page = self._total_pages  # lọc làm ít kết quả hơn → về trang cuối hợp lệ

        offset = (self._page - 1) * PAGE_SIZE
        self._events = self._service.list_events(
            query=query, source=source, status=status, date_range=date_range,
            limit=PAGE_SIZE, offset=offset, mode=mode,
        )
        self._fill_table()

        if self._table.rowCount() > 0:
            self._table.selectRow(0)  # tự chọn dòng đầu → hiện chi tiết ngay
        else:
            self._selected = None
            self._update_detail(None)
            self._update_delete_btn()

        self._update_pagination(total)
        if self._table.rowCount() == 0:
            if query or source or status or mode or date_range != "Tất cả":
                self._empty_label.setText("Không có sự kiện nào khớp bộ lọc.")
            else:
                self._empty_label.setText(
                    "Chưa có sự kiện nào — mở webcam để bắt đầu nhận diện."
                )
        self._empty_label.setVisible(self._table.rowCount() == 0)

    def _fill_table(self) -> None:
        """Đổ các sự kiện của trang hiện tại vào bảng."""
        self._table.setRowCount(len(self._events))
        for i, ev in enumerate(self._events):
            time_item = QTableWidgetItem(self._format_time(ev.detected_at))
            time_item.setData(Qt.ItemDataRole.UserRole, ev.id)

            person_item = QTableWidgetItem(f"⚠ {ev.label}" if ev.is_unknown else ev.label)
            source_item = QTableWidgetItem(self._service.source_label(ev.source))
            mode_item = QTableWidgetItem(self._service.mode_label(ev.mode))
            # Tô màu loại quét: xanh lá (vào ca) / đỏ nhạt (ra ca) / mờ (khác)
            if ev.mode == "checkin":
                mode_item.setForeground(QColor("#2e7d32"))
            elif ev.mode == "checkout":
                mode_item.setForeground(QColor("#c33"))
            sim_item = QTableWidgetItem(
                f"{ev.similarity:.2f}" if ev.similarity is not None else "—"
            )
            status_item = QTableWidgetItem(self._status_label(ev))
            status_item.setForeground(self._status_color(ev))

            items = [time_item, person_item, source_item, mode_item, sim_item, status_item]
            for col, item in enumerate(items):
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._table.setItem(i, col, item)

    @staticmethod
    def _is_blocked(ev: RecognitionEvent) -> bool:
        """Sự kiện bị CHẶN nhận diện: Giả mạo (B17) hoặc Mặt bị che (B18)."""
        return ev.label.startswith(("Giả mạo:", "Mặt bị che:"))

    def _status_label(self, ev: RecognitionEvent) -> str:
        """Cột 'Trạng thái': ✓ Xác nhận / ⚠ Chưa ĐK / ⚠ Bị chặn."""
        if ev.is_unknown:
            return "⚠ Chưa ĐK"
        if self._is_blocked(ev):
            return "⚠ Bị chặn"
        return "✓ Xác nhận"

    def _status_color(self, ev: RecognitionEvent) -> QColor:
        """Màu cột trạng thái: đỏ (chưa ĐK/bị chặn) hay xanh (xác nhận)."""
        if ev.is_unknown or self._is_blocked(ev):
            return QColor("#c33")
        return QColor("#2e7d32")

    # ---------------------------------------------------------
    # Hành động
    # ---------------------------------------------------------
    def _on_filter_changed(self, *_args) -> None:
        """Bất kỳ bộ lọc nào đổi → về trang 1 rồi tải lại."""
        self._page = 1
        self._refresh()

    def _selected_ids(self) -> list[str]:
        """Id các sự kiện đang được chọn (theo dòng trên trang hiện tại)."""
        rows = sorted({item.row() for item in self._table.selectedItems()})
        return [self._events[r].id for r in rows if 0 <= r < len(self._events)]

    def _update_delete_btn(self) -> None:
        """Bật/tắt + đổi nhãn nút xóa theo số dòng đang chọn."""
        count = len(self._selected_ids())
        self._delete_btn.setEnabled(count > 0)
        self._delete_btn.setText(
            "🗑 Xóa sự kiện…" if count <= 1 else f"🗑 Xóa {count} sự kiện…"
        )

    def _on_selection_changed(self) -> None:
        """Click/chọn dòng → cập nhật panel chi tiết + nút xóa."""
        row = self._table.currentRow()
        if 0 <= row < len(self._events):
            self._selected = self._events[row]
        else:
            self._selected = None
        self._update_detail(self._selected)
        self._update_delete_btn()

    def _on_delete(self) -> None:
        """Xóa (các) sự kiện đang chọn — yêu cầu mật khẩu trước (FR-11)."""
        event_ids = self._selected_ids()
        if not event_ids:
            return
        if len(event_ids) == 1:
            event = next((e for e in self._events if e.id == event_ids[0]), None)
            operation = (
                f"Xóa sự kiện '{event.label}' ({self._format_time(event.detected_at)})"
                if event is not None else "Xóa sự kiện"
            )
            note = "Sự kiện và ảnh snapshot sẽ bị xóa vĩnh viễn."
        else:
            operation = f"Xóa {len(event_ids)} sự kiện đã chọn"
            note = (
                f"{len(event_ids)} sự kiện và ảnh snapshot kèm theo sẽ "
                "bị xóa vĩnh viễn."
            )
        if not PasswordDialog.require(self._auth, operation, self, note=note):
            return
        deleted = self._service.delete_events(event_ids)
        logger.info("Đã xóa %d sự kiện nhận diện", deleted)
        self._refresh()

    def _go_prev(self) -> None:
        if self._page > 1:
            self._page -= 1
            self._refresh()

    def _go_next(self) -> None:
        if self._page < self._total_pages:
            self._page += 1
            self._refresh()

    # ---------------------------------------------------------
    # Panel chi tiết
    # ---------------------------------------------------------
    def _update_detail(self, event: RecognitionEvent | None) -> None:
        """Hiển thị snapshot + thông tin của sự kiện (hoặc trạng thái trống)."""
        if event is None:
            self._snapshot_label.setPixmap(QPixmap())
            self._snapshot_label.setText("Chọn một dòng để xem chi tiết")
            self._detail_sim.setText("Điểm tương đồng: —")
            self._detail_source.setText("Nguồn: —")
            self._detail_mode.setText("Loại quét: —")
            self._detail_event.setText("Sự kiện: —")
            self._detail_sync.setText("Đồng bộ cloud: —")
            return

        self._snapshot_label.setText("")
        pixmap = self._load_snapshot(event)
        self._snapshot_label.setPixmap(pixmap)

        # Người lạ → ảnh bị LÀM MỜ (riêng tư — nhất quán với CameraView)
        if event.is_unknown:
            effect = QGraphicsBlurEffect(self)
            effect.setBlurRadius(18)
            self._snapshot_label.setGraphicsEffect(effect)
        else:
            self._snapshot_label.setGraphicsEffect(None)

        sim = (
            f"{event.similarity:.2f}"
            if event.similarity is not None
            else "—"
        )
        self._detail_sim.setText(f"Điểm tương đồng: {sim}")
        self._detail_source.setText(
            f"Nguồn: {self._service.source_label(event.source)}"
        )
        self._detail_mode.setText(
            f"Loại quét: {self._service.mode_label(event.mode)}"
        )
        self._detail_event.setText(f"Sự kiện: {event.id}")
        # Đồng bộ cloud chưa triển khai (Bước 15) — hiển thị trạng thái thật
        self._detail_sync.setText("Đồng bộ cloud: — (chưa triển khai, Bước 15)")

    def _load_snapshot(self, event: RecognitionEvent) -> QPixmap:
        """Đọc ảnh snapshot từ đĩa; pixmap rỗng nếu thiếu/ngoài thư mục dữ liệu."""
        if not event.snapshot_path:
            return QPixmap()
        path = (DATA_DIR / event.snapshot_path).resolve()
        if not path.is_relative_to(DATA_DIR.resolve()) or not path.exists():
            return QPixmap()
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            return QPixmap()
        return pixmap.scaled(
            SNAPSHOT_WIDTH,
            SNAPSHOT_HEIGHT,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def _update_pagination(self, total: int) -> None:
        """Cập nhật nhãn phân trang + trạng thái nút Trước/Sau + tổng số."""
        self._page_label.setText(f"Trang {self._page}/{self._total_pages}")
        self._prev_btn.setEnabled(self._page > 1)
        self._next_btn.setEnabled(self._page < self._total_pages)
        self._count_label.setText(f"Tổng cộng: {total} sự kiện")

    # ---------------------------------------------------------
    # Tiện ích hiển thị
    # ---------------------------------------------------------
    @staticmethod
    def _format_time(iso: str) -> str:
        """detected_at (ISO, UTC) → '14:32:05 · 02/08' giờ địa phương."""
        try:
            dt = datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone()
        except ValueError:
            return iso[:19].replace("T", " ")
        return dt.strftime("%H:%M:%S · %d/%m")

    # ---------------------------------------------------------
    # Vòng đời
    # ---------------------------------------------------------
    def showEvent(self, event) -> None:  # noqa: N802 (chuẩn Qt)
        """Mỗi khi vào trang → tải lại dữ liệu mới nhất (sự kiện vừa ghi xuất hiện)."""
        super().showEvent(event)
        self._refresh()
