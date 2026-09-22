"""DashboardView — trang TỔNG QUAN điểm danh nhân viên (màn hình mở đầu).

Hiển thị:
  - 4 thẻ thống kê hôm nay: Đã điểm danh · Đang làm · Đã ra về · Chưa điểm danh.
  - Bảng điểm danh hôm nay: nhân viên, giờ vào, giờ ra, số lần quét, trạng thái.
  - Thẻ bên phải: tổng số nhân viên đã đăng ký + xuất CSV.
Dữ liệu đọc qua AttendanceService (view KHÔNG viết SQL). Nút "Làm mới"
hoặc quay lại trang này sẽ tải lại số liệu mới nhất.
"""
from __future__ import annotations

import logging
from datetime import date

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.infrastructure.db import Database
from app.services.attendance import (
    AttendanceService,
    fmt_datetime,
    fmt_time,
    is_still_working,
)
from app.infrastructure.repositories import PersonRepository

logger = logging.getLogger(__name__)

# Nhãn + màu trạng thái trên bảng (tách theo dấu | "text|color-hex")
# - "Đang làm"    : được thấy trong ACTIVE_WINDOW (cửa sổ hoạt động)
# - "Đã ra ca"    : đã bấm RA CA / giờ ra cố định đã xác nhận
# - "Chưa ra ca"  : đã vào ca nhưng chưa xác nhận RA CA
_ST_IN = "Đang làm|#f59e0b"
_ST_OUT = "Đã ra ca|#22c55e"
_ST_NOT_OUT = "Chưa ra ca|#64748b"


class DashboardView(QWidget):
    """Trang tổng quan: thống kê hôm nay + bảng điểm danh."""

    def __init__(self, db: Database, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._db = db
        self._attendance = AttendanceService(db)
        self._people = PersonRepository(db)
        self._build_ui()
        self.refresh_data()

    # ---------------------------------------------------------
    # Giao diện
    # ---------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(14)

        # ---- Tiêu đề trang + nút ----
        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("Tổng quan điểm danh")
        title.setObjectName("pageTitle")
        self._date_label = QLabel()
        self._date_label.setObjectName("pageSubtitle")
        title_box.addWidget(title)
        title_box.addWidget(self._date_label)
        header.addLayout(title_box)
        header.addStretch(1)

        export_btn = QPushButton("⬇  Xuất CSV")
        export_btn.setObjectName("secondaryBtn")
        export_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        export_btn.setToolTip("Xuất điểm danh 30 ngày gần nhất ra file CSV")
        export_btn.clicked.connect(self._on_export_csv)
        header.addWidget(export_btn)

        refresh_btn = QPushButton("⟳  Làm mới")
        refresh_btn.setObjectName("primaryBtn")
        refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        refresh_btn.clicked.connect(self.refresh_data)
        header.addWidget(refresh_btn)

        # Bộ lọc phòng ban (v3)
        self._dept_filter = QComboBox()
        self._dept_filter.addItem("Tất cả phòng ban")
        self._dept_filter.setToolTip("Chỉ hiện nhân viên của phòng ban được chọn")
        self._dept_filter.currentTextChanged.connect(lambda _: self.refresh_data())
        header.addWidget(self._dept_filter)
        root.addLayout(header)

        # ---- 4 thẻ thống kê ----
        cards = QHBoxLayout()
        cards.setSpacing(12)
        self._stat_cards: dict[str, QLabel] = {}
        for key, label in (
            ("checked_in", "ĐÃ ĐIỂM DANH"),
            ("still_working", "ĐANG LÀM"),
            ("checked_out", "ĐÃ RA VỀ"),
            ("not_yet", "CHƯA ĐIỂM DANH"),
        ):
            cards.addWidget(self._make_stat_card(key, label), stretch=1)
        root.addLayout(cards)

        # ---- Bảng điểm danh hôm nay ----
        table_title = QLabel("ĐIỂM DANH HÔM NAY")
        table_title.setObjectName("sectionTitle")
        root.addWidget(table_title)

        self._table = QTableWidget(0, 8)
        self._table.setObjectName("attendanceTable")
        self._table.setHorizontalHeaderLabels(
            ["Nhân viên", "Phòng ban · Chức vụ", "Giờ vào", "Giờ ra", "Ghi chú vào", "Ghi chú ra", "Thấy lần cuối", "Trạng thái"]
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.setShowGrid(False)
        header_view = self._table.horizontalHeader()
        header_view.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header_view.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        for col in (2, 3, 4, 5, 6, 7):
            header_view.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self._table.setStyleSheet("QTableWidget { border-radius: 12px; }")
        root.addWidget(self._table, stretch=1)

        # ---- Thanh dưới: tổng nhân viên ----
        footer = QHBoxLayout()
        self._total_label = QLabel()
        self._total_label.setObjectName("pageSubtitle")
        footer.addWidget(self._total_label)
        footer.addStretch(1)
        root.addLayout(footer)

    def _make_stat_card(self, key: str, label: str) -> QFrame:
        """Một thẻ thống kê bo góc: số to + nhãn nhỏ bên dưới."""
        card = QFrame()
        card.setObjectName("statCard")
        card.setMinimumHeight(92)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(2)

        value = QLabel("0")
        value.setObjectName("statValue")
        value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(value)

        text = QLabel(label)
        text.setObjectName("statLabel")
        text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(text)

        self._stat_cards[key] = value
        return card

    # ---------------------------------------------------------
    # Dữ liệu
    # ---------------------------------------------------------
    def refresh_data(self) -> None:
        """Tải lại số liệu + bảng (gọi khi vào trang / bấm Làm mới)."""
        try:
            records = self._attendance.list_today()
            stats = self._attendance.stats_today()
        except Exception:  # noqa: BLE001 — DB lỗi không được làm sập trang
            logger.exception("Lỗi tải dữ liệu điểm danh")
            records, stats = [], {}

        # Bộ lọc phòng ban (v3): "Tất cả phòng ban" + mọi phòng có trong DB
        try:
            departments = sorted(
                {p.department for p in self._people.list_all() if p.department}
            )
        except Exception:  # noqa: BLE001
            departments = []
        current = self._dept_filter.currentText()
        wanted = ["Tất cả phòng ban", *departments]
        if [self._dept_filter.itemText(i) for i in range(self._dept_filter.count())] != wanted:
            self._dept_filter.blockSignals(True)
            self._dept_filter.clear()
            self._dept_filter.addItems(wanted)
            # giữ lựa chọn cũ nếu vẫn còn, không thì về "Tất cả"
            idx = self._dept_filter.findText(current)
            self._dept_filter.setCurrentIndex(idx if idx >= 0 else 0)
            self._dept_filter.blockSignals(False)
        if self._dept_filter.currentIndex() > 0:
            selected = self._dept_filter.currentText()
            records = [r for r in records if r.department == selected]
        for key, value in stats.items():
            if key in self._stat_cards:
                self._stat_cards[key].setText(str(value))

        self._date_label.setText(f"Hôm nay · {date.today().strftime('%d/%m/%Y')}")
        total_txt = stats.get('total_employees', 0)
        filter_note = (
            f" · phòng {self._dept_filter.currentText()}"
            if self._dept_filter.currentIndex() > 0 else ""
        )
        self._total_label.setText(
            f"Tổng số nhân viên đã đăng ký: {total_txt}{filter_note}"
        )

        self._table.setRowCount(len(records))
        for row, rec in enumerate(records):
            # Trạng thái theo CỜ checked_out (v4): True = đã bấm RA CA (giờ ra
            # cố định đã xác nhận). VÀO CA / RA CA là 2 tính năng ĐỘC LẬP.
            if rec.checked_out:
                status_text, status_color = _ST_OUT.split("|")
            elif is_still_working(rec):
                status_text, status_color = _ST_IN.split("|")
            else:
                status_text, status_color = _ST_NOT_OUT.split("|")
            role = " · ".join(p for p in (rec.department, rec.position) if p)
            values = [
                rec.name,
                role,
                fmt_time(rec.check_in),
                fmt_time(rec.check_out),
                rec.note_in,
                rec.note_out,
                fmt_datetime(rec.last_seen),
            ]
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if col in (0, 1):
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
                    )
                self._table.setItem(row, col, item)
            status_item = QTableWidgetItem(status_text)
            status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            status_item.setForeground(Qt.GlobalColor.white)
            status_item.setBackground(QColor(status_color))
            self._table.setItem(row, 7, status_item)
        if not records:
            self._table.setRowCount(1)
            hint = QTableWidgetItem(
                "Chưa có ai điểm danh hôm nay — mở trang Điểm danh VÀO CA để bắt đầu."
            )
            hint.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(0, 0, hint)
            self._table.setSpan(0, 0, 1, 8)

    # ---------------------------------------------------------
    # Hành động
    # ---------------------------------------------------------
    def _on_export_csv(self) -> None:
        """Chọn nơi lưu + xuất CSV 30 ngày gần nhất."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Xuất điểm danh ra CSV",
            "diem_danh.csv",
            "CSV (*.csv)",
        )
        if not path:
            return
        try:
            count = self._attendance.export_csv(path, days=30)
        except Exception as exc:  # noqa: BLE001 — báo lỗi cho người dùng
            logger.exception("Lỗi xuất CSV")
            QMessageBox.warning(self, "Xuất CSV", f"Không xuất được file:\n{exc}")
            return
        QMessageBox.information(
            self, "Xuất CSV", f"Đã xuất {count} dòng điểm danh → {path}"
        )

    # ---------------------------------------------------------
    # Vòng đời
    # ---------------------------------------------------------
    def showEvent(self, event) -> None:  # noqa: N802 (chuẩn Qt)
        """Vào trang → tự tải dữ liệu mới nhất."""
        super().showEvent(event)
        self.refresh_data()
