"""PayrollView — trang KỲ CÔNG / chốt công nhận lương.

Chọn loại kỳ (tháng / 2 lần-tháng / tuần / tùy chọn) + 2 mốc ngày → tính
công từng nhân viên từ bảng attendance:
  - Ngày công, nghỉ (ngày làm việc của kỳ − ngày công, trừ CN nếu chọn).
  - Đi muộn / về sớm (khi bật "Áp dụng giờ ca" + dung sai phút).
  - Tổng giờ công (ngày chưa RA CA được cộng giờ credit — không thiệt cho
    người quét quên ra ca).
Lọc nhân viên theo Phòng ban + Chức vụ + từng Nhân viên (combo) + nhập
LƯƠNG/NGÀY → TỔNG TIỀN CÔNG = Tổng ngày công × Lương/ngày: bảng có cột
Tiền công riêng từng NV, thẻ thống kê (kèm nhãn phạm vi) tính SAU lọc —
chọn 1 NV → tiền công riêng người đó; chọn phòng/chức vụ → tổng của
nhóm đó. Lọc/lương chỉ ảnh hưởng hiển thị, không truy vấn lại DB.
Nút "Xuất CSV" ghi báo cáo chấm công kỳ ra file CSV (Excel mở được,
UTF-8 BOM). Nhấp đúp 1 dòng → hộp thoại chi tiết công từng ngày.

View KHÔNG viết SQL — mọi số liệu qua PayrollService (dùng AttendanceService).
"""
from __future__ import annotations

import logging
from datetime import date

from PySide6.QtCore import QDate, Qt, QTime
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTimeEdit,
    QVBoxLayout,
    QWidget,
)

from app.infrastructure.db import Database
from app.services.payroll import (
    PayPeriod,
    PayrollRow,
    PayrollService,
    current_period,
)
from app.services.attendance import fmt_time

logger = logging.getLogger(__name__)

# Nhãn + màu trạng thái từng ngày trong hộp chi tiết ("text|color-hex")
_D_OK = "Đủ công|#22c55e"
_D_LATE = "Đi muộn|#f59e0b"
_D_EARLY = "Về sớm|#f59e0b"
_D_LATE_EARLY = "Muộn + sớm|#f97316"
_D_NO_OUT = "Chưa ra ca|#3b82f6"


class PayrollView(QWidget):
    """Trang kỳ công: chọn kỳ → tổng hợp công + tính tiền công + xuất CSV."""

    def __init__(self, db: Database, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._service = PayrollService(db)
        self._rows: list[PayrollRow] = []       # các dòng ĐANG hiển thị (đã lọc)
        self._all_rows: list[PayrollRow] = []   # toàn bộ dòng của kỳ (chưa lọc)
        self._period: PayPeriod | None = None
        self._build_ui()
        self._on_period_type_changed()  # đặt mốc mặc định + tính lần đầu

    # ---------------------------------------------------------
    # Giao diện
    # ---------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(12)

        # ---- Tiêu đề ----
        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("Kỳ công / Chốt công")
        title.setObjectName("pageTitle")
        self._period_label = QLabel()
        self._period_label.setObjectName("pageSubtitle")
        title_box.addWidget(title)
        title_box.addWidget(self._period_label)
        header.addLayout(title_box)
        header.addStretch(1)

        export_btn = QPushButton("⬇  Xuất CSV")
        export_btn.setObjectName("secondaryBtn")
        export_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        export_btn.setToolTip("Xuất báo cáo chấm công của kỳ ra file CSV")
        export_btn.clicked.connect(self._on_export_csv)
        header.addWidget(export_btn)

        refresh_btn = QPushButton("⟳  Tính công")
        refresh_btn.setObjectName("primaryBtn")
        refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        refresh_btn.setToolTip("Tính lại công theo kỳ đang chọn")
        refresh_btn.clicked.connect(self.refresh_data)
        header.addWidget(refresh_btn)
        root.addLayout(header)

        # ---- Thanh chọn kỳ + quy tắc ----
        bar = QFrame()
        bar.setObjectName("statCard")
        grid = QGridLayout(bar)
        grid.setContentsMargins(16, 10, 16, 10)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        grid.addWidget(QLabel("Loại kỳ"), 0, 0)
        self._type_combo = QComboBox()
        for key, label in (
            ("monthly", "Kỳ tháng (26 → 25)"),
            ("twice_monthly", "Kỳ 2 lần/tháng"),
            ("weekly", "Kỳ tuần (T2 → CN)"),
            ("custom", "Tùy chọn ngày"),
        ):
            self._type_combo.addItem(label, key)
        self._type_combo.currentIndexChanged.connect(self._on_period_type_changed)
        grid.addWidget(self._type_combo, 0, 1)

        grid.addWidget(QLabel("Từ ngày"), 0, 2)
        self._start_edit = QDateEdit()
        self._start_edit.setCalendarPopup(True)
        self._start_edit.setDisplayFormat("dd/MM/yyyy")
        self._start_edit.dateChanged.connect(self.refresh_data)
        grid.addWidget(self._start_edit, 0, 3)

        grid.addWidget(QLabel("Đến ngày"), 0, 4)
        self._end_edit = QDateEdit()
        self._end_edit.setCalendarPopup(True)
        self._end_edit.setDisplayFormat("dd/MM/yyyy")
        self._end_edit.dateChanged.connect(self.refresh_data)
        grid.addWidget(self._end_edit, 0, 5)

        self._sunday_check = QCheckBox("Trừ Chủ nhật khỏi ngày làm việc")
        self._sunday_check.setChecked(True)
        self._sunday_check.toggled.connect(self.refresh_data)
        grid.addWidget(self._sunday_check, 0, 6, 1, 3)

        # Quy tắc giờ ca (đi muộn / về sớm) + loại lương
        self._shift_check = QCheckBox("Áp dụng giờ ca")
        self._shift_check.setToolTip(
            "Bật để đếm đi muộn / về sớm theo giờ ca khai bên dưới"
        )
        self._shift_check.toggled.connect(self.refresh_data)
        grid.addWidget(self._shift_check, 1, 0, 1, 2)

        grid.addWidget(QLabel("Vào ca"), 1, 2)
        self._shift_start_edit = QTimeEdit(QTime(8, 0))
        self._shift_start_edit.setDisplayFormat("HH:mm")
        self._shift_start_edit.timeChanged.connect(self.refresh_data)
        grid.addWidget(self._shift_start_edit, 1, 3)

        grid.addWidget(QLabel("Ra ca"), 1, 4)
        self._shift_end_edit = QTimeEdit(QTime(17, 0))
        self._shift_end_edit.setDisplayFormat("HH:mm")
        self._shift_end_edit.timeChanged.connect(self.refresh_data)
        grid.addWidget(self._shift_end_edit, 1, 5)

        grid.addWidget(QLabel("Dung sai (phút)"), 1, 6)
        self._grace_spin = QSpinBox()
        self._grace_spin.setRange(0, 120)
        self._grace_spin.setValue(0)
        self._grace_spin.setSuffix("′")
        self._grace_spin.valueChanged.connect(self.refresh_data)
        grid.addWidget(self._grace_spin, 1, 7)

        self._hourly_check = QCheckBox("Tính theo lương giờ (credit 4h/ngày)")
        self._hourly_check.setToolTip(
            "Bỏ chọn: ngày chưa RA CA được tính 8 giờ công (lương tháng).\n"
            "Chọn: ngày chưa RA CA được tính 4 giờ công (lương giờ)."
        )
        self._hourly_check.toggled.connect(self.refresh_data)
        grid.addWidget(self._hourly_check, 1, 8)
        root.addWidget(bar)

        # ---- Lọc nhân viên + Lương/ngày (tính tiền công ngay trên trang) ----
        bar2 = QFrame()
        bar2.setObjectName("statCard")
        grid2 = QGridLayout(bar2)
        grid2.setContentsMargins(16, 10, 16, 10)
        grid2.setHorizontalSpacing(8)

        grid2.addWidget(QLabel("Phòng ban"), 0, 0)
        self._dept_filter = QComboBox()
        self._dept_filter.addItem("Tất cả phòng ban", "")
        self._dept_filter.setToolTip(
            "Chỉ hiện nhân viên của phòng ban được chọn (danh sách lấy từ\n"
            "phòng ban đã khai của các nhân viên có công trong kỳ)."
        )
        self._dept_filter.currentIndexChanged.connect(self._on_view_filter_changed)
        grid2.addWidget(self._dept_filter, 0, 1)

        grid2.addWidget(QLabel("Chức vụ"), 0, 2)
        self._pos_filter = QComboBox()
        self._pos_filter.addItem("Tất cả chức vụ", "")
        self._pos_filter.setToolTip(
            "Chỉ hiện nhân viên có chức vụ được chọn (kết hợp với bộ lọc\n"
            "phòng ban nếu cả hai đều đặt)."
        )
        self._pos_filter.currentIndexChanged.connect(self._on_view_filter_changed)
        grid2.addWidget(self._pos_filter, 0, 3)

        grid2.addWidget(QLabel("Nhân viên"), 0, 4)
        self._emp_filter = QComboBox()
        self._emp_filter.addItem("Tất cả nhân viên", "")
        self._emp_filter.setToolTip(
            "Chọn 1 nhân viên (trong phòng ban/chức vụ đã lọc) để xem RIÊNG\n"
            "ngày công + TỔNG TIỀN CÔNG của người đó ở thẻ thống kê."
        )
        self._emp_filter.currentIndexChanged.connect(self._on_view_filter_changed)
        grid2.addWidget(self._emp_filter, 0, 5)

        grid2.addWidget(QLabel("Lương/ngày"), 0, 6)
        self._salary_spin = QDoubleSpinBox()
        self._salary_spin.setRange(0, 1_000_000_000)
        self._salary_spin.setDecimals(0)
        self._salary_spin.setSuffix(" đ")
        self._salary_spin.setToolTip(
            "Mức LƯƠNG/NGÀY (project chưa lưu lương trong DB nên nhập ở đây).\n"
            "TỔNG TIỀN CÔNG = Tổng ngày công × Lương/ngày."
        )
        self._salary_spin.valueChanged.connect(self._on_view_filter_changed)
        grid2.addWidget(self._salary_spin, 0, 7, 1, 2)
        root.addWidget(bar2)

        # ---- 5 thẻ thống kê kỳ ----
        cards = QHBoxLayout()
        cards.setSpacing(12)
        self._stat_cards: dict[str, QLabel] = {}   # giá trị từng thẻ
        self._stat_labels: dict[str, QLabel] = {}  # nhãn từng thẻ (đổi theo lọc)
        for key, label in (
            ("employees", "CÓ CÔNG TRONG KỲ"),
            ("work_days", "TỔNG NGÀY CÔNG"),
            ("total_hours", "TỔNG GIỜ CÔNG"),
            ("late", "SỐ LẦN ĐI MUỘN"),
            ("pay", "TỔNG TIỀN CÔNG"),
        ):
            cards.addWidget(self._make_stat_card(key, label), stretch=1)
        root.addLayout(cards)

        # ---- Bảng tổng hợp ----
        self._table = QTableWidget(0, 9)
        self._table.setObjectName("attendanceTable")
        self._table.setHorizontalHeaderLabels(
            [
                "Nhân viên",
                "Phòng ban · Chức vụ",
                "Ngày công",
                "Nghỉ (ngày)",
                "Đi muộn",
                "Về sớm",
                "Chưa ra ca",
                "Tổng giờ công",
                "Tiền công",
            ]
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.setShowGrid(False)
        self._table.doubleClicked.connect(self._on_row_double_clicked)
        header_view = self._table.horizontalHeader()
        header_view.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header_view.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        for col in range(2, 9):
            header_view.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self._table.setStyleSheet("QTableWidget { border-radius: 12px; }")
        self._table.setToolTip("Nhấp đúp 1 dòng để xem chi tiết công từng ngày")
        root.addWidget(self._table, stretch=1)

    def _make_stat_card(self, key: str, label: str) -> QFrame:
        """Thẻ thống kê bo góc (giống DashboardView)."""
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
        self._stat_labels[key] = text
        return card

    # ---------------------------------------------------------
    # Chọn kỳ + dữ liệu
    # ---------------------------------------------------------
    def _current_period(self) -> PayPeriod:
        """PayPeriod hiện tại theo combo + 2 QDateEdit (kind = 'custom' khi
        người dùng chỉnh tay mốc nào đó so với mặc định)."""
        start = self._start_edit.date().toPython()
        end = self._end_edit.date().toPython()
        if start > end:
            start, end = end, start  # đảo mốc nếu người dùng chọn ngược
        kind = self._type_combo.currentData() or "custom"
        label = self._type_combo.currentText()
        return PayPeriod(kind, start, end, label)

    def _on_period_type_changed(self) -> None:
        """Đổi loại kỳ → đặt 2 mốc ngày mặc định của kỳ chứa hôm nay."""
        kind = self._type_combo.currentData() or "custom"
        period = current_period(kind)
        for edit, d in ((self._start_edit, period.start), (self._end_edit, period.end)):
            edit.blockSignals(True)
            edit.setDate(QDate(d.year, d.month, d.day))
            edit.blockSignals(False)
        self.refresh_data()

    def refresh_data(self) -> None:
        """Tính công theo kỳ đang chọn (truy vấn DB) rồi dựng hiển thị."""
        period = self._current_period()
        self._period = period
        try:
            shift_start = (
                self._shift_start_edit.time().toPython()
                if self._shift_check.isChecked()
                else None
            )
            shift_end = (
                self._shift_end_edit.time().toPython()
                if self._shift_check.isChecked()
                else None
            )
            rows = self._service.summarize(
                period,
                exclude_sundays=self._sunday_check.isChecked(),
                shift_start=shift_start,
                shift_end=shift_end,
                late_grace_minutes=self._grace_spin.value(),
                hourly=self._hourly_check.isChecked(),
            )
        except Exception:  # noqa: BLE001 — DB lỗi không được làm sập trang
            logger.exception("Lỗi tính kỳ công")
            rows = []

        self._all_rows = rows
        self._period_label.setText(period.describe())
        self._render_rows()

    def _on_view_filter_changed(self, *_args) -> None:
        """Đổi bộ lọc NV hoặc Lương/ngày → dựng lại hiển thị (KHÔNG truy vấn DB)."""
        self._render_rows()

    def _sync_filter_combos(self) -> None:
        """Đổ lại combo Phòng ban/Chức vụ/Nhân viên từ các dòng của kỳ.

        Phòng ban + chức vụ lấy từ nhân viên CÓ CÔNG trong kỳ; combo Nhân
        viên chỉ liệt kê người khớp bộ lọc phòng ban/chức vụ đang chọn.
        Giữ lựa chọn hiện tại nếu còn hợp lệ; blockSignals khi rebuild để
        không tạo vòng lặp signal.
        """
        departments = sorted({r.department for r in self._all_rows if r.department})
        positions = sorted({r.position for r in self._all_rows if r.position})
        for combo, all_label, values in (
            (self._dept_filter, "Tất cả phòng ban", departments),
            (self._pos_filter, "Tất cả chức vụ", positions),
        ):
            current = combo.currentData() or ""
            wanted = [all_label, *values]
            if [combo.itemText(i) for i in range(combo.count())] == wanted:
                continue
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(all_label, "")
            for v in values:
                combo.addItem(v, v)
            idx = combo.findData(current)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            combo.blockSignals(False)

        # Combo Nhân viên: chỉ người khớp lọc phòng ban/chức vụ hiện tại
        base = self._apply_dept_pos_filter(self._all_rows)
        people = sorted(base, key=lambda r: r.name.lower())
        combo = self._emp_filter
        current_pid = combo.currentData() or ""
        wanted = [("Tất cả nhân viên", "")]
        wanted.extend((r.name, r.person_id) for r in people)
        have = [(combo.itemText(i), combo.itemData(i)) for i in range(combo.count())]
        if have != wanted:
            combo.blockSignals(True)
            combo.clear()
            for text, data in wanted:
                combo.addItem(text, data)
            idx = combo.findData(current_pid)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            combo.blockSignals(False)

    def _apply_dept_pos_filter(self, rows: list[PayrollRow]) -> list[PayrollRow]:
        """Lọc theo combo Phòng ban + Chức vụ (khớp chính xác; AND nếu cả
        hai đều đặt). 'Tất cả' → giữ nguyên danh sách."""
        out = rows
        dept = self._dept_filter.currentData() or ""
        if dept:
            out = [r for r in out if r.department == dept]
        pos = self._pos_filter.currentData() or ""
        if pos:
            out = [r for r in out if r.position == pos]
        return out

    def _apply_view_filter(self, rows: list[PayrollRow]) -> list[PayrollRow]:
        """Lọc danh sách: Phòng ban + Chức vụ + Nhân viên (nếu chọn riêng).
        Chọn 1 NV → kết quả chỉ còn dòng của người đó."""
        out = self._apply_dept_pos_filter(rows)
        emp_id = self._emp_filter.currentData() or ""
        if emp_id:
            out = [r for r in out if r.person_id == emp_id]
        return out

    def _render_rows(self) -> None:
        """Đổ thẻ thống kê + bảng từ self._all_rows áp dụng bộ lọc hiện có.

        Chỉ dùng dữ liệu đã tính — không truy vấn DB, nên gọi lại thoải mái
        khi người dùng gõ bộ lọc hoặc sửa Lương/ngày.
        """
        self._sync_filter_combos()
        frows = self._apply_view_filter(self._all_rows)
        self._rows = frows

        # Thẻ thống kê (theo kết quả SAU lọc — bỏ lọc = cả kỳ)
        total_hours = sum(r.total_hours for r in frows)
        stats = {
            "employees": len(frows),
            "work_days": sum(r.work_days for r in frows),
            "total_hours": f"{total_hours:.1f}",
            "late": sum(r.late_count for r in frows),
        }
        for key, value in stats.items():
            if key in self._stat_cards:
                self._stat_cards[key].setText(str(value))

        # Nhãn thẻ tiền công: nêu rõ phạm vi đang chọn (NV / phòng / chức vụ)
        pay_label = self._stat_labels.get("pay")
        if pay_label is not None:
            emp_id = self._emp_filter.currentData() or ""
            dept = self._dept_filter.currentData() or ""
            pos = self._pos_filter.currentData() or ""
            if emp_id:
                scope = self._emp_filter.currentText()  # tiền công RIÊNG NV này
            else:
                scope = " · ".join(
                    part
                    for part in (
                        f"phòng {dept}" if dept else "",
                        f"chức vụ {pos}" if pos else "",
                    )
                    if part
                )
            pay_label.setText(f"TỔNG TIỀN CÔNG — {scope}" if scope else "TỔNG TIỀN CÔNG")

        # TỔNG TIỀN CÔNG = Tổng ngày công × Lương/ngày
        money = sum(r.work_days for r in frows) * self._salary_spin.value()
        if "pay" in self._stat_cards:
            self._stat_cards["pay"].setText(f"{money:,.0f} đ".replace(",", "."))

        # Bảng tổng hợp
        self._table.setRowCount(len(frows))
        for row_i, r in enumerate(frows):
            money = r.work_days * self._salary_spin.value()
            values = [
                r.name,
                " · ".join(p for p in (r.department, r.position) if p),
                str(r.work_days),
                str(r.absent_days),
                str(r.late_count),
                str(r.early_leave_count),
                str(r.missing_checkout),
                f"{r.total_hours:.1f}",
                f"{money:,.0f} đ".replace(",", "."),
            ]
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if col in (0, 1):
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
                    )
                self._table.setItem(row_i, col, item)

        if not self._all_rows:
            self._table.setRowCount(1)
            hint = QTableWidgetItem(
                "Không có điểm danh nào trong kỳ — hãy quét VÀO CA trước, "
                "hoặc chọn kỳ khác."
            )
            hint.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(0, 0, hint)
            self._table.setSpan(0, 0, 1, 9)
        elif not frows:
            self._table.setRowCount(1)
            hint = QTableWidgetItem(
                "Không có ai khớp bộ lọc Phòng ban/Chức vụ/Nhân viên — "
                "đặt lại 'Tất cả' để xem toàn bộ."
            )
            hint.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(0, 0, hint)
            self._table.setSpan(0, 0, 1, 9)
        else:
            self._table.clearSpans()

    # ---------------------------------------------------------
    # Hộp thoại chi tiết công từng ngày
    # ---------------------------------------------------------
    def _on_row_double_clicked(self, index) -> None:
        """Nhấp đúp dòng → chi tiết công từng ngày của nhân viên."""
        row_i = index.row()
        if not (0 <= row_i < len(self._rows)):
            return  # dòng gợi ý (bảng rỗng) — không phải dữ liệu
        _show_day_detail_dialog(self, self._rows[row_i], self._period)

    # ---------------------------------------------------------
    # Xuất CSV
    # ---------------------------------------------------------
    def _on_export_csv(self) -> None:
        """Chọn nơi lưu + xuất CSV chấm công kỳ hiện tại (theo bộ lọc nếu có)."""
        if self._period is None:
            self.refresh_data()
            if self._period is None:
                return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Xuất chấm công kỳ ra CSV",
            f"cham_cong_{self._period.start.isoformat()}_{self._period.end.isoformat()}.csv",
            "CSV (*.csv)",
        )
        if not path:
            return
        try:
            count = self._service.export_csv(path, self._rows, self._period)
        except Exception as exc:  # noqa: BLE001 — báo lỗi cho người dùng
            logger.exception("Lỗi xuất CSV kỳ công")
            QMessageBox.warning(self, "Xuất CSV", f"Không xuất được file:\n{exc}")
            return
        QMessageBox.information(
            self, "Xuất CSV", f"Đã xuất {count} dòng chấm công → {path}"
        )

    # ---------------------------------------------------------
    # Vòng đời
    # ---------------------------------------------------------
    def showEvent(self, event) -> None:  # noqa: N802 (chuẩn Qt)
        """Vào trang → tự tính lại công mới nhất."""
        super().showEvent(event)
        self.refresh_data()


def _show_day_detail_dialog(parent: QWidget, row: PayrollRow, period: PayPeriod | None) -> None:
    """Hộp thoại chi tiết công từng ngày của 1 nhân viên trong kỳ."""
    dlg = QDialog(parent)
    dlg.setWindowTitle(f"Chi tiết công — {row.name}")
    dlg.resize(640, 420)
    layout = QVBoxLayout(dlg)

    info = QLabel(
        f"{row.name}"
        + (f" · {row.department} · {row.position}" if row.department or row.position else "")
        + (f"\n{period.describe()}" if period is not None else "")
    )
    info.setObjectName("pageSubtitle")
    layout.addWidget(info)

    table = QTableWidget(len(row.days), 6)
    table.setHorizontalHeaderLabels(
        ["Ngày", "Giờ vào", "Giờ ra", "Ghi chú vào", "Ghi chú ra", "Giờ công / Trạng thái"]
    )
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    table.setAlternatingRowColors(True)
    header_view = table.horizontalHeader()
    header_view.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    header_view.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
    header_view.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)

    for i, wd in enumerate(sorted(row.days)):
        d = row.days[wd]
        if d.late and d.early_leave:
            status_text, color = _D_LATE_EARLY.split("|")
        elif d.late:
            status_text, color = _D_LATE.split("|")
        elif d.early_leave:
            status_text, color = _D_EARLY.split("|")
        elif d.missing_checkout:
            status_text, color = _D_NO_OUT.split("|")
        else:
            status_text, color = _D_OK.split("|")
        try:
            y, m, dd = (int(p) for p in wd.split("-"))
            wd_label = date(y, m, dd).strftime("%d/%m")
        except ValueError:
            wd_label = wd
        values = [
            wd_label,
            fmt_time(d.check_in) if d.check_in else "—",
            fmt_time(d.check_out) if d.check_out else "—",
            d.note_in,
            d.note_out,
            f"{d.hours:.1f}h · {status_text}" if not d.missing_checkout
            else f"~{d.hours:.1f}h · {status_text}",
        ]
        for col, text in enumerate(values):
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            table.setItem(i, col, item)
        last = table.item(i, 5)
        last.setForeground(Qt.GlobalColor.white)
        last.setBackground(QColor(color))

    layout.addWidget(table)

    btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    btns.rejected.connect(dlg.reject)
    btns.clicked.connect(dlg.close)
    layout.addWidget(btns)

    dlg.exec()
