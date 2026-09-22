"""Quản lý giao diện Dark/Light theme (Bước 13, FR-9).

Một bảng màu duy nhất, áp toàn cục qua ``QApplication.setStyleSheet``.
Quy tắc thiết kế:
  - Màu sắc nằm Ở ĐÂU (QSS), KHÔNG rải rác trong từng view.
  - Các nút/panel đặc biệt được đánh dấu bằng ``setObjectName`` và style
    theo tên: ``primaryBtn`` (xanh) · ``accentBtn`` (cam) · ``dangerBtn``
    (đỏ) · ``secondaryBtn`` (viền mảnh) · ``card`` · ``videoLabel`` ·
    ``imageLabel`` · ``thumbLabel`` · ``navList``.
  - View chỉ giữ lại thuộc tính BỐ CỤC inline (font-size, padding, weight).

Bảng màu 2026 — phong cách app CHẤM CÔNG (dashboard hiện đại):
  - Sidebar gradient xanh đậm + item bo tròn, chọn = viên xanh gradient.
  - Thẻ thống kê (``statCard``) nền nổi, số to màu nhấn.
  - Tiêu đề trang (``pageTitle`` / ``pageSubtitle``) cỡ lớn, phân cấp rõ.

Cách dùng:
    from app.ui.theme import apply_theme
    apply_theme(QApplication.instance(), config.theme)   # khi khởi động
    apply_theme(QApplication.instance(), "light")        # khi bấm nút S/T
"""
from __future__ import annotations

# ============================================================
# THEME TỐI (dark) — mặc định
# ============================================================
DARK_QSS = """
QWidget {
    background-color: #0f1420;
    color: #e6eaf2;
    font-family: "Segoe UI", "Arial", sans-serif;
}
QLabel { background: transparent; }
QLabel#sectionTitle {
    color: #8b93a7;
    font-weight: bold;
    letter-spacing: 1px;
    font-size: 12px;
}
QLabel#pageTitle {
    font-size: 22px;
    font-weight: bold;
    color: #f2f5fb;
}
QLabel#pageSubtitle {
    font-size: 13px;
    color: #8b93a7;
}
QLabel#statValue {
    font-size: 26px;
    font-weight: bold;
}
QLabel#statLabel {
    font-size: 12px;
    color: #8b93a7;
    letter-spacing: 0.5px;
}
QToolTip {
    background-color: #1a2233; color: #e6eaf2;
    border: 1px solid #2c3852; border-radius: 6px;
    padding: 5px 8px;
}
QCheckBox { spacing: 6px; }
QCheckBox::indicator {
    width: 16px; height: 16px;
    border: 1px solid #2c3852; border-radius: 4px;
    background-color: #1a2233;
}
QCheckBox::indicator:checked {
    background-color: #3b82f6; border-color: #3b82f6;
    image: none;
}

QLineEdit, QComboBox, QSpinBox {
    background-color: #161d2d;
    color: #e6eaf2;
    border: 1px solid #2c3852;
    border-radius: 8px;
    padding: 7px 10px;
    selection-background-color: #3b82f6;
}
QLineEdit:focus, QComboBox:focus { border-color: #3b82f6; }
QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView {
    background-color: #161d2d; color: #e6eaf2;
    selection-background-color: #1d4ed8;
}

QPushButton {
    background-color: #1d2637;
    color: #dfe5f1;
    border: 1px solid #2c3852;
    border-radius: 8px;
    padding: 8px 16px;
}
QPushButton:hover { background-color: #242f45; }
QPushButton:pressed { background-color: #161d2d; }
QPushButton:disabled { background-color: #171d2b; color: #5d6679; }

QPushButton#primaryBtn {
    background-color: #3b82f6; color: white;
    border: none; font-weight: bold;
}
QPushButton#primaryBtn:hover { background-color: #2563eb; }
QPushButton#primaryBtn:disabled { background-color: #1e3a5f; color: #7ea6d9; }

QPushButton#accentBtn {
    background-color: #f59e0b; color: #1a1206;
    border: none; font-weight: bold;
}
QPushButton#accentBtn:hover { background-color: #d97706; }
QPushButton#accentBtn:disabled { background-color: #59431f; color: #b9a478; }

QPushButton#dangerBtn {
    background-color: transparent; color: #f87171;
    border: 1px solid #7f3b3b;
}
QPushButton#dangerBtn:hover { background-color: #3a2222; }

QPushButton#secondaryBtn {
    background-color: transparent; color: #c3ccdd;
    border: 1px solid #2c3852;
}
QPushButton#secondaryBtn:hover { border-color: #3b82f6; color: #7db1ff; }

QListWidget, QTableWidget, QTreeWidget {
    background-color: #121928;
    border: 1px solid #232d42;
    border-radius: 10px;
    alternate-background-color: #151d2e;
}
QListWidget::item { padding: 4px; }
QListWidget::item:selected, QTableWidget::item:selected {
    background-color: #1d4ed8; color: white;
}
QTableWidget { gridline-color: #1e2739; }
QHeaderView::section {
    background-color: #151d2e; color: #9aa3b8;
    border: none; border-bottom: 1px solid #232d42;
    padding: 8px; font-weight: bold;
}
QListWidget#navList {
    background-color: transparent;
    border: none;
    border-radius: 12px;
    font-size: 14px;
    padding: 6px;
}
QListWidget#navList::item {
    padding: 12px 14px;
    min-height: 24px;
    border-radius: 9px;
    margin: 2px 6px;
}
QListWidget#navList::item:hover {
    background-color: rgba(59, 130, 246, 0.18);
    color: #cfe0ff;
}
QListWidget#navList::item:selected {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #2563eb, stop:1 #3b82f6);
    color: white; font-weight: bold;
    border-left: none; padding-left: 14px;
}

QPushButton#sidebarBtn {
    background-color: rgba(255, 255, 255, 0.06);
    color: #cbd5e6;
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 8px;
    padding: 8px;
    font-size: 13px;
}
QPushButton#sidebarBtn:hover {
    background-color: rgba(255, 255, 255, 0.12); color: white;
}
QPushButton#sidebarBtn:pressed { background-color: #3b82f6; }

QFrame#sidebar {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #182136, stop:1 #0f1420);
    border-right: 1px solid #232d42;
}
QDialog { background-color: #0f1420; }
QMessageBox { background-color: #0f1420; }

QLabel#card, QFrame#card {
    background-color: #161d2d;
    border: 1px solid #232d42;
    border-radius: 12px;
}
QFrame#statCard {
    background-color: #161d2d;
    border: 1px solid #232d42;
    border-radius: 12px;
}
QLabel#videoLabel, QLabel#imageLabel {
    background-color: #05070c; color: #6d7688;
    border: 1px solid #232d42; border-radius: 10px;
}
QLabel#thumbLabel {
    background-color: #1d2637;
    border: 1px solid #2c3852;
    border-radius: 10px;
}

QSlider::groove:horizontal {
    height: 6px; background: #232d42;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    width: 16px; height: 16px; margin: -5px 0;
    background: #3b82f6; border-radius: 8px;
}
QSlider::sub-page:horizontal {
    background: #3b82f6; border-radius: 3px;
}
QSlider::handle:horizontal:hover { background: #60a5fa; }

QScrollBar:vertical { background: transparent; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: #2c3852; border-radius: 5px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: #3d4b6b; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 0; }
QScrollBar::handle:horizontal { background: #2c3852; border-radius: 5px; min-width: 30px; }
"""

# ============================================================
# THEME SÁNG (light)
# ============================================================
LIGHT_QSS = """
QWidget {
    background-color: #f3f5f9;
    color: #1c2333;
    font-family: "Segoe UI", "Arial", sans-serif;
}
QLabel { background: transparent; }
QLabel#sectionTitle {
    color: #69748c;
    font-weight: bold;
    letter-spacing: 1px;
    font-size: 12px;
}
QLabel#pageTitle {
    font-size: 22px;
    font-weight: bold;
    color: #10182b;
}
QLabel#pageSubtitle {
    font-size: 13px;
    color: #69748c;
}
QLabel#statValue {
    font-size: 26px;
    font-weight: bold;
}
QLabel#statLabel {
    font-size: 12px;
    color: #69748c;
    letter-spacing: 0.5px;
}
QToolTip {
    background-color: #ffffff; color: #1c2333;
    border: 1px solid #d3dae6; border-radius: 6px;
    padding: 5px 8px;
}
QCheckBox { spacing: 6px; }
QCheckBox::indicator {
    width: 16px; height: 16px;
    border: 1px solid #c9d2e0; border-radius: 4px;
    background-color: #ffffff;
}
QCheckBox::indicator:checked {
    background-color: #2563eb; border-color: #2563eb;
    image: none;
}

QLineEdit, QComboBox, QSpinBox {
    background-color: #ffffff;
    color: #1c2333;
    border: 1px solid #c9d2e0;
    border-radius: 8px;
    padding: 7px 10px;
    selection-background-color: #2563eb;
}
QLineEdit:focus, QComboBox:focus { border-color: #2563eb; }
QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView {
    background-color: #ffffff; color: #1c2333;
    selection-background-color: #d3e3fb;
}

QPushButton {
    background-color: #ffffff;
    color: #1c2333;
    border: 1px solid #c9d2e0;
    border-radius: 8px;
    padding: 8px 16px;
}
QPushButton:hover { background-color: #eef2f8; }
QPushButton:pressed { background-color: #e2e8f2; }
QPushButton:disabled { background-color: #f0f2f6; color: #9aa3b5; }

QPushButton#primaryBtn {
    background-color: #2563eb; color: white;
    border: none; font-weight: bold;
}
QPushButton#primaryBtn:hover { background-color: #1d4ed8; }
QPushButton#primaryBtn:disabled { background-color: #a9c6f0; color: #e8f0fa; }

QPushButton#accentBtn {
    background-color: #f59e0b; color: #241703;
    border: none; font-weight: bold;
}
QPushButton#accentBtn:hover { background-color: #d97706; }
QPushButton#accentBtn:disabled { background-color: #e0c290; color: #f8efe0; }

QPushButton#dangerBtn {
    background-color: #fdf2f2; color: #c02626;
    border: 1px solid #e3a1a1;
}
QPushButton#dangerBtn:hover { background-color: #fbe3e3; }

QPushButton#secondaryBtn {
    background-color: #fafbfd; color: #333c4f;
    border: 1px solid #b9c3d4;
}
QPushButton#secondaryBtn:hover { border-color: #2563eb; color: #2563eb; }

QListWidget, QTableWidget, QTreeWidget {
    background-color: #ffffff;
    border: 1px solid #d8dee9;
    border-radius: 10px;
    alternate-background-color: #f6f8fb;
}
QListWidget::item { padding: 4px; }
QListWidget::item:selected, QTableWidget::item:selected {
    background-color: #d3e3fb; color: #10254d;
}
QTableWidget { gridline-color: #e6eaf1; }
QHeaderView::section {
    background-color: #f2f5fa; color: #5c6680;
    border: none; border-bottom: 1px solid #d8dee9;
    padding: 8px; font-weight: bold;
}
QListWidget#navList {
    background-color: transparent;
    border: none;
    border-radius: 12px;
    font-size: 14px;
    padding: 6px;
}
QListWidget#navList::item {
    padding: 12px 14px;
    min-height: 24px;
    border-radius: 9px;
    margin: 2px 6px;
}
QListWidget#navList::item:hover {
    background-color: #e3edfc;
    color: #1a5fb4;
}
QListWidget#navList::item:selected {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #2563eb, stop:1 #3b82f6);
    color: white; font-weight: bold;
    border-left: none; padding-left: 14px;
}

QPushButton#sidebarBtn {
    background-color: #e9edf4;
    color: #3c4658;
    border: 1px solid #d5dbe7;
    border-radius: 8px;
    padding: 8px;
    font-size: 13px;
}
QPushButton#sidebarBtn:hover { background-color: #d9e1ee; color: #10182b; }
QPushButton#sidebarBtn:pressed { background-color: #2563eb; color: white; }

QFrame#sidebar {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #ffffff, stop:1 #eef1f7);
    border-right: 1px solid #d8dee9;
}
QDialog { background-color: #f3f5f9; }
QMessageBox { background-color: #f3f5f9; }

QLabel#card, QFrame#card {
    background-color: #ffffff;
    border: 1px solid #d8dee9;
    border-radius: 12px;
}
QFrame#statCard {
    background-color: #ffffff;
    border: 1px solid #d8dee9;
    border-radius: 12px;
}
QLabel#videoLabel, QLabel#imageLabel {
    background-color: #05070c; color: #8a92a2;
    border: 1px solid #232d42; border-radius: 10px;
}
QLabel#thumbLabel {
    background-color: #eef1f6;
    border: 1px solid #d8dee9;
    border-radius: 10px;
}

QSlider::groove:horizontal {
    height: 6px; background: #d8dee9;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    width: 16px; height: 16px; margin: -5px 0;
    background: #2563eb; border-radius: 8px;
}
QSlider::sub-page:horizontal {
    background: #2563eb; border-radius: 3px;
}
QSlider::handle:horizontal:hover { background: #1d4ed8; }

QScrollBar:vertical { background: transparent; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: #c3ccdb; border-radius: 5px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: #a8b3c6; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 0; }
QScrollBar::handle:horizontal { background: #c3ccdb; border-radius: 5px; min-width: 30px; }
"""


def apply_theme(app, theme: str) -> None:
    """Áp stylesheet cho toàn ứng dụng theo theme ('dark' hoặc 'light')."""
    stylesheet = DARK_QSS if theme == "dark" else LIGHT_QSS
    app.setStyleSheet(stylesheet)
