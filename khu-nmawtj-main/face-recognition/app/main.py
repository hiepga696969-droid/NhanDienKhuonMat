"""Entry point của ứng dụng — chạy bằng lệnh:  python -m app.main"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import PySide6  # noqa: F401  (cần để lấy đường dẫn thư mục plugins)

# Thêm thư mục gốc project vào sys.path để chạy được từ bất kỳ thư mục nào
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtWidgets import QApplication

from app.config import Config
from app.logger import setup_logging
from app.services.auth import AuthService
from app.ui.main_window import MainWindow
from app.ui.theme import apply_theme


def _configure_qt() -> None:
    """Khắc phục lỗi Qt không tìm thấy plugin khi đường dẫn có dấu cách.

    Qt tự suy diễn đường dẫn plugins từ vị trí file DLL, nhưng bị CẮT
    ngay tại dấu cách (ví dụ: `D:\\Html.css,js basic\\...`). Giải pháp:
    chỉ định đường dẫn plugins qua biến môi trường + thêm thư mục chứa
    DLL của PySide6 vào search path của Windows để nạp được các
    DLL phụ thuộc (Qt6Gui.dll...). Phải gọi TRƯỚC khi tạo QApplication.
    """
    pyside_dir = os.path.dirname(PySide6.__file__)
    os.environ.setdefault("QT_PLUGIN_PATH", os.path.join(pyside_dir, "plugins"))
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(pyside_dir)  # Windows: thêm vào DLL search path


def main() -> int:
    """Khởi động ứng dụng và trả về mã thoát."""
    setup_logging()
    _configure_qt()
    app = QApplication(sys.argv)
    app.setApplicationName("Điểm Danh Nhân Viên")
    app.setOrganizationName("EmployeeAttendance")

    # Nạp cấu hình + dịch vụ xác thực (Bước 2)
    config = Config.load()
    auth = AuthService(config)

    # Áp giao diện theo theme đã lưu (Bước 13 — Dark/Light)
    apply_theme(app, config.theme)

    window = MainWindow(auth, config)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
