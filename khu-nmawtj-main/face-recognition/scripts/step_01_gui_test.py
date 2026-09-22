"""
Bước 1 — Kiểm tra cửa sổ chính khởi tạo được.

Chạy ở chế độ "minimal" (không mở cửa sổ thật), tự động thoát sau 200ms.
Chỉ dùng để kiểm tra code — không phải một phần của ứng dụng.
"""
import os
import sys

# Thêm thư mục gốc project vào sys.path (script chạy từ scripts/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import PySide6

# Khắc phục lỗi đường dẫn plugins khi project path có dấu cách (giống main.py)
pyside_dir = os.path.dirname(PySide6.__file__)
os.environ["QT_PLUGIN_PATH"] = os.path.join(pyside_dir, "plugins")
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(pyside_dir)
os.environ["QT_QPA_PLATFORM"] = "minimal"  # chế độ ảo — không mở cửa sổ thật khi test

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from app.config import Config
from app.services.auth import AuthService
from app.ui.main_window import MainWindow

app = QApplication(sys.argv)
# Từ Bước 2 MainWindow cần auth + config (test cũ viết từ Bước 1)
window = MainWindow(AuthService(Config.load()), Config.load())  # KHÔNG show()
print(f"GUI OK — số màn hình: {window.stack.count()}")

window.nav.setCurrentRow(4)  # mô phỏng người dùng bấm chọn màn hình "Lịch sử"
print(f"Điều hướng OK — trang hiện tại: {window.stack.currentIndex()}")

QTimer.singleShot(200, app.quit)
sys.exit(app.exec())
