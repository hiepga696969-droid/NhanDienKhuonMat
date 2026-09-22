"""Bước 3 — Kiểm tra CameraView + CameraCapture (chế độ minimal, tự thoát).

Test kiểm tra hành vi KHÔNG crash (đường lỗi mềm) và quy trình dừng camera
khi khóa. Dùng camera index 99 (KHÔNG tồn tại) để kết quả XÁC ĐỊNH trên mọi
máy — không phụ thuộc webcam thật (mở webcam thật đồng bộ có thể treo do
MSMF, như đã gặp ở Bước 12). Trên máy thật, người dùng xác nhận bằng mắt
video hiển thị.
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import PySide6  # noqa: E402

# Khắc phục lỗi đường dẫn plugins khi project path có dấu cách (giống main.py)
pyside_dir = os.path.dirname(PySide6.__file__)
os.environ["QT_PLUGIN_PATH"] = os.path.join(pyside_dir, "plugins")
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(pyside_dir)
os.environ["QT_QPA_PLATFORM"] = "minimal"  # chế độ ảo — không mở cửa sổ thật

import time  # noqa: E402

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.config import Config  # noqa: E402
from app.infrastructure.camera import CameraCapture  # noqa: E402
from app.services.auth import AuthService  # noqa: E402
from app.ui.main_window import MainWindow  # noqa: E402

app = QApplication(sys.argv)

# 0) CameraCapture mở LỖI (index 99 không tồn tại) phải trả False — không crash
capture = CameraCapture(99)
opened = capture.open()
print(f"[0] CameraCapture(99).open() = {opened} (False — camera không tồn tại)")
assert opened is False, "Camera 99 không tồn tại thì open() phải trả False"
capture.release()

auth = AuthService(Config(password_hash="giả-lập"))
config = Config(camera_index=99, camera_width=320, camera_height=240)
window = MainWindow(auth, config)
window.show()

# 1) Mô phỏng mở khóa → vào CameraView
# (Bản điểm danh: trang 0 = Tổng quan dashboard, CameraView là trang 1)
window._root.setCurrentWidget(window._content)
window.nav.setCurrentRow(1)
camera_view = window._camera_view
assert window.stack.currentIndex() == 1, "Phải ở màn hình CameraView (trang 1)"
print("[1] Chuyển tới màn hình CameraView: OK")


def after_wait() -> None:
    """Chạy sau khi worker đã có thời gian khởi động (1.5s)."""
    try:
        status = camera_view._status_label.text()
        print(f"[2] Trạng thái camera sau khi mở: {status!r}")
        assert status, "Phải có trạng thái hiển thị"

        # Dừng camera an toàn — chờ worker kết thúc HẲN (tối đa 3s).
        # Lưu ý: stop_camera giữ tham chiếu thread nếu chưa dừng kịp trong
        # 1.5s (an toàn đa luồng — Bước 12) → test chờ thêm cho chắc.
        camera_view.stop_camera()
        deadline = time.time() + 3
        while (
            camera_view._thread is not None
            and camera_view._thread.isRunning()
            and time.time() < deadline
        ):
            QApplication.processEvents()
            time.sleep(0.05)
        assert (
            camera_view._thread is None
            or not camera_view._thread.isRunning()
        ), "Thread camera phải dừng hẳn sau stop_camera"
        print("[3] Dừng camera an toàn: OK")

        # Khóa app → camera phải dừng (dù đang chạy hay không)
        window._lock_now()
        assert window._root.currentWidget() is window._lock_screen
        print("[4] Khóa app → quay về màn hình khóa: OK")

        print("\n=== TEST BƯỚC 3 QUA (không crash, luồng an toàn) ===")
    finally:
        # LUÔN thoát event loop — kể cả khi assert thất bại (nếu không,
        # app.exec() chạy mãi và test treo vô hạn)
        app.quit()


QTimer.singleShot(1800, after_wait)
sys.exit(app.exec())
