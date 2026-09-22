"""Bước 2 — Kiểm tra màn hình khóa + PasswordDialog (chế độ minimal, tự thoát).

Viết lại theo spec `password-management-spec.md` (2026-09-03) cho LockScreen
MỚI — 4 chế độ:
  - setup             lần đầu dùng: đặt mật khẩu + 2 câu hỏi bảo mật (FR-5)
  - unlock            mở khóa bằng mật khẩu (link "Quên mật khẩu?" FR-6)
  - upgrade_unlock → upgrade_questions
                      người dùng cũ thiếu câu hỏi: mở khóa trước, rồi BẮT
                      BUỘC khai 2 câu hỏi mới vào app
  - recovery          quên mật khẩu: trả lời đúng CẢ 2 câu → đặt lại (FR-5/6)
Khóa MỀM (FR-2): 5 lần sai liên tiếp → chờ 10s giữa các lần thử; bộ đếm
dùng chung mật khẩu + câu trả lời. Hiển thị "còn N lần thử" (FR-3);
eye toggle (FR-4).

Chạy:  .venv\\Scripts\\python.exe scripts\\step_02_gui_test.py

QUAN TRỌNG: test dùng file config TẠM (data/test_config_step02.json) — KHÔNG
đụng vào config.json thật (trỏ CONFIG_PATH sang file tạm, an toàn).
"""
import os
import sys
import time
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

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QLineEdit  # noqa: E402

import app.config as config_module  # noqa: E402
from app.config import Config  # noqa: E402
from app.services.auth import AuthService  # noqa: E402
from app.ui.lock_screen import (  # noqa: E402
    MODE_RECOVERY,
    MODE_SETUP,
    MODE_UNLOCK,
    MODE_UPGRADE_QUESTIONS,
    MODE_UPGRADE_UNLOCK,
    LockScreen,
)
from app.ui.main_window import MainWindow  # noqa: E402
from app.ui.password_dialog import PasswordDialog  # noqa: E402

# File config TẠM — trỏ CONFIG_PATH sang đây (Config.save/load tìm lúc GỌI)
TEMP_CONFIG = PROJECT_ROOT / "data" / "test_config_step02.json"
TEMP_CONFIG.unlink(missing_ok=True)
config_module.CONFIG_PATH = TEMP_CONFIG

try:
    app = QApplication(sys.argv)
    auth = AuthService(Config(password_hash=None))  # giả lập lần đầu dùng

    # Camera 99 KHÔNG tồn tại → CameraView không mở webcam thật trong test
    # (tránh 2 luồng cv2 mở cùng webcam khi khóa/mở khóa nhanh → hỏng heap)
    window = MainWindow(
        auth, Config(camera_index=99, camera_width=320, camera_height=240)
    )
    window.show()

    # 1) Khởi động phải ở màn hình khóa (chế độ đặt mật khẩu lần đầu)
    assert window._root.currentWidget() is window._lock_screen, "Phải khởi động ở màn hình khóa"
    assert window._lock_screen._mode == MODE_SETUP, "Lần đầu dùng phải ở chế độ đặt mật khẩu"
    assert window._lock_screen._pw2.isVisible(), "Ô nhập lại mật khẩu phải hiện ở chế độ setup"
    assert window._lock_screen._q1_edit.isVisible(), "Câu hỏi bảo mật phải hiện ở chế độ setup"
    print("[1] Khởi động ở màn hình khóa (chế độ đặt mật khẩu + câu hỏi): OK")

    lock = window._lock_screen

    # 1b) Eye toggle (FR-4): bấm 👁 → mật khẩu hiện rõ, bấm lại → ẩn
    lock._eye_btn.click()
    assert lock._pw1.echoMode() == QLineEdit.EchoMode.Normal, "Eye toggle phải hiện mật khẩu"
    lock._eye_btn.click()
    assert lock._pw1.echoMode() == QLineEdit.EchoMode.Password, "Eye toggle phải ẩn mật khẩu lại"
    print("[1b] Eye toggle hiện/ẩn mật khẩu: OK")

    # 2) Mật khẩu yếu (không đủ chính sách FR-1) → báo lỗi, không lưu
    lock._pw1.setText("12345678")   # toàn số, không có chữ thường/hoa
    lock._pw2.setText("12345678")
    lock._on_submit()
    assert not auth.has_password, "Mật khẩu yếu không được lưu"
    assert lock._error_label.text(), "Phải hiện lỗi chính sách mật khẩu"
    print("[2] Mật khẩu không đạt chính sách (8 ký tự + thường/hoa/số) → báo lỗi: OK")

    # 3) Hai lần nhập không khớp → báo lỗi, không đặt được
    lock._pw1.setText("Mat-khau-123")
    lock._pw2.setText("Mat-khau-456")
    lock._on_submit()
    assert not auth.has_password, "Mật khẩu không khớp thì không được lưu"
    assert lock._error_label.text(), "Phải hiện thông báo lỗi không khớp"
    print("[3] Nhập 2 lần không khớp → báo lỗi: OK")

    # 4) Đặt mật khẩu + 2 câu hỏi bảo mật thành công → mở khóa vào nội dung
    lock._pw1.setText("Mat-khau-123")
    lock._pw2.setText("Mat-khau-123")
    lock._q1_edit.setText("Quê quán của bạn?")
    lock._a1_edit.setText("Hà Nội")
    lock._q2_edit.setText("Tên thú cưng đầu tiên?")
    lock._a2_edit.setText("Miu")
    lock._on_submit()
    assert auth.has_password, "Phải lưu được mật khẩu"
    assert auth.has_security_questions, "Phải lưu được 2 câu hỏi bảo mật"
    assert window._root.currentWidget() is window._content, "Phải chuyển vào nội dung"
    # Mật khẩu + câu hỏi phải nằm trong FILE TẠM, không phải config.json thật
    assert TEMP_CONFIG.exists(), "Mật khẩu phải được ghi vào file config tạm"
    print("[4] Đặt mật khẩu + câu hỏi bảo mật lần đầu + vào nội dung: OK")

    # 5) Khóa lại → về màn hình khóa (giờ là chế độ mở khóa)
    window._lock_now()
    assert window._root.currentWidget() is window._lock_screen
    assert window._lock_screen._mode == MODE_UNLOCK, "Giờ phải là chế độ mở khóa"
    assert not window._lock_screen._pw2.isVisible(), "Ô nhập lại phải ẩn ở chế độ mở khóa"
    assert not window._lock_screen._q1_edit.isVisible(), "Câu hỏi bảo mật phải ẩn khi mở khóa"
    assert window._lock_screen._forgot_btn.isVisible(), "Link 'Quên mật khẩu?' phải hiện"
    print("[5] Khóa bằng nút Khoá (chuyển sang chế độ mở khóa): OK")

    # 6) Mật khẩu sai → vẫn ở màn hình khóa + thông báo lỗi (còn 4 lần thử)
    lock._pw1.setText("sai")
    lock._on_submit()
    assert window._root.currentWidget() is window._lock_screen
    assert "còn 4 lần thử" in lock._error_label.text(), (
        f"Phải hiện 'còn 4 lần thử' (thực tế: {lock._error_label.text()!r})"
    )
    print("[6] Mật khẩu sai → báo lỗi + đếm lần thử còn lại: OK")

    # 7) Mật khẩu đúng → mở khóa
    lock._pw1.setText("Mat-khau-123")
    lock._on_submit()
    assert window._root.currentWidget() is window._content
    print("[7] Mở khóa với mật khẩu đúng: OK")

    # 8) PasswordDialog: mật khẩu đúng → Accepted
    dialog_ok = PasswordDialog(auth, "Xóa người dùng 'A'")
    dialog_ok._pw.setText("Mat-khau-123")
    dialog_ok._on_confirm()
    assert dialog_ok.result() == QDialog.DialogCode.Accepted
    print("[8] PasswordDialog mật khẩu đúng → Accepted: OK")

    # 9) PasswordDialog: mật khẩu sai → không Accepted + báo lỗi (1 lần sai)
    dialog_bad = PasswordDialog(auth, "Xóa lịch sử")
    dialog_bad._pw.setText("sai")
    dialog_bad._on_confirm()
    assert dialog_bad.result() != QDialog.DialogCode.Accepted
    assert dialog_bad._error.isVisible() or dialog_bad._error.text()
    print("[9] PasswordDialog mật khẩu sai → từ chối: OK")

    # 10) Khóa MỀM (FR-2): đủ 5 lần sai liên tiếp → phải chờ giữa 2 lần thử.
    #     (mở khóa đúng ở [7] đã reset bộ đếm; [9] sai 1 lần; thêm 4 = đủ 5)
    for _ in range(4):
        auth.verify("sai")
    assert auth.attempts_remaining == 0, "Hết lần thử sau 5 lần sai"
    assert auth.is_locked, "Phải kích hoạt khóa mềm (chờ 10s) sau 5 lần sai"
    window._lock_now()  # khóa lại trong lúc đang phải chờ
    assert not window._lock_screen._pw1.isEnabled(), "Input phải VẪN bị khóa khi đang chờ"
    assert window._lock_screen._lockout_label.text(), "Phải hiện đếm ngược thời gian chờ"
    print("[10] Khóa mềm 5 lần sai + khóa lại → input vẫn khóa, hiện đếm ngược: OK")

    # 10b) Hết thời gian chờ (giả lập trôi 11s) → input mở lại
    auth._last_fail_at = time.time() - 11  # 11s > SOFT_LOCK_WAIT_SECONDS=10
    window._lock_screen._update_lockout()
    assert not auth.is_locked, "Hết 10s chờ thì hết trạng thái khóa mềm"
    assert window._lock_screen._pw1.isEnabled(), "Input phải mở lại sau khi hết chờ"
    assert not window._lock_screen._lockout_label.text(), "Đếm ngược phải biến mất"
    print("[10b] Hết thời gian chờ → nhập lại được: OK")

    # 10c) Mở khóa bằng mật khẩu đúng → reset bộ đếm (để test khôi phục sạch)
    window._lock_screen._pw1.setText("Mat-khau-123")
    window._lock_screen._on_submit()
    assert window._root.currentWidget() is window._content
    assert auth.attempts_remaining == 5, "Mở khóa đúng phải reset bộ đếm"
    print("[10c] Mở khóa đúng → reset bộ đếm lần thử: OK")

    # 11) Quên mật khẩu → khôi phục bằng 2 câu hỏi (FR-5/6)
    window._lock_now()
    assert window._lock_screen._mode == MODE_UNLOCK
    window._lock_screen._on_forgot()
    assert window._lock_screen._mode == MODE_RECOVERY, "Phải vào chế độ khôi phục"
    assert window._lock_screen._q1_edit.text() == "Quê quán của bạn?"
    assert window._lock_screen._q1_edit.isReadOnly(), "Câu hỏi hiển thị nhưng không sửa được"
    # 11a) Sai câu trả lời → báo lỗi, KHÔNG cho đặt mật khẩu (dùng chung bộ đếm)
    window._lock_screen._a1_edit.setText("Sai rồi")
    window._lock_screen._a2_edit.setText("Sai luôn")
    window._lock_screen._on_submit()
    assert "còn 4 lần thử" in window._lock_screen._error_label.text(), (
        "Sai câu trả lời phải tính vào bộ đếm chung (thực tế: "
        f"{window._lock_screen._error_label.text()!r})"
    )
    print("[11] Chế độ khôi phục: hiện câu hỏi + sai câu trả lời → từ chối: OK")

    # 11b) Trả lời ĐÚNG CẢ 2 câu → đặt mật khẩu mới (chính sách mới)
    window._lock_screen._a1_edit.setText("Hà Nội")
    window._lock_screen._a2_edit.setText("Miu")
    window._lock_screen._pw1.setText("Mat-khau-moi-456")
    window._lock_screen._pw2.setText("Mat-khau-moi-456")
    window._lock_screen._on_submit()
    assert window._root.currentWidget() is window._content, "Phải mở khóa sau khi khôi phục"
    assert auth.verify("Mat-khau-moi-456"), "Mật khẩu mới phải hoạt động"
    assert not auth.verify("Mat-khau-123"), "Mật khẩu cũ không còn đúng"
    print("[11b] Trả lời đúng 2 câu hỏi → đặt lại mật khẩu + vào app: OK")

    # 12) Người dùng CŨ (có mật khẩu, chưa có câu hỏi) → bắt khai câu hỏi
    #     trước khi vào app (upgrade_unlock → upgrade_questions)
    from argon2 import PasswordHasher

    auth2 = AuthService(Config(password_hash=PasswordHasher().hash("mat-khau-cu")))
    assert not auth2.has_security_questions, "Giả lập người dùng cũ chưa có câu hỏi"
    lock2 = LockScreen(auth2)
    assert lock2._mode == MODE_UPGRADE_UNLOCK, "Phải vào chế độ mở khóa nâng cấp"
    unlocked2: list[bool] = []
    lock2.unlocked.connect(lambda: unlocked2.append(True))

    lock2._pw1.setText("mat-khau-cu")
    lock2._on_submit()
    assert lock2._mode == MODE_UPGRADE_QUESTIONS, "Sau mở khóa phải bắt khai câu hỏi"
    assert not unlocked2, "Chưa được vào app khi chưa khai xong câu hỏi"
    assert not lock2._pw1.isVisible(), "Ô mật khẩu phải ẩn khi đang khai câu hỏi"

    lock2._q1_edit.setText("Quê quán của bạn?")
    lock2._a1_edit.setText("Hà Nội")
    lock2._q2_edit.setText("Tên thú cưng đầu tiên?")
    lock2._a2_edit.setText("Miu")
    lock2._on_submit()
    assert unlocked2, "Phải vào app sau khi khai đủ 2 câu hỏi"
    assert auth2.has_security_questions, "Câu hỏi phải được lưu"
    print("[12] Người dùng cũ → mở khóa → bắt khai 2 câu hỏi → vào app: OK")

    print("\n=== TẤT CẢ TEST GUI BƯỚC 2 (LOCKSCREEN MỚI) ĐỀU QUA ===")
    # Dừng camera + đóng cửa sổ TRƯỚC khi thoát — tránh 'QThread: Destroyed
    # while thread is still running' (closeEvent của MainWindow dừng camera).
    window._camera_view.stop_camera()
    window.close()
    QTimer.singleShot(100, app.quit)
    sys.exit(app.exec())
finally:
    # Dọn file tạm — không cần khôi phục gì vì chưa bao giờ đụng config.json thật
    TEMP_CONFIG.unlink(missing_ok=True)