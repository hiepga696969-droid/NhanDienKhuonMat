"""Bước 14 — Kiểm tra SettingsView (trang Cài đặt, wireframe 5.5.7).

1. Widget <-> config: đổ giá trị config vào ô, đổi ô → [Lưu thay đổi] ghi
   đúng vào config.json (file TẠM — không đụng config thật của người dùng).
2. Camera index: phân tích chuỗi 'CAM 3' / '5' / 'abc'.
3. Thanh trượt ngưỡng: đổi giá trị → nhãn hiển thị đúng 0.xx.
4. [Khôi phục mặc định]: các ô về giá trị khởi tạo (chưa lưu).
5. Đổi mật khẩu: xác thực cũ (PasswordDialog) → nhập mới 2 lần → hash mới
   hoạt động, mật khẩu cũ không còn đúng.
6. Đồng bộ cloud: bật/tắt cần mật khẩu (FR-11); hủy xác thực → trả lại ô.
7. [Kết nối thử]: camera 99 (không tồn tại) → báo lỗi mềm, không crash.

Chạy:  .venv\\Scripts\\python.exe scripts\\step_14_gui_test.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import PySide6  # noqa: E402

# Khắc phục đường dẫn plugins + chế độ ảo (giống step_03/08)
pyside_dir = os.path.dirname(PySide6.__file__)
os.environ["QT_PLUGIN_PATH"] = os.path.join(pyside_dir, "plugins")
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(pyside_dir)
os.environ["QT_QPA_PLATFORM"] = "minimal"

from argon2 import PasswordHasher  # noqa: E402

import app.config as cfg_mod  # noqa: E402
import app.ui.settings_view as sv  # noqa: E402
from app.config import Config  # noqa: E402
from app.infrastructure.db import Database  # noqa: E402
from app.services.auth import AuthService  # noqa: E402
from app.services.sync import SyncService  # noqa: E402

# CHỈ ĐỊNH config.json TẠM — mọi thao tác Lưu/Đổi mật khẩu ghi vào đây,
# tuyệt đối không đụng config.json thật của người dùng (bài học Bước mật khẩu)
TEMP_CONFIG = PROJECT_ROOT / "data" / "test_config_14.json"
TEMP_DB = PROJECT_ROOT / "data" / "test_db_14.db"


def make_sync(config: Config) -> SyncService:
    """SyncService gắn DB TẠM (SettingsView giờ cần tham số sync — Bước 15)."""
    return SyncService(Database(TEMP_DB), config)

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [QUA] {name} {detail}")
    else:
        FAIL += 1
        print(f"  [THẤT] {name} {detail}")


def cleanup() -> None:
    TEMP_CONFIG.unlink(missing_ok=True)
    for suffix in ("", "-wal", "-shm"):  # SQLite WAL tạo thêm 2 file kèm
        Path(str(TEMP_DB) + suffix).unlink(missing_ok=True)


def test_config_widgets() -> None:
    """Widget <-> config: đổ giá trị, đổi, Lưu → file tạm đúng."""
    print("\n[1] Widget <-> config — lưu/đọc qua config.json tạm")
    cfg_mod.CONFIG_PATH = TEMP_CONFIG
    config = Config()
    auth = AuthService(config)
    auth.config.password_hash = PasswordHasher().hash("mat-khau-123")
    view = sv.SettingsView(config, auth, make_sync(config))

    # Khởi tạo: đổ từ config (mặc định)
    check("combo camera khởi tạo CAM 0",
          view._camera_combo.currentText() == "CAM 0",
          f"(text={view._camera_combo.currentText()!r})")
    check("combo độ phân giải khởi tạo 1280×720",
          view._res_combo.currentData() == (1280, 720))
    check("slider ngưỡng khởi tạo 0.40",
          view._threshold_slider.value() == 40
          and view._threshold_label.text() == "0.40",
          f"(slider={view._threshold_slider.value()}, label={view._threshold_label.text()!r})")
    check("checkbox cloud khởi tạo tắt", not view._sync_check.isChecked())
    check("trạng thái mật khẩu: đã đặt",
          "Đã đặt" in view._pw_status.text(),
          f"(text={view._pw_status.text()!r})")

    # Đổi các ô
    view._camera_combo.setCurrentText("CAM 2")
    for i in range(view._res_combo.count()):
        if view._res_combo.itemData(i) == (640, 480):
            view._res_combo.setCurrentIndex(i)
            break
    view._threshold_slider.setValue(55)
    check("nhãn ngưỡng cập nhật 0.55",
          view._threshold_label.text() == "0.55",
          f"(label={view._threshold_label.text()!r})")

    # Lưu → config.json tạm (patch QMessageBox modal để không chặn test)
    orig_info = sv.QMessageBox.information
    sv.QMessageBox.information = staticmethod(lambda *a, **k: None)
    try:
        view._on_save()
    finally:
        sv.QMessageBox.information = orig_info

    check("config.camera_index = 2", config.camera_index == 2,
          f"(={config.camera_index})")
    check("config độ phân giải 640×480",
          config.camera_width == 640 and config.camera_height == 480)
    check("config ngưỡng 0.55", config.recognition_threshold == 0.55,
          f"(={config.recognition_threshold})")

    # Đọc lại từ file tạm — dữ liệu đã GHI thật xuống đĩa
    loaded = Config.load()
    check("đọc lại từ file: camera 2", loaded.camera_index == 2)
    check("đọc lại từ file: 640×480",
          loaded.camera_width == 640 and loaded.camera_height == 480)
    check("đọc lại từ file: ngưỡng 0.55",
          abs(loaded.recognition_threshold - 0.55) < 1e-9)

    cleanup()


def test_camera_index_parse() -> None:
    """Phân tích chuỗi camera index từ combo (editable)."""
    print("\n[2] Camera index — phân tích chuỗi")
    check("'CAM 3' → 3", sv.SettingsView._camera_index_from_text("CAM 3") == 3)
    check("'5' → 5", sv.SettingsView._camera_index_from_text("5") == 5)
    check("'abc' → 0", sv.SettingsView._camera_index_from_text("abc") == 0)
    check("'' → 0", sv.SettingsView._camera_index_from_text("") == 0)
    check("'CAM 12' → 12", sv.SettingsView._camera_index_from_text("CAM 12") == 12)


def test_reset_defaults() -> None:
    """[Khôi phục mặc định]: các ô về giá trị khởi tạo, chưa lưu."""
    print("\n[3] [Khôi phục mặc định] — các ô về mặc định")
    cfg_mod.CONFIG_PATH = TEMP_CONFIG
    config = Config()
    auth = AuthService(config)
    view = sv.SettingsView(config, auth, make_sync(config))

    # Đổi lung tung rồi reset
    view._camera_combo.setCurrentText("CAM 4")
    view._threshold_slider.setValue(70)
    view._sync_check.blockSignals(True)
    view._sync_check.setChecked(True)
    view._sync_check.blockSignals(False)

    view._on_reset_defaults()

    check("combo camera về CAM 0",
          view._camera_combo.currentText() == "CAM 0",
          f"(text={view._camera_combo.currentText()!r})")
    check("slider ngưỡng về 0.40", view._threshold_slider.value() == 40)
    check("checkbox cloud về tắt", not view._sync_check.isChecked())
    # Chưa lưu → config object không đổi
    check("config chưa bị ghi (chưa bấm Lưu)",
          config.camera_index == 0 and config.recognition_threshold == 0.40)

    cleanup()


def test_change_password() -> None:
    """Đổi mật khẩu: xác thực cũ → nhập mới 2 lần → hash mới (FR-11)."""
    print("\n[4] Đổi mật khẩu — qua PasswordDialog + ChangePasswordDialog")
    cfg_mod.CONFIG_PATH = TEMP_CONFIG
    config = Config()
    auth = AuthService(config)
    auth.config.password_hash = PasswordHasher().hash("mat-khau-123")
    view = sv.SettingsView(config, auth, make_sync(config))

    # Giả lập: PasswordDialog.require trả True (đã xác thực mật khẩu cũ)
    orig_require = sv.PasswordDialog.require
    sv.PasswordDialog.require = staticmethod(lambda *a, **k: True)

    # Giả lập ChangePasswordDialog: nhập mới 2 lần → Accepted
    class FakeChangeDialog:
        def __init__(self, *a, **k):
            # Chính sách FR-1 mới: ≥8 ký tự + chữ thường + chữ hoa + chữ số
            self._new_pw = "Mat-khau-moi-456"
        def exec(self) -> int:
            return 1  # QDialog.DialogCode.Accepted
        def new_password(self) -> str:
            return self._new_pw

    orig_dialog = sv.ChangePasswordDialog
    sv.ChangePasswordDialog = FakeChangeDialog
    orig_info = sv.QMessageBox.information
    sv.QMessageBox.information = staticmethod(lambda *a, **k: None)

    try:
        view._on_change_password()
    finally:
        sv.PasswordDialog.require = orig_require
        sv.ChangePasswordDialog = orig_dialog
        sv.QMessageBox.information = orig_info

    check("mật khẩu mới hoạt động", auth.verify("Mat-khau-moi-456"))
    check("mật khẩu cũ không còn đúng", not auth.verify("mat-khau-123"))
    check("hash đã lưu vào file tạm",
          Config.load().password_hash is not None)
    # Hash mới KHÔNG chứa mật khẩu dạng thô (argon2)
    check("hash argon2 không chứa mật khẩu thô",
          "Mat-khau-moi-456" not in (Config.load().password_hash or ""))

    cleanup()


def test_sync_toggle_password() -> None:
    """Bật/tắt cloud cần mật khẩu; hủy xác thực → trả lại ô (FR-11)."""
    print("\n[5] Đồng bộ cloud — xác thực mật khẩu khi bật/tắt")
    cfg_mod.CONFIG_PATH = TEMP_CONFIG
    config = Config()
    auth = AuthService(config)
    auth.config.password_hash = PasswordHasher().hash("mat-khau-123")
    view = sv.SettingsView(config, auth, make_sync(config))

    orig_require = sv.PasswordDialog.require

    # 1) Cố bật khi HỦY xác thực (ô đang tắt mặc định) → ô phải trả về tắt
    sv.PasswordDialog.require = staticmethod(lambda *a, **k: False)
    view._sync_check.setChecked(True)  # cố bật nhưng bị từ chối
    check("hủy xác thực → ô trả về tắt", not view._sync_check.isChecked())

    # 2) Bật cloud + xác thực thành công → ô giữ trạng thái bật
    sv.PasswordDialog.require = staticmethod(lambda *a, **k: True)
    view._sync_check.setChecked(True)
    check("bật cloud (có mật khẩu) → ô bật", view._sync_check.isChecked())

    # 3) Cố tắt khi HỦY xác thực (ô đang bật) → ô phải giữ bật
    sv.PasswordDialog.require = staticmethod(lambda *a, **k: False)
    view._sync_check.setChecked(False)  # cố tắt nhưng bị từ chối
    check("hủy xác thực khi tắt → ô giữ bật", view._sync_check.isChecked())

    sv.PasswordDialog.require = orig_require
    cleanup()


def test_camera_probe(app) -> None:
    """[Kết nối thử] với camera 99 (không tồn tại) → báo lỗi mềm, không crash."""
    print("\n[6] [Kết nối thử] — camera 99 (không tồn tại) → lỗi mềm")

    cfg_mod.CONFIG_PATH = TEMP_CONFIG
    config = Config()
    auth = AuthService(config)
    view = sv.SettingsView(config, auth, make_sync(config))
    view.show()

    view._camera_combo.setCurrentText("CAM 99")
    view._on_test_camera()

    # Chờ worker kết thúc (open camera 99 thất bại nhanh) — xử lý event queue
    deadline = time.time() + 15
    while view._probe_thread is not None and view._probe_thread.isRunning():
        app.processEvents()
        if time.time() > deadline:
            break
        time.sleep(0.05)
    for _ in range(30):  # dọn nốt event còn treo
        app.processEvents()

    check("thread probe đã dọn sạch",
          view._probe_thread is None and view._probe_worker is None)
    check("nhãn trạng thái báo lỗi mềm",
          "Không mở được" in view._camera_status.text(),
          f"(text={view._camera_status.text()!r})")
    check("nút Kết nối thử bật lại", view._test_btn.isEnabled())

    view.close()
    cleanup()


def main() -> None:
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    print("=== TEST BƯỚC 14: TRANG CÀI ĐẶT (SETTINGSVIEW) ===")
    cleanup()
    test_config_widgets()
    test_camera_index_parse()
    test_reset_defaults()
    test_change_password()
    test_sync_toggle_password()
    test_camera_probe(app)
    cleanup()

    print(f"\nKẾT QUẢ: {PASS} qua / {FAIL} thất bại")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
