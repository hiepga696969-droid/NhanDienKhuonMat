"""Bước 21 — CLAHE preprocessing (chuẩn hóa ánh sáng trước khi embed).

1. Unit: preprocess_frame áp dụng CLAHE → ảnh đầu ra khác ảnh gốc,
   cùng shape + dtype. Ảnh rỗng/null → trả nguyên.
2. UNIT: ảnh tối + CLAHE → luminance tăng rõ rệt.
3. CLAHE giữ màu sắc (kênh a, b gần giữ nguyên).
4. Config: clahe_enabled lưu/đọc đúng.
5. SettingsView: checkbox CLAHE có + load/save/reset.

Chạy:  .venv/Scripts/python.exe scripts/step_21_gui_test.py
"""
from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["QT_QPA_PLATFORM"] = "minimal"

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.config import Config  # noqa: E402
from app.core.preprocess import preprocess_frame  # noqa: E402

TEMP_CONFIG = PROJECT_ROOT / "data" / "test_config_21.json"

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


def load_lena() -> np.ndarray:
    lena_path = PROJECT_ROOT / "lena_test.jpg"
    if not lena_path.exists() or lena_path.stat().st_size == 0:
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
            lena_path,
        )
    return cv2.imread(str(lena_path))


# ---------------------------------------------------------
# 1. preprocess_frame — unit
# ---------------------------------------------------------
def test_preprocess_unit() -> None:
    print("\n[1] preprocess_frame — đơn vị")
    lena = load_lena()

    result = preprocess_frame(lena)
    check("CLAHE giữ shape",
          result.shape == lena.shape,
          f"(in={lena.shape}, out={result.shape})")
    check("CLAHE giữ dtype uint8",
          result.dtype == np.uint8)
    check("CLAHE thay đổi pixel (ảnh khác ảnh gốc)",
          not np.array_equal(result, lena))

    check("ảnh rỗng → trả nguyên",
          preprocess_frame(np.array([], dtype=np.uint8)).size == 0)
    check("None → trả None",
          preprocess_frame(None) is None)


# ---------------------------------------------------------
# 2. Ảnh tối/sáng → CLAHE giúp tăng tương phản
# ---------------------------------------------------------
def test_clahe_brightness() -> None:
    print("\n[2] CLAHE — ảnh tối/sáng")
    lena = load_lena()

    dark = np.clip(lena.astype(np.int16) - 100, 0, 255).astype(np.uint8)
    dark_clahe = preprocess_frame(dark)

    dark_lab = cv2.cvtColor(dark, cv2.COLOR_BGR2LAB)
    dark_clahe_lab = cv2.cvtColor(dark_clahe, cv2.COLOR_BGR2LAB)
    l_orig = dark_lab[:, :, 0].mean()
    l_clahe = dark_clahe_lab[:, :, 0].mean()
    check("ảnh tối + CLAHE → luminance tăng rõ rệt",
          abs(l_clahe - l_orig) > 10.0,
          f"(L_orig={l_orig:.1f}, L_clahe={l_clahe:.1f}, diff={l_clahe-l_orig:+.1f})")

    bright = np.clip(lena.astype(np.int16) + 80, 0, 255).astype(np.uint8)
    bright_clahe = preprocess_frame(bright)
    b_lab = cv2.cvtColor(bright, cv2.COLOR_BGR2LAB)
    bc_lab = cv2.cvtColor(bright_clahe, cv2.COLOR_BGR2LAB)
    lb = b_lab[:, :, 0].mean()
    lbc = bc_lab[:, :, 0].mean()
    check("ảnh sáng + CLAHE → luminance thay đổi",
          abs(lbc - lb) > 1.0,
          f"(L_bright={lb:.1f}, L_clahe={lbc:.1f})")


# ---------------------------------------------------------
# 3. CLAHE giữ màu sắc
# ---------------------------------------------------------
def test_color_preservation() -> None:
    print("\n[3] CLAHE giữ màu sắc")
    lena = load_lena()
    original = preprocess_frame(lena)

    orig_lab = cv2.cvtColor(lena, cv2.COLOR_BGR2LAB)
    out_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
    a_diff = abs(orig_lab[:, :, 1].mean() - out_lab[:, :, 1].mean())
    b_diff = abs(orig_lab[:, :, 2].mean() - out_lab[:, :, 2].mean())
    check("kênh a (màu đỏ/xanh) giữ nguyên",
          a_diff < 2.0, f"(diff={a_diff:.2f})")
    check("kênh b (màu vàng/xanh) giữ nguyên",
          b_diff < 2.0, f"(diff={b_diff:.2f})")


# ---------------------------------------------------------
# 4. Config — clahe_enabled
# ---------------------------------------------------------
def test_config() -> None:
    print("\n[4] Config — clahe_enabled")
    cfg = Config()
    check("clahe_enabled mặc định = True",
          cfg.clahe_enabled is True)
    check("field tồn tại trong dataclass",
          "clahe_enabled" in Config.__dataclass_fields__)

    cfg.clahe_enabled = False
    cfg.save(TEMP_CONFIG)
    loaded = Config.load(TEMP_CONFIG)
    check("save + load round-trip",
          loaded.clahe_enabled is False)


# ---------------------------------------------------------
# 5. SettingsView — checkbox CLAHE (lazy Qt)
# ---------------------------------------------------------
def test_settings_checkbox() -> None:
    print("\n[5] SettingsView — checkbox CLAHE")
    # Lazy import PySide6 — chỉ cần cho test GUI
    import PySide6  # noqa: E402, F401
    pyside_dir = os.path.dirname(PySide6.__file__)
    os.environ["QT_PLUGIN_PATH"] = os.path.join(pyside_dir, "plugins")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(pyside_dir)

    from PySide6.QtWidgets import QApplication  # noqa: E402
    import app.config as cfg_mod  # noqa: E402

    from app.services.auth import AuthService  # noqa: E402
    from app.services.sync import SyncService  # noqa: E402
    from app.infrastructure.db import Database  # noqa: E402

    app = QApplication.instance() or QApplication(sys.argv)
    cfg_mod.CONFIG_PATH = TEMP_CONFIG

    db = Database(PROJECT_ROOT / "data" / "test_21.db")
    config = Config()
    import app.ui.settings_view as sv_mod
    from app.ui.settings_view import SettingsView
    view = SettingsView(config, AuthService(config), SyncService(db, config))

    check("SettingsView có checkbox clahe",
          hasattr(view, "_clahe_check"))
    check("CLAHE mặc định = checked",
          view._clahe_check.isChecked() is True)

    # _on_save kết thúc bằng QMessageBox.information MODAL → treo vô hạn dưới
    # platform "minimal" (không ai bấm được). Patch tạm như step_14.
    orig_info = sv_mod.QMessageBox.information
    sv_mod.QMessageBox.information = staticmethod(lambda *a, **k: None)
    try:
        view._clahe_check.setChecked(False)
        view._on_save()
    finally:
        sv_mod.QMessageBox.information = orig_info
    loaded = Config.load(TEMP_CONFIG)
    check("save + load: clahe_enabled=False",
          loaded.clahe_enabled is False)

    view._on_reset_defaults()
    check("reset defaults → clahe_enabled=True",
          view._clahe_check.isChecked() is True)

    view.close()
    db.close()
    cleanup()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main() -> None:
    print("=== TEST BƯỚC 21: CLAHE PREPROCESSING ===")
    cleanup()
    test_preprocess_unit()
    test_clahe_brightness()
    test_color_preservation()
    test_config()
    test_settings_checkbox()
    cleanup()
    print(f"\nKẾT QUẢ: {PASS} qua / {FAIL} thất bại")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
