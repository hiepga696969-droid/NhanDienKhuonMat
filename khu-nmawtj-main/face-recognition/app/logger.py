"""Cấu hình logging — ghi log ra console + file logs/app.log.

Lưu ý: console Windows mặc định dùng cp1252 (không in được tiếng Việt),
nên ta ép UTF-8 cho stdout/stderr trước khi cài handler.
"""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from app.config import APP_DIR

# Log đặt ở APP_DIR (cạnh .exe khi đã đóng gói — Bước 12), không nằm trong
# thư mục tạm _MEIPASS (nếu không, log sẽ biến mất mỗi lần chạy .exe).
LOG_DIR = APP_DIR / "logs"
LOG_FILE = LOG_DIR / "app.log"


def setup_logging(level: int = logging.INFO) -> None:
    """Khởi tạo logging: file xoay vòng (utf-8) + console (ép utf-8)."""
    # Ép console UTF-8 — tránh UnicodeEncodeError khi in tiếng Việt
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.stderr and hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    LOG_DIR.mkdir(exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler: xoay vòng khi đạt 1MB, giữ lại 3 file cũ
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()  # tránh trùng handler nếu setup_logging được gọi lại
    root.addHandler(file_handler)

    # Console handler: chỉ thêm khi có stdout (file .exe chạy --windowed sẽ
    # KHÔNG có console → sys.stdout = None, thêm handler sẽ gây lỗi lúc emit)
    if sys.stdout is not None and hasattr(sys.stdout, "write"):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)
