"""Cấu hình ứng dụng — đọc/ghi file config.json.

Các cài đặt như camera index, ngưỡng nhận diện, hash mật khẩu được lưu
trong config.json (KHÔNG nằm trong database) — dễ chỉnh tay, dễ backup,
và giúp tách "cấu hình" khỏi "dữ liệu người dùng" (Bước 7).

ĐƯỜNG DẪN (Bước 12 — đóng gói .exe): app phân biệt 2 chế độ chạy:
  - Từ MÃ NGUỒN (python -m app.main / test): mọi thứ nằm trong project.
  - Từ file .EXE (PyInstaller):
      * APP_DIR = thư mục chứa .exe → dữ liệu NGƯỜI DÙNG (config.json, data/,
        logs/) đặt ở đó — dễ mang đi, dễ backup, không bị mất khi rebuild.
      * RESOURCE_DIR = nơi PyInstaller giải nén bundle (onedir: cạnh exe;
        onefile: thư mục tạm _MEIPASS) → model ONNX (chỉ đọc) đặt ở đó.
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


def _is_frozen() -> bool:
    """Đang chạy từ file .exe đóng gói (PyInstaller) hay từ mã nguồn?"""
    return bool(getattr(sys, "frozen", False))


if _is_frozen():
    # --- Chạy từ .exe đã đóng gói (Bước 12) ---
    APP_DIR = Path(sys.executable).resolve().parent
    # onefile giải nén bundle vào _MEIPASS; onedir thì tài nguyên nằm cạnh exe
    RESOURCE_DIR = Path(getattr(sys, "_MEIPASS", APP_DIR))
    PROJECT_ROOT = APP_DIR  # giữ tên cũ cho tương thích với code hiện có
else:
    # --- Chạy từ mã nguồn: thư mục gốc project (app/config.py -> project/) ---
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    APP_DIR = PROJECT_ROOT
    RESOURCE_DIR = PROJECT_ROOT

# Cấu hình người dùng — đặt ở APP_DIR (cạnh .exe khi đã đóng gói)
CONFIG_PATH = APP_DIR / "config.json"


@dataclass
class Config:
    """Các giá trị cấu hình của ứng dụng (kèm giá trị mặc định)."""

    camera_index: int = 0            # chỉ số webcam (0 = camera mặc định)
    camera_width: int = 640          # độ phân giải ngang mong muốn nhẹ hơn
    camera_height: int = 480         # độ phân giải dọc mong muốn nhẹ hơn
    recognition_threshold: float = 0.40  # ngưỡng tương đồng tối thiểu (0..1)
    # Chống giả mạo (yêu cầu chớp mắt định kỳ, Bước 17) — mặc định TẮT:
    # người dùng thật không muốn bị làm phiền khi nhìn thẳng không chớp
    # (ai cần bảo mật thì tự bật trong Cài đặt → NHẬN DIỆN).
    anti_spoofing_enabled: bool = False
    smoothing_window: int = 3          # số khung temporal smoothing nhẹ hơn, giúp giảm lag
    clahe_enabled: bool = False        # tắt CLAHE mặc định để tránh nặng frame-by-frame
    password_hash: str | None = None  # hash mật khẩu (argon2) — None = chưa đặt
    # Câu hỏi bảo mật (2 câu, để khôi phục khi quên mật khẩu — spec quản lý
    # mật khẩu): câu hỏi lưu dạng chữ (hiển thị), câu trả lời CHỈ lưu hash
    # argon2 (không plaintext). Rỗng = chưa thiết lập.
    security_question_1: str = ""
    security_answer_hash_1: str | None = None
    security_question_2: str = ""
    security_answer_hash_2: str | None = None
    # Tự khóa app khi không dùng (phút) — 0 = tắt (mặc định); 1/5/15 = bật
    idle_lock_minutes: int = 0
    sync_enabled: bool = False       # bật/tắt đồng bộ cloud (Bước 15)
    # Thông tin Cloudflare D1 (Bước 15) — rỗng = chưa cấu hình
    cloud_account_id: str = ""       # Account ID (dash.cloudflare.com → trang tổng quan)
    cloud_database_id: str = ""      # D1 database ID (trang D1 → tên database)
    cloud_api_token: str = ""        # API token (quyền Account · D1 · Edit)
    # Thông tin Supabase Storage (Bước 16) — lưu ảnh snapshot/thumbnail
    # (free tier KHÔNG cần thẻ tín dụng; S3-compatible qua boto3)
    sb_endpoint: str = ""            # S3 endpoint: https://<ref>.supabase.co/storage/v1/s3
    sb_region: str = ""              # region của project (trang S3 Access Keys)
    sb_bucket: str = ""              # tên bucket (Storage → New bucket)
    sb_access_key_id: str = ""       # Access Key ID (Storage → S3 Access Keys)
    sb_secret_access_key: str = ""   # Secret Access Key (chỉ hiện 1 lần khi tạo)
    theme: str = "dark"              # chủ đề giao diện: dark / light (Bước 13)

    def __post_init__(self) -> None:
        """Chặn giá trị vô lý: ngưỡng tương đồng luôn nằm trong 0..1."""
        if not 0.0 <= self.recognition_threshold <= 1.0:
            logger.warning(
                "Ngưỡng %s ngoài khoảng 0..1 — dùng mặc định 0.40",
                self.recognition_threshold,
            )
            self.recognition_threshold = 0.40

    # -------------------------------------------------------------
    # Đọc / ghi config
    # -------------------------------------------------------------

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        """Đọc config từ file; nếu chưa có thì tạo file với giá trị mặc định.

        ``path=None`` → dùng CONFIG_PATH (đường dẫn mặc định). Tìm tại
        thời điểm GỌI (không cố định ở lúc định nghĩa hàm) để test có thể
        trỏ sang file tạm bằng cách gán lại ``app.config.CONFIG_PATH``.

        Chỉ nhận các key hợp lệ trong dataclass — bỏ qua key lạ để
        an toàn khi file bị chỉnh sửa tay (hoặc cấu hình từ bản cũ).
        """
        if path is None:
            path = CONFIG_PATH
        if not path.exists():
            config = cls()
            config.save(path)
            return config

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            # Chỉ lấy key hợp lệ, bỏ qua key lạ và giá trị null
            valid = {
                k: v
                for k, v in data.items()
                if k in cls.__dataclass_fields__ and v is not None
            }
            return cls(**valid)
        except (json.JSONDecodeError, TypeError, ValueError, OSError) as exc:
            # File hỏng (JSON lỗi / kiểu sai) → dùng mặc định, không làm sập app
            logger.warning("Config không đọc được (%s) — dùng giá trị mặc định", exc)
            return cls()

    def save(self, path: Path | None = None) -> None:
        """Ghi config ra file JSON (utf-8, có thụt lề để dễ đọc).

        ``path=None`` → dùng CONFIG_PATH (xem ghi chú của ``load``).
        """
        if path is None:
            path = CONFIG_PATH
        path.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
