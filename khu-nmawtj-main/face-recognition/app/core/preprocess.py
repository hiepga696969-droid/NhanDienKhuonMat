"""Tiền xử lý ảnh — CLAHE trên luminance channel (Bước 21).

Vấn đề: Webcam gặp ánh sáng thay đổi (mở/tắt đèn, ánh sáng tự nhiên)
→ cùng 1 mặt nhưng pixel khác nhau → embedding dao động → cosine không ổn định.

Giải pháp: CLAHE (Contrast Limited Adaptive Histogram Equalization) trên
kênh L (luminance) trong không gian LAB:
- Chỉ thay đổi độ sáng/tương phản, KHÔNG thay đổi màu sắc (kênh a, b giữ nguyên)
- Áp dụng cục bộ từng ô 8×8 → không bị overshot ở vùng sáng/tối
- clipLimit=3.0 → chống oversharpening

QUAN TRỌNG: CLAHE phải áp dụng CẢ KHI ĐĂNG KÝ VÀ KHI NHẬN DIỆN.
Nếu chỉ áp dụng 1 bên → embedding đăng ký và nhận diện sẽ khác → lỗi.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Singleton CLAHE instance (tái sử dụng — tránh tạo lại mỗi frame)
_clahe: cv2.CLAHE | None = None


def _get_clahe(clip_limit: float = 3.0, grid_size: int = 8) -> cv2.CLAHE:
    """Lazy-init CLAHE singleton."""
    global _clahe
    if _clahe is None:
        _clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(grid_size, grid_size),
        )
    return _clahe


def preprocess_frame(
    frame: np.ndarray,
    clip_limit: float = 3.0,
    grid_size: int = 8,
) -> np.ndarray:
    """Chuẩn hóa ánh sáng bằng CLAHE trên kênh L (LAB color space).

    Args:
        frame: ảnh BGR (uint8, shape H×W×3)
        clip_limit: giới hạn tương phản CLAHE (mặc định 3.0 — cân bằng)
        grid_size: kích thước ô cục bộ (8 = 8×8 ô)

    Returns:
        ảnh BGR đã chuẩn hóa ánh sáng (cùng kích thước)
    """
    if frame is None or frame.size == 0:
        return frame

    try:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = _get_clahe(clip_limit, grid_size)
        l_eq = clahe.apply(l)

        lab_eq = cv2.merge([l_eq, a, b])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    except Exception:  # noqa: BLE001
        # Fallback: trả frame gốc nếu có lỗi (không crash app)
        logger.exception("Lỗi CLAHE — giữ frame gốc")
        return frame
