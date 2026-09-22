"""Thu thập video từ webcam (OpenCV) — lớp hạ tầng.

Quan trọng: đối tượng VideoCapture của OpenCV KHÔNG an toàn đa luồng —
phải được tạo, đọc và đóng trong CÙNG một luồng. Vì vậy CameraCapture
được khởi tạo bên trong luồng worker (xem ui/camera_view.py).

Vấn đề đã gặp trên Windows (log MSMF "OnReadSample failed"): webcam
laptop báo MSMF "mở thành công" nhưng không bao giờ trả frame → màn
hình đen. Xử lý:
  1. Ưu tiên backend DSHOW (ổn định trên webcam laptop) trước MSMF.
  2. Xác minh backend THẬT SỰ đọc được 1 frame — không tin lời báo mở.
  3. RETRY đọc frame vài lần trước khi chịu thua (không thêm wait).
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Số lần thử lại khi đọc frame lỗi (MSMF thường hồi phục sau 1-2 lần)
READ_RETRIES = 5


class CameraCapture:
    """Bọc cv2.VideoCapture: mở camera, đọc frame BGR, đóng đúng cách."""

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720) -> None:
        self._index = index
        self._width = width
        self._height = height
        self._cap: cv2.VideoCapture | None = None

    # ---------------------------------------------------------
    # Vòng đời
    # ---------------------------------------------------------

    def open(self) -> bool:
        """Mở camera; đặt độ phân giải mong muốn. True nếu mở thành công.

        Ưu tiên DSHOW trước (ổn định trên webcam laptop), rồi MSMF. Quan
        trọng: sau khi mở phải THỬ ĐỌC 1 FRAME — vì trên một số máy MSMF
        báo isOpened()=True nhưng không bao giờ trả frame (OnReadSample
        failed) → màn hình đen nếu chỉ tin lời báo mở.
        """
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
            cap = cv2.VideoCapture(self._index, backend)
            if not cap.isOpened():
                logger.warning(
                    "Không mở được camera index=%d với backend %d",
                    self._index,
                    backend,
                )
                cap.release()
                continue
            # Đặt độ phân giải (camera có thể bỏ qua nếu không hỗ trợ)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            # Buffer 1 frame: giảm TRỄ — không đọc frame cũ tích trong bộ đệm
            # (webcam 30fps + buffer dài → ảnh hiển thị trễ cả nửa giây dù
            # máy khỏe). Không hỗ trợ trên vài backend → giá trị bị bỏ qua,
            # không lỗi. Thứ tự: đặt TRƯỚC khi mở/đọc frame đầu.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # Xác minh backend đọc được frame THẬT (chống bug MSMF "mở ảo").
            # Camera mới mở cần chút thời gian "warm-up" → thử vài lần.
            self._cap = cap  # gắn tạm để _read_frame hoạt động trong warm-up
            ok = False
            for _ in range(READ_RETRIES):
                frame_ok, _ = self._read_frame()
                if frame_ok:
                    ok = True
                    break
            if not ok:
                logger.warning(
                    "Backend %d mở được nhưng không đọc được frame — thử backend khác",
                    backend,
                )
                cap.release()
                self._cap = None
                continue
            self._cap = cap
            logger.info(
                "Đã mở camera %d (backend %d) — kích thước thực: %dx%d",
                self._index,
                backend,
                self.actual_width,
                self.actual_height,
            )
            return True
        return False

    def read(self) -> np.ndarray | None:
        """Đọc frame kế tiếp (BGR). Trả về None khi lỗi kéo dài hoặc camera đóng.

        Lỗi MSMF tạm thời (webcam bận) được xử lý bằng cách thử lại vài
        lần với khoảng nghỉ ngắn — tránh ngắt camera khi chỉ là giật nhịp.

        LƯU Ý: frame được LẬT NGANG (soi gương) trước khi trả về — webcam
        của người dùng hiển thị ảnh bị đảo trái-phải (đã xác nhận). Lật
        ngay tại nguồn để mọi màn hình dùng chung (preview + đăng ký)
        hiển thị đồng nhất, và việc phát hiện "quay trái/phải" khớp với
        những gì người dùng thấy trên màn hình.
        """
        if self._cap is None:
            return None
        for attempt in range(READ_RETRIES):
            ok, frame = self._read_frame()
            if ok:
                return cv2.flip(frame, 1)  # 1 = lật ngang (trái ↔ phải)
        logger.warning("Đọc frame thất bại liên tục (%d lần) — bỏ qua frame", READ_RETRIES)
        return None

    def _read_frame(self) -> tuple[bool, np.ndarray | None]:
        """Đọc 1 frame; trả (True, frame) hoặc (False, None).

        Trên Windows, MSMF đôi khi NÉM ngoại lệ C++ (cv2.error: Unknown
        C++ exception) thay vì trả cờ lỗi — nếu không bắt, ngoại lệ sẽ
        giết luồng worker camera (QThread chết bất thường). Xử lý: coi
        như một lần đọc lỗi, cho thử lại như bình thường.
        """
        if self._cap is None:
            return False, None
        try:
            ok, frame = self._cap.read()
            return bool(ok), frame
        except cv2.error as exc:
            logger.warning("Lỗi cv2 khi đọc frame (%s) — coi như lỗi tạm thời", exc)
            return False, None

    def release(self) -> None:
        """Đóng camera và giải phóng tài nguyên."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Đã đóng camera %d", self._index)

    # ---------------------------------------------------------
    # Thuộc tính
    # ---------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._cap is not None

    @property
    def actual_width(self) -> int:
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def actual_height(self) -> int:
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ---------------------------------------------------------
    # Tiện ích
    # ---------------------------------------------------------

    @staticmethod
    def list_available(max_index: int = 5) -> list[int]:
        """Quét các index camera khả dụng (thử mở từng cái rồi đóng ngay)."""
        available: list[int] = []
        for i in range(max_index):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
            cap.release()
        return available
