"""Embedding khuôn mặt — ArcFace (module recognition của bộ model buffalo_l).

Bước 5: sau khi PHÁT HIỆN được khuôn mặt (Bước 4, biết mặt ở ĐÂU),
ta trích "dấu vân tay số" của nó — vector 512 chiều (biết mặt LÀ AI).

Tính chất quan trọng:
  - Hai khuôn mặt CÙNG người  → 2 vector embedding rất gần nhau (cosine ~ 0.6+)
  - Hai khuôn mặt KHÁC người  → 2 vector embedding xa nhau (cosine thấp hơn nhiều)

Chi tiết: FaceAnalysis ĐẦY ĐỦ đã tính sẵn embedding ngay trong `get()`
(normed_embedding, đã chuẩn hóa L2). Vì vậy FaceEmbedder TÁI SỬ DỤNG chung
FaceAnalysis với FaceDetector — KHÔNG nạp model lần thứ hai (tốn RAM + thời gian).
"""
from __future__ import annotations

import logging

import numpy as np

from app.core.detector import FaceDetector

logger = logging.getLogger(__name__)

# Chiều vector embedding của buffalo_l/w600k_r50 (ArcFace)
EMBEDDING_DIM = 512


class FaceEmbedder:
    """Bọc module nhận diện: trích vector 512 chiều từ khuôn mặt đã phát hiện."""

    def __init__(self, detector: FaceDetector) -> None:
        # Dùng CHUNG FaceAnalysis với detector (model đã nạp từ Bước 4)
        self._app = detector.app
        logger.info("FaceEmbedder sẵn sàng (ArcFace w600k_r50, %d chiều)", EMBEDDING_DIM)

    def embed_face(self, face) -> np.ndarray | None:
        """Trả vector embedding chuẩn hóa L2 (512,); None nếu khuôn mặt thiếu embedding."""
        embedding = getattr(face, "normed_embedding", None)
        if embedding is None:
            return None
        return np.asarray(embedding, dtype=np.float32)

    def detect_and_embed(self, img_bgr: np.ndarray) -> list[tuple]:
        """Một lượt: phát hiện + trích embedding → [(face, embedding|None), ...]."""
        faces = self._app.get(img_bgr)
        return [(face, self.embed_face(face)) for face in faces]

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Độ giống nhau giữa 2 embedding trong khoảng [-1, 1].

        1.0 = giống hệt · gần 0 = không liên quan · gần -1 = đối nghịch.
        Với embedding đã chuẩn hóa L2, cosine = tích vô hướng của 2 vector.
        """
        a = np.asarray(a, dtype=np.float32).flatten()
        b = np.asarray(b, dtype=np.float32).flatten()
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return 0.0  # vector rỗng — không thể so sánh
        cosine = float(np.dot(a, b) / (na * nb))
        if not np.isfinite(cosine):
            return 0.0  # phòng NaN (dữ liệu hỏng)
        return cosine
