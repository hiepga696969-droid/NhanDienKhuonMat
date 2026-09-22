"""Bộ so khớp khuôn mặt — cosine similarity (Bước 6).

Sau khi có embedding (Bước 5) và CSDL lưu embedding đã đăng ký (Bước 6 trước),
bước này trả lời câu hỏi: *khuôn mặt trước camera giống AI nhất trong danh
sách đã đăng ký?*

ADR-5: với <50 người, brute-force NumPy (tính cosine với TOÀN BỘ embedding
đã lưu, lấy giá trị cao nhất) là đủ nhanh (hàng trăm nghìn phép tính/giây)
và không cần thêm thư viện. Giao diện ``Matcher`` cho phép đổi sang
FAISS/ChromaDB sau này mà không sửa nơi gọi.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from app.infrastructure.repositories import FaceSample

logger = logging.getLogger(__name__)

# Ngưỡng mặc định của InsightFace cho buffalo_l (sẽ hiệu chỉnh theo webcam thật)
DEFAULT_THRESHOLD = 0.40


def _normalize(vector: np.ndarray) -> np.ndarray:
    """Chuẩn hóa L2 (độ dài = 1). Với vector rỗng/all-zero, giữ nguyên (tránh chia 0)."""
    v = np.asarray(vector, dtype=np.float32).flatten()
    norm = float(np.linalg.norm(v))
    if norm < 1e-9:
        return v
    return v / norm


@dataclass(frozen=True)
class MatchResult:
    """Kết quả so khớp: ai giống nhất và điểm tương đồng (0..1)."""

    person_id: str
    similarity: float


class Matcher(ABC):
    """Giao diện chung cho bộ so khớp — đổi thuật toán mà không đổi nơi gọi."""

    @abstractmethod
    def register(self, person_id: str, embedding: np.ndarray) -> None:
        """Đăng ký một embedding của một người."""

    @abstractmethod
    def load_samples(self, samples: list[FaceSample]) -> None:
        """Nạp toàn bộ mẫu embedding từ CSDL (thay thế danh sách cũ)."""

    @abstractmethod
    def match(self, embedding: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> MatchResult | None:
        """Tìm người giống nhất; None nếu điểm cao nhất vẫn dưới ngưỡng."""

    @abstractmethod
    def clear(self) -> None:
        """Xóa toàn bộ danh sách đã đăng ký."""

    @property
    @abstractmethod
    def size(self) -> int:
        """Số mẫu embedding đang có."""


class CosineMatcher(Matcher):
    """So khớp cosine với Outlier Rejection (Bước 20).

    Thay vì dùng mean TẤT CẢ mẫu (cách cũ — dễ kéo trung bình xuống
    nếu có mẫu xấu), giờ:
    1. Lưu TỪNG mẫu embedding riêng (không chỉ tổng + số lượng)
    2. Khi so khớp: tính mean sơ bộ → loại outlier (cosine < mean - 2×std)
       → tính lại mean từ mẫu tốt → cosine với query.
    3. Nếu ≤3 mẫu: giữ nguyên mean (chưa đủ dữ liệu để loại outlier).

    Hiệu suất: với ≤50 người × ≤10 mẫu/người, brute-force NumPy vẫn
    hàng trăm nghìn phép tính/giây — không cần FAISS.
    """

    def __init__(self) -> None:
        self._person_ids: list[str] = []
        self._all_samples: dict[str, list[np.ndarray]] = {}  # TỪNG mẫu raw
        self._sums: dict[str, np.ndarray] = {}   # tổng embedding (chưa chuẩn hóa)
        self._counts: dict[str, int] = {}        # số mẫu mỗi người

    # ---------------------------------------------------------
    # Đăng ký / nạp dữ liệu
    # ---------------------------------------------------------
    def register(self, person_id: str, embedding: np.ndarray) -> None:
        """Thêm 1 mẫu embedding vào danh sách của người (lưu cả raw + tổng)."""
        emb = _normalize(embedding)
        if person_id in self._sums:
            self._all_samples[person_id].append(emb.copy())
            self._sums[person_id] = self._sums[person_id] + emb
            self._counts[person_id] += 1
        else:
            self._person_ids.append(person_id)
            self._all_samples[person_id] = [emb.copy()]
            self._sums[person_id] = emb.copy()
            self._counts[person_id] = 1

    def load_samples(self, samples: list[FaceSample]) -> None:
        self.clear()
        for sample in samples:
            self.register(sample.person_id, sample.embedding)
        logger.info(
            "Matcher nạp %d người (%d mẫu embedding)",
            len(self._person_ids),
            len(samples),
        )

    def load_samples_with_ids(
        self, samples: list[FaceSample], person_ids: list[str]
    ) -> None:
        """Nạp mẫu kèm person_id array (để không gọi label_of mỗi lần)."""
        # Placeholder — dùng load_samples cho đơn giản
        self.load_samples(samples)

    def clear(self) -> None:
        self._person_ids.clear()
        self._all_samples.clear()
        self._sums.clear()
        self._counts.clear()

    @property
    def size(self) -> int:
        """Số NGƯỜI đang có trong bộ so khớp (mỗi người 1 vector trung bình)."""
        return len(self._person_ids)

    # ---------------------------------------------------------
    # So khớp
    # ---------------------------------------------------------
    def _mean_of(self, person_id: str) -> np.ndarray:
        """Embedding trung bình — loại bỏ outlier trước khi tính (Bước 20).

        Thuật toán:
        1. Nếu ≤3 mẫu: dùng mean thô (chưa đủ dữ kiện để loại outlier).
        2. Tính mean sơ bộ từ TẤT CẢ mẫu.
        3. Tính cosine của từng mẫu với mean sơ bộ.
        4. Loại mẫu có cosine < mean(ces) - 2×std(ces) (outlier).
        5. Tính lại mean từ mẫu tốt; fallback giữ tất cả nếu <2 mẫu tốt.
        """
        samples = self._all_samples.get(person_id, [])
        n = len(samples)
        if n == 0:
            return np.zeros(512, dtype=np.float32)
        if n <= 3:
            # Chưa đủ mẫu → dùng mean thô
            return _normalize(self._sums[person_id] / self._counts[person_id])

        # Mean sơ bộ
        mean0 = _normalize(self._sums[person_id] / self._counts[person_id])

        # Cosine của từng mẫu với mean sơ bộ
        sims = np.array([float(np.dot(s, mean0)) for s in samples])

        # Outlier threshold: mean - 2×std
        mu, sigma = float(sims.mean()), float(sims.std())
        threshold = mu - 2.0 * sigma

        good = [s for s, sc in zip(samples, sims) if sc >= threshold]
        if len(good) < 2:
            good = samples  # fallback: giữ tất cả

        return _normalize(np.mean(good, axis=0))

    def _scores(self, embedding: np.ndarray) -> np.ndarray:
        """Cosine của embedding đầu vào với TỪNG NGƯỜI (mảng (P,))."""
        if not self._person_ids:
            return np.array([], dtype=np.float32)
        query = _normalize(embedding)
        matrix = np.stack([self._mean_of(pid) for pid in self._person_ids])
        return matrix @ query                    # (P,) — cosine vì đã L2 chuẩn hóa

    def match(self, embedding: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> MatchResult | None:
        scores = self._scores(embedding)
        if scores.size == 0:
            return None
        idx = int(np.argmax(scores))
        best = float(scores[idx])
        if best >= threshold:
            return MatchResult(person_id=self._person_ids[idx], similarity=best)
        return None  # người lạ — không đạt ngưỡng

    def rank(self, embedding: np.ndarray, top_k: int | None = None) -> list[MatchResult]:
        """Xếp hạng TỪNG NGƯỜI theo độ giống (trung bình mẫu) giảm dần."""
        scores = self._scores(embedding)
        order = np.argsort(-scores)
        results = [
            MatchResult(person_id=self._person_ids[i], similarity=float(scores[i]))
            for i in order
        ]
        return results[:top_k] if top_k is not None else results
