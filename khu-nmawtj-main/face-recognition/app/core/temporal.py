"""Bộ nhớ tạm thời + Theo dõi centroid — giải quyết nhảy tên (Bước 19).

Vấn đề: cosine similarity giữa cùng 1 khuôn mặt dao động giữa các frame
→ tên xuất hiện 1 frame rồi biến mất → rất khó chịu.

Giải pháp:
1. **TemporalBuffer** — lưu N kết quả gần nhất cho mỗi bbox, chỉ hiển thị
   khi ≥60% khung đồng ý (bỏ phiếu). Sau khi mất tín hiệu, giữ tên thêm
   PERSIST_FRAMES khung để tránh nhấp nháy.

2. **FaceTracker** — gán track_id cho mỗi khuôn mặt qua centroid tracking.
   Nhờ vậy temporal buffer phân biệt được 2 người khác nhau trong cùng khung.
"""
from __future__ import annotations

import logging
from collections import defaultdict, deque

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Hằng số — chỉnh trong Settings (slider "Độ ổn định")
# ─────────────────────────────────────────────────────────────
DEFAULT_WINDOW_SIZE = 5      # số khung trong cửa sổ bỏ phiếu
MIN_AGREE_RATIO = 0.60      # tối thiểu 60% khung đồng ý để hiển thị tên
PERSIST_FRAMES = 15          # giữ tên thêm N khung khi mất tín hiệu

# FaceTracker
MAX_TRACK_DISAPPEAR = 25     # xóa track sau N khung không thấy
MAX_TRACK_DISTANCE = 150.0   # centroid cách >150px → track mới


class TemporalBuffer:
    """Bộ nhớ tạm cho MỘT khuôn mặt — lưu N kết quả gần nhất, bỏ phiếu.

    Cách dùng::

        buf = TemporalBuffer(window_size=5)
        for frame in range(100):
            result = matcher.match(embedding, threshold)
            stable_id = buf.add(result)
            # stable_id = person_id (ổn định) hoặc None (chưa quyết định)
    """

    def __init__(self, window_size: int = DEFAULT_WINDOW_SIZE) -> None:
        self._window: deque = deque(maxlen=window_size)
        self._last_stable: str | None = None   # person_id ổn định cuối cùng
        self._persist_counter: int = 0         # đếm khung giữ tên sau khi mất

    def add(self, result) -> str | None:
        """Thêm kết quả MatchResult | None, trả về person_id ổn định (hoặc None)."""
        from app.core.matcher import MatchResult  # tránh circular import

        self._window.append(result)

        # Đếm vote trong cửa sổ hiện tại
        votes: dict[str, list[float]] = defaultdict(list)
        for r in self._window:
            if isinstance(r, MatchResult):
                votes[r.person_id].append(r.similarity)

        if not votes:
            # Không có ai trong cửa sổ → áp dụng persist
            return self._apply_persist(None)

        # Tìm person_id có nhiều vote nhất, overlap相似度 cao nhất
        best_id = max(
            votes,
            key=lambda pid: (len(votes[pid]), sum(votes[pid]) / len(votes[pid])),
        )
        best_count = len(votes[best_id])
        best_ratio = best_count / len(self._window)

        if best_ratio >= MIN_AGREE_RATIO:
            # Đủ đồng nhất → hiển thị tên
            self._last_stable = best_id
            self._persist_counter = PERSIST_FRAMES
            return best_id

        # Chưa đủ đồng nhất → kiểm tra persist
        return self._apply_persist(None)

    def _apply_persist(self, candidate: str | None) -> str | None:
        """Khi mất tín hiệu, giữ tên cũ thêm PERSIST_FRAMES khung."""
        if self._last_stable is not None and self._persist_counter > 0:
            self._persist_counter -= 1
            return self._last_stable
        return candidate

    def reset(self) -> None:
        """Xóa toàn bộ bộ nhớ (khi camera dừng/mở lại)."""
        self._window.clear()
        self._last_stable = None
        self._persist_counter = 0


class FaceTracker:
    """Theo dõi centroid đơn giản — gán track_id cho mỗi khuôn mặt.

    Thuật toán:
    1. Tính centroid (trung tâm) bbox mới.
    2. So sánh với centroid frame trước → bbox nào gần nhất = cùng người.
    3. bbox mới quá xa → gán track_id mới.
    4. bbox biến mất quá lâu → xóa track.
    """

    def __init__(self) -> None:
        self._next_id: int = 0
        # {track_id: [centroid_x, centroid_y, frames_disappeared]}
        self._tracks: dict[int, list] = {}

    def update(self, bboxes: list[np.ndarray]) -> dict[int, int]:
        """Nhận list bbox mới → trả về dict {bbox_index: track_id}.

        bbox định dạng [x1, y1, x2, y2] hoặc [x1, y1, x2, y2, ...].
        """
        result: dict[int, int] = {}

        if not bboxes:
            # Không có bbox nào → tăng frames_disappeared cho mọi track
            for tid in self._tracks:
                self._tracks[tid][2] += 1
            self._remove_stale()
            return result

        # Tính centroid bbox mới
        new_centroids: list[list[float]] = []
        for b in bboxes:
            cx = float((b[0] + b[2]) / 2)
            cy = float((b[1] + b[3]) / 2)
            new_centroids.append([cx, cy])

        # Nếu chưa có track nào → gán id mới cho tất cả
        if not self._tracks:
            for i in range(len(new_centroids)):
                tid = self._assign_new()
                self._tracks[tid] = [new_centroids[i][0], new_centroids[i][1], 0]
                result[i] = tid
            return result

        # Matching: tìm centroid mới gần nhất với mỗi track
        used_tracks: set[int] = set()
        used_bboxes: set[int] = set()

        # Tạo ma trận khoảng cách (tracks × bboxes)
        track_ids = list(self._tracks.keys())
        dist_matrix = np.full((len(track_ids), len(new_centroids)), 9999.0)
        for ti, tid in enumerate(track_ids):
            tx, ty = self._tracks[tid][0], self._tracks[tid][1]
            for bi, (nx, ny) in enumerate(new_centroids):
                dist_matrix[ti, bi] = np.sqrt((tx - nx) ** 2 + (ty - ny) ** 2)

        # Greedy matching: bbox nào gần nhất với track nào (không trùng)
        for _ in range(min(len(track_ids), len(new_centroids))):
            min_idx = np.unravel_index(dist_matrix.argmin(), dist_matrix.shape)
            ti, bi = int(min_idx[0]), int(min_idx[1])
            dist = dist_matrix[ti, bi]

            if dist > MAX_TRACK_DISTANCE:
                break  # bbox còn lại quá xa → là người mới

            tid = track_ids[ti]
            used_tracks.add(tid)
            used_bboxes.add(bi)

            # Cập nhật centroid + reset disappeared
            self._tracks[tid] = [
                new_centroids[bi][0], new_centroids[bi][1], 0
            ]
            result[bi] = tid

            # Đánh dấu đã dùng (đặt khoảng cách vô cực)
            dist_matrix[ti, :] = 9999.0
            dist_matrix[:, bi] = 9999.0

        #_bbox chưa match → track mới
        for bi in range(len(new_centroids)):
            if bi not in used_bboxes:
                tid = self._assign_new()
                self._tracks[tid] = [new_centroids[bi][0], new_centroids[bi][1], 0]
                result[bi] = tid

        # Track chưa match → tăng disappeared
        for tid in track_ids:
            if tid not in used_tracks:
                self._tracks[tid][2] += 1

        self._remove_stale()
        return result

    def _assign_new(self) -> int:
        """Gán track_id mới (tăng dần)."""
        tid = self._next_id
        self._next_id += 1
        return tid

    def _remove_stale(self) -> None:
        """Xóa track biến mất quá lâu."""
        stale = [
            tid for tid, (_, _, disp) in self._tracks.items()
            if disp > MAX_TRACK_DISAPPEAR
        ]
        for tid in stale:
            del self._tracks[tid]

    def reset(self) -> None:
        """Xóa toàn bộ tracks (khi camera dừng/mở lại)."""
        self._tracks.clear()
        self._next_id = 0
