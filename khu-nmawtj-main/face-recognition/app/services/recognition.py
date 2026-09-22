"""Dịch vụ nhận diện thời gian thực — use case FR-2 (Bước 9).

Nhiệm vụ chính:
  1. ``reload()``: nạp TOÀN BỘ embedding đã đăng ký từ CSDL vào bộ so khớp
     (CosineMatcher, Bước 6) — gọi ở luồng UI trước khi mở camera.
  2. ``match()``: so khớp embedding của khuôn mặt trước camera với danh
     sách — trả về người giống nhất (hoặc None = người lạ).
  3. ``save_event()``: ghi sự kiện nhận diện vào ``recognition_events``
     + lưu ảnh snapshot (crop khuôn mặt) ra đĩa — audit trail.

QUAN TRỌNG (đa luồng): service này chỉ được dùng từ LUỒNG UI. Worker
camera (QThread) KHÔNG gọi DB trực tiếp — worker dùng matcher trong bộ
nhớ (đã nạp sẵn) và phát tín hiệu sự kiện về UI; UI mới gọi save_event.
Lý do: kết nối sqlite3 gắn với luồng tạo ra nó (check_same_thread).
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from app.infrastructure.db import DATA_DIR, Database
from app.infrastructure.repositories import (
    FaceSampleRepository,
    PersonRepository,
    RecognitionEventRepository,
)
from app.core.matcher import CosineMatcher, MatchResult

logger = logging.getLogger(__name__)

SNAPSHOT_DIR = DATA_DIR / "snapshots"


class RecognitionService:
    """So khớp khuôn mặt với danh sách đã đăng ký + ghi lịch sử nhận diện."""

    def __init__(self, db: Database) -> None:
        self._samples = FaceSampleRepository(db)
        self._people = PersonRepository(db)
        self._events = RecognitionEventRepository(db)
        self._matcher = CosineMatcher()
        self._names: dict[str, str] = {}

    # ---------------------------------------------------------
    # Nạp dữ liệu & so khớp
    # ---------------------------------------------------------
    def reload(self) -> None:
        """Nạp lại toàn bộ embedding + bảng tên từ CSDL (gọi mỗi khi mở camera)."""
        self._matcher.load_samples(self._samples.all_samples())
        # Bảng tên để worker vẽ nhãn mà KHÔNG cần đụng DB (worker chạy ở luồng khác)
        self._names = {p.id: p.name for p in self._people.list_all()}

    def label_of(self, person_id: str) -> str:
        """Tên hiển thị của người theo id (dùng trong luồng worker — không đụng DB)."""
        return self._names.get(person_id, "?")

    @property
    def size(self) -> int:
        """Số người đang có trong bộ so khớp (mỗi người 1 vector trung bình)."""
        return self._matcher.size

    def match(self, embedding: np.ndarray, threshold: float) -> MatchResult | None:
        """Người giống nhất với embedding; None nếu dưới ngưỡng (người lạ)."""
        return self._matcher.match(embedding, threshold=threshold)

    # ---------------------------------------------------------
    # Ghi sự kiện + snapshot (gọi từ luồng UI)
    # ---------------------------------------------------------
    def save_event(
        self,
        person_id: str | None,
        similarity: float | None,
        face_crop: np.ndarray,
        is_unknown: bool = False,
        is_spoof: bool = False,
        is_occluded: bool = False,
        source: str = "webcam",
        mode: str = "",
    ) -> str | None:
        """Ghi sự kiện nhận diện + lưu ảnh snapshot; trả về id sự kiện.

        ``face_crop`` là ảnh crop khuôn mặt (BGR) — worker cắt và gửi
        bản SAO qua tín hiệu, UI lưu file ở đây (không đụng frame gốc).
        ``is_spoof`` (Bước 17): nhận diện được người ĐÃ BIẾT nhưng nghi
        ngờ giả mạo — label "Giả mạo: <tên>" (giữ person_id để lịch sử
        biết là ai bị nghi ngờ), is_unknown vẫn là False.
        ``is_occluded`` (Bước 18): người ĐÃ BIẾT nhưng mặt bị che khuất
        nhiều (không đủ tin cậy) — label "Mặt bị che: <tên>", KHÔNG hiện
        tên trong nhận diện nhưng vẫn ghi audit trail để biết là ai.
        ``mode``: v4 — 'checkin'/'checkout' (loại quét VÀO/RA CA), rỗng =
        không phải quét điểm danh (hiển thị '—' trong Lịch sử quét).
        """
        try:
            if person_id is not None:
                person = self._people.get(person_id)
                if person is None:
                    return None  # người vừa bị xóa — bỏ qua
                if is_spoof:
                    label = f"Giả mạo: {person.name}"
                elif is_occluded:
                    label = f"Mặt bị che: {person.name}"
                else:
                    label = person.name
            else:
                label = "Người lạ"

            snapshot_path = self._save_snapshot(face_crop)
            return self._events.add(
                person_id=person_id,
                label=label,
                source=source,
                similarity=similarity,
                snapshot_path=snapshot_path,
                is_unknown=is_unknown,
                mode=mode,
            )
        except Exception:  # noqa: BLE001 — lỗi ghi log KHÔNG được làm chết camera
            logger.exception("Lỗi ghi sự kiện nhận diện")
            return None

    def _save_snapshot(self, face_crop: np.ndarray) -> str:
        """Lưu ảnh crop khuôn mặt vào data/snapshots/{uuid}.jpg; trả đường dẫn tương đối.

        Tên file dùng UUID — KHÔNG dùng timestamp: 2 sự kiện cùng mili-giây
        (2 người được nhận diện cùng lúc) sẽ trùng tên và ghi đè lẫn nhau.
        """
        import uuid

        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"{uuid.uuid4().hex}.jpg"
        abs_path = SNAPSHOT_DIR / filename
        cv2.imwrite(str(abs_path), face_crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return str(SNAPSHOT_DIR.relative_to(DATA_DIR) / filename)
