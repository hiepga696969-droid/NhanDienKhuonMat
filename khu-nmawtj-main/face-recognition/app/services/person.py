"""Dịch vụ quản lý người dùng — use case FR-6 (Bước 8).

PersonListView KHÔNG thao tác repository trực tiếp mà gọi qua service
này (đúng kiến trúc phân lớp — spec 5.1):

  - ``list_with_stats()``: danh sách người kèm số mẫu embedding + thời
    điểm nhận diện gần nhất (phục vụ dòng \"✓ Nhận diện gần nhất...\").
  - ``rename()``: đổi tên người (chặn tên trống).
  - ``delete()``: xóa người — xóa luôn file thumbnail trên đĩa.
    Các face_samples tự xóa theo ON DELETE CASCADE; recognition_events
    giữ lại với person_id = NULL (ON DELETE SET NULL — không mất lịch sử).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from app.infrastructure.db import DATA_DIR, Database
from app.infrastructure.repositories import (
    FaceSampleRepository,
    Person,
    PersonRepository,
    RecognitionEventRepository,
)

logger = logging.getLogger(__name__)


@dataclass
class PersonStats:
    """Thông tin hiển thị của một người trong danh sách."""

    person: Person
    sample_count: int
    last_detected_at: str | None  # ISO; None = chưa từng được nhận diện


class PersonService:
    """Các use case thao tác trên danh sách người đã đăng ký."""

    def __init__(self, db: Database) -> None:
        self._people = PersonRepository(db)
        self._samples = FaceSampleRepository(db)
        self._events = RecognitionEventRepository(db)

    # ---------------------------------------------------------
    # Đọc
    # ---------------------------------------------------------
    def list_with_stats(self) -> list[PersonStats]:
        """Toàn bộ người (mới nhất trước) kèm số mẫu + lần nhận diện gần nhất."""
        people = self._people.list_all()
        stats: list[PersonStats] = []
        for person in people:
            stats.append(
                PersonStats(
                    person=person,
                    sample_count=self._samples.count_by_person(person.id),
                    last_detected_at=self._events.last_detected_at(person.id),
                )
            )
        return stats

    def count(self) -> int:
        """Tổng số người đã đăng ký."""
        return self._people.count()



    # ---------------------------------------------------------
    # Ghi
    # ---------------------------------------------------------
    def rename(self, person_id: str, new_name: str) -> bool:
        """Đổi tên người. Trả về True nếu thành công (tên trống → False)."""
        return self._people.rename(person_id, new_name)

    def update_info(
        self, person_id: str, name: str, department: str = "", position: str = ""
    ) -> bool:
        """Cập nhật tên + phòng ban + chức vụ nhân viên (v3)."""
        return self._people.update_info(
            person_id, name, department=department, position=position
        )

    def delete(self, person_id: str) -> bool:
        """Xóa người + file thumbnail trên đĩa (nếu có).

        Trả về True nếu xóa được; False nếu người không tồn tại.
        """
        person = self._people.get(person_id)
        if person is None:
            return False
        ok = self._people.delete(person_id)
        if ok and person.thumbnail_path:
            thumb = (DATA_DIR / person.thumbnail_path).resolve()
            # Chỉ xóa khi file nằm trong thư mục dữ liệu (chống đường dẫn lạ)
            if thumb.is_relative_to(DATA_DIR.resolve()) and thumb.exists():
                try:
                    thumb.unlink()
                    logger.info("Đã xóa thumbnail %s", thumb)
                except OSError:
                    logger.warning("Không xóa được thumbnail %s", thumb)
        return ok
