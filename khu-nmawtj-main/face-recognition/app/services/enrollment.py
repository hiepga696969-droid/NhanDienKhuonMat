"""Dịch vụ đăng ký khuôn mặt — use case FR-1 (Bước 7).

Nhận danh sách các mẫu đã thu (embedding + chất lượng + ảnh crop khuôn
mặt) từ EnrollmentDialog, rồi:
  1. Tạo bản ghi ``persons`` (tên, thumbnail = ảnh crop mặt tốt nhất).
  2. Tạo các bản ghi ``face_samples`` (3–5 mẫu embedding/người).

UI không thao tác repository trực tiếp mà gọi qua service này
(theo kiến trúc phân lớp — spec mục 5.1).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.infrastructure.db import DATA_DIR, Database
from app.infrastructure.repositories import (
    FaceSampleRepository,
    Person,
    PersonRepository,
)

logger = logging.getLogger(__name__)

THUMBS_DIR = DATA_DIR / "thumbs"


@dataclass
class CapturedSample:
    """Một khung hình tốt đã được chọn trong lúc đăng ký."""

    embedding: np.ndarray  # float32 (512,)
    quality: float         # điểm tin cậy của phát hiện (0..1)
    face_crop: np.ndarray  # ảnh crop khuôn mặt (BGR) — dùng làm thumbnail


class EnrollmentService:
    """Lưu người mới cùng các mẫu embedding vào CSDL."""

    def __init__(self, db: Database, thumbs_dir: Path = THUMBS_DIR) -> None:
        self._db = db
        self._people = PersonRepository(db)
        self._samples = FaceSampleRepository(db)
        self._thumbs_dir = thumbs_dir

    # ---------------------------------------------------------
    # Use case chính
    # ---------------------------------------------------------
    def save_person(
        self,
        name: str,
        samples: list[CapturedSample],
        department: str = "",
        position: str = "",
    ) -> Person:
        """Lưu nhân viên + embedding. Trả về Person đã lưu.

        Ném ValueError nếu tên trống hoặc không có mẫu nào.
        ``department``/``position``: phòng ban + chức vụ (tùy chọn — v3).
        """
        name = name.strip()
        if not name:
            raise ValueError("Tên không được để trống")
        if not samples:
            raise ValueError("Không có mẫu embedding nào để lưu")

        # 1) Tạo thumbnail (ảnh crop mặt của mẫu đầu tiên — đại diện)
        self._thumbs_dir.mkdir(parents=True, exist_ok=True)
        person = self._people.add(
            name=name, thumbnail_path="", department=department, position=position
        )
        thumb_path = self._save_thumbnail(person.id, samples[0].face_crop)
        self._people.update_thumbnail(person.id, thumb_path)

        # 2) Lưu từng mẫu embedding
        for sample in samples:
            self._samples.add(
                person_id=person.id,
                embedding=sample.embedding,
                # Chuyển quality từ thang 0-100 (SampleQuality.overall)
                # sang 0-1 (DB CHECK constraint: 0.0 ≤ quality ≤ 1.0)
                quality=float(sample.quality) / 100.0 if sample.quality > 1.0 else float(sample.quality),
            )

        # Đọc lại từ DB để trả Person với thumbnail_path đã cập nhật
        person = self._people.get(person.id)
        logger.info(
            "Đăng ký thành công: '%s' (id=%s) với %d mẫu",
            person.name, person.id, len(samples),
        )
        return person

    def add_samples(self, person: Person, samples: list[CapturedSample]) -> int:
        """Thêm thêm mẫu embedding cho người ĐÃ CÓ (đăng ký bổ sung).

        Dùng khi đăng ký hàng loạt gặp tên trùng người sẵn trong DB:
        không tạo hồ sơ mới — thêm mẫu vào hồ sơ cũ (nhận diện chính xác
        hơn với ảnh từ nhiều góc/điều kiện sáng khác nhau).

        Trả về số mẫu đã lưu thực sự (mẫu lỗi bị bỏ qua, không giết lô).
        """
        saved = 0
        for sample in samples:
            try:
                self._samples.add(
                    person_id=person.id,
                    embedding=sample.embedding,
                    quality=float(sample.quality) / 100.0
                    if sample.quality > 1.0 else float(sample.quality),
                )
                saved += 1
            except Exception:  # noqa: BLE001 — 1 mẫu hỏng không chặn phần còn lại
                logger.exception("Bỏ qua mẫu lỗi khi thêm vào '%s'", person.name)
        # Thumbnail: giữ ảnh cũ nếu đã có; chưa có (hiếm) thì cập nhật
        if saved and not person.thumbnail_path:
            thumb_path = self._save_thumbnail(person.id, samples[0].face_crop)
            self._people.update_thumbnail(person.id, thumb_path)
        logger.info(
            "Đã thêm %d/%d mẫu vào người có sẵn '%s' (id=%s)",
            saved, len(samples), person.name, person.id,
        )
        return saved

    def find_by_name(self, name: str) -> Person | None:
        """Tìm người theo TÊN (không phân biệt hoa/thường) — None nếu chưa có.

        Dùng khi đăng ký hàng loạt từ ảnh: trùng tên người có sẵn → thêm
        mẫu vào hồ sơ cũ thay vì tạo hồ sơ trùng.
        """
        return self._people.find_by_name(name)

    def update_info(
        self, person: Person, department: str = "", position: str = ""
    ) -> bool:
        """Cập nhật phòng ban/chức vụ người CÓ SẴN (giữ nguyên tên).

        Dùng khi lưu hàng loạt: tên trùng hồ sơ cũ nhưng người dùng có điền
        thêm phòng ban/chức vụ trong dialog đặt tên → áp dụng luôn.
        """
        return self._people.update_info(
            person.id, person.name,
            department=department or person.department,
            position=position or person.position,
        )

    # ---------------------------------------------------------
    # Nội bộ
    # ---------------------------------------------------------
    def _save_thumbnail(self, person_id: str, face_crop: np.ndarray) -> str:
        """Lưu ảnh crop khuôn mặt làm thumbnail; trả đường dẫn tương đối với DATA_DIR.

        Dùng ``cv2.imencode + tofile`` thay vì ``cv2.imwrite`` — imwrite
        FAIL IM LẶNG với đường dẫn tiếng Việt/khoảng trắng trên Windows
        (cùng lớp lỗi với cv2.imread mà luồng đọc ảnh đã tránh).
        """
        filename = f"{person_id}.jpg"
        abs_path = self._thumbs_dir / filename
        # Nén JPEG chất lượng cao (ảnh nhỏ ~10-30KB)
        ok, buf = cv2.imencode(".jpg", face_crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            logger.warning("Không mã hóa được thumbnail cho người %s", person_id)
            return ""
        buf.tofile(str(abs_path))  # ghi file hỗ trợ Unicode path
        # Lưu đường dẫn TƯƠNG ĐỐI với thư mục dữ liệu (di động khi đổi máy)
        return str(self._thumbs_dir.relative_to(DATA_DIR) / filename)
