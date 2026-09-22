"""Tạo dữ liệu mẫu (seed) cho CSDL — dùng để thử nghiệm trước khi có
tính năng đăng ký thật (Bước 8).

Chạy:  .venv\\Scripts\\python.exe scripts\\seed_demo.py

⚠ QUAN TRỌNG: embedding ở đây là VECTOR NGẪU NHIÊN (mô phỏng), KHÔNG
phải embedding thật của khuôn mặt — chỉ để thử CRUD, danh sách và so
khớp giả lập. Người demo có thể xóa qua giao diện sau này.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.infrastructure.db import Database  # noqa: E402
from app.infrastructure.repositories import (  # noqa: E402
    EMBEDDING_DIM,
    FaceSampleRepository,
    PersonRepository,
)

DEMO_NAME = "Nguyễn Văn Demo"
DEMO_SAMPLES = 4  # 4 mẫu như đăng ký thật (3–5 mẫu/người)


def _random_embedding(rng: np.random.Generator) -> np.ndarray:
    """Vector ngẫu nhiên chuẩn hóa L2 = 1 (mô phỏng embedding thật)."""
    vec = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    return vec / np.linalg.norm(vec)


def main() -> None:
    db = Database()
    people = PersonRepository(db)
    samples = FaceSampleRepository(db)

    # Không ghi đè nếu đã có dữ liệu (an toàn khi chạy lại nhiều lần)
    if people.count() > 0:
        print(f"CSDL đã có {people.count()} người — bỏ qua seed (chạy thử với DB trống).")
        return

    rng = np.random.default_rng(seed=42)  # cố định seed → chạy lại giống nhau

    # 1) Thêm người demo
    person = people.add(name=DEMO_NAME, thumbnail_path="data/demo_thumbnail.png")
    print(f"[1] Đã thêm người: {person.name} (id={person.id})")

    # 2) Thêm 4 mẫu embedding ngẫu nhiên
    for i in range(DEMO_SAMPLES):
        sample = samples.add(
            person_id=person.id,
            embedding=_random_embedding(rng),
            quality=round(0.85 + 0.03 * i, 2),  # tăng dần để giống thực tế
        )
        print(f"    Mẫu {i + 1}: {sample.embedding.shape} · chất lượng {sample.quality}")

    # 3) Kiểm chứng đọc lại
    loaded = samples.list_by_person(person.id)
    assert len(loaded) == DEMO_SAMPLES, "Đọc lại phải đủ số mẫu"
    assert loaded[0].embedding.shape == (EMBEDDING_DIM,)
    print(f"[2] Đọc lại: {len(loaded)} mẫu, vector {loaded[0].embedding.shape} ✓")
    print(f"\n✅ Đã seed dữ liệu demo vào {db.path}")

    db.close()


if __name__ == "__main__":
    main()
