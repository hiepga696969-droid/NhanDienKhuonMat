"""Kiểm tra tầng CSDL + repository (Bước 6) — dùng DB TẠM, không đụng data/app.db.

Chạy:  .venv\\Scripts\\python.exe scripts\\test_repository.py

Kiểm tra:
1. Schema: đủ 5 bảng + index + PRAGMA (foreign_keys, WAL)
2. Person CRUD: thêm / đọc / sửa tên / xóa
3. Embedding roundtrip: vector float32 (512,) → BLOB → vector như cũ
4. Cascade: xóa người → mẫu embedding tự xóa
5. Khóa ngoại: thêm mẫu cho người không tồn tại → lỗi
"""
from __future__ import annotations

import sqlite3
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

# DB tạm riêng — tự xóa sau khi test
TEMP_DB = PROJECT_ROOT / "data" / "test_repository.db"

EXPECTED_TABLES = {
    "persons",
    "face_samples",
    "recognition_events",
    "sync_outbox",
    "settings",
}

passed = True


def check(label: str, ok: bool) -> None:
    global passed
    if ok:
        print(f"  ✓ {label}")
    else:
        passed = False
        print(f"  ✗ THẤT BẠI: {label}")


def main() -> None:
    for suffix in ("", "-wal", "-shm"):  # dọn file DB tạm (kể cả WAL/SHM)
        Path(str(TEMP_DB) + suffix).unlink(missing_ok=True)

    db = Database(TEMP_DB)
    people = PersonRepository(db)
    samples = FaceSampleRepository(db)

    # ---- 1) Schema ----
    print("\n[1] Khởi tạo schema")
    with db.session() as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        tables = {r["name"] for r in rows if not r["name"].startswith("sqlite_")}
        check("đủ 5 bảng", EXPECTED_TABLES.issubset(tables))
        check(
            "PRAGMA foreign_keys = ON",
            conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1,
        )
        check(
            "PRAGMA journal_mode = WAL",
            conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal",
        )
        check("user_version = 3", conn.execute("PRAGMA user_version").fetchone()[0] == 3)  # v3: + phòng ban/chức vụ

    # ---- 2) Person CRUD ----
    print("\n[2] Person CRUD")
    # created_at khác nhau rõ ràng → thứ tự list_all xác định (không flaky
    # khi 2 insert rơi vào cùng mili-giây — ORDER BY created_at sẽ bằng nhau)
    p1 = people.add("Nguyễn Văn A", "data/thumbs/a.png", created_at="2026-01-01T00:00:00.000Z")
    p2 = people.add("Trần Thị B", "data/thumbs/b.png", created_at="2026-01-02T00:00:00.000Z")
    check("thêm 2 người → count = 2", people.count() == 2)
    check("get theo id trả đúng tên", people.get(p1.id).name == "Nguyễn Văn A")
    check("get id không tồn tại → None", people.get("khong-co") is None)
    check("list_all sắp mới nhất trước", people.list_all()[0].id == p2.id)

    check("rename thành công", people.rename(p1.id, "Nguyễn Văn A2") is True)
    check("rename tên trống bị từ chối", people.rename(p1.id, "   ") is False)
    check("tên đã đổi", people.get(p1.id).name == "Nguyễn Văn A2")

    # ---- 3) Embedding roundtrip ----
    print("\n[3] Embedding roundtrip (512 × float32)")
    rng = np.random.default_rng(7)
    emb = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    sample = samples.add(p1.id, emb, quality=0.92)
    loaded = samples.list_by_person(p1.id)[0]
    check("đúng số chiều sau khi đọc lại", loaded.embedding.shape == (EMBEDDING_DIM,))
    check("đúng dtype float32", loaded.embedding.dtype == np.float32)
    check(
        "giá trị giữ nguyên (tối đa sai lệch 1e-6)",
        np.max(np.abs(loaded.embedding - emb)) < 1e-6,
    )
    check("chất lượng lưu đúng", loaded.quality == 0.92)

    # ---- 4) Cascade delete ----
    print("\n[4] Xóa người → cascade xóa mẫu")
    before = samples.count_by_person(p1.id)
    people.delete(p1.id)
    after = samples.count_by_person(p1.id)
    check(f"xóa người → mẫu biến mất ({before} → {after})", after == 0)

    # ---- 5) Khóa ngoại ----
    print("\n[5] Khóa ngoại")
    try:
        samples.add("nguoi-khong-ton-tai", emb)
        check("thêm mẫu cho người không tồn tại phải lỗi", False)
    except sqlite3.IntegrityError:
        check("thêm mẫu cho người không tồn tại bị chặn (FK)", True)

    db.close()
    for suffix in ("", "-wal", "-shm"):
        Path(str(TEMP_DB) + suffix).unlink(missing_ok=True)

    print("\n" + ("=== TẤT CẢ TEST REPOSITORY ĐỀU QUA ===" if passed else "=== CÓ TEST THẤT BẠI ==="))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
