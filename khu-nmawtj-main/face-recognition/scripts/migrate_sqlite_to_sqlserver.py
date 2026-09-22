"""Di chuyển dữ liệu SQLite cũ (data/app.db) → SQL Server FaceRecognitionDB.

Chạy:  .venv\\Scripts\\python.exe scripts\\migrate_sqlite_to_sqlserver.py

CHỈ chạy khi bạn có dữ liệu đăng ký cũ trong SQLite muốn giữ lại. Bỏ qua
hoàn toàn nếu bắt đầu từ đầu — app tự dùng SQL Server.

Nguyên tắc an toàn:
- KHÔNG tạo/xóa/sửa cấu trúc bảng nào (cả 2 phía).
- Insert vào SQL Server theo kiểu "bỏ qua nếu id đã tồn tại" — chạy lại
  nhiều lần không bị trùng, không ghi đè dữ liệu SQL Server hiện có.
- Bảng attendance và sync_outbox KHÔNG chuyển (attendance là dữ liệu
  local theo thiết kế; outbox là hàng đợi tạm — để trống để app tự sync).
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.infrastructure.db import DB_PATH  # noqa: E402
from app.infrastructure.db import Database  # noqa: E402


def _ts(db: Database, table: str, col: str, val: str | None) -> str | None:
    """Chuẩn hóa timestamp theo kiểu cột đích (bỏ 'Z' nếu DATETIME)."""
    return db.ts(table, col, val)


def main() -> int:
    src_path = DB_PATH
    if not src_path.exists():
        print(f"Không thấy SQLite cũ tại {src_path} — không cần di chuyển.")
        return 0
    rows_total = 0
    src = sqlite3.connect(str(src_path))
    src.row_factory = sqlite3.Row
    dst = Database()
    dst.connect()

    def insert(db: Database, table: str, cols: list[str], row: sqlite3.Row) -> bool:
        """Insert idempotent theo id; False nếu id đã có (bỏ qua)."""
        exists = db.connect().execute(
            f"SELECT 1 FROM {table} WHERE id = ?", (row["id"],)
        ).fetchone()
        if exists:
            return False
        col_sql = ", ".join(cols)
        marks = ", ".join("?" for _ in cols)
        vals = [
            _ts(db, table, c, row[c]) if c.endswith(("_at", "_date")) and row[c] is not None
            else row[c]
            for c in cols
        ]
        with db.session() as conn:
            conn.execute(f"INSERT INTO {table} ({col_sql}) VALUES ({marks})", vals)
        return True

    # 1) persons
    cols = ["id", "name", "created_at", "thumbnail_path", "thumbnail_r2_key",
            "department", "position"]
    for row in src.execute(f"SELECT {', '.join(cols)} FROM persons"):
        r = dict(row)
        r.setdefault("department", "")
        r.setdefault("position", "")
        if insert(dst, "persons", cols, r):
            rows_total += 1
            print("  + persons", r["id"][:8], r["name"])

    # 2) face_samples — embedding đọc bytes từ SQLite, truyền nguyên
    cols = ["id", "person_id", "embedding", "quality", "captured_at"]
    col_type = (dst.column_type("face_samples", "embedding") or "").lower()
    as_hex = "varchar" in col_type or col_type in ("text", "ntext")
    for row in src.execute(f"SELECT {', '.join(cols)} FROM face_samples"):
        r = dict(row)
        if as_hex and isinstance(r["embedding"], (bytes, bytearray)):
            r["embedding"] = bytes(r["embedding"]).hex()
        if insert(dst, "face_samples", cols, r):
            rows_total += 1
            print("  + face_sample", r["id"][:8])

    # 3) recognition_events
    cols = ["id", "person_id", "label", "source", "detected_at", "similarity",
            "snapshot_path", "snapshot_r2_key", "is_unknown", "mode"]
    for row in src.execute(f"SELECT {', '.join(cols)} FROM recognition_events"):
        r = dict(row)
        r.setdefault("mode", "")
        if insert(dst, "recognition_events", cols, r):
            rows_total += 1
            print("  + recognition_event", r["id"][:8], r["label"][:30])

    src.close()
    dst.close()
    print(f"Xong — đã chuyển {rows_total} dòng mới (bỏ qua dòng đã tồn tại).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
