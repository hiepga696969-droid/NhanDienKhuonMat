"""Repository pattern — tầng truy xuất dữ liệu người + embedding (Bước 6).

UI/service KHÔNG viết câu lệnh SQL trực tiếp mà gọi qua repository —
mỗi repository gói trọn các thao tác SQL của một "thực thể" (person,
face_sample). Lợi ích: code sạch, dễ đổi nguồn dữ liệu sau này (ví dụ
D1 cloud), dễ kiểm thử.

Embedding được lưu dạng BLOB: mảng numpy.float32 (512 phần tử) → bytes
(2048 bytes) bằng ``tobytes()``; khi đọc dùng ``np.frombuffer``.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

import numpy as np

from app.infrastructure.db import Database
from app.infrastructure.sqlserver_db import utc_now_iso as _utc_now_iso

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 512  # kích thước vector ArcFace (buffalo_l)


# =============================================================
# Mô hình dữ liệu (dataclass — đối tượng thuần, không liên quan SQL)
# =============================================================

@dataclass
class Person:
    """Một người đã đăng ký khuôn mặt (nhân viên)."""

    id: str
    name: str
    created_at: str
    thumbnail_path: str
    thumbnail_r2_key: str | None = None
    department: str = ""   # phòng/ban (rỗng = chưa khai)
    position: str = ""     # chức vụ (rỗng = chưa khai)


@dataclass
class FaceSample:
    """Một mẫu embedding của một người (mỗi người có 3–5 mẫu)."""

    id: str
    person_id: str
    embedding: np.ndarray  # float32, hình dạng (EMBEDDING_DIM,)
    quality: float
    captured_at: str


@dataclass
class RecognitionEvent:
    """Một sự kiện nhận diện — audit trail (wireframe 5.5.4).

    is_unknown = True khi là người lạ (person_id = None, label = 'Người lạ').
    """

    id: str
    person_id: str | None
    label: str
    source: str
    detected_at: str
    similarity: float | None
    snapshot_path: str
    is_unknown: bool
    snapshot_r2_key: str | None = None
    mode: str = ""  # v4: 'checkin' / 'checkout' (loại quét VÀO/RA CA), rỗng = khác


def _new_id() -> str:
    """Tạo ID UUID v4 dạng hex — tránh xung đột khi đồng bộ nhiều máy."""
    return uuid.uuid4().hex


def to_iso_str(value: Any) -> str:
    """Chuẩn hóa giá trị thời gian đọc từ CSDL về chuỗi ISO.

    SQLite trả str; SQL Server có thể trả ``datetime`` (cột DATETIME/
    DATETIME2) hoặc ``date`` (cột DATE) — đổi về chuỗi ISO tương ứng để
    phần còn lại của app (so chuỗi, hiển thị, đồng bộ D1) hoạt động như cũ.
    """
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return value.isoformat(sep="T", timespec="microseconds") + "Z"
    if isinstance(value, date):
        return value.isoformat()  # 'YYYY-MM-DD' (cột DATE)
    return str(value)


def _embedding_to_blob(embedding: np.ndarray) -> bytes:
    """Chuyển vector float32 (512,) thành bytes để lưu BLOB."""
    arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if arr.size != EMBEDDING_DIM:
        raise ValueError(
            f"Embedding phải có {EMBEDDING_DIM} chiều, nhận được {arr.size}"
        )
    return arr.tobytes()


def _blob_to_embedding(blob: bytes) -> np.ndarray:
    """Đọc lại vector float32 (512,) từ bytes đã lưu.

    SQL Server: VARBINARY trả ``bytes``; nếu cột được tạo kiểu
    NVARCHAR/VARCHAR lưu chuỗi hex thì chấp nhận thêm (chuyển ngược).
    """
    if isinstance(blob, str):
        blob = bytes.fromhex(blob.removeprefix("0x"))
    elif isinstance(blob, (bytearray, memoryview)):
        blob = bytes(blob)
    return np.frombuffer(blob, dtype=np.float32)


# =============================================================
# Repository
# =============================================================

class PersonRepository:
    """Thao tác bảng ``persons`` (thêm / xem / sửa tên / xóa)."""

    def __init__(self, db: Database) -> None:
        self._db = db

    # -- Ghi -------------------------------------------------
    def add(
        self,
        name: str,
        thumbnail_path: str,
        person_id: str | None = None,
        created_at: str | None = None,
        thumbnail_r2_key: str | None = None,
        record_outbox: bool = True,
        department: str = "",
        position: str = "",
    ) -> Person:
        """Thêm người mới, trả về đối tượng Person đã lưu (kèm id).

        ``person_id``/``created_at``: chỉ dùng khi KÉO dữ liệu từ cloud về
        (phải giữ nguyên id để không trùng lặp — Bước 15). ``record_outbox``
        = False khi thao tác do chính sync tạo ra (tránh vòng lặp đẩy lại).
        ``thumbnail_r2_key``: key ảnh đại diện trên R2 khi kéo người từ
        cloud về (Bước 16) — lưu ngay để lần sync sau không upload lại.
        ``department``/``position``: phòng ban + chức vụ nhân viên
        (v3) — rỗng = chưa khai.
        """
        person = Person(
            id=person_id or _new_id(),
            name=name.strip(),
            created_at=created_at or "",  # rỗng → INSERT tự sinh UTC ISO
            thumbnail_path=thumbnail_path,
            thumbnail_r2_key=thumbnail_r2_key,
            department=department.strip(),
            position=position.strip(),
        )
        with self._db.session() as conn:
            conn.execute(
                "INSERT INTO persons (id, name, created_at, thumbnail_path,"
                " thumbnail_r2_key, department, position)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    person.id,
                    person.name,
                    # SQL Server không có strftime() của SQLite — app tự
                    # sinh timestamp ISO UTC; db.ts() bỏ hậu tố 'Z' khi cột
                    # là DATETIME (SQL Server không cast được chuỗi có 'Z').
                    self._db.ts("persons", "created_at",
                                person.created_at or _utc_now_iso()),
                    person.thumbnail_path,
                    person.thumbnail_r2_key,
                    person.department,
                    person.position,
                ),
            )
            row = conn.execute(
                "SELECT * FROM persons WHERE id = ?", (person.id,)
            ).fetchone()
        if record_outbox:
            self._outbox().add("person", person.id, "upsert")
        logger.info("Đã thêm người '%s' (id=%s)", person.name, person.id)
        return _row_to_person(row)

    def set_thumbnail_r2_key(self, person_id: str, r2_key: str) -> bool:
        """Ghi key ảnh trên R2 sau khi đã upload thành công (Bước 16).

        KHÔNG ghi outbox — key là kết quả của chính quá trình sync, ghi
        outbox sẽ tạo vòng lặp đẩy lại. Lần sync sau đọc từ DB và thấy
        key đã có → bỏ qua upload.
        """
        with self._db.session() as conn:
            cur = conn.execute(
                "UPDATE persons SET thumbnail_r2_key = ? WHERE id = ?",
                (r2_key, person_id),
            )
        return cur.rowcount > 0

    def update_thumbnail(
        self, person_id: str, thumbnail_path: str, record_outbox: bool = True
    ) -> bool:
        """Cập nhật đường dẫn ảnh đại diện (gọi sau khi đã lưu file thumbnail)."""
        with self._db.session() as conn:
            cur = conn.execute(
                "UPDATE persons SET thumbnail_path = ? WHERE id = ?",
                (thumbnail_path, person_id),
            )
        if cur.rowcount and record_outbox:
            self._outbox().add("person", person_id, "upsert")
        return cur.rowcount > 0

    def rename(
        self, person_id: str, new_name: str, record_outbox: bool = True
    ) -> bool:
        """Đổi tên người. Trả về True nếu tìm thấy và đã đổi."""
        new_name = new_name.strip()
        if not new_name:
            return False
        with self._db.session() as conn:
            cur = conn.execute(
                "UPDATE persons SET name = ? WHERE id = ?",
                (new_name, person_id),
            )
        if cur.rowcount and record_outbox:
            self._outbox().add("person", person_id, "upsert")
        if cur.rowcount:
            logger.info("Đã đổi tên người %s → '%s'", person_id, new_name)
        return cur.rowcount > 0

    def update_info(
        self,
        person_id: str,
        name: str,
        department: str = "",
        position: str = "",
        record_outbox: bool = True,
    ) -> bool:
        """Cập nhật tên + phòng ban + chức vụ (v3). Trả True nếu có dòng được đổi."""
        name = name.strip()
        if not name:
            return False
        with self._db.session() as conn:
            cur = conn.execute(
                "UPDATE persons SET name = ?, department = ?, position = ?"
                " WHERE id = ?",
                (name, department.strip(), position.strip(), person_id),
            )
        if cur.rowcount and record_outbox:
            self._outbox().add("person", person_id, "upsert")
        if cur.rowcount:
            logger.info(
                "Đã cập nhật người %s: '%s' · %s · %s",
                person_id, name, department, position,
            )
        return cur.rowcount > 0

    def delete(self, person_id: str, record_outbox: bool = True) -> bool:
        """Xóa người — các face_samples tự xóa (ON DELETE CASCADE).

        Ghi chú: recognition_events giữ lại với person_id = NULL
        (ON DELETE SET NULL — không làm mất lịch sử).
        """
        with self._db.session() as conn:
            cur = conn.execute("DELETE FROM persons WHERE id = ?", (person_id,))
        if cur.rowcount and record_outbox:
            self._outbox().add("person", person_id, "delete")
        if cur.rowcount:
            logger.info("Đã xóa người %s (kèm face_samples)", person_id)
        return cur.rowcount > 0

    def _outbox(self) -> SyncOutboxRepository:
        return SyncOutboxRepository(self._db)

    # -- Đọc -------------------------------------------------
    def get(self, person_id: str) -> Person | None:
        """Lấy một người theo id; None nếu không tồn tại."""
        with self._db.session() as conn:
            row = conn.execute(
                "SELECT * FROM persons WHERE id = ?", (person_id,)
            ).fetchone()
        return _row_to_person(row) if row else None

    def list_all(self) -> list[Person]:
        """Danh sách toàn bộ người, sắp theo thời gian tạo mới nhất."""
        with self._db.session() as conn:
            rows = conn.execute(
                "SELECT * FROM persons ORDER BY created_at DESC"
            ).fetchall()
        return [_row_to_person(r) for r in rows]

    def find_by_name(self, name: str) -> Person | None:
        """Tìm người theo TÊN (không phân biệt hoa/thường) — None nếu chưa có.

        So khớp trong Python (bỏ khoảng trắng + lower) để hành vi GIỐNG NHAU
        trên cả SQLite (phân biệt hoa/thường) lẫn SQL Server (collation không
        phân biệt) — dùng khi đăng ký hàng loạt: trùng tên → thêm mẫu vào
        hồ sơ cũ thay vì tạo trùng.
        """
        key = name.strip().lower()
        if not key:
            return None
        with self._db.session() as conn:
            rows = conn.execute("SELECT * FROM persons").fetchall()
        for r in rows:
            p = _row_to_person(r)
            if p.name.strip().lower() == key:
                return p
        return None

    def count(self) -> int:
        """Tổng số người đã đăng ký."""
        with self._db.session() as conn:
            row = conn.execute("SELECT COUNT(*) FROM persons").fetchone()
        return int(row[0])


class FaceSampleRepository:
    """Thao tác bảng ``face_samples`` (mẫu embedding của từng người)."""

    def __init__(self, db: Database) -> None:
        self._db = db

    # -- Ghi -------------------------------------------------
    def add(
        self,
        person_id: str,
        embedding: np.ndarray,
        quality: float = 1.0,
        sample_id: str | None = None,
        captured_at: str | None = None,
        record_outbox: bool = True,
    ) -> FaceSample:
        """Thêm một mẫu embedding cho người, trả về đối tượng đã lưu.

        ``sample_id``/``captured_at``: giữ nguyên id khi KÉO từ cloud về
        (Bước 15); ``record_outbox=False`` cho thao tác do sync tạo.
        """
        sample = FaceSample(
            id=sample_id or _new_id(),
            person_id=person_id,
            embedding=np.asarray(embedding, dtype=np.float32),
            quality=float(quality),
            captured_at=captured_at or "",
        )
        blob = _embedding_to_blob(sample.embedding)
        # Tự thích ứng kiểu cột embedding: VARBINARY nhận bytes trực tiếp;
        # nếu bảng được tạo với NVARCHAR/VARCHAR thì lưu chuỗi hex (đọc lại
        # _blob_to_embedding tự chuyển ngược — chấp nhận cả 2 định dạng).
        col_type = (self._db.column_type("face_samples", "embedding") or "").lower()
        emb_param: Any = blob.hex() if ("varchar" in col_type or col_type in ("text", "ntext")) else blob
        with self._db.session() as conn:
            conn.execute(
                "INSERT INTO face_samples (id, person_id, embedding, quality, captured_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (sample.id, sample.person_id, emb_param, sample.quality,
                 self._db.ts("face_samples", "captured_at",
                             sample.captured_at or _utc_now_iso())),
            )
            row = conn.execute(
                "SELECT * FROM face_samples WHERE id = ?", (sample.id,)
            ).fetchone()
        if record_outbox:
            self._outbox().add("face_sample", sample.id, "upsert")
        logger.debug("Đã thêm mẫu embedding %s cho người %s", sample.id, person_id)
        return _row_to_sample(row)

    def get(self, sample_id: str) -> FaceSample | None:
        """Lấy một mẫu embedding theo id; None nếu không tồn tại."""
        with self._db.session() as conn:
            row = conn.execute(
                "SELECT * FROM face_samples WHERE id = ?", (sample_id,)
            ).fetchone()
        return _row_to_sample(row) if row else None

    def _outbox(self) -> SyncOutboxRepository:
        return SyncOutboxRepository(self._db)

    # -- Đọc -------------------------------------------------
    def list_by_person(self, person_id: str) -> list[FaceSample]:
        """Tất cả mẫu embedding của một người."""
        with self._db.session() as conn:
            rows = conn.execute(
                "SELECT * FROM face_samples WHERE person_id = ?"
                " ORDER BY captured_at ASC",
                (person_id,),
            ).fetchall()
        return [_row_to_sample(r) for r in rows]

    def count_by_person(self, person_id: str) -> int:
        """Số mẫu embedding của một người."""
        with self._db.session() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM face_samples WHERE person_id = ?",
                (person_id,),
            ).fetchone()
        return int(row[0])

    def all_samples(self) -> list[FaceSample]:
        """Toàn bộ mẫu embedding của mọi người (dùng cho so khớp Bước 6)."""
        with self._db.session() as conn:
            rows = conn.execute("SELECT * FROM face_samples").fetchall()
        return [_row_to_sample(r) for r in rows]


class RecognitionEventRepository:
    """Thao tác bảng ``recognition_events`` (lịch sử nhận diện).

    Bước 9: ghi sự kiện khi nhận diện real-time (kèm snapshot).
    Bước 11 (HistoryView) sẽ mở rộng thêm truy vấn tìm kiếm.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    # -- Ghi -------------------------------------------------
    def add(
        self,
        person_id: str | None,
        label: str,
        source: str,
        similarity: float | None = None,
        snapshot_path: str = "",
        is_unknown: bool = False,
        event_id: str | None = None,
        detected_at: str | None = None,
        snapshot_r2_key: str | None = None,
        record_outbox: bool = True,
        mode: str = "",
    ) -> str:
        """Ghi một sự kiện nhận diện; trả về id sự kiện vừa tạo.

        person_id = None + is_unknown = 1 khi gặp người lạ.
        ``event_id``/``detected_at``: giữ nguyên khi KÉO từ cloud về
        (Bước 15); ``record_outbox=False`` cho thao tác do sync tạo.
        ``snapshot_r2_key``: key ảnh chụp trên R2 khi kéo sự kiện từ cloud
        (Bước 16) — lưu ngay để lần sync sau không upload lại.
        ``mode``: v4 — loại quét VÀO/RA CA ('checkin'/'checkout', rỗng =
        không phải quét điểm danh) để Lịch sử quét lọc/hiển thị riêng.
        """
        event_id = event_id or _new_id()
        with self._db.session() as conn:
            conn.execute(
                "INSERT INTO recognition_events"
                " (id, person_id, label, source, similarity, snapshot_path,"
                " is_unknown, detected_at, snapshot_r2_key, mode)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event_id,
                    person_id,
                    label,
                    source,
                    similarity,
                    snapshot_path,
                    1 if is_unknown else 0,
                    self._db.ts("recognition_events", "detected_at",
                                detected_at or _utc_now_iso()),
                    snapshot_r2_key,
                    mode,
                ),
            )
        if record_outbox:
            self._outbox().add("recognition_event", event_id, "upsert")
        logger.debug("Đã ghi sự kiện nhận diện %s (%s)", event_id, label)
        return event_id

    def set_snapshot_r2_key(self, event_id: str, r2_key: str) -> bool:
        """Ghi key ảnh snapshot trên R2 sau khi upload thành công (Bước 16).

        KHÔNG ghi outbox — key là kết quả của chính quá trình sync (lý do
        giống set_thumbnail_r2_key của PersonRepository).
        """
        with self._db.session() as conn:
            cur = conn.execute(
                "UPDATE recognition_events SET snapshot_r2_key = ? WHERE id = ?",
                (r2_key, event_id),
            )
        return cur.rowcount > 0

    def list_all(self) -> list[RecognitionEvent]:
        """Toàn bộ sự kiện, mới nhất trước (dùng cho đợt đồng bộ đầu tiên)."""
        with self._db.session() as conn:
            rows = conn.execute(
                "SELECT * FROM recognition_events ORDER BY detected_at ASC"
            ).fetchall()
        return [_row_to_event(r) for r in rows]

    def _outbox(self) -> SyncOutboxRepository:
        return SyncOutboxRepository(self._db)

    # -- Đọc -------------------------------------------------
    def last_detected_at(self, person_id: str) -> str | None:
        """Thời điểm nhận diện gần nhất của người (ISO); None nếu chưa từng.

        Dialect: SQL Server dùng SELECT TOP 1; SQLite (script test) dùng
        LIMIT 1 (không hỗ trợ TOP).
        """
        with self._db.session() as conn:
            if self._db.is_sqlite:
                sql = (
                    "SELECT detected_at FROM recognition_events"
                    " WHERE person_id = ? ORDER BY detected_at DESC LIMIT 1"
                )
            else:
                sql = (
                    "SELECT TOP 1 detected_at FROM recognition_events"
                    " WHERE person_id = ? ORDER BY detected_at DESC"
                )
            row = conn.execute(sql, (person_id,)).fetchone()
        return to_iso_str(row["detected_at"]) if row else None

    def get(self, event_id: str) -> RecognitionEvent | None:
        """Lấy một sự kiện theo id; None nếu không tồn tại."""
        with self._db.session() as conn:
            row = conn.execute(
                "SELECT * FROM recognition_events WHERE id = ?", (event_id,)
            ).fetchone()
        return _row_to_event(row) if row else None

    @staticmethod
    def _status_where(status: str) -> tuple[str, list[Any]]:
        """Điều kiện SQL cho bộ lọc trạng thái sự kiện (Bước 18 mở rộng).

        - ``blocked``: sự kiện bị CHẶN nhận diện — Giả mạo (B17) hoặc Mặt
          bị che (B18) — nhận biết qua tiền tố label (xem HistoryView).
        - ``unknown``: người lạ (is_unknown = 1).
        - ``confirmed``: nhận diện thành công (không phải người lạ, không
          bị chặn).
        Trả về (điều kiện SQL, tham số) — '' + [] khi status rỗng (tất cả).
        """
        if status == "blocked":
            # Chuỗi literal '{tiếng Việt}:%' truyền dạng THAM SỐ (không nhúng
            # vào câu SQL) — an toàn encoding với driver ODBC.
            return " AND (label LIKE ? OR label LIKE ?)", ["Giả mạo:%", "Mặt bị che:%"]
        if status == "unknown":
            return " AND is_unknown = 1", []
        if status == "confirmed":
            return (
                " AND is_unknown = 0"
                " AND label NOT LIKE ?"
                " AND label NOT LIKE ?",
                ["Giả mạo:%", "Mặt bị che:%"],
            )
        return "", []

    def list_events(
        self,
        query: str = "",
        source: str = "",
        status: str = "",
        since: str | None = None,
        limit: int = 50,
        offset: int = 0,
        mode: str = "",
    ) -> list[RecognitionEvent]:
        """Danh sách sự kiện MỚI NHẤT TRƯỚC (Bước 11 — HistoryView).

        Bộ lọc (kết hợp được với nhau):
        - ``query``: khớp tên hiển thị (label) — không phân biệt hoa thường
        - ``source``: '' = tất cả, ngược lại khớp chính xác (webcam/photo/mobile)
        - ``status``: '' = tất cả, 'blocked'/'unknown'/'confirmed' (xem
          _status_where) — lọc nhanh các lần bị CHẶN (Giả mạo / Mặt bị che)
        - ``mode``: v4 — 'checkin'/'checkout' (quét VÀO/RA CA), '' = tất cả
        - ``since``: mốc thời gian ISO — chỉ lấy sự kiện từ mốc này trở đi
        - ``limit`` / ``offset``: phân trang
        """
        sql = "SELECT * FROM recognition_events WHERE 1=1"
        params: list[Any] = []
        if query:
            # SQL Server: collation mặc định đã không phân biệt hoa/thường
            # → bỏ COLLATE NOCASE của SQLite.
            sql += " AND label LIKE ?"
            params.append(f"%{query}%")
        if source:
            sql += " AND source = ?"
            params.append(source)
        status_sql, status_params = self._status_where(status)
        sql += status_sql
        params.extend(status_params)
        if mode:
            sql += " AND mode = ?"
            params.append(mode)
        if since:
            sql += " AND detected_at >= ?"
            params.append(since)
        # Dialect phân trang: SQL Server dùng OFFSET...FETCH (bắt buộc có
        # ORDER BY); SQLite dùng LIMIT ? OFFSET ? (không hỗ trợ FETCH).
        if self._db.is_sqlite:
            sql += " ORDER BY detected_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
        else:
            sql += " ORDER BY detected_at DESC OFFSET ? ROWS FETCH NEXT ? ROWS ONLY"
            params.extend([offset, limit])
        with self._db.session() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_event(r) for r in rows]

    def count_events(
        self,
        query: str = "",
        source: str = "",
        status: str = "",
        since: str | None = None,
        mode: str = "",
    ) -> int:
        """Đếm sự kiện theo CÙNG bộ lọc của list_events (dùng cho phân trang)."""
        sql = "SELECT COUNT(*) FROM recognition_events WHERE 1=1"
        params: list[Any] = []
        if query:
            sql += " AND label LIKE ?"
            params.append(f"%{query}%")
        if source:
            sql += " AND source = ?"
            params.append(source)
        status_sql, status_params = self._status_where(status)
        sql += status_sql
        params.extend(status_params)
        if mode:
            sql += " AND mode = ?"
            params.append(mode)
        if since:
            sql += " AND detected_at >= ?"
            params.append(since)
        with self._db.session() as conn:
            row = conn.execute(sql, params).fetchone()
        return int(row[0])

    # -- Xóa -------------------------------------------------
    def delete(self, event_id: str, record_outbox: bool = True) -> bool:
        """Xóa một sự kiện; trả True nếu tìm thấy và đã xóa."""
        with self._db.session() as conn:
            cur = conn.execute(
                "DELETE FROM recognition_events WHERE id = ?", (event_id,)
            )
        if cur.rowcount and record_outbox:
            self._outbox().add("recognition_event", event_id, "delete")
        return cur.rowcount > 0

    def count_today(self) -> int:
        """Số sự kiện nhận diện từ đầu ngày HÔM NAY (giờ địa phương).

        detected_at lưu theo UTC (hậu tố 'Z') → đổi mốc 00:00 giờ địa
        phương sang UTC rồi so chuỗi ISO (cùng định dạng, so được).
        """
        from datetime import datetime, timezone

        midnight_local = datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        ).astimezone()
        midnight_utc = midnight_local.astimezone(timezone.utc)
        boundary = midnight_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")
        with self._db.session() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM recognition_events WHERE detected_at >= ?",
                (boundary,),
            ).fetchone()
        return int(row[0])


# =============================================================
# SyncOutboxRepository — hàng đợi đồng bộ cloud (Bước 15, outbox pattern)
# =============================================================

class SyncOutboxRepository:
    """Hàng đợi "cần đồng bộ lên cloud".

    Mọi thao tác GHI local (thêm/sửa/xóa người, mẫu, sự kiện) ghi 1 dòng
    vào đây; SyncService (Bước 15) đọc các dòng chưa đồng bộ (synced_at
    IS NULL), đẩy lên D1 rồi đánh dấu đã xong. Đây là **outbox pattern**:
    local ghi trước (offline OK), cloud đồng bộ sau khi có mạng.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    def add(self, entity: str, entity_id: str, op: str) -> None:
        """Ghi 1 thao tác cần đồng bộ (upsert / delete).

        Nếu đã có dòng CHƯA đồng bộ cho cùng thực thể → xóa dòng cũ rồi
        ghi dòng mới (chỉ giữ thao tác MỚI NHẤT — đổi tên 3 lần trước khi
        sync chỉ cần đẩy 1 lần với tên cuối cùng).
        """
        with self._db.session() as conn:
            conn.execute(
                "DELETE FROM sync_outbox"
                " WHERE entity = ? AND entity_id = ? AND synced_at IS NULL",
                (entity, entity_id),
            )
            # created_at truyền TƯỜNG MINH (không dựa vào DEFAULT của bảng
            # — bảng SQL Server do người dùng tạo, có thể không có DEFAULT).
            conn.execute(
                "INSERT INTO sync_outbox (id, entity, entity_id, op, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (_new_id(), entity, entity_id, op,
                 self._db.ts("sync_outbox", "created_at", _utc_now_iso())),
            )

    def list_pending(self) -> list[tuple[str, str, str, str]]:
        """Các thao tác chưa đồng bộ: (outbox_id, entity, entity_id, op)."""
        with self._db.session() as conn:
            rows = conn.execute(
                "SELECT id, entity, entity_id, op FROM sync_outbox"
                " WHERE synced_at IS NULL ORDER BY created_at ASC"
            ).fetchall()
        return [(r["id"], r["entity"], r["entity_id"], r["op"]) for r in rows]

    def mark_synced(self, outbox_ids: list[str], synced_at: str) -> int:
        """Đánh dấu các dòng đã đẩy lên cloud thành công; trả về số dòng.

        synced_at được chuẩn hóa theo kiểu cột thực tế (db.ts — bỏ 'Z'
        khi cột là DATETIME trên SQL Server).
        """
        if not outbox_ids:
            return 0
        placeholders = ",".join("?" for _ in outbox_ids)
        with self._db.session() as conn:
            cur = conn.execute(
                f"UPDATE sync_outbox SET synced_at = ? WHERE id IN ({placeholders})",
                [self._db.ts("sync_outbox", "synced_at", synced_at), *outbox_ids],
            )
        return cur.rowcount

    def count_pending(self) -> int:
        """Số thao tác đang chờ đồng bộ."""
        with self._db.session() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM sync_outbox WHERE synced_at IS NULL"
            ).fetchone()
        return int(row[0])


# =============================================================
# Hàm phụ chuyển hàng SQL → dataclass
# =============================================================

def _row_to_person(row: Any) -> Person:
    return Person(
        id=row["id"],
        name=row["name"],
        created_at=to_iso_str(row["created_at"]),
        thumbnail_path=row["thumbnail_path"] or "",
        thumbnail_r2_key=row["thumbnail_r2_key"],
        # Nhánh "or": DB cũ chưa migration (không có cột) → sqlite3.Row
        # truy cập key thiếu sẽ raise; dùng get-an-log thôi không đủ →
        # kiểm tra presence qua keys() cho an toàn cả 2 giai đoạn.
        department=row["department"] if "department" in row.keys() else "",
        position=row["position"] if "position" in row.keys() else "",
    )


def _row_to_sample(row: Any) -> FaceSample:
    return FaceSample(
        id=row["id"],
        person_id=row["person_id"],
        embedding=_blob_to_embedding(row["embedding"]),
        # Cột REAL/DECIMAL của SQL Server có thể trả decimal.Decimal → float
        quality=float(row["quality"]),
        captured_at=to_iso_str(row["captured_at"]),
    )


def _row_to_event(row: Any) -> RecognitionEvent:
    return RecognitionEvent(
        id=row["id"],
        person_id=row["person_id"],
        label=row["label"],
        source=row["source"],
        detected_at=to_iso_str(row["detected_at"]),
        # REAL/DECIMAL → float (SQL Server trả decimal.Decimal cho DECIMAL)
        similarity=float(row["similarity"]) if row["similarity"] is not None else None,
        snapshot_path=row["snapshot_path"] or "",
        snapshot_r2_key=row["snapshot_r2_key"],
        is_unknown=bool(row["is_unknown"]),
        # DB cũ chưa migration v4 thì không có cột mode → rỗng
        mode=row["mode"] if "mode" in row.keys() else "",
    )
