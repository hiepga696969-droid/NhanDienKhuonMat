"""Kết nối CSDL + kiểm tra lược đồ — SQL Server (pyodbc) cho app chính.

BACKEND:
- App chính dùng **SQL Server** (Windows Authentication, ODBC Driver 18):
  gọi ``Database()`` KHÔNG truyền path. Bảng đã được tạo sẵn trong
  FaceRecognitionDB (persons, face_samples, recognition_events,
  sync_outbox, settings, attendance) — app CHỈ KẾT NỐI và KIỂM TRA,
  KHÔNG tạo bảng mới, KHÔNG xóa/sửa bảng hiện có (yêu cầu người dùng).
- Các script kiểm thử (scripts/step_*.py, test_*.py...) vẫn dùng SQLite
  bằng cách truyền đường dẫn file: ``Database(TEMP_DB)`` — giữ nguyên
  hành vi cũ (PRAGMA WAL + tự tạo schema v4) để không phải sửa test.

SQL DIALECT (đã chuyển sang SQL Server ở tầng repository/service):
- ``strftime('%Y-%m-%dT%H:%M:%fZ', 'now')`` (DEFAULT của SQLite) → app
  tự sinh timestamp ISO UTC ở tầng Python (sqlserver_db.utc_now_iso);
  ``db.ts()`` tự bỏ hậu tố 'Z' khi cột là DATETIME (SQL Server không
  cast được chuỗi có 'Z' sang datetime).
- ``LIMIT ? OFFSET ?`` → ``OFFSET ? ROWS FETCH NEXT ? ROWS ONLY``.
- ``SELECT ... LIMIT 1`` → ``SELECT TOP 1 ...``.
- ``COLLATE NOCASE`` (SQLite) → collation SQL Server mặc định đã không
  phân biệt hoa thường → bỏ mệnh đề.
- ``ON CONFLICT ... DO UPDATE`` (chạy local trên bảng settings) →
  UPDATE trước, INSERT khi chưa có (SQL Server không hỗ trợ MERGE an toàn
  với tham số ?).

Định dạng thời gian: app luôn ghi/đọc chuỗi ISO ``YYYY-MM-DDTHH:MM:SS.ffffff``
(và ``...Z`` khi DEFAULT của DB sinh ra) — so chuỗi ISO vẫn đúng như cũ.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from app.config import APP_DIR
from app.infrastructure.sqlserver_db import get_connection

logger = logging.getLogger(__name__)

# -------------------------------------------------------------
# Đường dẫn & phiên bản schema (backend SQLite cho script test)
# -------------------------------------------------------------
# Dữ liệu người dùng đặt ở APP_DIR (cạnh .exe khi đã đóng gói — Bước 12)
DATA_DIR = APP_DIR / "data"
DB_PATH = DATA_DIR / "app.db"

SCHEMA_VERSION = 5  # tăng khi có thay đổi cấu trúc bảng (PRAGMA user_version)
# v3: persons.department/position
# v4: attendance.checked_out + note_in/note_out (ghi chú riêng vào/ra ca);
#     recognition_events.mode ('checkin'/'checkout' — lọc Lịch sử quét)
# v5: attendance NHIỀU PHIÊN trong ngày — thêm session_no (số thứ tự
#     phiên), BỎ UNIQUE(person_id, work_date) để 1 người có nhiều dòng
#     trong ngày (vào trưa ra, vào lại chiều = 2 phiên đều tính công).
#     SQL Server cũ: ALTER TABLE attendance ADD session_no INT NOT NULL DEFAULT 1;

# Các bảng + cột BẮT BUỘC phải có sẵn trong SQL Server (FaceRecognitionDB).
# App không tự tạo — thiếu thì báo lỗi rõ ràng để người dùng tạo trong SSMS.
REQUIRED_TABLES: dict[str, set[str]] = {
    "persons": {
        "id", "name", "created_at", "thumbnail_path", "thumbnail_r2_key",
        "department", "position",
    },
    "face_samples": {
        "id", "person_id", "embedding", "quality", "captured_at",
    },
    "recognition_events": {
        "id", "person_id", "label", "source", "detected_at", "similarity",
        "snapshot_path", "snapshot_r2_key", "is_unknown", "mode",
    },
    "sync_outbox": {
        "id", "entity", "entity_id", "op", "created_at", "synced_at",
    },
    "settings": {"key", "value"},
    "attendance": {
        "id", "person_id", "work_date", "session_no", "check_in", "check_out",
        "checked_out", "note_in", "note_out", "last_seen", "seen_count",
    },
}

# Khóa toàn cục tuần tự hóa giao dịch giữa các THREAD (Bước 15).
# SyncService chạy trong QThread (không đơ UI) nhưng dùng chung connection
# với main thread → mọi transaction phải qua session() và nằm trong khóa
# này để không interleave lẫn nhau (BEGIN/COMMIT không được lồng nhau).
_DB_LOCK = threading.RLock()


# -------------------------------------------------------------
# Lược đồ CSDL SQLite — CHỈ dùng cho backend SQLite (script test).
# Giữ NGUYÊN cú pháp như spec 5.4 (dùng chung D1).
# -------------------------------------------------------------
SCHEMA_SQL = """
-- 1) persons — người đã đăng ký khuôn mặt
CREATE TABLE IF NOT EXISTS persons (
    id               TEXT PRIMARY KEY,  -- UUID v4 (hex) — KHÔNG dùng AUTOINCREMENT
    name             TEXT NOT NULL CHECK (length(trim(name)) > 0),
    created_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    thumbnail_path   TEXT NOT NULL,     -- đường dẫn ảnh đại diện (local)
    thumbnail_r2_key TEXT               -- NULL = chưa upload lên R2
);

-- 2) face_samples — các mẫu embedding (3–5 mẫu/người)
CREATE TABLE IF NOT EXISTS face_samples (
    id          TEXT PRIMARY KEY,       -- UUID v4
    person_id   TEXT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    embedding   BLOB NOT NULL,          -- 512 × float32 = 2048 bytes
    quality     REAL NOT NULL DEFAULT 0.0 CHECK (quality >= 0.0 AND quality <= 1.0),
    captured_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_face_samples_person ON face_samples(person_id);

-- 3) recognition_events — lịch sử nhận diện (audit trail, kèm ảnh snapshot)
CREATE TABLE IF NOT EXISTS recognition_events (
    id              TEXT PRIMARY KEY,   -- UUID v4
    person_id       TEXT REFERENCES persons(id) ON DELETE SET NULL,
    label           TEXT NOT NULL,      -- tên người, hoặc 'Người lạ' nếu is_unknown = 1
    source          TEXT NOT NULL CHECK (source IN ('webcam', 'photo', 'mobile')),
    detected_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    similarity      REAL,               -- điểm tương đồng cao nhất (0..1)
    snapshot_path   TEXT,               -- ảnh chụp local
    snapshot_r2_key TEXT,               -- NULL = chưa upload lên R2
    is_unknown      INTEGER NOT NULL DEFAULT 0 CHECK (is_unknown IN (0, 1)),
    mode            TEXT NOT NULL DEFAULT ''  -- v4: 'checkin'/'checkout' (loại quét)
);
CREATE INDEX IF NOT EXISTS idx_events_detected_at ON recognition_events(detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_person      ON recognition_events(person_id);
CREATE INDEX IF NOT EXISTS idx_events_source      ON recognition_events(source);

-- 4) sync_outbox — hàng đợi đồng bộ cloud (outbox pattern, dùng ở Bước 15)
CREATE TABLE IF NOT EXISTS sync_outbox (
    id         TEXT PRIMARY KEY,        -- UUID v4
    entity     TEXT NOT NULL CHECK (entity IN ('person', 'face_sample', 'recognition_event')),
    entity_id  TEXT NOT NULL,
    op         TEXT NOT NULL CHECK (op IN ('upsert', 'delete')),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    synced_at  TEXT                     -- NULL = chưa đồng bộ
);
-- Partial index: chỉ quét hàng chưa đồng bộ — nhanh cho SyncService
CREATE INDEX IF NOT EXISTS idx_outbox_pending
    ON sync_outbox(synced_at) WHERE synced_at IS NULL;

-- 5) settings — khóa-giá trị (ngưỡng, camera idx, ... — bổ sung cho config.json)
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- 6) attendance — điểm danh nhân viên (v5: MỖI PHIÊN một dòng)
--    v5 nghiệp vụ NHIỀU PHIÊN VÀO/RA TRONG NGÀY:
--    - VÀO CA khi không có phiên mở → INSERT dòng mới (check_out NULL).
--    - RA CA → UPDATE dòng CHƯA RA GẦN NHẤT (check_out + checked_out = 1).
--    - Về trưa vào lại chiều = 2 dòng trong cùng ngày (bỏ UNIQUE
--      person_id + work_date của v4) — đều được tính công.
--    - session_no: số thứ tự phiên trong ngày (1, 2, ...).
CREATE TABLE IF NOT EXISTS attendance (
    id           TEXT PRIMARY KEY,      -- UUID v4
    person_id    TEXT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    work_date    TEXT NOT NULL,         -- 'YYYY-MM-DD' giờ ĐỊA PHƯƠNG
    session_no   INTEGER NOT NULL DEFAULT 1,  -- v5: phiên thứ mấy trong ngày
    check_in     TEXT NOT NULL,         -- ISO UTC lần VÀO CA (giờ địa phương lưu UTC)
    check_out    TEXT,                  -- ISO UTC lần RA CA (NULL = phiên đang mở)
    checked_out  INTEGER NOT NULL DEFAULT 0,  -- 1 = phiên đã xác nhận RA CA
    note_in      TEXT,                  -- ghi chú khi VÀO CA
    note_out     TEXT,                  -- ghi chú khi RA CA
    last_seen    TEXT NOT NULL,         -- ISO UTC — cập nhật mỗi lần nhận diện
    seen_count   INTEGER NOT NULL DEFAULT 1   -- số lần nhận diện của phiên
);
CREATE INDEX IF NOT EXISTS idx_attendance_date   ON attendance(work_date DESC);
CREATE INDEX IF NOT EXISTS idx_attendance_person ON attendance(person_id);
"""

# Phần schema bổ sung của v2 — chạy RIÊNG khi migration v1→v2 (DB cũ đã có
# các bảng 1-5 rồi thì executescript SCHEMA_SQL với CREATE TABLE IF NOT
# EXISTS vẫn chạy được, nhưng tách ra cho rõ ràng + nhanh).
ATTENDANCE_SQL = """
CREATE TABLE IF NOT EXISTS attendance (
    id           TEXT PRIMARY KEY,      -- UUID v4
    person_id    TEXT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    work_date    TEXT NOT NULL,         -- 'YYYY-MM-DD' giờ ĐỊA PHƯƠNG
    check_in     TEXT NOT NULL,         -- ISO UTC lần VÀO CA
    check_out    TEXT,                  -- ISO UTC lần RA CA (NULL = chưa ra ca)
    checked_out  INTEGER NOT NULL DEFAULT 0,  -- 1 = đã xác nhận RA CA
    note_in      TEXT,                  -- ghi chú khi VÀO CA
    note_out     TEXT,                  -- ghi chú khi RA CA
    last_seen    TEXT NOT NULL,         -- ISO UTC — cập nhật mỗi lần nhận diện
    seen_count   INTEGER NOT NULL DEFAULT 1,  -- số lần nhận diện trong ngày
    UNIQUE (person_id, work_date)
);
CREATE INDEX IF NOT EXISTS idx_attendance_date   ON attendance(work_date DESC);
CREATE INDEX IF NOT EXISTS idx_attendance_person ON attendance(person_id);
"""


# =============================================================
# Lớp Row kiểu sqlite3.Row cho pyodbc (đọc cột theo TÊN)
# =============================================================
class DictRow(dict):
    """Một dòng kết quả: đọc theo tên (row["name"]) lẫn theo chỉ số (row[0]).

    Thay thế sqlite3.Row cho pyodbc — hỗ trợ ``"cột" in row.keys()``
    mà repository/service đang dùng để kiểm tra cột có tồn tại không.
    """

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            try:
                return dict.__getitem__(self, key)
            except KeyError:
                raise KeyError(f"Không có cột '{key}' trong dòng kết quả") from None
        if isinstance(key, int):
            # Chỉ số nguyên (row[0]) — theo THỨ TỰ cột của câu SELECT
            return list(dict.values(self))[key]
        raise KeyError(f"Key không hợp lệ: {key!r}")

    def keys(self):  # noqa: D102 — giống sqlite3.Row.keys()
        return list(dict.keys(self))


class _DictCursor:
    """Bọc pyodbc.Cursor: fetchone/fetchall trả DictRow (đọc cột theo tên)."""

    def __init__(self, cur: Any) -> None:
        self._cur = cur
        # pyodbc chuyển NVARCHAR thành str, VARBINARY thành bytes,
        # INT thành int — khớp với kiểu sqlite3.Row trả về trước đây.

    def execute(self, sql: str, params: Any = None) -> "_DictCursor":
        if params is None:
            self._cur.execute(sql)
        else:
            self._cur.execute(sql, params)
        return self

    @property
    def rowcount(self) -> int:
        return self._cur.rowcount

    def _wrap(self, row: Any) -> DictRow | None:
        if row is None:
            return None
        cols = [c[0] for c in self._cur.description]
        return DictRow(zip(cols, row))

    def fetchone(self) -> DictRow | None:
        return self._wrap(self._cur.fetchone())

    def fetchall(self) -> list[DictRow]:
        rows = self._cur.fetchall()
        cols = [c[0] for c in self._cur.description]
        return [DictRow(zip(cols, r)) for r in rows]


class _SqlConnection:
    """Bọc pyodbc.Connection trông giống sqlite3.Connection (execute trực tiếp)."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def execute(self, sql: str, params: Any = None) -> _DictCursor:
        cur = self._conn.cursor()
        if params is None:
            cur.execute(sql)
        else:
            cur.execute(sql, params)
        return _DictCursor(cur)

    def cursor(self) -> _DictCursor:
        return _DictCursor(self._conn.cursor())

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def close(self) -> None:
        self._conn.close()


# =============================================================
# Database — facade chọn backend
# =============================================================
class Database:
    """Quản lý kết nối CSDL (SQL Server cho app, SQLite cho script test).

    Cách dùng:
        db = Database()            # SQL Server FaceRecognitionDB (app chính)
        db = Database(TEMP_DB)     # SQLite file tạm (script kiểm thử)
        with db.session() as conn: # tự commit (hoặc rollback nếu có lỗi)
            conn.execute(...)
    """

    def __init__(self, path: Path | None = None) -> None:
        # path=None → SQL Server (app chính); có path → SQLite (script test)
        self._path = path
        self._conn: Any = None
        # (bảng, cột) có kiểu DATETIME/DATETIME2... trên SQL Server — dùng
        # để tự thích ứng định dạng timestamp (có/không hậu tố 'Z').
        self._datetime_cols: set[tuple[str, str]] = set()
        # Kiểu dữ liệu đầy đủ của (bảng, cột) — tra cứu khi cần thích ứng
        # (ví dụ embedding lưu VARBINARY hay NVARCHAR).
        self._column_types: dict[tuple[str, str], str] = {}

    @property
    def path(self) -> Path | str:
        """Đường dẫn file SQLite, hoặc tên database SQL Server."""
        return self._path if self._path is not None else "FaceRecognitionDB (SQL Server)"

    @property
    def is_sqlite(self) -> bool:
        """Đang dùng backend SQLite (script test) hay SQL Server (app chính)."""
        return self._path is not None

    # ---------------------------------------------------------
    # Kết nối
    # ---------------------------------------------------------
    def connect(self) -> Any:
        """Mở kết nối (nếu chưa mở) và kiểm tra/tạo schema khi cần.

        SQL Server: chỉ KIỂM TRA các bảng bắt buộc đã tồn tại (không tạo,
        không xóa, không ALTER — bảng do người dùng quản lý trong SSMS).
        SQLite: mở file, bật PRAGMA + tự tạo/migrate schema như cũ.
        """
        if self._conn is None:
            if self._path is None:
                self._conn = _SqlConnection(get_connection())
                self._verify_schema_sqlserver()
            else:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                # check_same_thread=False: SyncService (QThread) dùng chung kết nối
                # với main thread — an toàn vì session() luôn giữ _DB_LOCK
                conn = sqlite3.connect(str(self._path), check_same_thread=False)
                conn.row_factory = sqlite3.Row  # đọc cột theo tên (row["name"])
                conn.execute("PRAGMA foreign_keys = ON")   # bật khóa ngoại
                conn.execute("PRAGMA journal_mode = WAL")  # ghi bền, đọc/ghi song song
                self._conn = conn
                self._init_schema_sqlite()
        return self._conn

    # ----------------- SQL Server -----------------
    def _verify_schema_sqlserver(self) -> None:
        """Kiểm tra các bảng + cột bắt buộc đã có trong SQL Server.

        KHÔNG tạo/xóa bảng. Thiếu CỘT đã biết (schema v5) → TỰ ADD
        (xem ``_auto_add_sqlserver_columns``); thiếu cột lạ/bảng → báo
        lỗi rõ ràng để người dùng tạo trong SSMS.
        """
        cur = self._conn.cursor()
        cur.execute(
            "SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS"
            " WHERE TABLE_CATALOG = DB_NAME()"
        )
        actual: dict[str, set[str]] = {}
        datetime_types = {"datetime", "datetime2", "smalldatetime", "datetimeoffset"}
        for row in cur.fetchall():
            table, column, dtype = str(row[0]), str(row[1]), str(row[2])
            actual.setdefault(table, set()).add(column)
            self._column_types[(table.lower(), column.lower())] = dtype
            if dtype in datetime_types:
                self._datetime_cols.add((table.lower(), column.lower()))

        missing_tables = [t for t in REQUIRED_TABLES if t not in actual]
        if missing_tables:
            raise RuntimeError(
                "Thiếu bảng trong SQL Server "
                f"({self.path}): {', '.join(missing_tables)}. "
                "Hãy tạo các bảng trong SSMS theo schema dự án (không để app tự tạo)."
            )
        missing_cols = {
            t: sorted(REQUIRED_TABLES[t] - actual.get(t, set()))
            for t in REQUIRED_TABLES
            if REQUIRED_TABLES[t] - actual.get(t, set())
        }
        if missing_cols:
            # Tự nâng cấp: ADD các cột THIẾU đã biết (chỉ THÊM, không sửa/
            # xóa gì — an toàn với dữ liệu có sẵn). Lý do: schema v5 thêm
            # session_no — bắt người dùng chạy SQL tay mỗi lần nâng cấp
            # là phiền; app tự thêm khi có quyền, lỗi quyền mới báo rõ.
            remaining = self._auto_add_sqlserver_columns(missing_cols)
            if remaining:
                detail = "; ".join(
                    f"{t} thiếu {', '.join(c)}" for t, c in remaining.items()
                )
                raise RuntimeError(
                    f"Thiếu cột trong SQL Server ({self.path}): {detail}. "
                    "Tài khoản kết nối không đủ quyền ALTER — hãy chạy lệnh "
                    "ALTER TABLE ... ADD trong SSMS theo schema dự án."
                )
        # v5 nhiều phiên/ngày: bỏ UNIQUE(person_id, work_date) trên
        # attendance nếu còn (không bỏ sẽ INSERT phiên 2 lỗi vi phạm unique).
        self._drop_sqlserver_attendance_unique()
        logger.info("Kết nối SQL Server OK — đủ 6 bảng bắt buộc (%s)", self.path)

    # DDL tạo cột khi tự nâng cấp SQL Server (chỉ ADD COLUMN — không bao
    # giờ sửa/xóa cột có sẵn). Key (bảng, cột) khớp REQUIRED_TABLES.
    _SQLSERVER_COLUMN_DDL: dict[tuple[str, str], str] = {
        ("attendance", "session_no"): "INT NOT NULL DEFAULT 1",
        ("attendance", "checked_out"): "INT NOT NULL DEFAULT 0",
        ("attendance", "note_in"): "NVARCHAR(MAX) NULL",
        ("attendance", "note_out"): "NVARCHAR(MAX) NULL",
        ("persons", "department"): "NVARCHAR(200) NOT NULL DEFAULT ''",
        ("persons", "position"): "NVARCHAR(200) NOT NULL DEFAULT ''",
        ("recognition_events", "mode"): "NVARCHAR(20) NOT NULL DEFAULT ''",
    }

    def _auto_add_sqlserver_columns(
        self, missing_cols: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        """Tự ADD các cột thiếu trên SQL Server (schema v5 tự nâng cấp).

        Chỉ thêm cột ĐÃ ĐỊNH NGHĨA trong ``_SQLSERVER_COLUMN_DDL`` — cột
        lạ trả về trong dict kết quả (caller báo lỗi rõ ràng). Trả về
        dict các cột CÒN THIẾU (không add được).
        """
        remaining: dict[str, list[str]] = {}
        for table, cols in missing_cols.items():
            for col in cols:
                ddl = self._SQLSERVER_COLUMN_DDL.get((table, col))
                if ddl is None:
                    remaining.setdefault(table, []).append(col)
                    continue
                try:
                    cur = self._conn.cursor()
                    cur.execute(f"ALTER TABLE [{table}] ADD [{col}] {ddl}")
                    self._conn.commit()  # DDL trong pyodbc là transactional
                    # Đăng ký kiểu để db.ts()/column_type() hoạt động ngay
                    dtype = ddl.split()[0].upper()
                    self._column_types[(table, col)] = dtype
                    logger.info(
                        "Đã tự thêm cột %s.%s (%s) vào SQL Server — schema v5",
                        table, col, ddl,
                    )
                except Exception:
                    logger.exception(
                        "Không tự thêm được cột %s.%s — cần chạy tay trong SSMS",
                        table, col,
                    )
                    remaining.setdefault(table, []).append(col)
        return remaining

    def _drop_sqlserver_attendance_unique(self) -> None:
        """v5: bỏ UNIQUE(person_id, work_date) trên attendance (SQL Server).

        Nhiều phiên/ngày = nhiều dòng CÙNG person + ngày → ràng buộc UNIQUE
        cũ của v1-v4 sẽ chặn INSERT phiên thứ 2 (lỗi vi phạm unique lúc
        quét VÀO CA lần nữa). Tìm index UNIQUE chứa cả 2 cột và DROP.
        Không có → không làm gì (bảng mới đã không có ràng buộc này).
        """
        try:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT i.name, i.is_unique_constraint"
                " FROM sys.indexes i"
                " WHERE i.object_id = OBJECT_ID('attendance')"
                "   AND i.is_unique = 1 AND i.is_primary_key = 0"
            )
            indexes = cur.fetchall()
        except Exception:
            logger.exception("Không kiểm tra được index bảng attendance")
            return
        # LƯU Ý: DictRow kế thừa dict — unpack trực tiếp sẽ lấy NHẦM key;
        # phải đọc qua chỉ số row[0], row[1] (thứ tự cột của SELECT).
        for idx_row in indexes:
            name = str(idx_row[0]) if idx_row[0] is not None else None
            is_constraint = bool(idx_row[1])
            if name is None:
                continue
            try:
                cur.execute(
                    "SELECT c.name FROM sys.index_columns ic"
                    " JOIN sys.columns c"
                    "   ON c.object_id = ic.object_id AND c.column_id = ic.column_id"
                    " WHERE ic.object_id = OBJECT_ID('attendance')"
                    "   AND ic.index_id = ("
                    "       SELECT i2.index_id FROM sys.indexes i2"
                    "       WHERE i2.object_id = OBJECT_ID('attendance')"
                    "         AND i2.name = ?)",
                    (name,),
                )
                cols = {str(r[0]).lower() for r in cur.fetchall()}
            except Exception:
                logger.exception("Không đọc được cột của index %s", name)
                continue
            if {"person_id", "work_date"} <= cols:
                stmt = (
                    f"ALTER TABLE [attendance] DROP CONSTRAINT [{name}]"
                    if is_constraint
                    else f"DROP INDEX [{name}] ON [attendance]"
                )
                try:
                    cur.execute(stmt)
                    self._conn.commit()  # DDL trong pyodbc là transactional
                    logger.info(
                        "Đã bỏ ràng buộc UNIQUE '%s' trên attendance "
                        "(v5 — hỗ trợ nhiều phiên vào/ra trong ngày)",
                        name,
                    )
                except Exception:
                    logger.exception(
                        "Không bỏ được ràng buộc UNIQUE '%s' — xóa tay trong "
                        "SSMS để quét được nhiều phiên/ngày",
                        name,
                    )

    # ----------------- SQLite (script test) -----------------
    def _init_schema_sqlite(self) -> None:
        """Tạo bảng nếu chưa có; quản lý phiên bản qua PRAGMA user_version.

        Migration TỪNG BƯỚC (mỗi phiên bản một nhánh) — DB cũ mở lên tự
        nâng cấp, không mất dữ liệu:
          v1 → v2: thêm bảng attendance (điểm danh)
          v2 → v3: thêm cột persons.department / persons.position
          v3 → v4: attendance.checked_out + note_in/note_out;
                   recognition_events.mode (loại quét VÀO/RA CA)
          v4 → v5: attendance nhiều phiên/ngày — thêm session_no, BỎ
                   UNIQUE(person_id, work_date) (dựng lại bảng)
        """
        assert isinstance(self._conn, sqlite3.Connection)
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if version < 1:
            self._conn.executescript(SCHEMA_SQL)
        if version < 2:
            self._conn.executescript(ATTENDANCE_SQL)
        if version < 3:
            # ALTER TABLE ... ADD COLUMN giữ nguyên dữ liệu (NULL = chưa khai)
            self._conn.execute(
                "ALTER TABLE persons ADD COLUMN department TEXT NOT NULL DEFAULT ''"
            )
            self._conn.execute(
                "ALTER TABLE persons ADD COLUMN position TEXT NOT NULL DEFAULT ''"
            )
        if version < 4:
            self._migrate_v4()
        if version < 5:
            self._migrate_v5()
        if version < SCHEMA_VERSION:
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self._conn.commit()
            logger.info("Đã nâng cấp CSDL lên schema v%d tại %s", SCHEMA_VERSION, self._path)

    def _migrate_v4(self) -> None:
        """v3 → v4: thêm cột mới nếu CHƯA tồn tại.

        DB mới tạo bằng SCHEMA_SQL (version 0) đã có sẵn các cột này —
        ALTER lại sẽ lỗi 'duplicate column name', nên kiểm tra qua
        PRAGMA table_info trước khi ALTER (an toàn cả 2 trường hợp).
        """
        assert isinstance(self._conn, sqlite3.Connection)
        attendance_cols = self._table_columns("attendance")
        events_cols = self._table_columns("recognition_events")
        if "checked_out" not in attendance_cols:
            self._conn.execute(
                "ALTER TABLE attendance ADD COLUMN checked_out INTEGER NOT NULL DEFAULT 0"
            )
        if "note_in" not in attendance_cols:
            self._conn.execute("ALTER TABLE attendance ADD COLUMN note_in TEXT")
        if "note_out" not in attendance_cols:
            self._conn.execute("ALTER TABLE attendance ADD COLUMN note_out TEXT")
        if "mode" not in events_cols:
            self._conn.execute(
                "ALTER TABLE recognition_events ADD COLUMN mode TEXT NOT NULL DEFAULT ''"
            )

    def _migrate_v5(self) -> None:
        """v4 → v5: attendance hỗ trợ NHIỀU PHIÊN trong ngày.

        - Thêm cột ``session_no`` nếu chưa có (DB cũ).
        - Đánh số phiên theo thứ tự giờ vào trong mỗi ngày.
        - BỎ ràng buộc UNIQUE(person_id, work_date): SQLite không DROP
          constraint được → dựng bảng MỚI không UNIQUE, chép nguyên dữ
          liệu, đổi tên (an toàn, giữ toàn bộ dòng cũ).
        """
        assert isinstance(self._conn, sqlite3.Connection)
        cols = self._table_columns("attendance")
        if not cols:
            return  # bảng chưa tồn tại (DB mới — SCHEMA_SQL đã tạo dạng v5)
        if "session_no" not in cols:
            self._conn.execute(
                "ALTER TABLE attendance ADD COLUMN session_no INTEGER NOT NULL DEFAULT 1"
            )
            # Đánh số phiên: 1, 2, ... theo thứ tự giờ vào trong ngày
            self._conn.execute(
                "UPDATE attendance SET session_no = ("
                "  SELECT COUNT(*) FROM attendance a2"
                "  WHERE a2.person_id = attendance.person_id"
                "    AND a2.work_date = attendance.work_date"
                "    AND a2.check_in <= attendance.check_in)"
            )
        # Dựng lại bảng KHÔNG có UNIQUE(person_id, work_date) — kiểm tra
        # qua sqlite_master (cách duy nhất phát hiện UNIQUE trong SQLite).
        has_unique = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'attendance'"
        ).fetchone()
        if has_unique is not None and "UNIQUE" in (has_unique["sql"] or "").upper():
            self._conn.executescript(
                """
                CREATE TABLE attendance_v5_new (
                    id           TEXT PRIMARY KEY,
                    person_id    TEXT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
                    work_date    TEXT NOT NULL,
                    session_no   INTEGER NOT NULL DEFAULT 1,
                    check_in     TEXT NOT NULL,
                    check_out    TEXT,
                    checked_out  INTEGER NOT NULL DEFAULT 0,
                    note_in      TEXT,
                    note_out     TEXT,
                    last_seen    TEXT NOT NULL,
                    seen_count   INTEGER NOT NULL DEFAULT 1
                );
                INSERT INTO attendance_v5_new
                    (id, person_id, work_date, session_no, check_in, check_out,
                     checked_out, note_in, note_out, last_seen, seen_count)
                SELECT id, person_id, work_date, session_no, check_in, check_out,
                       checked_out, note_in, note_out, last_seen, seen_count
                FROM attendance;
                DROP TABLE attendance;
                ALTER TABLE attendance_v5_new RENAME TO attendance;
                CREATE INDEX IF NOT EXISTS idx_attendance_date
                    ON attendance(work_date DESC);
                CREATE INDEX IF NOT EXISTS idx_attendance_person
                    ON attendance(person_id);
                """
            )
        logger.info("Migration v5 xong — attendance hỗ trợ nhiều phiên/ngày")

    def _table_columns(self, table: str) -> set[str]:
        """Tên các cột hiện có của 1 bảng (PRAGMA table_info)."""
        assert isinstance(self._conn, sqlite3.Connection)
        rows = self._conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {str(r["name"]) for r in rows}

    # ---------------------------------------------------------
    # Tiện ích tự thích ứng kiểu cột (SQL Server)
    # ---------------------------------------------------------
    def column_type(self, table: str, column: str) -> str | None:
        """Kiểu SQL của 1 cột (lowercase), None nếu không rõ/SQLite."""
        if not self._column_types:
            return None
        return self._column_types.get((table.lower(), column.lower()))

    def ts(self, table: str, column: str, iso: str | None) -> str | None:
        """Chuẩn hóa timestamp ISO theo KIỂU CỘT thực tế trong SQL Server.

        - Cột NVARCHAR/VARCHAR (giống TEXT của SQLite): giữ nguyên hậu tố
          'Z' — so chuỗi ISO đúng thứ tự thời gian như trước.
        - Cột DATETIME/DATETIME2/...: BỎ hậu tố 'Z' (SQL Server KHÔNG cast
          được chuỗi có 'Z' sang datetime) — giá trị vẫn là UTC vì app
          luôn ghi giờ UTC.
        SQLite (script test): trả nguyên giá trị (không có khai báo kiểu).
        """
        if iso is None or not self._datetime_cols:
            return iso
        if (table.lower(), column.lower()) in self._datetime_cols:
            return iso[:-1] if iso.endswith("Z") else iso
        return iso

    # ---------------------------------------------------------
    # Session (dùng chung cho cả 2 backend)
    # ---------------------------------------------------------
    @contextmanager
    def session(self) -> Iterator[Any]:
        """Ngữ cảnh giao dịch: commit khi thành công, rollback khi có lỗi.

        Toàn bộ thân giao dịch nằm trong ``_DB_LOCK`` để các THREAD khác
        nhau (UI + QThread đồng bộ) không xen kẽ transaction của nhau.
        """
        conn = self.connect()
        with _DB_LOCK:
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def close(self) -> None:
        """Đóng kết nối (gọi khi thoát app)."""
        with _DB_LOCK:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
