"""D1Client — gọi Cloudflare D1 REST API từ app desktop (Bước 15).

Thay vì deploy một Worker trung gian, app gọi TRỰC TIẾP endpoint HTTP
chính thức của Cloudflare:

    POST /accounts/{account_id}/d1/database/{database_id}/query

D1 là SQLite serverless → cú pháp SQL giống hệt SQLite local (đúng tinh
thần ADR-4: "cùng cú pháp SQL → logic đồng bộ đơn giản nhất"). Xác thực
bằng API token (quyền Account · D1 · Edit) qua header ``Authorization:
Bearer <token>``.

Đặc điểm quan trọng của REST API:
- Tham số (``params``) chỉ nhận CHUỖI → số/boolean truyền dạng chuỗi, SQLite
  tự ép kiểu theo type affinity của cột (giá trị '0.91' vào cột REAL → 0.91).
- BLOB (embedding 2048 bytes) không truyền qua JSON được → nhúng literal
  ``X'<hex>'`` ngay trong câu SQL (embedding chỉ ~4KB hex < giới hạn 100KB/câu);
  khi đọc dùng ``SELECT hex(embedding)`` rồi ``bytes.fromhex()``.
- Giới hạn D1: tối đa 100 tham số/câu, câu tối đa 100KB, hàng tối đa 2MB —
  dữ liệu của app (<50 người, vài nghìn sự kiện) thoải mái nằm trong mức này.

Lỗi mạng (mất kết nối) và lỗi D1 (sai token/quyền, SQL sai) được gói thành
``D1Error`` — SyncService bắt và báo trạng thái, không làm sập app.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# URL gốc của Cloudflare API v4
API_BASE = "https://api.cloudflare.com/client/v4"

# Thời gian chờ mỗi request (giây) — D1 trả lời nhanh, 15s là quá đủ
DEFAULT_TIMEOUT = 15.0


class D1Error(Exception):
    """Lỗi khi gọi D1: sai thông tin, mất mạng, hoặc SQL bị từ chối."""


# ---------------------------------------------------------------------------
# Lược đồ D1 — chỉ 3 bảng DÙNG CHUNG (persons/face_samples/recognition_events).
# sync_outbox và settings là riêng của bản local (hàng đợi + khóa-giá trị)
# nên KHÔNG tạo trên D1. Cú pháp giữ nguyên như SCHEMA_SQL của db.py
# (khóa ngoại + index) để mobile sau này đọc được đúng cấu trúc.
# ---------------------------------------------------------------------------
D1_DDL = """
CREATE TABLE IF NOT EXISTS persons (
    id               TEXT PRIMARY KEY,
    name             TEXT NOT NULL CHECK (length(trim(name)) > 0),
    created_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    thumbnail_path   TEXT NOT NULL,
    thumbnail_r2_key TEXT
);

CREATE TABLE IF NOT EXISTS face_samples (
    id          TEXT PRIMARY KEY,
    person_id   TEXT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    embedding   BLOB NOT NULL,
    quality     REAL NOT NULL DEFAULT 0.0 CHECK (quality >= 0.0 AND quality <= 1.0),
    captured_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_face_samples_person ON face_samples(person_id);

CREATE TABLE IF NOT EXISTS recognition_events (
    id              TEXT PRIMARY KEY,
    person_id       TEXT REFERENCES persons(id) ON DELETE SET NULL,
    label           TEXT NOT NULL,
    source          TEXT NOT NULL CHECK (source IN ('webcam', 'photo', 'mobile')),
    detected_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    similarity      REAL,
    snapshot_path   TEXT,
    snapshot_r2_key TEXT,
    is_unknown      INTEGER NOT NULL DEFAULT 0 CHECK (is_unknown IN (0, 1))
);
CREATE INDEX IF NOT EXISTS idx_events_detected_at ON recognition_events(detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_person      ON recognition_events(person_id);
CREATE INDEX IF NOT EXISTS idx_events_source      ON recognition_events(source);
"""

# BLOB -> literal X'hex' nhúng trong SQL (thay cho tham số — REST chỉ nhận string)
def blob_hex_literal(blob: bytes) -> str:
    """Chuyển bytes thành literal SQLite ``X'...'`` để nhúng vào câu SQL."""
    return f"X'{blob.hex()}'"


def blob_to_hex(blob: bytes) -> str:
    """Hex string của bytes (dùng làm tham số khi cần)."""
    return blob.hex()


def hex_to_bytes(hex_str: str) -> bytes:
    """Đọc lại bytes từ hex string (SELECT hex(embedding) → bytes)."""
    return bytes.fromhex(hex_str)


class D1Client:
    """Khách gọi D1 qua REST API — gói trọn xác thực, lỗi, và SQL.

    Cách dùng:
        client = D1Client(account_id, database_id, api_token)
        client.ensure_schema()                 # tạo bảng nếu chưa có
        rows = client.query("SELECT * FROM persons")
        client.query_batch([("INSERT ... VALUES (?, ?)", ["a", "b"]), ...])
    """

    def __init__(
        self,
        account_id: str,
        database_id: str,
        api_token: str,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._url = (
            f"{API_BASE}/accounts/{account_id}/d1/database/{database_id}/query"
        )
        self._headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }
        self._timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    # ------------------------------------------------------------------
    # Gọi API
    # ------------------------------------------------------------------
    def _post(self, body: dict[str, Any]) -> list[dict[str, Any]]:
        """Gửi POST tới /query; trả về mảng ``result`` (mỗi phần tử 1 query).

        Ném ``D1Error`` khi: mất mạng/timeout, HTTP lỗi, hoặc D1 báo lỗi.
        """
        try:
            response = self._client.post(self._url, json=body, headers=self._headers)
        except httpx.HTTPError as exc:
            raise D1Error(f"Mất kết nối tới Cloudflare: {exc}") from exc

        payload = response.json() if response.content else {}
        if response.status_code != 200:
            message = _extract_error(payload) or f"HTTP {response.status_code}"
            raise D1Error(f"D1 từ chối yêu cầu ({message})")
        if not payload.get("success"):
            message = _extract_error(payload) or "success=false"
            raise D1Error(f"D1 báo lỗi ({message})")

        # Mỗi phần tử của result tương ứng 1 câu SQL trong batch
        results = payload.get("result", [])
        for item in results:
            if not item.get("success", True):
                raise D1Error(f"D1 báo lỗi khi chạy SQL: {item.get('meta', {})}")
        return results

    def query(
        self,
        sql: str,
        params: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Chạy MỘT câu SQL (có thể nhiều câu cách dấu ';' — D1 chạy như batch).

        Trả về danh sách các HÀNG (mỗi hàng là dict cột → giá trị).
        """
        body: dict[str, Any] = {"sql": sql}
        if params:
            body["params"] = params
        results = self._post(body)
        # results[0].results chứa các hàng của câu đầu tiên
        rows = results[0].get("results", []) if results else []
        return [dict(r) for r in rows]

    def query_batch(self, statements: list[tuple[str, list[str] | None]]) -> None:
        """Chạy nhiều câu SQL trong MỘT request (batch) — giảm số lượt gọi.

        ``statements``: list[(sql, params | None)]. D1 chạy tuần tự trong
        cùng transaction → an toàn cho việc đẩy hàng loạt outbox.
        """
        if not statements:
            return
        self._post({"batch": [{"sql": s, "params": p} if p else {"sql": s} for s, p in statements]})

    # ------------------------------------------------------------------
    # Tiện ích cấp cao
    # ------------------------------------------------------------------
    def test_connection(self) -> None:
        """Kiểm tra thông tin kết nối: SELECT 1 — lỗi → D1Error (sai token/ID)."""
        self.query("SELECT 1")

    def ensure_schema(self) -> None:
        """Tạo 3 bảng + index trên D1 nếu chưa có (chạy lại vô hại)."""
        self._post({"sql": D1_DDL})


def _extract_error(payload: dict[str, Any]) -> str:
    """Lấy thông điệp lỗi đầu tiên từ payload Cloudflare (errors/messages)."""
    for key in ("errors", "messages"):
        entries = payload.get(key) or []
        if entries and isinstance(entries, list):
            msg = entries[0].get("message")
            if msg:
                return str(msg)
    return ""
