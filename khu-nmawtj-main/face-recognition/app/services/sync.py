"""SyncService — đồng bộ dữ liệu local ↔ Cloudflare D1 + object storage (Bước 15–16).

Kiến trúc: **local-first + outbox pattern** (ADR-4):
- PUSH (local → cloud): đọc ``sync_outbox`` — các thao tác ghi local chưa
  đồng bộ — đẩy lên D1 dạng upsert/delete, rồi đánh dấu ``synced_at``.
  Lần đồng bộ ĐẦU TIÊN còn đẩy TOÀN BỘ dữ liệu hiện có (persons +
  face_samples + recognition_events) để cloud có bản sao đầy đủ.
- ẢNH (Bước 16): trước khi đẩy 1 người/sự kiện, upload thumbnail/snapshot
  lên object storage S3-compatible — mặc định **Supabase Storage** (free
  tier không cần thẻ tín dụng; key quy ước ``thumbnails/{id}.jpg`` /
  ``snapshots/{id}.jpg``) và ghi key vào DB → câu SQL upsert mang đúng
  ``*_r2_key``. Xóa dữ liệu → xóa luôn ảnh tương ứng. Kéo dữ liệu có key
  ảnh → tải ảnh về local để Lịch sử/Danh sách hiển thị. Storage chưa cấu
  hình → chỉ đồng bộ D1 (text).
- PULL (cloud → local): kéo sự kiện từ cloud về (sau này app mobile ghi sự
  kiện quét — desktop phải thấy trong Lịch sử). Người + embedding chỉ kéo
  về khi local CHƯA có (khôi phục sau khi cài lại) — desktop là nơi QUẢN
  LÝ nên local thắng khi trùng id.
- Xung đột: mỗi dòng có id UUID riêng và sự kiện bất biến → "chèn nếu chưa
  có" là đủ an toàn cho quy mô <50 người (không cần cột updated_at phức tạp).

Không có mạng / sai token / lỗi D1 / lỗi storage → bắt ``D1Error``/``S3Error``
và trả về ``SyncResult`` kèm danh sách lỗi — KHÔNG ném ra ngoài (app không
được sập vì mất mạng). Nếu upload ảnh thất bại, CẢ lượt sync coi như thất bại
(outbox giữ nguyên) để lần sau thử lại — không đánh dấu "đã sync" khi ảnh
chưa lên được cloud.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.config import Config
from app.infrastructure.d1_client import D1Client, D1Error, blob_hex_literal
from app.infrastructure.db import DATA_DIR, Database
from app.infrastructure.s3_client import S3Client, S3Error
from app.infrastructure.repositories import (
    FaceSampleRepository,
    PersonRepository,
    RecognitionEventRepository,
    SyncOutboxRepository,
    _blob_to_embedding,
)

logger = logging.getLogger(__name__)

# Khóa trong bảng settings (key-value local) — đánh dấu trạng thái đồng bộ
KEY_INITIAL_SYNCED = "cloud_initial_synced"  # '1' = đã đẩy toàn bộ dữ liệu 1 lần
KEY_LAST_SYNC = "cloud_synced_at"            # thời điểm đồng bộ thành công gần nhất


@dataclass
class SyncResult:
    """Kết quả một lượt đồng bộ — UI hiển thị trực tiếp."""

    ok: bool = True
    pushed: int = 0    # số thao tác đã đẩy lên cloud
    pulled: int = 0    # số dòng đã kéo từ cloud về local
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.ok:
            return "Lỗi: " + "; ".join(self.errors[:3])
        return f"✓ Đã đồng bộ · đẩy {self.pushed} · kéo {self.pulled}"


class SyncService:
    """Điều phối đồng bộ 2 chiều với D1.

    Cách dùng (từ QThread để không đơ UI):
        service = SyncService(db, config)
        result = service.run_full_sync()
    """

    def __init__(self, db: Database, config: Config) -> None:
        self._db = db
        self._config = config
        self._persons = PersonRepository(db)
        self._samples = FaceSampleRepository(db)
        self._events = RecognitionEventRepository(db)
        self._outbox = SyncOutboxRepository(db)

    # ------------------------------------------------------------------
    # Đầu vào
    # ------------------------------------------------------------------
    def is_configured(self) -> bool:
        """Đã có đủ 3 thông tin cloud chưa (account_id, database_id, token)?"""
        return bool(
            self._config.cloud_account_id.strip()
            and self._config.cloud_database_id.strip()
            and self._config.cloud_api_token.strip()
        )

    def make_client(self) -> D1Client | None:
        if not self.is_configured():
            return None
        return D1Client(
            account_id=self._config.cloud_account_id.strip(),
            database_id=self._config.cloud_database_id.strip(),
            api_token=self._config.cloud_api_token.strip(),
        )

    def is_storage_configured(self) -> bool:
        """Đã đủ thông tin object storage chưa (endpoint + bucket + 2 key)."""
        return bool(
            self._config.sb_endpoint.strip()
            and self._config.sb_bucket.strip()
            and self._config.sb_access_key_id.strip()
            and self._config.sb_secret_access_key.strip()
        )

    def make_storage_client(self) -> S3Client | None:
        """Khách S3 (upload/tải ảnh); None nếu chưa cấu hình storage.

        Provider-agnostic (Supabase Storage / Cloudflare R2 / MinIO... —
        mọi dịch vụ S3-compatible): endpoint + region + bucket + credential
        đọc từ config, xem s3_client.py. Chưa cấu hình → chỉ đồng bộ D1
        (dữ liệu text), ảnh bỏ qua (key giữ NULL).
        """
        if not self.is_storage_configured():
            return None
        return S3Client(
            endpoint_url=self._config.sb_endpoint.strip(),
            region=self._config.sb_region.strip(),
            bucket=self._config.sb_bucket.strip(),
            access_key_id=self._config.sb_access_key_id.strip(),
            secret_access_key=self._config.sb_secret_access_key.strip(),
        )

    # ------------------------------------------------------------------
    # Đồng bộ chính
    # ------------------------------------------------------------------
    def run_full_sync(self) -> SyncResult:
        """Một lượt đồng bộ đầy đủ: đảm bảo schema → push → pull."""
        client = self.make_client()
        if client is None:
            return SyncResult(ok=False, errors=["Chưa cấu hình đủ thông tin cloud (Cài đặt)."])
        storage = self.make_storage_client()  # None = chưa cấu hình → chỉ sync D1

        result = SyncResult()
        try:
            client.ensure_schema()
            self._push(client, storage, result)
            self._pull(client, storage, result)
            _set_setting(self._db, KEY_LAST_SYNC, _now_iso())
            logger.info("Đồng bộ xong: %s", result.summary())
        except D1Error as exc:
            result.ok = False
            result.errors.append(str(exc))
            logger.warning("Đồng bộ thất bại: %s", exc)
        except S3Error as exc:
            result.ok = False
            result.errors.append(f"Storage: {exc}")
            logger.warning("Đồng bộ ảnh thất bại: %s", exc)
        except Exception as exc:  # noqa: BLE001 — mọi lỗi bất ngờ đều thành lỗi mềm
            result.ok = False
            result.errors.append(f"Lỗi nội bộ: {exc}")
            logger.exception("Đồng bộ lỗi bất ngờ")
        return result

    # ------------------------------------------------------------------
    # PUSH — local → cloud
    # ------------------------------------------------------------------
    def _push(self, client: D1Client, storage: S3Client | None, result: SyncResult) -> None:
        statements: list[tuple[str, list[str] | None]] = []

        # Lần đầu tiên: đẩy TOÀN BỘ dữ liệu hiện có (cloud rỗng → cần bản sao)
        first_full = _get_setting(self._db, KEY_INITIAL_SYNCED) != "1"
        if first_full:
            for person in self._persons.list_all():
                self._upload_person_thumbnail(storage, person)
                statements.append(_person_upsert_stmt(person))
            for sample in self._samples.all_samples():
                statements.append(_sample_upsert_stmt(sample))
            for event in self._events.list_all():
                self._upload_event_snapshot(storage, event)
                statements.append(_event_upsert_stmt(event))
            result.pushed += len(statements)
            logger.info("Đồng bộ đầu tiên: đẩy toàn bộ %d dòng lên D1", len(statements))

        # Outbox: các thay đổi phát sinh sau lần sync trước
        pending = self._outbox.list_pending()
        for outbox_id, entity, entity_id, op in pending:
            stmt = self._outbox_stmt(storage, entity, entity_id, op)
            if stmt is not None:
                statements.append(stmt)
                result.pushed += 1

        # Đẩy theo batch (D1 chạy tuần tự trong cùng transaction). Chỉ đánh
        # dấu "đã full sync" / "đã đẩy" SAU KHI batch thành công — nếu mạng
        # lỗi giữa chừng thì lần sau tự đẩy lại toàn bộ.
        if statements:
            client.query_batch(statements)
            self._outbox.mark_synced([oid for oid, *_ in pending], _now_iso())
            if first_full:
                _set_setting(self._db, KEY_INITIAL_SYNCED, "1")

    def _outbox_stmt(
        self, storage: S3Client | None, entity: str, entity_id: str, op: str
    ) -> tuple[str, list[str] | None] | None:
        """Dựng câu SQL cho 1 dòng outbox; None nếu dữ liệu gốc đã biến mất.

        Trước khi dựng lệnh upsert, upload ảnh (thumbnail/snapshot) lên
        object storage nếu chưa có key (Bước 16) — để câu SQL chứa đúng
        ``*_r2_key``. Thao tác ``delete`` còn xóa luôn ảnh tương ứng.
        """
        if entity == "person":
            person = self._persons.get(entity_id)
            if op == "delete":
                self._delete_storage_object(storage, f"thumbnails/{entity_id}.jpg")
                # Xóa kèm mẫu embedding (CASCADE) + ngắt liên kết sự kiện
                return (
                    "DELETE FROM face_samples WHERE person_id = ?;"
                    " UPDATE recognition_events SET person_id = NULL WHERE person_id = ?;"
                    " DELETE FROM persons WHERE id = ?",
                    [entity_id, entity_id, entity_id],
                )
            if person is not None:
                self._upload_person_thumbnail(storage, person)
                return _person_upsert_stmt(person)
        elif entity == "face_sample":
            sample = self._samples.get(entity_id)
            if op == "delete":
                return ("DELETE FROM face_samples WHERE id = ?", [entity_id])
            if sample is not None:
                return _sample_upsert_stmt(sample)
        elif entity == "recognition_event":
            event = self._events.get(entity_id)
            if op == "delete":
                self._delete_storage_object(storage, f"snapshots/{entity_id}.jpg")
                return ("DELETE FROM recognition_events WHERE id = ?", [entity_id])
            if event is not None:
                self._upload_event_snapshot(storage, event)
                return _event_upsert_stmt(event)
        # Dữ liệu gốc đã bị xóa trước khi sync (outbox 'upsert' mồ côi) → bỏ qua
        return None

    # ------------------------------------------------------------------
    # Ảnh object storage — upload / xóa / tải về (Bước 16)
    # ------------------------------------------------------------------
    def _upload_person_thumbnail(self, storage: S3Client | None, person: Any) -> None:
        """Upload thumbnail của người lên storage nếu có ảnh local + chưa có key.

        Sau khi upload thành công ghi key vào DB (KHÔNG ghi outbox — key là
        kết quả của sync, ghi outbox sẽ tạo vòng lặp). Lỗi storage → S3Error
        lan lên → cả lượt sync thất bại → outbox giữ nguyên → lần sau thử lại.
        """
        if storage is None or person.thumbnail_r2_key:
            return
        if not person.thumbnail_path:
            return
        path = (DATA_DIR / person.thumbnail_path).resolve()
        if not path.is_relative_to(DATA_DIR.resolve()) or not path.exists():
            return  # ảnh local không còn — bỏ qua (key vẫn NULL)
        key = f"thumbnails/{person.id}.jpg"
        storage.upload(key, path.read_bytes(), "image/jpeg")
        self._persons.set_thumbnail_r2_key(person.id, key)
        person.thumbnail_r2_key = key

    def _upload_event_snapshot(self, storage: S3Client | None, event: Any) -> None:
        """Upload snapshot của sự kiện lên storage (tương tự thumbnail)."""
        if storage is None or event.snapshot_r2_key:
            return
        if not event.snapshot_path:
            return
        path = (DATA_DIR / event.snapshot_path).resolve()
        if not path.is_relative_to(DATA_DIR.resolve()) or not path.exists():
            return
        key = f"snapshots/{event.id}.jpg"
        storage.upload(key, path.read_bytes(), "image/jpeg")
        self._events.set_snapshot_r2_key(event.id, key)
        event.snapshot_r2_key = key

    @staticmethod
    def _delete_storage_object(storage: S3Client | None, key: str) -> None:
        """Xóa 1 object khi xóa dữ liệu local (delete_object idempotent — xóa
        key không tồn tại cũng không lỗi). Chưa cấu hình storage → bỏ qua."""
        if storage is not None:
            storage.delete(key)

    # ------------------------------------------------------------------
    # PULL — cloud → local
    # ------------------------------------------------------------------
    def _pull(self, client: D1Client, storage: S3Client | None, result: SyncResult) -> None:
        """Kéo dữ liệu từ D1 về: chèn những dòng local CHƯA có.

        - Sự kiện: kéo hết (mobile sẽ ghi sự kiện quét vào đây).
        - Người/mẫu: chỉ kéo khi local chưa có id đó (desktop quản lý chính).
        - Ảnh (Bước 16): dòng có ``*_r2_key`` nhưng local chưa có file → tải
          từ storage về đúng thư mục (snapshots/ thumbs/) để Lịch sử/Danh
          sách hiển thị được. Chưa cấu hình storage hoặc tải lỗi → giữ key,
          bỏ ảnh (sự kiện vẫn kéo về, chỉ thiếu hình).
        Không ghi outbox cho các dòng vừa kéo (tránh đẩy ngược lại — vòng lặp).

        THỨ TỰ quan trọng: người → mẫu → sự kiện (người trước vì sự kiện
        có khóa ngoại person_id; kéo events TRƯỚC persons sẽ vi phạm
        FOREIGN KEY khi sự kiện mobile tham chiếu người chưa kéo về).
        """
        # 1) Người (khôi phục khi local trống/mới)
        for row in client.query(
            "SELECT id, name, created_at, thumbnail_path, thumbnail_r2_key,"
            " department, position FROM persons"
        ):
            if self._persons.get(row["id"]) is not None:
                continue
            r2_key = row.get("thumbnail_r2_key")
            thumbnail_path = self._download_thumbnail(storage, row["id"], r2_key)
            self._persons.add(
                name=row["name"],
                thumbnail_path=thumbnail_path or row.get("thumbnail_path") or "",
                person_id=row["id"],
                created_at=row.get("created_at"),
                thumbnail_r2_key=r2_key,
                # Cloud có thể chưa có 2 cột mới (schema cũ) → mặc định rỗng
                department=row.get("department") or "",
                position=row.get("position") or "",
                record_outbox=False,
            )
            result.pulled += 1

        # 2) Mẫu embedding (chỉ khi người đó CÓ ở local — tránh mồ côi)
        for row in client.query(
            "SELECT id, person_id, hex(embedding) AS embedding_hex, quality,"
            " captured_at FROM face_samples"
        ):
            if self._samples.get(row["id"]) is not None:
                continue
            if self._persons.get(row["person_id"]) is None:
                continue  # người chưa kéo về — bỏ mẫu này (an toàn hơn)

            self._samples.add(
                person_id=row["person_id"],
                embedding=_blob_to_embedding(bytes.fromhex(row["embedding_hex"])),
                quality=row.get("quality") or 0.0,
                sample_id=row["id"],
                captured_at=row.get("captured_at"),
                record_outbox=False,
            )
            result.pulled += 1

        # 3) Sự kiện (cuối — sau persons để khóa ngoại person_id hợp lệ)
        for row in client.query(
            "SELECT id, person_id, label, source, detected_at, similarity,"
            " snapshot_path, snapshot_r2_key, is_unknown FROM recognition_events"
        ):
            if self._events.get(row["id"]) is not None:
                continue
            # Người tham chiếu chưa có ở local (đã xóa trên cloud?) → ngắt
            # liên kết, giữ sự kiện với person_id = NULL (không mất lịch sử)
            person_id = row.get("person_id")
            if person_id is not None and self._persons.get(person_id) is None:
                person_id = None
            r2_key = row.get("snapshot_r2_key")
            snapshot_path = self._download_snapshot(storage, row["id"], r2_key)
            self._events.add(
                person_id=person_id,
                label=row["label"],
                source=row["source"],
                similarity=row.get("similarity"),
                snapshot_path=snapshot_path or row.get("snapshot_path") or "",
                is_unknown=bool(row.get("is_unknown")),
                event_id=row["id"],
                detected_at=row.get("detected_at"),
                snapshot_r2_key=r2_key,
                record_outbox=False,
            )
            result.pulled += 1

    # ------------------------------------------------------------------
    # Tải ảnh storage về local (Bước 16)
    # ------------------------------------------------------------------
    def _download_snapshot(self, storage: S3Client | None, event_id: str, r2_key: Any) -> str:
        """Tải snapshot về data/snapshots/{event_id}.jpg; trả đường dẫn tương
        đối (rỗng nếu chưa cấu hình storage / tải lỗi / không có key)."""
        if not r2_key or storage is None:
            return ""
        try:
            data = storage.download(r2_key)
        except S3Error as exc:
            logger.warning("Không tải được snapshot %s: %s", r2_key, exc)
            return ""
        return self._save_downloaded("snapshots", f"{event_id}.jpg", data)

    def _download_thumbnail(self, storage: S3Client | None, person_id: str, r2_key: Any) -> str:
        """Tải thumbnail về data/thumbs/{person_id}.jpg (tương tự snapshot)."""
        if not r2_key or storage is None:
            return ""
        try:
            data = storage.download(r2_key)
        except S3Error as exc:
            logger.warning("Không tải được thumbnail %s: %s", r2_key, exc)
            return ""
        return self._save_downloaded("thumbs", f"{person_id}.jpg", data)

    @staticmethod
    def _save_downloaded(subdir: str, filename: str, data: bytes) -> str:
        """Ghi ảnh tải từ storage vào data/{subdir}/{filename}; trả đường dẫn tương đối."""
        folder = DATA_DIR / subdir
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / filename
        path.write_bytes(data)
        return str(path.relative_to(DATA_DIR))


# ---------------------------------------------------------------------------
# Dựng câu SQL upsert (chạy được trên cả SQLite local lẫn D1)
# ---------------------------------------------------------------------------

def _bind(sql: str, params: list[str | None]) -> tuple[str, list[str]]:
    """Thay None trong params bằng literal ``NULL`` ngay trong SQL.

    REST API của D1 khai báo ``params`` là mảng STRING — JSON null không
    được đảm bảo chấp nhận (embedding/quality/similarity... có thể rỗng)
    → None được nhúng thẳng thành ``NULL``, chỉ giữ lại giá trị thật làm
    tham số. An toàn vì mọi ``?`` trong câu đều là placeholder (không có
    trong chuỗi literal khác).
    """
    out_params: list[str] = []
    pieces = sql.split("?")
    parts: list[str] = []
    for i, piece in enumerate(pieces):
        if i < len(pieces) - 1:  # mọi phần trừ phần cuối đều có 1 placeholder
            param = params[i]
            if param is None:
                parts.append(piece + "NULL")
            else:
                parts.append(piece + "?")
                out_params.append(param)
        else:
            parts.append(piece)
    return "".join(parts), out_params


def _person_upsert_stmt(person: Any) -> tuple[str, list[str] | None]:
    sql = (
        "INSERT INTO persons (id, name, created_at, thumbnail_path, thumbnail_r2_key)"
        " VALUES (?, ?, ?, ?, ?)"
        " ON CONFLICT(id) DO UPDATE SET name = excluded.name,"
        " thumbnail_path = excluded.thumbnail_path,"
        " thumbnail_r2_key = excluded.thumbnail_r2_key"
    )
    return _bind(sql, [
        person.id, person.name, person.created_at,
        person.thumbnail_path, person.thumbnail_r2_key,
    ])


def _sample_upsert_stmt(sample: Any) -> tuple[str, list[str] | None]:
    # Embedding BLOB nhúng literal X'hex' — REST API không nhận binary qua params
    sql = (
        "INSERT INTO face_samples (id, person_id, embedding, quality, captured_at)"
        f" VALUES (?, ?, {blob_hex_literal(sample.embedding.astype('float32').tobytes())}, ?, ?)"
        " ON CONFLICT(id) DO UPDATE SET person_id = excluded.person_id,"
        " embedding = excluded.embedding, quality = excluded.quality,"
        " captured_at = excluded.captured_at"
    )
    return _bind(sql, [sample.id, sample.person_id, str(sample.quality), sample.captured_at])


def _event_upsert_stmt(event: Any) -> tuple[str, list[str] | None]:
    sql = (
        "INSERT INTO recognition_events (id, person_id, label, source, detected_at,"
        " similarity, snapshot_path, snapshot_r2_key, is_unknown)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        " ON CONFLICT(id) DO UPDATE SET label = excluded.label,"
        " similarity = excluded.similarity, is_unknown = excluded.is_unknown"
    )
    return _bind(
        sql,
        [
            event.id,
            event.person_id,
            event.label,
            event.source,
            event.detected_at,
            str(event.similarity) if event.similarity is not None else None,
            event.snapshot_path,
            event.snapshot_r2_key,
            "1" if event.is_unknown else "0",
        ],
    )


# ---------------------------------------------------------------------------
# Bảng settings (key-value) — trạng thái đồng bộ
# ---------------------------------------------------------------------------

def _get_setting(db: Database, key: str) -> str | None:
    with db.session() as conn:
        row = conn.execute(
            "SELECT value FROM settings WHERE key = ?", (key,)
        ).fetchone()
    return str(row["value"]) if row else None


def _set_setting(db: Database, key: str, value: str) -> None:
    """Ghi giá trị vào bảng settings (key-value).

    SQL Server không hỗ trợ ``ON CONFLICT ... DO UPDATE`` của SQLite (và
    MERGE có nhiều edge case với tham số) → UPDATE trước; không có dòng
    nào được đổi thì INSERT mới (rowcount an toàn trong cùng session).
    """
    with db.session() as conn:
        cur = conn.execute(
            "UPDATE settings SET value = ? WHERE key = ?", (value, key)
        )
        if cur.rowcount == 0:
            conn.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?)",
                (key, value),
            )


def _now_iso() -> str:
    """Thời điểm hiện tại UTC dạng ISO (khớp định dạng DEFAULT của bảng).

    Chuẩn '%Y-%m-%dT%H:%M:%S.%fZ' — trước đây dùng '%f' thiếu dấu chấm
    (chuỗi sai chuẩn, SQLite vẫn lưu được nhưng SQL Server DATETIME
    không cast được) → sửa đúng chuẩn cho cả 2.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
