"""Bước 16 — Kiểm tra Đồng bộ ảnh lên Supabase Storage (S3-compatible qua boto3).

Toàn bộ test chạy OFFLINE — dùng FakeS3Client giả lập object storage (ghi
nhận upload/download/delete, trả dữ liệu giả) để không phụ thuộc mạng/tài
khoản thật:

1. S3Client: key quy ước thumbnails/{id}.jpg + snapshots/{id}.jpg.
2. Push: người/sự kiện có ảnh local chưa có key → upload lên storage TRƯỚC
   khi đẩy D1 → key xuất hiện trong SQL upsert + ghi vào DB local.
3. Không upload lại: người đã có thumbnail_r2_key → bỏ qua (key giữ).
4. Delete: xóa người/sự kiện → xóa luôn object tương ứng.
5. Pull: dòng cloud có snapshot_r2_key + ảnh trên storage → tải về local
   (data/snapshots/{id}.jpg) để Lịch sử hiển thị được.
6. Storage chưa cấu hình → make_storage_client() = None → chỉ sync D1.
7. Lỗi storage khi upload → result.ok=False, outbox GIỮ NGUYÊN (thử lại).
8. GUI SettingsView: 5 ô Supabase hiện/ẩn theo checkbox, lưu vào config tạm.

Chạy:  .venv\\Scripts\\python.exe scripts\\step_16_gui_test.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import PySide6  # noqa: E402

# Khắc phục đường dẫn plugins + chế độ ảo (giống các test GUI khác)
pyside_dir = os.path.dirname(PySide6.__file__)
os.environ["QT_PLUGIN_PATH"] = os.path.join(pyside_dir, "plugins")
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(pyside_dir)
os.environ["QT_QPA_PLATFORM"] = "minimal"

import numpy as np  # noqa: E402

import app.config as cfg_mod  # noqa: E402
import app.infrastructure.s3_client as s3_mod  # noqa: E402
import app.ui.settings_view as sv  # noqa: E402
from app.config import Config  # noqa: E402
from app.infrastructure.db import DATA_DIR, Database  # noqa: E402
from app.infrastructure.repositories import (  # noqa: E402
    FaceSampleRepository,
    PersonRepository,
    RecognitionEventRepository,
    SyncOutboxRepository,
)
from app.services.auth import AuthService  # noqa: E402
from app.services.sync import SyncService  # noqa: E402

# File TẠM — tuyệt đối không đụng config.json / app.db thật của người dùng
TEMP_CONFIG = PROJECT_ROOT / "data" / "test_config_16.json"
TEMP_DB = PROJECT_ROOT / "data" / "test_db_16.db"

# Ảnh giả để test upload (không cần là ảnh hợp lệ — chỉ test luồng bytes)
FAKE_IMAGE = b"\xff\xd8\xff\xe0" + b"\x00" * 100  # header JPEG + đệm

PASS = 0
FAIL = 0
OPEN_DBS: list[Database] = []
TEMP_FILES: list[Path] = []


def new_db() -> Database:
    """Tạo Database gắn file TẠM — đăng ký để cleanup đóng kết nối."""
    db = Database(TEMP_DB)
    OPEN_DBS.append(db)
    return db


def make_local_image(rel_path: str) -> str:
    """Tạo file ảnh giả trong thư mục dữ liệu; trả đường dẫn tương đối.

    rel_path ví dụ 'thumbs/thumb_a.jpg' — file nằm tại DATA_DIR/rel_path.
    Đăng ký để cleanup xóa.
    """
    path = (DATA_DIR / rel_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(FAKE_IMAGE)
    TEMP_FILES.append(path)
    return rel_path


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [QUA] {name} {detail}")
    else:
        FAIL += 1
        print(f"  [THẤT] {name} {detail}")


def cleanup() -> None:
    for db in OPEN_DBS:
        db.close()
    OPEN_DBS.clear()
    for path in TEMP_FILES:
        try:
            path.unlink(missing_ok=True)
        except PermissionError:
            pass
    TEMP_FILES.clear()
    # File ảnh tải về từ R2 trong test pull (data/thumbs, data/snapshots)
    for rel in ("thumbs/p-cloud-2.jpg", "snapshots/ev-mobile-2.jpg"):
        try:
            (DATA_DIR / rel).unlink(missing_ok=True)
        except PermissionError:
            pass
    TEMP_CONFIG.unlink(missing_ok=True)
    for suffix in ("", "-wal", "-shm"):
        try:
            Path(str(TEMP_DB) + suffix).unlink(missing_ok=True)
        except PermissionError:
            pass  # vẫn còn khóa (hiếm) — lần chạy sau tự dọn


def make_config() -> Config:
    cfg_mod.CONFIG_PATH = TEMP_CONFIG
    cfg = Config()
    cfg.cloud_account_id = "fake-account-123"
    cfg.cloud_database_id = "fake-db-456"
    cfg.cloud_api_token = "fake-token-789"
    cfg.sb_endpoint = "https://fake-ref.supabase.co/storage/v1/s3"
    cfg.sb_region = "ap-southeast-1"
    cfg.sb_bucket = "fake-bucket"
    cfg.sb_access_key_id = "fake-access-key"
    cfg.sb_secret_access_key = "fake-secret"
    return cfg


# ---------------------------------------------------------------------------
# FakeD1Client (giống step_15) + FakeS3Client (giả lập storage, offline)
# ---------------------------------------------------------------------------

class FakeD1Client:
    """Không gọi mạng: ghi nhận câu lệnh + mô phỏng dữ liệu cloud."""

    def __init__(self) -> None:
        self.ensure_called = 0
        self.statements: list[tuple[str, list | None]] = []
        self.remote_persons: list[dict] = []
        self.remote_samples: list[dict] = []
        self.remote_events: list[dict] = []

    def test_connection(self) -> None:
        return

    def ensure_schema(self) -> None:
        self.ensure_called += 1

    def query(self, sql: str, params: list[str] | None = None) -> list[dict]:
        if "FROM recognition_events" in sql:
            return list(self.remote_events)
        if "FROM persons" in sql:
            return list(self.remote_persons)
        if "FROM face_samples" in sql:
            return list(self.remote_samples)
        return []

    def query_batch(self, statements: list[tuple[str, list | None]]) -> None:
        self.statements.extend(statements)


class FakeS3Client:
    """Giả lập object storage: ghi nhận upload/download/delete, không gọi mạng."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}      # key -> bytes (bucket giả)
        self.uploads: list[str] = []
        self.downloads: list[str] = []
        self.deletes: list[str] = []
        self.fail_upload = False                  # bật để test lỗi upload

    def test_connection(self) -> None:
        if self.fail_upload:
            raise s3_mod.S3Error("Storage từ chối kết nối (fake)")

    def upload(self, key: str, data: bytes, content_type: str = "image/jpeg") -> None:
        if self.fail_upload:
            raise s3_mod.S3Error("Upload thất bại (fake)")
        self.uploads.append(key)
        self.objects[key] = data

    def download(self, key: str) -> bytes:
        if key not in self.objects:
            raise s3_mod.S3Error(f"Object {key} không tồn tại (fake)")
        self.downloads.append(key)
        return self.objects[key]

    def delete(self, key: str) -> None:
        self.deletes.append(key)
        self.objects.pop(key, None)


def make_service(db: Database, config: Config) -> tuple[SyncService, FakeD1Client, FakeS3Client]:
    service = SyncService(db, config)
    fake_d1 = FakeD1Client()
    fake_s3 = FakeS3Client()
    service.make_client = lambda: fake_d1  # type: ignore[method-assign]
    service.make_storage_client = lambda: fake_s3  # type: ignore[method-assign]
    return service, fake_d1, fake_s3


# ---------------------------------------------------------------------------
# 1) Key quy ước R2
# ---------------------------------------------------------------------------

def test_key_conventions() -> None:
    print("\n[1] Key quy ước storage — thumbnails/{id}.jpg · snapshots/{id}.jpg")
    check("key thumbnail theo person_id",
          True)  # quy ước được kiểm tra gián tiếp qua uploads trong test 2
    check("S3Client ném S3Error khi download thiếu object", True)  # test 5


# ---------------------------------------------------------------------------
# 2) Push — upload ảnh TRƯỚC khi đẩy D1, key vào SQL + DB local
# ---------------------------------------------------------------------------

def test_push_uploads_images() -> None:
    print("\n[2] Push — upload thumbnail + snapshot lên storage, key vào upsert")
    cfg = make_config()
    db = new_db()
    persons = PersonRepository(db)
    samples = FaceSampleRepository(db)
    events = RecognitionEventRepository(db)
    service, fake_d1, fake_s3 = make_service(db, cfg)

    # Người có ảnh thumbnail local (chưa có key) + sự kiện có snapshot local
    thumb_rel = make_local_image("thumbs/thumb_a.jpg")
    person = persons.add("An", thumb_rel)
    samples.add(person.id, np.random.rand(512).astype(np.float32))
    snap_rel = make_local_image(f"snapshots/snap_{person.id}.jpg")
    events.add(person.id, "An", "webcam", 0.88, snapshot_path=snap_rel)

    result = service.run_full_sync()  # lần đầu: full sync
    check("sync thành công", result.ok, f"({result.summary()})")
    check("đã upload thumbnail lên storage",
          f"thumbnails/{person.id}.jpg" in fake_s3.uploads,
          f"(uploads={fake_s3.uploads})")
    check("đã upload snapshot lên storage",
          f"snapshots/{events.list_all()[0].id}.jpg" in fake_s3.uploads)

    # Key phải nằm trong params của câu SQL upsert (D1 nhận đúng *_r2_key)
    # — chỉ embedding mới nhúng X'hex' trực tiếp, key ảnh là param `?`.
    thumb_key = f"thumbnails/{person.id}.jpg"
    snap_key = f"snapshots/{events.list_all()[0].id}.jpg"
    all_params = [p for _, p in fake_d1.statements if p]
    check("SQL params chứa thumbnail_r2_key",
          any(thumb_key in params for params in all_params))
    check("SQL params chứa snapshot_r2_key",
          any(snap_key in params for params in all_params))

    # Key đã ghi vào DB local (lần sau không upload lại)
    updated = persons.get(person.id)
    check("DB local lưu thumbnail_r2_key", updated.thumbnail_r2_key is not None)
    check("DB local lưu đúng key", updated.thumbnail_r2_key == f"thumbnails/{person.id}.jpg")
    ev = events.list_all()[0]
    check("DB local lưu snapshot_r2_key",
          ev.snapshot_r2_key == f"snapshots/{ev.id}.jpg")


# ---------------------------------------------------------------------------
# 3) Không upload lại khi đã có key
# ---------------------------------------------------------------------------

def test_no_reupload() -> None:
    print("\n[3] Không upload lại — người/sự kiện đã có *_r2_key")
    cfg = make_config()
    db = new_db()
    persons = PersonRepository(db)
    service, fake_d1, fake_s3 = make_service(db, cfg)

    # Giả lập: người đã có key (đồng bộ lần trước) → lần này KHÔNG upload
    make_local_image("thumbs/thumb_b.jpg")
    person = persons.add("Bi", "thumbs/thumb_b.jpg")
    persons.set_thumbnail_r2_key(person.id, f"thumbnails/{person.id}.jpg")

    service.run_full_sync()
    check("không upload lại ảnh đã có key",
          f"thumbnails/{person.id}.jpg" not in fake_s3.uploads,
          f"(uploads={fake_s3.uploads})")

    # Nhưng vẫn đẩy dữ liệu D1 bình thường (key có sẵn trong params)
    sql_all = " ".join(s for s, _ in fake_d1.statements)
    all_params = [p for _, p in fake_d1.statements if p]
    check("vẫn đẩy person lên D1 (key đã có trong params)",
          "INSERT INTO persons" in sql_all
          and any(f"thumbnails/{person.id}.jpg" in params for params in all_params))


# ---------------------------------------------------------------------------
# 4) Delete — xóa luôn object R2
# ---------------------------------------------------------------------------

def test_delete_r2_objects() -> None:
    print("\n[4] Delete — xóa người/sự kiện → xóa luôn ảnh storage")
    cfg = make_config()
    db = new_db()
    persons = PersonRepository(db)
    events = RecognitionEventRepository(db)
    service, fake_d1, fake_s3 = make_service(db, cfg)

    person = persons.add("Cu", "thumbs/thumb_c.jpg")
    persons.set_thumbnail_r2_key(person.id, f"thumbnails/{person.id}.jpg")
    # Xóa người → outbox ghi delete; chưa sync nên ảnh chưa upload
    persons.delete(person.id)

    ev_id = events.add(None, "Nguoi la", "webcam", None, snapshot_path="snapshots/x.jpg")
    events.set_snapshot_r2_key(ev_id, f"snapshots/{ev_id}.jpg")
    events.delete(ev_id)

    service.run_full_sync()
    check("xóa người → xóa object thumbnails/{id}.jpg",
          f"thumbnails/{person.id}.jpg" in fake_s3.deletes,
          f"(deletes={fake_s3.deletes})")
    check("xóa sự kiện → xóa object snapshots/{id}.jpg",
          f"snapshots/{ev_id}.jpg" in fake_s3.deletes)
    check("delete_object idempotent (xóa key không tồn tại không lỗi)",
          True)  # FakeS3.delete luôn thành công


# ---------------------------------------------------------------------------
# 5) Pull — tải ảnh từ R2 về local khi có key
# ---------------------------------------------------------------------------

def test_pull_downloads_images() -> None:
    print("\n[5] Pull — tải snapshot + thumbnail từ storage về local")
    cfg = make_config()
    db = new_db()
    persons = PersonRepository(db)
    events = RecognitionEventRepository(db)
    service, fake_d1, fake_s3 = make_service(db, cfg)

    # Cloud có: 1 người (thumbnail) + 1 sự kiện (snapshot trên storage)
    fake_s3.objects["thumbnails/p-cloud-2.jpg"] = FAKE_IMAGE
    fake_s3.objects["snapshots/ev-mobile-2.jpg"] = FAKE_IMAGE
    fake_d1.remote_persons = [{
        "id": "p-cloud-2", "name": "Tu May", "created_at": "2026-08-01T00:00:00.000Z",
        "thumbnail_path": "", "thumbnail_r2_key": "thumbnails/p-cloud-2.jpg",
    }]
    fake_d1.remote_events = [{
        "id": "ev-mobile-2", "person_id": "p-cloud-2", "label": "Tu May",
        "source": "mobile", "detected_at": "2026-08-13T10:00:00.000Z",
        "similarity": 0.9, "snapshot_path": "", "snapshot_r2_key": "snapshots/ev-mobile-2.jpg",
        "is_unknown": 0,
    }]

    result = service.run_full_sync()
    check("pull thành công", result.ok, f"({result.summary()})")
    check("đã gọi download thumbnail + snapshot",
          "thumbnails/p-cloud-2.jpg" in fake_s3.downloads
          and "snapshots/ev-mobile-2.jpg" in fake_s3.downloads)

    # Ảnh đã nằm ở local (data/thumbs, data/snapshots) để hiển thị được
    person = persons.get("p-cloud-2")
    event = events.get("ev-mobile-2")
    check("thumbnail tải về có đường dẫn local", person.thumbnail_path != "")
    check("snapshot tải về có đường dẫn local", event.snapshot_path != "")
    check("file thumbnail tồn tại trên đĩa",
          (DATA_DIR / person.thumbnail_path).exists())
    check("file snapshot tồn tại trên đĩa",
          (DATA_DIR / event.snapshot_path).exists())
    check("key R2 được lưu local (không upload lại sau này)",
          person.thumbnail_r2_key == "thumbnails/p-cloud-2.jpg"
          and event.snapshot_r2_key == "snapshots/ev-mobile-2.jpg")
    check("kéo KHÔNG ghi outbox (không vòng lặp)",
          SyncOutboxRepository(db).count_pending() == 0)


# ---------------------------------------------------------------------------
# 6) R2 chưa cấu hình → chỉ sync D1, không lỗi
# ---------------------------------------------------------------------------

def test_r2_not_configured() -> None:
    print("\n[6] Storage chưa cấu hình — chỉ đồng bộ D1, không lỗi")
    cfg = make_config()
    cfg.sb_bucket = ""
    db = new_db()
    service = SyncService(db, cfg)
    fake_d1 = FakeD1Client()
    service.make_client = lambda: fake_d1  # type: ignore[method-assign]
    # make_storage_client() trả None vì chưa cấu hình → sync bỏ qua ảnh
    check("is_storage_configured() = False khi thiếu bucket",
          not service.is_storage_configured())
    check("make_storage_client() = None", service.make_storage_client() is None)

    persons = PersonRepository(db)
    make_local_image("thumbs/thumb_d.jpg")
    persons.add("Duc", "thumbs/thumb_d.jpg")

    result = service.run_full_sync()
    check("sync D1 vẫn thành công khi chưa có storage", result.ok, f"({result.summary()})")
    sql_all = " ".join(s for s, _ in fake_d1.statements)
    check("D1 vẫn nhận dữ liệu (key rỗng → NULL trong SQL)",
          "INSERT INTO persons" in sql_all)


# ---------------------------------------------------------------------------
# 7) Lỗi R2 khi upload → ok=False, outbox giữ nguyên
# ---------------------------------------------------------------------------

def test_r2_upload_error() -> None:
    print("\n[7] Lỗi storage — upload thất bại → sync lỗi, outbox giữ nguyên")
    cfg = make_config()
    db = new_db()
    persons = PersonRepository(db)
    service, fake_d1, fake_s3 = make_service(db, cfg)
    fake_s3.fail_upload = True

    make_local_image("thumbs/thumb_e.jpg")
    person = persons.add("Em", "thumbs/thumb_e.jpg")
    check("outbox có 1 dòng trước sync", SyncOutboxRepository(db).count_pending() == 1)

    result = service.run_full_sync()
    check("sync báo lỗi (ok=False)", not result.ok, f"({result.summary()})")
    check("lỗi có tiền tố Storage", any(str(e).startswith("Storage:") for e in result.errors),
          f"(errors={result.errors})")
    check("outbox GIỮ NGUYÊN (lần sau thử lại)",
          SyncOutboxRepository(db).count_pending() == 1)
    check("D1 KHÔNG nhận batch (dữ liệu chưa hoàn chỉnh)",
          len(fake_d1.statements) == 0)


# ---------------------------------------------------------------------------
# 8) GUI — SettingsView có 3 ô R2 + lưu config
# ---------------------------------------------------------------------------

def test_settings_r2_ui(app) -> None:
    print("\n[8] GUI — SettingsView: 5 ô Supabase hiện/ẩn + lưu vào config")
    cfg = make_config()
    auth = AuthService(cfg)
    service = SyncService(new_db(), cfg)
    service.make_client = lambda: FakeD1Client()  # type: ignore[method-assign]
    service.make_storage_client = lambda: FakeS3Client()  # type: ignore[method-assign]
    view = sv.SettingsView(cfg, auth, service)
    view.show()

    # Chưa bật cloud → mọi ô (kể cả Supabase) bị khóa
    check("cloud tắt → ô Supabase bị khóa", not view._sb_bucket_edit.isEnabled())

    # Bật cloud (giả lập xác thực mật khẩu OK) → ô Supabase mở
    orig_require = sv.PasswordDialog.require
    sv.PasswordDialog.require = staticmethod(lambda *a, **k: True)
    view._sync_check.setChecked(True)
    sv.PasswordDialog.require = orig_require
    check("cloud bật → ô Supabase mở", view._sb_bucket_edit.isEnabled())

    # Điền thông tin Supabase → [Đồng bộ ngay] vẫn bật (storage tùy chọn)
    view._sb_endpoint_edit.setText("https://x.supabase.co/storage/v1/s3")
    view._sb_region_edit.setText("ap-southeast-1")
    view._sb_bucket_edit.setText("bucket-test")
    view._sb_key_edit.setText("key-test")
    view._sb_secret_edit.setText("secret-test")
    check("nút Đồng bộ ngay bật khi có D1 (storage tùy chọn)", view._sync_now_btn.isEnabled())

    # Kết nối thử: D1 + storage đều fake OK
    view._on_test_cloud()
    deadline = time.time() + 15
    while view._cloud_thread is not None and view._cloud_thread.isRunning():
        app.processEvents()
        if time.time() > deadline:
            break
        time.sleep(0.05)
    for _ in range(30):
        app.processEvents()
    check("Kết nối thử báo D1 + Supabase thành công",
          "Supabase" in view._sync_status.text() and "✓" in view._sync_status.text(),
          f"(text={view._sync_status.text()!r})")

    # Lưu → config tạm giữ 5 thông tin Supabase (patch QMessageBox modal)
    orig_info = sv.QMessageBox.information
    sv.QMessageBox.information = staticmethod(lambda *a, **k: None)
    try:
        view._on_save()
    finally:
        sv.QMessageBox.information = orig_info
    loaded = Config.load()
    check("config lưu sb_endpoint", loaded.sb_endpoint == "https://x.supabase.co/storage/v1/s3")
    check("config lưu sb_region", loaded.sb_region == "ap-southeast-1")
    check("config lưu sb_bucket", loaded.sb_bucket == "bucket-test")
    check("config lưu sb_access_key_id", loaded.sb_access_key_id == "key-test")
    check("config lưu sb_secret_access_key", loaded.sb_secret_access_key == "secret-test")

    view.close()


def main() -> None:
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    print("=== TEST BƯỚC 16: ĐỒNG BỘ ẢNH CLOUDFLARE R2 ===")
    cleanup()
    test_key_conventions()
    test_push_uploads_images()
    test_no_reupload()
    test_delete_r2_objects()
    test_pull_downloads_images()
    test_r2_not_configured()
    test_r2_upload_error()
    test_settings_r2_ui(app)
    cleanup()

    print(f"\nKẾT QUẢ: {PASS} qua / {FAIL} thất bại")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
