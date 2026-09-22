"""Bước 15 — Kiểm tra Đồng bộ Cloud (Cloudflare D1).

Toàn bộ test chạy OFFLINE — dùng FakeD1Client giả lập D1 (ghi nhận câu SQL,
trả dữ liệu giả) để không phụ thuộc mạng/tài khoản thật:

1. BLOB hex: literal X'...' dựng đúng + đọc lại bytes.
2. Outbox: mọi thao tác ghi local (thêm/sửa/xóa người, mẫu, sự kiện) đều
   ghi hàng đợi; thao tác mới nhất thay thế thao tác cũ cùng thực thể.
3. SyncService.push: lần đầu đẩy TOÀN BỘ dữ liệu + đánh dấu đã đồng bộ;
   lần sau chỉ đẩy outbox; SQL upsert đúng (embedding dạng X'hex').
4. SyncService.pull: kéo sự kiện/người/mẫu từ cloud về khi local chưa có;
   KHÔNG tạo vòng lặp outbox; local thắng khi trùng id.
5. Lỗi mềm: mất kết nối / sai token → result.ok=False kèm lỗi, không crash.
6. GUI SettingsView: 3 ô thông tin cloud hiện/ẩn theo checkbox, lưu vào
   config tạm, nút [Đồng bộ ngay] chạy + hiển thị trạng thái.

Chạy:  .venv\\Scripts\\python.exe scripts\\step_15_gui_test.py
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
import app.infrastructure.d1_client as d1  # noqa: E402
import app.ui.settings_view as sv  # noqa: E402
from app.config import Config  # noqa: E402
from app.infrastructure.db import Database  # noqa: E402
from app.infrastructure.repositories import (  # noqa: E402
    FaceSampleRepository,
    PersonRepository,
    RecognitionEventRepository,
    SyncOutboxRepository,
)
from app.services.auth import AuthService  # noqa: E402
from app.services.sync import SyncService  # noqa: E402

# File TẠM — tuyệt đối không đụng config.json / app.db thật của người dùng
TEMP_CONFIG = PROJECT_ROOT / "data" / "test_config_15.json"
TEMP_DB = PROJECT_ROOT / "data" / "test_db_15.db"

PASS = 0
FAIL = 0
OPEN_DBS: list[Database] = []


def new_db() -> Database:
    """Tạo Database gắn file TẠM — đăng ký để cleanup đóng kết nối (Windows
    khóa file đang mở → không xóa được nếu quên close)."""
    db = Database(TEMP_DB)
    OPEN_DBS.append(db)
    return db


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
    return cfg


# ---------------------------------------------------------------------------
# FakeD1Client — giả lập D1: ghi nhận SQL, trả dữ liệu giả cho SELECT
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


def make_service(config: Config) -> SyncService:
    service = SyncService(new_db(), config)
    service.make_client = lambda: FakeD1Client()  # type: ignore[method-assign]
    return service


# ---------------------------------------------------------------------------
# 1) BLOB hex
# ---------------------------------------------------------------------------

def test_blob_hex() -> None:
    print("\n[1] BLOB hex — literal X'...' + đọc lại bytes")
    blob = np.arange(8, dtype=np.float32).tobytes()
    literal = d1.blob_hex_literal(blob)
    check("literal bắt đầu X'", literal.startswith("X'") and literal.endswith("'"),
          f"({literal[:20]}...)")
    check("độ dài literal = X'...' (3 ký tự thêm) + hex gấp đôi bytes",
          len(literal) == 3 + 2 * len(blob))
    back = d1.hex_to_bytes(literal[2:-1])
    check("đọc lại đúng bytes", back == blob)

    # _bind: None → NULL literal trong SQL, param giữ lại (REST chỉ nhận string)
    from app.services.sync import _bind
    sql, params = _bind("INSERT INTO t (a, b) VALUES (?, ?)", ["x", None])
    check("_bind: None → NULL trong SQL", sql == "INSERT INTO t (a, b) VALUES (?, NULL)",
          f"({sql!r})")
    check("_bind: param thật giữ lại", params == ["x"], f"({params!r})")


# ---------------------------------------------------------------------------
# 2) Outbox — mọi thao tác ghi đều vào hàng đợi
# ---------------------------------------------------------------------------

def test_outbox_recording() -> None:
    print("\n[2] Outbox — ghi hàng đợi cho mọi thao tác ghi local")
    cfg = make_config()
    db = new_db()
    persons = PersonRepository(db)
    samples = FaceSampleRepository(db)
    events = RecognitionEventRepository(db)
    outbox = SyncOutboxRepository(db)

    person = persons.add("Nguyen Van A", "thumbs/a.jpg")
    pending = outbox.list_pending()
    check("thêm người → outbox upsert",
          outbox.count_pending() == 1
          and pending[0][1] == "person" and pending[0][2] == person.id
          and pending[0][3] == "upsert")

    persons.rename(person.id, "Nguyen Van B")
    pending = outbox.list_pending()
    check("đổi tên → vẫn 1 dòng (thay thế dòng cũ)",
          outbox.count_pending() == 1 and pending[0][3] == "upsert")

    sample = samples.add(person.id, np.random.rand(512).astype(np.float32))
    event_id = events.add(person.id, "Nguyen Van B", "webcam", 0.91)
    check("thêm mẫu + sự kiện → +2 dòng", outbox.count_pending() == 3)

    events.delete(event_id)
    check("xóa sự kiện → op delete", outbox.count_pending() == 3)

    persons.delete(person.id)
    pending = outbox.list_pending()
    person_row = [p for p in pending if p[1] == "person" and p[2] == person.id]
    check("xóa người → op delete (thay thế upsert trước đó)",
          len(person_row) == 1 and person_row[0][3] == "delete")
    # 3 dòng còn lại: person-delete + sample-upsert + event-delete
    check("tổng 3 dòng còn lại (person delete + mẫu + sự kiện delete)",
          outbox.count_pending() == 3)


# ---------------------------------------------------------------------------
# 3) Push — lần đầu đẩy toàn bộ, lần sau chỉ outbox
# ---------------------------------------------------------------------------

def test_push() -> None:
    print("\n[3] Push — lần đầu full sync, lần sau chỉ outbox")
    cfg = make_config()
    db = new_db()
    persons = PersonRepository(db)
    samples = FaceSampleRepository(db)
    events = RecognitionEventRepository(db)
    service = SyncService(db, cfg)
    fake = FakeD1Client()
    service.make_client = lambda: fake  # type: ignore[method-assign]

    person = persons.add("Chi", "thumbs/c.jpg")
    samples.add(person.id, np.random.rand(512).astype(np.float32))
    samples.add(person.id, np.random.rand(512).astype(np.float32))
    events.add(person.id, "Chi", "webcam", 0.88)

    # Lần 1: full sync
    result = service.run_full_sync()
    check("lần 1 thành công", result.ok, f"({result.summary()})")
    check("ensure_schema được gọi", fake.ensure_called >= 1)
    sql_all = " ".join(s for s, _ in fake.statements)
    check("đẩy person (INSERT persons)", "INSERT INTO persons" in sql_all)
    check("đẩy embedding dạng X'hex'", "X'" in sql_all)
    check("đẩy sự kiện (INSERT recognition_events)", "INSERT INTO recognition_events" in sql_all)
    check("result.pushed đủ số dòng (1 người + 2 mẫu + 1 sự kiện + 1 outbox)",
          result.pushed >= 5, f"(pushed={result.pushed})")
    check("outbox sạch sau sync", SyncOutboxRepository(db).count_pending() == 0)

    # Lần 2: không có thay đổi mới → không đẩy gì thêm
    fake.statements.clear()
    result2 = service.run_full_sync()
    check("lần 2 không đẩy thêm (không có outbox)",
          len(fake.statements) == 0 and result2.ok, f"(statements={len(fake.statements)})")

    # Thêm 1 sự kiện mới → chỉ đẩy đúng sự kiện đó
    events.add(person.id, "Chi", "photo", 0.75)
    fake.statements.clear()
    service.run_full_sync()
    sql2 = " ".join(s for s, _ in fake.statements)
    check("lần 3 chỉ đẩy outbox mới (không đẩy lại toàn bộ)",
          "INSERT INTO recognition_events" in sql2
          and sql2.count("INSERT INTO persons") == 0)


# ---------------------------------------------------------------------------
# 4) Pull — kéo dữ liệu cloud về khi local chưa có, không vòng lặp
# ---------------------------------------------------------------------------

def test_pull() -> None:
    print("\n[4] Pull — kéo sự kiện/người/mẫu từ cloud, local thắng khi trùng")
    cfg = make_config()
    db = new_db()
    persons = PersonRepository(db)
    events = RecognitionEventRepository(db)
    service = SyncService(db, cfg)
    fake = FakeD1Client()

    # Cloud có: 1 sự kiện mobile + 1 người + 1 mẫu embedding
    emb = np.random.rand(512).astype(np.float32)
    fake.remote_events = [{
        "id": "ev-mobile-1", "person_id": None, "label": "Nguoi la",
        "source": "mobile", "detected_at": "2026-08-13T10:00:00.000Z",
        "similarity": None, "snapshot_path": "", "snapshot_r2_key": None,
        "is_unknown": 1,
    }]
    fake.remote_persons = [{
        "id": "p-cloud-1", "name": "Tu Cloud", "created_at": "2026-08-01T00:00:00.000Z",
        "thumbnail_path": "", "thumbnail_r2_key": None,
    }]
    fake.remote_samples = [{
        "id": "s-cloud-1", "person_id": "p-cloud-1",
        "embedding_hex": emb.tobytes().hex(), "quality": 0.9,
        "captured_at": "2026-08-01T00:00:00.000Z",
    }]
    service.make_client = lambda: fake  # type: ignore[method-assign]

    result = service.run_full_sync()
    check("pull thành công", result.ok, f"({result.summary()})")
    check("kéo 3 dòng về", result.pulled == 3, f"(pulled={result.pulled})")
    check("sự kiện mobile có trong local",
          events.get("ev-mobile-1") is not None
          and events.get("ev-mobile-1").label == "Nguoi la")
    check("người từ cloud có trong local", persons.get("p-cloud-1") is not None)
    check("kéo KHÔNG ghi outbox (không vòng lặp)",
          SyncOutboxRepository(db).count_pending() == 0)

    # Local thắng khi trùng id: cloud sửa tên nhưng local giữ nguyên
    fake.remote_persons[0]["name"] = "Ten Da Doi Tren Cloud"
    fake.remote_events = []  # không còn sự kiện mới
    fake.remote_samples = []
    result2 = service.run_full_sync()
    check("lần 2 không kéo gì mới", result2.pulled == 0, f"(pulled={result2.pulled})")
    check("local giữ tên cũ (local thắng)", persons.get("p-cloud-1").name == "Tu Cloud")


# ---------------------------------------------------------------------------
# 5) Lỗi mềm
# ---------------------------------------------------------------------------

def test_errors() -> None:
    print("\n[5] Lỗi mềm — mất mạng / sai token / chưa cấu hình")
    # a) Chưa cấu hình đủ
    cfg = make_config()
    cfg.cloud_api_token = ""
    service = SyncService(new_db(), cfg)
    result = service.run_full_sync()
    check("thiếu token → ok=False + lỗi rõ ràng",
          not result.ok and any("cấu hình" in e for e in result.errors))

    # b) D1 từ chối (sai token / SQL lỗi)
    class RejectClient(FakeD1Client):
        def query(self, sql, params=None):
            raise d1.D1Error("HTTP 400 — token không hợp lệ")
        def query_batch(self, statements):
            raise d1.D1Error("HTTP 400 — token không hợp lệ")

    cfg2 = make_config()
    service2 = SyncService(new_db(), cfg2)
    service2.make_client = lambda: RejectClient()  # type: ignore[method-assign]
    result2 = service2.run_full_sync()
    check("sai token → ok=False + có lỗi",
          not result2.ok and len(result2.errors) >= 1,
          f"({result2.errors[:1]})")
    check("không crash — app vẫn chạy tiếp", True)


# ---------------------------------------------------------------------------
# 6) GUI — SettingsView với thông tin cloud
# ---------------------------------------------------------------------------

def test_settings_cloud_ui(app) -> None:
    print("\n[6] GUI — SettingsView: ô cloud + [Đồng bộ ngay] hiển thị trạng thái")
    cfg = make_config()
    auth = AuthService(cfg)
    service = make_service(cfg)  # make_client → FakeD1Client
    view = sv.SettingsView(cfg, auth, service)
    view.show()

    # Checkbox tắt → ô nhập bị khóa
    check("cloud tắt → ô nhập bị khóa", not view._account_edit.isEnabled())
    check("nút Đồng bộ ngay bị khóa khi chưa bật", not view._sync_now_btn.isEnabled())

    # Bật cloud (giả lập xác thực mật khẩu OK) → ô nhập mở
    orig_require = sv.PasswordDialog.require
    sv.PasswordDialog.require = staticmethod(lambda *a, **k: True)
    view._sync_check.setChecked(True)
    sv.PasswordDialog.require = orig_require
    check("cloud bật → ô nhập mở", view._account_edit.isEnabled())

    # Đã có sẵn thông tin trong config → nút [Đồng bộ ngay] bật
    check("có đủ thông tin → nút Đồng bộ ngay bật", view._sync_now_btn.isEnabled())

    # Bấm [Đồng bộ ngay] → chờ worker xong → trạng thái "Đã đồng bộ"
    view._on_sync_now()
    deadline = time.time() + 15
    while view._cloud_thread is not None and view._cloud_thread.isRunning():
        app.processEvents()
        if time.time() > deadline:
            break
        time.sleep(0.05)
    for _ in range(30):
        app.processEvents()

    check("thread cloud đã dọn sạch",
          view._cloud_thread is None and view._cloud_worker is None)
    check("nhãn trạng thái báo đồng bộ thành công",
          "Đã đồng bộ" in view._sync_status.text(),
          f"(text={view._sync_status.text()!r})")

    # Lưu → config tạm giữ 3 thông tin cloud (patch QMessageBox modal)
    orig_info = sv.QMessageBox.information
    sv.QMessageBox.information = staticmethod(lambda *a, **k: None)
    try:
        view._on_save()
    finally:
        sv.QMessageBox.information = orig_info
    loaded = Config.load()
    check("config lưu account_id", loaded.cloud_account_id == cfg.cloud_account_id)
    check("config lưu database_id", loaded.cloud_database_id == cfg.cloud_database_id)
    check("config lưu api_token", loaded.cloud_api_token == cfg.cloud_api_token)

    view.close()


def main() -> None:
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    print("=== TEST BƯỚC 15: ĐỒNG BỘ CLOUD (CLOUDFLARE D1) ===")
    cleanup()
    test_blob_hex()
    test_outbox_recording()
    test_push()
    test_pull()
    test_errors()
    test_settings_cloud_ui(app)
    cleanup()

    print(f"\nKẾT QUẢ: {PASS} qua / {FAIL} thất bại")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
