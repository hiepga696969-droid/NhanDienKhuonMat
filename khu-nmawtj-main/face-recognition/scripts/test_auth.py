"""Kiểm tra logic xác thực (app/services/auth.py) — không cần GUI.

Chạy:  .venv\\Scripts\\python.exe scripts\\test_auth.py

QUAN TRỌNG: test dùng file config TẠM (data/test_auth_config.json) — KHÔNG
đụng vào config.json thật. Trước đây set_password ghi đè mật khẩu thật của
người dùng (sao lưu/khôi phục mong manh — nếu test bị ngắt giữa chừng là
mất mật khẩu); giờ trỏ CONFIG_PATH sang file tạm, an toàn tuyệt đối.

Cập nhật theo spec password-management-spec.md: khóa MỀM 10s (không còn khóa
cứng 60s), chính sách mật khẩu mới (8 ký tự + thường/hoa/số), 2 câu hỏi bảo
mật hash argon2 (so khớp trim+lowercase, KHÔNG bỏ dấu tiếng Việt).
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from argon2 import PasswordHasher  # noqa: E402

import app.config as config_module  # noqa: E402
from app.config import Config  # noqa: E402
from app.services.auth import (  # noqa: E402
    MIN_ANSWER_LENGTH,
    MIN_PASSWORD_LENGTH,
    SOFT_LOCK_AFTER,
    SOFT_LOCK_WAIT_SECONDS,
    AuthService,
    normalize_answer,
    validate_password_strength,
)

# File config TẠM — xóa nếu còn từ lần chạy trước
TEMP_CONFIG = PROJECT_ROOT / "data" / "test_auth_config.json"
for suffix in ("", "-bak"):
    Path(str(TEMP_CONFIG) + suffix).unlink(missing_ok=True)
# Trỏ CONFIG_PATH sang file tạm (Config.save/load tìm đường dẫn lúc GỌI)
config_module.CONFIG_PATH = TEMP_CONFIG

passed = True

try:
    # Dùng Config tạm (không đọc file thật — người dùng đã đặt mật khẩu thật)
    config = Config(password_hash=None)
    auth = AuthService(config)

    # 1) Chưa có mật khẩu
    assert not auth.has_password, "Lần đầu dùng phải chưa có mật khẩu"
    assert auth.verify("anything") is False, "Chưa đặt mật khẩu thì verify phải False"
    assert not auth.has_security_questions, "Chưa có câu hỏi bảo mật"
    print("[1] Trạng thái chưa đặt mật khẩu: OK")

    # 2) Chính sách mật khẩu mới (FR-1)
    weak = "abc123"  # thiếu chữ hoa + quá ngắn
    errors = validate_password_strength(weak)
    assert errors, "abc123 phải bị từ chối theo chính sách mới"
    assert validate_password_strength("Long2026") == [], "Long2026 phải hợp lệ"
    try:
        auth.set_password(weak)
        raise AssertionError("set_password phải ném ValueError với mật khẩu yếu")
    except ValueError:
        pass
    print("[2] Chính sách mật khẩu mới (≥8 + thường/hoa/số): OK")

    # 3) Mật khẩu cũ yếu vẫn mở khóa được (không áp chính sách cho hash cũ)
    old_hash = PasswordHasher().hash("abcd")  # giả lập mật khẩu 4 ký tự từ bản cũ
    old_auth = AuthService(Config(password_hash=old_hash))
    assert old_auth.verify("abcd") is True, "Mật khẩu cũ ngắn vẫn phải mở khóa được"
    print("[3] Mật khẩu cũ yếu vẫn mở khóa bình thường: OK")

    # 4) Đặt mật khẩu hợp lệ + verify đúng/sai
    auth.set_password("Long2026")
    assert auth.has_password, "Sau khi đặt phải có mật khẩu"
    assert auth.verify("Long2026") is True, "Mật khẩu đúng phải verify True"
    assert auth.verify("sai-mat-khau") is False, "Mật khẩu sai phải verify False"
    assert auth.attempts_remaining == SOFT_LOCK_AFTER - 1, "Sai 1 lần → còn N-1 lần thử"
    assert TEMP_CONFIG.exists(), "Mật khẩu phải được ghi vào file config tạm"
    print("[4] Đặt mật khẩu + verify + đếm lần thử còn lại: OK")

    # 5) Khóa MỀM: sau 5 lần sai liên tiếp → phải chờ 10s giữa lần thử
    for _ in range(SOFT_LOCK_AFTER):
        auth.verify("sai")
    assert auth.is_locked, f"Sau {SOFT_LOCK_AFTER} lần sai phải kích hoạt chờ"
    assert auth.attempts_remaining == 0
    remaining = auth.lockout_remaining
    assert 0 < remaining <= SOFT_LOCK_WAIT_SECONDS, f"Thời gian chờ lạ: {remaining}"
    # Trong lúc chờ: mật khẩu đúng cũng không mở được (UI chặn — verify trả False)
    assert auth.verify("Long2026") is False, "Đang chờ thì không verify được"
    # Giả lập hết thời gian chờ → mật khẩu đúng mở khóa + reset bộ đếm
    auth._last_fail_at = 0.0  # chỉ để test nhanh (không phải đợi 10s thật)
    assert auth.verify("Long2026") is True
    assert auth.attempts_remaining == SOFT_LOCK_AFTER, "Đúng mật khẩu → reset bộ đếm"
    print(f"[5] Khóa mềm sau {SOFT_LOCK_AFTER} lần sai (chờ ≤{SOFT_LOCK_WAIT_SECONDS}s): OK")

    # 6) Câu hỏi bảo mật (FR-5/6) — hash + so khớp chuẩn hóa, KHÔNG bỏ dấu
    auth.set_security_questions(
        "Quê quán của bạn?", "  Hà Nội ",
        "Tên vật nuôi đầu tiên?", "mèo mun",
    )
    assert auth.has_security_questions, "Phải có đủ 2 câu hỏi sau khi thiết lập"
    assert normalize_answer("  Hà Nội ") == "hà nội"
    assert auth.verify_security_answers("ha noi", "meo mun") is False, \
        "Không bỏ dấu: 'ha noi' ≠ 'hà nội'"
    assert auth.verify_security_answers("hà nội", "mèo mun") is True, \
        "Đúng cả 2 câu (trim+lowercase) phải True"
    # Lưu hash — không lưu câu trả lời plaintext trong config
    raw = TEMP_CONFIG.read_text(encoding="utf-8")
    assert "hà nội" not in raw and "mèo mun" not in raw, "Câu trả lời phải hash"
    print("[6] Câu hỏi bảo mật: hash + so khớp không bỏ dấu: OK")

    # 7) Bộ đếm sai DÙNG CHUNG giữa mật khẩu và câu trả lời
    auth._reset_attempts()  # bắt đầu sạch
    for _ in range(SOFT_LOCK_AFTER):
        auth.verify_security_answers("sai1", "sai2")
    assert auth.is_locked, "Sai câu trả lời nhiều lần phải kích hoạt chờ chung"
    assert auth.verify("Long2026") is False, "Đang chờ do bảo mật sai → chặn cả mật khẩu"
    auth._last_fail_at = 0.0
    assert auth.verify("Long2026") is True, "Hết chờ → mật khẩu đúng vẫn mở khóa"
    print("[7] Bộ đếm dùng chung giữa mật khẩu & câu hỏi bảo mật: OK")

    # 8) Câu hỏi trùng nhau / trả lời quá ngắn bị từ chối
    try:
        auth.set_security_questions("Câu 1?", "trả lời một", "câu 1?", "trả lời hai")
        raise AssertionError("Hai câu hỏi trùng nhau phải bị từ chối")
    except ValueError:
        pass
    try:
        auth.set_security_questions("Câu 1?", "ab", "Câu 2?", "câu trả lời")
        raise AssertionError(f"Trả lời < {MIN_ANSWER_LENGTH} ký tự phải bị từ chối")
    except ValueError:
        pass
    print("[8] Validate câu hỏi (không trùng, trả lời đủ dài): OK")

except AssertionError as exc:
    passed = False
    print(f"THẤT BẠI: {exc}")

# Dọn file tạm — không cần khôi phục gì vì chưa bao giờ đụng config.json thật
TEMP_CONFIG.unlink(missing_ok=True)

if passed:
    print("\n=== TẤT CẢ TEST XÁC THỰC ĐỀU QUA ===")
else:
    print("\n=== CÓ TEST THẤT BẠI ===")
    sys.exit(1)
