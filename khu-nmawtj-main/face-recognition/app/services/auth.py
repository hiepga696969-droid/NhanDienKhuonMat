"""Xác thực mật khẩu ứng dụng — argon2 + khóa MỀM + câu hỏi bảo mật.

Thay đổi theo spec `password-management-spec.md` (2026-09-03):
  - FR-1  Chính sách mật khẩu MỚI: ≥ 8 ký tự + chữ thường + chữ hoa + chữ số.
         Mật khẩu CŨ yếu (đặt trước bản này) VẪN mở khóa bình thường —
         chính sách chỉ áp cho mật khẩu được tạo/đổi từ bản này.
  - FR-2  Khóa MỀM thay khóa cứng 60s: sau ``SOFT_LOCK_AFTER`` lần sai liên
         tiếp, mỗi lần thử kế tiếp phải chờ ``SOFT_LOCK_WAIT_SECONDS`` giây
         (không khóa hẳn — chỉ làm chậm kẻ dò). Bộ đếm reset khi xác thực
         đúng và DÙNG CHUNG cho mọi đường xác thực (mật khẩu + câu hỏi bảo
         mật) để không né được qua đường khác (đã chốt).
  - FR-5/6  2 câu hỏi bảo mật tự đặt để khôi phục khi quên mật khẩu. Câu
         trả lời chỉ lưu hash argon2; so khớp: trim + gộp khoảng trắng +
         lowercase, KHÔNG bỏ dấu tiếng Việt (đã chốt).

Nguyên tắc an toàn giữ nguyên:
  - KHÔNG lưu mật khẩu / câu trả lời dạng thô — chỉ lưu hash argon2.
  - Hash lưu trong config.json (cấu hình), tách khỏi database người dùng.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

from app.config import Config

# ── Chính sách mật khẩu (FR-1 — ĐƠN GIẢN HÓA theo yêu cầu người dùng) ─
# Chỉ cần tối thiểu 4 ký tự bất kỳ — KHÔNG bắt buộc chữ hoa / chữ số.
MIN_PASSWORD_LENGTH = 4
MIN_ANSWER_LENGTH = 3        # câu trả lời bảo mật tối thiểu (sau chuẩn hóa)

# Chuỗi mô tả quy tắc — hiển thị ở màn hình khóa / hộp thoại đổi mật khẩu
REQUIREMENTS_TEXT = (
    f"Tối thiểu {MIN_PASSWORD_LENGTH} ký tự (tùy ý — không bắt buộc chữ hoa/chữ số)"
)

# ── Khóa MỀM (FR-2) ────────────────────────────────────────────────
SOFT_LOCK_AFTER = 5          # sai liên tiếp bao nhiêu lần thì kích hoạt chờ
SOFT_LOCK_WAIT_SECONDS = 10  # phải chờ bao lâu giữa 2 lần thử khi đã sai nhiều


def validate_password_strength(password: str) -> list[str]:
    """Trả danh sách lỗi (rỗng = hợp lệ) theo chính sách ĐƠN GIẢN.

    Chỉ yêu cầu độ dài tối thiểu — UI gọi hàm này để hiện lỗi chi tiết;
    ``set_password`` cũng dùng để chặn (phòng người gọi bỏ qua validate).
    """
    errors: list[str] = []
    if len(password) < MIN_PASSWORD_LENGTH:
        errors.append(f"Mật khẩu tối thiểu {MIN_PASSWORD_LENGTH} ký tự")
    return errors


def normalize_answer(text: str) -> str:
    """Chuẩn hóa câu trả lời bảo mật trước khi hash/so khớp.

    Trim + gộp khoảng trắng thừa + lowercase. KHÔNG bỏ dấu tiếng Việt
    (đã chốt): "Hà Nội" khác "ha noi".
    """
    return " ".join(text.split()).lower()


@dataclass
class AuthService:
    """Quản lý mật khẩu + câu hỏi bảo mật + khóa mềm khi nhập sai nhiều."""

    config: Config
    _hasher: PasswordHasher = field(default_factory=PasswordHasher)
    # Số lần nhập sai LIÊN TIẾP (chưa reset) — dùng chung mọi đường xác thực
    _failed_attempts: int = 0
    # Thời điểm lần thất bại gần nhất — tính cooldown giữa 2 lần thử (FR-2)
    _last_fail_at: float = 0.0

    # ---------------------------------------------------------
    # Trạng thái
    # ---------------------------------------------------------

    @property
    def has_password(self) -> bool:
        """Đã đặt mật khẩu chưa (lần đầu dùng → chưa)."""
        return bool(self.config.password_hash)

    @property
    def has_security_questions(self) -> bool:
        """Đã thiết lập đủ 2 câu hỏi + hash câu trả lời chưa."""
        cfg = self.config
        return bool(
            cfg.security_question_1
            and cfg.security_question_2
            and cfg.security_answer_hash_1
            and cfg.security_answer_hash_2
        )

    @property
    def attempts_remaining(self) -> int:
        """Số lần thử còn lại TRƯỚC khi bị ép chờ (FR-3 hiển thị)."""
        return max(0, SOFT_LOCK_AFTER - self._failed_attempts)

    @property
    def is_locked(self) -> bool:
        """Đang phải CHỜ giữa 2 lần thử (khóa mềm) hay không.

        Giữ tên ``is_locked`` cho tương thích UI cũ; ngữ nghĩa mới = đang
        trong khoảng chờ SOFT_LOCK_WAIT_SECONDS sau lần sai thứ
        SOFT_LOCK_AFTER trở đi.
        """
        if self._failed_attempts < SOFT_LOCK_AFTER:
            return False
        return time.time() - self._last_fail_at < SOFT_LOCK_WAIT_SECONDS

    @property
    def lockout_remaining(self) -> int:
        """Số giây còn phải chờ (ceil); 0 = hết chờ."""
        if not self.is_locked:
            return 0
        return max(
            0,
            math.ceil(
                SOFT_LOCK_WAIT_SECONDS - (time.time() - self._last_fail_at)
            ),
        )

    # ---------------------------------------------------------
    # Mật khẩu
    # ---------------------------------------------------------

    def set_password(self, new_password: str) -> None:
        """Đặt mật khẩu MỚI (chính sách FR-1); ném ValueError nếu yếu."""
        errors = validate_password_strength(new_password)
        if errors:
            raise ValueError("\n".join(errors))
        self.config.password_hash = self._hasher.hash(new_password)
        self.config.save()
        # Đặt lại trạng thái thử — mật khẩu vừa đổi, không giữ nợ thất bại
        self._failed_attempts = 0
        self._last_fail_at = 0.0

    def verify(self, password: str) -> bool:
        """Kiểm tra mật khẩu; ghi nhận lần sai và kích hoạt chờ khi đủ ngưỡng.

        Lưu ý: UI nên kiểm tra ``is_locked`` TRƯỚC và hiện đếm ngược;
        nếu gọi trong lúc đang chờ, trả False (không đếm thêm).
        """
        if not self.config.password_hash:
            return False
        if self.is_locked:
            return False  # đang phải chờ giữa 2 lần thử — phòng thủ thêm
        try:
            self._hasher.verify(self.config.password_hash, password)
        except (VerificationError, InvalidHashError):
            self._record_failure()
            return False
        self._reset_attempts()
        return True

    # ---------------------------------------------------------
    # Câu hỏi bảo mật (FR-5/6)
    # ---------------------------------------------------------

    def set_security_questions(
        self, q1: str, a1: str, q2: str, a2: str
    ) -> None:
        """Thiết lập / cập nhật 2 câu hỏi + câu trả lời (hash argon2).

        Ném ValueError kèm lý do nếu: câu hỏi trống / trùng nhau, hoặc
        câu trả lời (sau chuẩn hóa) ngắn hơn MIN_ANSWER_LENGTH.
        """
        q1, q2 = q1.strip(), q2.strip()
        if not q1 or not q2:
            raise ValueError("Câu hỏi bảo mật không được để trống")
        if q1.lower() == q2.lower():
            raise ValueError("Hai câu hỏi bảo mật không được trùng nhau")
        n1, n2 = normalize_answer(a1), normalize_answer(a2)
        if len(n1) < MIN_ANSWER_LENGTH or len(n2) < MIN_ANSWER_LENGTH:
            raise ValueError(
                f"Câu trả lời tối thiểu {MIN_ANSWER_LENGTH} ký tự"
            )
        self.config.security_question_1 = q1
        self.config.security_question_2 = q2
        self.config.security_answer_hash_1 = self._hasher.hash(n1)
        self.config.security_answer_hash_2 = self._hasher.hash(n2)
        self.config.save()

    def verify_security_answers(self, a1: str, a2: str) -> bool:
        """Kiểm tra trả lời CẢ 2 câu (đã chốt) — dùng chung bộ đếm sai.

        Sai (hoặc thiếu câu hỏi) → ghi nhận như 1 lần sai mật khẩu và chịu
        cooldown giống hệt (không né được qua đường bảo mật).
        """
        if not self.has_security_questions:
            return False
        if self.is_locked:
            return False
        try:
            ok1 = self._hasher.verify(
                self.config.security_answer_hash_1, normalize_answer(a1)
            )
            ok2 = self._hasher.verify(
                self.config.security_answer_hash_2, normalize_answer(a2)
            )
        except (VerificationError, InvalidHashError):
            self._record_failure()
            return False
        if ok1 and ok2:
            self._reset_attempts()
            return True
        self._record_failure()
        return False

    # ---------------------------------------------------------
    # Nội bộ — bộ đếm sai dùng chung
    # ---------------------------------------------------------

    def _record_failure(self) -> None:
        """Ghi nhận 1 lần sai (mật khẩu hoặc câu trả lời)."""
        self._failed_attempts += 1
        self._last_fail_at = time.time()

    def _reset_attempts(self) -> None:
        """Xác thực đúng → reset bộ đếm sai + hết trạng thái chờ."""
        self._failed_attempts = 0
        self._last_fail_at = 0.0
