"""Màn hình khóa — hiển thị khi mở app và khi bấm Ctrl+L / nút Khoá (FR-11).

Cập nhật theo spec `password-management-spec.md` (2026-09-03):

Các CHẾ ĐỘ (tự chọn mỗi khi ``reset()``):
  - ``setup``          chưa có mật khẩu (lần đầu dùng): đặt mật khẩu mới +
                       2 câu hỏi bảo mật (bắt buộc — FR-5).
  - ``unlock``         đã có đủ mật khẩu + câu hỏi: mở khóa, có link
                       "Quên mật khẩu?".
  - ``upgrade_unlock`` có mật khẩu NHƯNG thiếu câu hỏi bảo mật (người dùng
                       cũ trước bản này): mở khóa bằng mật khẩu cũ, rồi
                       BẮT BUỘC khai 2 câu hỏi trước khi vào app (đã chốt).
  - ``recovery``       quên mật khẩu: trả lời đúng CẢ 2 câu (không bỏ dấu
                       tiếng Việt) → đặt mật khẩu mới (chính sách mới).

Khóa MỀM (FR-2): nhập sai quá 5 lần liên tiếp → phải chờ 10s giữa 2 lần
thử (không còn khóa cứng 60s); bộ đếm dùng chung mật khẩu + câu hỏi.
Hiển thị "còn N lần thử" trước ngưỡng (FR-3); eye toggle (FR-4).
"""
from __future__ import annotations

import logging

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.services.auth import (
    REQUIREMENTS_TEXT,
    SOFT_LOCK_AFTER,
    SOFT_LOCK_WAIT_SECONDS,
    AuthService,
    validate_password_strength,
)

logger = logging.getLogger(__name__)

# Các chế độ giao diện
MODE_SETUP = "setup"              # chưa có mật khẩu — tạo mới + 2 câu hỏi
MODE_UNLOCK = "unlock"            # mở khóa bằng mật khẩu
MODE_UPGRADE_UNLOCK = "upgrade_unlock"  # có mật khẩu, thiếu câu hỏi — bước 1
MODE_UPGRADE_QUESTIONS = "upgrade_questions"  # bước 2: khai 2 câu hỏi
MODE_RECOVERY = "recovery"        # quên mật khẩu — trả lời 2 câu + đặt lại


class LockScreen(QWidget):
    """Màn hình khóa toàn cửa sổ; phát tín hiệu `unlocked` khi mở khóa thành công."""

    unlocked = Signal()

    def __init__(self, auth: AuthService, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._auth = auth
        self._mode = MODE_UNLOCK
        self._build_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_lockout)
        self.reset()

    # ---------------------------------------------------------
    # Giao diện
    # ---------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.addStretch(3)

        title = QLabel("⏱ ĐIỂM DANH NHÂN VIÊN")
        title.setObjectName("pageTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 28px; font-weight: bold; letter-spacing: 2px;")
        outer.addWidget(title)

        self._subtitle = QLabel()
        self._subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._subtitle.setStyleSheet("font-size: 14px;")
        outer.addWidget(self._subtitle)
        outer.addSpacing(24)

        # Khung chứa các trường nhập (thẻ bo góc) — màu do theme QSS quyết định
        card = QFrame()
        card.setFixedWidth(420)
        card.setObjectName("card")
        card.setStyleSheet("border-radius: 10px;")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(24, 24, 24, 24)
        card_layout.setSpacing(10)

        # ---- Vùng mật khẩu (ô 1 + ô 2 + eye toggle) ----
        self._pw1 = QLineEdit()
        self._pw1.setEchoMode(QLineEdit.EchoMode.Password)
        self._pw1.returnPressed.connect(self._on_submit)
        card_layout.addWidget(self._pw1)

        self._pw2 = QLineEdit()
        self._pw2.setEchoMode(QLineEdit.EchoMode.Password)
        self._pw2.setPlaceholderText("Nhập lại mật khẩu")
        self._pw2.returnPressed.connect(self._on_submit)
        card_layout.addWidget(self._pw2)

        self._eye_btn = self._make_eye_button(self._pw1, self._pw2)
        card_layout.addWidget(self._eye_btn, alignment=Qt.AlignmentFlag.AlignRight)

        # ---- Vùng câu hỏi bảo mật ----
        self._sec_title = QLabel()
        self._sec_title.setStyleSheet("font-size: 13px; font-weight: bold; margin-top: 6px;")
        card_layout.addWidget(self._sec_title)

        self._q1_edit = QLineEdit()
        self._q1_edit.setPlaceholderText("Câu hỏi bảo mật 1 (vd: Quê quán của bạn?)")
        card_layout.addWidget(self._q1_edit)
        self._a1_edit = QLineEdit()
        self._a1_edit.setPlaceholderText("Câu trả lời 1")
        self._a1_edit.returnPressed.connect(self._on_submit)
        card_layout.addWidget(self._a1_edit)

        self._q2_edit = QLineEdit()
        self._q2_edit.setPlaceholderText("Câu hỏi bảo mật 2 (khác câu 1)")
        card_layout.addWidget(self._q2_edit)
        self._a2_edit = QLineEdit()
        self._a2_edit.setPlaceholderText("Câu trả lời 2")
        self._a2_edit.returnPressed.connect(self._on_submit)
        card_layout.addWidget(self._a2_edit)

        # ---- Nút chính / link phụ ----
        self._back_btn = QPushButton("‹ Quay lại mở khóa")
        self._back_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._back_btn.setStyleSheet("QPushButton { border: none; color: #888; font-size: 12px; }")
        self._back_btn.clicked.connect(self._on_back)
        card_layout.addWidget(self._back_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        self._submit_btn = QPushButton()
        self._submit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._submit_btn.setObjectName("primaryBtn")
        self._submit_btn.setStyleSheet("border-radius: 6px; padding: 10px; font-size: 14px;")
        self._submit_btn.clicked.connect(self._on_submit)
        card_layout.addWidget(self._submit_btn)

        self._forgot_btn = QPushButton("Quên mật khẩu?")
        self._forgot_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._forgot_btn.setStyleSheet("QPushButton { border: none; color: #888; font-size: 12px; }")
        self._forgot_btn.clicked.connect(self._on_forgot)
        card_layout.addWidget(self._forgot_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        self._error_label = QLabel()
        self._error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._error_label.setStyleSheet("color: #d33; font-weight: bold;")
        self._error_label.setWordWrap(True)
        card_layout.addWidget(self._error_label)

        self._lockout_label = QLabel()
        self._lockout_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lockout_label.setStyleSheet("color: #d33; font-weight: bold;")
        card_layout.addWidget(self._lockout_label)

        center = QHBoxLayout()
        center.addStretch(1)
        center.addWidget(card)
        center.addStretch(1)
        outer.addLayout(center)

        self._hint = QLabel()
        self._hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hint.setStyleSheet("font-size: 12px; color: #888;")
        self._hint.setWordWrap(True)
        outer.addWidget(self._hint)
        outer.addStretch(3)

    @staticmethod
    def _make_eye_button(*edits: QLineEdit) -> QPushButton:
        """Nút 👁 bật/tắt hiện mật khẩu cho các ô được gắn (FR-4)."""

        def toggle() -> None:
            show = edits[0].echoMode() == QLineEdit.EchoMode.Password
            mode = QLineEdit.EchoMode.Normal if show else QLineEdit.EchoMode.Password
            for edit in edits:
                edit.setEchoMode(mode)
            btn.setText("🙈 Ẩn" if show else "👁 Hiện")

        btn = QPushButton("👁 Hiện")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet("QPushButton { border: none; color: #888; font-size: 12px; }")
        btn.clicked.connect(toggle)
        return btn

    # ---------------------------------------------------------
    # Chế độ
    # ---------------------------------------------------------

    def _pick_mode(self) -> str:
        """Chọn chế độ theo trạng thái xác thực hiện tại."""
        auth = self._auth
        if not auth.has_password:
            return MODE_SETUP
        if not auth.has_security_questions:
            # Người dùng cũ (bản trước chưa có câu hỏi) — bắt khai trước
            return MODE_UPGRADE_UNLOCK
        return MODE_UNLOCK

    def _apply_mode(self, mode: str) -> None:
        """Hiển thị đúng nhóm trường + văn bản theo chế độ."""
        self._mode = mode
        setup_pw = mode in (MODE_SETUP, MODE_RECOVERY)      # cần nhập lại pw mới
        questions_form = mode in (
            MODE_SETUP, MODE_UPGRADE_QUESTIONS, MODE_RECOVERY
        )
        auth = self._auth

        if mode == MODE_SETUP:
            self._subtitle.setText("Thiết lập mật khẩu + câu hỏi bảo mật")
            self._pw1.setPlaceholderText("Mật khẩu mới")
            self._q1_edit.setReadOnly(False)
            self._q2_edit.setReadOnly(False)
            self._q1_edit.clear()
            self._q2_edit.clear()
            self._a1_edit.clear()
            self._a2_edit.clear()
            self._sec_title.setText(
                "Câu hỏi bảo mật (dùng khi QUÊN mật khẩu — hãy nhớ kỹ)"
            )
            self._submit_btn.setText("Tạo mật khẩu")
            self._hint.setText(REQUIREMENTS_TEXT)
        elif mode == MODE_UNLOCK:
            self._subtitle.setText("Nhập mật khẩu để mở khóa")
            self._pw1.setPlaceholderText("Mật khẩu")
            self._submit_btn.setText("MỞ KHÓA")
            self._hint.setText(
                f"Nhập sai {SOFT_LOCK_AFTER} lần sẽ phải chờ "
                f"{SOFT_LOCK_WAIT_SECONDS} giây giữa mỗi lần thử"
            )
        elif mode == MODE_UPGRADE_UNLOCK:
            self._subtitle.setText("Nhập mật khẩu để mở khóa")
            self._pw1.setPlaceholderText("Mật khẩu")
            self._submit_btn.setText("MỞ KHÓA")
            self._hint.setText(
                "Sau khi mở khóa, bạn cần thiết lập câu hỏi bảo mật "
                "(tính năng mới cho phép khôi phục khi quên mật khẩu)"
            )
        elif mode == MODE_UPGRADE_QUESTIONS:
            self._subtitle.setText("Hoàn tất bảo mật — tạo 2 câu hỏi")
            self._q1_edit.setReadOnly(False)
            self._q2_edit.setReadOnly(False)
            self._sec_title.setText(
                "Câu hỏi bảo mật (dùng khi QUÊN mật khẩu — hãy nhớ kỹ)"
            )
            self._submit_btn.setText("Hoàn tất")
            self._hint.setText(REQUIREMENTS_TEXT)
        elif mode == MODE_RECOVERY:
            self._subtitle.setText("Quên mật khẩu — trả lời 2 câu hỏi để đặt lại")
            self._pw1.setPlaceholderText("Mật khẩu mới")
            # Hiển thị câu hỏi đã lưu (không cho sửa)
            self._q1_edit.setText(auth.config.security_question_1)
            self._q2_edit.setText(auth.config.security_question_2)
            self._q1_edit.setReadOnly(True)
            self._q2_edit.setReadOnly(True)
            self._a1_edit.clear()
            self._a2_edit.clear()
            self._sec_title.setText("Trả lời cả 2 câu hỏi:")
            self._submit_btn.setText("Đặt lại mật khẩu")
            self._hint.setText(REQUIREMENTS_TEXT)

        # Nhóm trường: nhập lại mật khẩu chỉ cần ở chế độ tạo mới
        self._pw1.setVisible(mode != MODE_UPGRADE_QUESTIONS)
        self._pw2.setVisible(setup_pw)
        self._eye_btn.setVisible(mode != MODE_UPGRADE_QUESTIONS)
        self._sec_title.setVisible(questions_form)
        self._q1_edit.setVisible(questions_form)
        self._a1_edit.setVisible(questions_form)
        self._q2_edit.setVisible(questions_form)
        self._a2_edit.setVisible(questions_form)
        # Link phụ
        self._forgot_btn.setVisible(
            mode == MODE_UNLOCK and auth.has_security_questions
        )
        self._back_btn.setVisible(mode == MODE_RECOVERY)

        self._pw1.setEchoMode(QLineEdit.EchoMode.Password)
        self._pw2.setEchoMode(QLineEdit.EchoMode.Password)
        self._eye_btn.setText("👁 Hiện")

    def reset(self) -> None:
        """Xóa trường nhập + đồng bộ chế độ (gọi mỗi khi quay lại màn hình khóa)."""
        self._apply_mode(self._pick_mode())
        self._pw1.clear()
        self._pw2.clear()
        if self._mode != MODE_RECOVERY:
            self._q1_edit.clear()
            self._q2_edit.clear()
        self._a1_edit.clear()
        self._a2_edit.clear()
        self._error_label.clear()
        self._lockout_label.clear()
        if self._auth.is_locked:
            # Đang phải chờ giữa 2 lần thử → khóa nhập + hiện đếm ngược
            self._start_lockout_timer()
            return
        self._set_inputs_enabled(True)
        if self._mode == MODE_SETUP or self._mode == MODE_UPGRADE_QUESTIONS:
            self._q1_edit.setFocus()
        else:
            self._pw1.setFocus()

    # ---------------------------------------------------------
    # Hành vi
    # ---------------------------------------------------------

    def _on_submit(self) -> None:
        if self._auth.is_locked:
            self._update_lockout()
            return
        if self._mode == MODE_SETUP:
            self._submit_setup()
        elif self._mode in (MODE_UNLOCK, MODE_UPGRADE_UNLOCK):
            self._submit_unlock()
        elif self._mode == MODE_UPGRADE_QUESTIONS:
            self._submit_upgrade_questions()
        elif self._mode == MODE_RECOVERY:
            self._submit_recovery()

    def _submit_setup(self) -> None:
        """Tạo mật khẩu + 2 câu hỏi (lần đầu dùng)."""
        password = self._pw1.text()
        errors = validate_password_strength(password)
        if errors:
            self._show_error("\n".join(errors))
            return
        if password != self._pw2.text():
            self._show_error("Hai lần nhập mật khẩu không khớp")
            return
        try:
            self._auth.set_password(password)
            self._auth.set_security_questions(
                self._q1_edit.text(), self._a1_edit.text(),
                self._q2_edit.text(), self._a2_edit.text(),
            )
        except ValueError as exc:
            self._show_error(str(exc))
            return
        logger.info("Đã đặt mật khẩu mới (lần đầu dùng) + câu hỏi bảo mật")
        self.unlocked.emit()

    def _submit_unlock(self) -> None:
        """Mở khóa bằng mật khẩu (chế độ unlock / upgrade_unlock)."""
        if not self._auth.verify(self._pw1.text()):
            self._handle_wrong_attempt("mật khẩu")
            return
        logger.info("Mở khóa thành công")
        if self._mode == MODE_UPGRADE_UNLOCK:
            # Bước 2: bắt khai 2 câu hỏi trước khi vào app
            self._apply_mode(MODE_UPGRADE_QUESTIONS)
            self._set_inputs_enabled(True)
            self._q1_edit.setFocus()
            return
        self.unlocked.emit()

    def _submit_upgrade_questions(self) -> None:
        """Hoàn tất 2 câu hỏi bảo mật (người dùng cũ nâng cấp)."""
        try:
            self._auth.set_security_questions(
                self._q1_edit.text(), self._a1_edit.text(),
                self._q2_edit.text(), self._a2_edit.text(),
            )
        except ValueError as exc:
            self._show_error(str(exc))
            return
        logger.info("Đã thiết lập câu hỏi bảo mật (nâng cấp)")
        self.unlocked.emit()

    def _submit_recovery(self) -> None:
        """Quên mật khẩu: trả lời đúng 2 câu → đặt mật khẩu mới."""
        # Xác thực câu trả lời TRƯỚC (dùng chung bộ đếm sai — không né cooldown)
        if not self._auth.verify_security_answers(
            self._a1_edit.text(), self._a2_edit.text()
        ):
            self._handle_wrong_attempt("câu trả lời bảo mật")
            return
        password = self._pw1.text()
        errors = validate_password_strength(password)
        if errors:
            self._show_error("\n".join(errors))
            return
        if password != self._pw2.text():
            self._show_error("Hai lần nhập mật khẩu không khớp")
            return
        try:
            self._auth.set_password(password)
        except ValueError as exc:
            self._show_error(str(exc))
            return
        logger.info("Đã đặt lại mật khẩu qua câu hỏi bảo mật")
        self.unlocked.emit()

    def _handle_wrong_attempt(self, what: str) -> None:
        """Xử lý chung sau 1 lần xác thực sai (mật khẩu / câu trả lời)."""
        if self._auth.is_locked:
            self._show_error(
                f"Sai {what} quá nhiều lần — phải chờ giữa mỗi lần thử"
            )
            self._start_lockout_timer()
        else:
            self._show_error(
                f"Sai {what} — còn {self._auth.attempts_remaining} lần thử"
            )

    def _on_forgot(self) -> None:
        """Link 'Quên mật khẩu?' (chỉ ở chế độ unlock khi có đủ câu hỏi)."""
        self._apply_mode(MODE_RECOVERY)
        self._pw1.clear()
        self._pw2.clear()
        self._error_label.clear()
        self._lockout_label.clear()
        self._set_inputs_enabled(True)
        self._a1_edit.setFocus()

    def _on_back(self) -> None:
        """Quay lại màn mở khóa từ màn quên mật khẩu."""
        self.reset()

    # ---------------------------------------------------------
    # Khóa mềm — đếm ngược
    # ---------------------------------------------------------

    def _start_lockout_timer(self) -> None:
        self._set_inputs_enabled(False)
        self._timer.start(1000)
        self._update_lockout()

    def _update_lockout(self) -> None:
        remaining = self._auth.lockout_remaining
        if remaining <= 0:
            self._timer.stop()
            self._lockout_label.clear()
            self._set_inputs_enabled(True)
            self._pw1.setFocus()
            return
        self._lockout_label.setText(f"Thử lại sau {remaining} giây")

    def _set_inputs_enabled(self, enabled: bool) -> None:
        for w in (self._pw1, self._pw2, self._q1_edit, self._a1_edit,
                  self._q2_edit, self._a2_edit, self._submit_btn):
            w.setEnabled(enabled)
        self._forgot_btn.setEnabled(enabled)
        self._back_btn.setEnabled(enabled)

    def _show_error(self, message: str) -> None:
        self._error_label.setText(message)
