"""Hộp thoại xác thực mật khẩu cho các thao tác nhạy cảm (FR-11).

Sẽ được tái sử dụng ở: xóa/sửa người (Bước 7), xóa lịch sử (Bước 11),
đổi mật khẩu / bật cloud (Bước 14). Cách dùng:

    if PasswordDialog.require(auth, "Xóa người dùng 'A'", self):
        # thực hiện thao tác nhạy cảm
"""
from __future__ import annotations

import logging

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.services.auth import AuthService

logger = logging.getLogger(__name__)


class PasswordDialog(QDialog):
    """Hộp thoại modal yêu cầu nhập mật khẩu trước khi thực hiện thao tác."""

    def __init__(
        self,
        auth: AuthService,
        operation: str,
        note: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._auth = auth
        self.setWindowTitle("XÁC NHẬN MẬT KHẨU")
        self.setModal(True)
        self.setFixedWidth(420)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        op_label = QLabel(f"Thao tác:  {operation}")
        op_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(op_label)

        if note:
            note_label = QLabel(note)
            note_label.setWordWrap(True)
            layout.addWidget(note_label)

        pw_row = QHBoxLayout()
        pw_row.setSpacing(6)
        self._pw = QLineEdit()
        self._pw.setEchoMode(QLineEdit.EchoMode.Password)
        self._pw.setPlaceholderText("Mật khẩu")
        self._pw.returnPressed.connect(self._on_confirm)
        pw_row.addWidget(self._pw, stretch=1)

        def _toggle_visible() -> None:
            show = self._pw.echoMode() == QLineEdit.EchoMode.Password
            self._pw.setEchoMode(
                QLineEdit.EchoMode.Normal
                if show else QLineEdit.EchoMode.Password
            )
            eye_btn.setText("🙈 Ẩn" if show else "👁 Hiện")

        eye_btn = QPushButton("👁 Hiện")
        eye_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        eye_btn.setStyleSheet("QPushButton { border: none; color: #888; font-size: 12px; }")
        eye_btn.clicked.connect(_toggle_visible)
        pw_row.addWidget(eye_btn)
        layout.addLayout(pw_row)

        self._error = QLabel()
        self._error.setStyleSheet("color: #d33; font-weight: bold;")
        self._error.setVisible(False)
        layout.addWidget(self._error)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel = QPushButton("Hủy")
        cancel.clicked.connect(self.reject)
        confirm = QPushButton("Xác nhận")
        confirm.setDefault(True)
        confirm.clicked.connect(self._on_confirm)
        buttons.addWidget(cancel)
        buttons.addWidget(confirm)
        layout.addLayout(buttons)

        self._pw.setFocus()

    # ---------------------------------------------------------
    # Hành vi
    # ---------------------------------------------------------

    def _on_confirm(self) -> None:
        if self._auth.is_locked:
            self._show_error(
                f"Phải chờ — thử lại sau {self._auth.lockout_remaining}s"
            )
            return
        if self._auth.verify(self._pw.text()):
            logger.info("Xác thực mật khẩu thành công")
            self.accept()
        elif self._auth.is_locked:
            self._show_error(
                f"Sai quá nhiều lần — thử lại sau {self._auth.lockout_remaining}s"
            )
        else:
            self._show_error(
                f"Sai mật khẩu — còn {self._auth.attempts_remaining} lần thử"
            )

    def _show_error(self, message: str) -> None:
        self._error.setText(message)
        self._error.setVisible(True)

    @staticmethod
    def require(
        auth: AuthService,
        operation: str,
        parent: QWidget | None = None,
        note: str | None = None,
    ) -> bool:
        """Hiện dialog xác thực; trả về True nếu mật khẩu đúng."""
        dialog = PasswordDialog(auth, operation, note=note, parent=parent)
        return dialog.exec() == QDialog.DialogCode.Accepted
