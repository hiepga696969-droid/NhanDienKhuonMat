"""PersonListView — danh sách người đã đăng ký (Bước 8, wireframe 5.5.5).

Mỗi dòng hiển thị: ảnh đại diện (thumbnail) + tên + số mẫu + ngày đăng ký
+ lần nhận diện gần nhất, kèm nút [Sửa] và [Xóa]:
  - [Sửa] → mở ``PasswordDialog`` rồi đổi tên (QInputDialog).
  - [Xóa] → mở ``PasswordDialog`` (thông báo không thể hoàn tác) rồi xóa.
  - [＋ Đăng ký mới] → gọi callback do MainWindow cung cấp (mở
    EnrollmentDialog), sau khi đóng thì làm mới danh sách.

Tìm kiếm lọc theo tên (không phân biệt hoa thường). Danh sách tự làm mới
mỗi khi trang được hiển thị (showEvent) — dữ liệu luôn mới.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.infrastructure.db import DATA_DIR, Database
from app.services.auth import AuthService
from app.services.person import PersonService, PersonStats
from app.ui.password_dialog import PasswordDialog

logger = logging.getLogger(__name__)

THUMB_SIZE = 64  # kích thước ảnh đại diện (px)


class PersonListView(QWidget):
    """Trang quản lý danh sách người dùng (FR-6)."""

    # Phát ra khi danh sách người thay đổi (thêm/xóa/sửa) — MainWindow
    # kết nối để reload Matcher nhận diện (tránh ghost match sau xóa).
    person_changed = Signal()

    def __init__(
        self,
        db: Database,
        auth: AuthService,
        enroll_callback: Callable[[], None] | None = None,
        enroll_photo_callback: Callable[[], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._auth = auth
        self._service = PersonService(db)
        self._enroll_callback = enroll_callback
        # Callback mở dialog đăng ký từ NHIỀU ẢNH (MainWindow cung cấp —
        # cần dừng camera + tái sử dụng model đã nạp của CameraView).
        self._enroll_photo_callback = enroll_photo_callback
        self._all_stats: list[PersonStats] = []
        self._build_ui()

    # ---------------------------------------------------------
    # Giao diện (wireframe 5.5.5)
    # ---------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        # Tiêu đề + tổng số
        header = QHBoxLayout()
        title = QLabel("DANH SÁCH NGƯỜI DÙNG")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        header.addWidget(title)
        header.addStretch(1)
        self._count_label = QLabel()
        self._count_label.setStyleSheet("font-size: 13px;")
        header.addWidget(self._count_label)
        layout.addLayout(header)

        # Thanh công cụ: tìm kiếm + nút đăng ký mới
        toolbar = QHBoxLayout()
        self._search = QLineEdit()
        self._search.setPlaceholderText("🔍 Tìm kiếm theo tên...")
        self._search.setClearButtonEnabled(True)
        # textChanged truyền 1 tham số str — dùng lambda để bỏ qua (rõ ràng hơn)
        self._search.textChanged.connect(lambda _: self._render())
        toolbar.addWidget(self._search, stretch=1)

        enroll_btn = QPushButton("＋ Đăng ký mới")
        enroll_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        enroll_btn.setObjectName("primaryBtn")
        enroll_btn.setStyleSheet("border-radius: 6px; padding: 8px 14px; font-size: 13px;")
        enroll_btn.clicked.connect(self._on_enroll)
        toolbar.addWidget(enroll_btn)

        # Nút đăng ký từ NHIỀU ẢNH (không cần camera — ảnh có sẵn)
        photo_btn = QPushButton("🖼️ Thêm từ ảnh")
        photo_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        photo_btn.setToolTip(
            "Thêm nhân viên từ ảnh có sẵn (jpg/png...) — không cần camera.\n"
            "— 1 người: mỗi ảnh nét thêm 1 mẫu nhận diện.\n"
            "— Thư mục nhiều người: app tự gom nhóm theo khuôn mặt + gợi ý tên\n"
            "  từ tên file, đăng ký TẤT CẢ cùng lúc."
        )
        photo_btn.setStyleSheet("border-radius: 6px; padding: 8px 14px; font-size: 13px;")
        photo_btn.clicked.connect(self._on_enroll_from_photos)
        toolbar.addWidget(photo_btn)
        layout.addLayout(toolbar)

        # Danh sách
        self._list = QListWidget()
        self._list.setSpacing(4)
        # Màu nền/viền do theme QSS quyết định — chỉ giữ bo góc
        self._list.setStyleSheet("border-radius: 8px;")
        layout.addWidget(self._list, stretch=1)

        # Trạng thái rỗng (hiện khi không có ai / không khớp tìm kiếm)
        self._empty_label = QLabel("Chưa có người nào — bấm [＋ Đăng ký mới] để bắt đầu.")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet("font-size: 14px; padding: 24px;")
        self._empty_label.hide()
        layout.addWidget(self._empty_label)

        # Chân trang: tổng cộng + cảnh báo bảo mật
        footer = QVBoxLayout()
        self._shown_label = QLabel()
        self._shown_label.setStyleSheet("font-size: 13px;")
        footer.addWidget(self._shown_label)
        warning = QLabel("⚠ Xóa / sửa người dùng yêu cầu mật khẩu")
        warning.setStyleSheet("font-size: 12px; color: #b06a00;")  # hổ phách — đọc được cả 2 theme
        footer.addWidget(warning)
        layout.addLayout(footer)

    # ---------------------------------------------------------
    # Dữ liệu & hiển thị
    # ---------------------------------------------------------
    def _on_enroll_from_photos(self) -> None:
        """Mở EnrollmentPhotoDialog — thêm nhân viên từ NHIỀU ảnh có sẵn.

        Callback do MainWindow cung cấp (dừng camera + tái sử dụng model);
        không có callback → mở dialog trực tiếp (model tự nạp lần đầu).
        Xong → làm mới danh sách + báo "người thay đổi" để reload matcher.
        """
        if self._enroll_photo_callback is not None:
            self._enroll_photo_callback()
        else:
            # Không có callback (trường hợp test) → mở dialog trực tiếp;
            # cần Config để dialog tự nạp model khi CameraView chưa chạy.
            from app.config import Config
            from app.ui.enrollment_photo_dialog import EnrollmentPhotoDialog

            dlg = EnrollmentPhotoDialog(
                Config(), self._service._db, detector=None, parent=self
            )
            dlg.exec()
        self._refresh()
        self.person_changed.emit()

    def _refresh(self) -> None:
        """Đọc lại dữ liệu từ DB rồi vẽ lại danh sách."""
        self._all_stats = self._service.list_with_stats()
        self._render()

    def refresh(self) -> None:
        """API công khai cho MainWindow: làm mới danh sách sau khi thêm/sửa/xóa
        người từ DIALOG (đăng ký từ ảnh, đăng ký camera...) — các luồng đó
        KHÔNG phát person_changed vì dialog không nằm trong view này."""
        self._refresh()

    def _render(self) -> None:
        """Vẽ lại danh sách theo dữ liệu đã có + bộ lọc tìm kiếm hiện tại."""
        query = self._search.text().strip().lower()
        self._list.clear()

        shown = 0
        for stats in self._all_stats:
            name = stats.person.name
            if query and query not in name.lower():
                continue
            self._add_row(stats)
            shown += 1

        total = len(self._all_stats)
        self._count_label.setText(f"Tổng cộng: {total} người")
        self._shown_label.setText(f"Tổng cộng: {total} người · đang hiển thị: {shown}")
        if shown == 0:
            if query:
                self._empty_label.setText("Không có người nào khớp từ khóa tìm kiếm.")
            else:
                self._empty_label.setText(
                    "Chưa có người nào — bấm [＋ Đăng ký mới] để bắt đầu."
                )
        self._empty_label.setVisible(shown == 0)

    def _add_row(self, stats: PersonStats) -> None:
        """Thêm một người vào danh sách dưới dạng dòng tùy biến."""
        person = stats.person
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(8, 8, 8, 8)
        row_layout.setSpacing(12)

        # Ảnh đại diện
        thumb = QLabel()
        thumb.setFixedSize(THUMB_SIZE, THUMB_SIZE)
        thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thumb.setPixmap(self._load_thumbnail(person.thumbnail_path))
        thumb.setObjectName("thumbLabel")
        thumb.setStyleSheet("border-radius: 8px;")
        row_layout.addWidget(thumb)

        # Thông tin
        info = QVBoxLayout()
        info.setSpacing(2)

        name_label = QLabel(person.name)
        name_label.setStyleSheet("font-size: 15px; font-weight: bold;")
        info.addWidget(name_label)

        # Phòng ban · chức vụ (hiện dòng chữ xám khi chưa khai — v3)
        role_parts = [p for p in (person.department, person.position) if p]
        if role_parts:
            role_label = QLabel(" · ".join(role_parts))
            role_label.setStyleSheet("color: #6c9bd2; font-size: 12px; font-weight: bold;")
            info.addWidget(role_label)

        created = self._format_created(person.created_at)
        meta = QLabel(f"{stats.sample_count} mẫu · {created}")
        meta.setStyleSheet("font-size: 12px;")
        info.addWidget(meta)

        last_seen = self._format_last_seen(stats.last_detected_at)
        seen_label = QLabel(last_seen)
        # Xanh lá khi có lần nhận diện; xám nhạt khi chưa (đọc được cả 2 theme)
        seen_label.setStyleSheet(
            "color: #4caf50; font-size: 12px;"
            if stats.last_detected_at
            else "color: #8a8a92; font-size: 12px;"
        )
        info.addWidget(seen_label)

        row_layout.addLayout(info, stretch=1)

        # Nút thao tác
        edit_btn = QPushButton("Sửa")
        edit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        edit_btn.setObjectName("secondaryBtn")
        edit_btn.setStyleSheet("border-radius: 5px; padding: 5px 14px; font-size: 12px;")
        edit_btn.clicked.connect(
            lambda _=False, pid=person.id, nm=person.name,
                   dp=person.department, ps=person.position:
            self._on_edit(pid, nm, dp, ps)
        )
        row_layout.addWidget(edit_btn)

        delete_btn = QPushButton("Xóa")
        delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        delete_btn.setObjectName("dangerBtn")
        delete_btn.setStyleSheet("border-radius: 5px; padding: 5px 14px; font-size: 12px;")
        delete_btn.clicked.connect(
            lambda _=False, pid=person.id, nm=person.name: self._on_delete(pid, nm)
        )
        row_layout.addWidget(delete_btn)

        # Lưu id người vào item để lấy lại khi cần
        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, person.id)
        item.setSizeHint(row.sizeHint())
        self._list.addItem(item)
        self._list.setItemWidget(item, row)

    # ---------------------------------------------------------
    # Hành động
    # ---------------------------------------------------------
    def _on_enroll(self) -> None:
        """Mở EnrollmentDialog (qua callback của MainWindow) rồi làm mới."""
        if self._enroll_callback is not None:
            self._enroll_callback()  # đồng bộ (dialog.exec() chặn) — xong là làm mới
        self._refresh()

    def _on_edit(self, person_id: str, name: str, department: str = "", position: str = "") -> None:
        """Sửa nhân viên: mật khẩu trước, rồi nhập tên + phòng ban + chức vụ."""
        if not PasswordDialog.require(
            self._auth, f"Sửa thông tin nhân viên '{name}'", self
        ):
            return
        new_name, ok = QInputDialog.getText(
            self, "Đổi tên", "Tên mới:", text=name
        )
        if not ok:
            return
        new_dept, ok = QInputDialog.getText(
            self, "Phòng ban", "Phòng ban mới (để trống nếu chưa có):",
            text=department,
        )
        if not ok:
            return
        new_pos, ok = QInputDialog.getText(
            self, "Chức vụ", "Chức vụ mới (để trống nếu chưa có):",
            text=position,
        )
        if not ok:
            return
        if self._service.update_info(person_id, new_name, new_dept, new_pos):
            logger.info("Đã cập nhật nhân viên '%s' → '%s'", name, new_name)
            self._refresh()
            self.person_changed.emit()
        else:
            QMessageBox.warning(self, "Không đổi được", "Tên mới không hợp lệ (trống).")

    def _on_delete(self, person_id: str, name: str) -> None:
        """Xóa người: xác nhận trước → mật khẩu → xóa (không thể hoàn tác)."""
        # Bước 1: Hộp thoại xác nhận (trước khi hỏi mật khẩu)
        answer = QMessageBox.question(
            self,
            "Xác nhận xóa",
            f"Bạn có chắc muốn xóa người dùng '{name}'?\n\n"
            "• Ảnh đại diện sẽ bị xóa\n"
            "• Tất cả mẫu khuôn mặt sẽ bị xóa\n"
            "• Lịch sử nhận diện sẽ giữ nguyên (không gắn tên nữa)\n\n"
            "Hành động này không thể hoàn tác.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        # Bước 2: Yêu cầu mật khẩu
        if not PasswordDialog.require(
            self._auth,
            f"Xóa người dùng '{name}'",
            self,
            note="Nhập mật khẩu để xác nhận xóa.",
        ):
            return
        # Bước 3: Thực hiện xóa
        if self._service.delete(person_id):
            logger.info("Đã xóa người dùng '%s'", name)
            self._refresh()
            self.person_changed.emit()  # reload Matcher nhận diện
            QMessageBox.information(
                self, "Đã xóa", f"Đã xóa '{name}' thành công."
            )
        else:
            QMessageBox.warning(self, "Không xóa được", "Người dùng không tồn tại.")

    # ---------------------------------------------------------
    # Tiện ích hiển thị
    # ---------------------------------------------------------
    def _load_thumbnail(self, thumbnail_path: str) -> QPixmap:
        """Đọc ảnh đại diện từ đĩa; ảnh giữ chỗ màu xám nếu thiếu/hỏng.

        Chỉ đọc khi đường dẫn nằm TRONG thư mục dữ liệu (chống đường dẫn
        lạ "../..." thoát ra ngoài — dù DB tự ghi nên rất hiếm gặp).
        """
        if not thumbnail_path:
            return self._placeholder_pixmap()
        path = (DATA_DIR / thumbnail_path).resolve()
        if path.is_relative_to(DATA_DIR.resolve()) and path.exists():
            pixmap = QPixmap(str(path))
            if not pixmap.isNull():
                return pixmap.scaled(
                    THUMB_SIZE,
                    THUMB_SIZE,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
        return self._placeholder_pixmap()

    @staticmethod
    def _placeholder_pixmap() -> QPixmap:
        """Ảnh giữ chỗ màu xám (khi chưa có thumbnail hoặc file hỏng)."""
        placeholder = QPixmap(THUMB_SIZE, THUMB_SIZE)
        placeholder.fill(QColor("#ccd2da"))
        return placeholder

    @staticmethod
    def _format_created(iso: str) -> str:
        """created_at (ISO) → 'DD/MM/YYYY'."""
        try:
            dt = datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone()
        except ValueError:
            return iso[:10]
        return dt.strftime("%d/%m/%Y")

    @staticmethod
    def _format_last_seen(iso: str | None) -> str:
        """Lần nhận diện gần nhất → 'hôm nay 14:32' / 'hôm qua' / 'DD/MM/YYYY'."""
        if not iso:
            return "— Chưa từng được nhận diện"
        try:
            dt = datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone()
        except ValueError:
            return iso[:16].replace("T", " ")
        today = datetime.now().astimezone().date()
        if dt.date() == today:
            return f"✓ Nhận diện gần nhất: hôm nay {dt:%H:%M}"
        if dt.date() == today - timedelta(days=1):
            return f"✓ Nhận diện gần nhất: hôm qua {dt:%H:%M}"
        return f"✓ Nhận diện gần nhất: {dt:%d/%m/%Y}"

    # ---------------------------------------------------------
    # Vòng đời
    # ---------------------------------------------------------
    def showEvent(self, event) -> None:  # noqa: N802 (chuẩn Qt)
        """Mỗi khi trang hiển thị → làm mới dữ liệu (người mới đăng ký xuất hiện)."""
        super().showEvent(event)
        self._refresh()
