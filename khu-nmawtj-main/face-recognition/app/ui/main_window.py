"""Cửa sổ chính — màn hình khóa + điều hướng giữa các màn hình (Bước 3).

Cấu trúc: QMainWindow chứa QStackedWidget với 2 tầng:
  1. LockScreen  — luôn hiển thị khi mở app / bấm Khoá / Ctrl+L
  2. Content     — sidebar điều hướng + các màn hình:
       DashboardView (tổng quan điểm danh) · CameraView (Bước 3)
       PhotoView (Bước 10) · EnrollmentDialog (Bước 8)
       PersonListView (Bước 7) · HistoryView (Bước 11) · SettingsView (Bước 14)

Chế độ ĐIỂM DANH NHÂN VIÊN: webcam nhận diện tự động ghi check-in/out
(app/services/attendance.py) — toàn bộ logic camera/giải pháp AI giữ nguyên.
"""
from __future__ import annotations

import logging

from PySide6.QtCore import QThread, Qt
from PySide6.QtGui import QCloseEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.config import Config
from app.infrastructure.db import Database
from app.services.auth import AuthService
from app.services.sync import SyncService
from app.ui.camera_view import CameraView
from app.ui.dashboard_view import DashboardView
from app.ui.enrollment_dialog import EnrollmentDialog
from app.ui.history_view import HistoryView
from app.ui.lock_screen import LockScreen
from app.ui.payroll_view import PayrollView
from app.ui.person_list_view import PersonListView
from app.ui.photo_view import PhotoView
from app.ui.settings_view import SettingsView

logger = logging.getLogger(__name__)

# Danh sách màn hình: (tiêu đề hiển thị trên sidebar, tên trang)
# Tổng quan (điểm danh) là màn hình MẶC ĐỊNH khi mở khóa.
# "CameraView" và "CheckoutView" CÙNG dùng 1 trang webcam (self._camera_view)
# — chỉ khác CHẾ ĐỘ điểm danh đặt sẵn khi bấm vào menu (xem _switch_page).
PAGES: list[tuple[str, str]] = [
    ("🏠  Tổng quan", "DashboardView"),
    ("🟢  Vào ca", "CameraView"),
    ("🔴  Ra ca", "CheckoutView"),
    ("🖼️  Điểm danh ảnh", "PhotoView"),
    ("➕  Thêm nhân viên", "EnrollmentDialog"),
    ("👥  Nhân viên", "PersonListView"),
    ("📋  Lịch sử quét", "HistoryView"),
    ("💰  Kỳ công", "PayrollView"),
    ("⚙️  Cài đặt", "SettingsView"),
]
# Icon-only version for collapsed sidebar (emoji chi, khong text)
_ICONS: list[str] = ["🏠", "🟢", "🔴", "🖼️", "➕", "👥", "📋", "💰", "⚙️"]
# Ngưỡng chiều rộng (px) — dưới mức này sidebar thu gọn chỉ còn icon
_SIDEBAR_COLLAPSE_WIDTH = 700
_SIDEBAR_EXPANDED_WIDTH = 200
_SIDEBAR_COLLAPSED_WIDTH = 58


class MainWindow(QMainWindow):
    """Cửa sổ chính: màn hình khóa bao quanh vùng điều hướng nội dung."""

    def __init__(self, auth: AuthService, config: Config) -> None:
        super().__init__()
        self._auth = auth
        self._config = config
        self.setWindowTitle("Điểm Danh Nhân Viên · Employee Attendance")
        # Cửa sổ khởi động VỪA màn hình (trừ 40px viền). Trước đây cố định
        # 1100x700: trên màn hình nhỏ / scale 125% chiều cao logical chỉ ~660px
        # → cửa sổ cao hơn màn hình, đáy bị cắt và không kéo nhỏ được.
        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(
            min(1100, screen.width() - 40),
            min(700, screen.height() - 40),
        )

        self._root = QStackedWidget()
        self._lock_screen = LockScreen(auth)
        self._content = self._build_content()

        self._root.addWidget(self._lock_screen)
        self._root.addWidget(self._content)
        self.setCentralWidget(self._root)

        self._lock_screen.unlocked.connect(
            lambda: self._root.setCurrentWidget(self._content)
        )
        # Bật cloud → tự đồng bộ 1 lần khi mở khóa (Bước 15)
        self._lock_screen.unlocked.connect(self._maybe_auto_sync)
        # Nút [Dang ky ngay] tren man hinh webcam -> mo EnrollmentDialog
        # voi anh nguoi la lam mau 1 (phai dung camera truoc, mo lai sau)
        self._camera_view.enroll_requested.connect(self._open_enrollment_from_unknown)
        # Danh sách người thay đổi (xóa/sửa) → reload Matcher nhận diện
        # (tránh ghost match sau khi xóa người — embedding cũ vẫn còn trong bộ nhớ)
        self._person_list_view.person_changed.connect(
            self._camera_view._service.reload
        )
        # Danh sách nhân viên thay đổi → dashboard cập nhật số liệu luôn
        self._person_list_view.person_changed.connect(
            self._dashboard_view.refresh_data
        )

        # Luôn khởi động ở màn hình khóa (lần đầu dùng → đặt mật khẩu mới)
        self._root.setCurrentWidget(self._lock_screen)

        # Phím tắt Ctrl+L khóa ứng dụng
        QShortcut(QKeySequence("Ctrl+L"), self, activated=self._lock_now)
        logger.info("Khởi tạo cửa sổ chính hoàn tất (%d màn hình)", len(PAGES))

    # ---------------------------------------------------------
    # Vùng nội dung (sidebar + các trang)
    # ---------------------------------------------------------

    def _build_content(self) -> QWidget:
        """Xây dựng phần nội dung: sidebar điều hướng + các trang."""
        central = QWidget()
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Cột trái: sidebar + nút Khóa
        left = QVBoxLayout()
        left.setSpacing(0)

        # Logo/khối thương hiệu đầu sidebar
        brand = QLabel("⏱ ĐIỂM DANH")
        brand.setObjectName("brandLabel")
        brand.setAlignment(Qt.AlignmentFlag.AlignCenter)
        brand.setFixedHeight(56)
        brand.setToolTip("Điểm Danh Nhân Viên")
        self._brand_label = brand

        self.nav = QListWidget()
        self.nav.setObjectName("navList")
        self.nav.setFixedWidth(_SIDEBAR_EXPANDED_WIDTH)
        for title, _ in PAGES:
            item = QListWidgetItem(title)
            item.setToolTip(title.strip())  # hiển thị khi sidebar thu gọn
            self.nav.addItem(item)
        self.nav.currentRowChanged.connect(self._switch_page)

        self._lock_btn = QPushButton("🔒 Khoá")
        self._lock_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._lock_btn.setObjectName("sidebarBtn")
        self._lock_btn.setToolTip("Khoá ứng dụng (Ctrl+L)")
        self._lock_btn.clicked.connect(self._lock_now)

        # Nút S/T — chuyển đổi Dark/Light theme (FR-9, wireframe [S/T])
        self._theme_btn = QPushButton("☀️")
        self._theme_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._theme_btn.setObjectName("sidebarBtn")
        self._theme_btn.setToolTip("Chuyển đổi chủ đề Sáng / Tối")
        self._theme_btn.clicked.connect(self._toggle_theme)
        self._update_theme_btn()

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        btn_row.addWidget(self._lock_btn, stretch=1)
        btn_row.addWidget(self._theme_btn)

        left.addWidget(self._brand_label)
        left.addWidget(self.nav)
        left.addLayout(btn_row)

        # Sidebar là 1 khung gradient riêng (QSS `QFrame#sidebar`)
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setLayout(left)

        # Vùng nội dung: Dashboard điểm danh + CameraView + các trang còn lại
        self._db = Database()
        self._dashboard_view = DashboardView(self._db)
        self._camera_view = CameraView(self._config, db=self._db)
        self._enrollment_page = self._make_enrollment_page()
        self._person_list_view = PersonListView(
            self._db,
            self._auth,
            enroll_callback=self._open_enrollment,
            enroll_photo_callback=self._open_enrollment_photo,
        )
        self._photo_view = PhotoView(
            self._config,
            self._db,
            detector=self._camera_view.detector,
        )
        self._history_view = HistoryView(self._db, self._auth)
        self._payroll_view = PayrollView(self._db)
        self._sync_service = SyncService(self._db, self._config)
        self._settings_view = SettingsView(self._config, self._auth, self._sync_service)
        self._settings_view.settings_saved.connect(self._on_settings_saved)
        self.stack = QStackedWidget()
        # Map tên trang → index trong stack. "CameraView" và "CheckoutView"
        # DÙNG CHUNG 1 widget (trang webcam) — chỉ khác chế độ điểm danh đặt
        # sẵn khi bấm menu. Map này PHẢI khớp thứ tự addWidget (bỏ tên trùng).
        self._page_index: dict[str, int] = {}
        for _, name in PAGES:
            if name in self._page_index:
                continue  # trang đã có (CameraView/CheckoutView dùng chung)
            if name == "DashboardView":
                page = self._dashboard_view
            elif name == "CameraView":
                page = self._camera_view
            elif name == "CheckoutView":
                page = self._camera_view  # DÙNG CHUNG — chỉ khác chế độ
            elif name == "EnrollmentDialog":
                page = self._enrollment_page
            elif name == "PersonListView":
                page = self._person_list_view
            elif name == "PhotoView":
                page = self._photo_view
            elif name == "HistoryView":
                page = self._history_view
            elif name == "PayrollView":
                page = self._payroll_view
            elif name == "SettingsView":
                page = self._settings_view
            else:
                page = self._make_placeholder(name)
            self._page_index[name] = self.stack.addWidget(page)

        # Mặc định mở trang Tổng quan (dashboard điểm danh) + tô sáng mục đầu
        self.nav.setCurrentRow(0)

        root_layout.addWidget(sidebar)
        root_layout.addWidget(self.stack, stretch=1)
        return central

    # ---------------------------------------------------------
    # Hàm nội bộ
    # ---------------------------------------------------------

    def _make_placeholder(self, name: str) -> QWidget:
        """Tạo một trang placeholder cho màn hình chưa triển khai."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addStretch(1)

        label = QLabel(f"{name}\n(sẽ triển khai ở các bước tiếp theo)")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("font-size: 18px;")
        layout.addWidget(label)

        layout.addStretch(1)
        return page

    def _make_enrollment_page(self) -> QWidget:
        """Trang 'Thêm nhân viên': nút mở EnrollmentDialog (FR-1)."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addStretch(1)

        title = QLabel("Thêm nhân viên mới")
        title.setObjectName("pageTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        hint = QLabel(
            "Nhập tên nhân viên, nhìn thẳng vào camera — hệ thống tự chụp "
            "ngay khi thấy mặt rõ và lưu vào cơ sở dữ liệu."
        )
        hint.setObjectName("pageSubtitle")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setWordWrap(True)
        layout.addWidget(hint)

        open_btn = QPushButton("＋ Thêm nhân viên (quét khuôn mặt)")
        open_btn.setObjectName("primaryBtn")
        open_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        open_btn.clicked.connect(self._open_enrollment)
        layout.addWidget(open_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        # Đăng ký từ NHIỀU ẢNH có sẵn (không cần camera — FR mới)
        or_label = QLabel("hoặc")
        or_label.setStyleSheet("color: #888; font-size: 12px;")
        or_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(or_label)

        photo_btn = QPushButton("🖼️ Thêm nhân viên từ NHIỀU ẢNH")
        photo_btn.setObjectName("primaryBtn")
        photo_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        photo_btn.setToolTip(
            "Chọn nhiều ảnh có sẵn (jpg/png...) của MỘT hoặc NHIỀU nhân viên:\n"
            "— 1 người: mỗi ảnh nét thêm 1 mẫu nhận diện.\n"
            "— Thư mục nhiều người: app tự gom nhóm theo khuôn mặt + gợi ý "
            "tên từ tên file, lưu TẤT CẢ cùng lúc."
        )
        photo_btn.clicked.connect(self._open_enrollment_photo)
        layout.addWidget(photo_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        photo_hint = QLabel(
            "Phù hợp khi có ảnh nhân viên sẵn (ảnh CV, ảnh chụp điện thoại...).\n"
            "Yêu cầu: ảnh nét · 1 người/ảnh · khuôn mặt chiếm phần lớn khung."
        )
        photo_hint.setStyleSheet("color: #888; font-size: 11px;")
        photo_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        photo_hint.setWordWrap(True)
        layout.addWidget(photo_hint)

        layout.addStretch(1)
        return page

    def _open_enrollment(self) -> None:
        """Mở EnrollmentDialog — tái sử dụng model đã nạp của CameraView nếu có.

        QUAN TRỌNG: dừng hẳn camera của CameraView TRƯỚC khi mở dialog —
        tránh 2 luồng cùng mở 1 webcam (gây lỗi MSMF "OnReadSample failed").
        """
        self._camera_view.stop_camera()
        dialog = EnrollmentDialog(
            self._config,
            self._db,
            detector=self._camera_view.detector,
            parent=self,
        )
        if dialog.exec() == dialog.DialogCode.Accepted:
            # Đăng ký thành công → reload matcher + dashboard + danh sách nhân viên
            self._camera_view._service.reload()
            self._dashboard_view.refresh_data()
            self._person_list_view.refresh()
        # Camera của CameraView sẽ tự mở lại khi quay về trang webcam (showEvent)

    def _open_enrollment_photo(self) -> None:
        """Mở EnrollmentPhotoDialog — THÊM NHÂN VIÊN TỪ NHIỀU ẢNH.

        Tái sử dụng model đã nạp của CameraView (nếu có) — không nạp lại
        model 300MB lần 2. Dừng camera trước như luồng đăng ký thường.
        Lưu thành công → reload matcher + dashboard (qua person_changed).
        """
        from app.ui.enrollment_photo_dialog import EnrollmentPhotoDialog

        self._camera_view.stop_camera()
        dialog = EnrollmentPhotoDialog(
            self._config,
            self._db,
            detector=self._camera_view.detector,
            parent=self,
        )
        dialog.exec()
        if dialog.saved:
            # Person mới → matcher đang nạp trong CameraView thiếu embedding
            # mới → reload ngay (tránh quét không nhận người vừa thêm).
            # Đăng ký HÀNG LOẠT (nhiều người) → reload tối đa 1 lần cho tất cả.
            n = dialog.saved_count
            self._camera_view._service.reload()
            self._dashboard_view.refresh_data()
            # Làm mới danh sách NHÂN VIÊN — trước đây thiếu nên người vừa lưu
            # KHÔNG hiện trong trang 👥 (nhìn như "lưu không thành công").
            self._person_list_view.refresh()
            if n > 1:
                self.statusBar().showMessage(
                    f"Đã thêm {n} nhân viên từ ảnh — nhận diện sẵn sàng", 5000
                )

    def _open_enrollment_from_unknown(
        self, crop, embedding, quality: float
    ) -> None:
        """[Dang ky ngay]: anh crop nguoi la lam mau 1 cua EnrollmentDialog.

        Camera dang chay tren trang webcam -> dung truoc (tranh 2 luong cung
        mo webcam), mo dialog, roi mo lai camera khi dong.
        """
        from app.services.enrollment import CapturedSample

        self._camera_view.stop_camera()
        seed = CapturedSample(
            embedding=embedding,
            quality=quality,
            face_crop=crop,
        )
        dialog = EnrollmentDialog(
            self._config,
            self._db,
            detector=self._camera_view.detector,
            seed_sample=seed,
            parent=self,
        )
        dialog.exec()
        # Van dang o trang webcam (dialog modal khong lam page an) -> mo lai tu dong
        self._camera_view.start_camera()
        if dialog.result() == dialog.DialogCode.Accepted:
            # Dang ky nguoi la thanh cong -> cap nhat toan bo view phu thuoc
            self._camera_view._service.reload()
            self._dashboard_view.refresh_data()
            self._person_list_view.refresh()

    def _switch_page(self, index: int) -> None:
        """Chuyển màn hình khi người dùng chọn trong sidebar.

        Riêng 'Vào ca' / 'Ra ca' dùng CHUNG trang webcam — bấm vào menu
        nào thì đặt CHẾ ĐỘ điểm danh tương ứng TRƯỚC khi chuyển (camera
        quét xong ghi đúng chế độ đang chọn, kể cả khi camera đang chạy).
        """
        if not 0 <= index < len(PAGES):
            return
        name = PAGES[index][1]
        if name == "CameraView":
            self._camera_view.force_checkin_mode()
        elif name == "CheckoutView":
            self._camera_view.force_checkout_mode()
        # Dùng map tên → index đã build khi tạo stack (CameraView/CheckoutView
        # trùng index — cùng 1 widget, chỉ khác chế độ)
        stack_idx = self._page_index.get(name)
        if stack_idx is not None:
            self.stack.setCurrentIndex(stack_idx)
            logger.debug("Chuyển đến màn hình: %s", PAGES[index][0])

    def _toggle_theme(self) -> None:
        """Chuyển đổi Dark/Light theme (FR-9) — áp ngay + lưu vào config.json."""
        from PySide6.QtWidgets import QApplication

        from app.ui.theme import apply_theme

        new_theme = "light" if self._config.theme == "dark" else "dark"
        self._config.theme = new_theme
        apply_theme(QApplication.instance(), new_theme)
        self._config.save()
        self._update_theme_btn()
        logger.info("Đã chuyển theme → %s", new_theme)

    def _update_theme_btn(self) -> None:
        """Nhãn nút S/T: hiển thị theme sẽ chuyển tới (ngược với theme hiện tại)."""
        self._theme_btn.setText(
            "🌙 Tối" if self._config.theme == "light" else "☀️ Sáng"
        )

    def _maybe_auto_sync(self) -> None:
        """Tự đồng bộ khi mở khóa nếu đã bật cloud + đủ thông tin (Bước 15)."""
        if not self._config.sync_enabled or not self._sync_service.is_configured():
            return
        if getattr(self, "_auto_sync_thread", None) is not None \
                and self._auto_sync_thread.isRunning():
            return  # đang đồng bộ rồi — không chạy song song
        from app.ui.settings_view import SyncWorker

        self._auto_sync_thread = QThread(self)
        worker = SyncWorker(self._sync_service)
        worker.moveToThread(self._auto_sync_thread)
        self._auto_sync_thread.started.connect(worker.run)
        worker.finished.connect(self._on_auto_sync_finished)
        self._auto_sync_thread.start()
        logger.info("Tự động đồng bộ cloud sau khi mở khóa")

    def _on_auto_sync_finished(self, ok: bool, summary: str) -> None:
        """Kết quả tự đồng bộ → cập nhật trạng thái ở trang Cài đặt + log."""
        thread, self._auto_sync_thread = self._auto_sync_thread, None
        if thread is not None:
            thread.quit()
            thread.wait(1500)
            thread.deleteLater()
        self._settings_view.set_sync_status(summary)
        logger.info("Tự đồng bộ hoàn tất (%s): %s", "OK" if ok else "LỖI", summary)

    def _on_settings_saved(self) -> None:
        """Cài đặt vừa lưu: dừng camera để lần mở sau dùng cấu hình mới.

        CameraView đọc camera_index/ngưỡng tại thời điểm start_camera —
        nếu camera đang chạy với cấu hình cũ thì phải dừng (trang Cài đặt
        vừa lưu cấu hình mới, quay lại webcam sẽ mở đúng).
        """
        self._camera_view.stop_camera()
        # Vừa bật cloud trong Cài đặt → đồng bộ ngay (không chờ lần mở sau)
        self._maybe_auto_sync()

    def _lock_now(self) -> None:
        """Khóa ứng dụng: dừng camera (riêng tư) + về màn hình khóa."""
        self._camera_view.stop_camera()
        self._root.setCurrentWidget(self._lock_screen)
        self._lock_screen.reset()
        logger.info("Đã khóa ứng dụng")

    # ---------------------------------------------------------
    # Responsive sidebar — tự thu gọn khi cửa sổ nhỏ
    # ---------------------------------------------------------
    def _update_sidebar_mode(self) -> None:
        """Thu gọn sidebar (icon only) khi cửa sổ nhỏ, mở rộng khi to.

        Ngưỡng: _SIDEBAR_COLLAPSE_WIDTH (700px). Khi thu gọn:
          - nav chỉ hiển thị emoji icon (không chữ)
          - nút Khoá / theme thu nhỏ lại
        Khi mở rộng:
          - nav hiển thị đầy đủ icon + chữ
          - nút về kích thước bình thường
        """
        w = self.width()
        collapsed = w < _SIDEBAR_COLLAPSE_WIDTH
        if collapsed == getattr(self, "_sidebar_collapsed", None):
            return  # chưa thay đổi — bỏ qua
        self._sidebar_collapsed = collapsed
        if collapsed:
            self.nav.setFixedWidth(_SIDEBAR_COLLAPSED_WIDTH)
            for i, icon in enumerate(_ICONS):
                if i < self.nav.count():
                    self.nav.item(i).setText(icon)
            self._lock_btn.setText("🔒")
            self._theme_btn.setText("☀️" if self._config.theme == "light" else "🌙")
            self._brand_label.setText("⏱")
        else:
            self.nav.setFixedWidth(_SIDEBAR_EXPANDED_WIDTH)
            for i, (title, _) in enumerate(PAGES):
                if i < self.nav.count():
                    self.nav.item(i).setText(title)
            self._lock_btn.setText("🔒 Khoá")
            self._update_theme_btn()
            self._brand_label.setText("⏱ ĐIỂM DANH")

    def resizeEvent(self, event) -> None:  # noqa: N802 (chuẩn Qt)
        """Phát hiện thay đổi kích thước → cập nhật sidebar."""
        super().resizeEvent(event)
        self._update_sidebar_mode()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 (chuẩn Qt)
        """Đóng app: dừng camera + thread đồng bộ và giải phóng tài nguyên."""
        self._camera_view.stop_camera()
        self._settings_view.stop_probe()
        auto = getattr(self, "_auto_sync_thread", None)
        if auto is not None and auto.isRunning():
            auto.quit()
            auto.wait(1500)
        super().closeEvent(event)
