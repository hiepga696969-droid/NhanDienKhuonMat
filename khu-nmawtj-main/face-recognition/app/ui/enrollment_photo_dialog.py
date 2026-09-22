# -*- coding: utf-8 -*-
"""EnrollmentPhotoDialog — THÊM NHÂN VIÊN TỪ NHIỀU ẢNH (không cần camera).

HAI CHẾ ĐỘ:
  1. MỘT nhân viên (mặc định) — ảnh cùng 1 người → 1 hồ sơ nhiều mẫu.
  2. NHIỀU nhân viên — đẩy cả THƯ MỤC chứa ảnh của nhiều người khác nhau:
     app tự GOM NHÓM theo khuôn mặt (AI clustering bằng embedding ArcFace),
     gợi ý tên từ TÊN FILE ("NguyenVanA_1.jpg" → "NguyenVanA"), người dùng
     rà/sửa tên từng nhóm rồi LƯU TẤT CẢ cùng lúc.

Quy trình chung:
  1. [Chọn ảnh...] / [Chọn thư mục...] → 2. [⚡ Xử lý ảnh] chạy QThread
  (SCRFD + ArcFace, UI không đơ, thanh tiến độ) → 3. Danh sách kết quả
  (ảnh ĐẠT: thumbnail + checkbox ✅; ảnh TRƯỢT: lý do) → 4. Xác nhận & Lưu.

LUỒNG MODEL: dialog nhận detector CHUNG (đã nạp bởi CameraView nếu người
dùng đã vào trang webcam) — chưa có thì tự nạp lần đầu (mất vài giây).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QSize, Qt, QThread, Signal, Slot
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.config import Config
from app.core.detector import FaceDetector, MODELS_ROOT, ensure_model_available
from app.core.embedder import FaceEmbedder
from app.infrastructure.db import Database
from app.services.enrollment import EnrollmentService
from app.services.enrollment_photos import (
    EnrollmentPhotoService,
    PersonGroup,
    PhotoProcessResult,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Worker — xử lý ảnh trong QThread (UI không đơ)
# ═══════════════════════════════════════════════════════════════
class PhotoEnrollWorker(QObject):
    """Chạy EnrollmentPhotoService trong luồng phụ (xử lý + gom nhóm)."""

    progress = Signal(int, int, str)      # (xong, tổng, tên file hiện tại)
    one_done = Signal(object)             # PhotoProcessResult từng ảnh
    finished_all = Signal(int, int)       # (số đạt, tổng)
    groups_ready = Signal(list)           # list[PersonGroup] — chế độ NHIỀU người
    error = Signal(str)

    def __init__(
        self,
        paths: list[str],
        service: EnrollmentPhotoService,
        mode: str = "single",  # "single" | "multi"
    ) -> None:
        super().__init__()
        self._paths = paths
        self._service = service
        self._mode = mode
        self._cancelled = False

    def cancel(self) -> None:
        """Hủy xử lý nửa chừng (ảnh đang xử lý sẽ hoàn tất rồi dừng)."""
        self._cancelled = True

    @Slot()
    def run(self) -> None:
        try:
            files = self._service.list_image_files(self._paths)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Lỗi liệt kê ảnh")
            self.error.emit(f"Lỗi liệt kê ảnh: {exc}")
            return
        total = len(files)
        if total == 0:
            self.error.emit("Không có ảnh nào được chọn (hỗ trợ jpg/png/bmp/webp)")
            return
        ok_count = 0
        results: list[PhotoProcessResult] = []
        for i, f in enumerate(files, start=1):
            if self._cancelled:
                break
            self.progress.emit(i, total, f.name)
            try:
                result = self._service.process_photo(str(f))
            except Exception as exc:  # noqa: BLE001 — 1 ảnh lỗi không giết lô
                logger.exception("Lỗi xử lý ảnh %s", f.name)
                from app.services.enrollment_photos import PhotoProcessResult
                result = PhotoProcessResult(str(f), False, f"Lỗi xử lý: {exc}")
            results.append(result)
            if result.ok:
                ok_count += 1
            self.one_done.emit(result)

        # Chế độ NHIỀU người → gom nhóm NGƯỜI ngay trong worker (toán nhẹ,
        # không đụng model) rồi phát về UI
        if self._mode == "multi" and not self._cancelled:
            try:
                groups = self._service.group_into_persons(results)
                self.groups_ready.emit(groups)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Lỗi gom nhóm khuôn mặt")
                self.error.emit(f"Lỗi gom nhóm: {exc}")
                return
        self.finished_all.emit(ok_count, total)


def _icon_from_bgr(crop, size: int = 96) -> QIcon | None:
    """BGR crop → QIcon cho gallery nhóm (None nếu ảnh hỏng — không giết dialog)."""
    try:
        import cv2

        if crop is None or crop.size == 0:
            return None
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        return QIcon(QPixmap.fromImage(img).scaled(
            size, size, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))
    except Exception:  # noqa: BLE001
        return None


class GroupNamingDialog(QDialog):
    """Đặt TÊN + PHÒNG BAN + CHỨC VỤ cho một nhóm — CÓ HIỂN THỊ CÁC ẢNH MẶT
    của nhóm để người dùng nhìn mặt ai rồi mới nhập (không đoán mò).

    - [✔ Chấp nhận]  → dùng thông tin đã nhập (Enter cũng chạy).
    - [⏭ Bỏ qua nhóm] → bỏ nhóm này, các nhóm khác vẫn được lưu.
    """

    def __init__(
        self,
        suggested_name: str,
        crops: list,
        paths: list[str],
        suggested_department: str = "",
        suggested_position: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Đặt tên nhân viên — xem ảnh nhóm")
        self.setMinimumSize(560, 500)
        layout = QVBoxLayout(self)

        head = QLabel(
            f"Nhóm gồm {len(crops)} ảnh đạt — xem mặt bên dưới, nhập thông tin rồi bấm [✔ Chấp nhận]."
        )
        head.setWordWrap(True)
        layout.addWidget(head)

        # Gallery các MẶT trong nhóm (icon + tên file + tooltip đường dẫn)
        self._gallery = QListWidget()
        self._gallery.setViewMode(QListWidget.ViewMode.IconMode)
        self._gallery.setIconSize(QSize(96, 96))
        self._gallery.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._gallery.setMovement(QListWidget.Movement.Static)
        self._gallery.setSpacing(10)
        self._gallery.setMinimumHeight(220)
        for crop, path in zip(crops, paths):
            it = QListWidgetItem(Path(path).name)
            icon = _icon_from_bgr(crop)
            if icon is not None:
                it.setIcon(icon)
            it.setToolTip(path)
            self._gallery.addItem(it)
        layout.addWidget(self._gallery, stretch=1)

        # Thông tin nhập: họ tên (bắt buộc) + phòng ban + chức vụ
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("👤 Họ tên:"))
        self._name_edit = QLineEdit(suggested_name)
        self._name_edit.selectAll()
        row1.addWidget(self._name_edit, stretch=2)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Phòng ban:"))
        self._dept_edit = QLineEdit(suggested_department)
        self._dept_edit.setPlaceholderText("tùy chọn")
        row2.addWidget(self._dept_edit, stretch=1)
        row2.addWidget(QLabel("Chức vụ:"))
        self._pos_edit = QLineEdit(suggested_position)
        self._pos_edit.setPlaceholderText("tùy chọn")
        row2.addWidget(self._pos_edit, stretch=1)
        layout.addLayout(row2)

        btns = QHBoxLayout()
        self._btn_ok = QPushButton("✔ Chấp nhận")
        self._btn_ok.setObjectName("primaryBtn")
        self._btn_ok.setDefault(True)
        self._btn_ok.clicked.connect(self.accept)
        btns.addWidget(self._btn_ok, stretch=1)
        btn_skip = QPushButton("⏭ Bỏ qua nhóm")
        btn_skip.clicked.connect(self.reject)
        btns.addWidget(btn_skip)
        layout.addLayout(btns)

        # Chưa nhập tên → không cho chấp nhận
        self._btn_ok.setEnabled(bool(suggested_name.strip()))
        self._name_edit.textChanged.connect(
            lambda t: self._btn_ok.setEnabled(bool(t.strip()))
        )

    def values(self) -> tuple[str, str, str]:
        """(họ tên, phòng ban, chức vụ) người dùng đã nhập."""
        return (
            self._name_edit.text().strip(),
            self._dept_edit.text().strip(),
            self._pos_edit.text().strip(),
        )


# ═══════════════════════════════════════════════════════════════
# Dialog chính
# ═══════════════════════════════════════════════════════════════
class EnrollmentPhotoDialog(QDialog):
    """Dialog đăng ký nhân viên từ NHIỀU ảnh có XÁC NHẬN từng mẫu.

    Chế độ "single": 1 hồ sơ từ ảnh cùng 1 người (hành vi cũ).
    Chế độ "multi":  THƯ MỤC nhiều người → tự gom nhóm + lưu HÀNG LOẠT.
    """

    def __init__(
        self,
        config: Config,
        db: Database,
        detector: FaceDetector | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._db = db
        self._detector = detector
        self._service: EnrollmentPhotoService | None = None
        self._results: list[PhotoProcessResult] = []
        self._groups: list[PersonGroup] = []   # chế độ multi — nhóm người
        self._mode = "single"                  # "single" | "multi"
        self._worker: PhotoEnrollWorker | None = None
        self._thread: QThread | None = None
        self._saved = False
        self._saved_count = 0                  # số hồ sơ đã lưu (thành công)
        self._last_paths: list[str] = []       # ảnh đã chọn gần nhất (nút Xử lý chạy lại)
        self._build_ui()
        # Chưa có model (người dùng mở thẳng dialog) → nạp nền ngay
        if self._detector is None:
            self._status.setText("⏳ Đang nạp model nhận diện lần đầu (vài giây)...")
            QApplication.processEvents()
            self._load_detector()

    # ---------------------------------------------------------
    # Giao diện
    # ---------------------------------------------------------
    def _build_ui(self) -> None:
        self.setWindowTitle("Thêm nhân viên từ ảnh")
        self.resize(760, 680)
        layout = QVBoxLayout(self)

        title = QLabel("Đăng ký từ NHIỀU ảnh nhân viên")
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        self._hint = QLabel()
        self._hint.setObjectName("pageSubtitle")
        self._hint.setWordWrap(True)
        layout.addWidget(self._hint)

        # ── Chọn CHẾ ĐỘ ──
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Chế độ:"))
        self._rb_single = QRadioButton("👤 Một nhân viên")
        self._rb_single.setChecked(True)
        self._rb_single.toggled.connect(self._on_mode_changed)
        mode_row.addWidget(self._rb_single)
        self._rb_multi = QRadioButton("👥 Nhiều nhân viên (thư mục nhiều người)")
        self._rb_multi.toggled.connect(self._on_mode_changed)
        mode_row.addWidget(self._rb_multi)
        mode_row.addStretch(1)
        layout.addLayout(mode_row)

        # Thanh công cụ chọn ảnh
        tools = QHBoxLayout()
        btn_pick = QPushButton("📁 Chọn ảnh...")
        btn_pick.clicked.connect(self._on_pick_files)
        tools.addWidget(btn_pick)
        btn_dir = QPushButton("📂 Chọn thư mục...")
        btn_dir.clicked.connect(self._on_pick_folder)
        tools.addWidget(btn_dir)
        self._btn_process = QPushButton("⚡ Xử lý ảnh")
        self._btn_process.setObjectName("primaryBtn")
        self._btn_process.setEnabled(False)
        self._btn_process.clicked.connect(self._on_process)
        tools.addWidget(self._btn_process)
        tools.addStretch(1)
        layout.addLayout(tools)

        # Trạng thái + tiến độ
        self._status = QLabel("Chưa chọn ảnh.")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        self._progress = QProgressBar()
        self._progress.hide()
        layout.addWidget(self._progress)

        # Danh sách kết quả (thumbnail + checkbox từng ảnh)
        self._list = QListWidget()
        self._list.setMinimumHeight(300)
        self._list.setSpacing(2)
        self._list.itemDoubleClicked.connect(self._on_rename_group)
        layout.addWidget(self._list, stretch=1)

        # Thông tin nhập (chế độ MỘT người — ẩn khi chọn NHIỀU người).
        # Bọc layout trong QWidget để show/hide CẢ HÀNG (QHBoxLayout không
        # có .show()/.hide() — chỉ widget mới ẩn/hiện được).
        self._info_widget = QWidget()
        self._info_row = QHBoxLayout(self._info_widget)
        self._info_row.setContentsMargins(0, 0, 0, 0)
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("👤 Họ tên nhân viên (bắt buộc)")
        self._info_row.addWidget(self._name_edit, stretch=2)
        self._dept_edit = QLineEdit()
        self._dept_edit.setPlaceholderText("Phòng ban (tùy chọn)")
        self._info_row.addWidget(self._dept_edit, stretch=1)
        self._pos_edit = QLineEdit()
        self._pos_edit.setPlaceholderText("Chức vụ (tùy chọn)")
        self._info_row.addWidget(self._pos_edit, stretch=1)
        layout.addWidget(self._info_widget)

        # Nút lưu/hủy
        btns = QHBoxLayout()
        self._btn_save = QPushButton("💾 Lưu nhân viên")
        self._btn_save.setObjectName("primaryBtn")
        self._btn_save.setEnabled(False)
        self._btn_save.clicked.connect(self._on_save)
        btns.addWidget(self._btn_save, stretch=1)
        btn_cancel = QPushButton("Đóng")
        btn_cancel.clicked.connect(self.reject)
        btns.addWidget(btn_cancel)
        layout.addLayout(btns)

        self._apply_mode_texts()

    def _apply_mode_texts(self) -> None:
        """Cập nhật gợi ý + nhãn nút theo chế độ đang chọn."""
        if self._rb_multi.isChecked():
            self._hint.setText(
                "Đẩy cả thư mục chứa ảnh của NHIỀU nhân viên khác nhau.\n"
                "App tự GOM NHÓM theo khuôn mặt (AI) + gợi ý tên từ TÊN FILE\n"
                '(vd "NguyenVanA_1.jpg" → "NguyenVanA"). Rà lại tên từng nhóm rồi lưu TẤT CẢ.'
            )
            self._btn_save.setText(f"💾 Lưu TẤT CẢ nhóm ({len(self._groups)} người)" if self._groups else "💾 Lưu TẤT CẢ nhóm")
            self._info_widget.hide()
        else:
            self._hint.setText(
                "Chọn nhiều ảnh CÙNG một nhân viên (nét, 1 người/ảnh, mặt chiếm phần lớn khung).\n"
                "Mỗi ảnh đạt → 1 mẫu nhận diện. Nhiều mẫu = quét điểm danh chính xác hơn."
            )
            self._btn_save.setText("💾 Lưu nhân viên")
            self._info_widget.show()

    def _on_mode_changed(self) -> None:
        """Đổi chế độ → dọn kết quả cũ (nhóm/nhập tên của chế độ kia không còn hợp lệ)."""
        new_mode = "multi" if self._rb_multi.isChecked() else "single"
        if new_mode == self._mode:
            return
        self._mode = new_mode
        self._results.clear()
        self._groups.clear()
        self._list.clear()
        self._btn_save.setEnabled(False)
        self._status.setText("Chưa chọn ảnh.")
        self._apply_mode_texts()

    def _load_detector(self) -> None:
        """Nạp model SCRFD + ArcFace (chung với CameraView nếu đã nạp)."""
        try:
            if not (MODELS_ROOT / "models" / "buffalo_l").exists():
                ensure_model_available(MODELS_ROOT)
            self._detector = FaceDetector(MODELS_ROOT, det_size=(256, 256))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Không nạp được model")
            self._status.setText(f"⚠ Không nạp được model: {exc}")

    # ---------------------------------------------------------
    # Chọn ảnh
    # ---------------------------------------------------------
    def _on_pick_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Chọn ảnh nhân viên (chọn nhiều ảnh bằng Ctrl/Shift)",
            "",
            "Ảnh (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)",
        )
        if paths:
            self._start_process(paths)

    def _on_pick_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Chọn thư mục chứa ảnh nhân viên")
        if d:
            # Thư mục nhiều người → tự chuyển sang chế độ NHIỀU nhân viên
            if not self._rb_multi.isChecked():
                self._rb_multi.setChecked(True)
            self._start_process([d])

    # ---------------------------------------------------------
    # Xử lý (QThread)
    # ---------------------------------------------------------
    def _on_process(self) -> None:
        """Nút [⚡ Xử lý ảnh]: chạy lại với ảnh đã chọn gần nhất.

        Dùng khi: lần nạp model đầu chưa xong/lỗi, hoặc người dùng muốn
        xử lý lại sau khi đã đóng kết quả cũ. Chưa chọn ảnh bao giờ →
        nhắc chọn trước.
        """
        if not self._last_paths:
            self._status.setText(
                "Hãy bấm [📁 Chọn ảnh...] hoặc [📂 Chọn thư mục...] trước."
            )
            return
        self._start_process(self._last_paths)

    def _start_process(self, paths: list[str]) -> None:
        """Khởi động worker xử lý ảnh (nạp service nếu chưa có)."""
        self._last_paths = list(paths)  # lưu lại cho nút [Xử lý ảnh] chạy lại
        if self._detector is None:
            self._load_detector()
            if self._detector is None:
                self._btn_process.setEnabled(True)  # nút còn bấm được để thử lại
                return
        if self._service is None:
            self._service = EnrollmentPhotoService(self._detector)

        # Dọn kết quả cũ
        self._results.clear()
        self._groups.clear()
        self._list.clear()
        self._btn_process.setEnabled(False)
        self._btn_save.setEnabled(False)
        self._progress.setValue(0)
        self._progress.show()

        self._thread = QThread(self)
        self._worker = PhotoEnrollWorker(paths, self._service, mode=self._mode)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.one_done.connect(self._on_one_done)
        self._worker.finished_all.connect(self._on_finished_all)
        self._worker.groups_ready.connect(self._on_groups_ready)
        self._worker.error.connect(self._on_error)
        self._thread.start()

    @Slot(int, int, str)
    def _on_progress(self, done: int, total: int, name: str) -> None:
        self._progress.setMaximum(total)
        self._progress.setValue(done - 1)
        self._status.setText(f"Đang xử lý {done}/{total}: {name}")

    @Slot(object)
    def _on_one_done(self, result: PhotoProcessResult) -> None:
        """Một ảnh xong → thêm dòng kết quả (thumbnail + checkbox nếu đạt).

        QUAN TRỌNG: ``self._results`` và các item trong ``_list`` phải LUÔN
        đi đôi với nhau (mỗi result có đúng 1 item) — mọi lỗi vẽ thumbnail
        đều được nuốt để ``addItem`` vẫn chạy. Trước đây nếu vẽ icon lỗi
        giữa chừng thì item không được tạo → lúc bấm Lưu ``item(i)`` trả
        None → crash "'NoneType' object has no attribute 'checkState'".
        """
        self._results.append(result)

        item = QListWidgetItem()
        try:
            if result.ok and result.face_crop is not None:
                import cv2

                rgb = cv2.cvtColor(result.face_crop, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
                icon = QPixmap.fromImage(qimg).scaled(
                    64, 64, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                item.setIcon(icon)
                item.setText(f"✅ OK — {result.path}")
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked)  # mặc định CHỌN
            else:
                item.setText(f"❌ {result.path}\n      Lý do: {result.reason}")
                item.setForeground(Qt.GlobalColor.gray)
        except Exception:  # noqa: BLE001 — lỗi vẽ không được phá cặp result/item
            logger.exception("Lỗi hiển thị kết quả ảnh %s", result.path)
            item.setText(f"✅ OK (không xem trước được) — {result.path}")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
        item.setData(Qt.ItemDataRole.UserRole, result.path)  # tra cứu nhóm sau
        self._list.addItem(item)
        self._progress.setValue(self._progress.value() + 1)

    @Slot(list)
    def _on_groups_ready(self, groups: list) -> None:
        """Worker gom nhóm xong (chế độ NHIỀU người) → gắn tên nhóm vào nhãn."""
        self._groups = list(groups)
        # Ánh xạ path → tên nhóm (path là duy nhất trong lô)
        path2group: dict[str, PersonGroup] = {}
        for g in self._groups:
            for r in g.results:
                path2group[r.path] = g
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item is None or not (item.flags() & Qt.ItemFlag.ItemIsUserCheckable):
                continue
            path = item.data(Qt.ItemDataRole.UserRole)
            g = path2group.get(path)
            if g is not None:
                badge = (
                    f"[{g.name}]" if g.name_source == "filename"
                    else f"[{g.name} — AI chưa biết tên]"
                )
                item.setText(f"{badge} ✅ {path}")
        self._apply_mode_texts()

    @Slot(object)
    def _on_rename_group(self, item: QListWidgetItem) -> None:
        """NHẤP ĐÚP vào 1 ảnh (chế độ NHIỀU người) → SỬA THÔNG TIN NHÓM chứa
        ảnh đó (tên + phòng ban + chức vụ) kèm xem lại các ảnh của nhóm.

        Dùng khi AI gợi ý tên từ tên file bị SAI (vd file đặt tên khác
        người trong ảnh) — đổi cả NHÓM, tên sẽ được dùng khi lưu.
        """
        if self._mode != "multi" or not self._groups:
            return
        if not (item.flags() & Qt.ItemFlag.ItemIsUserCheckable):
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        target: PersonGroup | None = None
        for g in self._groups:
            if any(r.path == path for r in g.results):
                target = g
                break
        if target is None:
            return
        dlg = GroupNamingDialog(
            target.name,
            [r.face_crop for r in target.results if r.face_crop is not None],
            [r.path for r in target.results],
            suggested_department=target.department,
            suggested_position=target.position,
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        name, dept, pos = dlg.values()
        if not name:
            return
        target.name = name
        target.department = dept
        target.position = pos
        target.name_source = "filename"  # tên đã do người dùng chốt — lưu luôn
        self._refresh_group_labels()
        self._apply_mode_texts()
        self._status.setText(f"Đã đổi nhóm → '{name}' (áp dụng khi lưu)")

    def _refresh_group_labels(self) -> None:
        """Cập nhật nhãn [tên nhóm] trên TẤT CẢ item (sau khi đổi nhóm)."""
        path2group: dict[str, PersonGroup] = {}
        for g in self._groups:
            for r in g.results:
                path2group[r.path] = g
        for i in range(self._list.count()):
            it = self._list.item(i)
            if it is None or not (it.flags() & Qt.ItemFlag.ItemIsUserCheckable):
                continue
            p = it.data(Qt.ItemDataRole.UserRole)
            g = path2group.get(p)
            if g is not None:
                it.setText(f"[{g.name}] ✅ {p}")

    @Slot(int, int)
    def _on_finished_all(self, ok_count: int, total: int) -> None:
        self._thread.quit()
        self._thread.wait(3000)
        self._thread = None
        self._worker = None
        self._progress.hide()
        fail = total - ok_count
        summary = f"Hoàn tất: {ok_count}/{total} ảnh đạt"
        if fail:
            summary += f" · {fail} ảnh bị loại (xem lý do bên dưới)"
        if self._mode == "multi":
            summary += f" · gom được {len(self._groups)} NGƯỜI"
            if ok_count > 0 and len(self._groups) == 1:
                summary += " — thư mục này chỉ có 1 người, có thể chuyển chế độ 👤"
        self._status.setText(summary)
        self._btn_process.setEnabled(True)
        self._btn_save.setEnabled(ok_count > 0)
        if ok_count == 0:
            self._status.setText(
                summary + "\n⚠ Không có mẫu nào — chọn ảnh khác nét hơn hoặc chụp lại."
            )

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        self._thread.quit()
        self._thread.wait(3000)
        self._thread = None
        self._worker = None
        self._progress.hide()
        self._btn_process.setEnabled(True)
        self._status.setText(f"⚠ {msg}")

    # ---------------------------------------------------------
    # Lưu
    # ---------------------------------------------------------
    def _chosen_indices(self) -> list[int]:
        """Chỉ số các ảnh ĐẠT đang được CHECK trong danh sách.

        Ánh xạ qua PATH (UserRole) thay vì giả định item thứ i ứng với
        ``results[i]`` — chống lệch pha giữa 2 danh sách (trước đây khi 1
        slot vẽ item lỗi giữa chừng, item không được tạo → ``item(i)``
        trả None → crash "'NoneType' object has no attribute 'checkState'"
        lúc bấm Lưu).
        """
        by_path = {r.path: i for i, r in enumerate(self._results)}
        out: list[int] = []
        for row in range(self._list.count()):
            it = self._list.item(row)
            if it is None or not (it.flags() & Qt.ItemFlag.ItemIsUserCheckable):
                continue
            if it.checkState() != Qt.CheckState.Checked:
                continue
            idx = by_path.get(it.data(Qt.ItemDataRole.UserRole))
            if idx is not None:
                r = self._results[idx]
                if r.ok and r.sample is not None:
                    out.append(idx)
        return out

    def _on_save_single(self) -> None:
        """Chế độ MỘT nhân viên — hành vi cũ giữ nguyên."""
        name = self._name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Thiếu thông tin", "Vui lòng nhập họ tên nhân viên.")
            return
        chosen = [self._results[i].sample for i in self._chosen_indices()]
        if not chosen:
            QMessageBox.warning(
                self, "Không có mẫu",
                "Không có ảnh nào được chọn — tick ✅ vào ít nhất 1 ảnh đạt.",
            )
            return
        n_deselect = 0
        for row in range(self._list.count()):
            it = self._list.item(row)
            if (
                it is not None
                and (it.flags() & Qt.ItemFlag.ItemIsUserCheckable)
                and it.checkState() != Qt.CheckState.Checked
            ):
                n_deselect += 1
        msg = f"Lưu '{name}' với {len(chosen)} mẫu khuôn mặt?"
        if n_deselect:
            msg += f"\n({n_deselect} mẫu bị bỏ chọn sẽ KHÔNG được lưu)"
        if QMessageBox.question(self, "Xác nhận lưu", msg) != QMessageBox.StandardButton.Yes:
            return
        try:
            service = EnrollmentService(self._db)
            person = service.save_person(
                name=name,
                samples=chosen,
                department=self._dept_edit.text().strip(),
                position=self._pos_edit.text().strip(),
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Không lưu được", str(exc))
            return
        except Exception as exc:  # noqa: BLE001 — lỗi DB
            logger.exception("Lỗi lưu nhân viên từ ảnh")
            QMessageBox.critical(self, "Lỗi", f"Không lưu được nhân viên:\n{exc}")
            return
        self._saved = True
        self._saved_count = 1
        QMessageBox.information(
            self, "Thành công",
            f"✅ Đã lưu nhân viên '{person.name}'\n"
            f"Mã NV: NV{person.id[:6].upper()}\n"
            f"Số mẫu: {len(chosen)}",
        )
        self.accept()

    def _on_save_multi(self) -> None:
        """Chế độ NHIỀU nhân viên — lưu HÀNG LOẠT từng nhóm đã gom.

        MỖI NHÓM hiện hộp XEM ẢNH (các mặt trong nhóm) + nhập TÊN,
        PHÒNG BAN, CHỨC VỤ — bỏ qua nhóm nào thì nhóm đó không lưu.
        """
        if not self._groups:
            QMessageBox.warning(self, "Chưa có nhóm", "Chưa gom được nhóm nào — bấm ⚡ Xử lý ảnh trước.")
            return
        chosen_idx = self._chosen_indices()
        if not chosen_idx:
            QMessageBox.warning(
                self, "Không có mẫu",
                "Không có ảnh nào được chọn — tick ✅ vào ít nhất 1 ảnh đạt.",
            )
            return
        chosen_set = set(chosen_idx)

        # Xây danh sách (nhóm, mẫu) — chỉ giữ nhóm còn ≥ 1 mẫu được tick.
        # Ánh xạ path → chỉ số (KHÔNG dùng list.index() — dataclass chứa
        # numpy array, so sánh __eq__ từng phần tử gây lỗi truth-value).
        path2idx = {r.path: i for i, r in enumerate(self._results)}
        pending: list[tuple[PersonGroup, list]] = []  # (nhóm, mẫu được tick)
        for g in self._groups:
            samples = [
                r.sample for r in g.results
                if r.sample is not None
                and path2idx.get(r.path) in chosen_set
            ]
            if not samples:
                continue
            pending.append((g, samples))
        if not pending:
            QMessageBox.warning(self, "Không có nhóm", "Không còn nhóm nào có ảnh được chọn.")
            return

        # HIỂN THỊ ẢNH TỪNG NHÓM → người dùng xem mặt rồi đặt TÊN + PHÒNG
        # BAN + CHỨC VỤ (GroupNamingDialog). Bỏ qua nhóm nào → nhóm đó
        # không được lưu, các nhóm khác vẫn lưu bình thường.
        plan: list[tuple[str, list, str, str]] = []  # (tên, mẫu, phòng ban, chức vụ)
        skipped: list[str] = []
        for g, samples in pending:
            crops = [r.face_crop for r in g.results if r.face_crop is not None]
            paths = [r.path for r in g.results]
            dlg = GroupNamingDialog(
                g.name, crops, paths,
                suggested_department=g.department,
                suggested_position=g.position,
                parent=self,
            )
            if dlg.exec() != QDialog.DialogCode.Accepted:
                skipped.append(g.name)
                continue
            name, dept, pos = dlg.values()
            if not name:
                skipped.append(g.name)
                continue
            plan.append((name, samples, dept, pos))
        if not plan:
            QMessageBox.warning(
                self, "Không có nhóm nào được lưu",
                "Bạn đã bỏ qua MỌI nhóm nên không có gì được lưu.\n\n"
                "Mẹo: đặt tên file dạng TenNhanVien_1.jpg để app gợi ý sẵn tên.",
            )
            return

        # Xác nhận tổng thể (tên/phòng ban/chức vụ đã là giá trị cuối)
        detail_lines = []
        for n, s, d, p in plan:
            extra = f" — {d}" + (f" · {p}" if p else "") if (d or p) else ""
            detail_lines.append(f"• {n}{extra}: {len(s)} mẫu")
        msg = f"Lưu {len(plan)} NHÂN VIÊN cùng lúc?\n\n" + "\n".join(detail_lines)
        if QMessageBox.question(self, "Xác nhận lưu hàng loạt", msg) != QMessageBox.StandardButton.Yes:
            return

        # Lưu từng nhóm — trùng tên người CÓ SẴN trong DB → thêm mẫu vào hồ sơ cũ
        service = EnrollmentService(self._db)
        saved_names: list[str] = []
        failed: list[str] = []
        added_to_existing: list[str] = []
        for name, samples, dept, pos in plan:
            try:
                existing = service.find_by_name(name)
                if existing is not None:
                    service.add_samples(existing, samples)
                    if dept or pos:
                        # Người có sẵn + điền phòng ban/chức vụ mới → cập nhật luôn
                        service.update_info(
                            existing,
                            department=dept or existing.department,
                            position=pos or existing.position,
                        )
                    added_to_existing.append(name)
                else:
                    service.save_person(
                        name=name, samples=samples, department=dept, position=pos
                    )
                saved_names.append(name)
            except ValueError as exc:
                failed.append(f"{name}: {exc}")
            except Exception as exc:  # noqa: BLE001 — lỗi DB từng người
                logger.exception("Lỗi lưu nhân viên %s (ảnh)", name)
                failed.append(f"{name}: {exc}")

        self._saved = len(saved_names) > 0
        self._saved_count = len(saved_names)
        if saved_names:
            names_list = ", ".join(saved_names)
            extra = (
                f"\n(+{len(added_to_existing)} người trùng tên → thêm mẫu vào hồ sơ cũ)"
                if added_to_existing else ""
            )
            skipped_note = (
                f"\n\n⏭ Đã bỏ qua {len(skipped)} nhóm do bạn hủy đặt tên: "
                + ", ".join(skipped)
                if skipped else ""
            )
            QMessageBox.information(
                self, "Thành công",
                f"✅ Đã lưu {len(saved_names)} nhân viên:\n{names_list}{extra}{skipped_note}"
                + (f"\n\n⚠ Lỗi {len(failed)} người:\n" + "\n".join(failed) if failed else ""),
            )
            self.accept()
        else:
            QMessageBox.critical(
                self, "Lỗi", f"Không lưu được nhân viên nào:\n" + "\n".join(failed)
            )

    def _on_save(self) -> None:
        """Xác nhận + lưu theo chế độ đang chọn.

        Bắt MỌI exception không lường trước → hiện hộp thoại ĐỎ kèm
        traceback: slot Qt mặc định chỉ in traceback ra CONSOLE — người
        dùng chạy run.bat trên Windows sẽ không thấy gì nếu không bắt
        ở đây (trông như "bấm lưu nhưng không có gì xảy ra").
        """
        import traceback

        try:
            if self._rb_multi.isChecked():
                self._on_save_multi()
            else:
                self._on_save_single()
        except Exception as exc:  # noqa: BLE001 — lỗi bất ngờ phải HIỆN RA
            logger.exception("Lỗi không mong đợi khi lưu nhân viên từ ảnh")
            QMessageBox.critical(
                self, "Lỗi không mong đợi",
                f"Không lưu được nhân viên:\n{exc}\n\n"
                f"Chi tiết kỹ thuật (gửi hỗ trợ nếu cần):\n"
                f"{traceback.format_exc()[-1000:]}",
            )

    # ---------------------------------------------------------
    # Kết quả cho caller
    # ---------------------------------------------------------
    @property
    def saved(self) -> bool:
        """True nếu đã lưu ÍT NHẤT 1 nhân viên thành công."""
        return self._saved

    @property
    def saved_count(self) -> int:
        """Số hồ sơ nhân viên đã lưu thành công (chế độ nhiều người)."""
        return self._saved_count

    def closeEvent(self, event) -> None:  # noqa: N802
        """Đóng dialog khi worker đang chạy → hủy an toàn."""
        if self._worker is not None:
            self._worker.cancel()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(2000)
        super().closeEvent(event)
