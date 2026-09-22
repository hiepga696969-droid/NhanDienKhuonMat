"""PhotoView — nhận diện ảnh tĩnh (Bước 10, wireframe 5.5.8).

Luồng dùng: [ Mở ảnh… ] HOẶC kéo-thả ảnh vào trang → hiển thị ảnh gốc →
[ Nhận diện ] chạy pipeline MỘT LẦN trên QThread (không đơ UI với ảnh lớn)
→ mỗi khuôn mặt một khung + tên + điểm (hoặc "Người lạ" khung đỏ) → tự
động lưu recognition_events với ``source='photo'``.

Với người lạ trong ảnh: [ Đăng ký ngay ] mở EnrollmentDialog với ảnh crop
làm MẪU 1 (giống Bước 9) — sau khi đăng ký xong, tự chạy lại nhận diện để
người đó giờ được nhận diện trong ảnh.

Tái sử dụng (không viết lại): RecognitionService (nạp embedding + so khớp
+ ghi sự kiện), FaceDetector/FaceEmbedder (Bước 4-5), các hàm vẽ/crop của
CameraWorker (Bước 9), seed_sample của EnrollmentDialog (Bước 9).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import cv2
import numpy as np
from PySide6.QtCore import QObject, QThread, Qt, QTimer, Signal, Slot
from PySide6.QtGui import (
    QDragEnterEvent,
    QDropEvent,
    QImage,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.config import Config
from app.core import face_metrics
from app.core.detector import FaceDetector, MODELS_ROOT, ensure_model_available
from app.core.embedder import FaceEmbedder
from app.core.preprocess import preprocess_frame
from app.infrastructure.db import Database
from app.services.enrollment import CapturedSample
from app.services.attendance import AttendanceService
from app.services.recognition import RecognitionService
from app.ui.camera_view import (
    KNOWN_COLOR,
    SPOOF_COLOR,
    UNKNOWN_COLOR,
    CameraWorker,
)
from app.ui.enrollment_dialog import EnrollmentDialog

logger = logging.getLogger(__name__)

# Phần mở rộng file ảnh nhận khi KÉO-THẢ (khớp bộ lọc hộp thoại Mở ảnh +
# thêm .bmp — cv2 đọc tốt; imdecode vẫn chặn file giả đuôi nên an toàn).
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _imread_unicode(path: str) -> np.ndarray | None:
    """Đọc ảnh an toàn với đường dẫn CÓ DẤU TIẾNG VIỆT (Windows).

    ``cv2.imread`` dùng fopen ANSI → đường dẫn có ký tự ngoài ASCII (vd
    'D:/OneDrive/Máy tính/anh.jpg') trả về None ở trạng thái 'Không đọc
    được ảnh' dù file có tồn tại. Giải pháp chuẩn của OpenCV: đọc byte
    bằng np.fromfile (hỗ trợ Unicode) rồi giải mã bằng cv2.imdecode.
    Trả về None khi file hỏng/không phải ảnh — caller hiện thông báo.
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        logger.error("Không đọc được file ảnh %s: %s", path, exc)
        return None
    if data.size == 0:
        logger.error("File ảnh rỗng: %s", path)
        return None
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        logger.error("File không phải ảnh hợp lệ (imdecode thất bại): %s", path)
    return image


@dataclass
class PhotoFaceResult:
    """Kết quả nhận diện MỘT khuôn mặt trong ảnh."""

    index: int                # số thứ tự hiển thị (1-based)
    label: str                # tên người hoặc "Người lạ"
    similarity: float | None  # điểm tương đồng (người lạ → None)
    is_unknown: bool
    person_id: str | None
    bbox: tuple               # (x1, y1, x2, y2)
    crop: np.ndarray | None   # ảnh crop khuôn mặt (làm mẫu 1 khi đăng ký ngay)
    embedding: np.ndarray | None
    quality: float
    occluded: bool = False    # mặt bị che khuất / không rõ (Bước 18)


class PhotoWorker(QObject):
    """Nhận diện một ảnh tĩnh MỘT LẦN — chạy trong QThread (ảnh lớn không đơ UI)."""

    finished = Signal(object, object)  # (ảnh đã vẽ BGR, list[PhotoFaceResult])
    error = Signal(str)
    progress = Signal(str)  # tiến trình (nạp model, số mặt tìm thấy...)

    def __init__(
        self,
        image: np.ndarray,
        detector: FaceDetector,
        embedder: FaceEmbedder,
        service: RecognitionService | None,
        threshold: float,
    ) -> None:
        super().__init__()
        self._image = image
        self._detector = detector
        self._embedder = embedder
        self._service = service
        self._threshold = threshold

    @Slot()
    def run(self) -> None:
        """Nhận diện toàn bộ ảnh — MỌI exception được bắt để không kẹt UI.

        Trước đây chỉ bắt lỗi ở bước detect: exception ở bước sau (occlusion,
        vẽ, so khớp) thoát khỏi run() → thread không phát 'finished/error'
        → nút Quét kẹt 'Đang nhận diện...' vĩnh viễn + rò rỉ luồng. Now:
        try/except bao toàn bộ → lỗi nào cũng trả về UI qua error signal.
        """
        try:
            self._run_pipeline()
        except Exception as exc:  # noqa: BLE001 — bắt tất cả để UI không kẹt
            logger.exception("Lỗi pipeline nhận diện ảnh")
            self.error.emit(f"Không nhận diện được ảnh: {exc}")

    def _run_pipeline(self) -> None:
        # Bước 21: CLAHE preprocessing — detect trên ảnh đã chuẩn hóa ánh sáng
        # (đồng bộ với luồng đăng ký — embedding cùng điều kiện ánh sáng)
        detect_image = preprocess_frame(self._image)
        try:
            faces = self._detector.detect(detect_image)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Lỗi phát hiện khuôn mặt trong ảnh")
            self.error.emit(f"Không nhận diện được ảnh: {exc}")
            return

        self.progress.emit(f"Đã tìm thấy {len(faces)} khuôn mặt — đang so khớp...")

        frame = self._image.copy()
        results: list[PhotoFaceResult] = []
        # Sobel gradient dùng chung CHO CẢ ẢNH (tính 1 LẦN): trước đây hàm
        # texture_anomaly tính lại Sobel toàn ảnh cho TỪNG mặt → ảnh lớn
        # nhiều mặt quét chậm như treo (ảnh 4K × 5 mặt = 5 lần Sobel 4K).
        sobel_mag = face_metrics.sobel_magnitude(self._image)
        for i, face in enumerate(faces, start=1):
            x1, y1, x2, y2 = face.bbox.astype(int)
            embedding = self._embedder.embed_face(face) if self._embedder else None

            result = None
            if self._service is not None and embedding is not None:
                result = self._service.match(embedding, self._threshold)

            crop = CameraWorker._crop_face(self._image, face)  # crop từ ảnh GỐC
            occ = face_metrics.occlusion_score(face, self._image, sobel_mag=sobel_mag)
            occluded = occ >= face_metrics.OCCLUSION_WARN
            if result is not None:
                label = self._service.label_of(result.person_id) if self._service else "?"
                if occluded:
                    label += " ⚠ bị che"
                CameraWorker._draw_label(
                    frame,
                    f"{label} ({result.similarity:.2f})",
                    x1, max(0, y1 - 8),
                    SPOOF_COLOR if occluded else KNOWN_COLOR,
                )
                cv2.rectangle(frame, (x1, y1), (x2, y2), KNOWN_COLOR, 2)
                results.append(
                    PhotoFaceResult(
                        index=i, label=label, similarity=result.similarity,
                        is_unknown=False, person_id=result.person_id,
                        bbox=(x1, y1, x2, y2), crop=crop,
                        embedding=embedding, quality=float(face.det_score),
                        occluded=occluded,
                    )
                )
            else:
                label = "⚠ Mặt bị che / không rõ" if occluded else "Người lạ"
                CameraWorker._draw_label(
                    frame, label, x1, max(0, y1 - 8),
                    SPOOF_COLOR if occluded else UNKNOWN_COLOR,
                )
                cv2.rectangle(frame, (x1, y1), (x2, y2), UNKNOWN_COLOR, 2)
                results.append(
                    PhotoFaceResult(
                        index=i, label=label, similarity=None,
                        is_unknown=True, person_id=None,
                        bbox=(x1, y1, x2, y2), crop=crop,
                        embedding=embedding, quality=float(face.det_score),
                        occluded=occluded,
                    )
                )

        self.finished.emit(frame, results)


class PhotoView(QWidget):
    """Trang nhận diện ảnh tĩnh (FR-3)."""

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
        self._embedder: FaceEmbedder | None = None
        self._service = RecognitionService(db)
        self._attendance = AttendanceService(db)
        self._thread: QThread | None = None
        self._worker: PhotoWorker | None = None
        self._image: np.ndarray | None = None
        self._image_path: str = ""
        self._results: list[PhotoFaceResult] = []
        # Kéo-thả ảnh trực tiếp vào trang (thay/cạnh nút [ Mở ảnh… ]):
        # chấp nhận 1+ URL kéo từ Explorer/Desktop vào khung xem ảnh.
        self.setAcceptDrops(True)
        self._build_ui()

    # ---------------------------------------------------------
    # Giao diện (wireframe 5.5.8)
    # ---------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        # Thanh công cụ: mở ảnh + tên file + nút nhận diện
        toolbar = QHBoxLayout()
        open_btn = QPushButton("📂 Mở ảnh…")
        open_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        open_btn.clicked.connect(self._open_image)
        toolbar.addWidget(open_btn)

        # Nút NHANH: chọn file ảnh từ máy tính → tải + quét nhận diện +
        # điểm danh LUÔN (không phải bấm thêm [Quét Face ID] như Mở ảnh).
        self._quick_open_btn = QPushButton("📤 Tải ảnh từ máy & quét ngay")
        self._quick_open_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._quick_open_btn.setObjectName("accentBtn")
        self._quick_open_btn.setToolTip(
            "Chọn 1 file ảnh từ máy tính (jpg/png/webp/bmp)\n"
            "→ nhận diện + điểm danh theo chế độ VÀO/RA CA đang chọn."
        )
        self._quick_open_btn.clicked.connect(self._open_and_analyze)
        toolbar.addWidget(self._quick_open_btn)

        self._file_label = QLabel("Chưa chọn ảnh")
        self._file_label.setStyleSheet("font-size: 15px;")
        toolbar.addWidget(self._file_label, stretch=1)

        # QUAN TRỌNG: thanh công cụ phải được GẮN vào layout của trang —
        # trước đây thiếu dòng này → cả nút Mở ảnh lẫn tên file KHÔNG HIỆN
        # (chỉ kéo-thả ảnh vào trang mới dùng được).
        layout.addLayout(toolbar)

        # Tiêu đề trang + nút quét kèm CHẾ ĐỘ đang chọn (nhãn đổi theo nút mode)
        title_row = QHBoxLayout()
        title = QLabel("Điểm danh bằng ảnh")
        title.setObjectName("pageTitle")
        title_row.addWidget(title)
        title_row.addStretch(1)
        self._analyze_btn = QPushButton("▶ Quét Face ID — VÀO CA")
        self._analyze_btn.setObjectName("primaryBtn")
        self._analyze_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._analyze_btn.setEnabled(False)  # chưa có ảnh
        self._analyze_btn.clicked.connect(self._analyze)
        title_row.addWidget(self._analyze_btn)
        layout.addLayout(title_row)

        # ── Chế độ điểm danh ảnh: VÀO CA / RA CA ──
        mode_row = QHBoxLayout()
        mode_label = QLabel("Chế độ:")
        mode_label.setStyleSheet("font-size: 14px;")
        mode_row.addWidget(mode_label)
        self._checkin_btn = QPushButton("🟢 VÀO CA")
        self._checkin_btn.setCheckable(True)
        self._checkin_btn.setChecked(True)
        self._checkin_btn.setToolTip("Nhận diện người đã biết → ghi GIỜ VÀO")
        self._checkin_btn.clicked.connect(self._on_mode_checkin)
        mode_row.addWidget(self._checkin_btn)
        self._checkout_btn = QPushButton("🔴 RA CA")
        self._checkout_btn.setCheckable(True)
        self._checkout_btn.setToolTip("Nhận diện người đã biết → ghi GIỜ RA = lúc quét")
        self._checkout_btn.clicked.connect(self._on_mode_checkout)
        mode_row.addWidget(self._checkout_btn)
        mode_row.addStretch(1)
        layout.addLayout(mode_row)

        # ── Ghi chú cho lần quét ảnh (v4) ──
        # Ghi chú lưu RIÊNG theo tính năng: VÀO CA → note_in, RA CA → note_out.
        note_row = QHBoxLayout()
        note_label = QLabel("Ghi chú:")
        note_label.setStyleSheet("font-size: 14px;")
        note_row.addWidget(note_label)
        self._note_edit = QLineEdit()
        self._note_edit.setPlaceholderText("Tùy chọn — ví dụ: 'ca đêm', 'làm thêm giờ'...")
        self._note_edit.setClearButtonEnabled(True)
        self._note_edit.setToolTip(
            "Ghi chú lưu kèm điểm danh theo chế độ đang chọn:\n"
            "VÀO CA → ghi chú GIỜ VÀO · RA CA → ghi chú GIỜ RA."
        )
        note_row.addWidget(self._note_edit, stretch=1)
        layout.addLayout(note_row)

        # Vùng chính: ảnh + bảng kết quả
        body = QHBoxLayout()
        body.setSpacing(12)

        self._image_label = QLabel(
            "📂 Mở ảnh… hoặc KÉO-THẢ ảnh vào đây (jpg/png/webp/bmp)\nđể nhận diện khuôn mặt"
        )
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setMinimumSize(560, 420)
        self._image_label.setObjectName("imageLabel")
        self._image_label.setStyleSheet("font-size: 18px;")
        self._image_label.setWordWrap(True)
        body.addWidget(self._image_label, stretch=1)

        # Bảng kết quả bên phải
        panel = QVBoxLayout()
        panel.setSpacing(8)
        results_title = QLabel("KẾT QUẢ")
        results_title.setObjectName("sectionTitle")
        panel.addWidget(results_title)

        self._results_list = QListWidget()
        # Màu nền/viền do theme QSS quyết định — chỉ giữ bo góc
        self._results_list.setStyleSheet("border-radius: 6px;")
        panel.addWidget(self._results_list, stretch=1)

        self._status_label = QLabel("—")
        self._status_label.setStyleSheet("font-size: 15px;")
        panel.addWidget(self._status_label)
        body.addLayout(panel)
        layout.addLayout(body, stretch=1)

        hint = QLabel("Mẹo: ảnh càng rõ mặt, càng ít người → kết quả càng chính xác")
        hint.setStyleSheet("font-size: 14px;")
        layout.addWidget(hint)

    # ---------------------------------------------------------
    # Chế độ điểm danh VÀO CA / RA CA (ảnh)
    # ---------------------------------------------------------
    @Slot()
    def _on_mode_checkin(self) -> None:
        self._checkin_btn.setChecked(True)
        self._checkout_btn.setChecked(False)
        self._analyze_btn.setText("▶ Quét Face ID — VÀO CA")

    def _current_note(self) -> str:
        """Nội dung ghi chú đang nhập (rỗng = không có ghi chú)."""
        return self._note_edit.text().strip()

    @Slot()
    def _on_mode_checkout(self) -> None:
        self._checkout_btn.setChecked(True)
        self._checkin_btn.setChecked(False)
        self._analyze_btn.setText("▶ Quét Face ID — RA CA")

    @property
    def _is_checkin_mode(self) -> bool:
        return self._checkin_btn.isChecked()

    # ---------------------------------------------------------
    # Mở ảnh (hộp thoại) + KÉO-THẢ ảnh vào trang
    # ---------------------------------------------------------
    def _open_image(self) -> None:
        """Chọn file ảnh → hiển thị ảnh gốc (chưa nhận diện)."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn ảnh",
            "",
            "Ảnh (*.jpg *.jpeg *.png *.webp *.bmp);;Tất cả (*.*)",
        )
        if not path:
            return
        self._load_image(path)

    @Slot()
    def _open_and_analyze(self) -> None:
        """[Tải ảnh từ máy & quét ngay]: chọn file → tải + chạy nhận diện LUÔN.

        Khác [ Mở ảnh… ]: không dừng ở bước hiển thị ảnh gốc — tự chạy
        pipeline 'Quét Face ID' giúp người dùng, kết quả kèm điểm danh
        theo chế độ VÀO CA / RA CA đang chọn (cùng luồng _on_finished).
        Ảnh lỗi/không đọc được → chỉ hiện cảnh báo, KHÔNG quét ảnh cũ còn
        sót trong bộ nhớ (tránh điểm danh nhầm người trong ảnh trước).
        """
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn ảnh từ máy tính để nhận diện",
            "",
            "Ảnh (*.jpg *.jpeg *.png *.webp *.bmp);;Tất cả (*.*)",
        )
        if not path:
            return
        if not self._load_image(path):
            return  # ảnh lỗi — _load_image đã hiện cảnh báo
        # Ảnh tải OK → quét luôn (nút Quét vừa được _load_image bật)
        self._analyze()

    def _load_image(self, path: str) -> bool:
        """Đọc + hiển thị ảnh từ đường dẫn (dùng chung hộp thoại & kéo-thả).

        Đọc qua ``_imread_unicode`` — hỗ trợ đường dẫn CÓ DẤU tiếng Việt
        trên Windows (cv2.imread chỉ đọc được đường dẫn ASCII).
        Trả True khi ảnh tải OK (caller 'quét ngay' dựa vào giá trị này);
        False khi file hỏng (đã hiện cảnh báo, trạng thái cũ giữ nguyên).
        """
        image = _imread_unicode(path)
        if image is None:
            QMessageBox.warning(
                self, "Lỗi ảnh",
                f"Không đọc được ảnh (file hỏng hoặc định dạng không hỗ trợ):\n{path}",
            )
            return False
        self._image = image
        self._image_path = path
        # os.path.basename đúng cả '\' (Windows) lẫn '/' — split('/') cũ
        # chỉ hiện 'Máy tính\anh.jpg' khi đường dẫn dùng backslash.
        self._file_label.setText(
            f"Ảnh: {os.path.basename(path)} ({image.shape[1]}×{image.shape[0]})"
        )
        self._analyze_btn.setEnabled(True)
        self._results_list.clear()
        self._status_label.setText("Đã tải ảnh — bấm [Quét Face ID]")
        self._display(image)
        return True

    # ---------------------------------------------------------
    # Kéo-thả ảnh (drag & drop từ Explorer/Desktop)
    # ---------------------------------------------------------
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802 (chuẩn Qt)
        """Chấp nhận kéo khi có ít nhất 1 file ảnh trong dữ liệu kéo.

        Kiểm tra đuôi file qua ``_first_image_path`` — file khác/URL remote
        bị bỏ qua; không có ảnh nào → ignore (trang khác tự xử lý).
        """
        if self._first_image_path(event.mimeData()) is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802 (chuẩn Qt)
        """Thả: lấy file ảnh ĐẦU TIÊN trong danh sách → tải + hiển thị.

        QUrl.toLocalFile() tự decode %20/Unicode trong file:/// URL —
        đường dẫn tiếng Việt vẫn đọc đúng qua ``_imread_unicode``.
        """
        path = self._first_image_path(event.mimeData())
        if path is None:
            QMessageBox.warning(
                self, "Không phải ảnh",
                "File thả vào không phải ảnh (jpg/png/webp/bmp) — hãy thả file ảnh.",
            )
            event.ignore()
            return
        event.acceptProposedAction()
        logger.info("Kéo-thả ảnh vào Điểm danh ảnh: %s", path)
        self._load_image(path)

    @staticmethod
    def _first_image_path(mime):
        """Đường dẫn file ảnh ĐẦU TIÊN trong mimeData; None nếu không có.

        Bỏ qua URL remote (http://...); lọc theo đuôi file trong
        ``_IMAGE_EXTS`` (imdecode vẫn chặn file giả đuôi nên an toàn).
        """
        if not mime.hasUrls():
            return None
        for url in mime.urls():
            path = url.toLocalFile()
            if not path:
                continue  # URL remote — bỏ qua
            if os.path.splitext(path)[1].lower() in _IMAGE_EXTS:
                return path
        return None

    # ---------------------------------------------------------
    # Nhận diện
    # ---------------------------------------------------------
    def _analyze(self) -> None:
        """Chạy pipeline nhận diện ảnh MỘT LẦN trên QThread."""
        if self._image is None:
            return
        # Chống bấm [Nhận diện] 2 lần (kể cả trong lúc đang nạp model —
        # nếu không sẽ có 2 worker chạy song song và ghi TRÙNG sự kiện)
        if self._thread is not None and self._thread.isRunning():
            return
        self._analyze_btn.setEnabled(False)
        self._status_label.setText("Đang nhận diện...")

        # Nạp model (lần đầu) + embedding đã đăng ký (LUỒNG UI — trước khi chạy)
        if self._detector is None:
            self._status_label.setText("Đang nạp model nhận diện...")
            QApplication.processEvents()
            self._detector = self._load_detector()
            if self._detector is None:
                self._analyze_btn.setEnabled(True)  # bật lại nếu model lỗi
                self._status_label.setText("⚠ Không nạp được model")
                return
        if self._embedder is None:
            self._embedder = FaceEmbedder(self._detector)
        try:
            self._service.reload()
        except Exception:  # noqa: BLE001
            logger.exception("Lỗi nạp embedding cho nhận diện ảnh")
            self._service = RecognitionService(self._db)

        self._thread = QThread(self)
        self._worker = PhotoWorker(
            self._image,
            detector=self._detector,
            embedder=self._embedder,
            service=self._service,
            threshold=self._config.recognition_threshold,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.progress.connect(self._status_label.setText)
        # Mỗi lần _analyze() tạo cặp thread/worker MỚI (worker cũ bị GC khi
        # self._worker được gán lại) — không tái dùng → không dính kết nối cũ.
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _load_detector(self) -> FaceDetector | None:
        """Tạo FaceDetector; None nếu model thiếu/lỗi (hiện thông báo).

        Nếu thiếu model buffalo_l → tự tải về (ensure_model_available, cần internet).
        """
        try:
            if not (MODELS_ROOT / "models" / "buffalo_l").exists():
                self._status_label.setText("⏳ Đang tải model buffalo_l (~300MB, lần đầu)...")
                ensure_model_available(MODELS_ROOT)
            detector = FaceDetector(MODELS_ROOT)
            logger.info("Đã nạp model phát hiện khuôn mặt (photo)")
            return detector
        except Exception as exc:  # noqa: BLE001
            logger.exception("Không nạp được model: %s", exc)
            self._status_label.setText(f"⚠ {exc}")
            return None

    # ---------------------------------------------------------
    # Xử lý kết quả
    # ---------------------------------------------------------
    @Slot(object, object)
    def _on_finished(self, frame: np.ndarray, results: list) -> None:
        """Hiển thị ảnh đã vẽ + bảng kết quả + lưu sự kiện (source='photo')."""
        self._stop_thread()
        self._analyze_btn.setEnabled(True)
        self._results = results
        self._display(frame)
        self._populate_results(results)

        # Ghi sự kiện nhận diện + ĐIỂM DANH theo chế độ (LUỒNG UI — worker
        # không đụng DB). VÀO CA → giờ vào; RA CA → giờ ra = lúc quét ảnh.
        # Kết quả từng người gom lại để báo 1 hộp thoại tổng (ảnh có thể
        # nhiều mặt — không hiện dialog từng người).
        saved = 0
        lines: list[str] = []
        warnings = 0
        errors = 0
        note = self._current_note()
        mode_key = "checkin" if self._is_checkin_mode else "checkout"
        for r in results:
            if r.is_unknown:
                if r.crop is not None:
                    self._service.save_event(
                        person_id=None, similarity=None,
                        face_crop=r.crop, is_unknown=True, source="photo",
                    )
                    saved += 1
                    lines.append(f"• {r.label} — không điểm danh (người lạ)")
            else:
                self._service.save_event(
                    person_id=r.person_id, similarity=r.similarity,
                    face_crop=r.crop, source="photo", mode=mode_key,
                )
                saved += 1
                name = self._service.label_of(r.person_id)
                mode_txt = "VÀO CA" if self._is_checkin_mode else "RA CA"
                res = (
                    self._attendance.record_check_in(r.person_id, note=note)
                    if self._is_checkin_mode
                    else self._attendance.record_check_out(r.person_id, note=note)
                )
                msg = res.get("message", "")
                if not res.get("ok"):
                    errors += 1
                    lines.append(f"• ✗ {name} — {mode_txt} THẤT BẠI: {msg}")
                elif res.get("already_done") or res.get("was_not_checked_in"):
                    warnings += 1
                    lines.append(f"• ⚠ {name} — {msg}")
                else:
                    lines.append(f"• ✓ {name} — {msg}")
        mode_txt = "VÀO CA" if self._is_checkin_mode else "RA CA"
        summary = f"Đã lưu {saved} sự kiện · điểm danh {mode_txt}: "
        if errors:
            self._status_label.setText(f"⚠ {summary}{errors} lỗi")
            QMessageBox.critical(
                self, f"Điểm danh {mode_txt} — CÓ LỖI",
                "\n".join(lines),
            )
        else:
            self._status_label.setText(f"✓ {summary}xong")
            icon = QMessageBox.Warning if warnings else QMessageBox.Information
            title = (
                f"Điểm danh {mode_txt} thành công"
                if not warnings else f"Điểm danh {mode_txt} — có mục cần lưu ý"
            )
            self._show_auto_close_box(icon, title, "\n".join(lines), timeout_ms=5000)

    def _show_auto_close_box(
        self, icon, title: str, text: str, timeout_ms: int = 5000
    ) -> None:
        """QMessageBox TỰ ĐÓNG sau ``timeout_ms`` — xem kết quả nhanh mà
        không phải bấm OK cho từng hộp thoại."""
        box = QMessageBox(self)
        box.setIcon(icon)
        box.setWindowTitle(title)
        box.setText(text)
        box.setModal(True)
        box.show()
        QTimer.singleShot(timeout_ms, box.close)

    @Slot(str)
    def _on_error(self, message: str) -> None:
        self._stop_thread()
        self._analyze_btn.setEnabled(True)
        self._status_label.setText(f"⚠ {message}")

    def _stop_thread(self) -> None:
        """Dừng luồng nhận diện: worker đã xong → thoát event loop của QThread.

        QUAN TRỌNG: QThread mặc định chạy event loop MÃI (exec()) — việc
        worker.run() kết thúc KHÔNG tự dừng thread. Nếu không gọi quit(),
        thread rò rỉ và bị hủy khi đang chạy lúc view đóng → crash
        'QThread: Destroyed while thread is still running'.
        """
        thread = self._thread
        self._thread = None
        self._worker = None
        if thread is not None:
            thread.quit()
            if not thread.wait(1500):
                # Chưa kịp dừng (đang xử lý ảnh lớn) → GIỮ tham chiếu, tránh
                # hủy C++ object khi luồng còn chạy; sẽ tự dừng khi xong
                self._thread = thread

    def _populate_results(self, results: list) -> None:
        """Bảng kết quả: mỗi khuôn mặt một dòng; người lạ kèm [Đăng ký ngay]."""
        self._results_list.clear()
        for r in results:
            row = QWidget()
            layout = QHBoxLayout(row)
            layout.setContentsMargins(8, 6, 8, 6)
            layout.setSpacing(8)

            if r.is_unknown:
                text = QLabel(f"{r.index}. ⚠ {r.label}")
                text.setStyleSheet("color: #c33; font-weight: bold; font-size: 14px;")
            else:
                score = f"{r.similarity:.2f}" if r.similarity is not None else "—"
                warn = " ⚠ bị che" if r.occluded else ""
                text = QLabel(f"{r.index}. {r.label}{warn} · {score}")
                text.setStyleSheet("font-size: 14px;")
            layout.addWidget(text, stretch=1)

            if r.is_unknown and r.embedding is not None:
                enroll_btn = QPushButton("Đăng ký ngay")
                enroll_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                enroll_btn.setObjectName("accentBtn")
                enroll_btn.setStyleSheet("border-radius: 5px; padding: 4px 10px; font-size: 13px;")
                enroll_btn.clicked.connect(
                    lambda _=False, r=r: self._on_enroll(r)
                )
                layout.addWidget(enroll_btn)

            item = QListWidgetItem()
            item.setSizeHint(row.sizeHint())
            self._results_list.addItem(item)
            self._results_list.setItemWidget(item, row)

    def _on_enroll(self, r: PhotoFaceResult) -> None:
        """[Đăng ký ngay]: ảnh crop người lạ làm MẪU 1 → sau đó chạy lại nhận diện."""
        seed = CapturedSample(
            embedding=r.embedding,
            quality=r.quality,
            face_crop=r.crop,
        )
        dialog = EnrollmentDialog(
            self._config,
            self._db,
            detector=self._detector,
            seed_sample=seed,
            parent=self,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            logger.info("Đăng ký ngay từ ảnh thành công — chạy lại nhận diện")
            self._analyze()  # người giờ đã đăng ký → nhận diện lại để thấy tên

    # ---------------------------------------------------------
    # Hiển thị ảnh
    # ---------------------------------------------------------
    def _display(self, image: np.ndarray) -> None:
        """Chuyển ảnh BGR → pixmap vừa khung hiển thị."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self._image_label.setPixmap(
            pixmap.scaled(
                self._image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    # ---------------------------------------------------------
    # Vòng đời
    # ---------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802 (chuẩn Qt)
        """Đóng view: dừng luồng nhận diện nếu còn chạy (không để crash khi thoát)."""
        self._stop_thread()
        super().closeEvent(event)
