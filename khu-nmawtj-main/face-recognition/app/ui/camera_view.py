"""CameraView — nhận diện thời gian thực hoàn chỉnh (Bước 9, wireframe 5.5.2).

Kiến trúc luồng: CameraWorker chạy trong QThread — đọc frame từ webcam,
vẽ overlay và phát preview về UI. Vòng lặp preview KHÔNG BAO GIỜ bị
chặn (video không khựng): phát hiện + so khớp khuôn mặt (SCRFD + ArcFace,
chậm ~200-300ms) chạy trong LUỒNG PHỤ RIÊNG (threading) với cơ chế
"frame MỚI NHẤT THẮNG" — frame cũ bị ghi đè thay vì xếp hàng chờ.

Bước 9 bổ sung so với Bước 3-5:
  - NHIỀU mặt cùng lúc: mỗi mặt một khung + tên + điểm tương đồng.
  - NGƯỜI LẠ: làm mờ tự động (riêng tư) + khung đỏ + nút [ Đăng ký ngay ]
    (ảnh người lạ làm MẪU 1 khi mở EnrollmentDialog).
  - GHI SỰ KIỆN: mỗi lần nhận diện (đã biết hoặc người lạ) → lưu
    recognition_events + snapshot (debounce 5s/người — tránh tràn DB).

QUAN TRỌNG (đa luồng + sqlite): worker KHÔNG đụng CSDL. Worker dùng bộ
so khớp trong BỘ NHỚ (nạp trước ở luồng UI) và phát tín hiệu sự kiện;
CameraView (luồng UI) mới gọi RecognitionService.save_event.
"""
from __future__ import annotations

import logging
import math
import threading
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PySide6.QtCore import QObject, QThread, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.config import Config
from app.core import face_metrics
from app.core.detector import FaceDetector, MODELS_ROOT, ensure_model_available
from app.core.embedder import EMBEDDING_DIM, FaceEmbedder
from app.core.preprocess import preprocess_frame
from app.core.liveness import LivenessTracker
from app.core.matcher import MatchResult
from app.core.temporal import DEFAULT_WINDOW_SIZE, FaceTracker, TemporalBuffer
from app.infrastructure.camera import CameraCapture
from app.infrastructure.db import Database
from app.services.attendance import AttendanceService
from app.services.recognition import RecognitionService

logger = logging.getLogger(__name__)

# Màu vẽ (BGR)
KNOWN_COLOR = (0, 200, 0)      # xanh — người đã đăng ký
UNKNOWN_COLOR = (0, 60, 255)   # đỏ — người lạ
SPOOF_COLOR = (0, 165, 255)    # cam — nghi ngờ giả mạo (Bước 17)
TEXT_COLOR = (255, 255, 255)

# ── Confidence colors (BGR) ──
#   ≥ 0.80  xanh lá  — tin cậy cao
#   0.60–0.80  vàng  — tin cậy trung bình
#   0.40–0.60  cam   — tin cậy thấp
#   < 0.40  đỏ       — gần ngưỡng, có thể nhầm
# LƯU Ý: các màu KNOWN/UNKNOWN/SPOOF/CONF phía trên hiện chỉ còn được dùng
# bởi photo_view.py và script test (scripts/step_23_gui_test.py). Overlay
# nhận dạng THỜI GIAN THỰC dưới đây theo ảnh tham chiếu: đồng màu TRẮNG.
CONF_HIGH = (94, 208, 69)     # #45D05A — xanh lá (tin cậy cao ≥0.80)
CONF_MED = (0, 191, 255)      # #FFBF00 — vàng (0.60–0.80)
CONF_LOW = (0, 140, 255)      # #FF8C00 — cam (0.40–0.60)
CONF_VERY_LOW = (0, 69, 255)  # #FF4500 — đỏ (< 0.40)


# ── Overlay nhận dạng thời gian thực (kiểu ảnh tham chiếu) ─────────
# Khung + nhãn chip đồng màu TRẮNG cho MỌI trạng thái (người quen / người
# lạ / giả mạo / bị che) — trạng thái thể hiện bằng CHỮ trên chip (Unicode).
BOX_COLOR = (255, 255, 255)   # BGR — khung viền khuôn mặt
CHIP_BG = (255, 255, 255)     # BGR — nền chip nhãn
CHIP_TEXT = (18, 18, 18)      # BGR — chữ đậm trên nền trắng
CHIP_FONT_PX = 22             # cỡ chữ trên chip (px)


def _confidence_color(similarity: float) -> tuple[int, int, int]:
    """Trả về màu BGR theo mức similarity (dùng cho photo_view + script test)."""
    if similarity >= 0.80:
        return CONF_HIGH
    if similarity >= 0.60:
        return CONF_MED
    if similarity >= 0.40:
        return CONF_LOW
    return CONF_VERY_LOW


# ── Font Unicode (tiếng Việt có dấu) — cv2.putText KHÔNG render được dấu ──
_FONT_CACHE: dict[int, ImageFont.FreeTypeFont] = {}

# Cache ảnh chip ĐÃ RENDER (text → mảng BGR): nền chip TRẮNG ĐẶC nên
# bitmap giống hệt bất kể nền video — render PIL đúng MỘT lần rồi CHÉP
# trực tiếp vào frame. Trước đây mỗi frame preview (15/s × mỗi mặt) đều
# chạy vòng cvtColor → PIL → vẽ → cvtColor — tốn trên máy yếu.
_CHIP_CACHE: dict[str, np.ndarray] = {}
_CHIP_CACHE_MAX = 64  # tên + điểm đổi liên tục → dọn khi đầy


def _pil_font(size_px: int) -> ImageFont.FreeTypeFont:
    """Font TrueType hỗ trợ tiếng Việt, cache theo cỡ chữ.

    Thứ tự ưu tiên font Windows: Segoe UI → Arial → Tahoma.
    Fallback: font mặc định của Pillow (không dấu nhưng không crash).
    """
    if size_px not in _FONT_CACHE:
        font = None
        for name in ("segoeui.ttf", "arial.ttf", "tahoma.ttf"):
            try:
                font = ImageFont.truetype(name, size_px)
                break
            except OSError:
                continue
        _FONT_CACHE[size_px] = font if font is not None else ImageFont.load_default()
    return _FONT_CACHE[size_px]

# Debounce ghi sự kiện: tối đa 1 sự kiện / người / 5 giây.
# (Trước đây là 0.0 = ghi MỖI frame xử lý → ~7 lần ghi DB + chụp JPEG mỗi
# giây → giật camera và phình DB. 5s vẫn đủ cho điểm danh: check-in/check-out
# lệch tối đa 5s, cửa sổ "đang làm" là 300s.)
EVENT_DEBOUNCE_SECONDS = 5.0

# Khoảng làm mới ĐỊNH DANH (giây): ArcFace (nhúng 512 chiều — phần NẶNG
# nhất của đường ống, ~vài trăm ms trên CPU) chỉ chạy lại mỗi khoảng này
# cho một track đã nhận diện rõ. Giữa 2 lần làm mới: chỉ chạy SCRFD rẻ
# (~vài chục ms) để khung bám vị trí mặt + hiển thị tên từ cache.
EMBED_REFRESH_SECONDS = 2.0

# Khoảng kiểm tra lại NGƯỜI LẠ (giây): track chưa khớp ai cũng KHÔNG nhúng
# ArcFace mỗi chu kỳ — chỉ thỉnh thoảng (1s/lần) so khớp lại để phát hiện
# đăng ký mới. Giữa 2 lần kiểm tra: hiển thị "Người lạ" từ cache (rẻ).
UNKNOWN_RECHECK_SECONDS = 1.0

# Số khuôn mặt TỐI ĐA được nhận diện mỗi khung hình (sắp theo diện tích
# giảm dần — mặt to gần camera ưu tiên trước). Chặn trên để SCRFD phát
# hiện hàng chục mặt nhỏ ở xa không làm CPU quá tải và sinh so khớp sai.
MAX_FACES_PER_FRAME = 5

# ── Bước 25: Cảnh báo "Ảnh mờ" realtime dùng ngưỡng THÍCH NGHI ──────
# Quality gate kiểm tra mỗi ~2s; cảnh báo mờ khi độ nét hiện tại
# < max(40, 50% độ nét CAO NHẤT đã thấy). Máy nét (max ~104) giữ ngưỡng
# ~52 như trước (lọc rung); webcam nền mềm (max ~45) KHÔNG bị cảnh báo
# oan khi người dùng đã giữ yên — vì ~43 là độ nét nền của chính nó.
QUALITY_BLUR_WARN_FLOOR = 40.0
QUALITY_BLUR_WARN_RATIO = 0.50

# ── Khung nhận diện chạy MƯỢT theo mặt (motion extrapolation) ──────
# Vị trí khung chỉ được đo mỗi chu kỳ nhận diện (mặt di chuyển giữa 2 lần
# đo → khung "nhảy bước"). Giải pháp: ước lượng vận tốc (px/s) của TỪNG
# track từ 2 lần đo liên tiếp, rồi giữa 2 lần đo DỰ ĐOÁN vị trí khung
# theo công thức tiệm cận lead = τ·(1 − e^(−dt/τ)):
#   - dt nhỏ → lead ≈ dt (theo sát mặt như linear);
#   - dt lớn → lead bão hòa ở τ (mặt DỪNG đột ngột thì khung không trôi)
# — mượt tuyệt đối, chi phí ~0 (vài phép nhân mỗi frame preview).
MOTION_TAU = 0.30          # giây — khoảng dẫn tối đa của dự đoán
MOTION_MAX_AHEAD = 0.60    # giây — kẹp dt (luồng nhận diện bị trễ lâu)
MOTION_SMOOTH_ALPHA = 0.5  # EMA làm mượt vận tốc giữa 2 lần đo


class CameraWorker(QObject):
    """Vòng lặp đọc frame + preview — chạy trong QThread.

    CameraCapture được tạo TRONG run() (đúng luồng worker) vì
    VideoCapture của OpenCV không an toàn đa luồng.

    Nhận diện (model chậm) chạy trong threading.Thread phụ với cơ chế
    "frame mới nhất thắng" — vòng lặp preview KHÔNG bao giờ bị chặn.
    """

    frame_ready = Signal(object)          # numpy.ndarray BGR (đã vẽ)
    started = Signal(int, int)            # độ phân giải thực tế (w, h)
    error = Signal(str)
    fps_updated = Signal(float)
    # Có người lạ trong khung gần nhất không (bật/tắt nút [Đăng ký ngay])
    unknown_face = Signal(bool)
    # Người lạ vừa xuất hiện: (ảnh crop, embedding, chất lượng) — làm mẫu 1
    unknown_crop = Signal(object, object, float)
    # Sự kiện nhận diện (debounce) — gửi về UI để lưu CSDL (tránh đụng DB ở worker)
    recognition_hit = Signal(str, float, object)      # (person_id, similarity, crop)
    recognition_unknown = Signal(object)              # crop người lạ
    # Nghi ngờ giả mạo (Bước 17): nhận diện được NGƯỜI ĐÃ BIẾT nhưng không
    # thấy dấu hiệu sống (chớp mắt) → (person_id, similarity, crop) để UI
    # ghi sự kiện "Giả mạo" vào lịch sử (audit trail).
    spoof_suspected = Signal(str, float, object)
    # Mặt bị che NHIỀU (Bước 18): nhận diện được NGƯỜI ĐÃ BIẾT nhưng
    # occlusion cao (không đủ tin cậy) → (person_id, similarity, crop) để
    # UI ghi sự kiện "Mặt bị che" vào lịch sử (audit trail).
    occlusion_blocked = Signal(str, float, object)

    def __init__(
        self,
        camera_index: int,
        width: int,
        height: int,
        detector: FaceDetector | None = None,
        embedder: FaceEmbedder | None = None,
        service: RecognitionService | None = None,
        threshold: float = 0.40,
        anti_spoofing_enabled: bool = True,
        smoothing_window: int = DEFAULT_WINDOW_SIZE,
        clahe_enabled: bool = True,
        process_every: int = 2,
        preview_every: int = 1,
    ) -> None:
        super().__init__()
        self._camera_index = camera_index
        self._width = width
        self._height = height
        self._detector = detector
        self._embedder = embedder
        self._service = service
        self._threshold = threshold
        self._anti_spoofing_enabled = anti_spoofing_enabled
        self._smoothing_window = smoothing_window
        self._clahe_enabled = clahe_enabled
        self._process_every = max(1, int(process_every))
        self._preview_every = max(1, int(preview_every))
        # Tracker liveness RIÊNG CHO TỪNG NGƯỜI (2 người trong khung hình
        # mỗi người có chu kỳ chớp mắt của riêng mình — không gộp chung).
        self._liveness: dict[str, LivenessTracker] = {}
        # Temporal smoothing + face tracking (Bước 19)
        self._tracker = FaceTracker()
        self._buffers: dict[int, TemporalBuffer] = {}  # track_id → buffer
        self._running = False
        # Nhận diện KHÔNG chặn vòng lặp đọc camera: frame "mới nhất thắng"
        # (latest-wins) — preview luôn mượt, không khựng mỗi lần nhận diện.
        self._pending_lock = threading.Lock()
        self._pending: np.ndarray | None = None
        self._frame_available = threading.Event()
        self._shutdown = threading.Event()
        self._rec_thread: threading.Thread | None = None
        # Vận tốc từng track: {track_id: (cx, cy, vx, vy, ts)} — px/giây.
        # Dùng nội suy vị trí khung giữa 2 lần nhận diện (xem MOTION_TAU).
        self._track_motion: dict[int, tuple[float, float, float, float, float]] = {}
        # Định danh cache từng track: {track_id: {person_id, similarity, ts}}
        # — giúp BỎ ArcFace (nặng) khi đã biết ai, chỉ chạy SCRFD (rẻ).
        self._track_state: dict[int, dict] = {}
        # Thời điểm chu kỳ nhận diện ĐẦY ĐỦ (có ArcFace) gần nhất — các
        # chu kỳ giữa chỉ chạy SCRFD nhẹ (xem UNKNOWN_RECHECK_SECONDS).
        self._last_full_ts: float = 0.0
        self._was_unknown = False
        self._last_unknown_seed: tuple | None = None
        # Cache overlay lần nhận diện GẦN NHẤT (khung + chip + vùng mờ):
        # giữa 2 chu kỳ nhận diện preview vẫn chạy — vẽ lại cache để khung
        # hiển thị LIÊN TỤC thay vì nháy theo nhịp process_every.
        self._last_overlays: list[tuple] = []
        self._last_event_at: dict[str, float] = {}
        self._last_quality_check: float = 0.0  # Bước 22: Quality Gate
        # Bước 25: độ nét (Laplacian) cao nhất đã thấy — nền của webcam để
        # cảnh báo "Ảnh mờ" không kêu oan trên webcam nền mềm
        self._best_sharpness = 0.0

    # ---------------------------------------------------------
    # Vòng lặp chính
    # ---------------------------------------------------------
    @Slot()
    def run(self) -> None:
        capture = CameraCapture(self._camera_index, self._width, self._height)
        try:
            if not capture.open():
                self.error.emit("Không mở được webcam — kiểm tra camera / chỉ số camera")
                return
            self.started.emit(capture.actual_width, capture.actual_height)

            self._running = True
            frame_count = 0
            frame_index = 0
            t0 = time.time()
            self._best_sharpness = 0.0  # reset mỗi phiên mở camera

            # Nhận diện chạy ở LUỒNG PHỤ — tách khỏi vòng lặp preview
            self._shutdown.clear()
            with self._pending_lock:
                self._pending = None
            self._frame_available.clear()
            if self._detector is not None:
                self._rec_thread = threading.Thread(
                    target=self._recognize_loop,
                    name="face-recognition",
                    daemon=True,  # không chặn thoát app khi camera bị treo
                )
                self._rec_thread.start()

            while self._running:
                frame = capture.read()
                if frame is None:
                    self.error.emit("Không đọc được khung hình từ camera")
                    break
                # Nhận diện: chỉ ĐƯA frame vào ô chờ theo chu kỳ nặng rồi đi
                # tiếp — KHÔNG chờ model (SCRFD+ArcFace ~200-300ms) như trước,
                # nên video preview không còn khựng mỗi lần nhận diện.
                if self._detector is not None and frame_index % self._process_every == 0:
                    self._offer_frame(frame)
                # Vẽ LẠI overlay lần gần nhất để khung + tên hiện LIÊN TỤC
                # trên MỌI frame preview (không frame trần → không nháy).
                if frame_index % self._preview_every == 0:
                    self._redraw_overlays(frame)
                    self.frame_ready.emit(frame)

                # Cập nhật FPS mỗi ~1 giây
                frame_count += 1
                frame_index += 1
                elapsed = time.time() - t0
                if elapsed >= 1.0:
                    self.fps_updated.emit(frame_count / elapsed)
                    frame_count = 0
                    t0 = time.time()
        finally:
            # Dừng luồng nhận diện TRƯỚC khi đóng camera (nó đang dùng ảnh
            # crop từ frame — không được để chạy đua khi capture đã release).
            self._running = False
            self._shutdown.set()
            self._frame_available.set()
            rec = self._rec_thread
            if rec is not None and rec.is_alive():
                rec.join(timeout=1.0)
            self._rec_thread = None
            # LUÔN đóng camera — kể cả khi có lỗi bất ngờ trong vòng lặp
            # (nếu không, MSMF giữ webcam và thread chết không sạch).
            capture.release()

    def stop(self) -> None:
        """Yêu cầu dừng vòng lặp (gọi từ luồng UI)."""
        self._running = False
        # Đánh thức luồng nhận diện đang chờ frame để nó thấy cờ dừng NGAY
        # (không phải đợi hết timeout chờ frame). Tracker/buffers do LUỒNG
        # NHẬN DIỆN tự dọn khi thoát — luồng UI không đụng vào (an toàn).
        self._shutdown.set()
        self._frame_available.set()

    # ---------------------------------------------------------
    # Luồng nhận diện riêng (frame "mới nhất thắng")
    # ---------------------------------------------------------
    def _offer_frame(self, frame: np.ndarray) -> None:
        """Đưa frame vào ô chờ cho luồng nhận diện (MỚI NHẤT THẮNG).

        Nếu luồng nhận diện còn đang xử lý frame TRƯỚ và chưa lấy, frame
        mới GHI ĐÈ frame cũ — ô chờ luôn tối đa 1 frame nên độ trễ nhận
        diện KHÔNG tăng dần theo thời gian (khác hàng đợi FIFO).

        Đưa BẢN SAO vào: vòng lặp preview vẽ overlay trực tiếp lên frame
        gốc — tránh 2 luồng cùng ghi/đọc 1 vùng nhớ ảnh.
        """
        with self._pending_lock:
            self._pending = frame.copy()
        self._frame_available.set()

    def _recognize_loop(self) -> None:
        """Vòng lặp nhận diện — chạy trong luồng riêng, tách khỏi preview.

        Lấy frame MỚI NHẤT trong ô chờ → phát hiện + so khớp (chậm) → vẽ
        overlay + phát tín hiệu. Xử lý xong mà camera đã đưa frame mới hơn
        thì frame này được bỏ — preview KHÔNG bao giờ bị chặn bởi model.
        """
        while not self._shutdown.is_set() and self._running:
            frame: np.ndarray | None = None
            with self._pending_lock:
                if self._pending is not None:
                    frame = self._pending
                    self._pending = None
            if frame is None:
                # Chưa có frame mới — đợi được đánh thức (không quay vòng ăn CPU)
                self._frame_available.wait(timeout=0.2)
                self._frame_available.clear()
                continue
            try:
                self._process_frame(frame)
            except Exception:  # noqa: BLE001 — 1 frame lỗi không được giết luồng
                logger.exception("Lỗi xử lý nhận diện — bỏ qua frame này")
        # Dọn trạng thái theo dõi khi luồng kết thúc (an toàn — luồng nhận
        # diện là nơi DUY NHẤT đụng tracker/buffers/motion lúc này).
        self._tracker.reset()
        self._buffers.clear()
        self._track_motion.clear()
        self._track_state.clear()
        self._last_overlays = []

    # ---------------------------------------------------------
    # Xử lý khung hình: phát hiện → so khớp → vẽ
    # ---------------------------------------------------------
    def _process_frame(self, frame: np.ndarray) -> bool:
        """Nhận diện khuôn mặt trong frame; trả True nếu có người lạ.

        CHẾ ĐỘ ĐIỂM DANH NHIỀU NGƯỜI (cả nhóm cùng quét):
          - Nhận diện TẤT CẢ các mặt trong khung (tối đa MAX_FACES_PER_FRAME
            mặt to nhất) — mỗi người được ghi điểm danh riêng theo chế độ
            VÀO CA / RA CA đang chọn.
          - Bỏ qua liveness/occlusion khi quét (các tính năng này vẫn chạy
            đầy đủ khi ĐĂNG KÝ qua EnrollmentDialog dùng chung detector
            nhưng có vòng lặp riêng). Temporal smoothing VẪN CHẠY để chặn
            ghi oan điểm danh khi so khớp sai 1 khung hình.
        """
        # CLAHE chỉ bật khi người dùng tự bật trong Cài đặt (mặc định TẮT)
        detect_frame = preprocess_frame(frame) if self._clahe_enabled else frame
        # CHU KỲ ĐẦY ĐỦ (SCRFD + ArcFace) CHỈ khi CÓ track cần làm mới định
        # danh: người quen hết hạn làm mới (2s), người lạ hết hạn kiểm tra
        # (1s), hoặc chưa định danh được ai. Cảnh chỉ toàn người quen còn
        # tươi → chạy SCRFD NHẸ SUỐT — ArcFace r50 (phần NẶNG nhất của đường
        # ống trên máy yếu) gần như không chạy. Không service (không so khớp
        # được) → luôn ĐẦY ĐỦ để mẫu [Đăng ký ngay] vẫn có embedding.
        detect_ts = time.time()
        if self._service is None or not self._track_state:
            full_cycle = True
        else:
            full_cycle = any(
                (detect_ts - s["ts"])
                >= (
                    EMBED_REFRESH_SECONDS
                    if s["person_id"] is not None
                    else UNKNOWN_RECHECK_SECONDS
                )
                for s in self._track_state.values()
            )
        try:
            faces = self._detector.detect(detect_frame, with_recognition=full_cycle)
        except Exception:  # noqa: BLE001 — lỗi GPU không được giết luồng camera
            logger.exception("Lỗi phát hiện khuôn mặt — tạm tắt nhận diện")
            self._detector = None
            self._last_overlays = []  # không vẽ lại khung cũ khi nhận diện chết
            return False

        # NHIỀU MẶT: nhận diện TẤT CẢ các mặt trong khung — sắp theo DIỆN
        # TÍCH GIẢM DẦN (mặt to = gần camera = ưu tiên trước) và chặn trên
        # MAX_FACES_PER_FRAME để CPU không quá tải khi SCRFD phát hiện
        # hàng chục mặt nhỏ ở xa (nhúng mặt nhỏ chất lượng thấp, dễ so
        # khớp sai, chỉ tốn thêm CPU vô ích).
        if len(faces) > 1:
            faces = sorted(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                reverse=True,
            )
        if len(faces) > MAX_FACES_PER_FRAME:
            faces = faces[:MAX_FACES_PER_FRAME]

        # Cache overlay TẠO MỚI cho mỗi lần nhận diện — build trong biến CỤC
        # BỘ rồi gán MỘT LẦN ở cuối: luồng preview đọc _last_overlays liên
        # tục, gán một lần tránh nó đọc phải danh sách đang build dở.
        # Mỗi phần tử: (track_id, x1, y1, x2, y2, chip_text, vùng_mờ|None)
        # — track_id để preview NỘI SUY vị trí (khung chạy mượt theo mặt).
        overlays: list[tuple] = []
        current_ids: set[int] = set()

        # Bước 19: FaceTracker — gán track_id cho mỗi bbox
        bboxes = [face.bbox for face in faces]
        track_map = self._tracker.update(bboxes)  # {bbox_idx: track_id}

        has_unknown = False
        # Mẫu [Đăng ký ngay]: giữ người lạ LỚN NHẤT trong khung (gần camera
        # nhất, ảnh nét nhất) — nhiều người lạ cùng khung không ghi đè nhau.
        largest_unknown_seed: tuple | None = None
        for i, face in enumerate(faces):
            x1, y1, x2, y2 = face.bbox.astype(int)
            track_id = track_map.get(i, -1)

            # Cập nhật vận tốc track (px/giây) từ 2 lần đo liên tiếp —
            # dữ liệu cho nội suy vị trí khung mượt giữa 2 chu kỳ nhận diện.
            if track_id >= 0:
                current_ids.add(track_id)
                mcx = (x1 + x2) / 2.0
                mcy = (y1 + y2) / 2.0
                prev = self._track_motion.get(track_id)
                if prev is not None:
                    pdt = detect_ts - prev[4]
                    if pdt > 0.01:
                        nvx = (mcx - prev[0]) / pdt
                        nvy = (mcy - prev[1]) / pdt
                        # EMA: vận tốc đo được có nhiễu (bbox rung ±vài px)
                        vx = MOTION_SMOOTH_ALPHA * nvx + (1 - MOTION_SMOOTH_ALPHA) * prev[2]
                        vy = MOTION_SMOOTH_ALPHA * nvy + (1 - MOTION_SMOOTH_ALPHA) * prev[3]
                    else:  # 2 lần đo trùng thời điểm — giữ vận tốc cũ
                        vx, vy = prev[2], prev[3]
                else:
                    vx = vy = 0.0  # track mới — chưa đủ 2 điểm để đo
                self._track_motion[track_id] = (mcx, mcy, vx, vy, detect_ts)

            # ── TỐI ƯU NHẸ (quan trọng nhất): ArcFace CHỈ chạy KHI CẦN ──
            # SCRFD (phát hiện) rẻ — chạy MỖI chu kỳ để khung bám vị trí.
            # ArcFace (nhúng + so khớp) ĐẮT — bỏ hẳn khi định danh track
            # đã rõ và còn tươi: hiển thị tên từ cache, CPU gần như rảnh.
            state = self._track_state.get(track_id) if track_id >= 0 else None
            embedding = None
            ttl = (
                EMBED_REFRESH_SECONDS
                if state is not None and state["person_id"] is not None
                else UNKNOWN_RECHECK_SECONDS
            )
            # Chu kỳ NHẸ không có embedding → LUÔN dùng cache (kể cả quá hạn
            # TTL — chu kỳ ĐẦY ĐỦ kế tiếp sẽ làm mới; tránh chớp "Người lạ").
            use_cache = state is not None and (
                not full_cycle or (detect_ts - state["ts"]) < ttl
            )
            if use_cache:
                if state["person_id"] is not None:
                    # Đã biết ai → KHÔNG nhúng, KHÔNG so khớp (tiết kiệm lớn)
                    stable_result = MatchResult(
                        person_id=state["person_id"],
                        similarity=state["similarity"],
                    )
                    display_result = stable_result
                else:
                    # Người lạ + mới kiểm tra gần đây → hiển thị lại từ cache,
                    # rơi xuống nhánh người lạ bên dưới (không nhúng ArcFace)
                    stable_result = None  # tránh rò rỉ kết quả của mặt trước
                    display_result = None
            else:
                # Trích embedding (dùng chung cho so khớp + mẫu đăng ký người lạ)
                embedding = self._embedder.embed_face(face) if self._embedder else None

                # So khớp thô (trước khi smoothing)
                raw_result: MatchResult | None = None
                if self._service is not None and embedding is not None:
                    raw_result = self._service.match(embedding, self._threshold)

                # Bước 19: Temporal smoothing — thêm raw_result vào buffer.
                # TÁCH RIÊNG hiển thị và ghi điểm danh:
                #   - Hiển thị: dùng kết quả thô NGAY LẬP TỨC — tên hiện LIÊN TỤC,
                #     không nháy "Người lạ" 1-2 giây đầu khi buffer chưa đủ phiếu.
                #   - Ghi điểm danh: chỉ khi buffer đã bỏ phiếu ổn định — khớp sai
                #     1 khung hình không ghi oan điểm danh.
                stable_result = None
                if self._smoothing_window > 0 and track_id >= 0:
                    if track_id not in self._buffers:
                        self._buffers[track_id] = TemporalBuffer(self._smoothing_window)
                    stable_id = self._buffers[track_id].add(raw_result)
                    if stable_id is not None and raw_result is not None:
                        stable_result = MatchResult(
                            person_id=stable_id,
                            similarity=raw_result.similarity,
                        )
                else:
                    stable_result = raw_result  # không smoothing — kết quả thô là kết quả chốt

                # Ưu tiên tên đã bỏ phiếu ổn định; chưa có thì hiển thị kết quả thô
                display_result = stable_result or raw_result
                # Cập nhật cache định danh: khớp được → người quen TTL 2s;
                # trượt → người lạ, sau 1s sẽ chạy chu kỳ ĐẦY ĐỦ kiểm tra lại
                # (bắt đăng ký mới / người mới bước vào khung).
                if track_id >= 0:
                    chosen = stable_result or raw_result
                    if chosen is not None:
                        self._track_state[track_id] = {
                            "person_id": chosen.person_id,
                            "similarity": chosen.similarity,
                            "ts": detect_ts,
                        }
                    else:
                        # Vẫn người lạ → LƯU trạng thái (KHÔNG pop — pop làm
                        # chu kỳ sau nhúng lại ngay, mất công dụng throttling)
                        self._track_state[track_id] = {
                            "person_id": None,
                            "similarity": None,
                            "ts": detect_ts,
                        }
            if display_result is not None:
                # CHẾ ĐỘ NHẸ: bỏ kiểm tra occlusion + liveness (chớp mắt) khi
                # quét điểm danh — tiết kiệm CPU đáng kể (occlusion tính Sobel
                # cả khung mỗi lần). Tên vẫn vẽ + sự kiện vẫn ghi bình thường.
                chip_text = self._draw_known(frame, face, display_result)
                # Cache khung + nhãn — preview giữa 2 chu kỳ vẽ lại cho liên tục
                overlays.append((track_id, x1, y1, x2, y2, chip_text, None))
                if stable_result is not None:
                    self._maybe_emit_hit(frame, face, stable_result)
            else:
                # NGƯỜI LẠ (hoặc chưa có dữ liệu đăng ký)
                has_unknown = True
                seed = self._mark_unknown(frame, face, embedding)
                # Cache khung + nhãn + VÙNG MỜ (preview giữa 2 chu kỳ vẽ lại
                # cả vùng mờ — không lộ mặt người lạ nét giữa 2 lần nhận diện)
                bx1, by1 = max(0, x1), max(0, y1)
                bx2, by2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                overlays.append(
                    (track_id, x1, y1, x2, y2, "Người lạ", (bx1, by1, bx2, by2))
                )
                if seed is not None and (
                    largest_unknown_seed is None
                    or (x2 - x1) * (y2 - y1) > largest_unknown_seed[0]
                ):
                    largest_unknown_seed = ((x2 - x1) * (y2 - y1), *seed)

        # Gán cache overlay MỘT LẦN (an toàn đa luồng — xem chú thích trên)
        self._last_overlays = overlays
        # Dọn vận tốc + định danh của track đã biến mất (tránh phình dict)
        self._track_motion = {
            tid: m for tid, m in self._track_motion.items() if tid in current_ids
        }
        self._track_state = {
            tid: s for tid, s in self._track_state.items() if tid in current_ids
        }

        # [Đăng ký ngay] dùng mẫu người lạ LỚN NHẤT vừa thấy trong khung.
        # Khung không còn người lạ → GIỮ seed cũ (nút vẫn dùng được ngay).
        if largest_unknown_seed is not None:
            self._last_unknown_seed = largest_unknown_seed[1:]

        # Phát tín hiệu khi trạng thái người lạ THAY ĐỔI (tránh spam mỗi frame)
        if has_unknown != self._was_unknown:
            self._was_unknown = has_unknown
            self.unknown_face.emit(has_unknown)
            if has_unknown and self._last_unknown_seed is not None:
                self.unknown_crop.emit(*self._last_unknown_seed)

        # (Đã bỏ Quality Gate realtime khi quét — tính Laplacian/brightness
        # mỗi 2s cũng tốn CPU; cảnh báo này vẫn có ở luồng đăng ký.)

        return has_unknown

    # ---------------------------------------------------------
    # Vẽ lại overlay giữa 2 chu kỳ nhận diện (khung liên tục)
    # ---------------------------------------------------------
    def _redraw_overlays(self, frame: np.ndarray) -> None:
        """Vẽ lại khung + chip + vùng mờ của lần nhận diện GẦN NHẤT.

        Nhận diện nặng chạy ở LUỒNG PHỤ mỗi ``process_every`` khung; trong
        khoảng đó cache overlay giúp khung + tên hiện trên MỌI frame preview
        (không frame trần → không nháy).

        KHUNG CHẠY MƯỢT THEO MẶT: vị trí vẽ được NỘI SUY theo vận tốc track
        (xem MOTION_TAU) — mặt di chuyển giữa 2 lần đo thì khung vẫn bám
        theo liền mạch thay vì "nhảy bước" mỗi chu kỳ nhận diện.
        """
        now = time.time()
        for tid, x1, y1, x2, y2, text, blur in self._last_overlays:
            # Nội suy vị trí: dịch (dx, dy) theo vận tốc đã đo của track
            dx = dy = 0.0
            motion = self._track_motion.get(tid) if tid >= 0 else None
            if motion is not None:
                _, _, vx, vy, ts = motion
                dt = now - ts
                if dt > 0:
                    dt = min(dt, MOTION_MAX_AHEAD)
                    # Dẫn tiệm cận: ≈dt khi mới đo, bão hòa ở MOTION_TAU
                    lead = MOTION_TAU * (1.0 - math.exp(-dt / MOTION_TAU))
                    dx = vx * lead
                    dy = vy * lead
                    # Không cho khung trôi quá 1/2 kích thước (dự đoán sai)
                    half_w, half_h = (x2 - x1) * 0.5, (y2 - y1) * 0.5
                    dx = max(-half_w, min(half_w, dx))
                    dy = max(-half_h, min(half_h, dy))
            px1, py1 = x1 + dx, y1 + dy
            px2, py2 = x2 + dx, y2 + dy
            if blur is not None:
                bx1, by1, bx2, by2 = blur
                bx1, by1, bx2, by2 = (
                    bx1 + dx, by1 + dy, bx2 + dx, by2 + dy,
                )
                # Kẹp vào biên frame — chỉ số âm khiến numpy slice WRAP
                # (frame[-5:...] = lấy từ cuối ảnh → hỏng vùng mờ)
                fh, fw = frame.shape[:2]
                bx1, by1 = max(0, int(bx1)), max(0, int(by1))
                bx2, by2 = min(fw, int(bx2)), min(fh, int(by2))
                if bx2 > bx1 and by2 > by1:
                    region = frame[by1:by2, bx1:bx2]
                    if region.size:
                        # Tọa độ đã ép int trước phép // | 1 (float → lỗi)
                        k = max(25, min(bx2 - bx1, by2 - by1) // 4 | 1)
                        if k % 2 == 0:
                            k += 1  # kernel Gaussian phải là số LẺ
                        frame[by1:by2, bx1:bx2] = cv2.GaussianBlur(region, (k, k), 0)
            self._draw_box(frame, px1, py1, px2, py2)
            if text:
                self._draw_chip(frame, text, px1, py1)

    # ---------------------------------------------------------
    # Che khuất / occlusion (Bước 18)
    # ---------------------------------------------------------
    def _mark_occluded(
        self, frame: np.ndarray, face, result: MatchResult
    ) -> None:
        """Mặt bị che NHIỀU: không nhận diện sai — khung trắng + chip "Mặt bị che".

        KHÔNG làm mờ (đây là chính người dùng, không phải người lạ). Vẫn GHI
        sự kiện "Mặt bị che: <tên>" vào lịch sử (Bước 18) — audit trail
        biết là AI bị chặn vì che, kèm snapshot (cắt BẢN SAO trước khi vẽ
        khung để ảnh còn sạch).
        """
        x1, y1, x2, y2 = face.bbox.astype(int)
        crop = self._crop_face(frame, face)
        self._draw_box(frame, x1, y1, x2, y2)
        self._draw_chip(frame, "Mặt bị che", x1, y1)

        # Ghi sự kiện bị che (debounce theo từng người — tránh tràn DB)
        if crop is not None and self._should_save(f"occluded:{result.person_id}"):
            self.occlusion_blocked.emit(
                result.person_id, result.similarity, crop
            )

    # ---------------------------------------------------------
    # Chống giả mạo (Bước 17)
    # ---------------------------------------------------------
    def _is_live(self, person_id: str, face) -> bool:
        """Cập nhật tracker liveness của người này với frame hiện tại.

        Tracker theo dõi chu kỳ chớp mắt (mở→nhắm→mở) trong cửa sổ trượt
        ~10 giây + KHOAN DUNG 6 giây khi mới xuất hiện (người thật vừa
        vào khung chưa kịp chớp không bị bắt vội). Ảnh tĩnh giơ trước
        camera không bao giờ chớp → hết khoan dung, trả False.
        """
        tracker = self._liveness.setdefault(person_id, LivenessTracker())
        tracker.update(face)
        return tracker.is_live()

    def _mark_spoof(self, frame: np.ndarray, face, result: MatchResult) -> None:
        """Đánh dấu khuôn mặt NGHI NGỜ GIẢ MẠO: mờ + khung trắng + chip "Giả mạo".

        Người dùng đã chốt chính sách "Chặn": nhận diện bị chặn (không
        hiển thị tên) và ghi sự kiện "Giả mạo" vào lịch sử (debounce).
        """
        x1, y1, x2, y2 = face.bbox.astype(int)
        # Cắt BẢN SAO TRƯỚC khi làm mờ để snapshot còn nét (giống người lạ)
        crop = self._crop_face(frame, face)

        self._blur_face(frame, face)
        self._draw_box(frame, x1, y1, x2, y2)
        self._draw_chip(frame, "Giả mạo", x1, y1)

        # Ghi sự kiện giả mạo (debounce theo từng người — tránh tràn DB)
        if crop is not None and self._should_save(f"spoof:{result.person_id}"):
            self.spoof_suspected.emit(result.person_id, result.similarity, crop)

    # ---------------------------------------------------------
    # Vẽ & ghi sự kiện
    # ---------------------------------------------------------
    def _draw_known(
        self,
        frame: np.ndarray,
        face,
        result: MatchResult,
        warn: bool = False,
    ) -> str:
        """Vẽ người đã biết: khung trắng + chip nhãn "Tên điểm" (kiểu tham chiếu).

        ``warn=True`` (mặt hơi bị che) → thêm "!" sau tên.
        Trả về nhãn chip — caller cache lại để preview giữa 2 chu kỳ nhận
        diện vẽ lại (khung liên tục).
        """
        x1, y1, x2, y2 = face.bbox.astype(int)
        self._draw_box(frame, x1, y1, x2, y2)
        name = self._label_for(result.person_id)
        score = f" {result.similarity:.2f}" if result.similarity is not None else ""
        text = f"{name}!{score}" if warn else f"{name}{score}"
        self._draw_chip(frame, text, x1, y1)
        return text

    def _label_for(self, person_id: str) -> str:
        """Tên người theo id (qua service — worker không đụng DB)."""
        if self._service is not None:
            return self._service.label_of(person_id)
        return "?"

    def _mark_unknown(
        self, frame: np.ndarray, face, embedding
    ) -> tuple | None:
        """Làm mờ vùng mặt (riêng tư) + khung trắng + chip 'Người lạ'.

        Trả về (crop, embedding, det_score) làm mẫu [Đăng ký ngay]; None
        khi không trích được embedding. KHÔNG tự ghi vào
        ``_last_unknown_seed`` — nhiều người lạ cùng khung, caller (vòng
        lặp _process_frame) tự chọn mẫu LỚN NHẤT.
        """
        x1, y1, x2, y2 = face.bbox.astype(int)
        # Cắt BẢN SAO TRƯỚC khi làm mờ để snapshot còn nét
        crop = self._crop_face(frame, face) if embedding is not None else None

        self._blur_face(frame, face)
        self._draw_box(frame, x1, y1, x2, y2)
        # (CHẾ ĐỘ NHẸ: bỏ occlusion_score — Sobel cả khung mỗi lần quá tốn
        # CPU cho một nhãn phụ. Luồng đăng ký vẫn kiểm tra đầy đủ.)
        self._draw_chip(frame, "Người lạ", x1, y1)

        # Sự kiện người lạ (debounce) — dùng đúng crop NÉT đã cắt
        if crop is not None and self._should_save("unknown"):
            self.recognition_unknown.emit(crop)

        if embedding is not None:
            return (crop, embedding, float(face.det_score))
        return None

    def _maybe_emit_hit(self, frame: np.ndarray, face, result: MatchResult) -> None:
        """Ghi sự kiện nhận diện người đã biết (debounce 5s/người)."""
        if not self._should_save(result.person_id):
            return
        crop = self._crop_face(frame, face)
        self.recognition_hit.emit(result.person_id, result.similarity, crop)

    def _should_save(self, key: str) -> bool:
        now = time.time()
        if now - self._last_event_at.get(key, 0.0) < EVENT_DEBOUNCE_SECONDS:
            return False
        self._last_event_at[key] = now
        return True

    # ---------------------------------------------------------
    # Vẽ trợ giúp
    # ---------------------------------------------------------
    @staticmethod
    def _draw_label(
        frame: np.ndarray, text: str, x: int, y: int, color: tuple,
        scale: float = 0.8,
    ) -> None:
        """Vẽ nhãn nền màu đặc + chữ Unicode (tiếng Việt KHÔNG bị mất dấu).

        Thay cv2.putText (font HERSHEY không vẽ được dấu tiếng Việt) bằng
        PIL + font hệ thống (Segoe UI...). Giữ nguyên hình dạng cũ: nền màu
        theo ``color``, chữ trắng, vị trí ``y`` gần đáy nhãn như trước.
        Ký tự '⚠' được đổi thành '!' trước khi vẽ — đa số font hệ thống
        (Segoe UI/Arial/Tahoma) không có glyph này nên giữ nguyên sẽ vẽ
        thành ô trống/rác.
        """
        text = text.replace("⚠", "!")
        # HERSHEY cũ vẽ nét dày (thickness 2) → dùng cỡ lớn hơn một chút
        # để chữ nét đều, dễ đọc (20px ≈ nét của HERSHEY scale 0.8)
        font_px = max(14, int(scale * 25 + 0.5))
        font = _pil_font(font_px)
        asc, desc = font.getmetrics()
        th = asc + desc
        tw = float(font.getlength(text))
        baseline = int(y) - 2
        # Khung nền giống cv2 cũ: trên baseline - th - 8, dưới + 2 (chừa
        # thêm descender để chữ như g/y/j không bị cắt)
        top = baseline - th - 8
        bottom = baseline + desc + 2
        left = int(x)
        right = left + int(tw) + 10
        h, w = frame.shape[:2]
        if bottom <= top or left >= w or top >= h or right <= 0:
            return
        left, top = max(0, left), max(0, top)
        right, bottom = min(w, right), min(h, bottom)
        if right <= left or bottom <= top:
            return
        sub = frame[top:bottom, left:right].copy()
        pil = Image.fromarray(cv2.cvtColor(sub, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        draw.rectangle(
            [0, 0, right - left - 1, bottom - top - 1],
            fill=(color[2], color[1], color[0]),
        )
        # anchor "ls" = trái-baseline: (3, baseline - top) trong tọa độ sub
        draw.text(
            (3, baseline - top), text, font=font,
            fill=(TEXT_COLOR[2], TEXT_COLOR[1], TEXT_COLOR[0]),
            anchor="ls",
        )
        frame[top:bottom, left:right] = cv2.cvtColor(
            np.asarray(pil), cv2.COLOR_RGB2BGR
        )

    @staticmethod
    def _draw_box(
        frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        color: tuple = BOX_COLOR,
        thickness: int = 2,
    ) -> None:
        """Khung viền ĐỦ 4 cạnh, mảnh — giống ảnh tham chiếu (thay 4 góc L).

        Màu đồng nhất (BOX_COLOR trắng) cho MỌI trạng thái; trạng thái
        được thể hiện bằng chữ trên chip nhãn (_draw_chip).
        """
        cv2.rectangle(
            frame, (int(x1), int(y1)), (int(x2), int(y2)),
            color, thickness, cv2.LINE_AA,
        )

    @staticmethod
    def _draw_chip(
        frame: np.ndarray,
        text: str,
        x1: int,
        y1: int,
    ) -> None:
        """Chip nhãn TRẮNG ngay trên mép trái khung — giống ảnh tham chiếu.

        Toàn bộ chip (nền trắng + chữ đậm) vẽ TRONG MỘT LẦN bằng PIL —
        sửa lỗi nhãn cũ bị mất/cắt ký tự (vd "Long" hiện thành chữ rác):
          - bề rộng chip đo bằng CHÍNH font dùng để vẽ → luôn đủ chỗ;
          - chữ đặt cách lề trái ``pad_x - left`` (left = left-bearing của
            ký tự đầu) → ký tự đầu tiên luôn hiển thị đầy đủ;
          - chip bị kẹp trong biên frame (không vẽ tràn / cắt ngoài mép).
        Không đủ chỗ phía trên khung → dời xuống ngay dưới mép trên khung.
        """
        chip = _CHIP_CACHE.get(text)
        if chip is None:
            # Render MỘT LẦN bằng PIL (chữ tiếng Việt nét, không mất dấu)
            font = _pil_font(CHIP_FONT_PX)
            left, top, right, bottom = font.getbbox(text)
            tw, th = right - left, bottom - top
            if tw <= 0 or th <= 0:
                return
            pad_x, pad_y = 8, 4
            pil = Image.new(
                "RGB", (tw + pad_x * 2, th + pad_y * 2),
                (CHIP_BG[2], CHIP_BG[1], CHIP_BG[0]),
            )
            ImageDraw.Draw(pil).text(
                (pad_x - left, pad_y - top), text, font=font,
                fill=(CHIP_TEXT[2], CHIP_TEXT[1], CHIP_TEXT[0]),
            )
            # RGB → BGR, bản sao liền mạch để chép thẳng vào frame
            chip = np.asarray(pil)[:, :, ::-1].copy()
            if len(_CHIP_CACHE) >= _CHIP_CACHE_MAX:
                _CHIP_CACHE.clear()
            _CHIP_CACHE[text] = chip
        # Chép trực tiếp — chip là nền trắng ĐẶC, không cần trộn alpha
        fh, fw = frame.shape[:2]
        ch_, cw_ = chip.shape[:2]
        if ch_ >= fh or cw_ >= fw:
            return  # frame quá nhỏ so với chữ — bỏ qua
        cx = max(1, min(int(x1), fw - cw_ - 1))
        cy = int(y1) - ch_ - 4
        if cy < 1:
            cy = min(int(y1) + 4, max(1, fh - ch_ - 1))
        frame[cy:cy + ch_, cx:cx + cw_] = chip

    @staticmethod
    def _blur_face(frame: np.ndarray, face) -> None:
        """Làm mờ Gaussian MẠNH vùng khuôn mặt (riêng tư người lạ).

        Kernel tỉ lệ theo kích thước mặt (tối thiểu 25px) — mờ rõ rệt,
        không nhìn được ai là ai (mục đích làm mờ riêng tư).
        """
        x1, y1, x2, y2 = face.bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        w, h = x2 - x1, y2 - y1
        if w < 4 or h < 4:
            return
        region = frame[y1:y2, x1:x2].copy()
        k = max(25, min(w, h) // 4 | 1)
        if k % 2 == 0:
            k += 1  # kernel Gaussian phải là số LẺ
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(region, (k, k), 0)

    @staticmethod
    def _crop_face(frame: np.ndarray, face) -> np.ndarray:
        """Cắt ảnh khuôn mặt (có viền) — bản SAO để gửi qua tín hiệu an toàn."""
        x1, y1, x2, y2 = face.bbox.astype(int)
        h, w = frame.shape[:2]
        pad_x = int((x2 - x1) * 0.25)
        pad_y = int((y2 - y1) * 0.25)
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        return frame[y1:y2, x1:x2].copy()

    # ---------------------------------------------------------
    # Bước 22: Quality Gate — cảnh báo realtime
    # ---------------------------------------------------------
    QUALITY_GATE_INTERVAL = 2.0  # giây giữa 2 lần check (tránh spam)

    def _draw_quality_gate(self, frame: np.ndarray, faces: list) -> None:
        """Kiểm tra chất lượng khung hình + vẽ cảnh báo realtime.

        Chỉ check mỗi ~2 giây để không spam.
        """
        now = time.time()
        if now - self._last_quality_check < self.QUALITY_GATE_INTERVAL:
            return
        self._last_quality_check = now

        warnings: list[str] = []

        # 1. Không thấy mặt nào
        if not faces:
            warnings.append("⚠ Không thấy khuôn mặt — đưa mặt vào khung")
        else:
            face = faces[0]  # check mặt lớn nhất
            # 2. Mặt quá nhỏ (xa camera)
            x1, y1, x2, y2 = face.bbox.astype(int)
            w, h = x2 - x1, y2 - y1
            if w < 100 or h < 100:
                warnings.append("⚠ Khuôn mặt quá xa — tiến lại gần camera")
            # 3. Ảnh mờ — ngưỡng THÍCH NGHI theo độ nét nền của webcam
            #    (Bước 25): webcam nền mềm (Laplacian ~40-50 dù giữ yên)
            #    không bị cảnh báo oan, máy nét vẫn lọc rung như trước.
            sharp = face_metrics._sharpness(frame, face)
            if sharp > self._best_sharpness:
                self._best_sharpness = sharp
            blur_warn = max(
                QUALITY_BLUR_WARN_FLOOR,
                self._best_sharpness * QUALITY_BLUR_WARN_RATIO,
            )
            if sharp < blur_warn:
                warnings.append("⚠ Ảnh mờ — giữ yên đầu, tránh rung")

        # 4. Ánh sáng yếu/tối
        brightness = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
        if brightness < 50:
            warnings.append("⚠ Ánh sáng yếu — bật thêm đèn")
        elif brightness > 230:
            warnings.append("⚠ Quá sáng — tránh ngược sáng")

        # Vẽ cảnh báo (nếu có)
        if warnings:
            h, w = frame.shape[:2]
            y_start = 30
            for msg in warnings[:2]:  # tối đa 2 cảnh báo
                self._draw_label(frame, msg, 10, y_start, SPOOF_COLOR)
                y_start += 35


class CameraView(QWidget):
    """Khu vực xem webcam: video preview + điều khiển."""

    # Người dùng nhấn [ Đăng ký ngay ]: (ảnh crop, embedding, chất lượng)
    enroll_requested = Signal(object, object, float)

    def __init__(
        self,
        config: Config,
        db: Database | None = None,
        detector: FaceDetector | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._db = db
        self._detector = detector
        self._embedder: FaceEmbedder | None = None
        self._service: RecognitionService | None = None
        self._thread: QThread | None = None
        self._worker: CameraWorker | None = None
        self._last_unknown_seed: tuple | None = None
        # Hiển thị MỖI frame preview nhận được — preview mượt tối đa
        # (worker giờ phát MỖI khung vì nhận diện đã tách sang luồng phụ).
        self._frame_display_every = 1
        self._frame_display_counter = 0
        # Điểm danh tự động (tính năng mới): người đã biết được nhận diện
        # qua webcam → ghi check-in/out (chỉ khi có CSDL).
        self._attendance: AttendanceService | None = (
            AttendanceService(db) if db is not None else None
        )
        if db is not None:
            self._service = RecognitionService(db)
        # Thời điểm thông báo điểm danh gần nhất THEO TÊN NGƯỜI (anti-spam
        # dialog — nhận diện lặp mỗi vài giây không được hiện hộp thoại liên tục)
        self._last_notify_at: dict[str, float] = {}  # (giữ chỗ cho tương lai)
        self._build_ui()

    # ---------------------------------------------------------
    # Giao diện
    # ---------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # --- Vùng video (nền tối cố định — video cần nền tối ở cả 2 theme) ---
        self._video_label = QLabel("Webcam chưa mở")
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Minimum NHỎ (không phải 640x480): ảnh vẫn scale giữ tỉ lệ khi hiển thị,
        # nhưng layout không bị ép rộng → cửa sổ co nhỏ được (kéo góc hoạt động
        # trên màn hình nhỏ / scale 125%).
        self._video_label.setMinimumSize(400, 300)
        self._video_label.setObjectName("videoLabel")
        self._video_label.setStyleSheet("font-size: 18px;")
        layout.addWidget(self._video_label, stretch=1)

        # --- Panel điều khiển bên phải ---
        panel = QVBoxLayout()
        panel.setSpacing(8)

        ctl_title = QLabel("ĐIỀU KHIỂN")
        ctl_title.setObjectName("sectionTitle")
        panel.addWidget(ctl_title)

        self._status_label = QLabel("Trạng thái: chưa mở")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("font-size: 14px;")
        panel.addWidget(self._status_label)

        # Nút bấm cámara: nhãn ĐỔI THEO CHẾ ĐỘ đang chọn (Vào ca / Ra ca)
        # để người dùng luôn thấy mình sắp thực hiện tác vụ gì.
        self._start_btn = QPushButton("▶ Bắt đầu quét — VÀO CA")
        self._start_btn.setObjectName("primaryBtn")
        self._start_btn.clicked.connect(self.start_camera)
        panel.addWidget(self._start_btn)

        self._stop_btn = QPushButton("⏸ Dừng quét")
        self._stop_btn.clicked.connect(self.stop_camera)
        self._stop_btn.setEnabled(False)
        panel.addWidget(self._stop_btn)

        # ── Chế độ điểm danh: VÀO CA / RA CA (face ID xác nhận) ──
        # Người dùng bấm chọn chế độ TRƯỚC; người bước vào khung được quét
        # face ID và tự ghi theo chế độ đang chọn (mượt — không bấm lại từng người).
        mode_title = QLabel("CHẾ ĐỘ ĐIỂM DANH")
        mode_title.setObjectName("sectionTitle")
        panel.addWidget(mode_title)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(6)
        self._checkin_btn = QPushButton("🟢 VÀO CA")
        self._checkin_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._checkin_btn.setCheckable(True)
        self._checkin_btn.setChecked(True)  # mặc định: vào ca
        self._checkin_btn.setToolTip(
            "Quét face ID tự ghi GIỜ VÀO (lần đầu trong ngày)."
            " Không đụng giờ ra đã xác nhận."
        )
        self._checkin_btn.clicked.connect(self._on_mode_checkin)
        mode_row.addWidget(self._checkin_btn, stretch=1)

        self._checkout_btn = QPushButton("🔴 RA CA")
        self._checkout_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._checkout_btn.setCheckable(True)
        self._checkout_btn.setToolTip(
            "Quét face ID tự ghi GIỜ RA = thời điểm nhận diện (cố định)."
        )
        self._checkout_btn.clicked.connect(self._on_mode_checkout)
        mode_row.addWidget(self._checkout_btn, stretch=1)
        panel.addLayout(mode_row)

        self._mode_hint = QLabel(
            "Đang chế độ VÀO CA — người quét được face ID sẽ ghi giờ vào."
        )
        self._mode_hint.setWordWrap(True)
        self._mode_hint.setStyleSheet("font-size: 12px; color: #888;")
        panel.addWidget(self._mode_hint)

        # ── Ghi chú cho lần quét (v4) ──
        # Ghi chú lưu RIÊNG theo tính năng: VÀO CA → note_in, RA CA → note_out.
        self._note_edit = QLineEdit()
        self._note_edit.setPlaceholderText("📝 Ghi chú (tùy chọn)...")
        self._note_edit.setClearButtonEnabled(True)
        self._note_edit.setToolTip(
            "Ghi chú lưu kèm lần quét theo chế độ đang chọn:\n"
            "VÀO CA → ghi chú GIỜ VÀO · RA CA → ghi chú GIỜ RA.\n"
            "Ví dụ: 'vào trễ 15 phút', 'ra sớm họ hàng'."
        )
        self._note_edit.textChanged.connect(self._update_note_hint)
        panel.addWidget(self._note_edit)

        self._note_hint = QLabel("")
        self._note_hint.setWordWrap(True)
        self._note_hint.setStyleSheet("font-size: 11px; color: #888;")
        panel.addWidget(self._note_hint)
        self._update_note_hint()  # khởi tạo nhãn theo chế độ mặc định (VÀO CA)

        # --- HÀNH VI CỦA CÁC NÚT (logic) ---
        # - Bấm VÀO CA / RA CA: đổi chế độ + đổi luôn nhãn nút "Bắt đầu quét".
        # - Bấm "Bắt đầu quét": mở webcam; trong lúc quét nút bị vô hiệu.
        # - "Dừng quét": đóng webcam; nhãn nút trở lại "Bắt đầu quét".
        # - Menu "Vào ca"/"Ra ca" cũng gọi 2 hàm _on_mode_* này → nhãn luôn khớp.

        # Nút [ Đăng ký ngay ] — chỉ hiện khi có người lạ trong khung hình
        self._enroll_now_btn = QPushButton("⚡ Đăng ký ngay")
        self._enroll_now_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._enroll_now_btn.setObjectName("accentBtn")
        self._enroll_now_btn.setToolTip(
            "Đăng ký người lạ đang ở trong khung hình (ảnh hiện tại làm mẫu 1)"
        )
        self._enroll_now_btn.setStyleSheet("border-radius: 6px; padding: 10px; font-size: 14px;")
        self._enroll_now_btn.clicked.connect(self._on_enroll_now)
        self._enroll_now_btn.hide()  # ẩn cho tới khi có người lạ
        panel.addWidget(self._enroll_now_btn)

        panel.addStretch(1)

        layout.addLayout(panel)

    # ---------------------------------------------------------
    # Điều khiển camera
    # ---------------------------------------------------------
    def start_camera(self) -> None:
        """Mở webcam và chạy vòng lặp nhận diện trong QThread."""
        if self._thread is not None and self._thread.isRunning():
            return  # đã chạy rồi (hoặc thread cũ còn sống do camera treo — KHÔNG mở song song)
        if self._thread is not None:
            # Thread cũ đã kết thúc nhưng chưa được dọn (stop_camera giữ lại
            # khi chờ dừng quá hạn) → dọn sạch trước khi tạo thread mới
            self._thread = None
            self._worker = None

        # Nạp model phát hiện khuôn mặt lần đầu (mất ~1-2 giây — chỉ 1 lần)
        if self._detector is None:
            self._status_label.setText("Đang nạp model nhận diện...")
            QApplication.processEvents()  # vẽ lại label ngay (UI bị chặn lúc nạp model)
            self._detector = self._load_detector()
            if self._detector is None:
                return  # giữ nguyên thông báo lỗi model, không mở camera
        # Embedder tái sử dụng FaceAnalysis của detector — không nạp model lần 2
        if self._embedder is None:
            self._embedder = FaceEmbedder(self._detector)

        # Nạp danh sách embedding đã đăng ký vào bộ so khớp (LUỒNG UI — trước khi mở)
        if self._service is not None:
            try:
                self._service.reload()
            except Exception:  # noqa: BLE001
                logger.exception("Lỗi nạp embedding cho bộ so khớp")
                self._service = None
        self._last_unknown_seed = None

        self._thread = QThread(self)
        self._worker = CameraWorker(
            self._config.camera_index,
            self._config.camera_width,
            self._config.camera_height,
            detector=self._detector,
            embedder=self._embedder,
            service=self._service,
            threshold=self._config.recognition_threshold,
            anti_spoofing_enabled=self._config.anti_spoofing_enabled,
            smoothing_window=self._config.smoothing_window,
            clahe_enabled=self._config.clahe_enabled,
            process_every=6,   # ~5 lần nhận diện/giây (luồng phụ) — khung bám nhanh hơn
            preview_every=2,   # ~15fps preview — mượt mà nửa tải UI (máy yếu vẫn khỏe)
        )
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.started.connect(self._on_camera_started)
        self._worker.fps_updated.connect(self._on_fps)
        self._worker.error.connect(self._on_error)
        self._worker.unknown_face.connect(self._on_unknown_face)
        self._worker.unknown_crop.connect(self._on_unknown_crop)
        self._worker.recognition_hit.connect(self._on_recognition_hit)
        self._worker.recognition_unknown.connect(self._on_recognition_unknown)
        self._worker.spoof_suspected.connect(self._on_spoof_suspected)
        self._worker.occlusion_blocked.connect(self._on_occlusion_blocked)

        self._thread.start()
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._status_label.setText("Đang mở webcam...")
        logger.info(
            "Bắt đầu camera (index=%d, chế độ=%s)",
            self._config.camera_index,
            "VÀO CA" if self._is_checkin_mode else "RA CA",
        )

    @property
    def detector(self) -> FaceDetector | None:
        """FaceDetector đã nạp (nếu có) — cho EnrollmentDialog tái sử dụng model."""
        return self._detector

    def _load_detector(self) -> FaceDetector | None:
        """Tạo FaceDetector; trả None nếu model thiếu/lỗi (camera vẫn chạy, không nhận diện).

        Nếu thiếu model buffalo_l → tự tải về (ensure_model_available, cần internet)
        thay vì bắt người dùng chạy Bước 0.4 thủ công.
        """
        try:
            if not (MODELS_ROOT / "models" / "buffalo_l").exists():
                self._status_label.setText("⏳ Đang tải model buffalo_l (~300MB, lần đầu)...")
                ensure_model_available(MODELS_ROOT)
            # det_size 256 (thay vì 320 mặc định): SCRFD nhẹ hơn ~36% mỗi lần
            # detect (chi phí tỉ lệ ~bình phương cạnh) — máy yếu vẫn mượt.
            # Điểm danh webcam mặt chiếm ≥1/4 khung vẫn phát hiện tốt;
            # ảnh đăng ký (mặt lớn) không bị ảnh hưởng đáng kể.
            detector = FaceDetector(MODELS_ROOT, det_size=(256, 256))
            logger.info("Đã nạp model phát hiện khuôn mặt")
            return detector
        except Exception as exc:  # noqa: BLE001 — hiển thị lỗi cho người dùng, không crash
            logger.exception("Không nạp được model: %s", exc)
            self._status_label.setText(f"⚠ Không nạp được model phát hiện khuôn mặt: {exc}")
            return None

    def stop_camera(self) -> None:
        """Dừng vòng lặp, đóng camera và giải phóng thread.

        QUAN TRỌNG (an toàn đa luồng): nếu worker KHÔNG kịp dừng trong
        thời gian chờ (ví dụ MSMF treo khi mở camera lỗi), KHÔNG được xóa
        thread — giữ tham chiếu cho nó tự kết thúc. Xóa sớm khi luồng còn
        chạy sẽ gây hỏng heap (cv2 không an toàn đa luồng) nếu mở lại ngay:
        2 luồng camera chạy song song → crash.
        """
        if self._thread is None and self._worker is None:
            return  # chưa chạy — không làm gì (tránh log/thay đổi UI thừa)
        if self._worker is not None:
            self._worker.stop()
        if self._thread is not None:
            self._thread.quit()
            stopped = self._thread.wait(1500)  # chờ tối đa 1.5s cho worker dừng
            if stopped:
                # Thread đã kết thúc sạch → dọn tham chiếu
                self._thread = None
                self._worker = None
            # else: thread vẫn chạy (camera treo) → GIỮ tham chiếu;
            #       start_camera kiểm tra isRunning() để không mở song song.

        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._status_label.setText("Trạng thái: đã dừng")
        # Khôi phục nhãn nút quét theo chế độ đang chọn (sau khi dừng)
        self._start_btn.setText(
            "▶ Bắt đầu quét — " + ("VÀO CA" if self._is_checkin_mode else "RA CA")
        )
        self._enroll_now_btn.hide()
        logger.info("Dừng camera")

    # ---------------------------------------------------------
    # Xử lý tín hiệu từ worker
    # ---------------------------------------------------------
    @Slot(object)
    def _on_frame(self, frame: np.ndarray) -> None:
        """Nhận frame BGR (đã vẽ) → chuyển RGB → hiển thị lên QLabel."""
        # Giảm tải UI preview: chỉ render lại UI theo chu kỳ để tránh lag
        # khi camera worker đọc frame quá nhanh trên máy yếu.
        self._frame_display_counter += 1
        if self._frame_display_counter % self._frame_display_every != 0:
            return

        # QImage bọc TRỰC TIẾP buffer BGR (Format_BGR888) — bỏ một lần
        # cvtColor CẢ KHUNG mỗi preview (15 lần/giây × 640×480 pixel).
        h, w, ch = frame.shape
        qimg = QImage(frame.data, w, h, ch * w, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(qimg)
        target_size = self._video_label.size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            return
        self._video_label.setPixmap(
            pixmap.scaled(
                target_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
        )

    @Slot(int, int)
    def _on_camera_started(self, width: int, height: int) -> None:
        mode = "VÀO CA" if self._is_checkin_mode else "RA CA"
        self._status_label.setText(
            f"Đang quét {mode}: CAM {self._config.camera_index} · {width}×{height}"
        )

    @Slot(float)
    def _on_fps(self, fps: float) -> None:
        # Giữ phần trước ' · FPS' (có thể vừa bị _on_camera_started/nhận diện
        # ghi đè) và nối FPS mới — tránh mất tiền tố chế độ.
        base = self._status_label.text().split(" · FPS")[0]
        self._status_label.setText(f"{base} · FPS: {fps:.1f}")

    @Slot(bool)
    def _on_unknown_face(self, has_unknown: bool) -> None:
        """Bật/tắt nút [ Đăng ký ngay ] theo sự hiện diện của người lạ."""
        self._enroll_now_btn.setVisible(has_unknown)

    @Slot(object, object, float)
    def _on_unknown_crop(self, crop, embedding, quality: float) -> None:
        """Lưu ảnh + embedding người lạ vừa xuất hiện (mẫu 1 cho đăng ký ngay)."""
        self._last_unknown_seed = (crop, embedding, quality)

    # ---------------------------------------------------------
    # Chế độ điểm danh VÀO CA / RA CA
    # ---------------------------------------------------------
    @Slot()
    def _on_mode_checkin(self) -> None:
        """Chọn chế độ VÀO CA: nút RA CA tự nhả + nhãn nút quét đổi theo."""
        self._checkin_btn.setChecked(True)
        self._checkout_btn.setChecked(False)
        self._mode_hint.setText(
            "Đang chế độ VÀO CA — người quét được face ID sẽ ghi giờ vào."
        )
        self._start_btn.setText("▶ Bắt đầu quét — VÀO CA")
        self._update_note_hint()

    @Slot()
    def _on_mode_checkout(self) -> None:
        """Chọn chế độ RA CA: nút VÀO CA tự nhả + nhãn nút quét đổi theo."""
        self._checkout_btn.setChecked(True)
        self._checkin_btn.setChecked(False)
        self._mode_hint.setText(
            "Đang chế độ RA CA — giờ ra ghi theo thời điểm nhận diện (cố định)."
        )
        self._start_btn.setText("▶ Bắt đầu quét — RA CA")
        self._update_note_hint()

    def _update_note_hint(self) -> None:
        """Gợi ý nơi ghi chú sẽ lưu, đổi theo chế độ đang chọn."""
        if not hasattr(self, "_note_hint"):
            return  # gọi từ __init__ trước khi tạo đủ widget
        if self._is_checkin_mode:
            self._note_hint.setText("Ghi chú sẽ lưu vào GIỜ VÀO (note_in)")
        else:
            self._note_hint.setText("Ghi chú sẽ lưu vào GIỜ RA (note_out)")

    def _current_note(self) -> str:
        """Nội dung ghi chú đang nhập (rỗng = không có ghi chú)."""
        return self._note_edit.text().strip()

    @property
    def _is_checkin_mode(self) -> bool:
        """Đang ở chế độ VÀO CA không? (False = RA CA)"""
        return self._checkin_btn.isChecked()

    def force_checkin_mode(self) -> None:
        """Ép chế độ VÀO CA (gọi từ MainWindow khi vào menu Điểm danh VÀO CA)."""
        self._on_mode_checkin()

    def force_checkout_mode(self) -> None:
        """Ép chế độ RA CA (gọi từ MainWindow khi vào menu Điểm danh RA CA)."""
        self._on_mode_checkout()

    @Slot(str, float, object)
    def _on_recognition_hit(self, person_id: str, similarity: float, crop) -> None:
        """Face ID XÁC NHẬN người đã biết → ghi sự kiện + ĐIỂM DANH NGẦM.

        VÀO CA: phiên đang mở → giữ nguyên giờ vào; không có phiên mở →
        tạo phiên mới. RA CA: đóng phiên chưa ra gần nhất; không có phiên
        mở → service TỰ MỞ PHIÊN BÙ (không mất công) nhưng UI KHÔNG lưu
        sự kiện (không phải quét vào hợp lệ).

        KHÔNG hiện hộp thoại xác nhận — quét mặt là nhận, kết quả chỉ cập
        nhật dòng trạng thái nhỏ (quét nhiều người liền mạch, không ngắt).
        Chỉ lỗi DB mới hiện hộp thoại đỏ (cần người đọc, không tự tắt).
        """
        # Tên người qua service (CameraView không có _label_for — đó của worker).
        # Service None (lỗi nạp embedding) → dùng id ngắn làm tên tạm.
        if self._service is not None:
            name = self._service.label_of(person_id)
        else:
            name = person_id[:8]
        mode_txt = "VÀO CA" if self._is_checkin_mode else "RA CA"
        mode_key = "checkin" if self._is_checkin_mode else "checkout"
        note = self._current_note()

        res: dict | None = None
        if self._attendance is not None:
            res = (
                self._attendance.record_check_in(person_id, note=note)
                if self._is_checkin_mode
                else self._attendance.record_check_out(person_id, note=note)
            )
            self._notify_attendance(name, mode_txt, res)

        # Quét KHÔNG hợp lệ theo logic vào/ra ca → bỏ qua TOÀN BỘ (không
        # lưu sự kiện, không snapshot):
        #  - blocked: người không tồn tại (đã bị xóa);
        #  - RA CA mà KHÔNG có phiên mở → service tự mở phiên BÙ (giữ công)
        #    nhưng đây KHÔNG phải quét RA hợp lệ → không lưu sự kiện;
        #  - VÀO CA mà phiên đang mở → đã ghi giờ vào rồi, quét lại là trùng
        #    → không lưu sự kiện (chỉ cập nhật last_seen).
        if res is not None and (
            res.get("blocked")
            or res.get("was_not_checked_in")
            or (self._is_checkin_mode and res.get("already_done"))
        ):
            return

        if self._service is not None:
            self._service.save_event(
                person_id=person_id,
                similarity=similarity,
                face_crop=crop,
                mode=mode_key,
            )

    def _notify_attendance(self, name: str, mode_txt: str, res: dict) -> None:
        """Kết quả điểm danh CHỈ cập nhật dòng trạng thái — KHÔNG hộp thoại.

        Quét mặt là nhận, quét người này đến người khác liền mạch không bị
        ngắt bởi popup. Hộp thoại chỉ xuất hiện khi CÓ LỖI DB (đỏ, phải
        bấm OK — vì lỗi cần người xử lý).
        """
        msg = res.get("message", "")
        if res.get("ok"):
            self._status_label.setText(f"✓ {mode_txt}: {name} — {msg}")
        else:
            self._status_label.setText(f"⚠ {mode_txt} LỖI: {name} — {msg}")
            QMessageBox.critical(self, f"{mode_txt} THẤT BẠI", f"{name}\n\n{msg}")

    @Slot(object)
    def _on_recognition_unknown(self, crop) -> None:
        """Người lạ → ghi sự kiện (is_unknown=1) + snapshot."""
        if self._service is not None:
            self._service.save_event(
                person_id=None,
                similarity=None,
                face_crop=crop,
                is_unknown=True,
            )

    @Slot(str, float, object)
    def _on_spoof_suspected(self, person_id: str, similarity: float, crop) -> None:
        """Nghi ngờ giả mạo → ghi sự kiện 'Giả mạo' vào lịch sử (audit trail)."""
        if self._service is not None:
            self._service.save_event(
                person_id=person_id,
                similarity=similarity,
                face_crop=crop,
                is_spoof=True,
            )

    @Slot(str, float, object)
    def _on_occlusion_blocked(self, person_id: str, similarity: float, crop) -> None:
        """Mặt bị che nhiều → ghi sự kiện 'Mặt bị che' vào lịch sử (audit trail)."""
        if self._service is not None:
            self._service.save_event(
                person_id=person_id,
                similarity=similarity,
                face_crop=crop,
                is_occluded=True,
            )

    @Slot(str)
    def _on_error(self, message: str) -> None:
        """Lỗi camera: dừng trước, rồi hiện thông báo (tránh bị ghi đè)."""
        logger.error("Lỗi camera: %s", message)
        self.stop_camera()
        self._status_label.setText(f"⚠ {message}")

    # ---------------------------------------------------------
    # Đăng ký ngay
    # ---------------------------------------------------------
    def _on_enroll_now(self) -> None:
        """Người dùng nhấn [ Đăng ký ngay ] → gửi ảnh + embedding người lạ."""
        if self._last_unknown_seed is not None:
            crop, embedding, quality = self._last_unknown_seed
            self.enroll_requested.emit(crop, embedding, quality)

    # ---------------------------------------------------------
    # Vòng đời widget
    # ---------------------------------------------------------
    def hideEvent(self, event) -> None:  # noqa: N802 (chuẩn Qt)
        """Tự dừng camera khi rời màn hình (tiết kiệm tài nguyên + riêng tư)."""
        self.stop_camera()
        super().hideEvent(event)

    def showEvent(self, event) -> None:  # noqa: N802 (chuẩn Qt)
        """Tự mở webcam khi vào màn hình CameraView."""
        super().showEvent(event)
        if self._thread is None or not self._thread.isRunning():
            self.start_camera()

    def closeEvent(self, event) -> None:  # noqa: N802 (chuẩn Qt)
        self.stop_camera()
        super().closeEvent(event)
