"""LivenessTracker — phát hiện dấu hiệu SỐNG để chống giả mạo (Bước 17, FR-8).

Ý tưởng chống tấn công bằng ẢNH/VIDEO giả (photo attack):
  - Ảnh in / ảnh trên màn hình: khuôn mặt BẤT ĐỘNG — không bao giờ chớp mắt.
  - Người thật: mắt chớp tự nhiên (mỗi vài giây đến ~10 giây) — chu kỳ
    mở → nhắm → mở.

Tracker theo dõi CHU KỲ chớp mắt đầy đủ (không chỉ "mắt đang nhắm" — người
dụi mắt/mí rũ cũng nhắm): chỉ tính là 1 lần chớp khi mắt chuyển MỞ → NHẮM
→ MỞ với thời gian nhắm trong khoảng hợp lý (blink thật ~0.1–0.4s; nhắm
lâu hơn 1s là nhắm có chủ đích, không phải chớp).

Một khuôn mặt được coi là "SỐNG" nếu:
  - có ÍT NHẤT 1 lần chớp mắt trong cửa sổ trượt gần đây (10 giây), HOẶC
  - mới bắt đầu được theo dõi chưa đủ THỜI GIAN KHOAN DUNG (6 giây) — chưa
    đủ căn cứ để kết luận giả mạo (người thật vừa bước vào chưa kịp chớp
    không bị bắt vội). Chỉ sau 6 giây theo dõi liên tục KHÔNG thấy chớp mắt
    nào thì mới báo "nghi giả mạo". Ảnh tĩnh không bao giờ chớp → luôn bị
    chặn sau thời gian khoan dung.

Độ nhạy: dùng ngưỡng EAR RIÊNG (nhắm < 0.28, mở > 0.29) — nhạy hơn ngưỡng
chung của face_metrics (0.25/0.30) để bắt được chớp mắt tự nhiên trên
webcam thường (EAR lúc chớp có thể chỉ hạ xuống ~0.26–0.28).

QUAN TRỌNG (đa luồng): tracker KHÔNG giữ tham chiếu frame — chỉ lưu số
EAR + thời điểm → an toàn dùng trong luồng worker camera.
"""
from __future__ import annotations

import time
from typing import Callable

from app.core import face_metrics

# Cửa sổ trượt (giây): cần ít nhất 1 lần chớp trong khoảng này mới coi là sống.
# 10s (thay vì 3s) — người thật có thể nhìn chăm chú vài giây không chớp;
# chỉ cần chớp 1 lần trong 10 giây là hết nghi ngờ.
LIVENESS_WINDOW_SECONDS = 10.0
# Thời gian nhắm tối đa được tính là 1 lần chớp (blink thật rất nhanh;
# nhắm lâu hơn = nhắm có chủ đích / ảnh có mắt nhắm sẵn → KHÔNG đếm)
MAX_BLINK_DURATION = 1.0
# Khoan dung (giây): theo dõi liên tục chưa đủ lâu này mà chưa thấy chớp
# mắt → VẪN coi là sống (người thật vừa vào khung chưa kịp chớp). Hết
# khoan dung mà vẫn không có lần chớp nào → nghi giả mạo.
GRACE_SECONDS = 6.0
# Người biến mất khỏi khung lâu hơn khoảng này rồi quay lại → coi là quan
# sát MỚI (reset thời gian khoan dung + danh sách chớp cũ).
RESET_SECONDS = 5.0
# Ngưỡng EAR riêng của tracker — NHẠY hơn ngưỡng chung (face_metrics dùng
# 0.25/0.30 cho đăng ký). Nhắm < CLOSED_EAR, mở > OPEN_EAR (chênh 0.01 để
# tránh nhiễu quanh ngưỡng).
CLOSED_EAR = 0.28
OPEN_EAR = 0.29


class LivenessTracker:
    """Theo dõi chu kỳ chớp mắt của MỘT khuôn mặt và báo "còn sống" hay không.

    Cách dùng: gọi ``update(face)`` MỖI FRAME (với face của người đang
    được nhận diện), rồi đọc ``is_live()`` — True nếu có chớp mắt gần đây.
    """

    def __init__(
        self,
        window_seconds: float = LIVENESS_WINDOW_SECONDS,
        now_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        """Khởi tạo tracker.

        ``window_seconds``: cửa sổ trượt (mặc định 3s).
        ``now_fn``: hàm lấy thời gian hiện tại — inject để TEST (không phải
        chờ thật 3 giây); mặc định ``time.monotonic`` (không bị đổi giờ hệ thống).
        """
        self._window = window_seconds
        self._now_fn = now_fn
        self._blink_times: list[float] = []   # thời điểm các lần chớp hoàn chỉnh
        self._eye_state: str = "unknown"      # unknown / open / closed / closed_too_long
        self._closed_since: float | None = None  # thời điểm bắt đầu nhắm (state=closed)
        self._first_seen: float | None = None    # frame ĐẦU TIÊN thấy mặt (khoan dung)
        self._last_update: float | None = None   # frame GẦN NHẤT cập nhật (phát hiện biến mất)

    # ---------------------------------------------------------
    # Cập nhật mỗi frame
    # ---------------------------------------------------------
    def update(self, face) -> None:
        """Cập nhật trạng thái mắt với khuôn mặt mới nhất.

        ``face``: đối tượng Face của InsightFace (đã có landmark_3d_68).
        Gọi mỗi frame; không trả về gì — đọc kết quả bằng ``is_live()``.
        """
        ear = face_metrics.eye_aspect_ratio(face)
        if ear <= 0.0:
            return  # không đo được landmark (thiếu điểm mắt) — bỏ qua frame này
        now = self._now_fn()

        # Người vừa biến mất khỏi khung một lúc (> RESET_SECONDS) rồi quay
        # lại → là quan sát MỚI: reset thời gian khoan dung + chớp cũ (nếu
        # không, người bị báo giả mạo rồi quay lại sẽ bị báo lại ngay)
        if self._last_update is not None \
                and now - self._last_update > RESET_SECONDS:
            self._blink_times = []
            self._eye_state = "unknown"
            self._closed_since = None
            self._first_seen = None
        self._last_update = now
        if self._first_seen is None:
            self._first_seen = now

        if ear < CLOSED_EAR:
            # ---- MẮT ĐANG NHẮM ----
            if self._eye_state == "open":
                # Mở → nhắm: bắt đầu một chu kỳ chớp tiềm năng
                self._eye_state = "closed"
                self._closed_since = now
            elif self._eye_state == "closed":
                # Nhắm quá lâu (không phải chớp) → đánh dấu để khi mở không đếm.
                # _closed_since có thể None khi mắt NHẮM NGAY TỪ FRAME ĐẦU (trạng
                # thái unknown → closed, chưa từng thấy mở) → bỏ qua phép so
                # sánh (tránh TypeError now - None làm chết luồng camera).
                if self._closed_since is not None \
                        and now - self._closed_since > MAX_BLINK_DURATION:
                    self._eye_state = "closed_too_long"
            elif self._eye_state == "unknown":
                # Frame đầu tiên đã thấy mắt nhắm: KHÔNG có bằng chứng mắt từng
                # mở → không đếm khi mở lại (chỉ đếm chu kỳ MỞ→NHẮM→MỞ đầy đủ)
                self._eye_state = "closed"
                self._closed_since = None
        elif ear > OPEN_EAR:
            # ---- MẮT MỞ ----
            if self._eye_state == "closed":
                # Nhắm → mở: hoàn thành 1 lần chớp (nếu chu kỳ hợp lệ)
                self._eye_state = "open"
                if self._closed_since is not None:
                    self._blink_times.append(now)
                self._closed_since = None
            else:
                # open / unknown / closed_too_long → mở: reset trạng thái
                self._eye_state = "open"
                self._closed_since = None
        # ear trong vùng mờ (CLOSED_EAR..OPEN_EAR): không đổi trạng thái

    # ---------------------------------------------------------
    # Kết quả
    # ---------------------------------------------------------
    def is_live(self) -> bool:
        """Còn được coi là NGƯỜI THẬT không? (chớp gần đây HOẶC trong khoan dung)"""
        now = self._now_fn()
        cutoff = now - self._window
        # Cắt bỏ các lần chớp đã cũ (trượt cửa sổ theo thời gian)
        self._blink_times = [t for t in self._blink_times if t >= cutoff]
        if self._blink_times:
            return True
        # Khoan dung: mới theo dõi chưa đủ lâu → chưa đủ căn cứ kết luận
        # giả mạo (người thật vừa bước vào chưa kịp chớp mắt)
        if self._first_seen is not None and now - self._first_seen < GRACE_SECONDS:
            return True
        return False
