"""Đo chỉ số khuôn mặt để hướng dẫn đăng ký (Bước 7 nâng cao).

Từ đối tượng Face của InsightFace (buffalo_l có sẵn landmark_3d_68 và
kps 5 điểm), tính các chỉ số giúp NHẬN BIẾT hành động của người dùng:

  - ``eye_aspect_ratio`` (EAR): mắt mở hay nhắm → phát hiện CHỚP MẮT.
    Mắt mở bình thường EAR ≈ 0.3–0.5; nhắm → EAR < 0.25.
  - ``nose_shift``: vị trí mũi so với 2 mắt (chuẩn hóa) → phát hiện
    QUAY ĐẦU trái/phải. Nhìn thẳng ≈ 0; quay trái → âm; quay phải → dương.
  - ``mouth_ratio``: độ mở miệng (dọc/ngang) → phát hiện MỞ MIỆNG / CƯỜI.
  - ``head_yaw``: góc yaw (độ) từ face.pose — tham chiếu thêm.

Các ngưỡng được hiệu chỉnh bằng dữ liệu THẬT (lena: EAR≈0.47, mouth≈0.44,
nose_shift≈0.18 khi nhìn thẳng; người quay trái ≈ −0.43, quay phải ≈ +0.30).

Bước 18 nâng cao bổ sung phát hiện CHE KHUẤT (occlusion):
  - ``occlusion_score(face, frame)``: 0..1 mức nghi ngờ mặt bị che.
  - ``texture_anomaly(face, frame)``: tín hiệu KẾT CẤU ẢNH — đo edge
    density (Sobel) từng vùng MẮT/MIỆNG/TRÁN so với trung bình cả mặt.
    Bắt được che NHẸ mà det_score/landmark không đổi: ngón tay đè hốc
    mắt, bàn tay che miệng+mũi, tóc rủ trán (đo thực tế trên lena).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

# Ngưỡng phát hiện hành động (đã hiệu chỉnh thực nghiệm)
EAR_BLINK_THRESHOLD = 0.25      # mắt nhắm khi EAR < ngưỡng này
EAR_OPEN_THRESHOLD = 0.24       # mắt rõ ràng MỞ khi EAR > ngưỡng này
# Webcam laptop + đeo kính → landmark bị bóp méo nhẹ → EAR thường
# 0.27-0.30 (thay vì 0.35-0.50 như webcam ngoài). Hạ ngưỡng từ 0.30
# xuống 0.24 để chấp nhận người đeo kính (đo log thực tế: ear≈0.28-0.30).
NOSE_SHIFT_THRESHOLD = 0.25     # |nose_shift| > ngưỡng = đang quay đầu
# Ngưỡng mở miệng/cười. Hạ xuống 0.48 (từ 0.55) để NHẠY hơn — bắt được cả
# nụ cười nhẹ, không chỉ mở miệng to (lena mím môi = 0.44, mở miệng = 0.8).
MOUTH_OPEN_THRESHOLD = 0.48
# |yaw| dưới ngưỡng này coi là nhìn thẳng. LƯU Ý: pose của insightface
# chỉ là ước lượng (lena nhìn thẳng báo yaw≈30°) — để ngưỡng RỘNG, việc
# phân biệt trái/phải dựa chủ yếu vào nose_shift (chính xác hơn).
FRONTAL_YAW_DEG = 40.0
# Nới rộng yaw cho webcam laptop (pose estimator không chính xác —
# nhìn thẳng thật có thể báo yaw 5-10°). 40° đủ rộng để không blocking
# nhưng vẫn lọc được quay đầu rõ ràng.
# |roll| dưới ngưỡng này coi là không nghiêng đầu. Nghiêng đầu làm mẫu
# đăng ký lệch tư thế → nhận diện sau kém; người nhìn thẳng thật sự
# có |roll| < 10° (lena = 1.4°), webcam laptop có thể nghiêng nhẹ 15-20°
# do góc đặt camera → nới lên 30° để chấp nhận.
FRONTAL_ROLL_DEG = 30.0
# Ngưỡng cho bước "Nhìn thẳng" khi đăng ký — mũi phải gần tâm
# Webcam laptop đặt THẤP hơn mắt → nhìn thẳng tự nhiên thì mũi lệch
# lên ~0.25-0.30 → cần nới lên 0.30 để không phải cúi đầu.
STRICT_NOSE_SHIFT = 0.35
# Webcam laptop đặt THẤP hơn mắt → nhìn thẳng tự nhiên thì mũi lệch
# lên ~0.25-0.30. Nới lên 0.35 để người đeo kính không cần cúi đầu.
# Đo log thực tế: ns≈0.01 (rất tốt — kính không affect nose_shift).
# Ngưỡng occlusion (Bước 18 — mặt bị che khuất / không rõ):
#   score < OCCLUSION_WARN        → mặt rõ, nhận diện bình thường
#   WARN ≤ score < OCCLUSION_BLOCK → nghi bị che → tên kèm "⚠"
#   score ≥ OCCLUSION_BLOCK        → bị che nhiều → KHÔNG nhận diện
#                                    (hiện "⚠ Mặt bị che" — tránh nhận diện sai)
#
# Ngưỡng được HẠ XUỐNG sau khi đo thực tế (probe_occl2): công thức cũ
# (0.45/0.70) quá yếu — che 1 bên mặt chỉ đạt 0.43 (< 0.45) nên app vẫn
# nhận diện. Công thức mới thêm tín hiệu EAR từng mắt (mắt bị bóp méo khi
# che → EAR > 0.7, mắt thật không bao giờ vượt 0.6) + đường cong det dốc
# hơn → che 1 bên đạt 0.44-0.51 WARN, che mép/che 1 bên + EAR vọt lên
# BLOCK. Mặt RÕ giữ score rất thấp (0.07-0.14) nên hạ ngưỡng an toàn.
OCCLUSION_WARN = 0.30
OCCLUSION_BLOCK = 0.50
# Ngưỡng EAR "bất thường" (mắt bị bóp méo do che): mắt THẬT mở bình
# thường EAR ≈ 0.3-0.6 (lena 0.50/0.57); khi landmark bị kéo méo vì vật
# cản che 1 bên, EAR vọt lên 1.0-1.26. Ngưỡng rộng (0.70) để KHÔNG phiền
# người đeo kính (kính không làm EAR vọt — chỉ làm landmark lệch nhẹ).
EYE_EAR_ANOMALY_HIGH = 0.70
EYE_EAR_ANOMALY_LOW = 0.15
# Độ lệch EAR giữa 2 mắt coi là bất thường: mắt thật chênh nhau nhỏ
# (lena 0.50/0.57 → lệch 0.07); khi che 1 bên mắt, bên bị che sụp gần 0
# hoặc vọt cao → lệch lớn. Dùng tỉ lệ |L−R| / max(L,R) để chuẩn hóa.
EYE_EAR_ASYMMETRY = 0.25
# ------------------------------------------------------------------
# Ngưỡng KẾT CẤU ẢNH (texture) — phát hiện che nhẹ bằng edge density
# (đo thực tế trên lena — probe_tex2):
#   Vùng MẮT:  bình thường ~1.6 (lông mi, tròng đen, mí); ngón tay đè lên
#              hốc mắt → tụt ~0.8. KÍNH không kích hoạt (làm TĂNG edge).
#   Vùng MIỆNG: bình thường ~1.0 (môi); bàn tay che miệng+mũi → ~0.0.
#   Vùng TRÁN: bình thường ~1.1; tóc rủ/vật che → TĂNG (sợi tóc = edge).
# Các ngưỡng dùng RAMP (0→1) để mượt, không nhảy bậc.
EYE_TEXTURE_LOW_HI = 1.10   # eye ratio dưới mức này bắt đầu nghi bị che
EYE_TEXTURE_RAMP = 0.25     # ... xuống thêm 0.25 → chắc chắn bị che
EYE_TEXTURE_EAR_MIN = 0.15  # chỉ tính khi mắt MỞ (tránh nhầm chớp mắt)
MOUTH_TEXTURE_LOW_HI = 0.60
MOUTH_TEXTURE_RAMP = 0.20
FOREHEAD_TEXTURE_HIGH_LO = 1.45  # forehead ratio trên mức này = có vật trên trán
FOREHEAD_TEXTURE_RAMP = 0.25
# Ngưỡng bbox narrowness — khi vật cản lớn che 1 bên mặt, detector CO
# bbox lại只剩 vùng thấy được → w/h tụt (lena bình thường 0.68, che 45% → 0.47).
# Kích thước bbox co lại = tín hiệu occlusion đáng tin (đo thực tế).
BBOX_NARROWNESS_HI = 0.55  # w/h trên mức này = bình thường
BBOX_NARROWNESS_RAMP = 0.10  # ...dưới thêm 0.10 → chắc chắn bị che

# ── Bước 22: Chất lượng mẫu (SampleQuality) ──────────────────────
# Ngưỡng auto-reject khi đăng ký
MIN_QUALITY_FACE_SCORE = 0.60    # SCRFD confidence tối thiểu
# Blur: webcam laptop 720p đo thực tế chỉ ~104 (Laplacian) — ngưỡng cũ 100
# sát quá → ĐA SỐ frame bị reject → 3 lần reject → bỏ bước → "Không thu
# đủ mẫu" (bug thực tế 2026-08-25). Hạ 60: vẫn lọc frame rung mờ nặng
# nhưng chấp nhận webcam laptop bình thường.
MIN_QUALITY_BLUR = 60.0          # Laplacian variance tối thiểu (mờ < 60)
MIN_QUALITY_BRIGHTNESS = 40      # độ sáng tối thiểu — khớp MIN_BRIGHTNESS đăng ký
MAX_QUALITY_BRIGHTNESS = 220     # độ sáng tối đa — khớp MAX_BRIGHTNESS đăng ký
MIN_QUALITY_OCCLUSION = 0.30     # occlusion score phải < ngưỡng này

# Trọng số tính điểm tổng hợp (0..100)
W_FACE = 0.30   # face_score (detector confidence)
W_BLUR = 0.25   # blur (Laplacian variance, normalized)
W_BRIGHT = 0.20 # brightness (càng gần 128越好)
W_OCC = 0.25    # occlusion (càng thấp越好)


@dataclass
class SampleQuality:
    """Điểm chất lượng mẫu khi đăng ký — Used trong Quality Gate."""

    face_score: float     # SCRFD confidence (0..1)
    blur_score: float     # Laplacian variance (cao = nét)
    brightness: float     # Độ sáng trung bình (0..255)
    occlusion: float      # occlusion_score (0..1, thấp = rõ)
    overall: float = 0.0  # Tổng hợp 0..100

    def compute_overall(self) -> float:
        """Tính điểm tổng hợp 0..100 dựa trên 4 thành phần."""
        # Normalize face_score: 0..1 → 0..100
        f = max(0.0, min(1.0, self.face_score)) * 100.0
        # Normalize blur: 0..500 → 0..100 (Lena nét ≈ 537)
        b = max(0.0, min(1.0, self.blur_score / 500.0)) * 100.0
        # Normalize brightness: 60..200 → 100 (tối ưu ở 128), ngoài → giảm
        if MIN_QUALITY_BRIGHTNESS <= self.brightness <= MAX_QUALITY_BRIGHTNESS:
            # trong khoảng OK → 100 * (1 - distance_from_center / half_range)
            center = (MIN_QUALITY_BRIGHTNESS + MAX_QUALITY_BRIGHTNESS) / 2
            half = (MAX_QUALITY_BRIGHTNESS - MIN_QUALITY_BRIGHTNESS) / 2
            br = 100.0 * (1.0 - abs(self.brightness - center) / half)
        else:
            br = 0.0
        # Normalize occlusion: 0..0.5 → 100..0 (càng thấp越好)
        o = max(0.0, min(1.0, 1.0 - self.occlusion / 0.5)) * 100.0
        self.overall = W_FACE * f + W_BLUR * b + W_BRIGHT * br + W_OCC * o
        return self.overall

    def is_good(self, blur_min: float | None = None) -> bool:
        """Mẫu có đủ tốt để lưu không (auto-reject nếu False).

        ``blur_min``: ngưỡng độ nét tối thiểu — cho phép nơi gọi truyền
        ngưỡng THÍCH NGHI theo webcam (Bước 25, enrollment_dialog) thay vì
        hằng số tuyệt đối MIN_QUALITY_BLUR (60 — đo lệch nặng giữa các
        máy: ~104 máy dev, chỉ ~43 trên webcam người dùng).
        """
        if blur_min is None:
            blur_min = MIN_QUALITY_BLUR
        return (
            self.face_score >= MIN_QUALITY_FACE_SCORE
            and self.blur_score >= blur_min
            and MIN_QUALITY_BRIGHTNESS <= self.brightness <= MAX_QUALITY_BRIGHTNESS
            and self.occlusion < MIN_QUALITY_OCCLUSION
        )

    def label(self) -> str:
        """Nhãn chất lượng hiển thị: Tốt / Đạt / Kém."""
        if self.overall >= 75:
            return "Tốt"
        if self.overall >= 50:
            return "Đạt"
        return "Kém"


def compute_quality(face, frame: np.ndarray) -> SampleQuality:
    """Tính SampleQuality cho 1 khuôn mặt trong frame."""
    q = SampleQuality(
        face_score=float(face.det_score),
        blur_score=_sharpness(frame, face),
        brightness=float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()),
        occlusion=occlusion_score(face, frame),
    )
    q.compute_overall()
    return q


def _sharpness(frame: np.ndarray, face) -> float:
    """Độ nét vùng mặt — Laplacian variance (cao = nét)."""
    x1, y1, x2, y2 = face.bbox.astype(int)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return 0.0
    gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _dist(a, b) -> float:
    return float(math.dist(a[:2], b[:2]))  # chỉ dùng tọa độ x, y


def eye_aspect_ratio(face) -> float:
    """EAR trung bình 2 mắt (0 = nhắm hẳn, ~0.3-0.5 = mở)."""
    left, right = eye_aspect_ratios(face)
    return float((left + right) / 2.0)


def eye_aspect_ratios(face) -> tuple[float, float]:
    """EAR TỪNG MẮT (trái, phải) — 0 = nhắm hẳn, ~0.3-0.6 = mở.

    Dùng 68 điểm landmark chuẩn (dlib convention):
      mắt trái 36-41, mắt phải 42-47.
    Tách từng mắt để phát hiện che 1 bên: khi vật cản che một bên mặt,
    landmark bên đó bị bóp méo → EAR vọt lên 1.0+ (không thể có ở mắt
    thật) hoặc sụp gần 0 — tín hiệu occlusion mạnh nhất (đo thực tế).
    """
    lm = getattr(face, "landmark_3d_68", None)
    if lm is None or len(lm) < 48:
        return (0.0, 0.0)
    # EAR mắt trái (36..41): dọc (37,41) + (38,40) / 2*ngang (36,39)
    left = (_dist(lm[37], lm[41]) + _dist(lm[38], lm[40])) / (2.0 * _dist(lm[36], lm[39]) + 1e-6)
    # EAR mắt phải (42..47): dọc (43,47) + (44,46) / 2*ngang (42,45)
    right = (_dist(lm[43], lm[47]) + _dist(lm[44], lm[46])) / (2.0 * _dist(lm[42], lm[45]) + 1e-6)
    return (float(left), float(right))


def nose_shift(face) -> float:
    """Độ lệch ngang của mũi so với trung điểm 2 mắt (chuẩn hóa).

    > 0: mũi lệch phải (người quay sang trái) · < 0: mũi lệch trái
    (người quay sang phải) · ≈ 0: nhìn thẳng.
    """
    kps = getattr(face, "kps", None)
    if kps is None or len(kps) < 3:
        return 0.0
    # kps: [mắt phải, mắt trái, mũi, khóe miệng phải, khóe miệng trái]
    eye_r, eye_l, nose = kps[0], kps[1], kps[2]
    eye_dist = max(_dist(eye_r, eye_l), 1e-6)
    mid_x = (eye_r[0] + eye_l[0]) / 2.0
    return float((nose[0] - mid_x) / eye_dist)


def mouth_ratio(face) -> float:
    """Độ mở miệng: khoảng cách dọc / ngang (môi trong 60-67).

    Miệng khép ≈ 0.2–0.4; cười/mở miệng > 0.5–0.6.
    """
    lm = getattr(face, "landmark_3d_68", None)
    if lm is None or len(lm) < 68:
        return 0.0
    # Môi trong: 60 = khóe trái, 64 = khóe phải, 63 = giữa trên, 67 = giữa dưới
    vertical = _dist(lm[63], lm[67])
    horizontal = _dist(lm[60], lm[64])
    return float(vertical / (horizontal + 1e-6))


def head_yaw(face) -> float:
    """Góc yaw (độ) — face.pose = (pitch, yaw, roll). Âm = quay trái."""
    pose = getattr(face, "pose", None)
    if pose is None or len(pose) < 3:
        return 0.0
    return float(pose[1])


def head_roll(face) -> float:
    """Góc roll (độ) — face.pose = (pitch, yaw, roll). Dương = nghiêng phải."""
    pose = getattr(face, "pose", None)
    if pose is None or len(pose) < 3:
        return 0.0
    return float(pose[2])


def is_frontal(face) -> bool:
    """Nhìn thẳng: mũi ở giữa + mắt mở + không ngẩng/cúi quá mức."""
    return (
        abs(nose_shift(face)) < NOSE_SHIFT_THRESHOLD
        and eye_aspect_ratio(face) > EAR_BLINK_THRESHOLD
        and abs(head_yaw(face)) < FRONTAL_YAW_DEG
    )


def is_frontal_strict(face) -> bool:
    """Nhìn thẳng CHẶT (dùng khi đăng ký): mũi gần tâm + mắt mở rõ.

    Khắt khe hơn is_frontal để tránh nhận diện quá nhanh — mặt phải
    thực sự đối diện camera (không quay, không NGHIÊNG ĐẦU), mắt mở
    hẳn mới được chụp mẫu. Mẫu nhìn thẳng chất lượng cao là nền tảng
    cho nhận diện sau này.
    """
    return (
        abs(nose_shift(face)) < STRICT_NOSE_SHIFT
        and eye_aspect_ratio(face) > EAR_OPEN_THRESHOLD
        and abs(head_yaw(face)) < FRONTAL_YAW_DEG
        and abs(head_roll(face)) < FRONTAL_ROLL_DEG
    )


def is_blinking(face) -> bool:
    """Đang nhắm/chớp mắt (EAR thấp)."""
    return eye_aspect_ratio(face) < EAR_BLINK_THRESHOLD


def is_turning_left(face) -> bool:
    """Quay đầu sang trái (mũi lệch phải trong khung hình)."""
    return nose_shift(face) < -NOSE_SHIFT_THRESHOLD


def is_turning_right(face) -> bool:
    """Quay đầu sang phải (mũi lệch trái trong khung hình)."""
    return nose_shift(face) > NOSE_SHIFT_THRESHOLD


def is_mouth_open(face) -> bool:
    """Mở miệng / mỉm cười rõ."""
    return mouth_ratio(face) > MOUTH_OPEN_THRESHOLD


# ---------------------------------------------------------
# Che khuất / occlusion (Bước 18)
# ---------------------------------------------------------
def landmarks_outside_ratio(face) -> float:
    """Tỉ lệ landmark 68 điểm nằm NGOÀI khung mặt (0..1).

    Mặt bị cắt/che ở mép (tóc, tay, vật cản) làm nhiều landmark bị đẩy ra
    ngoài khung → tỉ lệ tăng. Bình thường ≈ 0.05-0.10 (sai số nhỏ).
    """
    lm = getattr(face, "landmark_3d_68", None)
    if lm is None or len(lm) < 68:
        return 0.0
    x1, y1, x2, y2 = face.bbox.astype(int)
    pad = 0.05 * (x2 - x1)
    outside = sum(
        1 for p in lm
        if p[0] < x1 - pad or p[0] > x2 + pad or p[1] < y1 - pad or p[1] > y2 + pad
    )
    return float(outside / len(lm))


def chin_nose_ratio(face) -> float:
    """Khoảng cách mũi (lm[30]) → cằm (lm[8]) chia chiều cao khung mặt.

    Bình thường ≈ 0.35-0.45; che phần dưới mặt (khẩu trang/vật cản) làm
    landmark cằm bị kéo lên → tỉ lệ tăng. Tín hiệu phụ của occlusion.
    """
    lm = getattr(face, "landmark_3d_68", None)
    if lm is None or len(lm) < 68:
        return 0.0
    h = face.bbox[3] - face.bbox[1]
    if h <= 0:
        return 0.0
    return float(np.linalg.norm(lm[8][:2] - lm[30][:2]) / h)


def eye_anomaly(face) -> float:
    """0..1 — mức độ bóp méo landmark MẮT (che 1 bên mặt).

    Tín hiệu MẠNH NHẤT khi che (đo thực tế trên lena):
      - Mặt rõ: EAR từng mắt 0.50 / 0.57 (đều trong 0.3-0.6, lệch 0.07).
      - Che 1 bên phải: EAR phải vọt 1.26 (landmark bị kéo méo).
      - Che mép phải: EAR phải 1.01.
    Ngưỡng rộng (0.70 / 0.15 / lệch 0.25) để không phiền người ĐEO KÍNH —
    kính làm landmark lệch nhẹ chứ không vọt EAR ra ngoài khoảng mắt thật.
    """
    ear_l, ear_r = eye_aspect_ratios(face)
    if ear_l <= 0.0 and ear_r <= 0.0:
        return 0.0
    high = max(ear_l, ear_r)
    low = min(ear_l, ear_r)
    high_comp = max(0.0, min(1.0, (high - EYE_EAR_ANOMALY_HIGH) / 0.30))
    low_comp = max(0.0, min(1.0, (EYE_EAR_ANOMALY_LOW - low) / 0.10))
    # Lệch 2 mắt: |L−R| / max(L,R) > ngưỡng → 1 bên mắt bị che/sụp
    asym = abs(ear_l - ear_r) / max(high, 1e-6)
    asym_comp = max(0.0, min(1.0, (asym - EYE_EAR_ASYMMETRY) / 0.25))
    return float(max(high_comp, low_comp, asym_comp))


def _bbox_narrowness(face) -> float:
    """Tỉ lệ w/h của bbox mặt — tín hiệu occlusion khi vật cản lớn.

    Khi vật cản che 1 bên mặt, detector CO bbox lại chỉ chứa vùng thấy
    được → w/h tụt. Đo thực tế (lena): bình thường 0.68; che 45% → 0.47.
    Gap đủ lớn để phát hiện.
    """
    x1, y1, x2, y2 = face.bbox
    w, h = x2 - x1, y2 - y1
    if h <= 0:
        return 1.0
    return float(w / h)


def _bbox_narrowness_score(face) -> float:
    """0..1 — mức độ nghi ngờ bbox bị co lại do vật cản.

    BBOX_NARROWNESS_HI = 0.55: trên mức này = bình thường (lena 0.68).
    Dưới thêm BBOX_NARROWNESS_RAMP = 0.10 → chắc chắn bị che (0.47).
    Dùng ramp 0→1 để mượt.
    """
    ratio = _bbox_narrowness(face)
    return float(max(0.0, min(1.0, (BBOX_NARROWNESS_HI - ratio) / BBOX_NARROWNESS_RAMP)))


def _expand_box(box: tuple, f: float) -> tuple:
    """Mở rộng khung (x1,y1,x2,y2) thêm f× chiều rộng/cao mỗi bên."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return (x1 - f * w, y1 - f * h, x2 + f * w, y2 + f * h)


def _landmark_bbox(lm: np.ndarray, idx) -> tuple:
    """Bounding box của nhóm landmark (mảng index)."""
    pts = lm[idx]
    return (float(pts[:, 0].min()), float(pts[:, 1].min()),
            float(pts[:, 0].max()), float(pts[:, 1].max()))


def _region_mean(mag: np.ndarray, box: tuple) -> float:
    """Giá trị TRUNG BÌNH của ảnh gradient trong khung (kẹp theo biên ảnh)."""
    H, W = mag.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return 0.0
    return float(mag[y1:y2, x1:x2].mean())


def sobel_magnitude(frame: np.ndarray) -> np.ndarray:
    """Magnitude gradient Sobel của CẢ ẢNH — tính 1 LẦN, dùng lại cho N mặt.

    Trước đây texture_anomaly tự tính Sobel mỗi lần gọi → ảnh lớn nhiều
    mặt (photo view) tính lại Sobel toàn ảnh cho từng mặt = chậm N lần.
    Hàm này tách phần tính ra để caller (photo_view) tính 1 lần rồi truyền
    qua ``sobel_mag`` của occlusion_score/texture_anomaly. Trả mảng rỗng
    khi ảnh rỗng/lỗi — caller coi như không có tín hiệu texture.
    """
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except Exception:  # noqa: BLE001 — ảnh None/rỗng/sai định dạng
        return np.empty(0, dtype=np.float32)
    if gray.ndim != 2 or gray.size == 0:
        return np.empty(0, dtype=np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def texture_anomaly(face, frame=None, sobel_mag: np.ndarray | None = None) -> float:
    """0..1 — mức nghi ngờ vật thể che vùng MẮT / MIỆNG / TRÁN.

    Tín hiệu KẾT CẤU ẢNH (pixel) — bắt được che NHẸ mà det_score và
    landmark vẫn "bình thường" (InsightFace bịa vị trí hợp lệ). Đo edge
    density (Sobel) từng vùng rồi CHUẨN HÓA theo trung bình cả mặt (tự
    thích ứng ánh sáng / độ nét — blur làm mọi vùng giảm đều nên tỉ lệ
    không đổi):
      - MẮT: bình thường ~1.6 (lông mi, tròng đen, mí); ngón tay đè hốc
        mắt → tụt ~0.8 (đo thực tế). KÍNH làm TĂNG edge → KHÔNG kích
        hoạt (đúng yêu cầu không phiền người đeo kính).
      - MIỆNG: bình thường ~1.0; bàn tay che miệng+mũi → ~0.0.
      - TRÁN: bình thường ~1.1; tóc rủ / vật che → TĂNG (sợi tóc = edge).

    ``sobel_mag``: magnitude Sobel TÍNH SẴN của cả ảnh (caller photo_view
    tính 1 lần cho nhiều mặt — tránh N lần Sobel toàn ảnh). None → tự
    tính từ ``frame`` (giữ tương thích với caller cũ + test fake face).
    Chỉ tính khi có ``frame`` (occlusion_score(face) không có frame thì
    bỏ qua tín hiệu này — giữ tương thích với test đơn vị fake face).
    """
    if frame is None:
        return 0.0
    lm = getattr(face, "landmark_3d_68", None)
    if lm is None or len(lm) < 68:
        return 0.0
    if sobel_mag is None:
        sobel_mag = sobel_magnitude(frame)
    mag = sobel_mag
    if mag.size == 0:
        return 0.0

    bx1, by1, bx2, by2 = face.bbox.astype(int)
    face_avg = _region_mean(mag, (bx1, by1, bx2, by2)) + 1e-6

    eye_l = _expand_box(_landmark_bbox(lm, range(36, 42)), 0.4)
    eye_r = _expand_box(_landmark_bbox(lm, range(42, 48)), 0.4)
    mouth = _expand_box(_landmark_bbox(lm, range(48, 68)), 0.15)
    eye_line = min(eye_l[1], eye_r[1])
    forehead = (min(eye_l[0], eye_r[0]), by1,
                max(eye_l[2], eye_r[2]), eye_line)

    r_eye_l = _region_mean(mag, eye_l) / face_avg
    r_eye_r = _region_mean(mag, eye_r) / face_avg
    r_mouth = _region_mean(mag, mouth) / face_avg
    r_forehead = _region_mean(mag, forehead) / face_avg

    ear_l, ear_r = eye_aspect_ratios(face)
    # Mắt MỞ (EAR > ngưỡng — không phải chớp/nhắm) mà mất kết cấu → bị che
    comp_eye_l = (
        max(0.0, min(1.0, (EYE_TEXTURE_LOW_HI - r_eye_l) / EYE_TEXTURE_RAMP))
        if ear_l > EYE_TEXTURE_EAR_MIN else 0.0
    )
    comp_eye_r = (
        max(0.0, min(1.0, (EYE_TEXTURE_LOW_HI - r_eye_r) / EYE_TEXTURE_RAMP))
        if ear_r > EYE_TEXTURE_EAR_MIN else 0.0
    )
    comp_mouth = max(0.0, min(1.0, (MOUTH_TEXTURE_LOW_HI - r_mouth)
                              / MOUTH_TEXTURE_RAMP))
    comp_forehead = max(0.0, min(1.0, (r_forehead - FOREHEAD_TEXTURE_HIGH_LO)
                                 / FOREHEAD_TEXTURE_RAMP))
    return float(max(comp_eye_l, comp_eye_r, comp_mouth, comp_forehead))


def occlusion_score(face, frame=None, sobel_mag: np.ndarray | None = None) -> float:
    """0..1 — mức độ nghi ngờ khuôn mặt bị che khuất / KHÔNG RÕ.

    Kết hợp 5 tín hiệu (hiệu chỉnh bằng đo thực tế trên lena — probe_occl3
    + probe_tex2):
      - ``det_score`` của SCRFD (0.40): mặt bị che làm detector hạ điểm
        (bình thường 0.75-0.90; bị che vừa 0.55-0.70).
      - ``eye_anomaly`` (0.20): EAR từng mắt vọt > 0.70 / sụp < 0.15 =
        landmark mắt bị bóp méo vì vật cản che 1 bên.
      - ``texture_anomaly`` (0.30, cần ``frame``): edge density vùng
        MẮT/MIỆNG/TRÁN — bắt che NHẸ (ngón tay đè hốc mắt, tay che
        miệng+mũi, tóc rủ trán) mà det/landmark không phản ứng.
      - Tỉ lệ landmark NGOÀI khung (0.05) + mũi-cằm (0.05): phụ trợ.

    Phân tầng: score < OCCLUSION_WARN → mặt rõ (nhận diện bình thường);
    WARN → hơi bị che (tên kèm ⚠, vẫn cố nhận diện); ≥ BLOCK → chặn
    (hiện "⚠ Mặt bị che", không đoán sai tên).

    ``sobel_mag``: magnitude Sobel tính sẵn (tối ưu photo view — xem
    texture_anomaly). None → tự tính từ ``frame``.

    LƯU Ý: vật cản LỚN (khẩu trang, tay che miệng) thường làm detector
    KHÔNG tìm thấy mặt — nằm NGOÀI phạm vi hàm này (không có face để
    tính). Che TÍ XÍU (ngón tay chạm khóe miệng) không thể phát hiện ở
    1 khung hình — mặt vẫn là khuôn mặt bình thường, mọi tín hiệu đều
    không đổi (đo thực tế).
    """
    det = float(getattr(face, "det_score", 0.0))
    det_component = max(0.0, min(1.0, (0.82 - det) / 0.25))
    outside_component = min(1.0, landmarks_outside_ratio(face) * 5.0)
    # Che phần DƯỚI mặt (tay che miệng/mũi): landmark cằm bị kéo lên →
    # chin_nose_ratio vọt (bình thường 0.39, che nửa dưới 0.47 — đo thực tế)
    chin_component = max(0.0, min(1.0, (chin_nose_ratio(face) - 0.43) / 0.12))
    return float(
        0.35 * det_component
        + 0.20 * eye_anomaly(face)
        + 0.05 * outside_component
        + 0.05 * chin_component
        + 0.25 * texture_anomaly(face, frame, sobel_mag=sobel_mag)
        + 0.10 * _bbox_narrowness_score(face)
    )
