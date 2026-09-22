# -*- coding: utf-8 -*-
"""EnrollmentPhotoService — đăng ký nhân viên từ NHIỀU ẢNH (không cần camera).

Dùng khi người dùng CÓ SẴN ảnh nhân viên (VD: ảnh chụp CV, ảnh chụp tại
công trường...). Cho phép chọn NHIỀU ảnh cùng lúc — mỗi ảnh có mặt rõ
đóng góp 1 mẫu embedding; NHIỀU MẪU = nhận diện chính xác hơn (đúng như
đăng ký bằng camera 3-5 mẫu).

LUỒNG XỬ LÝ:
  1. Đọc ảnh (hỗ trợ jpg/png/bmp/webp...), chuẩn hóa về 3 kênh BGR.
  2. SCRFD phát hiện khuôn mặt — KHÔNG yêu cầu nhìn thẳng: ảnh nghiêng/
     ngửa vẫn nhận (đa góc = mẫu đa dạng).
  3. LỌC ẢNH XẤU: không thấy mặt / nhiều mặt / mặt quá nhỏ / mờ / tối /
     quá sáng → bỏ qua kèm LÝ DO rõ ràng cho UI hiển thị.
  4. Nhúng embedding (ArcFace) cho mỗi ảnh ĐẠT → CapturedSample
     (tái sử dụng kiểu dữ liệu của luồng camera — save_person chạy như cũ).
  5. Người dùng XÁC NHẬN danh sách mẫu (xem ảnh + bỏ mẫu không muốn)
     → EnrollmentService.save_person() lưu như đăng ký thường.

KIẾN TRÚC: class thuần Python KHÔNG đụng Qt — chạy được trong QThread
(UI gọi qua worker) hoặc trực tiếp (script test). Exceptions ném ra
bên ngoài — caller (worker) bắt và chuyển thành thông báo lỗi UI.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.core.detector import FaceDetector
from app.core.embedder import FaceEmbedder
from app.core.preprocess import preprocess_frame
from app.services.enrollment import CapturedSample

logger = logging.getLogger(__name__)

# ── Ngưỡng lọc ảnh (mềm hơn luồng camera — ảnh đúng là ảnh chụp cẩn thận) ──
MIN_FACE_SIZE = 80        # cạnh bbox tối thiểu (px) — ảnh CV mặt thường ~150+
MIN_DET_SCORE = 0.50      # độ tin cậy SCRFD tối thiểu
MIN_SHARPNESS = 20.0      # Laplacian variance vùng mặt (webcam lớn 25 — ảnh chụp thường nét hơn)
MIN_BRIGHTNESS = 25       # độ sáng trung bình vùng mặt
MAX_BRIGHTNESS = 240
MAX_SPOOF_LIKE_BLUR = 1e6  # (giữ chỗ — không dùng, tương thích tương lai)

# Đuôi file ảnh hỗ trợ (dùng khi "chọn thư mục")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Ngưỡng gom nhóm NGƯỜI bằng embedding (cosine — embedding đã chuẩn hóa L2).
# 0.50 = cân bằng: khác người (cosine < 0.40 với ngưỡng nhận diện của app)
# gần như chắc chắn KHÔNG bị gộp; cùng người (cosine ≥ 0.6) luôn gộp đúng.
CLUSTER_THRESHOLD = 0.50

# Regex cắt hậu tố số/thứ tự trong tên file: "NguyenVanA_02", "tran b (1)",
# "long-3" → "NguyenVanA" / "tran b" / "long" (dùng làm TÊN GỢI Ý nhóm).
_FILENAME_SUFFIX_RE = re.compile(r"[\s_\-]*\(?\d+\)?$")

# Tiền tố tên file PHIỀM (ảnh điện thoại/app đặt tên máy móc) — KHÔNG dùng
# làm tên gợi ý vì mọi người cùng chung tiền tố này.
_GENERIC_PREFIXES = {
    "img", "image", "photo", "anh", "ảnh", "zalo", "fb", "avatar", "pic",
    "screenshot", "man hinh", "màn hình", "download", "camera",
}


@dataclass
class PhotoProcessResult:
    """Kết quả xử lý MỘT ảnh đăng ký."""

    path: str            # đường dẫn ảnh gốc
    ok: bool             # True = thu được mẫu; False = bị loại (xem reason)
    reason: str = ""     # lý do bị loại (tiếng Việt, hiển thị ở UI)
    sample: CapturedSample | None = None
    face_crop: np.ndarray | None = None  # ảnh mặt để UI xem trước
    det_score: float = 0.0


@dataclass
class PersonGroup:
    """MỘT NHÓM = một NGƯỜI: các ảnh được AI gom theo khuôn mặt giống nhau."""

    name: str                      # tên gợi ý (từ tên file hoặc "Nhóm k")
    results: list[PhotoProcessResult]  # các ảnh ĐẠT thuộc nhóm này
    name_source: str = "ai"        # "filename" | "ai" — nguồn tên gợi ý
    department: str = ""           # phòng ban (người dùng nhập qua dialog)
    position: str = ""             # chức vụ (người dùng nhập qua dialog)


class EnrollmentPhotoService:
    """Xử lý nhiều ảnh → danh sách mẫu embedding cho ĐĂNG KÝ nhân viên."""

    def __init__(self, detector: FaceDetector, embedder: FaceEmbedder | None = None) -> None:
        self._detector = detector
        self._embedder = embedder or FaceEmbedder(detector)

    # ---------------------------------------------------------
    # Tiện ích ảnh
    # ---------------------------------------------------------
    @staticmethod
    def list_image_files(paths: list[str]) -> list[Path]:
        """Sắp xếp + lọc danh sách đường dẫn chỉ giữ file ảnh hỗ trợ.

        Nếu phần tử là THƯ MỤC → quét toàn bộ ảnh trong thư mục (không
        đệ quy — người dùng chọn thư mục muốn nói "các ảnh trong này").
        """
        out: list[Path] = []
        for p in paths:
            path = Path(p)
            if path.is_dir():
                out.extend(
                    f for f in sorted(path.iterdir())
                    if f.suffix.lower() in IMAGE_EXTENSIONS
                )
            elif path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file():
                out.append(path)
        # Khử trùng lặp giữ thứ tự
        seen: set[str] = set()
        unique: list[Path] = []
        for f in out:
            key = str(f.resolve()).lower()
            if key not in seen:
                seen.add(key)
                unique.append(f)
        return unique

    # ---------------------------------------------------------
    # Xử lý chính
    # ---------------------------------------------------------
    def process_photo(self, image_path: str) -> PhotoProcessResult:
        """Xử lý 1 ảnh: đọc → phát hiện mặt → lọc → nhúng embedding.

        KHÔNG ném exception cho lỗi ảnh đơn lẻ (file hỏng, không đọc được)
        — trả ``ok=False`` kèm ``reason`` để 1 file xấu không giết cả lô.
        """
        path = Path(image_path)
        try:
            # cv2.imdecode + np.fromfile: đọc được đường dẫn TIẾNG VIỆT/
            # khoảng trắng trên Windows (cv2.imread fail ở đường dẫn Unicode)
            data = np.fromfile(str(path), dtype=np.uint8)
            if data.size == 0:
                return PhotoProcessResult(str(path), False, "File rỗng hoặc không đọc được")
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if image is None:
                return PhotoProcessResult(str(path), False, "File không phải ảnh hợp lệ")
        except OSError as exc:
            return PhotoProcessResult(str(path), False, f"Lỗi đọc file: {exc}")

        # Ảnh quá lớn (VD 12MP từ điện thoại) → thu nhỏ cho nhẹ, giữ tỉ lệ.
        # SCRFD chạy trên ảnh thu nhỏ nhưng bbox scale về ảnh gốc khi crop —
        # insightface tự scale bbox theo ảnh đầu vào nên crop từ ảnh gốc OK.
        h, w = image.shape[:2]
        if max(h, w) > 1600:
            scale = 1600 / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)))
        # Ảnh xám/quá nhỏ
        if image.shape[0] < 60 or image.shape[1] < 60:
            return PhotoProcessResult(str(path), False, "Ảnh quá nhỏ (<60px)")

        try:
            # CLAHE đồng bộ luồng đăng ký camera (ánh sáng cùng điều kiện)
            detect_image = preprocess_frame(image)
            faces = self._detector.detect(detect_image, with_recognition=True)
        except Exception as exc:  # noqa: BLE001 — lỗi model → lý do rõ ràng
            logger.exception("Lỗi phát hiện khuôn mặt trong %s", path.name)
            return PhotoProcessResult(str(path), False, f"Lỗi AI phát hiện: {exc}")

        if not faces:
            return PhotoProcessResult(str(path), False, "Không thấy khuôn mặt nào")
        if len(faces) > 1:
            return PhotoProcessResult(
                str(path), False, f"Ảnh có {len(faces)} khuôn mặt — cần ảnh 1 người"
            )

        face = faces[0]
        x1, y1, x2, y2 = face.bbox.astype(int)
        fw, fh = x2 - x1, y2 - y1
        if fw < MIN_FACE_SIZE or fh < MIN_FACE_SIZE:
            return PhotoProcessResult(
                str(path), False, f"Khuôn mặt quá nhỏ ({fw}×{fh}px)"
            )
        det_score = float(face.det_score)
        if det_score < MIN_DET_SCORE:
            return PhotoProcessResult(
                str(path), False, f"Mặt không rõ ràng (điểm {det_score:.2f})"
            )

        # Lọc chất lượng vùng mặt (mờ / tối / chói)
        crop = self._crop_face(image, x1, y1, x2, y2)
        if crop is None or crop.size == 0:
            return PhotoProcessResult(str(path), False, "Không cắt được vùng mặt")
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        if sharpness < MIN_SHARPNESS:
            return PhotoProcessResult(
                str(path), False, f"Ảnh mờ (độ nét {sharpness:.0f})"
            )
        if brightness < MIN_BRIGHTNESS:
            return PhotoProcessResult(str(path), False, "Ảnh quá tối")
        if brightness > MAX_BRIGHTNESS:
            return PhotoProcessResult(str(path), False, "Ảnh quá sáng/chói")

        embedding = self._embedder.embed_face(face)
        if embedding is None:
            return PhotoProcessResult(str(path), False, "Không trích được embedding")

        quality = min(1.0, det_score)  # 0..1 — khớp kiểu CapturedSample
        return PhotoProcessResult(
            str(path),
            True,
            sample=CapturedSample(embedding=embedding, quality=quality, face_crop=crop),
            face_crop=crop,
            det_score=det_score,
        )

    def process_many(self, paths: list[str]) -> list[PhotoProcessResult]:
        """Xử lý NHIỀU ảnh — mỗi ảnh 1 kết quả (ok/reason), không dừng giữa lô."""
        files = self.list_image_files(paths)
        return [self.process_photo(str(f)) for f in files]

    # ---------------------------------------------------------
    # Gom nhóm NGƯỜI (thư mục chứa ảnh NHIỀU nhân viên khác nhau)
    # ---------------------------------------------------------
    def group_into_persons(
        self, results: list[PhotoProcessResult]
    ) -> list[PersonGroup]:
        """Gom các ảnh ĐẠT thành NHÓM NGƯỜI — hỗ trợ đăng ký NHIỀU người
        CÙNG LÚC từ 1 thư mục (FR mới).

        Thuật toán 2 lớp:
          1. CLUSTERING THEO KHUÔN MẶT: so embedding (cosine) với tâm từng
             nhóm hiện có — giống nhau ≥ ``CLUSTER_THRESHOLD`` → gộp vào;
             khác → nhóm mới. Khác người gần như KHÔNG BAO GIỜ bị gộp
             (cosine khác người < 0.40, ngưỡng 0.50 dư an toàn).
          2. TÊN GỢI Ý từ TÊN FILE: đa số ảnh nhóm có tiền tố tên giống
             nhau (VD "NguyenVanA_1", "NguyenVanA_2" → "NguyenVanA") →
             dùng làm tên; tiền tố phiền (img/anh/zalo...) bị loại. Không
             có tên từ file → "Nhóm 1", "Nhóm 2"... người dùng tự sửa.
        """
        oks = [
            r for r in results
            if r.ok and r.sample is not None and r.sample.embedding is not None
        ]
        if not oks:
            return []

        # ── Lớp 1: clustering theo khuôn mặt (greedy, tâm chạy dần) ──
        clusters: list[dict] = []  # {"sum": vector, "n": int, "items": [kết quả]}
        for r in oks:
            emb = np.asarray(r.sample.embedding, dtype=np.float32).flatten()
            norm = float(np.linalg.norm(emb))
            if norm == 0.0:
                continue  # embedding hỏng — bỏ mẫu này khỏi gom nhóm
            emb = emb / norm  # chuẩn hóa L2 → cosine = tích vô hướng
            best: dict | None = None
            best_sim = -1.0
            for c in clusters:
                centroid = c["sum"] / c["n"]
                cnorm = float(np.linalg.norm(centroid)) or 1.0
                sim = float(np.dot(emb, centroid / cnorm))
                if sim > best_sim:
                    best_sim, best = sim, c
            if best is not None and best_sim >= CLUSTER_THRESHOLD:
                best["sum"] += emb
                best["n"] += 1
                best["items"].append(r)
            else:
                clusters.append({"sum": emb.copy(), "n": 1, "items": [r]})

        # ── Lớp 2: tên gợi ý từ tên file (đa số trong nhóm) ──
        groups: list[PersonGroup] = []
        used: set[str] = set()
        for i, c in enumerate(clusters):
            counts: dict[str, tuple[str, int]] = {}  # lower → (hiển thị, số lần)
            for r in c["items"]:
                nm = _name_from_filename(r.path)
                if nm is not None:
                    key = nm.lower()
                    if key in counts:
                        counts[key] = (counts[key][0], counts[key][1] + 1)
                    else:
                        counts[key] = (nm, 1)
            name: str | None = None
            source = "ai"
            if counts:
                # Ưu tiên tiền tố xuất hiện NHIỀU NHẤT trong nhóm
                disp, _cnt = max(counts.values(), key=lambda t: t[1])
                name, source = disp, "filename"
            # Tên trùng → đánh số thứ tự (2 nhóm cùng tên file hay gặp khi
            # tiền tố phiền bị loại — VD "anh (2)"...)
            base = name or f"Nhóm {i + 1}"
            candidate = base
            j = 2
            while candidate.lower() in used:
                candidate = f"{base} ({j})"
                j += 1
            used.add(candidate.lower())
            groups.append(
                PersonGroup(name=candidate, results=c["items"], name_source=source)
            )

        # Nhóm CÓ TÊN từ file đứng trước (alphabet), nhóm AI sau — UI dễ đọc
        groups.sort(key=lambda g: (g.name_source != "filename", g.name.lower()))
        logger.info("Gom nhóm %d ảnh đạt thành %d người", len(oks), len(groups))
        return groups

    # ---------------------------------------------------------
    # Nội bộ
    # ---------------------------------------------------------
    @staticmethod
    def _crop_face(
        image: np.ndarray, x1: int, y1: int, x2: int, y2: int, pad_ratio: float = 0.25
    ) -> np.ndarray | None:
        """Cắt vùng mặt + đệm 25% (giống CameraWorker._crop_face — tĩnh để test)."""
        h, w = image.shape[:2]
        pad_x = int((x2 - x1) * pad_ratio)
        pad_y = int((y2 - y1) * pad_ratio)
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        if x2 <= x1 or y2 <= y1:
            return None
        return image[y1:y2, x1:x2].copy()


def _name_from_filename(path: str) -> str | None:
    """Rút TÊN GỢI Ý từ tên file: "NguyenVanA_02.jpg" → "NguyenVanA".

    - Cắt đuôi số/thứ tự ("_1", "-02", " (3)"...).
    - Chỉ nhận khi còn ≥ 2 ký tự VÀ có chữ cái (loại "123.jpg").
    - Loại tiền tố PHIỀN máy móc ("img", "anh", "zalo"...) — mọi người
      cùng chung tiền tố này, dùng làm tên sẽ nhầm lẫn.
    """
    stem = Path(path).stem
    prefix = _FILENAME_SUFFIX_RE.sub("", stem).strip()
    if len(prefix) < 2 or not any(c.isalpha() for c in prefix):
        return None
    if prefix.lower().strip("_ -") in _GENERIC_PREFIXES:
        return None
    return prefix
