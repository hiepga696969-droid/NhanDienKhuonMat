"""Phát hiện khuôn mặt — SCRFD (module detection của bộ model buffalo_l).

Chỉ tải module `detection` (nhẹ hơn, nhanh hơn tải toàn bộ bộ model).
Kết quả mỗi khuôn mặt là đối tượng Face có:
  - face.bbox      : [x1, y1, x2, y2]
  - face.det_score : độ tin cậy (0..1)
  - face.kps       : 5 điểm landmark (mắt, mũi, miệng)
"""
from __future__ import annotations

import logging
import threading
import zipfile
from pathlib import Path

import numpy as np
import requests
from insightface.app import FaceAnalysis

from app.config import RESOURCE_DIR

logger = logging.getLogger(__name__)

# FaceAnalysis tìm model tại <root>/models/buffalo_l.
# Mã nguồn: RESOURCE_DIR = thư mục project; .exe: RESOURCE_DIR = bundle
# (cạnh exe đối với onedir, _MEIPASS đối với onefile — Bước 12).
MODELS_ROOT = RESOURCE_DIR / "models"

# URL tải bộ model buffalo_l (SCRFD + ArcFace, ~300MB) — cùng nguồn mà
# insightface dùng khi tự tải (github releases của deepinsight/insightface).
BUFFALO_L_ZIP_URL = (
    "https://github.com/deepinsight/insightface/releases/download/"
    "model-zoo/buffalo_l.zip"
)


def _model_dir_ready(model_dir: Path) -> bool:
    """Model coi như có mặt khi thư mục tồn tại VÀ có ít nhất 1 file .onnx.

    Kiểm tra này tránh trường hợp thư mục rỗng/tải dở dang (mất mạng giữa
    chừng) khiến app tưởng đã có model rồi lại crash khi FaceAnalysis nạp.
    """
    return model_dir.is_dir() and any(model_dir.glob("*.onnx"))


def ensure_model_available(models_root: Path | None = None, force: bool = False) -> Path:
    """Đảm bảo bộ model buffalo_l tồn tại — tự tải nếu thiếu.

    Trả về đường dẫn thư mục model (<models_root>/models/buffalo_l).
    Ném exception nếu tải thất bại (mất mạng, 404...) — caller hiển thị lỗi.
    """
    root = models_root if models_root is not None else MODELS_ROOT
    model_dir = root / "models" / "buffalo_l"
    if _model_dir_ready(model_dir) and not force:
        return model_dir

    # Có file zip nhưng chưa giải nén → giải nén luôn (tránh tải lại)
    zip_path = root / "models" / "buffalo_l.zip"
    if not zip_path.exists() or force:
        if not root.exists():
            root.mkdir(parents=True, exist_ok=True)
        (root / "models").mkdir(parents=True, exist_ok=True)
        logger.info("Đang tải model buffalo_l từ %s ...", BUFFALO_L_ZIP_URL)
        with requests.get(BUFFALO_L_ZIP_URL, stream=True, timeout=30) as resp:
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Tải model buffalo_l thất bại (HTTP {resp.status_code}) — "
                    "kiểm tra kết nối internet và thử lại."
                )
            with zip_path.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)
        logger.info("Đã tải model buffalo_l (%.1f MB)", zip_path.stat().st_size / 1e6)

    logger.info("Giải nén model buffalo_l vào %s ...", model_dir)
    if model_dir.exists():
        # Xóa nốt thư mục tải dở dang trước khi giải nén lại cho sạch
        import shutil

        shutil.rmtree(model_dir, ignore_errors=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(model_dir)
    if not _model_dir_ready(model_dir):
        raise RuntimeError(
            f"Giải nén model buffalo_l xong nhưng không thấy file .onnx tại {model_dir} — "
            "file zip có thể hỏng, xóa thư mục models/ và thử lại."
        )
    logger.info("Model buffalo_l sẵn sàng tại %s", model_dir)
    return model_dir

# ONNX Runtime chạy GPU qua DirectML (fallback CPU)
PROVIDERS = ["DmlExecutionProvider", "CPUExecutionProvider"]


class FaceDetector:
    """Bọc SCRFD: nhận ảnh BGR → trả về danh sách khuôn mặt phát hiện."""

    def __init__(
        self,
        models_root: Path = MODELS_ROOT,
        det_thresh: float = 0.5,
        # 320 thay 640 (mặc định): nhẹ hơn ~4 lần mỗi lần detect — đủ cho
        # webcam điểm danh (mặt chiếm ≥1/4 khung vẫn được phát hiện tốt).
        # Nếu cần nhận diện mặt rất nhỏ/xa, truyền det_size=(480, 480) hoặc lớn hơn.
        det_size: tuple[int, int] = (320, 320),
    ) -> None:
        # Lưu ý: insightface 1.0.1 bị segfault với allowed_modules=['detection']
        # ensure_model_available: nếu thiếu model sẽ tự tải về (cần internet).
        ensure_model_available(models_root)
        self._app = FaceAnalysis(
            name="buffalo_l",
            root=str(models_root),
            providers=PROVIDERS,
        )
        self._app.prepare(ctx_id=0, det_size=det_size, det_thresh=det_thresh)
        self._lock = threading.Lock()  # bảo vệ việc hoán đổi models khi detect
        logger.info("FaceDetector sẵn sàng (SCRFD buffalo_l)")

    def detect(self, img_bgr: np.ndarray, with_recognition: bool = True) -> list:
        """Phát hiện khuôn mặt trong ảnh BGR; trả danh sách Face (bbox, score, kps).

        ``with_recognition=False`` → CHẾ ĐỘ NHẸ: chỉ chạy SCRFD (phát hiện),
        tạm GỠ ArcFace + genderage + landmark khỏi FaceAnalysis — nhanh
        gấp nhiều lần (ArcFace r50 là phần NẶNG nhất trên CPU). Face trả
        về KHÔNG có ``normed_embedding`` — caller tự nhúng khi cần.

        Mặc định True → mọi caller cũ (đăng ký, ảnh chụp, script test)
        hoạt động như trước, không cần sửa gì.
        """
        if with_recognition:
            with self._lock:
                return self._app.get(img_bgr)
        # Chế độ nhẹ: lọc giữ lại CHỈ model detection (SCRFD có sẵn kps 5 điểm
        # nên liveness/occlusion vẫn dùng được nếu cần), ArcFace/genderage/
        # landmark bị gỡ tạm — KHÔNG hủy model, khôi phục ngay sau get().
        saved = self._app.models
        light = {k: v for k, v in saved.items() if "det" in k.lower()}
        with self._lock:
            self._app.models = light
            try:
                return self._app.get(img_bgr)
            finally:
                self._app.models = saved  # LUÔN khôi phục — kể cả khi lỗi

    @property
    def app(self) -> FaceAnalysis:
        """FaceAnalysis dùng chung — cho FaceEmbedder tái sử dụng model đã nạp."""
        return self._app
