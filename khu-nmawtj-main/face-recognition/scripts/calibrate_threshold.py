"""Hiệu chỉnh ngưỡng nhận diện (Bước 6) — đo bằng dữ liệu THẬT trên GPU.

Ý tưởng (kiến thức ROC):
  - "Genuine"   = cặp khuôn mặt CÙNG người → điểm cosine nên CAO.
  - "Impostor"  = cặp khuôn mặt KHÁC người → điểm cosine nên THẤP.
  - Ngưỡng lý tưởng nằm GIỮA hai cụm điểm: cao hơn max-impostor (ít nhầm
    người lạ thành quen) nhưng thấp hơn min-genuine (không bỏ sót người quen).

Script này chạy model thật trên máy (lena + ảnh nhóm 6 người trong
insightface) để đo 2 cụm điểm, đề xuất ngưỡng và GHI vào config.json
(giữ nguyên các cài đặt khác, kể cả mật khẩu).

Chạy:  .venv\\Scripts\\python.exe scripts\\calibrate_threshold.py
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Config  # noqa: E402
from app.core.detector import FaceDetector, MODELS_ROOT  # noqa: E402
from app.core.embedder import FaceEmbedder  # noqa: E402
from app.core.matcher import DEFAULT_THRESHOLD  # noqa: E402

# Khoảng ngưỡng hợp lý (giới hạn trên để không quá khắt khe)
THRESHOLD_MIN, THRESHOLD_MAX = 0.25, 0.60


def _load_lena(path: Path) -> np.ndarray:
    """Lấy ảnh lena (tự tải nếu thiếu/hỏng) và trả về mảng BGR."""
    if not path.exists() or path.stat().st_size == 0:
        print("  Tải ảnh test lena.jpg từ OpenCV samples...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
            path,
        )
    img = cv2.imread(str(path))
    assert img is not None, "thiếu lena_test.jpg"
    return img


def main() -> None:
    print("=== HIỆU CHỈNH NGƯỠNG NHẬN DIỆN (chạy model thật trên GPU) ===\n")

    detector = FaceDetector(MODELS_ROOT)
    embedder = FaceEmbedder(detector)

    # ---- Chuẩn bị ảnh ----
    lena_path = PROJECT_ROOT / "lena_test.jpg"
    lena = _load_lena(lena_path)
    h, w = lena.shape[:2]
    # Biến thể của lena: xoay 12° + thu nhỏ — CÙNG người (genuine)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), 12, 0.75)
    lena_twist = cv2.warpAffine(lena, m, (w, h))
    # Ảnh nhóm 6 người — các khuôn mặt KHÁC người (impostor)
    insightface_images = Path(sys.prefix) / "Lib" / "site-packages" / "insightface" / "data" / "images"
    group = cv2.imread(str(insightface_images / "t1.jpg"))
    assert group is not None, "thiếu ảnh t1.jpg trong insightface/data/images"

    # ---- Trích embedding ----
    def embed_face(img: np.ndarray):
        # Một lượt duy nhất qua từng face (tránh gọi model thừa lần 2)
        embeddings = []
        for face in detector.detect(img):
            emb = embedder.embed_face(face)
            if emb is not None:
                embeddings.append(emb)
        return embeddings

    emb_lena = embed_face(lena)
    emb_lena_twist = embed_face(lena_twist)
    emb_group = embed_face(group)
    print(f"  lena: {len(emb_lena)} mặt · lena xoay: {len(emb_lena_twist)} mặt · nhóm: {len(emb_group)} mặt")
    if not (emb_lena and emb_lena_twist and len(emb_group) >= 2):
        print("  ❌ Không đủ khuôn mặt để hiệu chỉnh — dừng, giữ ngưỡng mặc định.")
        sys.exit(1)

    # ---- Đo 2 cụm điểm ----
    cos = FaceEmbedder.cosine_similarity

    genuine = []    # cùng người (lena vs lena xoay)
    for a in emb_lena:
        for b in emb_lena_twist:
            genuine.append(cos(a, b))

    impostor = []   # khác người (mọi cặp khác nhau trong ảnh nhóm + lena vs nhóm)
    for i in range(len(emb_group)):
        for j in range(i + 1, len(emb_group)):
            impostor.append(cos(emb_group[i], emb_group[j]))
    for a in emb_lena:
        for b in emb_group:
            impostor.append(cos(a, b))

    genuine = np.array(genuine)
    impostor = np.array(impostor)
    print(f"\n  Cụm CÙNG người (genuine) : {len(genuine)} cặp — min {genuine.min():.3f} · max {genuine.max():.3f}")
    print(f"  Cụm KHÁC người (impostor): {len(impostor)} cặp — min {impostor.min():.3f} · max {impostor.max():.3f}")

    # ---- Đề xuất ngưỡng ----
    gap = genuine.min() - impostor.max()
    if gap <= 0:
        print("\n  ⚠ Hai cụm điểm CHỒNG LÊN NHAU (gap âm) — không có ngưỡng tách hoàn hảo.")
        print(f"     Đề xuất: điểm giữa 2 cụm = {(genuine.min() + impostor.max()) / 2:.3f} (chấp nhận đánh đổi).")
    else:
        print(f"\n  ✅ Cách biệt rõ ràng: gap = {gap:.3f}")

    # Điểm giữa hai cụm, giới hạn trong [THRESHOLD_MIN, THRESHOLD_MAX]
    suggested = float((genuine.min() + impostor.max()) / 2.0)
    suggested = max(THRESHOLD_MIN, min(THRESHOLD_MAX, suggested))
    print(f"  → Ngưỡng đề xuất: {suggested:.3f}")
    print("  ℹ Ước lượng DƯỚI ĐIỀU KIỆN LÝ TƯỞNG (ảnh tĩnh, sáng tốt) — điểm genuine")
    print("    từ webcam thật thường THẤP HƠN (ánh sáng, rung tay). Nếu nhận diện")
    print("    thật hay bỏ sót, hãy hạ ngưỡng xuống ~0.45–0.50 ở Cài đặt (Bước 14).")

    # ---- Lưu vào config.json (giữ nguyên mật khẩu + cài đặt khác) ----
    config = Config.load()
    old = config.recognition_threshold
    config.recognition_threshold = round(suggested, 3)
    config.save()
    print(f"  → Đã lưu config.json: recognition_threshold {old:.3f} → {config.recognition_threshold:.3f}")

    lena_path.unlink(missing_ok=True)  # dọn ảnh tạm
    print("\n✅ HOÀN TẤT HIỆU CHỈNH NGƯỠNG")


if __name__ == "__main__":
    main()
