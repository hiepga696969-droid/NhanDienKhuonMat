"""Test thực tế Bước 5 — Embedding khuôn mặt (chạy trên GPU DirectML).

Kiểm chứng:
  1. Face của lena có embedding 512 chiều, chuẩn L2 = 1 (đã chuẩn hóa).
  2. CÙNG người (lena vs lena xoay/nhỏ) → cosine CAO (>= 0.5).
  3. KHÁC người (lena vs Tom Hanks) → cosine THẤP hơn hẳn.
  4. Face không có embedding → embed_face trả None.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.detector import FaceDetector, MODELS_ROOT  # noqa: E402
from app.core.embedder import EMBEDDING_DIM, FaceEmbedder  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [QUA] {name} {detail}")
    else:
        FAIL += 1
        print(f"  [THẤT] {name} {detail}")


def main() -> None:
    print("=== TEST BƯỚC 5: EMBEDDING KHUÔN MẶT ===")

    detector = FaceDetector(MODELS_ROOT)
    embedder = FaceEmbedder(detector)

    # --- Ảnh test: lena (1 người) + biến thể, và t1.jpg (ảnh nhóm 6 người) ---
    lena_path = Path(__file__).resolve().parent.parent / "lena_test.jpg"
    if not lena_path.exists() or lena_path.stat().st_size == 0:
        print("  Tải ảnh test lena.jpg từ OpenCV samples...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
            lena_path,
        )
    lena = cv2.imread(str(lena_path))
    assert lena is not None, "thiếu lena_test.jpg"
    h, w = lena.shape[:2]
    # Biến thể: thu nhỏ 75% + xoay 12 độ (mô phỏng người hơi nghiêng đầu)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), 12, 0.75)
    lena_twist = cv2.warpAffine(lena, m, (w, h))
    # Ảnh test nằm trong gói insightface (di động được — không hardcode đường dẫn tuyệt đối)
    insightface_images = Path(sys.prefix) / "Lib" / "site-packages" / "insightface" / "data" / "images"
    group = cv2.imread(str(insightface_images / "t1.jpg"))
    assert group is not None, "thiếu ảnh t1.jpg trong insightface/data/images"

    # --- 1. Embedding lena ---
    faces = detector.detect(lena)
    check("phát hiện đúng 1 mặt trên lena", len(faces) == 1, f"(tìm thấy {len(faces)})")
    if not faces:
        print(f"KẾT QUẢ: {PASS} qua / {FAIL} thất bại")
        sys.exit(1)

    emb_a = embedder.embed_face(faces[0])
    check("embedding tồn tại", emb_a is not None)
    check("đúng 512 chiều", emb_a is not None and len(emb_a) == EMBEDDING_DIM,
          f"(len={len(emb_a)} nếu có)")
    check("chuẩn L2 ≈ 1.0 (đã chuẩn hóa)",
          emb_a is not None and abs(float(np.linalg.norm(emb_a)) - 1.0) < 1e-3,
          f"(norm={float(np.linalg.norm(emb_a)):.4f} nếu có)")

    # --- 2. CÙNG người: lena vs lena xoay/nhỏ ---
    faces_twist = detector.detect(lena_twist)
    check("phát hiện mặt trên lena biến thể", len(faces_twist) == 1,
          f"(tìm thấy {len(faces_twist)})")
    if faces_twist:
        emb_a2 = embedder.embed_face(faces_twist[0])
        sim_same = FaceEmbedder.cosine_similarity(emb_a, emb_a2)
        check("CÙNG người → cosine >= 0.5", sim_same >= 0.5,
              f"(cosine={sim_same:.3f})")

    # --- 3. KHÁC người: lena vs khuôn mặt khác trong ảnh nhóm t1.jpg ---
    faces_group = detector.detect(group)
    check("ảnh nhóm t1.jpg phát hiện >= 2 mặt", len(faces_group) >= 2,
          f"(tìm thấy {len(faces_group)})")
    if len(faces_group) >= 2:
        # Lấy 2 khuôn mặt khác nhau (lớn nhất = người đầu tiên, kế tiếp = người khác)
        faces_sorted = sorted(faces_group, key=lambda f: -(f.bbox[2] - f.bbox[0]))
        emb_b = embedder.embed_face(faces_sorted[0])
        emb_c = embedder.embed_face(faces_sorted[1])
        sim_diff = FaceEmbedder.cosine_similarity(emb_b, emb_c)
        check("KHÁC người → cosine < 0.5", sim_diff < 0.5,
              f"(cosine={sim_diff:.3f})")
        if faces_twist:
            check("cùng người cao hơn hẳn khác người", sim_same > sim_diff + 0.2,
                  f"(same={sim_same:.3f} > diff={sim_diff:.3f})")

    # --- 4. Face thiếu embedding → None ---
    class FakeFace:  # noqa: D401
        pass

    check("face không embedding → None", embedder.embed_face(FakeFace()) is None)

    # --- 5. detect_and_embed trả cặp (face, embedding) ---
    pairs = embedder.detect_and_embed(lena)
    check("detect_and_embed trả đúng số cặp", len(pairs) == len(faces),
          f"(cặp={len(pairs)})")
    if pairs:
        check("cặp đầu có embedding đủ chiều", pairs[0][1] is not None
              and len(pairs[0][1]) == EMBEDDING_DIM)

    print(f"\nKẾT QUẢ: {PASS} qua / {FAIL} thất bại")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
