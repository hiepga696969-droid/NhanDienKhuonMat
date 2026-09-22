"""Test thực tế Bước 6 — FaceMatcher so khớp cosine (chạy trên GPU DirectML).

Quy trình đúng theo kiến trúc:
  1. Trích embedding THẬT từ ảnh bằng GPU (detector + embedder).
  2. Lưu embedding vào CSDL (FaceSampleRepository) — như đăng ký thật.
  3. Nạp toàn bộ mẫu từ DB vào CosineMatcher (load_samples).
  4. So khớp: cùng người → đạt ngưỡng; khác người → bị chặn (None).

Chạy:  .venv\\Scripts\\python.exe scripts\\test_matcher.py
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.detector import FaceDetector, MODELS_ROOT  # noqa: E402
from app.core.embedder import FaceEmbedder  # noqa: E402
from app.core.matcher import CosineMatcher  # noqa: E402
from app.infrastructure.db import Database  # noqa: E402
from app.infrastructure.repositories import (  # noqa: E402
    FaceSampleRepository,
    PersonRepository,
)

TEMP_DB = PROJECT_ROOT / "data" / "test_matcher.db"

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


def sim_of(result) -> str:
    """Định dạng điểm tương đồng từ MatchResult; '—' nếu là None (người lạ)."""
    return f"{result.similarity:.3f}" if result is not None else "—"


def main() -> None:
    print("=== TEST BƯỚC 6: FACEMATCHER (cosine, dữ liệu từ DB) ===\n")

    detector = FaceDetector(MODELS_ROOT)
    embedder = FaceEmbedder(detector)

    # ---------- 1. Lấy embedding thật ----------
    lena_path = PROJECT_ROOT / "lena_test.jpg"
    if not lena_path.exists() or lena_path.stat().st_size == 0:
        print("  Tải ảnh test lena.jpg...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
            lena_path,
        )
    lena = cv2.imread(str(lena_path))
    h, w = lena.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), 12, 0.75)
    lena_twist = cv2.warpAffine(lena, m, (w, h))

    insightface_images = Path(sys.prefix) / "Lib" / "site-packages" / "insightface" / "data" / "images"
    group = cv2.imread(str(insightface_images / "t1.jpg"))

    faces_lena = detector.detect(lena)
    faces_twist = detector.detect(lena_twist)
    faces_group = detector.detect(group)

    emb_lena = embedder.embed_face(faces_lena[0]) if faces_lena else None
    emb_twist = embedder.embed_face(faces_twist[0]) if faces_twist else None
    check("có embedding lena + biến thể", emb_lena is not None and emb_twist is not None)
    if emb_lena is None or emb_twist is None:
        sys.exit(1)

    # 2 khuôn mặt khác nhau trong ảnh nhóm (người khác)
    faces_sorted = sorted(faces_group, key=lambda f: -(f.bbox[2] - f.bbox[0]))
    emb_other = embedder.embed_face(faces_sorted[0]) if faces_sorted else None
    emb_other2 = embedder.embed_face(faces_sorted[1]) if len(faces_sorted) > 1 else None
    check("có embedding 2 người khác", emb_other is not None and emb_other2 is not None)
    if emb_other is None or emb_other2 is None:
        sys.exit(1)

    # ---------- 2. Lưu vào DB tạm (giống đăng ký thật: vài mẫu/người) ----------
    for suffix in ("", "-wal", "-shm"):
        Path(str(TEMP_DB) + suffix).unlink(missing_ok=True)
    db = Database(TEMP_DB)
    people = PersonRepository(db)
    samples = FaceSampleRepository(db)

    p_alice = people.add("Alice", "data/thumbs/alice.png")
    p_bob = people.add("Bob", "data/thumbs/bob.png")
    # Alice: 2 mẫu (góc thẳng + nghiêng) — Bob: 1 mẫu
    samples.add(p_alice.id, emb_lena, quality=0.95)
    samples.add(p_alice.id, emb_twist, quality=0.88)
    samples.add(p_bob.id, emb_other, quality=0.90)
    check("DB tạm: 2 người, 3 mẫu embedding",
          people.count() == 2 and samples.count_by_person(p_alice.id) == 2)

    # ---------- 3. Nạp vào Matcher từ DB ----------
    matcher = CosineMatcher()
    matcher.load_samples(samples.all_samples())
    # Matcher trung bình mỗi người → size = SỐ NGƯỜI (2), không phải số mẫu (3)
    check("matcher nạp 2 người (3 mẫu)", matcher.size == 2)

    # ---------- 4. So khớp ----------
    # Cùng người (lena vs mẫu Alice) → phải nhận diện được Alice
    res = matcher.match(emb_lena, threshold=0.4)
    check("cùng người → trả Alice", res is not None and res.person_id == p_alice.id,
          f"(similarity={sim_of(res)})")

    # Biến thể lena (đã xoay) → vẫn nhận Alice
    res2 = matcher.match(emb_twist, threshold=0.4)
    check("biến thể xoay → vẫn trả Alice",
          res2 is not None and res2.person_id == p_alice.id,
          f"(similarity={sim_of(res2)})")

    # Khác người (người chưa đăng ký trong ảnh nhóm) → phải là None (người lạ)
    res3 = matcher.match(emb_other2, threshold=0.4)
    check("người lạ → None (dưới ngưỡng)", res3 is None,
          f"(similarity={sim_of(res3)})")

    # ---------- 5. Hành vi ngưỡng ----------
    # Ngưỡng quá cao (vượt cosine tối đa = 1.0) → kể cả cùng người bị từ chối.
    # (Không dùng sim_same + delta: với trung bình mẫu, cosine của một mẫu
    # với vector TRUNG BÌNH có thể CAO HƠN cosine giữa 2 mẫu với nhau.)
    res_high = matcher.match(emb_twist, threshold=1.01)
    check("ngưỡng quá cao → từ chối cùng người", res_high is None)
    # Ngưỡng quá thấp (0.0) → mọi thứ đều khớp (người lạ thành "quen")
    res_low = matcher.match(emb_other2, threshold=0.0)
    check("ngưỡng 0.0 → người lạ cũng khớp (giá trị thấp)",
          res_low is not None and res_low.similarity < 0.4,
          f"(similarity={res_low.similarity:.3f} nếu có)")

    # rank() xếp đúng thứ tự
    ranked = matcher.rank(emb_lena)
    check("rank: mẫu Alice đứng đầu", ranked[0].person_id == p_alice.id)

    # Matcher rỗng → match trả None
    matcher.clear()
    check("matcher rỗng → None", matcher.match(emb_lena) is None and matcher.size == 0)

    # ---------- Dọn dẹp ----------
    db.close()
    for suffix in ("", "-wal", "-shm"):
        Path(str(TEMP_DB) + suffix).unlink(missing_ok=True)
    lena_path.unlink(missing_ok=True)

    print(f"\nKẾT QUẢ: {PASS} qua / {FAIL} thất bại")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
