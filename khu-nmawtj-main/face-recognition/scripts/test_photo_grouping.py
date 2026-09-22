# -*- coding: utf-8 -*-
"""Kiểm tra logic GOM NHÓM NGƯỜI của EnrollmentPhotoService (không cần model).

Tạo embedding GIẢ cho 3 người × 3 ảnh + 2 ảnh "rác" (không đạt), chạy
``group_into_persons`` và đối chiếu:
  1. Gom đúng 3 nhóm (không gộp chéo người, không tách cùng người).
  2. Tên gợi ý từ tên file đúng ("NguyenVanA_1" → "NguyenVanA").
  3. Ảnh không đạt KHÔNG rơi vào nhóm nào.
  4. Tên nhóm không trùng nhau.

Chạy:  python scripts/test_photo_grouping.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.enrollment import CapturedSample
from app.services.enrollment_photos import (
    EnrollmentPhotoService,
    PhotoProcessResult,
    _name_from_filename,
)

rng = np.random.default_rng(42)
PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✓ PASS  {name}")
    else:
        FAIL += 1
        print(f"  ✗ FAIL  {name}  {detail}")


def make_result(path: str, base_vec: np.ndarray, noise: float = 0.02) -> PhotoProcessResult:
    """Ảnh ĐẠT giả: embedding = vector người + nhiễu nhỏ (cosine ~0.97)."""
    emb = base_vec + rng.normal(0, noise, base_vec.shape)
    emb = emb / np.linalg.norm(emb)
    return PhotoProcessResult(
        path,
        True,
        sample=CapturedSample(
            embedding=emb.astype(np.float32), quality=0.9, face_crop=np.zeros((4, 4, 3), np.uint8)
        ),
        face_crop=np.zeros((4, 4, 3), np.uint8),
        det_score=0.9,
    )


def main() -> int:
    print("═" * 60)
    print("KIỂM TRA GOM NHÓM NGƯỜI (không cần model)")
    print("═" * 60)

    # ── 1. Tên từ tên file ──
    print("\n[1] Rút tên từ tên file")
    check("NguyenVanA_1.jpg → NguyenVanA", _name_from_filename("C:/a/NguyenVanA_1.jpg") == "NguyenVanA")
    check("tran-b (3).jpg → 'tran-b'", (_name_from_filename("C:/x/tran-b (3).jpg") or "") == "tran-b")
    check("IMG_2024.jpg → None (tiền tố phiền)", _name_from_filename("C:/x/IMG_2024.jpg") is None)
    check("123.jpg → None (không có chữ)", _name_from_filename("C:/x/123.jpg") is None)
    check("long-02.jpg → long", (_name_from_filename("C:/x/long-02.jpg") or "") == "long")

    # ── 2. Gom nhóm 3 người ──
    print("\n[2] Gom nhóm 3 người × 3 ảnh")
    # 3 vector người khác nhau hoàn toàn (cosine ~0 giữa các người)
    vecs = []
    for _ in range(3):
        v = rng.normal(0, 1, 512).astype(np.float32)
        vecs.append(v / np.linalg.norm(v))
    people = ["NguyenVanA", "TranThiB", "LeVanC"]
    results: list[PhotoProcessResult] = []
    for pi, pname in enumerate(people):
        for k in range(3):
            results.append(make_result(f"C:/anh/{pname}_{k + 1}.jpg", vecs[pi]))
    # 2 ảnh KHÔNG ĐẠT
    results.append(PhotoProcessResult("C:/anh/blur.jpg", False, "Ảnh mờ"))
    results.append(PhotoProcessResult("C:/anh/noface.jpg", False, "Không thấy khuôn mặt nào"))

    # Service không cần detector vì chỉ gọi group_into_persons (thuần toán)
    class _NullDet:
        pass

    service = EnrollmentPhotoService.__new__(EnrollmentPhotoService)
    groups = service.group_into_persons(results)

    check("Gom đúng 3 nhóm", len(groups) == 3, f"nhận {len(groups)}")
    if len(groups) == 3:
        names = sorted(g.name for g in groups)
        expected = sorted(people)
        check("Tên gợi ý đúng từ tên file", names == expected, f"{names} != {expected}")
        check("Mỗi nhóm 3 ảnh", all(len(g.results) == 3 for g in groups),
              f"số ảnh mỗi nhóm: {[len(g.results) for g in groups]}")
        check("Mỗi nhóm đúng người (path chứa tên nhóm)",
              all(g.name in r.path for g in groups for r in g.results),
              "")
        check("Ảnh không đạt KHÔNG vào nhóm nào",
              all("blur.jpg" not in r.path and "noface.jpg" not in r.path
                  for g in groups for r in g.results))

    # ── 3. Nhiễu lớn hơn (cùng người nhưng ảnh rất khác góc) ──
    print("\n[3] Cùng người, ảnh khác góc nhiều (nhiễu 0.25)")
    v = rng.normal(0, 1, 512).astype(np.float32)
    v /= np.linalg.norm(v)
    same_person = [
        make_result("C:/x/anh goc 1.jpg", v, noise=0.25),
        make_result("C:/x/anh goc 2.jpg", v, noise=0.25),
        make_result("C:/x/anh goc 3.jpg", v, noise=0.25),
    ]
    groups2 = service.group_into_persons(same_person)
    check("Vẫn gom thành 1 nhóm", len(groups2) == 1, f"nhận {len(groups2)}")

    # ── 4. Khác người rất giống nhau (sinh đôi giả lập: cosine 0.55) ──
    print("\n[4] Hai người GIỐNG NHAU (cosine ~0.55 — dưới ngưỡng 0.50 khi so với tâm)")
    v1 = rng.normal(0, 1, 512).astype(np.float32)
    v1 /= np.linalg.norm(v1)
    v2 = v1 + 0.95 * rng.normal(0, 1, 512).astype(np.float32)
    v2 /= np.linalg.norm(v2)
    sim = float(np.dot(v1, v2))
    twins = [
        make_result("C:/x/sinh doi A.jpg", v1, noise=0.01),
        make_result("C:/x/sinh doi A 2.jpg", v1, noise=0.01),
        make_result("C:/x/sinh doi B.jpg", v2, noise=0.01),
    ]
    groups3 = service.group_into_persons(twins)
    check(f"cosine giữa 2 người = {sim:.2f} → gom 2 nhóm", len(groups3) == 2, f"nhận {len(groups3)}")

    print("\n" + "═" * 60)
    if FAIL == 0:
        print(f"KẾT QUẢ: TẤT CẢ PASS ({PASS} kiểm tra) ✅")
    else:
        print(f"KẾT QUẢ: {PASS} PASS, {FAIL} FAIL ❌")
    print("═" * 60)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
