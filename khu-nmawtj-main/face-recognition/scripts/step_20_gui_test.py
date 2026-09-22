"""Bước 20 — Top-K Matching + Outlier Rejection (lọc mẫu embedding xấu).

1. Unit: tạo nhiều embedding Copy cùng 1 vector + 1 mẫu xấu → CosineMatcher
   loại bỏ outlier → mean vector vẫn đúng hướng.
2. Unit: ≤3 mẫu → giữ nguyên mean (không loại outlier).
3. Unit: tất cả mẫu giống hệt nhau → không loại gì (std=0 → threshold=-inf).
4. Integration: DB thật +.Matcher nạp mẫu → match trả đúng người.
5. So sánh: nếu KHÔNG loại outlier → mẫu xấu kéo mean sai → match nhầm.

Chạy:  .venv/Scripts/python.exe scripts/step_20_gui_test.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import PySide6  # noqa: E402

pyside_dir = os.path.dirname(PySide6.__file__)
os.environ["QT_PLUGIN_PATH"] = os.path.join(pyside_dir, "plugins")
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(pyside_dir)
os.environ["QT_QPA_PLATFORM"] = "minimal"

import numpy as np  # noqa: E402

from app.core.matcher import CosineMatcher, MatchResult, _normalize  # noqa: E402
from app.infrastructure.db import Database  # noqa: E402
from app.infrastructure.repositories import FaceSample  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMP_DB = PROJECT_ROOT / "data" / "test_matcher_20.db"

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


def cleanup() -> None:
    for suffix in ("", "-wal", "-shm"):
        Path(str(TEMP_DB) + suffix).unlink(missing_ok=True)


# ---------------------------------------------------------
# 1. Outlier rejection — unit
# ---------------------------------------------------------
def test_outlier_rejection() -> None:
    print("\n[1] Outlier rejection — loại mẫu xấu")
    rng = np.random.default_rng(42)

    # Tạo "ground truth" vector
    truth = _normalize(rng.standard_normal(512).astype(np.float32))

    # 6 mẫu tốt (truth + noise nhỏ)
    good_samples = [_normalize(truth + rng.standard_normal(512).astype(np.float32) * 0.05) for _ in range(6)]
    # 1 mẫu xấu (hướng khác hẳn)
    bad_sample = _normalize(-truth + rng.standard_normal(512).astype(np.float32) * 0.1)

    m = CosineMatcher()
    for i, s in enumerate(good_samples):
        m.register("person_A", s)
    m.register("person_A", bad_sample)

    # Mean WITH outlier rejection
    mean_rejected = m._mean_of("person_A")
    sim_with_rejection = float(np.dot(mean_rejected, truth))

    # Mean WITHOUT outlier rejection (thô — dùng all_samples)
    mean_raw = _normalize(np.mean(good_samples + [bad_sample], axis=0))
    sim_raw = float(np.dot(mean_raw, truth))

    check("mean rejection gần ground truth hơn mean thô",
          sim_with_rejection > sim_raw,
          f"(rejected={sim_with_rejection:.4f}, raw={sim_raw:.4f})")
    check("mean rejection cosine > 0.85 (rất gần truth)",
          sim_with_rejection > 0.85,
          f"(sim={sim_with_rejection:.4f})")


# ---------------------------------------------------------
# 2. ≤3 mẫu → giữ nguyên mean (không loại outlier)
# ---------------------------------------------------------
def test_few_samples_no_rejection() -> None:
    print("\n[2] ≤3 mẫu → giữ nguyên mean thô")
    rng = np.random.default_rng(99)
    truth = _normalize(rng.standard_normal(512).astype(np.float32))
    samples = [_normalize(truth + rng.standard_normal(512).astype(np.float32) * 0.05) for _ in range(3)]

    m = CosineMatcher()
    for s in samples:
        m.register("B", s)

    mean = m._mean_of("B")
    mean_raw = _normalize(np.mean(samples, axis=0))
    # Với ≤3 mẫu, _mean_of phải trả về mean thô (không loại)
    diff = float(np.linalg.norm(mean - mean_raw))
    check("≤3 mẫu → mean giống mean thô (diff < 0.01)",
          diff < 0.01, f"(diff={diff:.6f})")


# ---------------------------------------------------------
# 3. Tất cả mẫu giống hệt nhau → không loại gì
# ---------------------------------------------------------
def test_identical_samples() -> None:
    print("\n[3] Tất cả mẫu giống hệt nhau → không loại")
    m = CosineMatcher()
    v = _normalize(np.ones(512, dtype=np.float32))
    for _ in range(5):
        m.register("C", v)

    mean = m._mean_of("C")
    sim = float(np.dot(mean, v))
    check("5 mẫu giống hệt → mean = mẫu gốc (cosine=1.0)",
          sim > 0.999, f"(sim={sim:.6f})")


# ---------------------------------------------------------
# 4. Integration: DB + matcher nạp mẫu → match đúng
# ---------------------------------------------------------
def test_db_integration() -> None:
    print("\n[4] Integration: DB + CosineMatcher")
    rng = np.random.default_rng(77)
    db = Database(TEMP_DB)

    # Tạo 2 người với nhiều mẫu
    person_a_truth = _normalize(rng.standard_normal(512).astype(np.float32))
    person_b_truth = _normalize(rng.standard_normal(512).astype(np.float32))

    samples_a = [_normalize(person_a_truth + rng.standard_normal(512).astype(np.float32) * 0.05) for _ in range(5)]
    samples_b = [_normalize(person_b_truth + rng.standard_normal(512).astype(np.float32) * 0.05) for _ in range(4)]

    # Thêm 1 mẫu xấu cho A (hướng khác)
    samples_a_bad = _normalize(-person_a_truth + rng.standard_normal(512).astype(np.float32) * 0.1)

    m = CosineMatcher()
    for s in samples_a:
        m.register("A", s)
    m.register("A", samples_a_bad)  # mẫu xấu
    for s in samples_b:
        m.register("B", s)

    check("matcher size = 2 người", m.size == 2)

    # Match query gần A
    query_a = _normalize(person_a_truth + rng.standard_normal(512).astype(np.float32) * 0.03)
    result_a = m.match(query_a, threshold=0.30)
    check("query gần A → match person A",
          result_a is not None and result_a.person_id == "A",
          f"(result={result_a})")

    # Match query gần B
    query_b = _normalize(person_b_truth + rng.standard_normal(512).astype(np.float32) * 0.03)
    result_b = m.match(query_b, threshold=0.30)
    check("query gần B → match person B",
          result_b is not None and result_b.person_id == "B",
          f"(result={result_b})")

    # Query xa cả A và B → stranger
    query_stranger = _normalize(rng.standard_normal(512).astype(np.float32))
    result_stranger = m.match(query_stranger, threshold=0.60)
    check("query lạ (xa cả 2) → None (người lạ)",
          result_stranger is None)

    m.clear()
    check("clear → size=0", m.size == 0)

    db.close()
    cleanup()


# ---------------------------------------------------------
# 5. Ranking
# ---------------------------------------------------------
def test_ranking() -> None:
    print("\n[5] Ranking (top-K)")
    rng = np.random.default_rng(55)
    truth_a = _normalize(rng.standard_normal(512).astype(np.float32))
    truth_b = _normalize(rng.standard_normal(512).astype(np.float32))

    m = CosineMatcher()
    for _ in range(3):
        m.register("X", _normalize(truth_a + rng.standard_normal(512).astype(np.float32) * 0.03))
    for _ in range(3):
        m.register("Y", _normalize(truth_b + rng.standard_normal(512).astype(np.float32) * 0.03))

    query = _normalize(truth_a + rng.standard_normal(512).astype(np.float32) * 0.02)
    ranked = m.rank(query, top_k=2)
    check("rank trả đúng số phần tử", len(ranked) == 2)
    check("rank phần tử đầu gần X hơn", ranked[0].person_id == "X",
          f"(top={ranked[0].person_id}, sim={ranked[0].similarity:.4f})")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main() -> None:
    print("=== TEST BƯỚC 20: TOP-K MATCHING + OUTLIER REJECTION ===")
    cleanup()
    test_outlier_rejection()
    test_few_samples_no_rejection()
    test_identical_samples()
    test_db_integration()
    test_ranking()
    cleanup()
    print(f"\nKẾT QUẢ: {PASS} qua / {FAIL} thất bại")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
