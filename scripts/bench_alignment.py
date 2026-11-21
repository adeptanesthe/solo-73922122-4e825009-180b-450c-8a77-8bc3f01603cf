"""Work-reduction benchmark for the pairwise overlap scan.

Reproduces the workload from the task statement and compares the optimized
:func:`app.alignment.scan_pair` against a naive full per-pixel scan:

* two 1024x1024 grayscale tiles, each split into four equal black/white
  quadrants, with the second tile inverted at the same location;
* a 9x9 search window centred on the zero shift;
* minimum overlap 1,000,000 pixels and a MAD cap of 3.

Every window position has enough overlap, no offset passes, and the response
must still report the lowest mean difference over the whole window and the
corresponding overlap size.  The script prints both implementations' per-pixel
absolute-difference counts and asserts the optimized scan uses at most half as
many while returning the identical diagnostic.

Run with:  python scripts/bench_alignment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from app.alignment import scan_pair
from app.imaging import DecodedImage, overlap_rect

SIZE = 1024
WINDOW_RADIUS = 4
MIN_OVERLAP = 1_000_000
MAX_MAD = 3.0


def inverted_quadrant_tiles() -> tuple[np.ndarray, np.ndarray]:
    img = np.zeros((SIZE, SIZE), dtype=np.uint8)
    half = SIZE // 2
    img[:half, :half] = 255
    img[half:, half:] = 255  # two white and two black equal quadrants
    return img, 255 - img    # inverted at every shared location


def naive_scan(a: DecodedImage, b: DecodedImage, offsets, min_overlap, max_mad):
    a16 = a.array.astype(np.int16)
    b16 = b.array.astype(np.int16)
    evals = 0
    best_mad = best_off = None
    best_n = 0
    scored = 0
    for dx, dy in offsets:
        rect = overlap_rect(a.width, a.height, b.width, b.height, dx, dy)
        if rect is None:
            continue
        x0, y0, x1, y1 = rect
        n = (x1 - x0) * (y1 - y0)
        if n < min_overlap:
            continue
        scored += 1
        diff = np.abs(a16[y0:y1, x0:x1] - b16[y0 - dy : y1 - dy, x0 - dx : x1 - dx])
        evals += int(diff.size)
        mad = float(diff.sum(dtype=np.int64)) / n
        if best_mad is None or mad < best_mad:
            best_mad, best_off, best_n = mad, (dx, dy), n
    return evals, scored, best_off, best_mad, best_n


def main() -> int:
    img, inv = inverted_quadrant_tiles()
    a = DecodedImage(0, img, "a")
    b = DecodedImage(1, inv, "b")
    offsets = [
        (dx, dy)
        for dy in range(-WINDOW_RADIUS, WINDOW_RADIUS + 1)
        for dx in range(-WINDOW_RADIUS, WINDOW_RADIUS + 1)
    ]

    naive_evals, scored, n_off, n_mad, n_n = naive_scan(
        a, b, offsets, MIN_OVERLAP, MAX_MAD
    )
    scan = scan_pair(
        0, a, b, offsets, MIN_OVERLAP, MAX_MAD,
        a_i16=img.astype(np.int16), b_i16=inv.astype(np.int16),
    )

    print(f"scored valid positions : {scored}")
    print(f"accepted offsets       : {scan.accepted} (expected [])")
    print(f"best offset            : naive={n_off} optimized={scan.best_offset}")
    print(f"best overlap pixels    : naive={n_n} optimized={scan.best_overlap}")
    print(f"best MAD               : naive={n_mad:.12f}")
    print(f"                         optim={scan.best_mad:.12f}")
    print(f"naive abs-diff evals   : {naive_evals:,}")
    print(f"optimized abs-diff evals: {scan.abs_diff_evaluations:,}")
    reduction = 1 - scan.abs_diff_evaluations / naive_evals
    print(f"reduction              : {reduction:.1%}")
    print(f"half-of-naive cap      : {naive_evals // 2:,}")

    assert scan.scored_positions == scored == 81
    assert scan.accepted == []
    assert scan.best_offset == n_off
    assert scan.best_overlap == n_n
    assert abs(scan.best_mad - n_mad) <= 1e-9
    assert scan.abs_diff_evaluations <= naive_evals // 2, "work must be at least halved"
    print("OK: identical diagnostics, and per-pixel work is at least halved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
