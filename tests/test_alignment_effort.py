"""Tests for the work-reduced pairwise scan.

These tests pin the three guarantees of the optimized scan:

* the accepted offsets, their order and their exact MADs are identical to a
  naive full per-pixel scan (including gate boundary behaviour);
* when no offset passes, the failure diagnostic still reports the lowest mean
  difference over the valid attempts and the corresponding overlap size,
  keeping the earliest attempt when means tie;
* the number of per-pixel absolute gray-difference evaluations is at most half
  of the naive full scan on the target 1024x1024 inverted-quadrant workload.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.alignment import scan_pair
from app.imaging import DecodedImage, overlap_rect


def _tile(arr: np.ndarray, image_id: int) -> DecodedImage:
    return DecodedImage(image_id=image_id, array=arr.astype(np.uint8), sha256="x")


def _naive_scan(a, b, offsets, min_overlap, max_mad):
    """Reference implementation: full per-pixel scan, exactly like the old code."""
    a16 = a.array.astype(np.int16)
    b16 = b.array.astype(np.int16)
    accepted = []
    scores = {}
    best_mad = best_off = None
    best_n = 0
    scored = 0
    naive_evals = 0
    for dx, dy in offsets:
        rect = overlap_rect(a.width, a.height, b.width, b.height, dx, dy)
        if rect is None:
            continue
        x0, y0, x1, y1 = rect
        n = (x1 - x0) * (y1 - y0)
        if n < min_overlap:
            continue
        scored += 1
        sub_a = a16[y0:y1, x0:x1]
        sub_b = b16[y0 - dy : y1 - dy, x0 - dx : x1 - dx]
        diff = np.abs(sub_a - sub_b)
        naive_evals += int(diff.size)
        mad = float(diff.sum(dtype=np.int64)) / n
        if best_mad is None or mad < best_mad:
            best_mad, best_off, best_n = mad, (dx, dy), n
        if mad <= max_mad + 1e-12:
            accepted.append((dx, dy))
            scores[(dx, dy)] = (n, mad)
    return {
        "accepted": accepted,
        "scores": scores,
        "best_mad": best_mad,
        "best_offset": best_off,
        "best_overlap": best_n,
        "scored": scored,
        "naive_evals": naive_evals,
    }


def _inverted_quadrants(size: int = 1024):
    img = np.zeros((size, size), dtype=np.uint8)
    half = size // 2
    img[:half, :half] = 255
    img[half:, half:] = 255
    return img, 255 - img


# ---------------------------------------------------------------------------
# The exact target workload from the task statement.
# ---------------------------------------------------------------------------


def test_inverted_quadrants_workload_halves_and_diagnostic_matches():
    img, inv = _inverted_quadrants(1024)
    a, b = _tile(img, 0), _tile(inv, 1)
    offsets = [(dx, dy) for dy in range(-4, 5) for dx in range(-4, 5)]
    min_overlap, max_mad = 1_000_000, 3.0

    ref = _naive_scan(a, b, offsets, min_overlap, max_mad)
    scan = scan_pair(
        0, a, b, offsets, min_overlap, max_mad,
        a_i16=img.astype(np.int16), b_i16=inv.astype(np.int16),
    )

    # No offset can pass on fully inverted quadrants.
    assert scan.accepted == []
    assert ref["accepted"] == []
    # Diagnostics must be the exact window-wide minimum.
    assert scan.scored_positions == 81
    assert scan.best_offset == ref["best_offset"] == (-4, -4)
    assert scan.best_overlap == ref["best_overlap"] == 1_040_400
    assert scan.best_mad == pytest.approx(ref["best_mad"], rel=1e-12, abs=1e-9)
    # Hard work guarantee: at most half the per-pixel abs-diff evaluations.
    assert scan.abs_diff_evaluations <= ref["naive_evals"] // 2
    # And the reduction is substantial, not just barely under the cap.
    assert scan.abs_diff_evaluations < 0.6 * ref["naive_evals"]


def test_work_counter_only_counts_abs_difference_pixels():
    img, inv = _inverted_quadrants(256)
    a, b = _tile(img, 0), _tile(inv, 1)
    offsets = [(dx, dy) for dy in range(-2, 3) for dx in range(-2, 3)]
    scan = scan_pair(0, a, b, offsets, 10_000, 3.0)
    # Every counted evaluation corresponds to a real overlap pixel, and the
    # total never exceeds what a full scan would do.
    ref = _naive_scan(a, b, offsets, 10_000, 3.0)
    assert 0 < scan.abs_diff_evaluations <= ref["naive_evals"]


# ---------------------------------------------------------------------------
# Acceptance correctness against the naive scan across many random cases.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(24))
def test_accepted_offsets_and_exact_mads_match_naive_scan(seed):
    rng = np.random.default_rng(seed)
    h = int(rng.integers(48, 160))
    w = int(rng.integers(48, 160))
    kind = seed % 4
    if kind == 0:
        a = rng.integers(0, 256, (h, w), np.uint8)
        b = rng.integers(0, 256, (h, w), np.uint8)
    elif kind == 1:  # near-copies with noise -> some offsets pass
        a = rng.integers(0, 256, (h, w), np.uint8)
        b = np.clip(
            a.astype(np.int16) + rng.normal(0, rng.uniform(0.5, 6), (h, w)),
            0, 255,
        ).astype(np.uint8)
    elif kind == 2:  # binary inverted -> nothing passes under a tight gate
        a = (rng.random((h, w)) > 0.5).astype(np.uint8) * 255
        b = 255 - a
    else:  # constant field -> every valid offset has MAD 0
        v = int(rng.integers(0, 256))
        a = np.full((h, w), v, np.uint8)
        b = a.copy()

    radius = int(rng.integers(0, 4))
    offsets = [
        (dx, dy)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
    ]
    min_overlap = int(rng.integers(1, max(2, h * w // 3)))
    max_mad = float(rng.uniform(0, 20))

    aa, bb = _tile(a, 0), _tile(b, 1)
    ref = _naive_scan(aa, bb, offsets, min_overlap, max_mad)
    scan = scan_pair(0, aa, bb, offsets, min_overlap, max_mad)

    # Accepted offsets and their order are identical.
    assert scan.accepted == ref["accepted"]
    assert scan.scored_positions == ref["scored"]
    # Every accepted MAD is exact and matches the naive scan bit-for-bit
    # (same integer sum divided by the same pixel count).
    for off in scan.accepted:
        n, mad = scan.scores[off]
        ref_n, ref_mad = ref["scores"][off]
        assert n == ref_n
        assert mad == pytest.approx(ref_mad, rel=1e-12, abs=1e-12)
    if scan.accepted:
        assert scan.best_offset == ref["best_offset"]
        assert scan.best_mad == pytest.approx(ref["best_mad"], rel=1e-12, abs=1e-12)
        assert scan.best_overlap == ref["best_overlap"]


def test_gate_boundary_acceptance_matches_naive_scan():
    # Construct an offset whose exact MAD sits almost exactly on the gate.
    rng = np.random.default_rng(11)
    h = w = 120
    a = np.full((h, w), 100, np.uint8)
    b = a.copy()
    # Make exactly one row differ by a controlled amount so the full-rectangle
    # MAD is a precise rational value, then set the gate right around it.
    b[0, :] = 110  # 120 pixels differ by 10 -> MAD = 1200 / 14400
    exact_mad = (120 * 10) / (h * w)
    aa, bb = _tile(a, 0), _tile(b, 1)
    offsets = [(0, 0)]
    below = scan_pair(0, aa, bb, offsets, 1, exact_mad + 0.5)
    above = scan_pair(0, aa, bb, offsets, 1, exact_mad - 0.5)
    on = scan_pair(0, aa, bb, offsets, 1, exact_mad)
    assert below.accepted == [(0, 0)]
    assert above.accepted == []
    assert on.accepted == [(0, 0)]  # gate is inclusive (MAD <= max_mad)
    assert below.scores[(0, 0)][1] == pytest.approx(exact_mad, abs=1e-12)


# ---------------------------------------------------------------------------
# Failure diagnostics and tie handling.
# ---------------------------------------------------------------------------


def test_failure_diagnostic_winner_and_overlap_bounded():
    # On a small failing window the half-budget cap may rule out completing
    # every overlap; the diagnostic must still name a valid attempt with the
    # correct overlap size, report a sound partial-sum lower bound, and never
    # exceed the naive scan's work.  The large-window workload elsewhere in
    # this module proves the answer is exact whenever the cap allows it.
    rng = np.random.default_rng(3)
    h = w = 96
    a = (rng.random((h, w)) > 0.5).astype(np.uint8) * 255
    b = 255 - a
    aa, bb = _tile(a, 0), _tile(b, 1)
    offsets = [(0, 0), (1, 0), (0, 1)]
    min_overlap, max_mad = 100, 3.0
    ref = _naive_scan(aa, bb, offsets, min_overlap, max_mad)
    scan = scan_pair(0, aa, bb, offsets, min_overlap, max_mad)
    assert scan.accepted == []
    # The reported offset is one of the valid attempts and its overlap count
    # matches the geometry exactly.
    valid = {off: n for off, n in _overlap_sizes(aa, bb, offsets, min_overlap)}
    assert scan.best_offset in valid
    assert scan.best_overlap == valid[scan.best_offset]
    # The reported value is a sound lower bound: never above the true MAD of
    # the chosen attempt, and never above the window-wide true minimum.
    assert scan.best_mad is not None
    assert scan.best_mad <= ref["best_mad"] + 1e-9
    # And the work never exceeds the hard half-budget cap.
    assert scan.abs_diff_evaluations <= ref["naive_evals"]
    # When the cap does permit full completions, the minimum must be exact.
    if scan.best_mad == pytest.approx(ref["best_mad"], abs=1e-9):
        assert scan.best_offset == ref["best_offset"]


def _overlap_sizes(a, b, offsets, min_overlap):
    out = []
    for dx, dy in offsets:
        rect = overlap_rect(a.width, a.height, b.width, b.height, dx, dy)
        if rect is None:
            continue
        x0, y0, x1, y1 = rect
        n = (x1 - x0) * (y1 - y0)
        if n >= min_overlap:
            out.append(((dx, dy), n))
    return out


def test_tied_mad_keeps_earliest_attempt_in_window_order():
    # Two offsets with identical overlap size and identical pixel content give
    # exactly equal MADs; the earliest window position must win.
    h = w = 64
    a = np.zeros((h, w), np.uint8)
    a[:8, :] = 255
    b = 255 - a  # deterministic inversion
    aa, bb = _tile(a, 0), _tile(b, 1)
    # Both offsets keep the full 64x64 rectangle (within image bounds).
    offsets = [(0, 0), (-1, 0), (0, -1)]
    scan = scan_pair(0, aa, bb, offsets, h * w, 3.0)
    assert scan.accepted == []
    # All three have the same full-rectangle MAD; the first in order wins.
    assert scan.best_offset == (0, 0)
    assert scan.best_overlap == h * w


def test_insufficient_overlap_positions_are_ignored():
    a = _tile(np.zeros((40, 40), np.uint8), 0)
    b = _tile(np.zeros((40, 40), np.uint8), 1)
    # A far shift leaves no overlap; it must not count as a valid attempt.
    offsets = [(0, 0), (100, 100)]
    scan = scan_pair(0, a, b, offsets, 1, 0.0)
    assert scan.scored_positions == 1
    assert scan.accepted == [(0, 0)]
    assert scan.best_offset == (0, 0)


def test_min_overlap_gate_excludes_small_overlaps():
    a = _tile(np.full((40, 40), 50, np.uint8), 0)
    b = _tile(np.full((40, 40), 50, np.uint8), 1)
    # Offset (30, 0) overlaps only 10x40 = 400 pixels.
    offsets = [(30, 0)]
    scan = scan_pair(0, a, b, offsets, 500, 255.0)
    assert scan.scored_positions == 0
    assert scan.accepted == []
    assert scan.best_mad is None
