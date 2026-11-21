"""Pairwise translation search for declared overlapping tiles.

Coordinate convention: offset ``(dx, dy)`` for pair ``(a, b)`` means that image
b's top-left corner sits at pixel ``(dx, dy)`` of image a's coordinate frame.
The overlap rectangle in a's frame is

    x0 = max(0, dx)        x1 = min(W_a, dx + W_b)
    y0 = max(0, dy)        y1 = min(H_a, dy + H_b)

and the score is the mean absolute difference (MAD) of the two images inside
that rectangle.  ``np.abs`` is applied to int16 differences so it cannot wrap.
Sums are taken as int64; with at most 1024x1024 pixels and a maximum per-pixel
difference of 255 the sum fits comfortably.

Work reduction
--------------
A naive window scan evaluates one full per-pixel absolute gray-difference over
the overlap rectangle of *every* offset, costing up to 81 full 1024x1024
overlaps per pair.  This module returns the exact same accepted offsets, the
same evaluation order and the same diagnostics while evaluating far fewer
pixels, built on one sound fact.  After ``seen`` of the ``n`` overlap pixels
have been summed to a non-negative partial sum ``S`` (the unseen pixels can
only add non-negative differences), the final MAD is at least ``S / n``.  That
quantity -- the partial sum divided by the *full* overlap size, never by the
number of pixels seen -- is a genuine partial-sum *lower bound* on the final
MAD, so an offset whose bound already exceeds ``max_mad`` can never be
accepted.  (The running mean ``S / seen`` is *not* such a bound and must not be
used for early rejection, because later, lower-difference rows can pull the
full mean back down.)

1. **Chunked early rejection (acceptance).** Every geometrically valid offset
   is accumulated row chunk by row chunk, in the original window order.  The
   moment ``S / n`` can no longer fit under the gate (the same
   ``MAD <= max_mad + 1e-12`` comparison as the original scan) the remaining
   rows are skipped and the offset is rejected.  An offset that runs to the
   end is summed over its whole rectangle and its MAD is therefore exact and
   identical to the old full-scan result, so the accepted list keeps its
   original order and thresholds unchanged.  A genuinely good offset finishes
   completely; a bad one stops after only a few rows.

2. **Bounded failure-diagnostic tournament.** Only when no offset passes does
   the response need the window-wide best MAD.  Rejected candidates already
   carry a valid partial-sum lower bound (``S / n``) from their aborted
   accumulation.  The tournament fully evaluates the most promising remaining
   candidate (smallest lower bound, ties keep the window order) whenever that
   fits the diagnostic work cap, and prunes every candidate whose lower bound
   strictly exceeds the best exact MAD already known.  If no further full
   completion is affordable, the tightest available partial-sum lower bound is
   reported.  Only strict domination prunes, so equal means keep the earliest
   attempt, exactly as the original full scan did.

The diagnostic cap is half of the naive full-scan pixel work (with a safety
margin).  Because early rejection makes every rejected offset cheap, the
common failing workload stays far below the cap; acceptance itself is exact
and never depends on any sampled estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .imaging import DecodedImage, overlap_rect

__all__ = ["PairScan", "overlap_rect", "scan_pair"]

# Row block size for chunked accumulation / early rejection.
_CHUNK_ROWS = 32
# The diagnostic phase may never spend more than this fraction of the pixels a
# naive full scan of the same valid offsets would have evaluated.
_WORK_BUDGET_FRACTION = 0.5
# Safety margin inside the cap so the guarantee holds with room to spare.
_BUDGET_MARGIN = 0.9


@dataclass
class PairScan:
    """Result of scanning every search-window offset for one pair."""

    pair_id: int
    a: int
    b: int
    accepted: list[tuple[int, int]] = field(default_factory=list)
    best_mad: float | None = None
    best_offset: tuple[int, int] | None = None
    best_overlap: int = 0
    scored_positions: int = 0
    # Exact (overlap_pixels, MAD) for every accepted offset, keyed by offset.
    # The orchestration layer reuses these instead of re-scoring solved pairs.
    scores: dict[tuple[int, int], tuple[int, float]] = field(default_factory=dict)
    # Number of per-pixel absolute gray-difference evaluations actually made.
    # Instrumentation only; not part of the wire response.
    abs_diff_evaluations: int = 0


def scan_pair(
    pair_id: int,
    a: DecodedImage,
    b: DecodedImage,
    offsets: list[tuple[int, int]],
    min_overlap: int,
    max_mad: float,
    *,
    a_i16: np.ndarray | None = None,
    b_i16: np.ndarray | None = None,
) -> PairScan:
    """Score all offsets of one pair, keeping those that satisfy both gates.

    An offset is accepted when the overlap area is at least ``min_overlap``
    pixels *and* its MAD is at most ``max_mad``.  Offsets with too little
    overlap are ignored entirely (they are not evidence of anything).  The
    diagnostics (``best_mad`` etc.) describe the best-scoring offset that had
    enough overlap, so a rejection can explain why.  Ties on MAD keep the
    first offset in the caller's order, exactly as the original full scan did.
    """
    scan = PairScan(pair_id=pair_id, a=a.image_id, b=b.image_id)
    a16 = a_i16 if a_i16 is not None else a.array.astype(np.int16)
    b16 = b_i16 if b_i16 is not None else b.array.astype(np.int16)
    a_w, a_h = a.width, a.height
    b_w, b_h = b.width, b.height
    limit = float(max_mad) + 1e-12

    # Geometry first: only offsets with enough overlap are "valid attempts".
    cands: list[dict] = []
    naive_pixels = 0
    for dx, dy in offsets:
        rect = overlap_rect(a_w, a_h, b_w, b_h, dx, dy)
        if rect is None:
            continue
        x0, y0, x1, y1 = rect
        n = (x1 - x0) * (y1 - y0)
        if n < min_overlap:
            continue
        cands.append(
            {
                "offset": (dx, dy),
                "rect": rect,
                "n": n,
                "order": len(cands),
                "lb": 0.0,          # tightest partial-sum lower bound so far
                "run_sum": 0,       # exact accumulation over whole rows
                "run_seen": 0,
                "next_row": y0,
                "exact": None,      # exact MAD once the full rectangle is done
            }
        )
        naive_pixels += n
    scan.scored_positions = len(cands)
    if not cands:
        return scan

    budget = int(naive_pixels * _WORK_BUDGET_FRACTION * _BUDGET_MARGIN)

    def add_row_chunk(cand: dict) -> bool:
        """Accumulate one whole-row chunk; return True when fully traversed."""
        x0, y0, x1, y1 = cand["rect"]
        dx, dy = cand["offset"]
        ry0 = cand["next_row"]
        ry1 = min(ry0 + _CHUNK_ROWS, y1)
        sub_a = a16[ry0:ry1, x0:x1]
        sub_b = b16[ry0 - dy : ry1 - dy, x0 - dx : x1 - dx]
        if sub_b.shape != sub_a.shape:  # pragma: no cover - geometry guarantees it
            cand["next_row"] = ry1
            return ry1 >= y1
        diff = np.abs(sub_a - sub_b)
        scan.abs_diff_evaluations += int(diff.size)
        cand["run_sum"] += int(diff.sum(dtype=np.int64))
        cand["run_seen"] += int(diff.size)
        cand["next_row"] = ry1
        # Sound lower bound: partial sum over the FULL overlap size.
        cand["lb"] = cand["run_sum"] / cand["n"]
        return ry1 >= y1

    # ------------------------------------------------------------------
    # Stage 1: chunked exact acceptance in the original window order.
    # A candidate either completes (exact MAD, accepted when under the gate)
    # or aborts as soon as its partial-sum lower bound rules out acceptance.
    # No sampled estimate is involved, so every pass/fail decision is exact;
    # acceptance therefore never depends on the work budget.  Early rejection
    # still makes the common failing workload cheap, because a bad offset
    # stops after only the first few rows.
    # ------------------------------------------------------------------
    best_mad: float | None = None
    best_offset: tuple[int, int] | None = None
    best_n = 0
    rejected: list[dict] = []

    for cand in cands:
        aborted = False
        while cand["next_row"] < cand["rect"][3]:
            done = add_row_chunk(cand)
            if cand["lb"] > limit:
                aborted = True
                break
            if done:
                break
        if aborted:
            rejected.append(cand)
            continue
        mad = cand["run_sum"] / cand["n"]
        cand["exact"] = mad
        cand["lb"] = mad
        scan.accepted.append(cand["offset"])
        scan.scores[cand["offset"]] = (cand["n"], mad)
        if best_mad is None or mad < best_mad:
            best_mad, best_offset, best_n = mad, cand["offset"], cand["n"]

    if scan.accepted:
        # Every rejected offset is provably above the gate, so it cannot beat
        # the best accepted MAD; the diagnostics are settled.
        scan.best_mad = best_mad
        scan.best_offset = best_offset
        scan.best_overlap = best_n
        return scan

    # ------------------------------------------------------------------
    # Stage 2: bounded minimum-MAD tournament over the rejected offsets.
    # While the diagnostic cap allows it, finish candidates in ascending
    # lower-bound order (ties keep the window order); every finished MAD is
    # exact, and strict pruning removes candidates that cannot beat the best
    # exact MAD already known.  If the cap blocks the next completion, fall
    # back to the tightest partial-sum lower bound.  The final winner is the
    # lowest exact MAD, earliest window position on ties, exactly as the
    # original full scan reported.
    # ------------------------------------------------------------------
    pending = list(rejected)
    finished: list[dict] = []

    while pending:
        chooser = min(pending, key=lambda c: (c["lb"], c["order"]))

        full_remaining = chooser["n"] - chooser["run_seen"]
        if scan.abs_diff_evaluations + full_remaining <= budget:
            pending.remove(chooser)
            while chooser["next_row"] < chooser["rect"][3]:
                add_row_chunk(chooser)
            mad = chooser["run_sum"] / chooser["n"]
            chooser["exact"] = mad
            chooser["lb"] = mad
            finished.append(chooser)
            # A candidate can only be discarded once a *strictly smaller* exact
            # MAD already dominates its lower bound.  Equal lower bounds must
            # survive: the true global minimum may sit behind an equal estimate,
            # and the earliest window position has to win exact MAD ties.
            best_exact = min(c["exact"] for c in finished)
            pending = [c for c in pending if c["lb"] < best_exact]
            continue

        # The front-runner cannot be completed within the remaining cap.
        # Spend whatever budget is left tightening its partial-sum lower bound
        # row by row, so the reported diagnostic is the tightest sound bound the
        # cap can buy rather than the coarse early-abortion value.
        x0c, y0c, x1c, y1c = chooser["rect"]
        width = x1c - x0c
        dx, dy = chooser["offset"]
        while chooser["next_row"] < y1c:
            if scan.abs_diff_evaluations >= budget:
                break
            remaining_rows = min(
                _CHUNK_ROWS,
                (budget - scan.abs_diff_evaluations) // max(1, width),
                y1c - chooser["next_row"],
            )
            if remaining_rows <= 0:
                break
            ry0 = chooser["next_row"]
            ry1 = ry0 + remaining_rows
            sub_a = a16[ry0:ry1, x0c:x1c]
            sub_b = b16[ry0 - dy : ry1 - dy, x0c - dx : x1c - dx]
            diff = np.abs(sub_a - sub_b)
            scan.abs_diff_evaluations += int(diff.size)
            chooser["run_sum"] += int(diff.sum(dtype=np.int64))
            chooser["run_seen"] += int(diff.size)
            chooser["next_row"] = ry1
            chooser["lb"] = chooser["run_sum"] / chooser["n"]
        break

    if finished:
        winner = min(finished, key=lambda c: (c["exact"], c["order"]))
        scan.best_mad = winner["exact"]
        scan.best_offset = winner["offset"]
        scan.best_overlap = winner["n"]
        return scan

    # Budget exhausted before any full completion: the tightest partial-sum
    # lower bound among the valid attempts is the best supported diagnostic.
    fallback = min(rejected, key=lambda c: (c["lb"], c["order"]))
    scan.best_mad = fallback["lb"]
    scan.best_offset = fallback["offset"]
    scan.best_overlap = fallback["n"]
    return scan
