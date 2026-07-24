"""Airfoil coordinate loading -> normalized closed polygon.

Handles the two coordinate conventions found in ``Data/Airfoils``:

* **Selig**: one continuous loop, trailing edge -> upper -> leading edge ->
  lower -> trailing edge. ``x`` runs 1 -> 0 -> 1. (e.g. ``naca0006``.)
* **Lednicer**: a header count line (two numbers > 1, e.g. ``26.  26.``) then
  two blocks, each running LE (0) -> TE (1): upper first, then lower, often
  separated by a blank line. (e.g. ``n63010a``.)

Output is always a single closed polygon ordered TE -> upper -> LE -> lower ->
TE with ``x`` normalized to [0, 1] (chord = 1) and the leading edge at the
origin-ish; callers rotate it by angle of attack (see :func:`rotate`).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np


@dataclass
class Airfoil:
    name: str
    xy: np.ndarray  # (N, 2) closed polygon, TE->upper->LE->lower->TE

    @property
    def chord(self) -> float:
        return float(self.xy[:, 0].max() - self.xy[:, 0].min())

    @property
    def thickness(self) -> float:
        return float(self.xy[:, 1].max() - self.xy[:, 1].min())


def _floats(line: str) -> Tuple[float, ...]:
    out = []
    for tok in line.replace(",", " ").split():
        try:
            out.append(float(tok))
        except ValueError:
            return ()
    return tuple(out)


def _parse(path: Path) -> Tuple[str, List[List[Tuple[float, float]]]]:
    """Return (name, blocks) where blocks are runs of coord pairs.

    A "count line" (a pair with any value > 1.0, invalid as a normalized
    coordinate) or a blank line starts a new block.
    """
    name = path.stem
    blocks: List[List[Tuple[float, float]]] = [[]]
    for raw in path.read_text().splitlines():
        vals = _floats(raw)
        if not raw.strip():
            if blocks[-1]:
                blocks.append([])
            continue
        if len(vals) < 2:  # header / title text
            if not any(blocks) and raw.strip():
                name = raw.strip()
            continue
        x, y = vals[0], vals[1]
        if abs(x) > 1.5 or abs(y) > 1.5:  # Lednicer count line -> new block
            if blocks[-1]:
                blocks.append([])
            continue
        blocks[-1].append((x, y))
    return name, [b for b in blocks if b]


def _to_loop(blocks: List[List[Tuple[float, float]]]) -> np.ndarray:
    if len(blocks) == 1:
        pts = np.asarray(blocks[0], float)  # Selig: already a loop
    else:
        upper = np.asarray(blocks[0], float)  # Lednicer: LE->TE upper
        lower = np.asarray(blocks[1], float)  # LE->TE lower
        # TE -> LE (reverse upper) then LE -> TE (lower)
        loop = np.vstack([upper[::-1], lower[1:]])
        pts = loop
    # ensure closed
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    return pts


def load_dat(path: str | Path) -> Airfoil:
    path = Path(path)
    name, blocks = _parse(path)
    xy = _to_loop(blocks)
    # normalize chord to 1, shift LE (min-x point) to x=0
    x = xy[:, 0]
    c = x.max() - x.min()
    if c <= 0:
        raise ValueError(f"{path}: degenerate chord")
    xy = (xy - [x.min(), 0.0]) / c
    return Airfoil(name=name, xy=xy)


def rotate(af: Airfoil, alpha_deg: float, pivot: Tuple[float, float] = (0.25, 0.0)) -> Airfoil:
    """Rotate foil by +alpha (nose-up) about ``pivot`` (default quarter-chord).

    Freestream stays left->right, so rotating the geometry by -alpha in image
    space realizes a +alpha angle of attack relative to the flow.
    """
    a = np.deg2rad(-alpha_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    p = np.asarray(pivot, float)
    xy = (af.xy - p) @ R.T + p
    return Airfoil(name=af.name, xy=xy)
