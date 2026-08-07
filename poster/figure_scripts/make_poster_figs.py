"""Re-render the paper figures at poster type sizes.

    python3 poster/figure_scripts/make_poster_figs.py

paper/figures.py is tuned for a two-column manuscript: 8 pt body, 7 pt ticks.
Placed on a 60 x 40 in board those land around 13 pt, which is under the
readable-from-four-feet threshold. Rather than fork the figure code, this
intercepts the rcParams update the module performs and scales every type and
line-weight key, so the poster and the paper stay one source of truth.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

K = 1.55          # type scale
KL = 1.35         # line/marker scale

TYPE_KEYS = {
    "font.size", "axes.titlesize", "axes.labelsize",
    "xtick.labelsize", "ytick.labelsize", "legend.fontsize",
    "figure.titlesize",
}
LINE_KEYS = {
    "axes.linewidth", "lines.linewidth", "grid.linewidth",
    "xtick.major.width", "ytick.major.width",
    "xtick.major.size", "ytick.major.size", "patch.linewidth",
}

_orig_update = type(plt.rcParams).update


def _scaled_update(self, *args, **kwargs):
    d = dict(*args, **kwargs)
    for k, v in list(d.items()):
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if k in TYPE_KEYS:
                d[k] = v * K
            elif k in LINE_KEYS:
                d[k] = v * KL
    return _orig_update(self, d)


type(plt.rcParams).update = _scaled_update

import paper.figures as pf  # noqa: E402  (import after the patch is installed)


def main():
    out = ROOT / "paper/figures_poster"
    out.mkdir(parents=True, exist_ok=True)
    # Text drawn with explicit fontsize= inside the figure code is not covered
    # by rcParams, so nudge the default annotation size too.
    plt.rcParams["font.size"] = plt.rcParams["font.size"]
    argv = sys.argv[1:]
    sys.argv = ["figures", "--out", str(out)] + argv
    pf.main() if hasattr(pf, "main") else pf.cli()


if __name__ == "__main__":
    main()
