"""Dataset plumbing for the U-Net: memmap cache, splits, normalisation.

The generator writes one compressed ``.npz`` per case, which is the right
archival format and the wrong training format -- decompressing 1,926 files every
epoch costs more than the forward pass. :func:`build_cache` flattens the whole
dataset into a handful of uncompressed ``.npy`` memmaps once, after which an
epoch is pure page-cache reads.

The cache also stores the input tensor. The SDF is a pure function of geometry
and incidence so it *could* be rebuilt on the fly, but at ~35 ms a case that is
a minute per epoch spent recomputing a constant.

Cache layout (N = number of cases, H x W = grid):

    inputs.npy   (N, 4, H, W) float16
    targets.npy  (N, 5, H, W) float16
    masks.npy    (N, 2, H, W) uint8      [fluid, solid]
    scalars.npy  (N, 5)       float32    [cl, cd, cm, w_max, twist_te_deg]
    done.npy     (N,)         bool       resume marker
    index.json                           case order, splits, grid

Building is resumable: ``done.npy`` marks completed rows, so a run interrupted
by a wall-clock limit picks up where it stopped.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .geometry import load_dat
from .sdf import Grid, input_tensor
from .simulate import CasePrep, SCALAR_NAMES

CACHE_FILES = {
    "inputs": ("inputs.npy", np.float16, 4),
    "targets": ("targets.npy", np.float16, 5),
    "masks": ("masks.npy", np.uint8, 2),
}


def read_manifest(manifest: str | Path) -> List[dict]:
    rows = [
        json.loads(line)
        for line in Path(manifest).read_text().splitlines()
        if line.strip()
    ]
    return [r for r in rows if r.get("ok")]


def build_cache(
    manifest: str | Path = "artifacts/cfd/manifest.jsonl",
    splits_path: str | Path = "artifacts/cfd/splits.json",
    out_dir: str | Path = "artifacts/sensor/cache",
    grid: Optional[Grid] = None,
    time_budget_s: Optional[float] = None,
    progress_every: int = 200,
) -> dict:
    """Materialise the memmap cache. Safe to call repeatedly to resume."""
    grid = grid or Grid()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(manifest)
    rows.sort(key=lambda r: (r["foil"], r["alpha_deg"]))
    n = len(rows)

    index_path = out / "index.json"
    fidelity_names = sorted({(r.get("fidelity") or "panel") for r in rows})
    fidelity_codes = {name: i for i, name in enumerate(fidelity_names)}
    def _mtime(path: str) -> float:
        try:
            return round(Path(path).stat().st_mtime, 3)
        except OSError:
            return 0.0

    index = {
        "cases": [{"foil": r["foil"], "alpha_deg": r["alpha_deg"], "npz": r["npz"],
                   "series": r["series"], "dat": r["dat"],
                   "fidelity": r.get("fidelity") or "panel",
                   "mtime": _mtime(r["npz"])} for r in rows],
        "splits": json.loads(Path(splits_path).read_text()),
        "grid": {"nx": grid.nx, "ny": grid.ny, "xlim": list(grid.xlim), "ylim": list(grid.ylim)},
        "scalar_names": list(SCALAR_NAMES),
        "fidelity_codes": fidelity_codes,
    }
    # The `done` flags are positional. If the manifest changed -- rows added, or
    # a (foil, alpha) upgraded from panel to RANS and now pointing at a
    # different .npz -- those flags refer to different cases and the cache would
    # silently keep serving stale targets. Compare the case list and start over
    # when it moved. (Symptom when this was missing: a RANS sweep merged, then
    # `sensor-cache` reported "cached 1926, remaining 0, seconds 0.0" and every
    # array still held the panel data.)
    stale = True
    changed_rows: List[int] = []
    if index_path.exists():
        try:
            old = json.loads(index_path.read_text())
            old_cases = old.get("cases", [])
            key = lambda c: (c["foil"], c["alpha_deg"], c.get("npz"))
            stale = [key(c) for c in old_cases] != [key(c) for c in index["cases"]]
            if not stale:
                # Same cases in the same order, but a case can have been
                # *re-solved in place*: identical foil, angle and path, new
                # contents. Comparing the list alone missed that entirely --
                # re-running two cases to fix a biased Cd left the cache serving
                # the old arrays and `build_cache` reporting "seconds: 0.0".
                changed_rows = [
                    i for i, (o, c) in enumerate(zip(old_cases, index["cases"]))
                    if o.get("mtime") != c.get("mtime")
                ]
        except Exception:
            stale = True
    index_path.write_text(json.dumps(index))
    if stale:
        print(f"case list changed -- rebuilding cache for {n} cases", flush=True)

    mode_for = lambda path: "w+" if (stale or not path.exists()) else "r+"

    maps = {}
    for key, (fname, dtype, ch) in CACHE_FILES.items():
        path = out / fname
        maps[key] = np.lib.format.open_memmap(
            path, mode=mode_for(path), dtype=dtype, shape=(n, ch, grid.ny, grid.nx)
        )
    sc_path = out / "scalars.npy"
    scalars = np.lib.format.open_memmap(
        sc_path, mode=mode_for(sc_path), dtype=np.float32, shape=(n, 5)
    )
    done_path = out / "done.npy"
    done = np.lib.format.open_memmap(
        done_path, mode=mode_for(done_path), dtype=bool, shape=(n,)
    )
    if stale:
        done[:] = False
    elif changed_rows:
        print(
            f"{len(changed_rows)} case(s) re-solved since the last cache build "
            f"-- refreshing them", flush=True,
        )
        for i in changed_rows:
            done[i] = False
    fid_path = out / "fidelity.npy"
    fidelity = np.lib.format.open_memmap(
        fid_path, mode=mode_for(fid_path), dtype=np.int8, shape=(n,)
    )
    for i, r in enumerate(rows):
        fidelity[i] = fidelity_codes[r.get("fidelity") or "panel"]
    fidelity.flush()

    t0 = time.time()
    prep_cache: Dict[str, CasePrep] = {}
    built = 0
    for i, r in enumerate(rows):
        if done[i]:
            continue
        if time_budget_s is not None and time.time() - t0 > time_budget_s:
            break
        with np.load(r["npz"]) as z:
            tgt = z["targets"] if "targets" in z else z["target"]
            maps["targets"][i] = tgt.astype(np.float16)
            maps["masks"][i, 0] = z["mask"]
            maps["masks"][i, 1] = z["solid_mask"]
            scalars[i] = z["scalars"]
        prep = prep_cache.get(r["dat"])
        if prep is None:
            prep = CasePrep.build(load_dat(r["dat"]))
            prep_cache = {r["dat"]: prep}  # foils are contiguous; keep one
        maps["inputs"][i] = input_tensor(
            prep.airfoil, r["alpha_deg"], r["reynolds"] if "reynolds" in r else 5e5, grid
        ).astype(np.float16)
        done[i] = True
        built += 1
        if progress_every and built % progress_every == 0:
            print(f"  cached {int(done.sum())}/{n}", flush=True)

    for m in maps.values():
        m.flush()
    scalars.flush()
    done.flush()
    return {
        "cases": n,
        "cached": int(done.sum()),
        "remaining": int(n - done.sum()),
        "seconds": round(time.time() - t0, 1),
        "dir": str(out),
    }


class SensorCache:
    """Read-only view over the memmap cache with split-aware indexing."""

    def __init__(self, cache_dir: str | Path = "artifacts/sensor/cache"):
        self.dir = Path(cache_dir)
        self.index = json.loads((self.dir / "index.json").read_text())
        self.cases = self.index["cases"]
        self.splits = self.index["splits"]
        self.scalar_names = self.index.get("scalar_names", list(SCALAR_NAMES))
        self.inputs = np.load(self.dir / "inputs.npy", mmap_mode="r")
        self.targets = np.load(self.dir / "targets.npy", mmap_mode="r")
        self.masks = np.load(self.dir / "masks.npy", mmap_mode="r")
        self.scalars = np.load(self.dir / "scalars.npy", mmap_mode="r")
        self.done = np.load(self.dir / "done.npy", mmap_mode="r")
        self.fidelity_codes = self.index.get("fidelity_codes", {"panel": 0})
        self.code_to_fidelity = {v: k for k, v in self.fidelity_codes.items()}
        fid_path = self.dir / "fidelity.npy"
        # Caches built before mixed fidelity existed have no such file; treat
        # every case in them as panel rather than refusing to load.
        self.fidelity = (
            np.load(fid_path, mmap_mode="r")
            if fid_path.exists()
            else np.zeros(len(self.cases), dtype=np.int8)
        )

    def indices(self, split: str, fidelity: Optional[str] = None) -> np.ndarray:
        """Cache rows for a split, optionally restricted to one fidelity."""
        foils = set(self.splits[split])
        code = self.fidelity_codes.get(fidelity) if fidelity else None
        idx = [
            i for i, c in enumerate(self.cases)
            if c["foil"] in foils and self.done[i]
            and (code is None or int(self.fidelity[i]) == code)
        ]
        return np.asarray(idx, dtype=np.int64)

    def fidelity_of(self, i: int) -> str:
        return self.code_to_fidelity.get(int(self.fidelity[i]), "panel")

    def fidelity_counts(self) -> Dict[str, int]:
        return {
            name: int((np.asarray(self.fidelity) == code).sum())
            for name, code in self.fidelity_codes.items()
        }

    def __len__(self) -> int:
        return len(self.cases)


def _guard_std(std: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Replace the standard deviation of an effectively constant channel with 1.

    The Reynolds channel is constant across the base sweep, so its measured
    standard deviation is zero. Dividing by ``std + 1e-6`` then multiplies
    whatever tiny difference exists between two representations of the same
    number by a million. That is not hypothetical: the cache stores inputs as
    float16, so the mean is a rounded value, and at inference time the channel
    is computed in float32 -- a 1e-4 rounding difference became an input of
    +/-100, far outside anything the network saw in training, and the predicted
    coefficients came out constant and wrong while the cached-input evaluation
    still scored R^2 = 0.99.

    Setting the scale to 1 leaves a constant channel as a harmless constant
    offset instead.
    """
    std = np.asarray(std, dtype=np.float64).copy()
    scale = np.maximum(np.abs(np.asarray(mean, dtype=np.float64)), 1.0)
    constant = std < 1e-4 * scale
    std[constant] = 1.0
    std[~constant] = std[~constant] + 1e-8
    return std


def compute_norm_stats(cache: SensorCache, split: str = "train") -> dict:
    """Per-channel mean/std over the training split only.

    Statistics are taken over *valid* pixels only: averaging the fluid channels
    over the solid interior (where they are identically zero) would shrink the
    apparent variance and mis-scale the loss.
    """
    idx = cache.indices(split)
    if idx.size == 0:
        raise ValueError(f"split '{split}' is empty -- build the cache first")
    sample = idx if idx.size <= 400 else np.random.default_rng(0).choice(idx, 400, False)

    x = np.asarray(cache.inputs[np.sort(sample)], dtype=np.float32)
    y = np.asarray(cache.targets[np.sort(sample)], dtype=np.float32)
    m = np.asarray(cache.masks[np.sort(sample)], dtype=np.float32)
    s = np.asarray(cache.scalars[np.sort(sample)], dtype=np.float32)

    fluid = m[:, 0:1]
    solid = m[:, 1:2]
    valid = np.concatenate([fluid.repeat(3, axis=1), solid.repeat(2, axis=1)], axis=1)

    def _masked_stats(arr, mask):
        w = mask.sum(axis=(0, 2, 3)) + 1e-8
        mean = (arr * mask).sum(axis=(0, 2, 3)) / w
        var = ((arr - mean[None, :, None, None]) ** 2 * mask).sum(axis=(0, 2, 3)) / w
        return mean, np.sqrt(var) + 1e-6

    y_mean, y_std = _masked_stats(y, valid)
    x_mean = x.mean(axis=(0, 2, 3))
    x_std = _guard_std(x.std(axis=(0, 2, 3)), x_mean)
    y_std = _guard_std(y_std, y_mean)
    return {
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
        "s_mean": s.mean(axis=0).tolist(),
        "s_std": (s.std(axis=0) + 1e-6).tolist(),
        "n_samples": int(len(sample)),
    }


def make_torch_dataset(
    cache: SensorCache,
    split: str,
    stats: dict,
    fidelity_weights: Optional[Dict[str, float]] = None,
):
    """Build a ``torch.utils.data.Dataset``; imported lazily so NumPy-only
    consumers of this module never pay for a torch import.

    ``fidelity_weights`` maps a fidelity name to a per-sample loss weight, e.g.
    ``{"panel": 1.0, "rans": 10.0}``. A hundred RANS cases alongside 1,926 panel
    cases contribute 5% of the gradient at equal weight, which is not enough to
    move the network off the cheap targets; up-weighting is how the expensive
    ones actually count. Leave it ``None`` for uniform weighting.
    """
    import torch
    from torch.utils.data import Dataset

    idx = cache.indices(split)
    weights = np.ones(len(idx), dtype=np.float32)
    if fidelity_weights:
        for k, j in enumerate(idx):
            weights[k] = float(fidelity_weights.get(cache.fidelity_of(int(j)), 1.0))
    x_mean = torch.tensor(stats["x_mean"], dtype=torch.float32)[:, None, None]
    x_std = torch.tensor(stats["x_std"], dtype=torch.float32)[:, None, None]
    y_mean = torch.tensor(stats["y_mean"], dtype=torch.float32)[:, None, None]
    y_std = torch.tensor(stats["y_std"], dtype=torch.float32)[:, None, None]
    s_mean = torch.tensor(stats["s_mean"], dtype=torch.float32)
    s_std = torch.tensor(stats["s_std"], dtype=torch.float32)

    class _DS(Dataset):
        def __len__(self):
            return len(idx)

        def __getitem__(self, i):
            j = int(idx[i])
            x = torch.from_numpy(np.asarray(cache.inputs[j], dtype=np.float32))
            y = torch.from_numpy(np.asarray(cache.targets[j], dtype=np.float32))
            m = torch.from_numpy(np.asarray(cache.masks[j], dtype=np.float32))
            s = torch.from_numpy(np.asarray(cache.scalars[j], dtype=np.float32))
            return {
                "x": (x - x_mean) / x_std,
                "y": (y - y_mean) / y_std,
                "mask": m,
                "s": (s - s_mean) / s_std,
                "index": j,
                "weight": torch.tensor(float(weights[i])),
                "fidelity": int(cache.fidelity[j]),
            }

    return _DS()
