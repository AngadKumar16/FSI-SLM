"""Batch driver for the panel backend: 107 sections x 18 angles = 1,926 cases.

Writes one ``.npz`` per case plus a ``manifest.jsonl`` and a ``splits.json``.
Cases are grouped by foil so the per-foil setup (trailing-edge closure, zero-lift
angle, lift slope, section properties) is computed once instead of eighteen
times, and foils are farmed out to a process pool.

Split policy: **hold out whole foils**, never individual angles. Splitting by
case would put the same shape in train and test at neighbouring incidences,
which inflates the score without measuring what the surrogate is for --
generalising to an unseen shape. NACA 0009 is forced into the test set because
it is the section the experimental rig uses.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .geometry import load_dat
from .sdf import Grid
from .simulate import CasePrep, pack_npz, simulate_case

# Dense where the physics is changing fastest (attached, pre-stall), coarse in
# the post-stall plateau. Exactly 18 stations.
DEFAULT_ALPHAS: Sequence[float] = (
    -8.0, -6.0, -4.0, -2.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0,
    6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0,
)
DEFAULT_REYNOLDS = 5.0e5
TEST_FOILS = ("n0009sm",)
AIRFOIL_ROOT = "Data/Airfoils"


@dataclass
class GenConfig:
    airfoil_root: str = AIRFOIL_ROOT
    out_dir: str = "artifacts/cfd"
    alphas: Sequence[float] = DEFAULT_ALPHAS
    reynolds: float = DEFAULT_REYNOLDS
    nx: int = 128
    ny: int = 128
    e_star: float = 2.0e4
    spar_x: float = 0.25
    max_workers: int = 4
    overwrite: bool = False

    def grid(self) -> Grid:
        base = Grid()
        return Grid(nx=self.nx, ny=self.ny, xlim=base.xlim, ylim=base.ylim)


def list_airfoils(root: str | Path = AIRFOIL_ROOT) -> List[Path]:
    """Every ``.dat``/``.dat.txt`` under the series directories, sorted."""
    root = Path(root)
    paths = sorted(
        p for p in root.rglob("*.dat*")
        if p.is_file() and not p.name.startswith(".")
    )
    return paths


def _series_of(path: Path) -> str:
    return path.parent.name


# Higher wins when the same (foil, alpha) exists at more than one fidelity.
# URANS outranks steady RANS everywhere, but it earns that rank only past stall:
# a steady solver does not reach a fixed point on massively separated flow, so a
# converged steady case at 4 degrees and a time-averaged unsteady case at 4
# degrees are the same answer, while at 16 degrees only one of them means
# anything.
FIDELITY_RANK = {"panel": 0, "rans": 1, "urans": 2}


def _rank(row: dict) -> tuple:
    """Merge precedence: fidelity first, then completeness."""
    return (FIDELITY_RANK.get(row.get("fidelity") or "panel", 0), _richness(row))


def _richness(row: dict) -> int:
    """How complete a manifest row is; used to break merge ties."""
    return int(bool(row.get("ok"))) + sum(
        1 for k in ("f_sep", "natural_frequency_hz", "stress_max", "cl")
        if row.get(k) is not None
    )


def _row_from_meta(meta: dict) -> dict:
    """Flatten the metadata stored inside an ``.npz`` into a manifest row."""
    k = meta.get("coeffs", {})
    m = meta.get("modal", {})
    return {
        "cl": k.get("cl"), "cd": k.get("cd"), "cm": k.get("cm"),
        "cl_inviscid": k.get("cl_inviscid"),
        "f_sep": k.get("f_sep"),
        "stalled": bool(k.get("stalled", False)),
        "deep_stall": bool((k.get("f_sep") or 1.0) < 0.15),
        "alpha_L0_deg": k.get("alpha_L0_deg"),
        "cl_alpha_per_rad": k.get("cl_alpha_per_rad"),
        "thickness": k.get("max_thickness"),
        "thickness_x": k.get("max_thickness_x"),
        "w_max": k.get("w_max"), "w_te": k.get("w_te"),
        "twist_te_deg": k.get("twist_te_deg"),
        "stress_max": k.get("stress_max"),
        "e_star": k.get("e_star"),
        "natural_frequency_hz": m.get("natural_frequency_hz"),
        "mass_ratio": m.get("mass_ratio"),
        "reduced_velocity": m.get("reduced_velocity"),
        "fidelity": meta.get("fidelity", "panel"),
        # Carried into the manifest so a whole sweep can be audited at once:
        # a RANS row whose structural channels came from "raster_probe" rather
        # than "openfoam_patch" has deflection targets biased low.
        "surface_source": meta.get("surface_source"),
        "converged": meta.get("converged"),
        "residual_max": meta.get("residual_max"),
        "cl_std": meta.get("cl_std"),
        "cd_std": meta.get("cd_std"),
        # Unsteady rows carry the one quantity a steady solve cannot produce:
        # the vortex-shedding frequency, measured rather than correlated. The
        # twin bridge downstream would otherwise fall back on Fage-Johansen.
        # ``organised_shedding`` is not decoration: the twin bridge refuses to
        # use a measured frequency without it, because the peak-finder returns
        # the largest bin whether or not a street exists. Dropping it from the
        # row made every URANS case fall back to the correlation it was solved
        # to replace.
        **{
            k: (meta.get("unsteady") or {}).get(k)
            for k in ("f_shed", "strouhal_chord", "strouhal_projected",
                      "cl_rms", "cd_rms", "n_cycles", "statistically_steady",
                      "organised_shedding", "shedding_prominence",
                      "shedding_note", "stationarity_note")
        },
    }


def _run_foil(args) -> List[dict]:
    """Solve every angle for one foil. Runs in a worker process."""
    path_str, cfg_dict = args
    cfg = GenConfig(**cfg_dict)
    path = Path(path_str)
    grid = cfg.grid()
    out_npz = Path(cfg.out_dir) / "npz"
    out_npz.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    try:
        af = load_dat(path)
        prep = CasePrep.build(af)
    except Exception as exc:  # unparseable geometry -> record and move on
        return [{
            "foil": path.stem.replace(".dat", ""),
            "series": _series_of(path),
            "dat": str(path),
            "error": f"{type(exc).__name__}: {exc}",
            "ok": False,
        }]

    for alpha in cfg.alphas:
        tag = f"{path.stem.replace('.dat', '')}_a{alpha:g}"
        npz_path = out_npz / f"{tag}.npz"
        row = {
            "foil": path.stem.replace(".dat", ""),
            "series": _series_of(path),
            "dat": str(path),
            "alpha_deg": float(alpha),
            "reynolds": float(cfg.reynolds),
            "npz": str(npz_path),
            "fidelity": "panel",
            "ok": False,
            "error": None,
        }
        if npz_path.exists() and not cfg.overwrite:
            try:
                with np.load(npz_path, allow_pickle=False) as z:
                    meta = json.loads(str(z["meta"]))
                row.update(_row_from_meta(meta))
                row.update({"ok": True, "cached": True})
                rows.append(row)
                continue
            except Exception:
                pass  # missing meta or corrupt file -> regenerate

        try:
            case = simulate_case(
                prep, alpha, cfg.reynolds, grid,
                e_star=cfg.e_star, spar_x=cfg.spar_x, with_modal=True,
            )
            packed = pack_npz(case)
            np.savez_compressed(npz_path, **packed)
            row.update(_row_from_meta(json.loads(str(packed["meta"]))))
            row.update({"ok": True, "cached": False})
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["traceback"] = traceback.format_exc(limit=3)
        rows.append(row)
    return rows


def generate_dataset(
    cfg: Optional[GenConfig] = None,
    limit_foils: Optional[int] = None,
    progress: bool = True,
    foil_start: int = 0,
    foil_count: Optional[int] = None,
    finalize: bool = True,
    time_budget_s: Optional[float] = None,
) -> dict:
    """Run the sweep (or a slice of it). Writes per-chunk rows, then a manifest.

    ``foil_start`` / ``foil_count`` let the sweep be run in pieces -- useful
    under a wall-clock limit, or to resume after an interruption. Each chunk
    drops its rows in ``rows/`` and :func:`finalize_manifest` stitches them
    together, so partial runs accumulate instead of clobbering each other.
    Completed cases are detected from their ``.npz`` and skipped.
    """
    cfg = cfg or GenConfig()
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    foils = list_airfoils(cfg.airfoil_root)
    if limit_foils:
        foils = foils[:limit_foils]
    foils = foils[foil_start: (foil_start + foil_count) if foil_count else None]
    if not foils:
        raise FileNotFoundError(f"no airfoil .dat files under {cfg.airfoil_root}")

    cfg_dict = {k: v for k, v in cfg.__dict__.items()}
    cfg_dict["alphas"] = list(cfg.alphas)
    jobs = [(str(p), cfg_dict) for p in foils]

    rows: List[dict] = []
    t0 = time.time()
    done = 0
    stopped_early = False

    def _over_budget() -> bool:
        return time_budget_s is not None and (time.time() - t0) > time_budget_s

    if cfg.max_workers > 1:
        # Pin every worker to a single BLAS thread. The panel solve calls
        # np.linalg.solve, which by default grabs one thread per core *inside
        # each process*: four workers then oversubscribe a four-core box 16-way
        # and the pool runs an order of magnitude slower than a serial loop
        # (measured: 28 s for four foils, against 2.5 s each serially). The
        # parallelism we want is across cases, not inside the linear algebra.
        # "spawn" is required so the children re-read these on import.
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[var] = "1"
        ctx = mp.get_context("spawn")
        # Submit in waves rather than all at once: with everything queued up
        # front there is no way to stop cleanly, and a run killed by a wall-clock
        # limit loses every row it had computed. Waves of `max_workers` cost one
        # barrier each and make the budget check meaningful.
        with ProcessPoolExecutor(max_workers=cfg.max_workers, mp_context=ctx) as pool:
            for i in range(0, len(jobs), cfg.max_workers):
                if _over_budget():
                    stopped_early = True
                    break
                wave = jobs[i: i + cfg.max_workers]
                futs = {pool.submit(_run_foil, j): j[0] for j in wave}
                for fut in as_completed(futs):
                    rows.extend(fut.result())
                    done += 1
                    if progress:
                        el = time.time() - t0
                        print(
                            f"[{done:3d}/{len(jobs)}] {Path(futs[fut]).stem:<22s} "
                            f"{el:6.1f}s elapsed, eta {el / done * (len(jobs) - done):6.1f}s",
                            flush=True,
                        )
    else:
        for j in jobs:
            if _over_budget():
                stopped_early = True
                break
            rows.extend(_run_foil(j))
            done += 1
            if progress:
                print(f"[{done:3d}/{len(jobs)}] {Path(j[0]).stem}", flush=True)

    rows.sort(key=lambda r: (r.get("foil", ""), r.get("alpha_deg", 0.0)))
    rows_dir = out / "rows"
    rows_dir.mkdir(exist_ok=True)
    chunk = rows_dir / f"chunk_{foil_start:04d}_{len(foils):04d}.jsonl"
    with chunk.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    if not finalize:
        return {
            "chunk": str(chunk),
            "cases": len(rows),
            "ok": sum(1 for r in rows if r.get("ok")),
            "foils_done": done,
            "foils_requested": len(jobs),
            "stopped_early": stopped_early,
            "seconds": round(time.time() - t0, 1),
        }
    return finalize_manifest(out, n_alphas=len(cfg.alphas))


def finalize_manifest(out_dir: str | Path, n_alphas: int = len(DEFAULT_ALPHAS)) -> dict:
    """Merge every chunk into ``manifest.jsonl``, write splits and a summary."""
    out = Path(out_dir)
    rows_dir = out / "rows"
    merged: Dict[tuple, dict] = {}
    # Any .jsonl in rows/, not just "chunk_*": the OpenFOAM driver writes
    # "rans_<timestamp>.jsonl", and a narrower glob silently ignored an entire
    # RANS sweep -- 15 solved cases, over an hour of compute, merged into
    # nothing with no error.
    for chunk in sorted(rows_dir.glob("*.jsonl")) if rows_dir.exists() else []:
        for line in chunk.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r.get("foil"), r.get("alpha_deg"))
            prev = merged.get(key)
            # Rank by fidelity first, richness second. A RANS row legitimately
            # lacks the panel backend's viscous-model scalars (f_sep and
            # friends), so it scores *lower* on richness than the panel row it
            # is meant to replace -- ranking on richness alone silently threw
            # away every expensive case at an angle the cheap sweep had already
            # covered. Chunks from an early run that predates the embedded
            # metadata are still resolved by richness within a fidelity.
            if prev is not None and _rank(prev) > _rank(r):
                continue
            merged[key] = r

    rows = sorted(merged.values(), key=lambda r: (r.get("foil") or "", r.get("alpha_deg") or 0.0))
    manifest = out / "manifest.jsonl"
    with manifest.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    splits = write_splits(manifest, out / "splits.json")
    ok = [r for r in rows if r.get("ok")]
    failed = [r for r in rows if not r.get("ok")]
    summary = {
        "cases": len(rows),
        "ok": len(ok),
        "failed": len(failed),
        "foils": len({r["foil"] for r in rows}),
        "alphas": n_alphas,
        "deep_stall": sum(1 for r in ok if r.get("deep_stall")),
        "manifest": str(manifest),
        "splits": {k: len(v) for k, v in splits.items()},
        "failures": [
            {"foil": r.get("foil"), "alpha": r.get("alpha_deg"), "error": r.get("error")}
            for r in failed[:20]
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def write_splits(
    manifest: str | Path,
    out_path: str | Path,
    test_foils: Sequence[str] = TEST_FOILS,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
) -> Dict[str, List[str]]:
    """Foil-disjoint train/val/test split, stratified by series.

    ``test_foils`` are pinned into the test set (NACA 0009, the section the rig
    uses) and the rest of the test set is drawn at random within each series, so
    a test score is an average over ~16 unseen shapes rather than a single lucky
    or unlucky one.
    """
    rows = [json.loads(l) for l in Path(manifest).read_text().splitlines() if l.strip()]
    by_foil: Dict[str, str] = {}
    for r in rows:
        if r.get("ok"):
            by_foil[r["foil"]] = r["series"]

    pinned = sorted(f for f in by_foil if f in set(test_foils))
    rest = sorted(f for f in by_foil if f not in set(pinned))
    rng = np.random.default_rng(seed)

    def _draw(pool: List[str], frac: float) -> List[str]:
        picked: List[str] = []
        for series in sorted({by_foil[f] for f in pool}):
            members = sorted(f for f in pool if by_foil[f] == series)
            k = max(1, int(round(frac * len(members))))
            picked.extend(rng.permutation(members)[:k].tolist())
        return sorted(picked)

    test_extra = _draw(rest, test_frac)
    rest2 = [f for f in rest if f not in set(test_extra)]
    val = _draw(rest2, val_frac)
    train = sorted(f for f in rest2 if f not in set(val))

    splits = {"train": train, "val": val, "test": sorted(pinned + test_extra)}
    Path(out_path).write_text(json.dumps(splits, indent=2))
    return splits
