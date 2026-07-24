"""Batch driver: generate the (input, target) dataset over foils x AoA.

For each case: build mesh -> write case -> convert -> solve -> sample to .npz,
appending a row to ``manifest.jsonl`` (foil, series, AoA, Re, Cl, Cd, converged,
npz path). Cases are independent, so this is embarrassingly parallel; a simple
serial loop is provided plus a ``max_workers`` process pool.

Split policy (see ``split_by_foil``): hold out entire foils -- never leak the
same shape across train/val/test. NACA 0009 is forced into the test set.
"""
from __future__ import annotations

import json
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ..geometry import load_dat
from ..sdf import Grid
from .case import CaseParams, write_case
from .mesh import MeshParams, build_mesh
from .run import convert_mesh, read_force_coeffs, solve
from .sample import sample_case, save_npz


def run_one(dat_path: str | Path, alpha_deg: float, work_dir: str | Path,
            reynolds: float = 5.0e5, keep_case: bool = False) -> dict:
    dat_path = Path(dat_path)
    work_dir = Path(work_dir)
    tag = f"{dat_path.stem}_a{alpha_deg:g}"
    case = work_dir / "cases" / tag
    npz = work_dir / "npz" / f"{tag}.npz"
    rec = {"foil": dat_path.stem, "series": dat_path.parent.name,
           "alpha_deg": float(alpha_deg), "reynolds": float(reynolds),
           "npz": str(npz), "converged": False, "error": None}
    try:
        af = load_dat(dat_path)
        msh = case.parent / f"{tag}.msh"
        case.mkdir(parents=True, exist_ok=True)
        build_mesh(af, alpha_deg, msh, MeshParams())
        write_case(case, CaseParams(reynolds=reynolds, span=MeshParams().span))
        convert_mesh(case, msh)
        res = solve(case)
        fc = read_force_coeffs(case)
        rec.update({"Cl": fc.get("Cl"), "Cd": fc.get("Cd"), "Cm": fc.get("Cm"),
                    "converged": res.returncode == 0})
        sample = sample_case(case, af, alpha_deg, reynolds, Grid())
        save_npz(npz, sample)
    except Exception as e:  # keep the batch going; record the failure
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()[-1500:]
    finally:
        if not keep_case:
            shutil.rmtree(case, ignore_errors=True)
    return rec


def build_dataset(dat_paths: Sequence[Path], alphas: Sequence[float],
                  work_dir: str | Path, reynolds: float = 5.0e5,
                  max_workers: int = 1) -> Path:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest = work_dir / "manifest.jsonl"
    jobs = [(p, a) for p in dat_paths for a in alphas]

    def _emit(rec):
        with manifest.open("a") as f:
            f.write(json.dumps({k: v for k, v in rec.items() if k != "traceback"}) + "\n")
        s = "ok" if rec["converged"] else f"FAIL {rec['error']}"
        print(f"[{rec['foil']} a={rec['alpha_deg']:g}] Cl={rec.get('Cl')} Cd={rec.get('Cd')} {s}")

    if max_workers == 1:
        for p, a in jobs:
            _emit(run_one(p, a, work_dir, reynolds))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(run_one, p, a, work_dir, reynolds): (p, a) for p, a in jobs}
            for fut in as_completed(futs):
                _emit(fut.result())
    return manifest


def split_by_foil(manifest: Path, test_foils=("n0009sm",), val_frac=0.15, seed=0):
    """Group-split so no foil shape appears in more than one split."""
    rng = np.random.default_rng(seed)
    rows = [json.loads(l) for l in Path(manifest).read_text().splitlines() if l.strip()]
    foils = sorted({r["foil"] for r in rows})
    rest = [f for f in foils if f not in test_foils]
    rng.shuffle(rest)
    n_val = max(1, int(len(rest) * val_frac))
    val, train = set(rest[:n_val]), set(rest[n_val:])
    def which(f):
        return "test" if f in test_foils else ("val" if f in val else "train")
    return {"train": train, "val": val, "test": set(test_foils),
            "rows": {r["npz"]: which(r["foil"]) for r in rows if r["converged"]}}
