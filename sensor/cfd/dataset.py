"""Batch driver for the OpenFOAM backend.

For each case: build mesh -> write case -> convert -> solve -> read force
coefficients -> sample onto the U-Net grid -> write the shared ``.npz``
contract, appending a row in exactly the schema :mod:`sensor.generate` produces.

Rows land in ``<out_dir>/rows/`` and the ``.npz`` files in ``<out_dir>/npz/``,
so a RANS sweep can be pointed at the *same* ``out_dir`` as the panel sweep and
:func:`sensor.generate.finalize_manifest` merges them into one manifest. Each
row carries ``fidelity: "rans"``; the merge prefers the richer row when a
(foil, alpha) pair exists at both fidelities, so re-running a case in RANS
upgrades it in place.

Split policy is not duplicated here -- :func:`sensor.generate.write_splits` owns
it, holds out whole foils, and pins NACA 0009 into the test set.

Requires external binaries: OpenFOAM (ESI v2606, via the ``openfoam2606``
wrapper) and gmsh. Nothing else in the repository does.
"""
from __future__ import annotations

import json
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..generate import _row_from_meta
from ..geometry import load_dat
from ..sdf import Grid
from ..simulate import pack_npz
from .case import CaseParams, write_case
from .mesh import MeshParams, build_mesh
from .run import convert_mesh, read_force_coeffs, read_residuals, solve
from .sample import sample_case

# Denser through stall than the panel sweep: that is the band where the two
# backends are expected to disagree, so it is where RANS cases buy the most.
DEFAULT_ALPHAS: Sequence[float] = (
    -8.0, -4.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0,
    12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0,
)


def run_one(
    dat_path: str | Path,
    alpha_deg: float,
    out_dir: str | Path,
    reynolds: float = 5.0e5,
    keep_case: bool = False,
    grid: Optional[Grid] = None,
    average_last: int = 500,
) -> dict:
    """Solve and pack one RANS case. Never raises; failures come back as rows."""
    dat_path = Path(dat_path)
    out_dir = Path(out_dir)
    grid = grid or Grid()
    foil = dat_path.stem.replace(".dat", "")
    tag = f"{foil}_a{alpha_deg:g}"
    case = out_dir / "cases" / tag
    npz = out_dir / "npz" / f"{tag}.npz"
    npz.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "foil": foil,
        "series": dat_path.parent.name,
        "dat": str(dat_path),
        "alpha_deg": float(alpha_deg),
        "reynolds": float(reynolds),
        "npz": str(npz),
        "fidelity": "rans",
        "ok": False,
        "error": None,
        "converged": False,
    }
    try:
        af = load_dat(dat_path)
        msh = case.parent / f"{tag}.msh"
        case.mkdir(parents=True, exist_ok=True)
        build_mesh(af, alpha_deg, msh, MeshParams())
        write_case(case, CaseParams(reynolds=reynolds, span=MeshParams().span))
        convert_mesh(case, msh)
        res = solve(case)
        if res.returncode != 0:
            raise RuntimeError(f"simpleFoam failed:\n{res.stdout[-1500:]}")

        # Return code says only that the binary ran. Convergence has to come
        # from the residuals.
        conv = read_residuals(case)
        fc = read_force_coeffs(case, average_last=average_last)
        row.update({
            "converged": conv["converged"],
            "residual_max": conv["residual_max"],
            "iterations": conv["iterations"],
            "convergence_note": conv["reason"],
            "cl_std": fc.get("Cl_std"),
            "cd_std": fc.get("Cd_std"),
            "n_averaged": fc.get("n_averaged"),
        })

        sample = sample_case(
            case, af, alpha_deg, reynolds, grid,
            cl=fc.get("Cl"), cd=fc.get("Cd"), cm=fc.get("Cm"),
            converged=conv["converged"],
            residual_max=conv["residual_max"],
            cl_std=fc.get("Cl_std"), cd_std=fc.get("Cd_std"),
        )
        packed = pack_npz(sample)
        np.savez_compressed(npz, **packed)
        row.update(_row_from_meta(json.loads(str(packed["meta"]))))
        row["fidelity"] = "rans"
        row["ok"] = True
    except Exception as exc:  # keep the batch going; record the failure
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc()[-1500:]
    finally:
        # Never delete a case that failed. Wiping it removes the .msh, the
        # OpenFOAM logs and constant/polyMesh -- i.e. everything needed to
        # diagnose why it failed -- and leaves only a stack trace.
        if not keep_case and row["ok"]:
            shutil.rmtree(case, ignore_errors=True)
    return row


def build_dataset(
    dat_paths: Sequence[str | Path],
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    out_dir: str | Path = "artifacts/cfd",
    reynolds: float = 5.0e5,
    max_workers: int = 1,
    keep_case: bool = False,
    progress: bool = True,
) -> dict:
    """Run a RANS sweep into the shared dataset directory.

    Point ``out_dir`` at the panel sweep's directory to mix fidelities, then run
    ``python cli.py sensor-cache`` to rebuild the training cache.
    """
    out_dir = Path(out_dir)
    rows_dir = out_dir / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(Path(p), float(a)) for p in dat_paths for a in alphas]

    rows = []
    t0 = time.time()
    if max_workers == 1:
        for p, a in jobs:
            rows.append(run_one(p, a, out_dir, reynolds, keep_case))
            if progress:
                _log(rows[-1], len(rows), len(jobs), t0)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futs = {
                pool.submit(run_one, p, a, out_dir, reynolds, keep_case): (p, a)
                for p, a in jobs
            }
            for fut in as_completed(futs):
                rows.append(fut.result())
                if progress:
                    _log(rows[-1], len(rows), len(jobs), t0)

    rows.sort(key=lambda r: (r["foil"], r["alpha_deg"]))
    chunk = rows_dir / f"rans_{int(t0)}.jsonl"
    with chunk.open("w") as fh:
        for r in rows:
            fh.write(json.dumps({k: v for k, v in r.items() if k != "traceback"}) + "\n")

    ok = [r for r in rows if r["ok"]]
    return {
        "chunk": str(chunk),
        "cases": len(rows),
        "ok": len(ok),
        "failed": len(rows) - len(ok),
        "seconds": round(time.time() - t0, 1),
        "failures": [
            {"foil": r["foil"], "alpha": r["alpha_deg"], "error": r["error"]}
            for r in rows if not r["ok"]
        ][:10],
        "next": "python cli.py sensor-cache   # after finalize_manifest",
    }


def plan_multifidelity_sweep(
    splits_path: str | Path = "artifacts/cfd/splits.json",
    manifest: str | Path = "artifacts/cfd/manifest.jsonl",
    n_foils: int = 4,
    steady_alphas: Sequence[float] = (-4.0, 0.0, 4.0, 6.0, 8.0),
    unsteady_alphas: Sequence[float] = (12.0, 16.0, 20.0),
    seed: int = 0,
) -> dict:
    """Choose which *training* foils to re-solve at high fidelity, and how.

    The reason this is a function and not a shell loop: the only RANS cases in
    the dataset so far are on NACA 0009, which is pinned to the *test* split.
    Training on them would be training on test, so the multi-fidelity path has
    been wired and tested but never actually used. What it needs is high-fidelity
    rows on foils the network is allowed to see.

    Foils are drawn one per series so the high-fidelity subset spans 4-digit,
    5-digit and 6-series sections rather than three variations of one shape, and
    the draw is seeded so the choice is reproducible and can be reported.

    The angle split is the measured one: below stall a steady solve converges
    and is a hundred times cheaper, above it a steady solve has no fixed point.
    """
    splits = json.loads(Path(splits_path).read_text())
    train = set(splits["train"])
    rows = [
        json.loads(l) for l in Path(manifest).read_text().splitlines() if l.strip()
    ]
    by_foil = {}
    for r in rows:
        if r.get("ok") and r["foil"] in train:
            by_foil.setdefault(r["foil"], r)

    rng = np.random.default_rng(seed)
    series = sorted({r["series"] for r in by_foil.values()})
    picked = []
    for s in series:
        members = sorted(f for f, r in by_foil.items() if r["series"] == s)
        if members:
            picked.append(str(rng.permutation(members)[0]))
    # Top up from whatever is left, still deterministically.
    rest = sorted(f for f in by_foil if f not in set(picked))
    picked.extend(str(f) for f in rng.permutation(rest)[: max(0, n_foils - len(picked))])
    picked = sorted(picked[:n_foils])

    dats = [by_foil[f]["dat"] for f in picked]
    return {
        "foils": picked,
        "dats": dats,
        "steady": {"alphas": list(steady_alphas), "cases": len(dats) * len(steady_alphas)},
        "unsteady": {"alphas": list(unsteady_alphas),
                     "cases": len(dats) * len(unsteady_alphas)},
        "note": (
            "these are train-split foils, so the resulting rows are legitimate "
            "training data; NACA 0009 stays in test and stays panel-free"
        ),
    }


def _log(row: dict, done: int, total: int, t0: float) -> None:
    status = "ok" if row["ok"] else f"FAIL {row['error']}"
    el = time.time() - t0
    print(
        f"[{done:3d}/{total}] {row['foil']} a={row['alpha_deg']:g} "
        f"Cl={row.get('cl')} Cd={row.get('cd')} {status}  "
        f"({el:.0f}s elapsed, eta {el / done * (total - done):.0f}s)",
        flush=True,
    )
