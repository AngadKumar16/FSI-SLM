"""Drive one OpenFOAM case: mesh convert -> patch types -> solve -> read Cl/Cd.

All OpenFOAM binaries are invoked through the gerlero wrapper
``openfoam2606 -c '<shell>'`` which activates the v2606 environment.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np

WRAP = "openfoam2606"


def foam(case_dir: Path, shell: str, timeout: int = 3600) -> subprocess.CompletedProcess:
    return subprocess.run(
        [WRAP, "-c", f"cd '{case_dir}' && {shell}"],
        capture_output=True, text=True, timeout=timeout,
    )


def convert_mesh(case_dir: Path, msh: Path) -> None:
    """gmshToFoam, then fix patch types (airfoil=wall, frontAndBack=empty)."""
    # Every command runs as `cd <case_dir> && ...`, so a mesh path relative to
    # the *repository* root resolves to nothing once inside the case directory.
    # gmshToFoam does not treat a missing file as an error: it opens an empty
    # stream, reads zero patches, and dies later on
    # UPtrList<polyPatch>::last() -- "Cannot dereference nullptr at index -1 in
    # range [0,0)" -- which reads exactly like a malformed mesh and is not.
    msh = Path(msh).resolve()
    if not msh.is_file():
        raise FileNotFoundError(f"mesh not found: {msh}")

    r = foam(case_dir, f"gmshToFoam '{msh}'")
    if r.returncode != 0:
        raise RuntimeError(f"gmshToFoam failed:\n{r.stdout[-2000:]}\n{r.stderr[-1000:]}")

    boundary = Path(case_dir) / "constant" / "polyMesh" / "boundary"
    if not boundary.is_file():
        raise RuntimeError(
            f"gmshToFoam returned 0 but wrote no {boundary}; the mesh was not read"
        )
    # polyBoundaryMesh list is the top entry "entry0" in the boundary file
    for patch, typ in (("frontAndBack", "empty"), ("airfoil", "wall")):
        r = foam(case_dir, f"foamDictionary constant/polyMesh/boundary "
                           f"-entry 'entry0/{patch}/type' -set {typ}")
        if r.returncode != 0:
            raise RuntimeError(f"set {patch} type failed: {r.stderr[-500:]}")


def check_mesh(case_dir: Path) -> str:
    return foam(case_dir, "checkMesh -constant").stdout


def solve(case_dir: Path, timeout: int = 5400) -> subprocess.CompletedProcess:
    return foam(case_dir, "renumberMesh -overwrite -constant > log.renumber 2>&1; "
                          "simpleFoam > log.simpleFoam 2>&1; tail -1 log.simpleFoam",
                timeout=timeout)


def read_force_coeffs(case_dir: Path, average_last: int = 500) -> dict:
    """Force coefficients from ``postProcessing/forceCoeffs*``.

    Averaged over the last ``average_last`` iterations rather than read off the
    final line. Steady SIMPLE does not settle to a fixed point on separated
    flow -- it limit-cycles -- so the last iteration is an arbitrary point on an
    oscillation. Taking the mean turns that into a usable number and the spread
    (returned as ``Cl_std``/``Cd_std``) says how far from converged it is.

    Set ``average_last=1`` for the old last-line behaviour.
    """
    files = list(case_dir.glob("postProcessing/forceCoeffs*/**/coefficient*.dat"))
    if not files:
        files = list(case_dir.glob("postProcessing/forceCoeffs*/**/forceCoeffs*.dat"))
    if not files:
        raise FileNotFoundError("no forceCoeffs output")

    text = files[0].read_text()
    lines = [l for l in text.splitlines() if l.strip() and not l.startswith("#")]
    if not lines:
        raise ValueError(f"{files[0]}: no data rows")
    header = [h for h in text.splitlines() if h.startswith("#")][-1]
    cols = header.lstrip("#").split()

    # Average over a trailing *fraction*, capped at `average_last`. A run that
    # converges early stops early, so a fixed 500-iteration window on a
    # 600-iteration case reaches back into the start-up transient and biases the
    # mean. That showed up as Cd(0 deg) = 0.0131 against Cd(2 deg) = 0.0085 --
    # a symmetric section cannot have its drag minimum away from zero
    # incidence. One third of the record is safely past the transient.
    n_rows = len(lines)
    window_size = max(1, min(average_last, n_rows // 3)) if n_rows >= 12 else n_rows
    window = lines[-window_size:]
    rows = []
    for line in window:
        vals = line.split()
        if len(vals) == len(cols):
            rows.append([float(v) for v in vals])
    if not rows:
        rows = [[float(v) for v in lines[-1].split()]]
    arr = np.asarray(rows, dtype=float)

    # map EXACT column names (avoid the (f)/(r) half-panel columns)
    alias = {"cl": "Cl", "cd": "Cd", "cmpitch": "Cm", "cm": "Cm"}
    out = {
        "iter": float(arr[-1, 0]),
        "n_averaged": int(arr.shape[0]),
    }
    for j, name in enumerate(cols):
        key = alias.get(name.strip().lower())
        if key and j < arr.shape[1]:
            out[key] = float(arr[:, j].mean())
            out[f"{key}_std"] = float(arr[:, j].std())
    return out


# Residual below which a field is considered settled. 1e-4 is the usual
# engineering bar for steady incompressible RANS force coefficients.
RESIDUAL_TOLERANCE = 1.0e-4
_RESIDUAL_RE = re.compile(
    r"Solving for (\w+),\s*Initial residual = ([0-9.eE+-]+)"
)


def read_residuals(case_dir: Path, log_name: str = "log.simpleFoam") -> dict:
    """Final initial-residual per field, and whether the run actually converged.

    ``simpleFoam`` exits 0 whether or not the residuals fell, so a return code
    says nothing about solution quality. On separated cases the run burns every
    iteration of ``n_iter`` and stops, and the resulting coefficients are a
    snapshot of an oscillation -- which is how a lift curve ends up
    non-monotonic through stall.
    """
    log = Path(case_dir) / log_name
    if not log.is_file():
        return {"converged": False, "reason": f"{log_name} missing", "residuals": {}}

    text = log.read_text(errors="ignore")
    residuals: dict = {}
    for field, value in _RESIDUAL_RE.findall(text):
        try:
            residuals[field] = float(value)
        except ValueError:
            continue

    declared = "SIMPLE solution converged" in text
    times = re.findall(r"^Time = ([0-9.eE+-]+)", text, flags=re.MULTILINE)
    iterations = int(float(times[-1])) if times else 0
    worst = max(residuals.values()) if residuals else float("inf")

    return {
        "converged": bool(declared or worst < RESIDUAL_TOLERANCE),
        "declared_converged": declared,
        "residual_max": worst,
        "residuals": residuals,
        "iterations": iterations,
        "reason": (
            "ok" if declared or worst < RESIDUAL_TOLERANCE
            else f"max initial residual {worst:.2e} > {RESIDUAL_TOLERANCE:g} "
                 f"after {iterations} iterations"
        ),
    }
