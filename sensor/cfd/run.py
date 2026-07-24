"""Drive one OpenFOAM case: mesh convert -> patch types -> solve -> read Cl/Cd.

All OpenFOAM binaries are invoked through the gerlero wrapper
``openfoam2606 -c '<shell>'`` which activates the v2606 environment.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

WRAP = "openfoam2606"


def foam(case_dir: Path, shell: str, timeout: int = 3600) -> subprocess.CompletedProcess:
    return subprocess.run(
        [WRAP, "-c", f"cd '{case_dir}' && {shell}"],
        capture_output=True, text=True, timeout=timeout,
    )


def convert_mesh(case_dir: Path, msh: Path) -> None:
    """gmshToFoam, then fix patch types (airfoil=wall, frontAndBack=empty)."""
    r = foam(case_dir, f"gmshToFoam '{msh}'")
    if r.returncode != 0:
        raise RuntimeError(f"gmshToFoam failed:\n{r.stdout[-2000:]}\n{r.stderr[-1000:]}")
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


def read_force_coeffs(case_dir: Path) -> dict:
    """Parse the last line of postProcessing/forceCoeffs*/0/coefficient.dat."""
    files = list(case_dir.glob("postProcessing/forceCoeffs*/**/coefficient*.dat"))
    if not files:
        files = list(case_dir.glob("postProcessing/forceCoeffs*/**/forceCoeffs*.dat"))
    if not files:
        raise FileNotFoundError("no forceCoeffs output")
    lines = [l for l in files[0].read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    header = [h for h in files[0].read_text().splitlines() if h.startswith("#")][-1]
    cols = header.lstrip("#").split()
    vals = lines[-1].split()
    rec = dict(zip(cols, map(float, vals)))
    # map EXACT column names (avoid the (f)/(r) half-panel columns)
    alias = {"cl": "Cl", "cd": "Cd", "cmpitch": "Cm", "cm": "Cm"}
    out = {"iter": rec.get("Time", float(vals[0]))}
    for k, v in rec.items():
        key = alias.get(k.strip().lower())
        if key:
            out[key] = v
    return out
