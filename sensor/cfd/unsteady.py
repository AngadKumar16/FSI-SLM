"""Unsteady RANS (``pimpleFoam``) for the post-stall band the steady solver cannot do.

Why this exists
---------------

Every steady case at ``alpha >= 10`` in the first RANS sweep hit the iteration
cap with initial residuals of order 1e-2, and lift oscillated by up to +-0.43.
That is not a convergence failure to be tuned away: massively separated flow has
no steady solution to converge *to*. ``simpleFoam`` limit-cycles, and whichever
iteration it stops on is an arbitrary point on the oscillation. Seven of fifteen
cases were usable and the entire stall region was thrown out.

The right answer is to stop asking for a fixed point. ``pimpleFoam`` integrates
in time, the wake sheds, and the quantities that mean something are *statistics*
of the resulting signal: the time-averaged fields, the mean and RMS of the force
coefficients, and the shedding frequency.

What comes out that a steady solve cannot give
----------------------------------------------

The shedding frequency is not a nicety here. The interpreter half of this
project classifies vortex-induced vibration by comparing the shedding frequency
to the structural natural frequency, and until now that frequency came from the
Fage-Johansen correlation ``St = 0.2`` on a projected height -- a correlation,
applied to an airfoil, in the regime where it is least reliable. A URANS case
*measures* it. :func:`shedding_frequency` reads it off the lift signal and the
value travels with the case into the manifest and the surrogate metadata.

Honesty about what URANS is
---------------------------

Two-dimensional URANS with a k-omega SST model resolves the primary von Karman
shedding and nothing smaller. Real post-stall flow is three-dimensional and
broadband; 2-D URANS is known to overpredict the coherence of the shedding and
therefore the lift fluctuation amplitude, typically by tens of percent. It gets
the *frequency* far better than it gets the amplitude. Cases are tagged
``fidelity="urans"`` so the distinction survives into training, and the
statistical-stationarity check below is reported per case rather than assumed.

Requires OpenFOAM (ESI v2606 via the ``openfoam2606`` wrapper) and gmsh.
"""
from __future__ import annotations

import json
import shutil
import time
import traceback
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from ..generate import _row_from_meta
from ..geometry import load_dat
from ..sdf import Grid
from ..simulate import pack_npz
from .adapt import build_case_arrays
from .case import HEADER, _f
from .sweep import run_sweep
from .mesh import MeshParams, build_mesh
from .run import foam

# The band where the steady solver failed. Below 10 degrees a converged
# ``simpleFoam`` case is the same answer for a hundredth of the cost, so URANS
# there would be spending an hour to reproduce a number already in hand.
DEFAULT_STALL_ALPHAS: Sequence[float] = (10.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0)


@dataclass
class UnsteadyParams:
    """Time integration and averaging window, in convective times ``c / Uinf``.

    Defaults are sized on the physics rather than on a guess. A stalled airfoil
    sheds at a chord-based Strouhal number around 0.15-0.25, i.e. a period of
    4-7 convective times. ``t_start_average = 15`` discards the start-up
    transient and the first couple of cycles; ``t_end = 45`` then leaves a
    30-convective-time window (~6 shedding cycles at St 0.2), which is what
    :func:`statistics_converged` needs to pass its ``min_cycles = 6`` gate on a
    strongly shedding case. The window used to be 15 (t_end 30) to keep the run
    tractable off a cluster, but that was ~3 cycles -- below the gate -- so every
    organised case came back ``converged = False``; the mesh coarsening that made
    a case ~10x cheaper is what buys the honest window back. The deepest-stall,
    lowest-St cases (period ~7) still clear only ~4 cycles and may flag -- those
    genuinely need longer, and the per-case stationarity report says so.
    """

    reynolds: float = 5.0e5
    span: float = 0.1              # must match MeshParams.span
    t_end: float = 45.0            # convective times
    t_start_average: float = 15.0
    max_courant: float = 2.0       # PIMPLE with outer correctors tolerates > 1
    max_delta_t: float = 0.01
    initial_delta_t: float = 1.0e-4
    n_outer_correctors: int = 2
    n_correctors: int = 2
    turb_intensity: float = 0.05
    turb_length: float = 0.1
    write_interval: float = 10.0   # checkpoint cadence for resume; only final avg needed

    @property
    def average_window(self) -> float:
        return max(self.t_end - self.t_start_average, 0.0)


def write_unsteady_case(case_dir: str | Path, up: UnsteadyParams | None = None) -> Path:
    """Write a transient ``pimpleFoam`` case for one foil condition.

    Mesh is nondimensional (chord ``c = 1``), so ``U = 1``, ``nu = 1/Re`` and
    time is measured directly in convective times. Geometry is pre-rotated by
    the incidence in the mesh, so the freestream stays axial.
    """
    up = up or UnsteadyParams()
    d = Path(case_dir)
    nu = 1.0 / up.reynolds
    U = 1.0
    k = 1.5 * (up.turb_intensity * U) ** 2
    omega = k ** 0.5 / (0.09 ** 0.25 * up.turb_length)
    nut = k / omega

    _f(d / "constant/transportProperties", "dictionary", "transportProperties",
       f"transportModel Newtonian;\nnu {nu:.8g};")
    _f(d / "constant/turbulenceProperties", "dictionary", "turbulenceProperties",
       "simulationType RAS;\nRAS { RASModel kOmegaSST; turbulence on; printCoeffs on; }")

    def field(obj, cls, dims, internal, bcs):
        _f(d / "0" / obj, cls, obj,
           f"dimensions {dims};\ninternalField uniform {internal};\n"
           f"boundaryField {{\n{bcs}\n}}")

    fs_U = ("    farfield { type freestreamVelocity; freestreamValue uniform (1 0 0); }\n"
            "    airfoil { type noSlip; }\n"
            "    frontAndBack { type empty; }")
    fs_p = ("    farfield { type freestreamPressure; freestreamValue uniform 0; }\n"
            "    airfoil { type zeroGradient; }\n"
            "    frontAndBack { type empty; }")
    field("U", "volVectorField", "[0 1 -1 0 0 0 0]", "(1 0 0)", fs_U)
    field("p", "volScalarField", "[0 2 -2 0 0 0 0]", "0", fs_p)
    field("k", "volScalarField", "[0 2 -2 0 0 0 0]", f"{k:.6g}",
          f"    farfield {{ type inletOutlet; inletValue uniform {k:.6g}; value uniform {k:.6g}; }}\n"
          "    airfoil { type kqRWallFunction; value uniform 1e-12; }\n"
          "    frontAndBack { type empty; }")
    field("omega", "volScalarField", "[0 0 -1 0 0 0 0]", f"{omega:.6g}",
          f"    farfield {{ type inletOutlet; inletValue uniform {omega:.6g}; value uniform {omega:.6g}; }}\n"
          "    airfoil { type omegaWallFunction; value uniform 1e-12; }\n"
          "    frontAndBack { type empty; }")
    field("nut", "volScalarField", "[0 2 -1 0 0 0 0]", f"{nut:.6g}",
          "    farfield { type calculated; value uniform 0; }\n"
          "    airfoil { type nutkWallFunction; value uniform 0; }\n"
          "    frontAndBack { type empty; }")

    # ---- controlDict --------------------------------------------------------
    # ``adjustableRunTime`` rather than ``timeStep`` for writing: the time step
    # is adaptive, so a write every N steps lands at unpredictable times and the
    # averaging window would not be reproducible between cases.
    _f(d / "system/controlDict", "dictionary", "controlDict",
       "application pimpleFoam;\n"
       "startFrom latestTime;\nstartTime 0;\nstopAt endTime;\n"
       f"endTime {up.t_end:g};\ndeltaT {up.initial_delta_t:g};\n"
       f"writeControl adjustableRunTime;\nwriteInterval {up.write_interval:g};\n"
       "purgeWrite 2;\nwriteFormat ascii;\nwritePrecision 8;\n"
       "runTimeModifiable true;\n"
       f"adjustTimeStep yes;\nmaxCo {up.max_courant:g};\n"
       f"maxDeltaT {up.max_delta_t:g};\n"
       "functions {\n"
       "  forceCoeffs {\n"
       "    type forceCoeffs; libs (forces);\n"
       # Every step: the lift signal *is* the measurement, and a decimated
       # record aliases the shedding peak.
       "    writeControl timeStep; writeInterval 1;\n"
       "    patches (airfoil); rho rhoInf; rhoInf 1; magUInf 1;\n"
       "    liftDir (0 1 0); dragDir (1 0 0); pitchAxis (0 0 1);\n"
       f"    CofR (0.25 0 0); lRef 1; Aref {up.span};\n"
       "  }\n"
       "  fieldAverage {\n"
       "    type fieldAverage; libs (fieldFunctionObjects);\n"
       "    writeControl writeTime;\n"
       # Averaging starts *after* the transient. Starting at t = 0 folds the
       # impulsive start into the mean and no amount of run time removes it.
       f"    timeStart {up.t_start_average:g};\n"
       "    fields ( U { mean on; prime2Mean off; base time; }\n"
       "             p { mean on; prime2Mean on;  base time; } );\n"
       "  }\n"
       "  yPlus { type yPlus; libs (fieldFunctionObjects); writeControl writeTime; }\n"
       "}")

    # ---- schemes ------------------------------------------------------------
    # Second-order backward in time: Euler damps the shedding amplitude badly at
    # the time steps an adaptive run actually takes. ``limitedLinearV`` on the
    # momentum convection is the standard compromise for separated URANS --
    # second order where the solution is smooth, bounded through the shear layer.
    _f(d / "system/fvSchemes", "dictionary", "fvSchemes",
       "ddtSchemes { default backward; }\n"
       "gradSchemes { default Gauss linear; grad(U) cellLimited Gauss linear 1; }\n"
       "divSchemes {\n"
       "  default none;\n"
       "  div(phi,U) Gauss limitedLinearV 1;\n"
       "  div(phi,k) Gauss limitedLinear 1;\n"
       "  div(phi,omega) Gauss limitedLinear 1;\n"
       "  div((nuEff*dev2(T(grad(U))))) Gauss linear;\n"
       "}\n"
       "laplacianSchemes { default Gauss linear corrected; }\n"
       "interpolationSchemes { default linear; }\n"
       "snGradSchemes { default corrected; }\n"
       "wallDist { method meshWave; }")

    _f(d / "system/fvSolution", "dictionary", "fvSolution",
       "solvers {\n"
       "  p { solver GAMG; tolerance 1e-6; relTol 0.01; smoother GaussSeidel; }\n"
       "  pFinal { $p; tolerance 1e-7; relTol 0; }\n"
       "  \"(U|k|omega)\" { solver smoothSolver; smoother symGaussSeidel;\n"
       "                   tolerance 1e-8; relTol 0.1; }\n"
       "  \"(U|k|omega)Final\" { $U; tolerance 1e-9; relTol 0; }\n"
       "}\n"
       f"PIMPLE {{ nOuterCorrectors {up.n_outer_correctors}; "
       f"nCorrectors {up.n_correctors}; nNonOrthogonalCorrectors 1; }}\n"
       # No under-relaxation on the final outer corrector, or the time
       # derivative is quietly damped and the shedding amplitude with it.
       "relaxationFactors { equations { \".*\" 1.0; } }")
    return d


def solve_unsteady(case_dir: Path, timeout: int = 39600):
    return foam(case_dir,
                "renumberMesh -overwrite -constant > log.renumber 2>&1; "
                "pimpleFoam > log.pimpleFoam 2>&1; tail -2 log.pimpleFoam",
                timeout=timeout)


# ------------------------------------------------------------------- analysis


def read_time_series(case_dir: str | Path) -> dict:
    """``t``, ``Cl``, ``Cd``, ``Cm`` from the ``forceCoeffs`` function object.

    A restarted run writes a second directory under ``postProcessing`` and the
    two overlap in time, so the records are concatenated and de-duplicated on
    time rather than the first file being taken.
    """
    case_dir = Path(case_dir)
    files = sorted(case_dir.glob("postProcessing/forceCoeffs*/**/coefficient*.dat"))
    if not files:
        files = sorted(case_dir.glob("postProcessing/forceCoeffs*/**/forceCoeffs*.dat"))
    if not files:
        raise FileNotFoundError(f"no forceCoeffs output under {case_dir}")

    times, rows, cols = [], [], None
    for f in files:
        text = f.read_text()
        heads = [h for h in text.splitlines() if h.startswith("#")]
        if heads:
            cols = heads[-1].lstrip("#").split()
        for line in text.splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            vals = line.split()
            if cols and len(vals) == len(cols):
                rows.append([float(v) for v in vals])
                times.append(float(vals[0]))
    if not rows:
        raise ValueError(f"{case_dir}: forceCoeffs wrote no data rows")

    arr = np.asarray(rows, dtype=float)
    order = np.argsort(arr[:, 0], kind="stable")
    arr = arr[order]
    keep = np.concatenate([[True], np.diff(arr[:, 0]) > 1e-12])
    arr = arr[keep]

    alias = {"cl": "Cl", "cd": "Cd", "cmpitch": "Cm", "cm": "Cm"}
    out = {"t": arr[:, 0]}
    for j, name in enumerate(cols or []):
        key = alias.get(name.strip().lower())
        if key and key not in out and j < arr.shape[1]:
            out[key] = arr[:, j]
    return out


def _uniform_resample(t: np.ndarray, y: np.ndarray, n: Optional[int] = None):
    """Put an adaptive-time-step signal on a uniform grid before the FFT.

    ``adjustTimeStep`` means the samples are not equispaced. Feeding them to an
    FFT as if they were smears the spectral peak across neighbouring bins and
    can move it outright; linear interpolation onto the median step is the
    cheapest correct fix and the signal is heavily oversampled (Courant-limited
    steps are far below the shedding period).
    """
    if t.size < 8:
        return t, y
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        return t, y
    n = n or max(int((t[-1] - t[0]) / dt) + 1, 8)
    tu = np.linspace(t[0], t[-1], n)
    return tu, np.interp(tu, t, y)


def shedding_frequency(
    t: np.ndarray,
    signal: np.ndarray,
    t_start: float = 0.0,
    f_min: float = 0.02,
    f_max: float = 2.0,
) -> dict:
    """Dominant frequency of a force signal, in cycles per convective time.

    Because the case is nondimensional (``c = 1``, ``Uinf = 1``), a frequency in
    cycles per unit time *is* the chord-based Strouhal number. No conversion, no
    chance of a units slip.

    The mean is removed first -- otherwise the DC bin dominates everything -- and
    a Hann window is applied, which is what keeps the finite record from leaking
    the peak across half the spectrum. ``f_min`` rejects the slow drift of a run
    that has not fully settled; ``f_max`` rejects solver noise at the time-step
    scale.
    """
    t = np.asarray(t, float)
    y = np.asarray(signal, float)
    sel = t >= t_start
    t, y = t[sel], y[sel]
    if t.size < 32 or (t[-1] - t[0]) <= 0:
        return {"f_shed": None, "reason": "record too short to transform"}

    t, y = _uniform_resample(t, y)
    y = y - y.mean()
    n = y.size
    win = np.hanning(n)
    # Coherent gain of a Hann window is 0.5; correcting for it keeps the
    # reported amplitude comparable with the RMS computed in the time domain.
    spec = np.abs(np.fft.rfft(y * win)) * 2.0 / (n * 0.5)
    dt = float(t[1] - t[0])
    freq = np.fft.rfftfreq(n, dt)

    band = (freq >= f_min) & (freq <= f_max)
    if not band.any():
        return {"f_shed": None, "reason": "no spectral bin inside the search band"}
    idx = np.flatnonzero(band)[int(np.argmax(spec[band]))]
    df = float(freq[1] - freq[0])
    # Sub-bin interpolation. The bin spacing is 1 / (record length), so an 80
    # convective-time record resolves frequency only to about 0.0125 -- 6% of a
    # typical shedding Strouhal number, and the lock-in test downstream compares
    # f_shed to f_n at the few-percent level. Fitting a parabola through the
    # peak bin and its two neighbours recovers the true peak to a fraction of a
    # bin, which is standard practice for a windowed periodogram and costs three
    # array lookups.
    offset = 0.0
    if 0 < idx < spec.size - 1:
        a, b, c = (float(np.log(max(spec[i], 1e-300))) for i in (idx - 1, idx, idx + 1))
        denom = a - 2.0 * b + c
        if denom != 0.0:
            offset = float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))
    f = float(freq[idx] + offset * df)

    # Peak prominence: a genuine shedding tone stands well clear of the local
    # background. Without this an attached case with no organised street returns
    # whichever noise bin happened to be largest, and it would look like a
    # measurement.
    background = float(np.median(spec[band]))
    prominence = float(spec[idx] / (background + 1e-12))
    return {
        "f_shed": f,
        "amplitude": float(spec[idx]),
        "prominence": prominence,
        "record_length": float(t[-1] - t[0]),
        "n_cycles": float((t[-1] - t[0]) * f),
        "organised": bool(prominence > 5.0),
        "reason": (
            "ok" if prominence > 5.0
            else f"peak is only {prominence:.1f}x the local background; this "
                 f"record has no organised shedding tone"
        ),
    }


def strouhal_projected(f_shed: float, alpha_deg: float, thickness: float) -> float:
    """Strouhal number on the Fage-Johansen projected height ``sin|a| + t/c``.

    The chord-based number is what the simulation produces directly; this is the
    one the bluff-body literature quotes, and the one that should land near 0.2
    if the case is really shedding a von Karman street.
    """
    d = abs(np.sin(np.deg2rad(alpha_deg))) + thickness
    return float(f_shed * max(d, 1e-6))


def statistics_converged(
    t: np.ndarray,
    signal: np.ndarray,
    t_start: float,
    tol: float = 0.05,
    min_cycles: float = 6.0,
    f_shed: Optional[float] = None,
) -> dict:
    """Is the averaging window long enough for its mean to mean anything?

    Two conditions, both necessary. The window has to cover at least
    ``min_cycles`` shedding periods -- averaging over two and a half cycles
    returns a number that depends on which phase the window happened to start
    in. And the mean over the first half of the window has to agree with the
    mean over the second half to within ``tol`` of the signal's own RMS
    fluctuation, which is what catches a case still drifting toward its limit
    cycle.

    This is the unsteady counterpart of the residual check on a steady run, and
    it exists for the same reason: ``pimpleFoam`` exits 0 whether or not the
    statistics settled.
    """
    t = np.asarray(t, float)
    y = np.asarray(signal, float)
    sel = t >= t_start
    t, y = t[sel], y[sel]
    if t.size < 32:
        return {"statistically_steady": False, "reason": "averaging window too short"}

    half = t[0] + 0.5 * (t[-1] - t[0])
    a, b = y[t < half], y[t >= half]
    if a.size < 8 or b.size < 8:
        return {"statistically_steady": False, "reason": "averaging window too short"}

    rms = float(y.std())
    drift = abs(float(a.mean()) - float(b.mean()))
    # A signal with no fluctuation at all (a genuinely steady case run
    # unsteadily) would divide by zero; fall back to the mean magnitude.
    scale = max(rms, 0.01 * abs(float(y.mean())), 1e-6)
    cycles = (t[-1] - t[0]) * f_shed if f_shed else None

    ok_drift = drift < tol * scale
    ok_cycles = cycles is None or cycles >= min_cycles
    return {
        "statistically_steady": bool(ok_drift and ok_cycles),
        "half_window_drift": drift,
        "rms": rms,
        "n_cycles": cycles,
        "reason": (
            "ok" if ok_drift and ok_cycles
            else "; ".join(
                ([] if ok_drift else
                 [f"mean drifts by {drift:.3g} between window halves "
                  f"({drift / scale:.2f} of the RMS, tolerance {tol})"])
                + ([] if ok_cycles else
                   [f"window covers only {cycles:.1f} shedding cycles, "
                    f"need {min_cycles}"])
            )
        ),
    }


def analyse_case(case_dir: str | Path, up: UnsteadyParams,
                 alpha_deg: float, thickness: float) -> dict:
    """Everything the time series has to say about one unsteady case."""
    ts = read_time_series(case_dir)
    t, cl = ts["t"], ts.get("Cl")
    cd = ts.get("Cd")
    if cl is None:
        raise ValueError(f"{case_dir}: forceCoeffs has no Cl column")

    shed = shedding_frequency(t, cl, up.t_start_average)
    conv = statistics_converged(t, cl, up.t_start_average, f_shed=shed.get("f_shed"))
    sel = t >= up.t_start_average
    out = {
        "cl_mean": float(cl[sel].mean()),
        "cl_rms": float(cl[sel].std()),
        "cd_mean": float(cd[sel].mean()) if cd is not None else None,
        "cd_rms": float(cd[sel].std()) if cd is not None else None,
        "cm_mean": float(ts["Cm"][sel].mean()) if "Cm" in ts else None,
        "t_end_reached": float(t[-1]),
        "n_samples": int(sel.sum()),
        "f_shed": shed.get("f_shed"),
        "shedding_prominence": shed.get("prominence"),
        "organised_shedding": shed.get("organised"),
        "shedding_note": shed.get("reason"),
        "n_cycles": shed.get("n_cycles"),
        "statistically_steady": conv["statistically_steady"],
        "stationarity_note": conv["reason"],
        "half_window_drift": conv.get("half_window_drift"),
    }
    if out["f_shed"]:
        out["strouhal_chord"] = out["f_shed"]  # c = 1, Uinf = 1
        out["strouhal_projected"] = strouhal_projected(
            out["f_shed"], alpha_deg, thickness
        )
    return out


def read_mean_fields(case_dir: str | Path, grid: Grid):
    """Sample ``UMean``/``pMean`` onto the U-Net grid, with a stated fallback.

    If the averaged fields are missing -- a run that stopped before the first
    write past ``timeStart`` -- the instantaneous fields are used and the caller
    is told, because a single snapshot of a shedding wake is not a mean and
    should not be trained on as if it were.
    """
    import pyvista as pv

    case_dir = Path(case_dir)
    foam_file = case_dir / "case.foam"
    foam_file.touch()
    reader = pv.OpenFOAMReader(str(foam_file))
    reader.set_active_time_value(reader.time_values[-1])
    reader.cell_to_point_creation = True
    mesh = reader.read()
    internal = mesh["internalMesh"] if "internalMesh" in mesh.keys() else mesh[0]

    names = set(internal.array_names)
    u_name, p_name, averaged = "UMean", "pMean", True
    if "UMean" not in names or "pMean" not in names:
        u_name, p_name, averaged = "U", "p", False

    zc = float(internal.points[:, 2].mean())
    img = pv.ImageData(
        dimensions=(grid.nx, grid.ny, 1),
        spacing=(grid.dx, grid.dy, 1.0),
        origin=(grid.xlim[0], grid.ylim[0], zc),
    )
    sampled = img.sample(internal)
    vel = np.asarray(sampled[u_name]).reshape(grid.ny, grid.nx, 3)
    p = np.asarray(sampled[p_name]).reshape(grid.ny, grid.nx)
    valid = np.asarray(sampled["vtkValidPointMask"]).reshape(grid.ny, grid.nx).astype(bool)
    return vel[..., 0], vel[..., 1], p, valid, averaged


def run_one(
    dat_path: str | Path,
    alpha_deg: float,
    out_dir: str | Path,
    up: Optional[UnsteadyParams] = None,
    keep_case: bool = False,
    grid: Optional[Grid] = None,
) -> dict:
    """Solve and pack one URANS case. Never raises; failures come back as rows."""
    up = up or UnsteadyParams()
    dat_path = Path(dat_path)
    out_dir = Path(out_dir)
    grid = grid or Grid()
    foil = dat_path.stem.replace(".dat", "")
    tag = f"{foil}_a{alpha_deg:g}_urans"
    case = out_dir / "cases" / tag
    npz = out_dir / "npz" / f"{tag}.npz"
    npz.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "foil": foil,
        "series": dat_path.parent.name,
        "dat": str(dat_path),
        "alpha_deg": float(alpha_deg),
        "reynolds": float(up.reynolds),
        "npz": str(npz),
        "fidelity": "urans",
        "ok": False,
        "error": None,
        "converged": False,
    }
    try:
        from .run import convert_mesh

        af = load_dat(dat_path)
        msh = case.parent / f"{tag}.msh"
        case.mkdir(parents=True, exist_ok=True)
        build_mesh(af, alpha_deg, msh, MeshParams())
        write_unsteady_case(case, up)
        convert_mesh(case, msh)
        res = solve_unsteady(case)
        if res.returncode != 0:
            raise RuntimeError(f"pimpleFoam failed:\n{res.stdout[-1500:]}")

        stats = analyse_case(case, up, alpha_deg, af.thickness)
        u, v, p, valid, averaged = read_mean_fields(case, grid)
        if not averaged:
            stats["stationarity_note"] = (
                "no UMean/pMean in the last write: the fields below are an "
                "instantaneous snapshot, not a time average. "
                + str(stats.get("stationarity_note", ""))
            )
            stats["statistically_steady"] = False

        sample = build_case_arrays(
            af, alpha_deg, up.reynolds, grid, u, v, p,
            valid=valid, case_dir=case,
            cl=stats["cl_mean"], cd=stats["cd_mean"], cm=stats["cm_mean"],
            fidelity="urans",
            # For an unsteady case "converged" means the statistics settled.
            # Reusing the field keeps one convergence switch downstream instead
            # of two that mean different things.
            converged=stats["statistically_steady"],
            cl_std=stats["cl_rms"], cd_std=stats["cd_rms"],
            unsteady=stats,
        )
        packed = pack_npz(sample)
        np.savez_compressed(npz, **packed)
        row.update(_row_from_meta(json.loads(str(packed["meta"]))))
        row.update({
            "fidelity": "urans",
            "ok": True,
            "converged": stats["statistically_steady"],
            "convergence_note": stats["stationarity_note"],
            "cl_std": stats["cl_rms"],
            "cd_std": stats["cd_rms"],
            "time_averaged": bool(averaged),
        })
    except Exception as exc:  # keep the batch going; record the failure
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc()[-1500:]
    finally:
        # Never delete a case that failed: the logs and the force history are
        # the only way to tell "diverged" from "not yet settled".
        if not keep_case and row["ok"]:
            shutil.rmtree(case, ignore_errors=True)
    return row


def build_unsteady_dataset(
    dat_paths: Sequence[str | Path],
    alphas: Sequence[float] = DEFAULT_STALL_ALPHAS,
    out_dir: str | Path = "artifacts/cfd",
    up: Optional[UnsteadyParams] = None,
    max_workers: int = 1,
    keep_case: bool = False,
    progress: bool = True,
    resume: bool = True,
) -> dict:
    """Run a URANS sweep into the shared dataset directory.

    Point ``out_dir`` at the panel sweep's directory; ``finalize_manifest``
    merges the rows and prefers the unsteady one wherever it exists, then
    ``python cli.py sensor-cache`` rebuilds the training cache.

    A case here costs hours, so rows are flushed to the chunk file the moment
    each one finishes and ``resume`` (default on) skips any (foil, alpha) that
    already has an ``ok`` URANS row. A steady or panel row does *not* count as
    solved -- re-solving those at URANS is the whole point.
    """
    up = up or UnsteadyParams()
    out_dir = Path(out_dir)
    rows_dir = out_dir / "rows"
    jobs = [(Path(p), float(a)) for p in dat_paths for a in alphas]

    t0 = time.time()
    worker = partial(run_one, out_dir=out_dir, up=up, keep_case=keep_case)
    rows, chunk, skipped = run_sweep(
        jobs, worker, rows_dir, prefix="urans", fidelity="urans",
        max_workers=max_workers, resume=resume, progress=progress, log=_log,
    )

    ok = [r for r in rows if r["ok"]]
    steady = [r for r in ok if r.get("converged")]
    return {
        "chunk": str(chunk),
        "cases": len(rows),
        "ok": len(ok),
        "statistically_steady": len(steady),
        "failed": len(rows) - len(ok),
        "skipped_already_solved": len(skipped),
        "seconds": round(time.time() - t0, 1),
        "shedding": [
            {"foil": r["foil"], "alpha": r["alpha_deg"],
             "St_c": r.get("strouhal_chord"), "St_d": r.get("strouhal_projected"),
             "cl_mean": r.get("cl"), "cl_rms": r.get("cl_std")}
            for r in ok
        ][:20],
        "failures": [
            {"foil": r["foil"], "alpha": r["alpha_deg"], "error": r["error"]}
            for r in rows if not r["ok"]
        ][:10],
        "next": "python cli.py sensor-cache   # after finalize_manifest",
    }


def _log(row: dict, done: int, total: int, t0: float) -> None:
    status = "ok" if row["ok"] else f"FAIL {row['error']}"
    if row["ok"] and not row.get("converged"):
        status = "ok (statistics not settled)"
    el = time.time() - t0
    st = row.get("strouhal_projected")
    print(
        f"[{done:3d}/{total}] {row['foil']} a={row['alpha_deg']:g} "
        f"Cl={row.get('cl')} St_d={st if st is None else round(st, 3)} {status}  "
        f"({el:.0f}s elapsed, eta {el / done * (total - done):.0f}s)",
        flush=True,
    )
