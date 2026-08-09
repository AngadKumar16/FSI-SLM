"""Real PIV ingestion: DaVis DSExport CSV frames -> RawExperiment -> FeatureRecord.

This is the concrete "future real-data pipeline" that :mod:`Data.schema` and
:mod:`features.extract` were written for. It converts a directory of DaVis PIV
vector exports (one CSV per snapshot) into a schema-valid
:class:`~Data.schema.RawExperiment` and, through the *unchanged*
:func:`features.extract.extract_features`, the model-facing
:class:`~Data.schema.FeatureRecord`.

What the DaVis files actually are
---------------------------------
The exports in ``Exports/`` are **mean-subtracted fluctuation fields**
(the frame ``Originator`` tag reads ``... Vector Arithmetic`` and the time-mean
velocity is ~0 at every domain border). This is not a defect -- it is exactly
the fluctuating-velocity product POD wants -- but it drives every modelling
choice below, all made honestly:

* ``u',v'`` snapshots feed POD (SVD) directly; they are already fluctuations.
* The vortex-shedding frequency is the dominant peak of the *wake* fluctuation
  spectrum (transverse velocity ``v'``), not of a raw velocity signal.
* Freestream velocity ``U`` and chord ``c`` cannot be recovered from a
  mean-subtracted field. They are **rig inputs** (documented defaults, both
  overridable) used only to non-dimensionalise (Strouhal, Reynolds). Everything
  else is measured from the frames.
* Mean-field-only quantities (chordwise separation point) are supplied from
  geometry/physics with a provenance note in ``RawExperiment.notes`` -- never
  fabricated as if measured.
* Structural displacement and force coefficients come from separate subsystems
  (load cell, video strain tracking) and are absent from a PIV-only export.
  They are written as zero signals with a provenance note so the single
  downstream code path runs unchanged and the record's own validators
  (``range_warnings`` / ``check_physical_consistency``) transparently flag them.

The result is a record whose PIV block is real measurement and whose absent
channels are explicit, not invented.
"""

from __future__ import annotations

import glob
import math
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from Data.schema import FlowConditions, FlowRegime, RawExperiment
from features.extract import extract_features

# --------------------------------------------------------------------------- #
# Rig / fluid defaults (documented, overridable)
# --------------------------------------------------------------------------- #
WATER_KINEMATIC_VISCOSITY_M2_S = 1.0e-6  # water at ~20 C

# Freestream is a rig constant (constant-flow pump through a fixed nozzle); it is
# taken from the same constant the synthetic generator uses so the real and
# synthetic pipelines share one documented operating point. It is NOT measurable
# from a mean-subtracted PIV export.
try:  # pragma: no cover - trivial import guard
    from Data.synthetic import FIXED_FREESTREAM_VELOCITY_M_S as _RIG_FREESTREAM
except Exception:  # pragma: no cover
    _RIG_FREESTREAM = 1.569

DEFAULT_FREESTREAM_M_S = float(_RIG_FREESTREAM)
DEFAULT_CHORD_M = 0.100  # NACA0009 printed foil (see project bill of materials)

# Number of spatial points kept for the POD snapshot stack. POD energy fractions
# are invariant to modest spatial subsampling; capping the point count keeps the
# RawExperiment JSON-serialisable and the SVD fast without changing the answer.
POD_MAX_POINTS = 3000


# --------------------------------------------------------------------------- #
# Loaded series container
# --------------------------------------------------------------------------- #
@dataclass
class PIVSeries:
    """A time-resolved DaVis PIV export loaded onto its regular grid.

    Arrays ``U``, ``V``, ``valid`` have shape ``(n_frames, height, width)``.
    Row 0 is the top of the field (largest Y); column 0 is the smallest X. ``U``
    is the streamwise (X) velocity fluctuation, ``V`` the transverse (Y).
    """

    X: np.ndarray            # (H, W) metres
    Y: np.ndarray            # (H, W) metres
    U: np.ndarray            # (T, H, W) m/s (fluctuation)
    V: np.ndarray            # (T, H, W) m/s (fluctuation)
    valid: np.ndarray        # (T, H, W) bool
    times: np.ndarray        # (T,) seconds
    aoa_deg: float
    run_id: str
    source_dir: str

    @property
    def n_frames(self) -> int:
        return self.U.shape[0]

    @property
    def fs_hz(self) -> float:
        dt = float(np.median(np.diff(self.times)))
        return 1.0 / dt if dt > 0 else 0.0

    @property
    def dx(self) -> float:
        return float(np.median(np.diff(self.X[0])))

    @property
    def dy(self) -> float:
        return float(abs(np.median(np.diff(self.Y[:, 0]))))

    def common_valid_mask(self, min_frac: float = 0.90) -> np.ndarray:
        """(H, W) mask of grid points valid in >= ``min_frac`` of frames."""
        return self.valid.mean(axis=0) >= min_frac


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
_TIME_RE = re.compile(r"-([0-9]+\.[0-9]+)\.csv$")
_AOA_RE = re.compile(r"AOA\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_RUN_RE = re.compile(r"(Run[^.]*)", re.IGNORECASE)


def frame_time_seconds(path: str) -> float:
    """Extract the timestamp (seconds) encoded in a DaVis export filename.

    Filenames look like ``Export.<id>.000042-2799.9710.csv`` where the trailing
    number is milliseconds. Returns 0.0 if not present.
    """
    m = _TIME_RE.search(os.path.basename(path))
    return float(m.group(1)) / 1000.0 if m else 0.0


def parse_frame_header(path: str) -> Dict[str, object]:
    """Read the small DaVis text header (grid size, angle of attack, run id)."""
    width = height = 0
    aoa_deg = float("nan")
    run_id = ""
    with open(path, "r", errors="replace") as fh:
        for _ in range(8):  # header is the first several lines before >>*DATA*<<
            line = fh.readline()
            if not line:
                break
            if line.startswith("GridSize"):
                gm = re.search(r"Width=(\d+),\s*Height=(\d+)", line)
                if gm:
                    width, height = int(gm.group(1)), int(gm.group(2))
            elif line.startswith("Originator"):
                am = _AOA_RE.search(line)
                if am:
                    aoa_deg = float(am.group(1))
                rm = _RUN_RE.search(line)
                if rm:
                    run_id = rm.group(1).strip()
    return {"width": width, "height": height, "aoa_deg": aoa_deg, "run_id": run_id}


def _read_frame_uvm(path: str, height: int, width: int):
    """Return (U, V, valid) each (height, width) for one DaVis CSV frame."""
    import pandas as pd

    # skiprows=8 leaves the "X (m)[m]",...,"Length[m/s]" line as the header row.
    df = pd.read_csv(path, skiprows=8, usecols=[3, 4, 5])
    status = df.iloc[:, 0].to_numpy().astype(str)
    u = df.iloc[:, 1].to_numpy(dtype=float)
    v = df.iloc[:, 2].to_numpy(dtype=float)
    valid = np.char.find(status, "Valid") >= 0  # "Valid" / "Valid subst" -> True
    n = height * width
    if u.size != n:
        raise ValueError(f"{path}: expected {n} vectors, got {u.size}")
    return (
        u.reshape(height, width),
        v.reshape(height, width),
        valid.reshape(height, width),
    )


def load_piv_series(
    export_dir: str,
    pattern: str = "Export.*.csv",
    max_frames: Optional[int] = None,
) -> PIVSeries:
    """Load a directory of DaVis DSExport CSV frames into a :class:`PIVSeries`.

    Frames are ordered by their embedded timestamp. The spatial grid and its
    coordinates are read once from the first frame (they are identical across
    frames); only U, V and the validity mask are read per frame.
    """
    files = sorted(glob.glob(os.path.join(export_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"no PIV frames matching {pattern!r} in {export_dir!r}")
    files.sort(key=frame_time_seconds)
    if max_frames is not None:
        files = files[:max_frames]

    hdr = parse_frame_header(files[0])
    width, height = int(hdr["width"]), int(hdr["height"])
    if width == 0 or height == 0:
        raise ValueError(f"could not parse GridSize from {files[0]}")

    import pandas as pd

    coords = pd.read_csv(files[0], skiprows=8, usecols=[0, 1])
    X = coords.iloc[:, 0].to_numpy(dtype=float).reshape(height, width)
    Y = coords.iloc[:, 1].to_numpy(dtype=float).reshape(height, width)

    n = len(files)
    U = np.empty((n, height, width), dtype=np.float32)
    V = np.empty((n, height, width), dtype=np.float32)
    valid = np.empty((n, height, width), dtype=bool)
    times = np.empty(n, dtype=float)
    for i, f in enumerate(files):
        u, v, m = _read_frame_uvm(f, height, width)
        U[i], V[i], valid[i] = u, v, m
        times[i] = frame_time_seconds(f)

    return PIVSeries(
        X=X, Y=Y, U=U, V=V, valid=valid, times=times,
        aoa_deg=float(hdr["aoa_deg"]),
        run_id=str(hdr["run_id"]) or "run",
        source_dir=export_dir,
    )


# --------------------------------------------------------------------------- #
# Field-derived quantities
# --------------------------------------------------------------------------- #
def _masked(series: PIVSeries) -> Tuple[np.ndarray, np.ndarray]:
    """Return (U, V) with invalid samples set to NaN, shape (T, H, W)."""
    inv = ~series.valid
    U = series.U.astype(float).copy()
    V = series.V.astype(float).copy()
    U[inv] = np.nan
    V[inv] = np.nan
    return U, V


def tke_field(series: PIVSeries) -> np.ndarray:
    """Per-point turbulent kinetic energy 0.5*(u'^2 + v'^2), shape (H, W) [m^2/s^2].

    The exports are already fluctuations, so the temporal std *is* the RMS
    fluctuation; TKE follows directly. Points with too few valid frames become
    NaN (all-NaN slices are expected at the field edges) and are ignored by
    downstream masks.
    """
    U, V = _masked(series)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 0.5 * (np.nanstd(U, axis=0) ** 2 + np.nanstd(V, axis=0) ** 2)


def wake_mask(series: PIVSeries, percentile: float = 60.0) -> np.ndarray:
    """(H, W) mask of the energetic wake: common-valid points whose TKE exceeds
    the given percentile of the common-valid TKE distribution."""
    common = series.common_valid_mask()
    tke = tke_field(series)
    vals = tke[common & np.isfinite(tke)]
    if vals.size == 0:
        return common
    thr = np.percentile(vals, percentile)
    return common & (tke > thr)


def _detrend_columns(x: np.ndarray) -> np.ndarray:
    """Remove a per-column linear trend (kills slow drift / record-length ramps).

    ``x`` is (T, N). Columns that are constant are returned mean-removed only.
    """
    t = np.arange(x.shape[0], dtype=float)
    t = t - t.mean()
    denom = float((t * t).sum())
    if denom <= 0:
        return x - x.mean(axis=0, keepdims=True)
    slope = (t[:, None] * (x - x.mean(axis=0, keepdims=True))).sum(axis=0) / denom
    return x - x.mean(axis=0, keepdims=True) - np.outer(t, slope)


def shedding_analysis(
    series: PIVSeries,
    wake: Optional[np.ndarray] = None,
    f_low_hz: float = 0.3,
) -> Dict[str, object]:
    """Locate the vortex-shedding tone and a representative wake probe signal.

    Strategy (robust to the low frame rate and broadband stall wake):
      1. Detrend every wake point's transverse fluctuation ``v'(t)``.
      2. Average their periodograms into one wake spectrum; its dominant peak
         above ``f_low_hz`` is ``shedding_frequency_hz`` (drift below f_low is
         ignored).
      3. Choose the single wake point with the most energy in a band around that
         peak as the probe, and return its *detrended* ``v'(t)`` so the shared
         :func:`features.extract.dominant_frequency` recovers the same tone.
    """
    if wake is None:
        wake = wake_mask(series)
    fs = series.fs_hz
    T = series.n_frames
    _, V = _masked(series)
    Vflat = V.reshape(T, -1)
    idx = np.where(wake.ravel())[0]
    # keep only fully-finite wake columns for the spectrum
    cols = idx[np.all(np.isfinite(Vflat[:, idx]), axis=0)]
    if cols.size == 0 or fs <= 0 or T < 8:
        return {
            "shedding_frequency_hz": 0.0,
            "peak_factor": 0.0,
            "probe_signal": [0.0] * T,
            "probe_xy": (float("nan"), float("nan")),
            "spectrum_freqs": [],
            "spectrum_power": [],
        }

    Vw = _detrend_columns(Vflat[:, cols])
    win = np.hanning(T)
    spec = np.abs(np.fft.rfft(Vw * win[:, None], axis=0)) ** 2  # (F, ncols)
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)
    wake_spec = spec.mean(axis=1)  # average periodogram over wake

    band = freqs > f_low_hz
    if not np.any(band):
        band = freqs > 0
    k_rel = int(np.argmax(wake_spec[band]))
    f_shed = float(freqs[band][k_rel])
    baseline = float(np.median(wake_spec[band]) + 1e-30)
    peak_factor = float(wake_spec[band][k_rel] / baseline)

    # Probe = wake point with most energy in +/-30% band around f_shed.
    lo, hi = 0.7 * f_shed, 1.3 * f_shed
    fband = (freqs >= lo) & (freqs <= hi)
    if not np.any(fband):
        fband = band
    energy = spec[fband, :].sum(axis=0)
    best = int(np.argmax(energy))
    probe_col = cols[best]
    probe_signal = Vw[:, best]
    j, i = np.unravel_index(probe_col, (series.U.shape[1], series.U.shape[2]))
    return {
        "shedding_frequency_hz": f_shed,
        "peak_factor": peak_factor,
        "probe_signal": [float(x) for x in probe_signal],
        "probe_xy": (float(series.X[j, i]), float(series.Y[j, i])),
        "spectrum_freqs": [float(x) for x in freqs],
        "spectrum_power": [float(x) for x in wake_spec],
    }


def pod_snapshot_stack(
    series: PIVSeries, max_points: int = POD_MAX_POINTS
) -> Tuple[np.ndarray, int]:
    """Build the (n_frames, 2*n_points) fluctuating-velocity POD stack.

    Uses points valid in every frame, spatially subsampled to at most
    ``max_points`` (POD energy fractions are invariant to modest subsampling).
    Returns (stack, n_points_used).
    """
    common = series.common_valid_mask(min_frac=1.0)
    idx = np.where(common.ravel())[0]
    if idx.size == 0:
        common = series.common_valid_mask(min_frac=0.95)
        idx = np.where(common.ravel())[0]
    if idx.size > max_points:
        stride = int(math.ceil(idx.size / max_points))
        idx = idx[::stride]
    T = series.n_frames
    U = series.U.reshape(T, -1)[:, idx].astype(float)
    V = series.V.reshape(T, -1)[:, idx].astype(float)
    stack = np.hstack([U, V])
    return stack, idx.size


def wake_geometry(
    series: PIVSeries, chord_m: float, wake: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """TKE-weighted PCA of the wake region -> streamwise length and width /c.

    Orientation-agnostic: the minor principal axis is the wake width, the major
    axis its streamwise length. 2-sigma extents are used.
    """
    if wake is None:
        wake = wake_mask(series)
    tke = tke_field(series)
    jj, ii = np.where(wake)
    if jj.size < 3:
        return {"wake_width_over_c": 0.0, "wake_length_over_c": 0.0,
                "centroid_x": float("nan"), "centroid_y": float("nan")}
    pts = np.c_[series.X[jj, ii], series.Y[jj, ii]]
    w = tke[jj, ii]
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    if w.sum() <= 0:
        w = np.ones_like(w)
    cen = np.average(pts, axis=0, weights=w)
    d = np.ascontiguousarray(pts - cen, dtype=np.float64)
    wsum = float(w.sum())
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        cov = (d * w[:, None]).T @ d / wsum
    val, _ = np.linalg.eigh(cov)  # ascending
    minor = 2.0 * math.sqrt(max(val[0], 0.0))
    major = 2.0 * math.sqrt(max(val[1], 0.0))
    return {
        "wake_width_over_c": float(minor / chord_m),
        "wake_length_over_c": float(major / chord_m),
        "centroid_x": float(cen[0]),
        "centroid_y": float(cen[1]),
    }


def wake_tke_normalized(
    series: PIVSeries, freestream_m_s: float, wake: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Wake-mean TKE, both dimensional [m^2/s^2] and normalised by U^2."""
    if wake is None:
        wake = wake_mask(series)
    tke = tke_field(series)
    vals = tke[wake & np.isfinite(tke)]
    tke_dim = float(vals.mean()) if vals.size else 0.0
    tke_norm = tke_dim / (freestream_m_s ** 2) if freestream_m_s > 0 else 0.0
    ti = (
        math.sqrt(max(2.0 / 3.0 * tke_dim, 0.0)) / freestream_m_s
        if freestream_m_s > 0 else 0.0
    )
    return {"tke_dimensional_m2_s2": tke_dim,
            "turbulent_kinetic_energy": tke_norm,
            "turbulence_intensity": ti}


def separation_from_physics(aoa_deg: float) -> Tuple[float, str]:
    """Chordwise separation x/c from stall physics (mean field absent in export).

    NACA0009 is a thin symmetric section with static stall near 9-10 deg. This
    returns a documented estimate, not a measurement, and says so.
    """
    a = abs(aoa_deg)
    if a >= 12.0:
        return 0.10, (
            f"separation_location x/c=0.10 is a physics-based estimate: AoA={aoa_deg:g} deg "
            "is deep stall for NACA0009 (static stall ~9-10 deg), so separation is near the "
            "leading edge. The mean velocity field needed to measure it directly was removed "
            "in this fluctuation-only DaVis export."
        )
    if a >= 9.0:
        return 0.35, (
            f"separation_location x/c=0.35 estimated: AoA={aoa_deg:g} deg is near NACA0009 "
            "static stall; mean field unavailable in this fluctuation-only export."
        )
    return 0.90, (
        f"separation_location x/c=0.90 estimated (attached): AoA={aoa_deg:g} deg is below "
        "NACA0009 static stall; mean field unavailable in this fluctuation-only export."
    )


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def build_raw_experiment(
    series: PIVSeries,
    chord_m: float = DEFAULT_CHORD_M,
    freestream_m_s: float = DEFAULT_FREESTREAM_M_S,
    nu_m2_s: float = WATER_KINEMATIC_VISCOSITY_M2_S,
    experiment_id: Optional[str] = None,
    separation_x_over_c: Optional[float] = None,
) -> Tuple[RawExperiment, Dict[str, object]]:
    """Assemble a schema-valid :class:`RawExperiment` from a loaded PIV series.

    Returns ``(raw_experiment, diagnostics)``. Structural and force channels are
    zero (absent from a PIV-only export); the PIV block is real measurement.
    """
    T = series.n_frames
    fs = series.fs_hz
    aoa = series.aoa_deg if np.isfinite(series.aoa_deg) else 0.0

    wake = wake_mask(series)
    shed = shedding_analysis(series, wake=wake)
    geom = wake_geometry(series, chord_m, wake=wake)
    tke = wake_tke_normalized(series, freestream_m_s, wake=wake)
    stack, n_pts = pod_snapshot_stack(series)

    if separation_x_over_c is None:
        sep_xc, sep_note = separation_from_physics(aoa)
    else:
        sep_xc, sep_note = float(separation_x_over_c), "separation_location x/c supplied by caller."

    reynolds = freestream_m_s * chord_m / nu_m2_s

    exp_id = experiment_id or f"real_piv_AOA{aoa:g}_{series.run_id}".replace(" ", "_").replace(":", "")

    notes = (
        "REAL PIV run ingested by features.piv_ingest. DaVis DSExport frames are "
        "mean-subtracted fluctuation fields (Originator 'Vector Arithmetic'); "
        f"n_frames={T}, fs={fs:.3f} Hz, duration={series.times[-1]-series.times[0]:.2f} s, "
        f"grid={series.U.shape[2]}x{series.U.shape[1]} (dx={series.dx*1e3:.2f} mm), "
        f"POD points={n_pts}. "
        f"freestream_velocity={freestream_m_s:.3f} m/s and chord={chord_m:g} m are RIG INPUTS "
        "(not recoverable from a mean-subtracted field), used only for St and Re. "
        f"Reynolds={reynolds:.3e}. "
        f"shedding tone peak/median={shed['peak_factor']:.1f}. "
        + sep_note
        + " Structural displacement and Cl/Cd/Cm are acquired by separate subsystems "
        "(load cell, video strain tracking) and are ABSENT here -- written as zeros so the "
        "shared feature-extraction path runs unchanged; downstream validators flag them."
    )

    conditions = FlowConditions(
        reynolds_number=float(reynolds),
        angle_of_attack_deg=float(aoa),
        freestream_velocity_m_s=float(freestream_m_s),
        reduced_velocity=0.0,  # requires structural f_n (not acquired) -> flagged by validators
        chord_m=float(chord_m),
    )

    zeros = [0.0] * T
    raw = RawExperiment(
        experiment_id=exp_id,
        conditions=conditions,
        sampling_rate_hz=float(fs),
        structural_displacement_signal=zeros,           # not acquired (PIV-only export)
        wake_velocity_probe_signal=list(shed["probe_signal"]),  # real: v'(t) at wake probe
        cl_signal=zeros,                                 # not acquired
        cd_signal=zeros,                                 # not acquired
        cm_signal=zeros,                                 # not acquired
        piv_snapshots=[[float(x) for x in row] for row in stack],  # real fluctuation stack -> POD
        natural_frequency_hz=0.0,                        # no structural channel -> lock-in undetermined
        mode_number=1,
        camber=0.0,                                      # NACA0009 is symmetric (real)
        wake_width_over_c=float(geom["wake_width_over_c"]),
        separation_location_x_over_c=float(sep_xc),
        turbulent_kinetic_energy=float(tke["turbulent_kinetic_energy"]),
        pressure=101325.0,
        strain=0.0,
        ground_truth_regime=None,  # real experiment: no answer key
        notes=notes,
    )

    diagnostics = {
        "n_frames": T,
        "fs_hz": fs,
        "duration_s": float(series.times[-1] - series.times[0]),
        "grid": [int(series.U.shape[2]), int(series.U.shape[1])],
        "dx_mm": series.dx * 1e3,
        "aoa_deg": aoa,
        "run_id": series.run_id,
        "chord_m": chord_m,
        "freestream_m_s": freestream_m_s,
        "reynolds_number": reynolds,
        "pod_points": n_pts,
        "shedding": {k: shed[k] for k in ("shedding_frequency_hz", "peak_factor", "probe_xy")},
        "spectrum": {"freqs": shed["spectrum_freqs"], "power": shed["spectrum_power"]},
        "wake_geometry": geom,
        "tke": tke,
        "separation_note": sep_note,
    }
    return raw, diagnostics


def ingest_piv_directory(
    export_dir: str,
    chord_m: float = DEFAULT_CHORD_M,
    freestream_m_s: float = DEFAULT_FREESTREAM_M_S,
    nu_m2_s: float = WATER_KINEMATIC_VISCOSITY_M2_S,
    pattern: str = "Export.*.csv",
    max_frames: Optional[int] = None,
    experiment_id: Optional[str] = None,
    separation_x_over_c: Optional[float] = None,
):
    """End-to-end: load DaVis frames -> RawExperiment -> FeatureRecord.

    Returns ``(feature_record, raw_experiment, series, diagnostics)``. The
    FeatureRecord is produced by the *unchanged* :func:`features.extract.extract_features`,
    so the real-data path is byte-for-byte the synthetic path from here on.
    """
    series = load_piv_series(export_dir, pattern=pattern, max_frames=max_frames)
    raw, diagnostics = build_raw_experiment(
        series,
        chord_m=chord_m,
        freestream_m_s=freestream_m_s,
        nu_m2_s=nu_m2_s,
        experiment_id=experiment_id,
        separation_x_over_c=separation_x_over_c,
    )
    feature_record = extract_features(raw)
    return feature_record, raw, series, diagnostics


__all__ = [
    "PIVSeries",
    "load_piv_series",
    "parse_frame_header",
    "frame_time_seconds",
    "tke_field",
    "wake_mask",
    "shedding_analysis",
    "pod_snapshot_stack",
    "wake_geometry",
    "wake_tke_normalized",
    "separation_from_physics",
    "build_raw_experiment",
    "ingest_piv_directory",
    "DEFAULT_CHORD_M",
    "DEFAULT_FREESTREAM_M_S",
    "WATER_KINEMATIC_VISCOSITY_M2_S",
]
