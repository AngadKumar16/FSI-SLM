"""Instrumentation-fault detection for FSI experiment records.

The interpreter is supposed to flag "likely sensor noise". That is a different
question from "is this record physically consistent" -- :func:`features.extract.
check_physical_consistency` already answers the second one by cross-checking
Strouhal against ``f c / U`` and so on. This module answers the first: it looks
at the *raw signals* for the failure modes that a PIV rig, a load cell and a
laser displacement sensor actually exhibit, and reports them with a severity so
the language model can say "the 43 Hz peak is mains pickup, not shedding"
instead of dutifully interpreting an artefact.

Detectors, and why each one is here:

``piv_dropout``
    Fraction of PIV vectors that are identically zero. Correlation failure in a
    low-seeding-density region gives exactly this signature. Above a few percent
    the POD modes are contaminated.
``piv_outliers``
    Universal median test (Westerweel & Scarano, *Exp. Fluids* 39, 2005): a
    vector is spurious if its residual from the local median exceeds a multiple
    of the local median residual. This is the standard PIV validation test.
``piv_frozen_pixels``
    Interrogation windows whose value never changes across the snapshot stack --
    a dead CCD region or a masked area that was not excluded.
``clipping``
    Samples sitting exactly on the extreme value of a signal. A load cell driven
    past its range or an ADC railing produces flat tops, which bias the RMS down
    and inject spurious harmonics.
``quantisation``
    Too few distinct levels for the number of samples: the ADC resolution is
    coarser than the signal being measured, so small-amplitude spectral content
    is not real.
``spikes``
    Isolated samples whose first difference is many robust standard deviations
    from typical -- electrical interference, or a laser dropout on a shiny
    surface.
``drift``
    A linear trend comparable to the fluctuation amplitude. Thermal drift in a
    strain gauge, or a slow zero shift in a load cell.
``flatline``
    A run of identical samples: the sensor stopped responding.
``spectral_snr``
    Peak-to-floor ratio at the dominant frequency. Below ~3 the "dominant
    frequency" is not distinguishable from broadband noise and any Strouhal
    number derived from it is meaningless.
``aliasing``
    Energy piling up at the top of the band, or a dominant peak above 40% of
    Nyquist. Anti-alias filtering or sample rate is wrong and the reported
    frequency may be a fold-down of a higher one.
``mains_interference``
    A narrow peak within a tolerance of 50 or 60 Hz (and harmonics) that is not
    explained by the shedding or structural frequency.

Every detector returns a normalised value, the threshold it was compared
against, and a severity in ``{"info", "warning", "critical"}``. Nothing here
modifies the record: detection and interpretation stay separate, so a flagged
experiment is still reported with its original numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

SEVERITY_ORDER = {"info": 0, "warning": 1, "critical": 2}

# Thresholds. Chosen from common practice rather than tuned: PIV validation
# literature treats >5% spurious vectors as a poor field, load-cell clipping is
# unacceptable at any level above a rounding artefact, and a spectral peak needs
# roughly a factor of 3 over the floor to be callable.
THRESHOLDS = {
    "piv_dropout": 0.02,
    "piv_outliers": 0.05,
    "piv_frozen_pixels": 0.01,
    "clipping": 0.005,
    "quantisation_levels": 32,
    "spikes": 0.005,
    "drift_ratio": 0.5,
    "flatline_run": 20,
    "spectral_snr": 3.0,
    "aliasing_band_fraction": 0.25,
    "mains_tolerance_hz": 1.5,
}
MAINS_FREQUENCIES_HZ = (50.0, 60.0)


@dataclass
class NoiseFlag:
    name: str
    severity: str
    value: float
    threshold: float
    channel: str
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class NoiseReport:
    flags: List[NoiseFlag] = field(default_factory=list)
    channels_checked: List[str] = field(default_factory=list)
    quality_score: float = 1.0
    verdict: str = "clean"
    summary: str = ""

    @property
    def critical(self) -> List[NoiseFlag]:
        return [f for f in self.flags if f.severity == "critical"]

    @property
    def warnings(self) -> List[NoiseFlag]:
        return [f for f in self.flags if f.severity == "warning"]

    def to_dict(self) -> dict:
        return {
            "flags": [f.to_dict() for f in self.flags],
            "channels_checked": self.channels_checked,
            "quality_score": round(self.quality_score, 3),
            "verdict": self.verdict,
            "summary": self.summary,
        }

    def to_prompt_block(self, max_lines: int = 8) -> str:
        """Render as a prompt section. Empty string when nothing was found."""
        if not self.flags:
            return ""
        lines = [
            "[data_quality]",
            f"verdict: {self.verdict} (score {self.quality_score:.2f})",
        ]
        for f in sorted(
            self.flags, key=lambda f: -SEVERITY_ORDER[f.severity]
        )[:max_lines]:
            lines.append(f"- {f.severity.upper()} {f.channel}/{f.name}: {f.message}")
        if len(self.flags) > max_lines:
            lines.append(f"- ... {len(self.flags) - max_lines} more")
        return "\n".join(lines)


# --------------------------------------------------------------------- helpers


def _robust_sigma(x: np.ndarray) -> float:
    """MAD-based standard deviation estimate; immune to the spikes we hunt."""
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad)


def _dominant_peak(x: np.ndarray, fs: float):
    """Return ``(freq, peak_power, noise_floor)`` excluding DC."""
    x = np.asarray(x, float)
    n = x.size
    if n < 16 or fs <= 0:
        return 0.0, 0.0, 0.0
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft((x - x.mean()) * win)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    if spec.size < 3:
        return 0.0, 0.0, 0.0
    spec[0] = 0.0
    k = int(np.argmax(spec))
    peak = float(spec[k])
    # noise floor = median power away from the peak and its immediate skirt
    keep = np.ones(spec.size, dtype=bool)
    lo, hi = max(1, k - 3), min(spec.size, k + 4)
    keep[lo:hi] = False
    keep[0] = False
    floor = float(np.median(spec[keep])) if keep.any() else 0.0
    return float(freqs[k]), peak, floor


def _longest_constant_run(x: np.ndarray) -> int:
    if x.size == 0:
        return 0
    same = np.diff(x) == 0
    if not same.any():
        return 1
    best = run = 1
    for s in same:
        run = run + 1 if s else 1
        best = max(best, run)
    return int(best)


# ------------------------------------------------------------------- detectors


def check_signal(
    x: Sequence[float],
    channel: str,
    sampling_rate_hz: float,
    expected_frequencies_hz: Iterable[float] = (),
) -> List[NoiseFlag]:
    """Run every time-series detector on one channel."""
    arr = np.asarray(list(x), dtype=float)
    flags: List[NoiseFlag] = []
    if arr.size < 8:
        return flags
    if not np.all(np.isfinite(arr)):
        bad = float(np.mean(~np.isfinite(arr)))
        flags.append(NoiseFlag(
            "non_finite", "critical", bad, 0.0, channel,
            f"{bad:.1%} of samples are NaN or Inf",
        ))
        arr = np.nan_to_num(arr)

    span = float(arr.max() - arr.min())
    sigma = _robust_sigma(arr)

    # clipping: samples exactly at either rail
    if span > 0:
        at_rail = float(np.mean((arr == arr.max()) | (arr == arr.min())))
        if at_rail > THRESHOLDS["clipping"]:
            flags.append(NoiseFlag(
                "clipping", "critical" if at_rail > 0.02 else "warning",
                at_rail, THRESHOLDS["clipping"], channel,
                f"{at_rail:.1%} of samples sit exactly on the signal's min/max "
                f"-- the channel is railing, so RMS is biased low",
            ))

    # quantisation
    levels = int(np.unique(arr).size)
    if arr.size >= 64 and levels < THRESHOLDS["quantisation_levels"]:
        flags.append(NoiseFlag(
            "quantisation", "warning", float(levels),
            float(THRESHOLDS["quantisation_levels"]), channel,
            f"only {levels} distinct values across {arr.size} samples -- ADC "
            f"resolution is coarser than the fluctuation being measured",
        ))

    # flatline
    run = _longest_constant_run(arr)
    if run >= THRESHOLDS["flatline_run"]:
        flags.append(NoiseFlag(
            "flatline", "critical", float(run), float(THRESHOLDS["flatline_run"]),
            channel, f"{run} consecutive identical samples -- sensor stopped responding",
        ))

    # spikes on the first difference
    d = np.diff(arr)
    ds = _robust_sigma(d)
    if ds > 0:
        frac = float(np.mean(np.abs(d - np.median(d)) > 8.0 * ds))
        if frac > THRESHOLDS["spikes"]:
            flags.append(NoiseFlag(
                "spikes", "warning" if frac < 0.02 else "critical",
                frac, THRESHOLDS["spikes"], channel,
                f"{frac:.1%} of samples jump more than 8 robust sigma between "
                f"consecutive points -- impulsive interference or dropout",
            ))

    # drift
    if sigma > 0:
        t = np.arange(arr.size, dtype=float)
        slope = float(np.polyfit(t, arr, 1)[0])
        drift = abs(slope) * arr.size / (sigma + 1e-12)
        if drift > THRESHOLDS["drift_ratio"]:
            flags.append(NoiseFlag(
                "drift", "warning", drift, THRESHOLDS["drift_ratio"], channel,
                f"end-to-end linear trend is {drift:.1f}x the fluctuation "
                f"amplitude -- thermal drift or zero shift, subtract before "
                f"trusting the mean",
            ))

    # spectral quality
    f_peak, peak, floor = _dominant_peak(arr, sampling_rate_hz)
    if floor > 0:
        snr = peak / floor
        # A flat periodogram of pure noise still has a largest bin: for m
        # exponentially distributed bins the expected maximum is ln(m) and the
        # median is ln 2, so peak/floor lands near ln(m)/ln 2 -- about 10 for a
        # 2048-sample record -- with no signal present at all. Comparing against
        # a fixed 3 would therefore never fire. The bar scales with record
        # length, times a 1.5 safety factor.
        m_bins = max(arr.size // 2, 2)
        snr_min = max(
            THRESHOLDS["spectral_snr"], 1.5 * float(np.log(m_bins) / np.log(2.0))
        )
        if snr < snr_min:
            flags.append(NoiseFlag(
                "spectral_snr", "critical" if snr < 0.5 * snr_min else "warning",
                float(snr), float(snr_min), channel,
                f"dominant peak at {f_peak:.1f} Hz is only {snr:.1f}x the "
                f"broadband floor (needs > {snr_min:.0f} to stand out of noise "
                f"in a {arr.size}-sample record) -- this frequency is not "
                f"reliably resolved",
            ))

    nyquist = 0.5 * sampling_rate_hz
    if nyquist > 0:
        if f_peak > 0.4 * nyquist:
            flags.append(NoiseFlag(
                "aliasing", "warning", f_peak / nyquist, 0.4, channel,
                f"dominant peak at {f_peak:.1f} Hz is {f_peak / nyquist:.0%} of "
                f"Nyquist -- may be a fold-down of a higher frequency",
            ))
        # energy piling at the top of the band
        win = np.hanning(arr.size)
        spec = np.abs(np.fft.rfft((arr - arr.mean()) * win)) ** 2
        if spec.sum() > 0:
            top = float(spec[int(0.8 * spec.size):].sum() / spec.sum())
            if top > THRESHOLDS["aliasing_band_fraction"]:
                flags.append(NoiseFlag(
                    "aliasing_band", "warning", top,
                    THRESHOLDS["aliasing_band_fraction"], channel,
                    f"{top:.0%} of the spectral energy is in the top 20% of the "
                    f"band -- anti-alias filtering looks absent",
                ))

        # mains pickup
        expected = [f for f in expected_frequencies_hz if f and np.isfinite(f)]
        for mains in MAINS_FREQUENCIES_HZ:
            if mains >= nyquist:
                continue
            if abs(f_peak - mains) <= THRESHOLDS["mains_tolerance_hz"] and not any(
                abs(mains - e) <= 2.0 for e in expected
            ):
                flags.append(NoiseFlag(
                    "mains_interference", "critical", f_peak, mains, channel,
                    f"dominant peak sits at {f_peak:.1f} Hz, within tolerance of "
                    f"{mains:.0f} Hz mains, and no physical frequency in this "
                    f"record explains it -- treat as electrical pickup",
                ))
                break
    return flags


def check_piv(snapshots: Sequence[Sequence[float]], channel: str = "piv") -> List[NoiseFlag]:
    """Dropout, spurious-vector and dead-pixel checks on a snapshot stack."""
    arr = np.asarray(snapshots, dtype=float)
    flags: List[NoiseFlag] = []
    if arr.ndim != 2 or arr.size < 16:
        return flags

    if not np.all(np.isfinite(arr)):
        bad = float(np.mean(~np.isfinite(arr)))
        flags.append(NoiseFlag(
            "piv_non_finite", "critical", bad, 0.0, channel,
            f"{bad:.1%} of PIV samples are NaN -- correlation failed there",
        ))
        arr = np.nan_to_num(arr)

    zero_frac = float(np.mean(arr == 0.0))
    if zero_frac > THRESHOLDS["piv_dropout"]:
        flags.append(NoiseFlag(
            "piv_dropout", "critical" if zero_frac > 0.10 else "warning",
            zero_frac, THRESHOLDS["piv_dropout"], channel,
            f"{zero_frac:.1%} of interrogation windows returned exactly zero -- "
            f"correlation dropout; POD mode energies are affected",
        ))

    # frozen (dead) columns: no variation across the snapshot stack
    if arr.shape[0] >= 4:
        var = arr.var(axis=0)
        frozen = float(np.mean(var == 0.0))
        if frozen > THRESHOLDS["piv_frozen_pixels"]:
            flags.append(NoiseFlag(
                "piv_frozen_pixels", "warning", frozen,
                THRESHOLDS["piv_frozen_pixels"], channel,
                f"{frozen:.1%} of windows never change across the snapshot "
                f"stack -- dead sensor region or an unmasked static area",
            ))

    # universal median test along the spatial axis
    med = np.median(arr, axis=1, keepdims=True)
    residual = np.abs(arr - med)
    med_res = np.median(residual, axis=1, keepdims=True) + 1e-6
    spurious = float(np.mean(residual / med_res > 8.0))
    if spurious > THRESHOLDS["piv_outliers"]:
        flags.append(NoiseFlag(
            "piv_outliers", "critical" if spurious > 0.15 else "warning",
            spurious, THRESHOLDS["piv_outliers"], channel,
            f"{spurious:.1%} of vectors fail a universal median test "
            f"(Westerweel & Scarano) -- spurious vectors, re-run validation "
            f"before trusting the wake statistics",
        ))
    return flags


# ------------------------------------------------------------------ aggregation


def analyse_raw_experiment(raw: Any) -> NoiseReport:
    """Full report for a :class:`Data.schema.RawExperiment` (or a dict).

    Accepts anything with the same attribute names, so a real-rig ingest can
    hand in its own object without importing the pydantic model.
    """
    def _get(name, default=None):
        if isinstance(raw, dict):
            return raw.get(name, default)
        return getattr(raw, name, default)

    fs = float(_get("sampling_rate_hz", 0.0) or 0.0)
    # Frequencies this experiment is *expected* to contain. Without them the
    # mains detector fires on any experiment whose shedding happens to land near
    # 50 or 60 Hz, which is common at laboratory scale. The wake probe and the
    # lift signal define the physical frequencies; a peak matching neither those
    # nor the natural frequency is what makes mains pickup suspicious.
    expected: List[float] = [float(_get("natural_frequency_hz", 0.0) or 0.0)]
    for probe in ("wake_velocity_probe_signal", "cl_signal"):
        sig = _get(probe)
        if sig is not None and len(sig) > 16 and fs > 0:
            f_peak, _, _ = _dominant_peak(np.asarray(list(sig), float), fs)
            if f_peak > 0:
                expected.append(float(f_peak))

    channels = {
        "structural_displacement": _get("structural_displacement_signal"),
        "wake_velocity_probe": _get("wake_velocity_probe_signal"),
        "cl": _get("cl_signal"),
        "cd": _get("cd_signal"),
        "cm": _get("cm_signal"),
    }

    flags: List[NoiseFlag] = []
    checked: List[str] = []
    for name, sig in channels.items():
        if sig is None or len(sig) == 0:
            continue
        checked.append(name)
        flags.extend(check_signal(sig, name, fs, expected))

    snaps = _get("piv_snapshots")
    if snaps is not None and len(snaps) > 0:
        checked.append("piv")
        flags.extend(check_piv(snaps))

    return _finish(flags, checked)


def _finish(flags: List[NoiseFlag], checked: List[str]) -> NoiseReport:
    """Score and summarise. One critical flag is enough to distrust a record."""
    penalty = 0.0
    for f in flags:
        penalty += {"info": 0.02, "warning": 0.10, "critical": 0.30}[f.severity]
    score = float(max(0.0, 1.0 - penalty))

    n_crit = sum(1 for f in flags if f.severity == "critical")
    n_warn = sum(1 for f in flags if f.severity == "warning")
    if n_crit >= 2 or score < 0.4:
        verdict = "unusable"
    elif n_crit == 1:
        verdict = "suspect"
    elif n_warn:
        verdict = "minor_issues"
    else:
        verdict = "clean"

    if flags:
        worst = sorted(flags, key=lambda f: -SEVERITY_ORDER[f.severity])[0]
        summary = (
            f"{len(flags)} instrumentation flag(s) across {len(checked)} channels; "
            f"most serious: {worst.channel}/{worst.name} ({worst.severity})."
        )
    else:
        summary = f"No instrumentation faults detected across {len(checked)} channels."

    return NoiseReport(
        flags=flags, channels_checked=checked,
        quality_score=score, verdict=verdict, summary=summary,
    )


def analyse_signals(
    signals: Dict[str, Sequence[float]],
    sampling_rate_hz: float,
    piv_snapshots: Optional[Sequence[Sequence[float]]] = None,
    expected_frequencies_hz: Iterable[float] = (),
) -> NoiseReport:
    """Report for loose arrays -- the entry point a real rig ingest would use."""
    flags: List[NoiseFlag] = []
    checked: List[str] = []
    for name, sig in signals.items():
        if sig is None or len(sig) == 0:
            continue
        checked.append(name)
        flags.extend(check_signal(sig, name, sampling_rate_hz, expected_frequencies_hz))
    if piv_snapshots is not None and len(piv_snapshots) > 0:
        checked.append("piv")
        flags.extend(check_piv(piv_snapshots))
    return _finish(flags, checked)
