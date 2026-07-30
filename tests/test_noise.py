"""Instrumentation-fault detector: does it fire when it should, and stay quiet
when it should?

The second half matters more than the first. A detector that flags everything
is worse than none, because the language model is told to caveat its answer
whenever a flag is present.
"""
from __future__ import annotations

import numpy as np
import pytest

from features.noise import (
    THRESHOLDS,
    analyse_signals,
    check_piv,
    check_signal,
)

FS = 1000.0
N = 2048


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def t():
    return np.arange(N) / FS


@pytest.fixture
def clean(rng, t):
    """A well-behaved shedding signal: one tone plus modest broadband noise."""
    return np.sin(2 * np.pi * 23.0 * t) + 0.25 * rng.standard_normal(N)


def _names(flags):
    return {f.name for f in flags}


# ----------------------------------------------------------------- no false alarms


def test_clean_signal_raises_nothing(clean):
    assert _names(check_signal(clean, "cl", FS)) == set()


def test_broadband_shedding_peak_is_not_flagged(rng, t):
    """A real wake signal is broadband; it must not trip the SNR detector."""
    broad = np.convolve(rng.standard_normal(N + 64), np.ones(20) / 20, "same")[:N]
    sig = broad + 0.6 * np.sin(2 * np.pi * 23.0 * t)
    assert "spectral_snr" not in _names(check_signal(sig, "wake", FS))


def test_shedding_near_mains_is_not_flagged_when_physically_expected(rng, t):
    """The give-away for mains pickup is a peak nothing physical explains."""
    sig = np.sin(2 * np.pi * 50.0 * t) + 0.2 * rng.standard_normal(N)
    assert "mains_interference" in _names(check_signal(sig, "disp", FS))
    assert "mains_interference" not in _names(
        check_signal(sig, "disp", FS, expected_frequencies_hz=[50.0])
    )


def test_clean_piv_stack_raises_nothing(rng):
    assert _names(check_piv(rng.standard_normal((64, 144)))) == set()


# ------------------------------------------------------------------- detections


def test_clipping(clean):
    rail = 0.5 * np.abs(clean).max()
    assert "clipping" in _names(check_signal(np.clip(clean, -rail, rail), "cl", FS))


def test_flatline(clean):
    sig = clean.copy()
    sig[500:600] = sig[500]
    assert "flatline" in _names(check_signal(sig, "disp", FS))


def test_spikes(clean):
    sig = clean.copy()
    sig[::150] += 12.0
    assert "spikes" in _names(check_signal(sig, "wake", FS))


def test_drift(clean):
    sig = clean + np.linspace(0.0, 8.0, N)
    assert "drift" in _names(check_signal(sig, "cd", FS))


def test_quantisation(clean):
    assert "quantisation" in _names(check_signal(np.round(clean * 4) / 4, "cl", FS))


def test_pure_noise_has_no_resolvable_frequency(rng):
    """White noise still has a largest spectral bin; the detector must know that
    peak-to-floor of ~ln(m)/ln 2 means *nothing is there*."""
    assert "spectral_snr" in _names(check_signal(rng.standard_normal(N), "cl", FS))


def test_non_finite_samples(clean):
    sig = clean.copy()
    sig[10] = np.nan
    assert "non_finite" in _names(check_signal(sig, "cl", FS))


def test_piv_dropout(rng):
    snaps = rng.standard_normal((64, 144))
    snaps[:, :20] = 0.0
    assert "piv_dropout" in _names(check_piv(snaps))


def test_piv_frozen_pixels(rng):
    snaps = rng.standard_normal((64, 144))
    snaps[:, 5] = 1.0
    snaps[:, 6] = -1.0
    assert "piv_frozen_pixels" in _names(check_piv(snaps))


def test_piv_spurious_vectors(rng):
    snaps = rng.standard_normal((64, 144))
    idx = rng.choice(snaps.size, int(0.10 * snaps.size), replace=False)
    flat = snaps.ravel()
    flat[idx] = 500.0
    assert "piv_outliers" in _names(check_piv(flat.reshape(snaps.shape)))


# ------------------------------------------------------------------- reporting


def test_verdict_escalates_with_severity(clean, rng):
    good = analyse_signals({"cl": clean}, FS)
    assert good.verdict == "clean" and good.quality_score == 1.0

    sig = clean.copy()
    sig[400:500] = sig[400]
    bad = analyse_signals({"cl": sig, "cd": clean + np.linspace(0, 8, N)}, FS)
    assert bad.verdict in {"suspect", "unusable"}
    assert bad.quality_score < good.quality_score


def test_prompt_block_is_empty_when_clean(clean):
    assert analyse_signals({"cl": clean}, FS).to_prompt_block() == ""


def test_prompt_block_lists_worst_first(clean):
    sig = clean.copy()
    sig[400:500] = sig[400]
    block = analyse_signals({"cl": sig, "cd": clean + np.linspace(0, 8, N)}, FS
                            ).to_prompt_block()
    assert block.startswith("[data_quality]")
    lines = [l for l in block.splitlines() if l.startswith("- ")]
    assert lines[0].startswith("- CRITICAL")


def test_report_is_json_serialisable(clean):
    import json

    json.dumps(analyse_signals({"cl": clean}, FS).to_dict())


def test_short_signals_are_skipped_not_crashed():
    assert check_signal([1.0, 2.0], "cl", FS) == []
    assert check_piv([[1.0, 2.0]]) == []
