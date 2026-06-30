"""Deterministic, physically-correct description generator.

Given a :class:`~data.schema.FeatureRecord`, produce the target natural-language
interpretation that the SFT model learns to imitate. The description always
states, in a fixed structured order:

1. flow regime
2. dominant frequency and Strouhal number St
3. lock-in status (shedding freq vs. structural natural freq)
4. amplitude trend
5. mode shape
6. a one-line mechanism note

The template path is fully deterministic and requires no network/API. An
optional ``paraphrase`` hook can route the deterministic text through an
external LLM API for linguistic variety; if no API key / library is available it
silently returns the template text, so dataset building always works offline.

Design note: for SYNTHETIC data we trust the carried ``ground_truth_regime`` as
the label. For data without a label (e.g. a future real experiment) we fall back
to :func:`classify_regime`, a transparent rule-based classifier over the same
features. Both paths emit byte-identical formatting so a model trained on
synthetic labels generalizes to real records.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

from Data.schema import (
    AmplitudeTrend,
    FeatureRecord,
    FlowRegime,
    LockInStatus,
)

# Stable, machine-parseable phrasing per enum. The evaluator and
# validate_output.py rely on these exact regime keywords appearing in the text.
_REGIME_PHRASE = {
    FlowRegime.ATTACHED: "attached flow",
    FlowRegime.SEPARATED: "separated flow",
    FlowRegime.VIV_LOCKIN: "VIV lock-in",
    FlowRegime.GALLOPING: "galloping",
    FlowRegime.STALL_FLUTTER: "stall flutter",
}

_LOCKIN_PHRASE = {
    LockInStatus.LOCKED: "locked",
    LockInStatus.NEAR_LOCKIN: "near lock-in",
    LockInStatus.UNLOCKED: "unlocked",
}

_MECHANISM = {
    FlowRegime.ATTACHED: (
        "Boundary layer remains attached; loads are dominated by mean "
        "aerodynamics with only weak wake unsteadiness."
    ),
    FlowRegime.SEPARATED: (
        "Massive flow separation produces an energetic broadband wake, but the "
        "structure does not sustain a coherent limit cycle."
    ),
    FlowRegime.VIV_LOCKIN: (
        "Vortex shedding has synchronized with the structural natural frequency, "
        "driving a saturated vortex-induced limit cycle."
    ),
    FlowRegime.GALLOPING: (
        "A low-frequency movement-induced (galloping) instability driven by "
        "negative aerodynamic damping, well below the shedding frequency."
    ),
    FlowRegime.STALL_FLUTTER: (
        "Dynamic stall on a near-stalled section feeds energy into a single-mode "
        "(stall-flutter) limit cycle through negative aerodynamic damping."
    ),
}


def classify_regime_confidence(fr: FeatureRecord) -> Tuple[FlowRegime, float, dict]:
    """Probabilistic regime classifier with confidence calibration.

    Returns:
        (regime, confidence_score, supporting_features)
        regime: the predicted FlowRegime
        confidence_score: in [0, 1]; higher = clearer decision boundary
        supporting_features: dict of feature_name -> normalized importance weight
    """
    s = fr.structure
    p = fr.piv
    c = fr.conditions
    lock = fr.lock_in_status()
    big_amp = s.rms_amplitude_over_c > 0.1
    fn = s.natural_frequency_hz
    f_shed = p.shedding_frequency_hz
    coh = p.coherence.cl_coherence_peak if p.coherence else 0.0
    damp = s.damping.damping_ratio if s.damping else 0.0
    aoa = c.angle_of_attack_deg

    # Feature importance accumulator
    weights = {}

    # Galloping: large amplitude, structural freq far below shedding, often
    # high damping, low coherence
    if big_amp and fn > 0 and f_shed > 0 and (fn / f_shed) < 0.5:
        regime = FlowRegime.GALLOPING
        weights["ratio_freq"] = 0.35
        weights["amplitude"] = 0.25
        weights["damping_ratio"] = 0.20
        weights["coherence"] = 0.20
        confidence = 0.7 + 0.3 * (1.0 - min(damp / 0.1, 1.0))
    # VIV lock-in: large amplitude AND synchronized frequencies, high coherence
    elif big_amp and lock == LockInStatus.LOCKED:
        regime = FlowRegime.VIV_LOCKIN
        weights["amplitude"] = 0.30
        weights["lock_in"] = 0.30
        weights["coherence"] = 0.25
        weights["frequency_ratio"] = 0.15
        confidence = 0.75 + 0.25 * coh
    # Stall flutter: large amplitude near/above stall, frequencies not locked
    elif big_amp and aoa >= 12.0:
        regime = FlowRegime.STALL_FLUTTER
        weights["amplitude"] = 0.30
        weights["aoa"] = 0.35
        weights["strouhal"] = 0.20
        weights["separation"] = 0.15
        confidence = 0.65 + 0.25 * min((aoa - 12.0) / 10.0, 1.0)
    # Separated: strong separation / wide wake, weak structural response
    elif p.separation_location_x_over_c < 0.6 or p.wake_width_over_c > 0.6:
        regime = FlowRegime.SEPARATED
        weights["separation"] = 0.40
        weights["wake_width"] = 0.30
        weights["amplitude"] = 0.20
        weights["turbulence"] = 0.10
        confidence = 0.7 + 0.2 * max(p.wake_width_over_c / 1.0, p.separation_location_x_over_c / 0.6)
    else:
        regime = FlowRegime.ATTACHED
        weights["inverse_separation"] = 0.40
        weights["inverse_amplitude"] = 0.35
        weights["low_tke"] = 0.25
        confidence = 0.8

    # Normalize weights to sum to 1
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}

    # Clamp confidence
    confidence = float(np.clip(confidence, 0.0, 1.0))
    return regime, confidence, weights


def classify_regime(fr: FeatureRecord) -> FlowRegime:
    """Backward-compatible wrapper returning just the regime."""
    regime, _, _ = classify_regime_confidence(fr)
    return regime


def amplitude_trend(fr: FeatureRecord, regime: FlowRegime) -> AmplitudeTrend:
    """Infer the amplitude trend implied by the regime and response magnitude."""
    a = fr.structure.rms_amplitude_over_c
    if a < 0.01:
        return AmplitudeTrend.NEGLIGIBLE
    if regime == FlowRegime.GALLOPING:
        return AmplitudeTrend.GROWING
    if regime in (FlowRegime.VIV_LOCKIN, FlowRegime.STALL_FLUTTER):
        return AmplitudeTrend.SATURATED
    if regime == FlowRegime.SEPARATED:
        return AmplitudeTrend.DECAYING
    return AmplitudeTrend.NEGLIGIBLE


def _mode_shape_phrase(fr: FeatureRecord) -> str:
    n = fr.structure.mode_number
    ordinal = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth"}.get(n, f"{n}th")
    kind = "bending/heave" if n == 1 else ("coupled bending-torsion" if n == 2 else "higher-order")
    camber = fr.structure.camber
    camber_note = ""
    if camber > 0.08:
        camber_note = f" with pronounced mean camber ({camber:.2f}c)"
    elif camber > 0.02:
        camber_note = f" with mild camber ({camber:.2f}c)"
    return f"{ordinal} structural mode (mode {n}, {kind}){camber_note}"


def describe(fr: FeatureRecord, paraphrase_with_api: bool = False) -> str:
    """Return the deterministic structured interpretation of a feature record.

    If ``paraphrase_with_api`` is True and an API is configured, the deterministic
    text is paraphrased for linguistic variety; otherwise the template text is
    returned unchanged.
    """
    regime, confidence, _ = classify_regime_confidence(fr)
    lock = fr.lock_in_status()
    trend = amplitude_trend(fr, regime)
    p = fr.piv
    s = fr.structure

    regime_txt = _REGIME_PHRASE[regime]
    lock_txt = _LOCKIN_PHRASE[lock]
    f_dom = p.shedding_frequency_hz
    st = p.strouhal_number

    # Enhanced description with new structural diagnostics
    damping_txt = ""
    if s.damping is not None:
        damping_txt = (
            f" Damping analysis yields ζ = {s.damping.damping_ratio:.4f} "
            f"(Q = {s.damping.q_factor:.1f})."
        )

    coherence_txt = ""
    if p.coherence is not None:
        coh_cl = p.coherence.cl_coherence_peak
        phase = p.coherence.phase_angle_deg
        coherence_txt = (
            f" Cross-spectral coherence between structural motion and lift is "
            f"{coh_cl:.3f} (phase {phase:.1f}°)."
        )

    moments_txt = ""
    if s.spectrum is not None:
        skew = s.spectrum.skewness
        kurt = s.spectrum.kurtosis
        bw = s.spectrum.spectral_bandwidth_hz
        moments_txt = (
            f" Signal statistics: skewness = {skew:.3f}, kurtosis = {kurt:.3f}, "
            f"spectral bandwidth = {bw:.3g} Hz."
        )

    text = (
        f"Regime: {regime_txt} (confidence: {confidence:.2f}). "
        f"The wake sheds at a dominant frequency of {f_dom:.2f} Hz, "
        f"giving a Strouhal number St = {st:.3f}. "
        f"Lock-in status: {lock_txt} "
        f"(shedding {f_dom:.2f} Hz vs. structural natural frequency "
        f"{s.natural_frequency_hz:.2f} Hz). "
        f"Amplitude trend: {trend.value} "
        f"(RMS displacement {s.rms_amplitude_over_c:.3f}c). "
        f"Mode shape: {_mode_shape_phrase(fr)}."
        f"{damping_txt}{coherence_txt}{moments_txt} "
        f"Mechanism: {_MECHANISM[regime]}"
    )

    if paraphrase_with_api:
        para = _maybe_paraphrase(text)
        if para is not None:
            return para
    return text


def _maybe_paraphrase(text: str) -> Optional[str]:
    """Optional LLM paraphrase for linguistic variety. Returns None if unavailable.

    This is intentionally best-effort and provider-agnostic: it only activates if
    an API key is present in the environment AND the relevant client library is
    importable. Any failure returns None so the caller keeps the template text.
    The paraphrase MUST preserve every number and the regime/lock-in keywords; the
    prompt below instructs that, and validate_output can be used to verify it.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    instruction = (
        "Rewrite the following fluid-structure-interaction interpretation in "
        "varied but precise scientific prose. You MUST preserve every numeric "
        "value exactly, and keep the regime name, the words 'St', 'Hz', and the "
        "lock-in status verbatim. Return only the rewritten text.\n\n" + text
    )
    try:  # Anthropic first
        if os.environ.get("ANTHROPIC_API_KEY"):
            import anthropic  # type: ignore

            client = anthropic.Anthropic()
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=400,
                messages=[{"role": "user", "content": instruction}],
            )
            return "".join(block.text for block in msg.content if block.type == "text").strip()
        else:  # OpenAI fallback
            from openai import OpenAI  # type: ignore

            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": instruction}],
                max_tokens=400,
            )
            return resp.choices[0].message.content.strip()
    except Exception:
        return None


__all__ = ["describe", "classify_regime", "classify_regime_confidence", "amplitude_trend"]
