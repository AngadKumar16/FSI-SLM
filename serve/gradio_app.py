"""Gradio UI for the FSI interpreter.

A single-page app that lets a researcher (1) provide **real FSI data** -- either
by uploading a ``RawExperiment`` JSON (run through ``features/extract.py``) or by
typing the measured features into a form -- and then (2) interact with the model
two ways:

* **Structured interpretation** -- the fixed 6-part description the trained LoRA
  adapter produces (regime, frequency/St, lock-in, amplitude trend, mode,
  mechanism), optionally RAG-grounded.
* **Free-text questions** -- open-ended questions about the active record,
  answered by the *base* instruct model with the record + retrieved literature
  passages as grounding (see :meth:`inference.runner.InferenceRunner.answer`).

Both paths share one :class:`~inference.runner.InferenceRunner`. Launch via
``python cli.py ui`` or ``python -m serve.gradio_app``.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

# --------------------------------------------------------------------------- #
# Compatibility shim: gradio 4.37's client chokes on JSON schemas that use a
# bare boolean (e.g. ``additionalProperties: true``, produced by pydantic for
# dict-typed fields) -- ``TypeError: argument of type 'bool' is not iterable``
# inside ``get_type``/``_json_schema_to_python_type``. That crashes the main
# page route (api_info) and aborts ``launch()``. Wrap both helpers to treat a
# boolean schema as "Any". Safe no-op on fixed gradio versions.
# --------------------------------------------------------------------------- #
import gradio_client.utils as _gcu  # noqa: E402

_orig_get_type = _gcu.get_type
_orig_js2pt = _gcu._json_schema_to_python_type


def _safe_get_type(schema):
    if isinstance(schema, bool):
        return "Any"
    return _orig_get_type(schema)


def _safe_js2pt(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    return _orig_js2pt(schema, defs)


_gcu.get_type = _safe_get_type
_gcu._json_schema_to_python_type = _safe_js2pt

from Data.describe import (
    classify_regime,
    describe as _describe_fn,
)
from Data.schema import (
    FlowRegime,
    FeatureRecord,
    RawExperiment,
)
from features.extract import extract_features


# --------------------------------------------------------------------------- #
# Record construction helpers
# --------------------------------------------------------------------------- #
def _parse_pod(text: str) -> List[float]:
    """Parse a comma-separated POD-amplitude string into a list of floats."""
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if not parts:
        return [0.0]
    return [float(p) for p in parts]


def feature_record_from_form(
    experiment_id: str,
    reynolds_number: float,
    angle_of_attack_deg: float,
    freestream_velocity_m_s: float,
    reduced_velocity: float,
    chord_m: float,
    shedding_frequency_hz: float,
    strouhal_number: float,
    wake_width_over_c: float,
    separation_location_x_over_c: float,
    pod_mode_amplitudes: str,
    turbulent_kinetic_energy: float,
    pressure: float,
    strain: float,
    rms_amplitude_over_c: float,
    dominant_frequency_hz: float,
    natural_frequency_hz: float,
    mode_number: float,
    camber: float,
    cl_mean: float,
    cl_rms: float,
    cl_dominant_frequency_hz: float,
    cd_mean: float,
    cd_rms: float,
    cd_dominant_frequency_hz: float,
    cm_mean: float,
    cm_rms: float,
    cm_dominant_frequency_hz: float,
) -> FeatureRecord:
    """Build a FeatureRecord from the manual-entry form fields."""
    return FeatureRecord.model_validate(
        {
            "experiment_id": experiment_id or "manual_entry",
            "conditions": {
                "reynolds_number": reynolds_number,
                "angle_of_attack_deg": angle_of_attack_deg,
                "freestream_velocity_m_s": freestream_velocity_m_s,
                "reduced_velocity": reduced_velocity,
                "chord_m": chord_m,
            },
            "pressure_strain": {
                "pressure": pressure,
                "strain": strain,
            },
            "piv": {
                "shedding_frequency_hz": shedding_frequency_hz,
                "strouhal_number": strouhal_number,
                "wake_width_over_c": wake_width_over_c,
                "separation_location_x_over_c": separation_location_x_over_c,
                "pod_mode_amplitudes": _parse_pod(pod_mode_amplitudes),
                "turbulent_kinetic_energy": turbulent_kinetic_energy,
            },
            "structure": {
                "rms_amplitude_over_c": rms_amplitude_over_c,
                "dominant_frequency_hz": dominant_frequency_hz,
                "natural_frequency_hz": natural_frequency_hz,
                "mode_number": int(mode_number),
                "camber": camber,
            },
            "forces": {
                "cl_mean": cl_mean,
                "cl_rms": cl_rms,
                "cl_dominant_frequency_hz": cl_dominant_frequency_hz,
                "cd_mean": cd_mean,
                "cd_rms": cd_rms,
                "cd_dominant_frequency_hz": cd_dominant_frequency_hz,
                "cm_mean": cm_mean,
                "cm_rms": cm_rms,
                "cm_dominant_frequency_hz": cm_dominant_frequency_hz,
            },
        }
    )


def _record_status(fr: FeatureRecord) -> str:
    """Markdown status line incl. any range warnings (units-bug detector)."""
    warns = fr.range_warnings()
    head = f"**Active record:** `{fr.experiment_id}`"
    if not warns:
        return head + " — all fields in range. ✅"
    bullets = "\n".join(f"- {w}" for w in warns)
    return (
        head
        + f" — ⚠️ {len(warns)} field(s) out of expected range "
        "(check for a units mismatch upstream):\n" + bullets
    )


def _esc(text: str) -> str:
    return html.escape(text, quote=False)


_TXT = "color:#000!important;"  # forced on every leaf tag — parent-inherited
                                 # color gets repainted by the Gradio theme's
                                 # own tag-level rules (h3/p/li/strong), same
                                 # bug as the card-contrast fix above.


def _info_section(heading: str, html_inner: str) -> str:
    return (
        f"<p style='margin:10px 0 4px 0; {_TXT} font-size:11px; "
        "text-transform:uppercase; letter-spacing:0.04em; opacity:0.65;'>"
        f"<strong style='{_TXT}'>{_esc(heading)}</strong></p>"
        f"{html_inner}"
    )


def _info_list(pairs) -> str:
    items = "".join(
        f"<li style='margin-bottom:6px; {_TXT}'>"
        f"<strong style='{_TXT}'>{_esc(name)}</strong> — "
        f"<span style='{_TXT}'>{_esc(desc)}</span></li>"
        for name, desc in pairs
    )
    return f"<ul style='margin:0 0 4px 0; padding-left:18px;'>{items}</ul>"


def _info_render(spec: dict, color: str = "#1a1a1a") -> str:
    """Click-to-open ⓘ badge (native <details>/<summary>, no JS needed).

    A `title`-attribute tooltip was tried first but Gradio's Markdown
    component strips it via its sanitizer even with sanitize_html=False in
    some builds, and native title tooltips are hover-timing dependent across
    browsers anyway. <details> is guaranteed to toggle on click/tap.

    `spec` is a structured dict (see _BOX_INFO), rendered as fully-labeled
    sections — every leaf tag carries an explicit forced black color so nothing
    gets silently repainted by the surrounding theme.
    """
    body = (
        f"<p style='margin:0 0 4px 0; {_TXT} font-size:13px; font-weight:600;'>"
        f"{_esc(spec['what'])}</p>"
    )

    if spec.get("source_fields"):
        code_items = " · ".join(
            f"<code style='background:#eee!important; {_TXT} padding:1px 5px; "
            f"border-radius:4px; font-size:11px;'>{_esc(f)}</code>"
            for f in spec["source_fields"]
        )
        body += _info_section("Source field(s) in FeatureRecord", f"<p style='margin:0 0 4px 0;'>{code_items}</p>")

    if spec.get("formula"):
        body += _info_section(
            "Formula",
            f"<p style='margin:0 0 4px 0;'><code style='background:#eee!important; "
            f"{_TXT} padding:3px 6px; border-radius:4px; font-size:12px;'>"
            f"{_esc(spec['formula'])}</code></p>",
        )

    if spec.get("options"):
        body += _info_section("Possible values", _info_list(spec["options"]))

    if spec.get("signals"):
        body += _info_section("Inputs to this score", _info_list(spec["signals"]))

    if spec.get("ratings"):
        body += _info_section("Overall rating thresholds", _info_list(spec["ratings"]))

    if spec.get("example"):
        body += _info_section(
            "Worked example",
            f"<p style='margin:0 0 4px 0; {_TXT}'>{_esc(spec['example'])}</p>",
        )

    if spec.get("edge_cases"):
        items = "".join(
            f"<li style='margin-bottom:6px; {_TXT}'>{_esc(c)}</li>" for c in spec["edge_cases"]
        )
        body += _info_section(
            "Edge cases / gotchas",
            f"<ul style='margin:0 0 4px 0; padding-left:18px;'>{items}</ul>",
        )

    if spec.get("how"):
        body += _info_section("How it's computed", f"<p style='margin:0 0 4px 0; {_TXT}'>{_esc(spec['how'])}</p>")

    if spec.get("why"):
        body += _info_section("Why it matters", f"<p style='margin:0 0 4px 0; {_TXT}'>{_esc(spec['why'])}</p>")

    if spec.get("see_also"):
        body += _info_section(
            "See also",
            f"<p style='margin:0; {_TXT}'>{_esc(', '.join(spec['see_also']))}</p>",
        )

    return (
        "<details style='display:inline-block; margin-left:6px; "
        "vertical-align:middle; font-size:12px; font-weight:normal;'>"
        f"<summary style='cursor:pointer; list-style:none; display:inline-flex; "
        f"align-items:center; justify-content:center; width:16px; height:16px; "
        f"border:1px solid {color}; border-radius:50%; color:{color}!important; "
        "opacity:0.8; font-family:serif;'>i</summary>"
        f"<div style='margin-top:6px; padding:12px 14px; background:#fff!important; "
        f"{_TXT} border:1px solid #ccc; border-radius:8px; "
        "font-size:12.5px; font-weight:normal; line-height:1.55; max-width:560px; "
        f"white-space:normal; box-shadow:0 2px 8px rgba(0,0,0,0.15);'>{body}</div></details>"
    )


def _info(key: str, color: str = "#1a1a1a") -> str:
    return _info_render(_BOX_INFO[key], color=color)


_BOX_INFO = {
    "regime": {
        "what": (
            "Classified flow-structure interaction regime for this operating "
            "point, produced by a rule-based decision tree (Data/describe.py: "
            "classify_regime_confidence) checked in fixed priority order."
        ),
        "source_fields": [
            "structure.rms_amplitude_over_c", "structure.natural_frequency_hz",
            "piv.shedding_frequency_hz", "piv.coherence.cl_coherence_peak",
            "structure.damping.damping_ratio", "conditions.angle_of_attack_deg",
            "piv.separation_location_x_over_c", "piv.wake_width_over_c",
        ],
        "options": [
            ("Attached flow", "Boundary layer stays attached to the surface. Low unsteady loading, no coherent wake instability — the benign baseline case."),
            ("Separated flow", "Flow separates early and/or the wake is wide (broadband), but the structure doesn't sustain a resonant limit cycle. Noisy, not resonant."),
            ("VIV lock-in", "Vortex-shedding frequency has synchronized (within ~10%) with the structure's natural frequency. Produces a resonant, amplitude-saturated limit cycle — the classic 'lock-in' phenomenon."),
            ("Galloping", "Large, growing-amplitude instability from negative aerodynamic damping. Structural frequency sits well below (<50% of) the shedding frequency — not a resonance effect."),
            ("Stall flutter", "Large single-mode limit cycle driven by dynamic stall at high angle of attack (≥12°). Also from negative aerodynamic damping, but tied to stall rather than frequency ratio."),
        ],
        "example": (
            "Default demo record: RMS amplitude 0.300c (>0.1 → 'big amplitude'), "
            "f_shed=12.00Hz, f_natural=12.00Hz → ratio 1.00 → locked. Both "
            "conditions for VIV lock-in are met, so regime = VIV lock-in with "
            "confidence = 0.75 + 0.25 × coherence."
        ),
        "edge_cases": [
            "Manual-entry records have no coherence/damping data (Optional fields default to 0.0), so those weighting terms silently drop to their floor value — confidence is effectively computed from amplitude + frequency ratio alone.",
            "Rules short-circuit at the first match: a record could technically satisfy both the galloping condition and the stall-flutter condition, but galloping is checked first and wins outright.",
            "ground_truth_regime overrides the classifier entirely when present (synthetic training data) — the rule-based classifier only runs when that field is absent, e.g. for uploaded or manually-entered real data.",
        ],
        "how": "Rules are checked in this order — galloping → VIV lock-in → stall flutter → separated → attached — and the first matching rule wins.",
        "why": "Regime sets the expected fatigue-loading character: lock-in and stall flutter cause sustained resonant stress cycling (fatigue risk); galloping risks unbounded amplitude growth; attached/separated are comparatively benign.",
        "see_also": ["Lock-In Status", "Amplitude Response", "Interpretation Confidence", "Regime score breakdown"],
    },
    "frequency": {
        "what": "Raw frequency measurements that feed the regime and lock-in calls.",
        "source_fields": [
            "piv.shedding_frequency_hz", "piv.strouhal_number",
            "structure.natural_frequency_hz", "conditions.chord_m",
            "conditions.freestream_velocity_m_s",
        ],
        "options": [
            ("Shedding frequency", "Dominant wake tone, extracted via FFT of the PIV/velocity time series."),
            ("Strouhal number (St)", "Dimensionless shedding rate. Classic 2D bluff-body baseline is ≈0.2."),
            ("Structural natural frequency", "The structure's own resonant frequency, shown for direct comparison against the shedding frequency — see the Lock-In Status card."),
        ],
        "formula": "St = f_shed · chord / freestream_velocity",
        "example": (
            "Default demo record: f_shed=12.00Hz, chord=0.100m, U∞=2.00 m/s → "
            "St = 12.00 × 0.100 / 2.00 = 0.600 — 3x the ~0.2 baseline, "
            "consistent with a bluff, separation-prone section rather than a "
            "slender streamlined one."
        ),
        "edge_cases": [
            "strouhal_number is a stored field here, not recomputed live from f_shed/chord/U∞ — if it disagrees with the formula above, that mismatch is itself a data-consistency issue (see check_physical_consistency, reflected in the warnings count on the Interpretation Confidence card).",
            "St is only physically meaningful for periodic vortex shedding — in attached flow there may be no clean shedding tone at all, so a St value is still shown but is harder to interpret.",
        ],
        "why": "A large deviation from St ≈ 0.2 flags an unusual body shape, a regime not dominated by clean shedding, or a possible unit error upstream in the data.",
        "see_also": ["Lock-In Status", "Frequency Lock-In Gauge"],
    },
    "lockin": {
        "what": "Categorizes the ratio f_shed / f_natural into 3 synchronization states.",
        "source_fields": ["piv.shedding_frequency_hz", "structure.natural_frequency_hz"],
        "formula": "ratio = shedding_frequency_hz / natural_frequency_hz",
        "options": [
            ("Locked", "ratio in [0.90, 1.10] — within 10%. Shedding has synchronized with the structure: resonance. This is what defines VIV lock-in."),
            ("Near lock-in", "ratio in [0.80, 0.90) or (1.10, 1.20] — approaching resonance, not fully synced."),
            ("Unlocked", "ratio outside [0.80, 1.20] — shedding and structural frequencies are independent, no synchronization."),
        ],
        "example": (
            "Default demo record: 12.00 / 12.00 = 1.000 → inside [0.90, 1.10] → "
            "locked. This is the maximally resonant case — the ratio sits "
            "exactly at the center of the green band on the gauge below."
        ),
        "edge_cases": [
            "If natural_frequency_hz is 0 or missing, the ratio is undefined; the implementation treats f_n ≤ 0 as ratio = 0.0, which reads as strongly 'unlocked' rather than as an error — check for a missing/zero natural frequency if lock-in looks unexpectedly unlocked.",
            "The 10% / 20% band widths (tol=0.10 default) are fixed thresholds inside FeatureRecord.lock_in_status() — this UI doesn't expose a per-record tolerance control.",
        ],
        "see_also": ["Frequency & Strouhal Number", "Frequency Lock-In Gauge", "Flow Regime"],
    },
    "amplitude": {
        "what": "Oscillation-amplitude trend, derived from RMS displacement / chord and the classified regime.",
        "source_fields": ["structure.rms_amplitude_over_c"],
        "options": [
            ("Negligible", "RMS amplitude/chord < 0.01 — no meaningful motion, response is noise-level."),
            ("Growing", "Assigned for galloping — amplitude increases without an obvious ceiling until structural nonlinearity intervenes."),
            ("Saturated", "Assigned for VIV lock-in and stall flutter — amplitude has plateaued into a stable, bounded limit cycle."),
            ("Decaying", "Assigned for separated flow — any structural response present is dying out, not sustained."),
        ],
        "example": (
            "Default demo record: RMS amplitude/chord = 0.3000c, regime = VIV "
            "lock-in → trend = saturated (VIV lock-in and stall flutter both "
            "map to 'saturated' regardless of the exact amplitude value, as "
            "long as it's ≥ 0.01c)."
        ),
        "edge_cases": [
            "The trend label is regime-derived, not magnitude-derived beyond the negligible cutoff — two VIV lock-in records with very different RMS amplitudes (say 0.05c and 0.5c) will both read 'saturated'. Read the raw RMS number alongside the label to judge actual severity.",
        ],
        "see_also": ["Flow Regime", "Lock-In Status"],
    },
    "mode_shape": {
        "what": "Which structural vibration mode dominates the response (mode_number field), and its physical deflection shape.",
        "source_fields": ["structure.mode_number"],
        "options": [
            ("Mode 1", "First bending/heave — the whole structure moves up-down together, no internal node."),
            ("Mode 2+", "Higher-order shapes (e.g. torsion, S-shaped bending with internal nodes), depending on the specific structure's mode set."),
        ],
        "example": "Default demo record: mode_number = 1 → 'first structural mode (mode 1, bending/heave)'.",
        "edge_cases": [
            "mode_number is entered as an integer (precision=0) on the manual form — double check it wasn't left as a stray decimal before trusting the mode-shape phrase.",
            "The phrase mapping for modes beyond 1 is a generic label from Data/describe.py (_mode_shape_phrase), not a structure-specific finite-element result — treat it as descriptive, not authoritative.",
        ],
        "why": "Which mode is excited changes where peak stress occurs — mode 1 concentrates stress near the root/base, higher modes create internal stress concentrations at node/antinode boundaries.",
        "see_also": ["POD Mode Energy Spectrum (fluid modes, not structural — don't conflate the two)"],
    },
    "gauge": {
        "what": "Visual read of the Lock-In Status ratio (f_shed / f_natural) on a 0.5-1.5 track. The black marker plots this record's exact ratio.",
        "source_fields": ["piv.shedding_frequency_hz", "structure.natural_frequency_hz"],
        "options": [
            ("Green band (0.90-1.10)", "Locked / resonant."),
            ("Amber bands (0.80-0.90 and 1.10-1.20)", "Near lock-in."),
            ("Grey (outside amber)", "Unlocked."),
        ],
        "example": "Default demo record: ratio 1.000 → marker sits exactly at gauge center, inside the green band.",
        "edge_cases": [
            "The track display is clamped to [0.5, 1.5] — a ratio outside that range (e.g. 2.0) still shows the marker pinned at the edge, so an extreme value can look visually identical to a moderately-unlocked one. Read the printed ratio number above the bar, not just marker position, for the true value.",
        ],
        "why": "Dead center of the green band (ratio = 1.0) is perfect resonance — the worst case for sustained fatigue cycling.",
        "see_also": ["Lock-In Status"],
    },
    "pod": {
        "what": (
            "Proper Orthogonal Decomposition (POD) mode-energy bars — a spatial "
            "decomposition of the wake velocity field, ranked by kinetic-energy "
            "share. This is a FLUID/wake mode breakdown, separate from the "
            "structural mode_number in the Mode Shape card."
        ),
        "source_fields": ["piv.pod_mode_amplitudes"],
        "options": [
            ("Mode 1 dominant (>60% of energy)", "One coherent wake structure — clean, periodic vortex shedding."),
            ("Energy spread across several modes", "A more complex, turbulent, or noisy wake that resists a single-frequency description."),
        ],
        "example": (
            "Default demo record: amplitudes [0.60, 0.20, 0.10, 0.05] → mode 1 "
            "share = 0.60 / (0.60+0.20+0.10+0.05) = 63% → labeled 'single "
            "dominant coherent structure'."
        ),
        "edge_cases": [
            "pod_mode_amplitudes is free-form comma-separated text on the manual-entry form — there's no enforced descending-energy ordering, so mode 1 in the chart is simply 'the first number you typed,' not necessarily the largest.",
        ],
        "see_also": ["Mode Shape (structural, not fluid — don't conflate the two)"],
    },
    "confidence": {
        "what": "Composite trust score for the regime call above, computed here from 4 signals — it is not part of the raw model output.",
        "source_fields": [
            "derived: classify_regime_confidence(), regime_scores(), "
            "range_warnings(), check_physical_consistency(), "
            "structure.spectrum, piv.coherence",
        ],
        "signals": [
            ("Classifier confidence [0,1]", "How strongly this record's features satisfy the winning regime's decision rule."),
            ("Margin over runner-up", "Gap between the winning regime's score and the 2nd-most-likely regime in the Regime Score Breakdown card. Margin < 0.15 means the record sits near a decision boundary and could flip with slightly different inputs."),
            ("Range/consistency warnings", "Count of measured fields outside physically expected bounds, or mutually inconsistent (e.g. a units mismatch). Each one lowers trust."),
            ("Noise/signal-quality flags", "Only available when the record has real spectral stats (skewness, kurtosis, spectral kurtosis, coherence) from an uploaded RawExperiment/PIV signal — not from manual form entry. Flags asymmetric/impulsive signals, flat/broadband spectra (weak tonal shedding), or weak structure-force coherence (undermines the claimed mechanism)."),
        ],
        "ratings": [
            ("HIGH", "confidence > 0.75 AND margin > 0.3 AND 0 warnings AND 0 noise flags"),
            ("MEDIUM", "everything that doesn't hit HIGH or LOW"),
            ("LOW", "confidence < 0.5 OR ≥2 warnings OR ≥2 noise flags"),
        ],
        "example": (
            "Default demo record: confidence=0.75, margin=0.69 (>0.3), 0 "
            "warnings, spectrum/coherence absent (manual entry) → rated "
            "MEDIUM rather than HIGH — see edge cases for why."
        ),
        "edge_cases": [
            "A manual-entry record can never score HIGH: absent spectrum/coherence isn't counted as a noise flag, but it also can't confirm a clean signal, so the rating logic keeps it at MEDIUM. Treat a MEDIUM rating on manual entries as 'no evidence found of noise' rather than 'confirmed clean.'",
            "Margin comes from the separate 5-way Regime Score Breakdown card, not directly from the confidence formula — the two numbers are related but computed by different functions and can diverge slightly at extreme values.",
        ],
        "see_also": ["Regime score breakdown", "Flow Regime"],
    },
    "key_insight": {
        "what": (
            "One-line physical mechanism for why the classified regime occurs. "
            "Fixed text per regime from a lookup table (Data/describe.py: "
            "_MECHANISM) — it explains the underlying physics of the regime in "
            "general, not this specific record's numbers."
        ),
        "source_fields": ["Data/describe.py: _MECHANISM[regime] — static lookup, no numeric inputs"],
        "example": (
            "For VIV lock-in: 'Vortex shedding has synchronized with the "
            "structural natural frequency, driving a saturated vortex-induced "
            "limit cycle.' — identical text for every VIV lock-in record "
            "regardless of its specific Re, St, or amplitude."
        ),
        "edge_cases": [
            "Because this text never changes within a regime, it can't distinguish two records that share a regime — use the numeric cards (frequency, amplitude, confidence) for that comparison instead.",
        ],
        "see_also": ["Flow Regime"],
    },
    "score_breakdown": {
        "what": "Normalized likelihood for all 5 possible regimes on this exact record — scores sum to 1.0.",
        "source_fields": [
            "structure.rms_amplitude_over_c", "structure.natural_frequency_hz",
            "piv.shedding_frequency_hz", "conditions.angle_of_attack_deg",
            "piv.separation_location_x_over_c", "piv.wake_width_over_c",
        ],
        "how": (
            "The winning regime keeps its calibrated confidence score from the "
            "classifier. The remaining probability mass (1 − confidence) is "
            "split across the other 4 regimes proportional to each one's own "
            "raw rule-based evidence score (Data/describe.py: regime_scores)."
        ),
        "example": (
            "Default demo record: VIV lock-in 0.75; attached flow, separated "
            "flow, galloping, and stall flutter each ≈0.06 (the remaining 0.25 "
            "splits roughly evenly since none of the other 4 regimes have "
            "strong raw evidence for this record)."
        ),
        "edge_cases": [
            "The per-regime 'evidence' formulas are independent heuristics, not a trained probabilistic model — treat the 4 non-winning scores as a rough relative ranking, not calibrated probabilities in the statistical sense.",
        ],
        "why": "One tall bar with the rest near-zero means an unambiguous call. Several comparable bars means the record sits near a multi-way boundary — treat the top regime as a best guess, not a certainty.",
        "see_also": ["Interpretation Confidence", "Flow Regime"],
    },
    "summary_lockin_amp": {
        "what": "Compact restatement of the Lock-In Status and Amplitude Response cards, for at-a-glance reading.",
        "why": "Open the Lock-In Status and Amplitude Response cards further down this page (or their own ⓘ badges) for the complete definitions, formulas, and worked examples for every possible value.",
        "see_also": ["Lock-In Status", "Amplitude Response"],
    },
}


def _passages_markdown(retrieved: List) -> str:
    """Render retrieved RAG passages as a citation block."""
    if not retrieved:
        return (
            "_No reference passages retrieved (RAG off, no index, or below "
            "score threshold)._"
        )
    lines = ["### Reference passages"]
    for i, r in enumerate(retrieved):
        lines.append(
            f"**[{i + 1}] {r.source}** (p. {r.page}, score {r.score:.3f})\n\n"
            f"> {r.passage_text.strip()}"
        )
    return "\n\n".join(lines)


# --------------------------------------------------------------------------- #
# App factory
# --------------------------------------------------------------------------- #
def create_ui(config: Optional[dict] = None) -> "gr.Blocks":
    """Build the Gradio Blocks app wired to a shared InferenceRunner."""
    _config = config or {}

    def _adapter_path() -> str:
        return _config.get("inference", {}).get("adapter_path", "artifacts/adapters")

    def _has_trained_model() -> bool:
        adapter = _adapter_path()
        return bool(adapter and Path(adapter, "adapters.safetensors").exists())

    # Lazily-built shared runner (heavy import / model load deferred to first use).
    _runner_holder: Dict[str, Any] = {"runner": None}

    def _get_runner():
        if _runner_holder["runner"] is None:
            from inference.runner import InferenceConfig, InferenceRunner

            cfg = _config.get("inference", {})
            rag_cfg = _config.get("rag", {})
            inf_cfg = InferenceConfig(
                model_name=cfg.get(
                    "model_name", "./artifacts/models/Phi-4-mini-instruct"
                ),
                adapter_path=cfg.get("adapter_path", "artifacts/adapters"),
                rag_index_dir=cfg.get(
                    "rag_index_dir", rag_cfg.get("index_dir", "artifacts/rag_index")
                ),
                max_tokens=cfg.get("max_tokens", 350),
                top_k=cfg.get("rag_top_k", rag_cfg.get("top_k", 3)),
                score_threshold=cfg.get(
                    "rag_score_threshold", rag_cfg.get("score_threshold", 0.3)
                ),
            )
            _runner_holder["runner"] = InferenceRunner(inf_cfg)
        return _runner_holder["runner"]

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #
    def build_from_form(*form_values):
        try:
            fr = feature_record_from_form(*form_values)
        except Exception as e:  # noqa: BLE001 - surface any validation error to UI
            return None, None, f"❌ Could not build record: {e}"
        return fr.model_dump(mode="json"), fr.model_dump(mode="json"), _record_status(fr)

    def build_from_upload(file_obj):
        if file_obj is None:
            return None, None, "❌ No file uploaded."
        # Gradio may hand back a path string (type="filepath") or a file object.
        file_path = file_obj if isinstance(file_obj, str) else file_obj.name
        try:
            with open(file_path) as f:
                data = json.load(f)
            raw = RawExperiment.model_validate(data)
            fr = extract_features(raw)
        except Exception as e:  # noqa: BLE001
            return None, None, (
                f"❌ Could not parse RawExperiment / extract features: {e}"
            )
        return fr.model_dump(mode="json"), fr.model_dump(mode="json"), _record_status(fr)

    def _build_metrics_table(fr: FeatureRecord) -> str:
        """Build a detailed markdown table with all key metrics organized by category."""
        p = fr.piv
        s = fr.structure
        c = fr.conditions
        f = fr.forces
        regime = fr.ground_truth_regime or classify_regime(fr)
        lock = fr.lock_in_status()
        from Data.describe import amplitude_trend, _LOCKIN_PHRASE, _mode_shape_phrase

        trend = amplitude_trend(fr, regime)

        lines = [
            "| Category | Metric | Value | Notes |",
            "|---|---|---|---|",
        ]

        freq_ratio = (s.natural_frequency_hz / p.shedding_frequency_hz) if p.shedding_frequency_hz else 0.0
        cl_ratio = (f.cl_rms / f.cl_mean) if f.cl_mean else float("inf")

        # Flow conditions
        re_note = (
            "laminar/transitional" if c.reynolds_number < 3e5
            else "subcritical turbulent" if c.reynolds_number < 3.5e5
            else "critical/supercritical turbulent"
        )
        lines.append(f"| **Conditions** | Reynolds number | {c.reynolds_number:.2e} | {re_note} |")
        aoa_note = (
            "near/past static stall (≥12°)" if c.angle_of_attack_deg >= 12.0
            else "attached-flow range" if c.angle_of_attack_deg < 8.0
            else "approaching stall"
        )
        lines.append(f"| | Angle of attack | {c.angle_of_attack_deg:.1f}° | {aoa_note} |")
        lines.append(f"| | Freestream velocity | {c.freestream_velocity_m_s:.2f} m/s | |")
        lines.append(f"| | Reduced velocity (U*) | {c.reduced_velocity:.2f} | {'lock-in-prone band (4-8)' if 4.0 <= c.reduced_velocity <= 8.0 else 'outside typical lock-in band'} |")
        lines.append(f"| | Chord | {c.chord_m:.3f} m | reference length for all `/c` quantities |")

        # PIV
        lines.append(f"| **PIV** | Shedding frequency | {p.shedding_frequency_hz:.2f} Hz | Dominant wake tone |")
        lines.append(f"| | Strouhal number | {p.strouhal_number:.3f} | St = fD/U; ~0.2 bluff-body baseline, this deviates {'a lot — check regime' if abs(p.strouhal_number - 0.2) > 0.15 else 'modestly'} |")
        lines.append(f"| | Wake width | {p.wake_width_over_c:.3f}c | {'>0.6 → separated/bluff wake' if p.wake_width_over_c > 0.6 else 'narrow, attached-like wake'} |")
        lines.append(f"| | Separation location | {p.separation_location_x_over_c:.3f}c | {'<0.6 → early separation' if p.separation_location_x_over_c < 0.6 else 'separation delayed toward TE'} |")
        lines.append(f"| | Turbulent KE | {p.turbulent_kinetic_energy:.4f} | {'elevated — noisy wake' if p.turbulent_kinetic_energy > 0.1 else 'low, coherent wake'} |")

        # Structure
        lines.append(f"| **Structure** | RMS amplitude | {s.rms_amplitude_over_c:.4f}c | trend: {trend.value} |")
        lines.append(f"| | Dominant frequency | {s.dominant_frequency_hz:.2f} Hz | response frequency of structure |")
        lines.append(f"| | Natural frequency | {s.natural_frequency_hz:.2f} Hz | f_n/f_shed = {freq_ratio:.2f} |")
        lines.append(f"| | Mode number | {int(s.mode_number)} | {_mode_shape_phrase(fr)} |")
        lines.append(f"| | Camber | {s.camber:.3f} | {'notable camber, expect mean-lift offset' if abs(s.camber) > 0.03 else 'near-symmetric section'} |")

        # Forces
        lines.append(f"| **Forces** | CL mean | {f.cl_mean:.3f} | steady lift coefficient |")
        lines.append(f"| | CL rms | {f.cl_rms:.3f} | unsteady/mean ratio = {cl_ratio:.2f} |")
        lines.append(f"| | CL dom. freq | {f.cl_dominant_frequency_hz:.2f} Hz | {'matches shedding' if abs(f.cl_dominant_frequency_hz - p.shedding_frequency_hz) < 0.5 else 'offset from shedding freq'} |")
        lines.append(f"| | CD mean | {f.cd_mean:.3f} | steady drag coefficient |")
        lines.append(f"| | CD rms | {f.cd_rms:.3f} | drag unsteadiness |")
        lines.append(f"| | CD dom. freq | {f.cd_dominant_frequency_hz:.2f} Hz | {'2x shedding (typical drag doubling)' if abs(f.cd_dominant_frequency_hz - 2 * p.shedding_frequency_hz) < 1.0 else ''} |")
        lines.append(f"| | CM mean | {f.cm_mean:.4f} | pitching moment bias |")
        lines.append(f"| | CM rms | {f.cm_rms:.4f} | pitching moment unsteadiness |")
        lines.append(f"| | CM dom. freq | {f.cm_dominant_frequency_hz:.2f} Hz | |")

        return "\n".join(lines)

    def _build_interpretation_cards(fr: FeatureRecord) -> str:
        """Build interpretation as visual cards/sections."""
        regime = fr.ground_truth_regime or classify_regime(fr)
        lock = fr.lock_in_status()
        from Data.describe import amplitude_trend, _LOCKIN_PHRASE, _MECHANISM, _REGIME_PHRASE, _mode_shape_phrase

        trend = amplitude_trend(fr, regime)
        p = fr.piv
        s = fr.structure

        # Regime badge color
        regime_colors = {
            "attached flow": "🟢",
            "separated flow": "🟡",
            "VIV lock-in": "🔴",
            "galloping": "🟠",
            "stall flutter": "🟣",
        }
        regime_emoji = regime_colors.get(_REGIME_PHRASE[regime], "⚪")

        md = "<div style='display:grid; grid-template-columns:repeat(auto-fit, minmax(320px, 1fr)); gap: 12px; align-items:stretch;'>\n\n"

        # Card 1: Regime
        md += f"<div style='background:#f7f8fa!important; color:#1a1a1a!important; padding: 12px; border-radius: 8px; border-left: 4px solid #2196F3; height:100%; box-sizing:border-box;'>\n"
        md += f"<h3 style='color:#1a1a1a!important; margin:0 0 4px 0;'>{regime_emoji} Flow Regime{_info('regime')}</h3>\n"
        md += f"<p style='font-size: 18px; margin: 4px 0; color:#1a1a1a!important;'><strong style='color:#1a1a1a!important;'>{_REGIME_PHRASE[regime].title()}</strong></p>\n"
        md += f"<p style='margin: 4px 0; color:#555!important;'>{_MECHANISM[regime]}</p>\n"
        md += "</div>\n\n"

        # Card 2: Frequency
        md += f"<div style='background:#f7f8fa!important; color:#1a1a1a!important; padding: 12px; border-radius: 8px; border-left: 4px solid #FFC107; height:100%; box-sizing:border-box;'>\n"
        md += f"<h3 style='color:#1a1a1a!important; margin:0 0 4px 0;'>📊 Frequency & Strouhal Number{_info('frequency')}</h3>\n"
        md += f"<table style='width: 100%; border-collapse: collapse;'>\n"
        md += f"<tr><td style='padding: 6px; color:#1a1a1a!important;'><strong style='color:#1a1a1a!important;'>Shedding frequency</strong></td><td style='padding: 6px; text-align: right; color:#1a1a1a!important;'><code>{p.shedding_frequency_hz:.2f} Hz</code></td></tr>\n"
        md += f"<tr><td style='padding: 6px; color:#1a1a1a!important;'><strong style='color:#1a1a1a!important;'>Strouhal number (St)</strong></td><td style='padding: 6px; text-align: right; color:#1a1a1a!important;'><code>{p.strouhal_number:.3f}</code></td></tr>\n"
        md += f"<tr><td style='padding: 6px; color:#1a1a1a!important;'><strong style='color:#1a1a1a!important;'>Structural natural freq</strong></td><td style='padding: 6px; text-align: right; color:#1a1a1a!important;'><code>{s.natural_frequency_hz:.2f} Hz</code></td></tr>\n"
        md += f"</table>\n"
        md += "</div>\n\n"

        # Card 3: Lock-in
        lock_color = "#4CAF50" if _LOCKIN_PHRASE[lock] == "locked" else ("#FF9800" if _LOCKIN_PHRASE[lock] == "near lock-in" else "#F44336")
        md += f"<div style='background:#f7f8fa!important; color:#1a1a1a!important; padding: 12px; border-radius: 8px; border-left: 4px solid {lock_color}; height:100%; box-sizing:border-box;'>\n"
        md += f"<h3 style='color:#1a1a1a!important; margin:0 0 4px 0;'>🔒 Lock-In Status{_info('lockin')}</h3>\n"
        md += f"<p style='font-size: 16px; margin: 4px 0; color:#1a1a1a!important;'><strong style='color:#1a1a1a!important;'>{_LOCKIN_PHRASE[lock].title()}</strong></p>\n"
        md += f"<p style='margin: 4px 0; color:#555!important;'>Shedding at {p.shedding_frequency_hz:.2f} Hz vs structural natural frequency {s.natural_frequency_hz:.2f} Hz</p>\n"
        md += "</div>\n\n"

        # Card 4: Amplitude
        md += f"<div style='background:#f7f8fa!important; color:#1a1a1a!important; padding: 12px; border-radius: 8px; border-left: 4px solid #4CAF50; height:100%; box-sizing:border-box;'>\n"
        md += f"<h3 style='color:#1a1a1a!important; margin:0 0 4px 0;'>📈 Amplitude Response{_info('amplitude')}</h3>\n"
        md += f"<p style='margin: 4px 0; color:#1a1a1a!important;'><strong style='color:#1a1a1a!important;'>Trend:</strong> <code>{trend.value}</code></p>\n"
        md += f"<p style='margin: 4px 0; color:#1a1a1a!important;'><strong style='color:#1a1a1a!important;'>RMS displacement:</strong> <code>{s.rms_amplitude_over_c:.4f}c</code></p>\n"
        md += "</div>\n\n"

        # Card 5: Mode shape
        md += f"<div style='background:#f7f8fa!important; color:#1a1a1a!important; padding: 12px; border-radius: 8px; border-left: 4px solid #9C27B0; height:100%; box-sizing:border-box;'>\n"
        md += f"<h3 style='color:#1a1a1a!important; margin:0 0 4px 0;'>🎭 Mode Shape{_info('mode_shape')}</h3>\n"
        md += f"<p style='margin: 4px 0; color:#1a1a1a!important;'>{_mode_shape_phrase(fr)}</p>\n"
        md += "</div>\n\n"

        md += "</div>"
        return md

    def _lockin_gauge(fr: FeatureRecord) -> str:
        """Horizontal gauge of the shedding/natural frequency ratio.

        The green band (0.9-1.1) is the locked region, amber (0.8-1.2) is
        near lock-in, matching FeatureRecord.lock_in_status (tol=0.10).
        """
        f_shed = fr.piv.shedding_frequency_hz
        f_n = fr.structure.natural_frequency_hz
        ratio = (f_shed / f_n) if f_n > 0 else 0.0
        # Track spans ratio 0.5 .. 1.5; clamp marker inside.
        lo, hi = 0.5, 1.5
        pos = max(0.0, min(1.0, (ratio - lo) / (hi - lo))) * 100
        # Band edges as track percentages.
        def _pct(r: float) -> float:
            return max(0.0, min(1.0, (r - lo) / (hi - lo))) * 100
        amber_l, amber_r = _pct(0.8), _pct(1.2)
        green_l, green_r = _pct(0.9), _pct(1.1)
        md = "<div style='background:#f7f8fa!important; color:#1a1a1a!important; padding: 12px; border-radius: 8px; border-left: 4px solid #E91E63; height:100%; box-sizing:border-box;'>\n"
        md += f"<h3 style='color:#1a1a1a!important; margin:0 0 4px 0;'>🎯 Frequency Lock-In Gauge{_info('gauge')}</h3>\n"
        md += (
            f"<p style='margin: 4px 0; color:#555!important;'>Ratio "
            f"f<sub>shed</sub>/f<sub>n</sub> = <code>{ratio:.3f}</code> "
            f"({f_shed:.2f} Hz / {f_n:.2f} Hz)</p>\n"
        )
        md += "<div style='position: relative; height: 26px; background: #eee; border-radius: 13px; margin: 10px 0;'>\n"
        md += f"<div style='position:absolute; left:{amber_l:.1f}%; width:{amber_r-amber_l:.1f}%; top:0; bottom:0; background:#FFE082; border-radius:13px;'></div>\n"
        md += f"<div style='position:absolute; left:{green_l:.1f}%; width:{green_r-green_l:.1f}%; top:0; bottom:0; background:#A5D6A7;'></div>\n"
        md += f"<div style='position:absolute; left:{pos:.1f}%; top:-4px; width:3px; height:34px; background:#000; transform:translateX(-1px);'></div>\n"
        md += "</div>\n"
        md += "<div style='display:flex; justify-content:space-between; font-size:11px; color:#777!important;'><span style='color:#777!important;'>0.5</span><span style='color:#777!important;'>1.0 (locked)</span><span style='color:#777!important;'>1.5</span></div>\n"
        md += "</div>\n\n"
        return md

    def _pod_spectrum(fr: FeatureRecord) -> str:
        """Horizontal energy bars for the POD mode amplitudes."""
        amps = list(fr.piv.pod_mode_amplitudes or [])
        if not amps:
            return ""
        mx = max(abs(a) for a in amps) or 1.0
        mode1_frac = abs(amps[0]) / sum(abs(a) for a in amps) if sum(abs(a) for a in amps) else 0.0
        md = "<div style='background:#f7f8fa!important; color:#1a1a1a!important; padding: 12px; border-radius: 8px; border-left: 4px solid #1976D2; height:100%; box-sizing:border-box;'>\n"
        md += f"<h3 style='color:#1a1a1a!important; margin:0 0 4px 0;'>🌊 POD Mode Energy Spectrum{_info('pod')}</h3>\n"
        md += f"<p style='margin:2px 0 8px 0; font-size:12px; color:#555!important;'>Mode 1 carries {mode1_frac*100:.0f}% of resolved energy — {'single dominant coherent structure' if mode1_frac > 0.6 else 'energy spread across several modes (more complex/noisier wake)'}.</p>\n"
        for i, a in enumerate(amps):
            w = abs(a) / mx * 100
            md += "<div style='display:flex; align-items:center; gap:8px; margin:4px 0;'>\n"
            md += f"<span style='width:56px; font-size:12px; color:#1a1a1a!important;'>Mode {i + 1}</span>\n"
            md += "<div style='flex:1; background:#eee; border-radius:6px; height:16px;'>\n"
            md += f"<div style='width:{w:.1f}%; height:16px; background:#1976D2; border-radius:6px;'></div>\n"
            md += "</div>\n"
            md += f"<span style='width:52px; font-size:12px; text-align:right; color:#1a1a1a!important;'><code>{a:.3f}</code></span>\n"
            md += "</div>\n"
        md += "</div>\n\n"
        return md

    def _build_confidence_card(fr: FeatureRecord) -> str:
        """Consolidated confidence / noise-corruption indicator.

        Combines: classifier confidence, margin over the runner-up regime
        (ambiguity), count of range/consistency warnings, and — when the
        record carries spectral stats (real PIV extraction, not manual
        entry) — skewness/kurtosis/coherence as a noise proxy.
        """
        from Data.describe import classify_regime_confidence, regime_scores, _REGIME_PHRASE
        from features.extract import check_physical_consistency

        regime, confidence, _weights = classify_regime_confidence(fr)
        scores = regime_scores(fr)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        runner_up_margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else 1.0

        warns = fr.range_warnings() + check_physical_consistency(fr)

        noise_bits = []
        s = fr.structure
        p = fr.piv
        if s.spectrum is not None:
            sp = s.spectrum
            if abs(sp.skewness) > 1.0:
                noise_bits.append(f"skewness {sp.skewness:.2f} — asymmetric, non-Gaussian response")
            if sp.kurtosis > 5.0:
                noise_bits.append(f"kurtosis {sp.kurtosis:.2f} — impulsive/intermittent signal")
            if sp.spectral_kurtosis < 1.0:
                noise_bits.append(f"spectral kurtosis {sp.spectral_kurtosis:.2f} — flat/broadband spectrum, weak tonal content")
        if p.coherence is not None:
            coh = p.coherence.cl_coherence_peak
            if coh < 0.5:
                noise_bits.append(f"CL coherence peak {coh:.2f} — weak structure/force coupling, low confidence in mechanism claim")

        overall = "high" if (confidence > 0.75 and runner_up_margin > 0.3 and not warns and not noise_bits) else \
                  "low" if (confidence < 0.5 or len(warns) >= 2 or len(noise_bits) >= 2) else "medium"
        overall_color = {"high": "#4CAF50", "medium": "#FF9800", "low": "#F44336"}[overall]
        overall_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}[overall]

        md = f"<div style='background:#f7f8fa!important; color:#1a1a1a!important; padding: 12px; border-radius: 8px; border-left: 4px solid {overall_color}; height:100%; box-sizing:border-box;'>\n"
        md += f"<h3 style='color:#1a1a1a!important; margin:0 0 6px 0;'>{overall_emoji} Interpretation Confidence: {overall.upper()}{_info('confidence')}</h3>\n"
        md += f"<p style='margin: 2px 0; color:#1a1a1a!important;'>Classifier confidence for <strong style='color:#1a1a1a!important;'>{_REGIME_PHRASE[regime]}</strong>: <code>{confidence:.2f}</code></p>\n"
        md += f"<p style='margin: 2px 0; color:#1a1a1a!important;'>Margin over runner-up regime: <code>{runner_up_margin:.2f}</code>{' — close call, borderline classification' if runner_up_margin < 0.15 else ''}</p>\n"
        md += f"<p style='margin: 2px 0; color:#1a1a1a!important;'>Range/consistency warnings: <code>{len(warns)}</code></p>\n"
        if noise_bits:
            md += "<p style='margin: 6px 0 2px 0; color:#1a1a1a!important;'><strong style='color:#1a1a1a!important;'>Noise / signal-quality flags:</strong></p>\n"
            md += "<ul style='margin:2px 0 0 18px; padding:0; color:#1a1a1a!important;'>\n"
            for b in noise_bits:
                md += f"<li style='color:#1a1a1a!important;'>{b}</li>\n"
            md += "</ul>\n"
        elif s.spectrum is None and p.coherence is None:
            md += "<p style='margin: 6px 0 0 0; font-size:12px; color:#555!important;'>No spectral/coherence stats on this record (manual entry) — noise indicators unavailable; confidence above is based on regime-boundary distance only.</p>\n"
        else:
            md += "<p style='margin: 6px 0 0 0; font-size:12px; color:#555!important;'>No noise flags — signal statistics within normal bounds.</p>\n"
        md += "</div>\n\n"
        return md

    def _build_summary_card(fr: FeatureRecord) -> str:
        """Build a one-glance-at-a-time summary that an engineer actually wants."""
        regime = fr.ground_truth_regime or classify_regime(fr)
        lock = fr.lock_in_status()
        from Data.describe import amplitude_trend, _LOCKIN_PHRASE, _MECHANISM, _REGIME_PHRASE
        trend = amplitude_trend(fr, regime)
        p = fr.piv
        s = fr.structure

        regime_emoji = {
            "attached flow": "🟢", "separated flow": "🟡",
            "VIV lock-in": "🔴", "galloping": "🟠", "stall flutter": "🟣",
        }.get(_REGIME_PHRASE[regime], "⚪")

        md = "<div style='display:grid; grid-template-columns:repeat(auto-fit, minmax(340px, 1fr)); gap: 10px; align-items:stretch;'>\n\n"
        md += f"<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color:#fff!important; padding: 14px; border-radius: 8px;'>\n"
        md += f"<p style='margin: 0; font-size: 14px; opacity: 0.9; color:#fff!important;'>Flow regime{_info('regime', color='#fff')}</p>\n"
        md += f"<p style='margin: 4px 0 0 0; font-size: 20px; font-weight: bold; color:#fff!important;'>{regime_emoji} {_REGIME_PHRASE[regime].title()}</p>\n"
        md += "</div>\n\n"

        md += "<div style='background:#f8f9fa!important; color:#1a1a1a!important; padding: 12px; border-radius: 8px;'>\n"
        md += f"<p style='margin:0 0 4px 0; font-size:11px; color:#888!important;'>Lock-in / Amplitude{_info('summary_lockin_amp')}</p>\n"
        md += "<table style='width: 100%;'>\n"
        md += f"<tr><td style='padding: 6px; color:#666!important;'>Lock-in</td><td style='padding: 6px; text-align: right; font-weight: bold; color:#1a1a1a!important;'>{_LOCKIN_PHRASE[lock]}</td></tr>\n"
        md += f"<tr><td style='padding: 6px; color:#666!important;'>Amplitude</td><td style='padding: 6px; text-align: right; font-weight: bold; color:#1a1a1a!important;'>{trend.value} ({s.rms_amplitude_over_c:.3f}c)</td></tr>\n"
        md += "</table>\n"
        md += "</div>\n\n"

        md += "<div style='background:#fff3cd!important; color:#1a1a1a!important; padding: 10px; border-radius: 6px; border-left: 3px solid #ffc107;'>\n"
        md += f"<p style='margin: 0; font-size: 13px; color:#1a1a1a!important;'><strong style='color:#1a1a1a!important;'>Key insight{_info('key_insight')}:</strong> {_MECHANISM[regime]}</p>\n"
        md += "</div>\n\n"

        md += _build_confidence_card(fr)

        # Regime probability breakdown (all five regimes, normalized).
        from Data.describe import regime_scores
        scores = regime_scores(fr)
        md += "<div style='background:#f8f9fa!important; color:#1a1a1a!important; padding: 12px; border-radius: 8px;'>\n"
        md += f"<p style='margin: 0 0 6px 0; font-size: 13px; color:#666!important;'><strong style='color:#666!important;'>Regime score breakdown</strong>{_info('score_breakdown')}</p>\n"
        for r, sc in sorted(scores.items(), key=lambda kv: -kv[1]):
            w = sc * 100
            md += "<div style='display:flex; align-items:center; gap:8px; margin:3px 0;'>\n"
            md += f"<span style='width:110px; font-size:12px; color:#1a1a1a!important;'>{_REGIME_PHRASE[r]}</span>\n"
            md += "<div style='flex:1; background:#eee; border-radius:5px; height:12px;'>\n"
            md += f"<div style='width:{w:.1f}%; height:12px; background:#667eea; border-radius:5px;'></div>\n"
            md += "</div>\n"
            md += f"<span style='width:44px; font-size:12px; text-align:right; color:#1a1a1a!important;'><code>{sc:.2f}</code></span>\n"
            md += "</div>\n"
        md += "</div>\n\n"

        md += "</div>"
        return md

    def _structured_breakdown(fr: FeatureRecord) -> Tuple[str, str, str, str]:
        """Return (summary, warnings, metrics_md, interpretation_md) for Tab 2."""
        from features.extract import check_physical_consistency

        warns = fr.range_warnings() + check_physical_consistency(fr)
        metrics_md = "### Complete Feature Record\n\n" + _build_metrics_table(fr)
        diagnostics = _lockin_gauge(fr) + _pod_spectrum(fr)
        interp_md = (
            "### Interpretation\n\n"
            + _build_interpretation_cards(fr)
            + "\n\n### Visual Diagnostics\n\n"
            + "<div style='display:grid; grid-template-columns:repeat(auto-fit, minmax(380px, 1fr)); gap: 12px; align-items:stretch;'>\n\n"
            + diagnostics
            + "</div>\n\n"
        )
        warnings_md = (
            "### ⚠️ Data Quality Warnings\n\n"
            + "\n".join(f"- {w}" for w in warns)
            if warns
            else ""
        )
        return _build_summary_card(fr), warnings_md, metrics_md, interp_md

    def _warn_update(warnings_md: str):
        # Hide the warnings column entirely when there's nothing to show —
        # keeps the wide right-hand column from sitting empty.
        return gr.update(value=warnings_md, visible=bool(warnings_md))

    def interpret(fr_dict, use_rag):
        empty = ("### Quick Summary", gr.update(value="", visible=False), "### Complete Feature Record", "", "")
        if not fr_dict:
            return ("Provide FSI data first (tab 1), then interpret.",) + empty[1:]
        fr = FeatureRecord.model_validate(fr_dict)
        summary, warnings, metrics_md, interp_md = _structured_breakdown(fr)
        warn_upd = _warn_update(warnings)

        if not _has_trained_model():
            raw_text = _describe_fn(fr)
            interp_full = (
                interp_md
                + "\n\n---\n\n### Raw Rule-Based Output\n\n"
                + raw_text.replace("\n", "  \n")
            )
            return summary, warn_upd, metrics_md, interp_full, _passages_markdown([])

        try:
            runner = _get_runner()
            retrieved = runner.retrieve(fr) if use_rag else []
            raw_text = runner.generate(fr, retrieved=retrieved)
            interp_full = (
                interp_md
                + "\n\n---\n\n### Model Output\n\n"
                + raw_text.replace("\n", "  \n")
            )
            return summary, warn_upd, metrics_md, interp_full, _passages_markdown(retrieved)
        except Exception as e:  # noqa: BLE001
            raw_text = _describe_fn(fr)
            interp_full = (
                interp_md
                + "\n\n---\n\n"
                + f"### ⚠️ Model inference failed: {e}\n\n"
                + "Falling back to rule-based output.\n\n"
                + raw_text.replace("\n", "  \n")
            )
            return summary, warn_upd, metrics_md, interp_full, _passages_markdown([])

    def download_report(fr_dict):
        """Write a clean markdown interpretation report and return its path."""
        if not fr_dict:
            return None
        import tempfile
        fr = FeatureRecord.model_validate(fr_dict)
        regime = fr.ground_truth_regime or classify_regime(fr)
        from Data.describe import _REGIME_PHRASE

        parts = [
            f"# FSI Interpretation Report — {fr.experiment_id}",
            "",
            f"**Flow regime:** {_REGIME_PHRASE[regime].title()}  ",
            f"**Lock-in:** {fr.lock_in_status().value}",
            "",
            "## Feature Record",
            "",
            _build_metrics_table(fr),
            "",
            "## Rule-Based Description",
            "",
            _describe_fn(fr),
        ]
        from features.extract import check_physical_consistency

        warns = fr.range_warnings() + check_physical_consistency(fr)
        if warns:
            parts += ["", "## Data Quality Warnings", ""]
            parts += [f"- {w}" for w in warns]
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in fr.experiment_id)
        path = Path(tempfile.gettempdir()) / f"fsi_report_{safe_id}.md"
        path.write_text("\n".join(parts))
        return str(path)

    def ask(fr_dict, question, use_rag, history, chat):
        history = history or []
        chat = chat or []
        if not question or not question.strip():
            return history, chat, ""
        if not fr_dict:
            chat = chat + [
                {"role": "user", "content": question},
                {"role": "assistant", "content": "Provide FSI data first (tab 1) before asking."},
            ]
            return history, chat, ""

        fr = FeatureRecord.model_validate(fr_dict)
        runner = _get_runner()
        answer_text, retrieved = runner.answer(
            fr, question, use_rag=use_rag, history=history
        )

        display = answer_text
        if retrieved:
            cites = ", ".join(sorted({r.source for r in retrieved}))
            display = f"{answer_text}\n\n_Sources: {cites}_"

        history = history + [(question, answer_text)]
        chat = chat + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": display},
        ]
        return history, chat, ""

    def reset_chat():
        return [], []

    _SWEEP_EMOJI = {
        "attached": "🟢", "separated": "🟡", "viv_lockin": "🔴",
        "galloping": "🟠", "stall_flutter": "🟣",
    }

    def run_sweep(fr_dict, param, lo, hi, n):
        if not fr_dict:
            return "Provide FSI data first (tab 1), then sweep."
        from features.sweep import find_transitions, sweep_parameter

        fr = FeatureRecord.model_validate(fr_dict)
        try:
            points = sweep_parameter(
                fr,
                param=param,
                lo=None if lo is None else float(lo),
                hi=None if hi is None else float(hi),
                n=int(n),
            )
        except Exception as e:  # noqa: BLE001
            return f"❌ Sweep failed: {e}"

        lines = [
            f"### Regime map — sweeping `{param}` "
            f"({points[0].value:.2f} → {points[-1].value:.2f}, {len(points)} points)",
            "",
            f"| {param} | regime | confidence | f_shed (Hz) | lock-in |",
            "|---|---|---|---|---|",
        ]
        for pt in points:
            emoji = _SWEEP_EMOJI.get(pt.regime.value, "⚪")
            lines.append(
                f"| {pt.value:.2f} | {emoji} {pt.regime.value} | {pt.confidence:.2f} "
                f"| {pt.shedding_frequency_hz:.2f} | {pt.lock_in} |"
            )
        transitions = find_transitions(points)
        if transitions:
            lines += ["", "**Regime transitions:**", ""]
            for value, frm, to in transitions:
                lines.append(f"- at `{param} ≈ {value:.2f}`: {frm.value} → {to.value}")
        else:
            lines += ["", "_No regime transition inside this range._"]
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Layout
    # ------------------------------------------------------------------ #
    with gr.Blocks(title="FSI-SLM Interpreter") as demo:
        gr.Markdown(
            "# FSI-SLM Interpreter\n"
            "TAMUK 2026\n"
            "Provide real fluid-structure-interaction measurements, then get a "
            "structured interpretation or ask free-text questions about the "
            "record. The feature record is the only source of truth for measured "
            "numbers; retrieved literature passages add domain context only.\n\n"
            f"**Model:** {'trained adapter' if _has_trained_model() else 'rule-based fallback (no adapter found)'}"
        )

        fr_state = gr.State(None)
        history_state = gr.State([])

        # ---- Tab 1: provide data ---- #
        with gr.Tab("1 · Provide FSI data"):
            with gr.Tab("Manual entry"):
                experiment_id = gr.Textbox(label="experiment_id", value="manual_entry")
                gr.Markdown("**Flow conditions**")
                with gr.Row():
                    reynolds_number = gr.Number(label="reynolds_number", value=1.5e5)
                    angle_of_attack_deg = gr.Number(label="angle_of_attack_deg", value=8.0)
                    freestream_velocity_m_s = gr.Number(label="freestream_velocity_m_s", value=2.0)
                    reduced_velocity = gr.Number(label="reduced_velocity (U*)", value=6.0)
                    chord_m = gr.Number(label="chord_m", value=0.1)
                gr.Markdown("**Pressure & Strain**")
                with gr.Row():
                    pressure = gr.Slider(label="Pressure (Pa)", minimum=0, maximum=200000, value=101325)
                    strain = gr.Slider(label="Strain", minimum=0, maximum=10, value=0.01)
                gr.Markdown("**Example Inputs**")
                examples = [
                    [101325, 0.01],
                    [50000, 0.005],
                ]
                gr.Examples(examples=examples, inputs=[pressure, strain], label="Select Example Data")
                
                gr.Markdown("**PIV-derived**")
                with gr.Row():
                    shedding_frequency_hz = gr.Number(label="shedding_frequency_hz", value=12.0)
                    strouhal_number = gr.Number(label="strouhal_number", value=0.6)
                    wake_width_over_c = gr.Number(label="wake_width_over_c", value=0.8)
                    separation_location_x_over_c = gr.Number(label="separation_location_x_over_c", value=0.6)
                    turbulent_kinetic_energy = gr.Number(label="turbulent_kinetic_energy", value=0.05)
                pod_mode_amplitudes = gr.Textbox(
                    label="pod_mode_amplitudes (comma-separated)", value="0.6, 0.2, 0.1, 0.05"
                )
                gr.Markdown("**Structure**")
                with gr.Row():
                    rms_amplitude_over_c = gr.Number(label="rms_amplitude_over_c", value=0.3)
                    dominant_frequency_hz = gr.Number(label="dominant_frequency_hz", value=12.0)
                    natural_frequency_hz = gr.Number(label="natural_frequency_hz", value=12.0)
                    mode_number = gr.Number(label="mode_number", value=1, precision=0)
                    camber = gr.Number(label="camber", value=0.02)
                gr.Markdown("**Force coefficients**")
                with gr.Row():
                    cl_mean = gr.Number(label="cl_mean", value=0.8)
                    cl_rms = gr.Number(label="cl_rms", value=0.4)
                    cl_dominant_frequency_hz = gr.Number(label="cl_dom_freq_hz", value=12.0)
                with gr.Row():
                    cd_mean = gr.Number(label="cd_mean", value=0.2)
                    cd_rms = gr.Number(label="cd_rms", value=0.1)
                    cd_dominant_frequency_hz = gr.Number(label="cd_dom_freq_hz", value=24.0)
                with gr.Row():
                    cm_mean = gr.Number(label="cm_mean", value=-0.05)
                    cm_rms = gr.Number(label="cm_rms", value=0.05)
                    cm_dominant_frequency_hz = gr.Number(label="cm_dom_freq_hz", value=12.0)
                build_form_btn = gr.Button("Build record from form", variant="primary")

            with gr.Tab("Upload RawExperiment JSON"):
                gr.Markdown(
                    "Upload a `RawExperiment` JSON (time-resolved signals + PIV "
                    "snapshots). It is run through `features/extract.py` to produce "
                    "the FeatureRecord."
                )
                raw_file = gr.File(label="RawExperiment JSON", file_types=[".json"])
                build_upload_btn = gr.Button("Extract features from upload", variant="primary")

            build_status = gr.Markdown()
            record_json = gr.JSON(label="Active FeatureRecord")

        # ---- Tab 2: structured interpretation ---- #
        with gr.Tab("2 · Structured interpretation"):
            use_rag_struct = gr.Checkbox(label="Use RAG (retrieve literature passages)", value=False)
            with gr.Row():
                interpret_btn = gr.Button("Interpret record", variant="primary")
                report_btn = gr.DownloadButton("⬇️ Download report (.md)")
            with gr.Row():
                summary_out = gr.Markdown("### Quick Summary", sanitize_html=False)
            warnings_out = gr.Markdown("", sanitize_html=False, visible=False)
            with gr.Row():
                metrics_out = gr.Markdown("### Complete Feature Record", sanitize_html=False)
            with gr.Row():
                interp_out = gr.Markdown(label="Structured interpretation", sanitize_html=False)
            interp_passages = gr.Markdown()

        # ---- Tab: what-if sweep ---- #
        with gr.Tab("What-if sweep"):
            gr.Markdown(
                "Vary one operating parameter across a range (holding the rest "
                "of the record fixed, with physically coupled quantities "
                "rescaled) and map the predicted regime at each point — e.g. "
                "find the lock-in window in reduced velocity. Instant, "
                "rule-based, no model load."
            )
            with gr.Row():
                sweep_param = gr.Dropdown(
                    label="Parameter",
                    choices=["reduced_velocity", "angle_of_attack_deg", "natural_frequency_hz"],
                    value="reduced_velocity",
                )
                sweep_lo = gr.Number(label="From (blank = auto)", value=None)
                sweep_hi = gr.Number(label="To (blank = auto)", value=None)
                sweep_n = gr.Slider(label="Points", minimum=5, maximum=41, step=2, value=15)
            sweep_btn = gr.Button("Run sweep", variant="primary")
            sweep_out = gr.Markdown()

        # ---- Tab 3: free-text Q&A ---- #
        with gr.Tab("3 · Ask questions"):
            gr.Markdown(
                "Ask open-ended questions about the active record. Answered by the "
                "base instruct model grounded in the record (+ optional retrieved "
                "passages). The structured adapter is **not** used here."
            )
            use_rag_qa = gr.Checkbox(label="Use RAG (ground answers in literature)", value=True)
            chatbot = gr.Chatbot(label="Q&A about the active record", height=380)
            question_box = gr.Textbox(
                label="Your question",
                placeholder="e.g. Why is the amplitude saturating? Is St consistent with lock-in?",
            )
            with gr.Row():
                ask_btn = gr.Button("Ask", variant="primary")
                clear_btn = gr.Button("Clear conversation")

        # ------------------------------------------------------------------ #
        # Wiring
        # ------------------------------------------------------------------ #
        form_inputs = [
            experiment_id,
            reynolds_number, angle_of_attack_deg, freestream_velocity_m_s,
            reduced_velocity, chord_m,
            shedding_frequency_hz, strouhal_number, wake_width_over_c,
            separation_location_x_over_c, pod_mode_amplitudes, turbulent_kinetic_energy,
            pressure, strain,
            rms_amplitude_over_c, dominant_frequency_hz, natural_frequency_hz,
            mode_number, camber,
            cl_mean, cl_rms, cl_dominant_frequency_hz,
            cd_mean, cd_rms, cd_dominant_frequency_hz,
            cm_mean, cm_rms, cm_dominant_frequency_hz,
        ]
        build_form_btn.click(
            build_from_form, inputs=form_inputs,
            outputs=[fr_state, record_json, build_status],
        )
        build_upload_btn.click(
            build_from_upload, inputs=[raw_file],
            outputs=[fr_state, record_json, build_status],
        )
        interpret_btn.click(
            interpret, inputs=[fr_state, use_rag_struct],
            outputs=[summary_out, warnings_out, metrics_out, interp_out, interp_passages],
        )
        report_btn.click(
            download_report, inputs=[fr_state], outputs=[report_btn],
        )
        ask_btn.click(
            ask,
            inputs=[fr_state, question_box, use_rag_qa, history_state, chatbot],
            outputs=[history_state, chatbot, question_box],
        )
        question_box.submit(
            ask,
            inputs=[fr_state, question_box, use_rag_qa, history_state, chatbot],
            outputs=[history_state, chatbot, question_box],
        )
        clear_btn.click(reset_chat, outputs=[history_state, chatbot])
        sweep_btn.click(
            run_sweep,
            inputs=[fr_state, sweep_param, sweep_lo, sweep_hi, sweep_n],
            outputs=[sweep_out],
        )

    return demo


# --------------------------------------------------------------------------- #
# Direct execution
# --------------------------------------------------------------------------- #
def main(
    config_path: str = "configs/default.yaml",
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
) -> None:
    """Launch the Gradio app directly."""
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)
    demo = create_ui(config=config)
    demo.launch(server_name=host, server_port=port, share=share)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FSI-SLM Gradio UI")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    main(config_path=args.config, host=args.host, port=args.port, share=args.share)
