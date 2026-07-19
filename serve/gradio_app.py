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


def _info(text: str, color: str = "#1a1a1a") -> str:
    """Hoverable ⓘ badge: native browser tooltip via the `title` attr, no JS."""
    t = html.escape(text, quote=True)
    return (
        f"<span title=\"{t}\" style='cursor:help; color:{color}!important; "
        f"opacity:0.7; font-size:12px; border:1px solid {color}; border-radius:50%; "
        "width:15px; height:15px; display:inline-flex; align-items:center; "
        "justify-content:center; margin-left:6px; vertical-align:middle; "
        "font-family:serif;'>i</span>"
    )


_BOX_INFO = {
    "regime": (
        "Classified flow regime. 5 possible values: (1) Attached flow — boundary "
        "layer stays attached, low unsteady loading, no coherent instability. "
        "(2) Separated flow — flow separates early / wide broadband wake, but no "
        "sustained resonant structural response. (3) VIV lock-in — shedding "
        "frequency has synchronized (within ~10%) with the structure's natural "
        "frequency, driving a resonant, amplitude-saturated limit cycle. "
        "(4) Galloping — growing-amplitude instability from negative aerodynamic "
        "damping, structural frequency well below (<50% of) shedding frequency. "
        "(5) Stall flutter — large single-mode limit cycle from dynamic stall at "
        "high angle of attack (>=12 degrees). Checked in that priority order."
    ),
    "frequency": (
        "Raw frequency data feeding the regime call. Shedding frequency = "
        "dominant wake tone from PIV/FFT. Strouhal number St = f_shed * chord / "
        "freestream_velocity — dimensionless shedding rate; ~0.2 is the classic "
        "2D bluff-body baseline, so a big deviation flags an unusual shape, a "
        "non-shedding-dominated regime, or a possible units error. Structural "
        "natural frequency is shown alongside for direct comparison against "
        "shedding frequency (see Lock-In Status)."
    ),
    "lockin": (
        "Ratio = f_shed / f_natural, three possible states. locked: ratio in "
        "0.90-1.10 (within 10%) — shedding has synchronized with the structure, "
        "i.e. resonance; this is what defines VIV lock-in. near lock-in: ratio "
        "in 0.80-1.20 but outside the locked band — approaching resonance, not "
        "fully synced. unlocked: ratio outside 0.80-1.20 — frequencies are "
        "independent, no synchronization."
    ),
    "amplitude": (
        "Oscillation amplitude trend (RMS displacement / chord), assigned from "
        "the classified regime. negligible: RMS/c < 0.01, no meaningful motion. "
        "growing: assigned for galloping — amplitude increases without an "
        "obvious ceiling until structural nonlinearity intervenes. saturated: "
        "assigned for VIV lock-in and stall flutter — amplitude has plateaued "
        "into a stable, bounded limit cycle. decaying: assigned for separated "
        "flow — any structural response is dying out, not sustained."
    ),
    "mode_shape": (
        "Which structural vibration mode dominates (mode_number) and its "
        "physical deflection shape. Mode 1 is typically first bending/heave — "
        "whole structure moves up-down together, no internal node. Higher mode "
        "numbers correspond to higher-order shapes (e.g. torsion, S-shaped "
        "bending with internal nodes); which mode is excited changes where "
        "peak stress occurs on the structure."
    ),
    "gauge": (
        "Visual version of Lock-In Status: track spans ratio f_shed/f_natural "
        "from 0.5 to 1.5. Green band (0.90-1.10) = locked/resonant. Amber bands "
        "(0.80-0.90 and 1.10-1.20) = near lock-in. Grey outside amber = "
        "unlocked. Black marker = this record's exact ratio. Dead center of "
        "green (ratio = 1.0) is perfect resonance."
    ),
    "pod": (
        "Proper Orthogonal Decomposition (POD) splits the wake's spatial "
        "velocity field into orthogonal energy modes ranked by kinetic energy "
        "share — this is a FLUID/wake mode decomposition, separate from the "
        "structural mode_number in the Mode Shape card. Bars = each mode's "
        "amplitude, normalized to the largest. Mode 1 holding most energy "
        "(e.g. >60%) means one coherent structure (clean periodic shedding). "
        "Energy spread evenly across many modes means a more complex, "
        "turbulent, or noisy wake that resists a single-frequency description."
    ),
    "confidence": (
        "Composite trust score for the regime call, built from 4 signals. "
        "Classifier confidence [0,1]: how strongly this record satisfies the "
        "winning regime's decision rule. Margin over runner-up: gap between "
        "the winner's score and the 2nd-most-likely regime in the breakdown "
        "below — small margin (<0.15) means the record sits near a decision "
        "boundary and could flip with slightly different inputs. Range/"
        "consistency warnings: count of fields outside physically expected "
        "bounds or mutually inconsistent (e.g. a units mismatch) — each one "
        "lowers trust. Noise/signal-quality flags: only available when the "
        "record has real spectral stats (skewness, kurtosis, spectral "
        "kurtosis, coherence) from an uploaded RawExperiment/PIV signal, not "
        "manual entry — flags asymmetric/impulsive signals, flat/broadband "
        "spectra (weak tonal shedding), or weak structure-force coherence "
        "(undermines the claimed mechanism). Overall: HIGH = confidence>0.75 "
        "AND margin>0.3 AND 0 warnings AND 0 noise flags. LOW = confidence<0.5 "
        "OR >=2 warnings OR >=2 noise flags. MEDIUM = everything in between."
    ),
    "key_insight": (
        "One-line physical mechanism for why the classified regime occurs. "
        "Fixed text per regime from a lookup table (not computed per-record) — "
        "explains the underlying physics, not this record's specific numbers."
    ),
    "score_breakdown": (
        "Normalized likelihood for all 5 possible regimes on this record; "
        "scores sum to 1.0. The winning regime keeps its calibrated confidence "
        "score; the rest of the probability mass (1 - confidence) is split "
        "across the other 4 regimes proportional to each one's own raw "
        "rule-based evidence. One tall bar with the rest near-zero = "
        "unambiguous call. Several comparable bars = the record sits near a "
        "multi-way boundary — treat the top regime as a best guess, not a "
        "certainty."
    ),
    "summary_lockin_amp": (
        "Compact restatement of Lock-In Status and Amplitude Response — "
        "hover the full cards further down this page for the complete "
        "definitions of every possible value."
    ),
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

        md = "<div style='display: flex; flex-direction: column; gap: 12px;'>\n\n"

        # Card 1: Regime
        md += f"<div style='background:#f0f8ff!important; color:#1a1a1a!important; padding: 12px; border-radius: 8px; border-left: 4px solid #2196F3;'>\n"
        md += f"<h3 style='color:#1a1a1a!important; margin:0 0 4px 0;'>{regime_emoji} Flow Regime{_info(_BOX_INFO['regime'])}</h3>\n"
        md += f"<p style='font-size: 18px; margin: 4px 0; color:#1a1a1a!important;'><strong style='color:#1a1a1a!important;'>{_REGIME_PHRASE[regime].title()}</strong></p>\n"
        md += f"<p style='margin: 4px 0; color:#555!important;'>{_MECHANISM[regime]}</p>\n"
        md += "</div>\n\n"

        # Card 2: Frequency
        md += f"<div style='background:#fff8e1!important; color:#1a1a1a!important; padding: 12px; border-radius: 8px; border-left: 4px solid #FFC107;'>\n"
        md += f"<h3 style='color:#1a1a1a!important; margin:0 0 4px 0;'>📊 Frequency & Strouhal Number{_info(_BOX_INFO['frequency'])}</h3>\n"
        md += f"<table style='width: 100%; border-collapse: collapse;'>\n"
        md += f"<tr><td style='padding: 6px; color:#1a1a1a!important;'><strong style='color:#1a1a1a!important;'>Shedding frequency</strong></td><td style='padding: 6px; text-align: right; color:#1a1a1a!important;'><code>{p.shedding_frequency_hz:.2f} Hz</code></td></tr>\n"
        md += f"<tr><td style='padding: 6px; color:#1a1a1a!important;'><strong style='color:#1a1a1a!important;'>Strouhal number (St)</strong></td><td style='padding: 6px; text-align: right; color:#1a1a1a!important;'><code>{p.strouhal_number:.3f}</code></td></tr>\n"
        md += f"<tr><td style='padding: 6px; color:#1a1a1a!important;'><strong style='color:#1a1a1a!important;'>Structural natural freq</strong></td><td style='padding: 6px; text-align: right; color:#1a1a1a!important;'><code>{s.natural_frequency_hz:.2f} Hz</code></td></tr>\n"
        md += f"</table>\n"
        md += "</div>\n\n"

        # Card 3: Lock-in
        lock_color = "#4CAF50" if _LOCKIN_PHRASE[lock] == "locked" else ("#FF9800" if _LOCKIN_PHRASE[lock] == "near lock-in" else "#F44336")
        md += f"<div style='background:#fce4ec!important; color:#1a1a1a!important; padding: 12px; border-radius: 8px; border-left: 4px solid {lock_color};'>\n"
        md += f"<h3 style='color:#1a1a1a!important; margin:0 0 4px 0;'>🔒 Lock-In Status{_info(_BOX_INFO['lockin'])}</h3>\n"
        md += f"<p style='font-size: 16px; margin: 4px 0; color:#1a1a1a!important;'><strong style='color:#1a1a1a!important;'>{_LOCKIN_PHRASE[lock].title()}</strong></p>\n"
        md += f"<p style='margin: 4px 0; color:#555!important;'>Shedding at {p.shedding_frequency_hz:.2f} Hz vs structural natural frequency {s.natural_frequency_hz:.2f} Hz</p>\n"
        md += "</div>\n\n"

        # Card 4: Amplitude
        md += f"<div style='background:#e8f5e9!important; color:#1a1a1a!important; padding: 12px; border-radius: 8px; border-left: 4px solid #4CAF50;'>\n"
        md += f"<h3 style='color:#1a1a1a!important; margin:0 0 4px 0;'>📈 Amplitude Response{_info(_BOX_INFO['amplitude'])}</h3>\n"
        md += f"<p style='margin: 4px 0; color:#1a1a1a!important;'><strong style='color:#1a1a1a!important;'>Trend:</strong> <code>{trend.value}</code></p>\n"
        md += f"<p style='margin: 4px 0; color:#1a1a1a!important;'><strong style='color:#1a1a1a!important;'>RMS displacement:</strong> <code>{s.rms_amplitude_over_c:.4f}c</code></p>\n"
        md += "</div>\n\n"

        # Card 5: Mode shape
        md += f"<div style='background:#f3e5f5!important; color:#1a1a1a!important; padding: 12px; border-radius: 8px; border-left: 4px solid #9C27B0;'>\n"
        md += f"<h3 style='color:#1a1a1a!important; margin:0 0 4px 0;'>🎭 Mode Shape{_info(_BOX_INFO['mode_shape'])}</h3>\n"
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
        md = "<div style='background:#fce4ec!important; color:#1a1a1a!important; padding: 12px; border-radius: 8px; border-left: 4px solid #E91E63;'>\n"
        md += f"<h3 style='color:#1a1a1a!important; margin:0 0 4px 0;'>🎯 Frequency Lock-In Gauge{_info(_BOX_INFO['gauge'])}</h3>\n"
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
        md = "<div style='background:#e3f2fd!important; color:#1a1a1a!important; padding: 12px; border-radius: 8px; border-left: 4px solid #1976D2;'>\n"
        md += f"<h3 style='color:#1a1a1a!important; margin:0 0 4px 0;'>🌊 POD Mode Energy Spectrum{_info(_BOX_INFO['pod'])}</h3>\n"
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

        md = f"<div style='background:#eceff1!important; color:#1a1a1a!important; padding: 12px; border-radius: 8px; border-left: 4px solid {overall_color};'>\n"
        md += f"<h3 style='color:#1a1a1a!important; margin:0 0 6px 0;'>{overall_emoji} Interpretation Confidence: {overall.upper()}{_info(_BOX_INFO['confidence'])}</h3>\n"
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

        md = "<div style='display: flex; flex-direction: column; gap: 8px;'>\n\n"
        md += f"<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color:#fff!important; padding: 14px; border-radius: 8px;'>\n"
        md += f"<p style='margin: 0; font-size: 14px; opacity: 0.9; color:#fff!important;'>Flow regime{_info(_BOX_INFO['regime'], color='#fff')}</p>\n"
        md += f"<p style='margin: 4px 0 0 0; font-size: 20px; font-weight: bold; color:#fff!important;'>{regime_emoji} {_REGIME_PHRASE[regime].title()}</p>\n"
        md += "</div>\n\n"

        md += "<div style='background:#f8f9fa!important; color:#1a1a1a!important; padding: 12px; border-radius: 8px;'>\n"
        md += f"<p style='margin:0 0 4px 0; font-size:11px; color:#888!important;'>Lock-in / Amplitude{_info(_BOX_INFO['summary_lockin_amp'])}</p>\n"
        md += "<table style='width: 100%;'>\n"
        md += f"<tr><td style='padding: 6px; color:#666!important;'>Lock-in</td><td style='padding: 6px; text-align: right; font-weight: bold; color:#1a1a1a!important;'>{_LOCKIN_PHRASE[lock]}</td></tr>\n"
        md += f"<tr><td style='padding: 6px; color:#666!important;'>Amplitude</td><td style='padding: 6px; text-align: right; font-weight: bold; color:#1a1a1a!important;'>{trend.value} ({s.rms_amplitude_over_c:.3f}c)</td></tr>\n"
        md += "</table>\n"
        md += "</div>\n\n"

        md += "<div style='background:#fff3cd!important; color:#1a1a1a!important; padding: 10px; border-radius: 6px; border-left: 3px solid #ffc107;'>\n"
        md += f"<p style='margin: 0; font-size: 13px; color:#1a1a1a!important;'><strong style='color:#1a1a1a!important;'>Key insight{_info(_BOX_INFO['key_insight'])}:</strong> {_MECHANISM[regime]}</p>\n"
        md += "</div>\n\n"

        md += _build_confidence_card(fr)

        # Regime probability breakdown (all five regimes, normalized).
        from Data.describe import regime_scores
        scores = regime_scores(fr)
        md += "<div style='background:#f8f9fa!important; color:#1a1a1a!important; padding: 12px; border-radius: 8px;'>\n"
        md += f"<p style='margin: 0 0 6px 0; font-size: 13px; color:#666!important;'><strong style='color:#666!important;'>Regime score breakdown</strong>{_info(_BOX_INFO['score_breakdown'])}</p>\n"
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
        interp_md = (
            "### Interpretation\n\n"
            + _build_interpretation_cards(fr)
            + "\n\n### Visual Diagnostics\n\n"
            + _lockin_gauge(fr)
            + _pod_spectrum(fr)
        )
        warnings_md = (
            "### ⚠️ Data Quality Warnings\n\n"
            + "\n".join(f"- {w}" for w in warns)
            if warns
            else ""
        )
        return _build_summary_card(fr), warnings_md, metrics_md, interp_md

    def interpret(fr_dict, use_rag):
        empty = ("### Quick Summary", "", "### Complete Feature Record", "", "")
        if not fr_dict:
            return ("Provide FSI data first (tab 1), then interpret.",) + empty[1:]
        fr = FeatureRecord.model_validate(fr_dict)
        summary, warnings, metrics_md, interp_md = _structured_breakdown(fr)

        if not _has_trained_model():
            raw_text = _describe_fn(fr)
            interp_full = (
                interp_md
                + "\n\n---\n\n### Raw Rule-Based Output\n\n"
                + raw_text.replace("\n", "  \n")
            )
            return summary, warnings, metrics_md, interp_full, _passages_markdown([])

        try:
            runner = _get_runner()
            retrieved = runner.retrieve(fr) if use_rag else []
            raw_text = runner.generate(fr, retrieved=retrieved)
            interp_full = (
                interp_md
                + "\n\n---\n\n### Model Output\n\n"
                + raw_text.replace("\n", "  \n")
            )
            return summary, warnings, metrics_md, interp_full, _passages_markdown(retrieved)
        except Exception as e:  # noqa: BLE001
            raw_text = _describe_fn(fr)
            interp_full = (
                interp_md
                + "\n\n---\n\n"
                + f"### ⚠️ Model inference failed: {e}\n\n"
                + "Falling back to rule-based output.\n\n"
                + raw_text.replace("\n", "  \n")
            )
            return summary, warnings, metrics_md, interp_full, _passages_markdown([])

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
                with gr.Column(scale=1):
                    summary_out = gr.Markdown("### Quick Summary", sanitize_html=False)
                with gr.Column(scale=2):
                    warnings_out = gr.Markdown("", sanitize_html=False)
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
