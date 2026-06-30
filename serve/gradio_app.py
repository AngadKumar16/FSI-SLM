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

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

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

        # Flow conditions
        lines.append(f"| **Conditions** | Reynolds number | {c.reynolds_number:.2e} | |")
        lines.append(f"| | Angle of attack | {c.angle_of_attack_deg:.1f}° | |")
        lines.append(f"| | Freestream velocity | {c.freestream_velocity_m_s:.2f} m/s | |")
        lines.append(f"| | Reduced velocity (U*) | {c.reduced_velocity:.2f} | |")
        lines.append(f"| | Chord | {c.chord_m:.3f} m | |")

        # PIV
        lines.append(f"| **PIV** | Shedding frequency | {p.shedding_frequency_hz:.2f} Hz | Dominant |")
        lines.append(f"| | Strouhal number | {p.strouhal_number:.3f} | St = fD/U |")
        lines.append(f"| | Wake width | {p.wake_width_over_c:.3f}c | >0.6 suggests separation |")
        lines.append(f"| | Separation location | {p.separation_location_x_over_c:.3f}c | <0.6 is early |")
        lines.append(f"| | Turbulent KE | {p.turbulent_kinetic_energy:.4f} | |")

        # Structure
        lines.append(f"| **Structure** | RMS amplitude | {s.rms_amplitude_over_c:.4f}c | {trend.value} |")
        lines.append(f"| | Dominant frequency | {s.dominant_frequency_hz:.2f} Hz | |")
        lines.append(f"| | Natural frequency | {s.natural_frequency_hz:.2f} Hz | |")
        lines.append(f"| | Mode number | {int(s.mode_number)} | {_mode_shape_phrase(fr)} |")
        lines.append(f"| | Camber | {s.camber:.3f} | |")

        # Forces
        lines.append(f"| **Forces** | CL mean | {f.cl_mean:.3f} | |")
        lines.append(f"| | CL rms | {f.cl_rms:.3f} | |")
        lines.append(f"| | CL dom. freq | {f.cl_dominant_frequency_hz:.2f} Hz | |")
        lines.append(f"| | CD mean | {f.cd_mean:.3f} | |")
        lines.append(f"| | CD rms | {f.cd_rms:.3f} | |")
        lines.append(f"| | CD dom. freq | {f.cd_dominant_frequency_hz:.2f} Hz | |")
        lines.append(f"| | CM mean | {f.cm_mean:.4f} | |")
        lines.append(f"| | CM rms | {f.cm_rms:.4f} | |")
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
        md += f"<div style='background: #f0f8ff; padding: 12px; border-radius: 8px; border-left: 4px solid #2196F3;'>\n"
        md += f"<h3>{regime_emoji} Flow Regime</h3>\n"
        md += f"<p style='font-size: 18px; margin: 4px 0;'><strong>{_REGIME_PHRASE[regime].title()}</strong></p>\n"
        md += f"<p style='margin: 4px 0; color: #555;'>{_MECHANISM[regime]}</p>\n"
        md += "</div>\n\n"

        # Card 2: Frequency
        md += f"<div style='background: #fff8e1; padding: 12px; border-radius: 8px; border-left: 4px solid #FFC107;'>\n"
        md += f"<h3>📊 Frequency & Strouhal Number</h3>\n"
        md += f"<table style='width: 100%; border-collapse: collapse;'>\n"
        md += f"<tr><td style='padding: 6px;'><strong>Shedding frequency</strong></td><td style='padding: 6px; text-align: right;'><code>{p.shedding_frequency_hz:.2f} Hz</code></td></tr>\n"
        md += f"<tr><td style='padding: 6px;'><strong>Strouhal number (St)</strong></td><td style='padding: 6px; text-align: right;'><code>{p.strouhal_number:.3f}</code></td></tr>\n"
        md += f"<tr><td style='padding: 6px;'><strong>Structural natural freq</strong></td><td style='padding: 6px; text-align: right;'><code>{s.natural_frequency_hz:.2f} Hz</code></td></tr>\n"
        md += f"</table>\n"
        md += "</div>\n\n"

        # Card 3: Lock-in
        lock_color = "#4CAF50" if _LOCKIN_PHRASE[lock] == "locked" else ("#FF9800" if _LOCKIN_PHRASE[lock] == "near lock-in" else "#F44336")
        md += f"<div style='background: #fce4ec; padding: 12px; border-radius: 8px; border-left: 4px solid {lock_color};'>\n"
        md += f"<h3>🔒 Lock-In Status</h3>\n"
        md += f"<p style='font-size: 16px; margin: 4px 0;'><strong>{_LOCKIN_PHRASE[lock].title()}</strong></p>\n"
        md += f"<p style='margin: 4px 0; color: #555;'>Shedding at {p.shedding_frequency_hz:.2f} Hz vs structural natural frequency {s.natural_frequency_hz:.2f} Hz</p>\n"
        md += "</div>\n\n"

        # Card 4: Amplitude
        md += f"<div style='background: #e8f5e9; padding: 12px; border-radius: 8px; border-left: 4px solid #4CAF50;'>\n"
        md += f"<h3>📈 Amplitude Response</h3>\n"
        md += f"<p style='margin: 4px 0;'><strong>Trend:</strong> <code>{trend.value}</code></p>\n"
        md += f"<p style='margin: 4px 0;'><strong>RMS displacement:</strong> <code>{s.rms_amplitude_over_c:.4f}c</code></p>\n"
        md += "</div>\n\n"

        # Card 5: Mode shape
        md += f"<div style='background: #f3e5f5; padding: 12px; border-radius: 8px; border-left: 4px solid #9C27B0;'>\n"
        md += f"<h3>🎭 Mode Shape</h3>\n"
        md += f"<p style='margin: 4px 0;'>{_mode_shape_phrase(fr)}</p>\n"
        md += "</div>\n\n"

        md += "</div>"
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
        md += f"<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 14px; border-radius: 8px;'>\n"
        md += f"<p style='margin: 0; font-size: 14px; opacity: 0.9;'>Flow regime</p>\n"
        md += f"<p style='margin: 4px 0 0 0; font-size: 20px; font-weight: bold;'>{regime_emoji} {_REGIME_PHRASE[regime].title()}</p>\n"
        md += "</div>\n\n"

        md += f"<div style='background: #f8f9fa; padding: 12px; border-radius: 8px;'>\n"
        md += f"<table style='width: 100%;'>\n"
        md += f"<tr><td style='padding: 6px; color: #666;'>Lock-in</td><td style='padding: 6px; text-align: right; font-weight: bold;'>{_LOCKIN_PHRASE[lock]}</td></tr>\n"
        md += f"<tr><td style='padding: 6px; color: #666;'>Amplitude</td><td style='padding: 6px; text-align: right; font-weight: bold;'>{trend.value} ({s.rms_amplitude_over_c:.3f}c)</td></tr>\n"
        md += f"</table>\n"
        md += "</div>\n\n"

        md += f"<div style='background: #fff3cd; padding: 10px; border-radius: 6px; border-left: 3px solid #ffc107;'>\n"
        md += f"<p style='margin: 0; font-size: 13px;'><strong>Key insight:</strong> {_MECHANISM[regime]}</p>\n"
        md += "</div>\n\n"

        md += "</div>"
        return md

    def _structured_breakdown(fr: FeatureRecord) -> Tuple[str, str, str]:
        """Return (summary, warnings, full_details) for Tab 2."""
        warns = fr.range_warnings()
        md = ""
        md += "### Complete Feature Record\n\n"
        md += _build_metrics_table(fr)
        md += "\n\n"
        md += "### Interpretation\n\n"
        md += _build_interpretation_cards(fr)
        md += "\n\n"
        if warns:
            md += "---\n\n### Data Quality Warnings\n\n"
            for w in warns:
                md += f"- {w}\n"
        return _build_summary_card(fr), (
            "### Warnings\n\n" + "\n".join(f"- {w}" for w in warns) if warns else ""
        ), md

    def interpret(fr_dict, use_rag):
        if not fr_dict:
            return "Provide FSI data first (tab 1), then interpret.", "", ""
        fr = FeatureRecord.model_validate(fr_dict)
        summary, warnings, details = _structured_breakdown(fr)

        if not _has_trained_model():
            raw_text = _describe_fn(fr)
            full_md = details + "\n\n---\n\n### Raw Rule-Based Output\n\n" + raw_text.replace("\n", "  \n")
            return summary, warnings, full_md

        runner = _get_runner()
        retrieved = runner.retrieve(fr) if use_rag else []
        raw_text = runner.generate(fr, use_rag=bool(use_rag and retrieved))
        full_md = details + "\n\n---\n\n### Model Output\n\n" + raw_text.replace("\n", "  \n")
        return summary, warnings, full_md

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

    # ------------------------------------------------------------------ #
    # Layout
    # ------------------------------------------------------------------ #
    with gr.Blocks(title="FSI-SLM Interpreter") as demo:
        gr.Markdown(
            "# FSI-SLM Interpreter\n"
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
                    ["Pressure: 101325, Strain: 0.01"],
                    ["Pressure: 50000, Strain: 0.005"],
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
            interpret_btn = gr.Button("Interpret record", variant="primary")
            with gr.Row():
                with gr.Column(scale=1):
                    summary_out = gr.Markdown("### Quick Summary")
                with gr.Column(scale=2):
                    warnings_out = gr.Markdown("")
            with gr.Row():
                metrics_out = gr.Markdown("### Complete Feature Record")
            with gr.Row():
                interp_out = gr.Markdown(label="Structured interpretation")
            interp_passages = gr.Markdown()

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
            outputs=[summary_out, warnings_out, interp_out, interp_passages],
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
