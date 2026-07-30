"""FastAPI inference server for the FSI interpreter.

Exposes the trained LoRA adapter (or rule-based fallback) over HTTP. Endpoints:

* ``POST /infer``        -- run one structured interpretation, return validated
  JSON (interpretation text + parsed regime + source citations).
* ``POST /infer/stream`` -- Server-Sent Events token stream of the same
  interpretation, terminated by a ``done`` event carrying the parsed fields.
* ``GET  /schema``       -- FeatureRecord JSON schema for API consumers.
* ``GET  /health``       -- liveness probe.

Build via :func:`create_app`; launch with ``python cli.py serve``. The heavy
InferenceRunner (model + adapter + RAG index) is loaded lazily on the first
request unless ``eager_load=True`` is passed to :func:`create_app`.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from Data.schema import FeatureRecord
from eval.validate_output import parse_structured_fields, validate_output

logger = logging.getLogger("FSI-Serving")
logging.basicConfig(level=logging.INFO)


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #
class InferRequest(BaseModel):
    """Body for /infer and /infer/stream."""

    feature_record: Dict[str, Any] = Field(
        ..., description="A FeatureRecord as JSON (see GET /schema)."
    )
    use_rag: bool = Field(
        False, description="Retrieve literature passages to ground the output."
    )
    max_tokens: Optional[int] = Field(
        None, description="Override generation length (defaults to config)."
    )
    max_retrieved: Optional[int] = Field(
        None, description="Override RAG top_k for this request."
    )


class SensorPredictRequest(BaseModel):
    """Body for /sensor/predict."""

    airfoil: str = Field(..., description="Path to an airfoil .dat file.")
    alpha_deg: float = Field(..., description="Angle of attack in degrees.")
    reynolds: float = Field(5.0e5, description="Chord Reynolds number.")
    return_fields: bool = Field(
        False, description="Include the predicted 2-D fields (large)."
    )
    field_downsample: int = Field(
        4, description="Average-pool factor applied to returned fields."
    )


class SensorPolarRequest(BaseModel):
    """Body for /sensor/polar."""

    airfoil: str
    alphas: List[float] = Field(
        default_factory=lambda: [-4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16]
    )
    reynolds: float = 5.0e5
    use_solver: bool = Field(
        False, description="Bypass the network and run the panel solver."
    )


class TwinRequest(BaseModel):
    """Body for /twin."""

    airfoil: str
    alpha_deg: float
    chord_m: float = 0.1
    span_m: float = 0.3
    u_inf_m_s: float = 5.0
    use_surrogate: bool = Field(
        True, description="Use the trained U-Net; false runs the solver."
    )
    interpret: bool = Field(
        False, description="Also run the language model on the derived record."
    )
    use_rag: bool = False
    inject_faults: Optional[List[str]] = Field(
        None,
        description=(
            "Deliberately corrupt the synthesised signals to exercise the "
            "noise detector: clipping, dropout, mains, drift, spike, flatline."
        ),
    )


def _citations(retrieved: List) -> List[Dict[str, Any]]:
    """Serialise RetrievalResult objects into JSON-friendly citation dicts."""
    return [
        {
            "source": r.source,
            "page": r.page,
            "score": round(float(r.score), 4),
            "passage": r.passage_text.strip(),
        }
        for r in retrieved
    ]


# --------------------------------------------------------------------------- #
# App factory
# --------------------------------------------------------------------------- #
def create_app(
    config: Optional[dict] = None, eager_load: bool = False
) -> FastAPI:
    """Build the FastAPI app wired to a lazily-loaded InferenceRunner."""
    _config = config or {}
    app = FastAPI(title="FSI-SLM Inference API", version="1.0.0")

    # Lazily-built shared runner (heavy model load deferred to first use).
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

    if eager_load:
        _get_runner()

    def _build_record(request: InferRequest) -> FeatureRecord:
        try:
            return FeatureRecord.model_validate(request.feature_record)
        except Exception as e:  # noqa: BLE001 - surface validation errors as 422
            raise HTTPException(status_code=422, detail=f"Invalid FeatureRecord: {e}")

    def _do_retrieve(runner, fr: FeatureRecord, request: InferRequest) -> List:
        if not request.use_rag:
            return []
        try:
            return runner.retrieve(fr, top_k=request.max_retrieved)
        except Exception as e:  # noqa: BLE001 - RAG is optional; degrade gracefully
            logger.warning(f"RAG retrieval failed: {e}")
            return []

    # ------------------------------------------------------------------ #
    # Endpoints
    # ------------------------------------------------------------------ #
    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "model_loaded": _runner_holder["runner"] is not None}

    @app.get("/schema")
    async def get_schema() -> dict:
        """Return the FeatureRecord JSON schema for API consumers."""
        return FeatureRecord.model_json_schema()

    @app.post("/infer")
    async def infer(request: InferRequest) -> dict:
        """Run one structured interpretation and return validated JSON."""
        fr = _build_record(request)
        runner = _get_runner()
        retrieved = _do_retrieve(runner, fr, request)
        try:
            text = runner.generate(
                fr,
                retrieved=retrieved,
                max_tokens=request.max_tokens,
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("Inference failed")
            raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

        from features.extract import check_physical_consistency

        result = validate_output(text)
        return {
            "interpretation": text,
            "regime": result.parsed_fields.get("regime"),
            "valid": result.valid,
            "errors": result.errors,
            "warnings": result.warnings,
            "record_consistency_warnings": check_physical_consistency(fr),
            "used_rag": bool(request.use_rag and retrieved),
            "citations": _citations(retrieved),
        }

    # ------------------------------------------------------------------ #
    # Surrogate ("sensor") endpoints -- geometry in, physics out, no LLM
    # ------------------------------------------------------------------ #
    @app.get("/sensor/status")
    async def sensor_status() -> dict:
        """Whether the dataset has been generated and the surrogate trained."""
        from sensor.api import checkpoint_info, dataset_summary

        return {
            "dataset": dataset_summary(),
            "surrogate": checkpoint_info(),
        }

    @app.post("/sensor/predict")
    async def sensor_predict(request: SensorPredictRequest) -> dict:
        """Fields and force coefficients for one (shape, incidence, Re) case."""
        from sensor.api import predict_case

        try:
            return predict_case(
                request.airfoil, request.alpha_deg, request.reynolds,
                return_fields=request.return_fields,
                field_downsample=request.field_downsample,
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:  # noqa: BLE001
            logger.exception("Surrogate prediction failed")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/sensor/polar")
    async def sensor_polar(request: SensorPolarRequest) -> dict:
        """Cl/Cd polar over a range of incidences."""
        from sensor.api import sweep_alpha

        try:
            return sweep_alpha(
                request.airfoil, request.alphas, request.reynolds,
                use_solver=request.use_solver,
            )
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/twin")
    async def twin(request: TwinRequest) -> dict:
        """Geometry + incidence -> FeatureRecord + data-quality report.

        With ``interpret=true`` the record and its ``[data_quality]`` block are
        passed to the language model, so this single call spans the whole
        digital twin: shape in, interpreted experiment out.
        """
        from features.noise import NoiseFlag, NoiseReport
        from sensor.api import twin_record

        try:
            out = twin_record(
                request.airfoil, request.alpha_deg,
                chord_m=request.chord_m, span_m=request.span_m,
                u_inf_m_s=request.u_inf_m_s,
                use_surrogate=request.use_surrogate,
                inject_faults=tuple(request.inject_faults or ()),
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("Twin construction failed")
            raise HTTPException(status_code=500, detail=str(e))

        if not request.interpret:
            return out

        fr = FeatureRecord.model_validate(out["feature_record"])
        report = NoiseReport(
            flags=[NoiseFlag(**f) for f in out["noise_report"]["flags"]],
            channels_checked=out["noise_report"]["channels_checked"],
            quality_score=out["noise_report"]["quality_score"],
            verdict=out["noise_report"]["verdict"],
            summary=out["noise_report"]["summary"],
        )
        runner = _get_runner()
        retrieved = runner.retrieve(fr) if request.use_rag else None
        prompt = runner.build_prompt(fr, retrieved, noise_report=report)
        try:
            text = runner.generate(prompt)
        except Exception as e:  # noqa: BLE001
            logger.exception("Inference failed")
            raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
        result = validate_output(text)
        out["interpretation"] = text
        out["regime_model"] = result.parsed_fields.get("regime")
        out["valid"] = result.valid
        out["citations"] = _citations(retrieved or [])
        return out

    @app.post("/infer/stream")
    async def infer_stream(request: InferRequest) -> StreamingResponse:
        """Stream the interpretation token-by-token over SSE."""
        fr = _build_record(request)
        runner = _get_runner()
        retrieved = _do_retrieve(runner, fr, request)

        def event_gen():
            pieces: List[str] = []
            try:
                for response in runner.stream(
                    fr,
                    retrieved=retrieved,
                    max_tokens=request.max_tokens,
                ):
                    text_piece = getattr(response, "text", "") or ""
                    pieces.append(text_piece)
                    yield f"data: {json.dumps({'text': text_piece, 'done': False})}\n\n"

                full = "".join(pieces)
                if not full:
                    yield "data: " + json.dumps(
                        {
                            "text": "",
                            "done": True,
                            "error": "No tokens produced.",
                        }
                    ) + "\n\n"
                    return

                parsed = parse_structured_fields(full)
                payload = {
                    "text": "",
                    "done": True,
                    "interpretation": full,
                    "regime": parsed.get("regime"),
                    "used_rag": bool(request.use_rag and retrieved),
                    "citations": _citations(retrieved),
                }
                yield f"data: {json.dumps(payload)}\n\n"

            except Exception as e:  # noqa: BLE001
                logger.exception("Streaming inference failed")
                yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    return app
