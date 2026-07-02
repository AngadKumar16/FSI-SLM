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
                use_rag=bool(request.use_rag and retrieved),
                max_tokens=request.max_tokens,
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("Inference failed")
            raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

        result = validate_output(text)
        return {
            "interpretation": text,
            "regime": result.parsed_fields.get("regime"),
            "valid": result.valid,
            "errors": result.errors,
            "warnings": result.warnings,
            "used_rag": bool(request.use_rag and retrieved),
            "citations": _citations(retrieved),
        }

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
                    use_rag=bool(request.use_rag and retrieved),
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
