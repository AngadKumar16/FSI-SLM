"""FastAPI inference server for the FSI interpreter.

Endpoints
---------
- ``GET  /health``        -- health check (model state, RAG state, load time).
- ``GET  /schema``        -- FeatureRecord JSON schema.
- ``POST /infer``         -- run inference on a FeatureRecord (single response).
- ``POST /infer/stream``  -- Server-Sent Events streaming inference.
- ``GET  /ready``         -- 200 once the model is loaded, 503 otherwise.

The inference backend is a single :class:`inference.runner.InferenceRunner`
shared by ``cli.py:infer`` and ``cli.py:eval``. The server can run in three
modes:

- **Trained model + adapter** (default when ``artifacts/adapters/adapters.safetensors``
  exists): load Phi-4-mini + LoRA, generate with mlx_lm.
- **Rule-based fallback**: when no adapter is found, falls back to
  :func:`data.describe.describe`.
- **Injected model**: pass a pre-loaded ``(model, tokenizer)`` pair to
  ``create_app(model=..., tokenizer=...)`` -- mostly for tests.

Usage
-----

::

    # Start the server (uses artifacts/adapters/ + artifacts/rag_index/)
    python cli.py serve --port 8000

    # Or run directly
    python -m serve.fastapi_app --port 8000

    # Then send requests
    curl -X POST http://localhost:8000/infer \\
        -H "Content-Type: application/json" \\
        -d '{"feature_record": {...}, "use_rag": false}'

    # Stream tokens
    curl -N -X POST http://localhost:8000/infer/stream \\
        -H "Content-Type: application/json" \\
        -d '{"feature_record": {...}, "use_rag": false}'
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from Data.schema import FeatureRecord

# ----------------------------------------------------------------- #
# Logging
# ----------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fsi.server")


# --------------------------------------------------------------------------- #
# Request / Response schemas
# --------------------------------------------------------------------------- #
class InferRequest(BaseModel):
    """Inference request body."""
    feature_record: dict = Field(..., description="A FeatureRecord-compatible JSON object")
    use_rag: bool = Field(default=False, description="Whether to use RAG augmentation")
    max_retrieved: int = Field(default=3, ge=1, le=20, description="Max RAG passages to retrieve")
    max_tokens: Optional[int] = Field(default=None, ge=1, le=2048, description="Override max tokens for this request")


class PassageInfo(BaseModel):
    """A single retrieved reference passage."""
    text: str = Field(..., description="Passage text")
    source: str = Field(..., description="Source document filename")
    score: float = Field(..., description="Relevance score")
    page: int = Field(default=0, description="Page number in source")


class InferResponse(BaseModel):
    """Inference response."""
    experiment_id: str = Field(..., description="Experiment identifier")
    interpretation: str = Field(..., description="Structured interpretation text")
    regime: Optional[str] = Field(default=None, description="Classified flow regime")
    used_rag: bool = Field(default=False, description="Whether RAG was used")
    retrieved_passages: List[PassageInfo] = Field(default_factory=list, description="Retrieved reference passages")
    model: str = Field(default="rule-based", description="Model used for inference")
    latency_ms: float = Field(default=0.0, description="End-to-end request latency in milliseconds")


# --------------------------------------------------------------------------- #
# App factory
# --------------------------------------------------------------------------- #
def create_app(
    config: Optional[dict] = None,
    runner: Optional["InferenceRunner"] = None,  # type: ignore[name-defined]
    eager_load: bool = False,
) -> FastAPI:
    """Create the FastAPI application.

    Parameters
    ----------
    config : dict, optional
        Application configuration (from YAML). Used for RAG settings and to
        build an InferenceRunner if none is injected.
    runner : InferenceRunner, optional
        Pre-built runner. If None, one is built lazily from ``config`` on
        first ``/infer`` call. Tests can inject a runner to avoid touching
        the model files.
    eager_load : bool
        If True, load the model + adapter at app startup (so first /infer
        is fast). If False (default), the model loads on the first /infer
        call (~30s warmup, then ~3s per call).
    """
    app = FastAPI(
        title="FSI-SLM Inference Server",
        description="Fluid-Structure Interaction interpretation via SLM",
        version="0.3.0",
    )

    _config = config or {}
    _runner: Any = runner  # built lazily on first /infer if None
    _load_started_at: Optional[float] = None
    _load_completed_at: Optional[float] = None

    def _adapter_path() -> str:
        return _config.get("inference", {}).get("adapter_path", "artifacts/adapters")

    def _rag_index_path() -> str:
        return _config.get("inference", {}).get(
            "rag_index_dir",
            _config.get("rag", {}).get("index_dir", "artifacts/rag_index"),
        )

    def _has_trained_model() -> bool:
        adapter = _adapter_path()
        return bool(adapter and Path(adapter, "adapters.safetensors").exists())

    def _get_runner() -> "InferenceRunner":  # type: ignore[name-defined]
        nonlocal _runner, _load_started_at, _load_completed_at
        if _runner is None:
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
                temperature=cfg.get("temperature", 0.0),
                top_p=cfg.get("top_p", 1.0),
                top_k=cfg.get("rag_top_k", rag_cfg.get("top_k", 3)),
                score_threshold=cfg.get(
                    "rag_score_threshold", rag_cfg.get("score_threshold", 0.3)
                ),
            )
            logger.info(f"Loading model: {inf_cfg.model_name} + adapter {inf_cfg.adapter_path}")
            _load_started_at = time.time()
            _runner = InferenceRunner(inf_cfg)
            _load_completed_at = time.time()
            logger.info(
                f"Model loaded in {_load_completed_at - _load_started_at:.1f}s"
            )
        return _runner

    @app.on_event("startup")
    async def _startup() -> None:
        if eager_load:
            logger.info("Eager-loading model at startup")
            _get_runner()
        logger.info(f"Server ready: model={'trained' if _has_trained_model() else 'rule-based'}, "
                    f"rag={Path(_rag_index_path(), 'index.faiss').exists()}")

    # ------------------------------------------------------------------ #
    # Health / readiness
    # ------------------------------------------------------------------ #
    @app.get("/health")
    async def health() -> Dict[str, Any]:
        """Health check with model load timing."""
        result: Dict[str, Any] = {
            "status": "ok",
            "model": "trained" if _has_trained_model() else "rule-based",
            "model_loaded": _runner is not None,
            "rag_index": Path(_rag_index_path(), "index.faiss").exists(),
        }
        if _load_completed_at is not None and _load_started_at is not None:
            result["load_completed_at"] = _load_completed_at
            result["load_duration_s"] = _load_completed_at - _load_started_at
        return result

    @app.get("/ready")
    async def ready() -> Dict[str, Any]:
        """Readiness probe: 200 only when model is loaded."""
        if _runner is not None:
            return {"ready": True}
        raise HTTPException(status_code=503, detail="Model not yet loaded")

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    @app.post("/infer", response_model=InferResponse)
    async def infer(request: InferRequest) -> InferResponse:
        """Run inference on a FeatureRecord and return structured interpretation."""
        t0 = time.time()
        log_extra = {"experiment_id": "?", "use_rag": request.use_rag}
        try:
            fr = FeatureRecord.model_validate(request.feature_record)
            log_extra["experiment_id"] = fr.experiment_id
        except Exception as e:
            logger.exception(f"Invalid FeatureRecord: {e}")
            raise HTTPException(status_code=422, detail=f"Invalid FeatureRecord: {e}")

        retrieved_passages: List[dict] = []

        if _has_trained_model():
            runner = _get_runner()
            if request.use_rag:
                try:
                    retrieved = runner.retrieve(fr, top_k=request.max_retrieved)
                    retrieved_passages = [
                        {
                            "text": r.passage_text,
                            "source": r.source,
                            "score": r.score,
                            "page": r.page,
                        }
                        for r in retrieved
                    ]
                except Exception as e:
                    logger.warning(f"RAG retrieval failed: {e}")
                    retrieved = []
            else:
                retrieved = []

            try:
                interpretation = runner.generate(
                    fr,
                    use_rag=bool(request.use_rag and retrieved),
                    max_tokens=request.max_tokens,
                )
            except Exception as e:
                logger.exception(f"Model inference failed, falling back: {e}")
                from Data.describe import describe
                interpretation = describe(fr)
            model_label = "trained"
        else:
            from Data.describe import describe
            interpretation = describe(fr)
            model_label = "rule-based"

        from eval.validate_output import parse_structured_fields
        parsed = parse_structured_fields(interpretation)
        regime = parsed.get("regime")

        latency_ms = (time.time() - t0) * 1000.0
        logger.info(
            f"infer exp={fr.experiment_id} rag={request.use_rag} "
            f"model={model_label} regime={regime} latency_ms={latency_ms:.0f}"
        )

        return InferResponse(
            experiment_id=fr.experiment_id,
            interpretation=interpretation,
            regime=regime,
            used_rag=request.use_rag and bool(retrieved_passages),
            retrieved_passages=[PassageInfo(**p) for p in retrieved_passages],
            model=model_label,
            latency_ms=latency_ms,
        )

    @app.post("/infer/stream")
    async def infer_stream(request: InferRequest):
        """Stream tokens via Server-Sent Events.

        Each event has the form ``data: <json>\\n\\n``. The JSON has
        ``text`` (the new token text) and ``done`` (bool). The last
        event has ``done: true`` and includes ``interpretation`` (full
        text) plus ``regime`` and ``used_rag``.
        """
        try:
            fr = FeatureRecord.model_validate(request.feature_record)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Invalid FeatureRecord: {e}")

        if not _has_trained_model():
            raise HTTPException(status_code=503, detail="Streaming requires a trained model")

        runner = _get_runner()
        retrieved: List = []
        if request.use_rag:
            try:
                retrieved = runner.retrieve(fr, top_k=request.max_retrieved)
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")

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
                from eval.validate_output import parse_structured_fields
                parsed = parse_structured_fields(full)
                payload = {
                    "text": "",
                    "done": True,
                    "interpretation": full,
                    "regime": parsed.get("regime"),
                    "used_rag": request.use_rag and bool(retrieved),
                }
                yield f"data: {json.dumps(payload)}\n\n"
            except Exception as e:
                logger.exception(f"Streaming inference failed: {e}")
                yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    @app.get("/schema")
    async def get_schema() -> dict:
        """Return the FeatureRecord JSON schema for API consumers."""
        return FeatureRecord.model_json_schema()

    return app


# --------------------------------------------------------------------------- #
# Direct execution
# --------------------------------------------------------------------------- #
def main(host: str = "0.0.0.0", port: int = 8000, config_path: str = "configs/default.yaml",
         eager_load: bool = False) -> None:
    """Start the server directly.

    Parameters
    ----------
    host : str
        Server host address.
    port : int
        Server port.
    config_path : str
        Path to YAML configuration file.
    eager_load : bool
        If True, load model + adapter at startup (slow start, fast first request).
    """
    import yaml
    import uvicorn

    with open(config_path) as f:
        config = yaml.safe_load(f)

    app = create_app(config=config, eager_load=eager_load)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FSI-SLM Inference Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--eager-load", action="store_true",
                        help="Load model + adapter at startup (slower boot, faster first request)")
    args = parser.parse_args()
    main(host=args.host, port=args.port, config_path=args.config, eager_load=args.eager_load)
