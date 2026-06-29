"""FastAPI inference server for the FSI interpreter.

Endpoints
---------
- ``GET  /health``  -- health check.
- ``GET  /schema``  -- FeatureRecord JSON schema.
- ``POST /infer``   -- run inference on a FeatureRecord, optionally with RAG.

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
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from Data.schema import FeatureRecord

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Request / Response schemas
# --------------------------------------------------------------------------- #
class InferRequest(BaseModel):
    """Inference request body."""
    feature_record: dict = Field(..., description="A FeatureRecord-compatible JSON object")
    use_rag: bool = Field(default=False, description="Whether to use RAG augmentation")
    max_retrieved: int = Field(default=3, ge=1, le=20, description="Max RAG passages to retrieve")


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


# --------------------------------------------------------------------------- #
# App factory
# --------------------------------------------------------------------------- #
def create_app(
    config: Optional[dict] = None,
    runner: Optional["InferenceRunner"] = None,  # type: ignore[name-defined]
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

    Returns
    -------
    FastAPI
        Configured FastAPI app.
    """
    app = FastAPI(
        title="FSI-SLM Inference Server",
        description="Fluid-Structure Interaction interpretation via SLM",
        version="0.2.0",
    )

    _config = config or {}
    _runner = runner  # built lazily on first /infer if None

    def _get_runner() -> "InferenceRunner":  # type: ignore[name-defined]
        nonlocal _runner
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
                    "rag_index_dir",
                    rag_cfg.get("index_dir", "artifacts/rag_index"),
                ),
                max_tokens=cfg.get("max_tokens", 350),
                temperature=cfg.get("temperature", 0.0),
                top_p=cfg.get("top_p", 1.0),
                top_k=cfg.get("rag_top_k", rag_cfg.get("top_k", 3)),
                score_threshold=cfg.get(
                    "rag_score_threshold", rag_cfg.get("score_threshold", 0.3)
                ),
            )
            _runner = InferenceRunner(inf_cfg)
        return _runner

    def _has_trained_model() -> bool:
        cfg = _config.get("inference", {})
        adapter = cfg.get("adapter_path")
        return bool(adapter and Path(adapter, "adapters.safetensors").exists())

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {
            "status": "ok",
            "model": "trained" if _has_trained_model() else "rule-based",
            "rag_index": Path(
                _config.get("inference", {}).get(
                    "rag_index_dir", "artifacts/rag_index"
                ),
                "index.faiss",
            ).exists(),
        }

    @app.post("/infer", response_model=InferResponse)
    async def infer(request: InferRequest):
        """Run inference on a FeatureRecord and return structured interpretation.

        Parameters
        ----------
        request : InferRequest
            The inference request body.

        Returns
        -------
        InferResponse
            The structured interpretation with optional RAG citations.
        """
        try:
            fr = FeatureRecord.model_validate(request.feature_record)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Invalid FeatureRecord: {e}")

        retrieved_passages: List[dict] = []

        # Decide predictor
        if _has_trained_model():
            runner = _get_runner()
            if request.use_rag:
                try:
                    retrieved = runner.retrieve(
                        fr, top_k=request.max_retrieved
                    )
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
                    fr, use_rag=bool(request.use_rag and retrieved)
                )
            except Exception as e:
                logger.exception(f"Model inference failed, falling back: {e}")
                from Data.describe import describe
                interpretation = describe(fr)
            model_label = "trained"
        else:
            # Rule-based fallback
            from Data.describe import describe
            interpretation = describe(fr)
            model_label = "rule-based"

        # Extract regime from interpretation
        from eval.validate_output import parse_structured_fields
        parsed = parse_structured_fields(interpretation)
        regime = parsed.get("regime")

        return InferResponse(
            experiment_id=fr.experiment_id,
            interpretation=interpretation,
            regime=regime,
            used_rag=request.use_rag and bool(retrieved_passages),
            retrieved_passages=[
                PassageInfo(**p) for p in retrieved_passages
            ],
            model=model_label,
        )

    @app.get("/schema")
    async def get_schema():
        """Return the FeatureRecord JSON schema for API consumers."""
        return FeatureRecord.model_json_schema()

    return app


# --------------------------------------------------------------------------- #
# Direct execution
# --------------------------------------------------------------------------- #
def main(host: str = "0.0.0.0", port: int = 8000, config_path: str = "configs/default.yaml") -> None:
    """Start the server directly.

    Parameters
    ----------
    host : str
        Server host address.
    port : int
        Server port.
    config_path : str
        Path to YAML configuration file.
    """
    import yaml
    import uvicorn

    with open(config_path) as f:
        config = yaml.safe_load(f)

    app = create_app(config=config)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FSI-SLM Inference Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(host=args.host, port=args.port, config_path=args.config)