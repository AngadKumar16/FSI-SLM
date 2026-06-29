"""FastAPI inference server for the FSI interpreter.

Provides a ``/infer`` endpoint that:
1. Accepts a ``FeatureRecord`` JSON and optional RAG toggle
2. Runs inference (currently rule-based fallback; in production uses the trained model)
3. Returns validated structured interpretation with source citations

Usage::

    python -m serve.fastapi_app
    # or via CLI: python cli.py serve
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
    model=None,
    tokenizer=None,
) -> FastAPI:
    """Create the FastAPI application.

    Parameters
    ----------
    config : dict, optional
        Application configuration (from YAML).
    model : optional
        Loaded MLX model for inference. If None, uses rule-based fallback.
    tokenizer : optional
        Tokenizer for the model.

    Returns
    -------
    FastAPI
        Configured FastAPI app.
    """
    app = FastAPI(
        title="FSI-SLM Inference Server",
        description="Fluid-Structure Interaction interpretation via SLM",
        version="0.1.0",
    )

    _config = config or {}
    _model = model
    _tokenizer = tokenizer

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "ok", "model": "rule-based" if _model is None else "trained"}

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

        # Build prompt
        from train.tokenization.prompt_template import apply_chat_template, SYSTEM_PROMPT

        retrieved_passages: List[dict] = []

        if request.use_rag:
            try:
                from rag.index import load_index
                from rag.retrieve import build_query, retrieve
                from rag.prompt_assembly import assemble_rag_prompt

                rag_cfg = _config.get("rag", {})
                index_dir = rag_cfg.get("index_dir", "artifacts/rag_index")
                index, metadata, emb_model = load_index(index_dir)
                query = build_query(fr)
                results = retrieve(
                    query=query,
                    index=index,
                    metadata=metadata,
                    model=emb_model,
                    top_k=request.max_retrieved,
                    score_threshold=rag_cfg.get("score_threshold", 0.3),
                )
                prompt = assemble_rag_prompt(fr, results)
                retrieved_passages = [
                    {
                        "text": r.passage_text,
                        "source": r.source,
                        "score": r.score,
                        "page": r.page,
                    }
                    for r in results
                ]
            except (FileNotFoundError, ImportError) as e:
                logger.warning(f"RAG unavailable: {e}")
                prompt = apply_chat_template(SYSTEM_PROMPT, fr.to_prompt_block(), add_generation_prompt=True)
        else:
            prompt = apply_chat_template(SYSTEM_PROMPT, fr.to_prompt_block(), add_generation_prompt=True)

        # Generate interpretation
        if _model is not None:
            # Production path: use trained model
            import mlx.core as mx

            # Tokenize prompt and generate
            prompt_tokens = _tokenizer.encode(prompt)
            # TODO: implement proper generation loop with configurable max_tokens, temperature, etc.
            # For now, use rule-based fallback
            raise NotImplementedError("Trained model inference not yet implemented in this endpoint")
        else:
            # Rule-based fallback
            from data.describe import describe
            interpretation = describe(fr)

        # Extract regime from interpretation
        regime = None
        from eval.validate_output import parse_structured_fields
        parsed = parse_structured_fields(interpretation)
        regime = parsed.get("regime")

        return InferResponse(
            experiment_id=fr.experiment_id,
            interpretation=interpretation,
            regime=regime,
            used_rag=request.use_rag,
            retrieved_passages=[
                PassageInfo(**p) for p in retrieved_passages
            ],
            model="rule-based" if _model is None else "trained",
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