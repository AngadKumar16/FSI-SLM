import logging
from fastapi.responses import StreamingResponse
import json

logger = logging.getLogger("FSI-Streaming")
logging.basicConfig(level=logging.INFO)

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

        # Validate and handle missing fields
        if not pieces:
            payload = {
                "text": "",
                "done": True,
                "error": "No valid tokens produced. Ensure pressure/strain are provided."
            }
            yield f"data: {json.dumps(payload)}\n\n"

        full = "".join(pieces)
        from eval.validate_output import parse_structured_fields
        parsed = parse_structured_fields(full)

        payload = {
            "text": "",
            "done": True,
            "interpretation": full,
            "regime": parsed.get("regime"),
            "used_rag": request.use_rag and bool(retrieved),
            "note": "Missing fields were filled with default values, where applicable."
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