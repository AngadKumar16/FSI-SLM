# Training provenance

Record of what the FSI-SLM retrieval index and fine-tune were built from. The
artifacts themselves (`artifacts/`) are too large to track; their SHA-256 hashes
are recorded here so a copy can be checked against this commit.

| File | What it records |
|---|---|
| `corpus.json` | Every PDF in `Data/Papers/`: DOI, licence, SHA-256, size, chunk count in the index |
| `artifacts.json` | SHA-256, size and build time of the FAISS index, its chunk metadata, the LoRA adapter and the SFT train/val/test splits; embedding model; chunks per source paper; SHA-256 of the merged model and adapter published on Hugging Face |
| `adapter_config.json` | Exact LoRA config used for the fine-tune (base model `microsoft/Phi-4-mini-instruct`, rank 8, 330 iters) |
| `eval_results.json`, `eval_results.md` | Evaluation of the fine-tuned model |

Build dates (from the artifacts' modification times): index and SFT splits
2026-07-08 00:14–00:16 UTC, adapter 2026-07-08 04:55 UTC.

The index's chunk metadata names exactly the 16 files in `Data/Papers/` as its
sources (1,840 chunks) — nothing else.

Check a local copy:

```bash
shasum -a 256 artifacts/rag_index/index.faiss artifacts/adapters/adapters.safetensors
python - <<'PY'
import json, collections
m = json.load(open("artifacts/rag_index/metadata.json"))
print(len(m), sorted(collections.Counter(c["source"] for c in m).items()))
PY
```
