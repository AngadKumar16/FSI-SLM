# Implementation Plan

[Overview]
Build the complete FSI-SLM pipeline: synthetic data generation → feature extraction → SFT dataset building → QLoRA fine-tuning of Phi-4-mini-instruct on Apple Silicon → RAG-augmented inference with FAISS over a paper corpus → evaluation and validation.

The existing codebase has a solid core (schema, synthetic generator, feature extraction, description generator, dataset builder) but is missing the training loop, RAG pipeline, evaluation tools, CLI, config system, tests, and documentation. This plan fills all gaps to deliver a runnable end-to-end system. The M4 Pro (Apple Silicon) target means we use MLX for training instead of bitsandbytes/TRL, and faiss-cpu for indexing. The plan prioritizes a minimal runnable milestone first (pipeline works end-to-end on synthetic data) then adds RAG, evaluation, and polish.

[Types]
No new Pydantic models or enums are needed; the existing schema.py types (RawExperiment, FeatureRecord, FlowRegime, LockInStatus, AmplitudeTrend, FlowConditions, StructuralSignal, ForceCoefficients, PIVDerived) are complete.

New data structures required:
- `rag/retrieve.py`: `RetrievalResult` dataclass with fields: `passage_text: str`, `score: float`, `source: str`, `page: int`, `chunk_id: int`
- `eval/validate_output.py`: `ValidationResult` dataclass with fields: `valid: bool`, `parsed_fields: dict`, `errors: list[str]`, `warnings: list[str]`
- `train/sft.py`: `TrainingConfig` dataclass mirroring the YAML config keys for MLX training
- `cli.py`: `CLIContext` dataclass holding parsed config, seed, and device info

[Files]
New files to create (18 files), existing files to modify (2 files).

**New files:**

1. `requirements.txt` — Pinned dependencies: torch, transformers, datasets, mlx, mlx-lm, faiss-cpu, sentence-transformers, pydantic, numpy, scipy, pyyaml, pdfminer.six, fastapi, uvicorn, pytest, fire
2. `configs/default.yaml` — Hyperparameters: model_name, seed, dataset paths, training (batch_size, lr, epochs, lora_rank, lora_alpha, lora_dropout, max_seq_length), RAG (top_k, index_type, embedding_model, chunk_size, chunk_overlap, score_threshold), eval (freq_tolerance, regime_f1_threshold)
3. `cli.py` — Unified CLI with subcommands: generate-data, build-dataset, index-corpus, train, eval, infer, serve. Uses `fire.Fire` for subcommand dispatch. Each subcommand loads config, sets seed, and calls the appropriate module.
4. `tools/utils/repro.py` — `set_seed(seed: int)` function that seeds numpy, random, torch, and sets deterministic flags. Called at CLI startup.
5. `rag/corpus_loader.py` — `load_corpus(directory: str) -> list[dict]`: walks Data/Papers/, extracts text from PDFs via pdfminer, chunks into passages (256 tokens, 50% overlap), returns list of dicts with keys: text, source, page, chunk_id. Includes fallback for non-PDF text files.
6. `rag/index.py` — `build_index(passages: list[dict], model_name: str, index_type: str) -> tuple[faiss.Index, list[dict]]`: embeds passages via sentence-transformers, builds FAISS index (Flat or IVF), saves index + metadata store (JSON) to disk. `load_index(index_dir: str) -> tuple[faiss.Index, list[dict], SentenceTransformer]`: loads persisted index and metadata.
7. `rag/retrieve.py` — `RetrievalResult` dataclass. `retrieve(query: str, index, metadata, model, top_k: int, score_threshold: float) -> list[RetrievalResult]`: builds query embedding from a query template combining FeatureRecord fields, searches FAISS index, returns top-k results with scores and metadata.
8. `rag/prompt_assembly.py` — `assemble_rag_prompt(feature_record: FeatureRecord, retrieved: list[RetrievalResult], system_prompt: str) -> str`: assembles the full prompt with system message, verbatim FeatureRecord block, delimited retrieved passages with source citations, and instruction block. Never allows passages to override FeatureRecord values.
9. `train/sft.py` — MLX-based LoRA training script. Loads Phi-4-mini-instruct via mlx-lm, applies LoRA adapters, trains on SFT dataset (JSONL format from build_dataset.py), saves adapter weights. Supports configurable lora_rank, lora_alpha, lora_dropout, batch_size, learning_rate, epochs, max_seq_length. Includes checkpointing and resume capability.
10. `train/tokenization/prompt_template.py` — Exact Phi-4-mini-instruct chat template implementation matching vendor format. Exports `apply_chat_template(messages: list[dict]) -> str` and `SYSTEM_PROMPT` constant (moved from build_dataset.py for shared access).
11. `eval/evaluate.py` — Runs model on held-out test set with/without RAG toggle. Extracts numeric claims from generated text via regex parsing. Compares to ground-truth FeatureRecord values within tolerances. Reports per-field accuracy, regime classification F1, lock-in precision/recall, and with-vs-without-RAG improvement. Outputs JSON results file.
12. `eval/validate_output.py` — Standalone CLI (no model deps). Accepts raw generated text via stdin or file. Parses formatted claims into structured fields. Checks numeric consistency (e.g., reported St vs computed from reported f_shed/U), lock-in logic vs claimed frequencies, and schema conformance. Exits non-zero if invalid. Prints machine-readable JSON summary. Uses `ValidationResult` dataclass.
13. `serve/fastapi_app.py` — FastAPI app with /infer endpoint. Loads merged model checkpoint or LoRA adapter + base model. Accepts FeatureRecord JSON, runs inference with optional RAG, returns validated structured interpretation plus source citations.
14. `tests/test_schema.py` — Unit tests: valid FeatureRecord passes validation, invalid units/ranges fail, range_warnings triggers correctly, lock_in_status logic, to_prompt_block format.
15. `tests/test_synthetic.py` — Unit tests: generate_experiment returns valid RawExperiment, generate_dataset produces correct counts per regime, all regimes produce physically plausible values.
16. `tests/test_features.py` — Unit tests: extract_features returns correct field count, dominant_frequency finds expected peaks, pod_energy_fractions returns plausible values.
17. `tests/test_describe.py` — Unit tests: describe returns deterministic output for canonical FeatureRecord, classify_regime matches ground_truth_regime, amplitude_trend returns correct enum.
18. `tests/test_validate_output.py` — Unit tests: validate_output accepts correct string, rejects malformed output, parses numeric fields correctly.
19. `docs/schema.md` — Expanded documentation of schema.py fields, mapping from PIV to FeatureRecord, units, valid ranges, and examples.

**Modified files:**

1. `Data/build_dataset.py` — Move SYSTEM_PROMPT and apply_phi4_chat_template to train/tokenization/prompt_template.py and import from there. Add `tokenizer` parameter passthrough for MLX tokenizer.
2. `README.md` — Complete rewrite with: project description, architecture overview, setup instructions, CLI usage examples, reproducibility notes, schema contract summary, how to add real corpus, training instructions, evaluation commands, and license.

**Files to delete/move:** None.

[Functions]
New functions to create (25+), existing functions to modify (2).

**New functions:**

- `tools/utils/repro.py`: `set_seed(seed: int) -> None`
- `rag/corpus_loader.py`: `load_corpus(directory: str) -> list[dict]`, `_extract_text_from_pdf(path: str) -> str`, `_chunk_text(text: str, chunk_size: int, overlap: int) -> list[dict]`
- `rag/index.py`: `build_index(passages: list[dict], model_name: str, index_type: str, output_dir: str) -> None`, `load_index(index_dir: str) -> tuple`
- `rag/retrieve.py`: `RetrievalResult` (dataclass), `retrieve(query: str, index, metadata, model, top_k: int, score_threshold: float) -> list[RetrievalResult]`, `build_query(feature_record: FeatureRecord) -> str`
- `rag/prompt_assembly.py`: `assemble_rag_prompt(fr: FeatureRecord, retrieved: list[RetrievalResult], system_prompt: str) -> str`
- `train/sft.py`: `train(config: dict) -> None`, `load_dataset(path: str, tokenizer) -> Dataset`, `save_adapter(adapter_path: str) -> None`, `merge_weights(base_path: str, adapter_path: str, output_path: str) -> None`
- `train/tokenization/prompt_template.py`: `apply_chat_template(messages: list[dict]) -> str`, `SYSTEM_PROMPT` (constant)
- `eval/evaluate.py`: `evaluate(model, test_dataset, config) -> dict`, `parse_numeric_claims(text: str) -> dict`, `compute_accuracy(predictions: list, ground_truth: list, tolerances: dict) -> dict`, `compute_confusion_matrix(predictions: list, ground_truth: list) -> dict`
- `eval/validate_output.py`: `validate_output(text: str) -> ValidationResult`, `parse_structured_fields(text: str) -> dict`, `check_numeric_consistency(fields: dict) -> list[str]`, `check_lock_in_logic(fields: dict) -> list[str]`, `main()` (CLI entry point)
- `serve/fastapi_app.py`: `create_app() -> FastAPI`, `/infer` endpoint handler
- `cli.py`: `main()`, `generate_data()`, `build_dataset()`, `index_corpus()`, `train()`, `eval()`, `infer()`, `serve()`

**Modified functions:**

- `Data/build_dataset.py/build_dataset()` — Accept optional `tokenizer` parameter; use `train/tokenization/prompt_template.apply_chat_template` instead of local `render_chat`.
- `Data/build_dataset.py/render_chat()` — Remove or redirect to `train/tokenization/prompt_template.apply_chat_template`.

[Classes]
No new classes required beyond the dataclasses listed in [Types]. The existing Pydantic models in schema.py are complete.

[Testing]
Unit tests for schema validation, synthetic generator, feature extraction, description generator, and output validator.

- `tests/test_schema.py`: 6+ test functions covering valid construction, invalid ranges, enum values, lock_in_status, to_prompt_block format, range_warnings
- `tests/test_synthetic.py`: 4+ test functions covering generate_experiment output validity, generate_dataset counts, per-regime physical plausibility, seed reproducibility
- `tests/test_features.py`: 4+ test functions covering extract_features field count, dominant_frequency peak detection, pod_energy_fractions, passthrough fields
- `tests/test_describe.py`: 4+ test functions covering deterministic output, classify_regime accuracy, amplitude_trend, mode_shape_phrase
- `tests/test_validate_output.py`: 4+ test functions covering valid input acceptance, malformed input rejection, numeric field parsing, lock-in logic detection

[Dependencies]
New packages to add to requirements.txt:

Core: torch, transformers, datasets, accelerate, scipy, pyyaml
Apple Silicon training: mlx, mlx-lm
RAG: faiss-cpu, sentence-transformers, pdfminer.six
Serving: fastapi, uvicorn
Testing: pytest
CLI: fire

All with pinned versions compatible with Python 3.11+ on macOS ARM64.

[Implementation Order]
Build in dependency order: infrastructure first, then RAG, then training, then evaluation, then tests and docs.

1. Create `requirements.txt`, `configs/default.yaml`, `tools/utils/repro.py` — foundational infrastructure
2. Create `train/tokenization/prompt_template.py` — shared prompt template (needed by build_dataset and RAG)
3. Modify `Data/build_dataset.py` — refactor to use shared prompt template
4. Create `rag/corpus_loader.py`, `rag/index.py`, `rag/retrieve.py`, `rag/prompt_assembly.py` — RAG pipeline
5. Create `train/sft.py` — MLX-based LoRA training
6. Create `eval/validate_output.py` — standalone output validator
7. Create `eval/evaluate.py` — evaluation harness
8. Create `cli.py` — unified CLI entry point
9. Create `serve/fastapi_app.py` — optional serving wrapper
10. Create `tests/test_schema.py`, `tests/test_synthetic.py`, `tests/test_features.py`, `tests/test_describe.py`, `tests/test_validate_output.py` — unit tests
11. Create `docs/schema.md` — schema documentation
12. Rewrite `README.md` — full project documentation