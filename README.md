# FSI-SLM: Fluid-Structure Interaction Interpretation via Small Language Models
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Apple Silicon](https://img.shields.io/badge/platform-Apple%20Silicon-orange)](https://www.apple.com/mac/)

Fine-tune **Phi-4-mini-instruct** (3.8B) to interpret fluid-structure interaction (FSI) experiments — classifying flow regimes, extracting vortex-shedding frequencies, detecting VIV lock-in, and more — all running on a single **Apple Silicon (M4 Pro)** MacBook.

The system generates synthetic training data with physically-consistent couplings (Strouhal number, lock-in, galloping, stall flutter), extracts features via signal processing (FFT, POD), builds an SFT dataset, trains with MLX-based LoRA, and optionally augments inference with RAG over a corpus of FSI papers indexed with FAISS.

There are two halves. The **interpreter** reads an experiment record and explains
it. The **surrogate** (`sensor/`) produces that record from geometry alone: a
U-Net trained on 1,926 solved cases predicts velocity, pressure, load and
displacement fields plus the force coefficients, and a bridge turns those steady
fields into a time-resolved `FeatureRecord` screened for instrumentation faults.
Together they close the loop from "NACA 0009 at 12 degrees" to an interpreted
experiment with no human step.

The surrogate is conditioned on an **inviscid physics prior** — the panel
solution on the same raster is an input channel, so the network learns the
viscous correction rather than the whole field — and on a **fidelity code**, so
one network can be asked for the panel, steady-RANS or URANS answer. Three
solvers write the same contract: a panel method everywhere, steady RANS below
stall, and unsteady RANS above it, where a steady solver has no fixed point to
converge to and the transient run also *measures* the vortex-shedding frequency
the lock-in classification depends on.

```bash
python cli.py twin --airfoil Data/Airfoils/4/n0009sm.dat.txt --alpha 12 --interpret
```

- **[docs/results.md](docs/results.md)** — every measured number, with the
  command that regenerates it
- **[docs/surrogate.md](docs/surrogate.md)** — what the surrogate computes, how it
  is validated, and what it is not
- **[docs/abstract_claims.md](docs/abstract_claims.md)** — every claim in the
  paper abstract mapped to the code that implements it, including the two that
  are not yet true

---

## Architecture

For a detailed end-to-end workflow diagram see:

- **[docs/workflow.md](docs/workflow.md)** — Mermaid diagrams (Renders in GitHub, GitLab, VSCode)
- **[docs/workflow.txt](docs/workflow.txt)** — ASCII for terminal viewing

```
┌─────────────────┐     ┌───────────────────┐     ┌─────────────────┐
│  RawExperiment   │────▶│  Feature Record    │────▶│  SFT Dataset     │
│  (synthetic or   │     │  (compact,         │     │  (prompt+target) │
│   real PIV data) │     │   model-facing)    │     │                  │
└─────────────────┘     └───────────────────┘     └────────┬─────────┘
                                                            │
                    ┌─────────────────────┐                  │
                    │  RAG (FAISS +       │                  ▼
                    │   sentence-transform)│          ┌─────────────────┐
                    │  over paper corpus   │          │  MLX LoRA        │
                    └──────────┬──────────┘          │  Training        │
                               │                     └────────┬─────────┘
                               │                              │
                    ┌──────────▼──────────┐                   │
                    │  Prompt Assembly    │◄──────────────────┘
                    │  (FeatureRecord     │
                    │   + retrieved       │
                    │    passages)        │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Inference / Serve   │
                    │  (rule-based or      │
                    │   trained model)     │
                    └─────────────────────┘
```

### Components

| Component | Description | Key Files |
|-----------|-------------|-----------|
| **Schema** | Pydantic models for RawExperiment and FeatureRecord with validated ranges and units | `Data/schema.py` |
| **Synthetic Generator** | Physically-informed experiment generator with 5 flow regimes | `Data/synthetic.py` |
| **Feature Extraction** | FFT-based frequency detection, POD via SVD, St calculation | `features/extract.py` |
| **Description Generator** | Deterministic structured interpretation text | `Data/describe.py` |
| **Dataset Builder** | Assembles SFT examples, splits by experiment ID | `Data/build_dataset.py` |
| **Prompt Template** | Phi-4-mini-instruct chat template (shared across pipeline) | `train/tokenization/prompt_template.py` |
| **RAG Pipeline** | PDF corpus loader, FAISS index, retriever, prompt assembly | `rag/` |
| **Training** | MLX-based LoRA fine-tuning with checkpointing | `train/sft.py` |
| **Validation** | Standalone output validator (regex parsing, physics checks) | `eval/validate_output.py` |
| **Evaluation** | Per-field accuracy, regime F1, lock-in metrics | `eval/evaluate.py` |
| **CLI** | Unified command-line interface via `fire.Fire` | `cli.py` |
| **Serve** | FastAPI inference server with RAG | `serve/fastapi_app.py` |
| **MCP Server** | Model Context Protocol server (tools for Claude Desktop/Code and other MCP clients) | `serve/mcp_server.py` |
| **Config** | YAML-based configuration with sensible defaults | `configs/default.yaml` |
| **Noise Detection** | Eleven instrumentation-fault detectors (PIV dropout and spurious vectors, ADC clipping and quantisation, spikes, drift, flatline, spectral SNR, aliasing, mains pickup) | `features/noise.py` |
| **Flow Solver** | Hess-Smith panel method + Beddoes-Leishman static stall + momentum-consistent wake | `sensor/flow/` |
| **Steady RANS** | gmsh + OpenFOAM `simpleFoam`, k-omega SST. Converges below stall | `sensor/cfd/` |
| **Unsteady RANS** | `pimpleFoam` + time-averaging + measured shedding frequency. The post-stall branch | `sensor/cfd/unsteady.py` |
| **Structural Solver** | Chordwise Euler-Bernoulli beam (load and displacement fields) + spanwise modal model | `sensor/structure/` |
| **Dataset Generator** | 107 sections x 18 angles = 1,926 cases, foil-disjoint splits | `sensor/generate.py` |
| **Input Contract** | Named, versioned channel set with mirror parities; inviscid prior + fidelity conditioning | `sensor/sdf.py` |
| **U-Net Surrogate** | Five field channels + five scalars; FiLM conditioning, bottleneck attention, physics-informed loss | `sensor/model.py`, `sensor/train.py` |
| **Symmetry** | Exact y-mirror as augmentation and test-time averaging, validated against the solver | `sensor/symmetry.py` |
| **Twin Bridge** | Steady surrogate fields -> time-resolved `FeatureRecord` + data-quality report | `sensor/bridge.py` |

---

## Setup

### Prerequisites

- Python 3.11+
- macOS with Apple Silicon (M1/M2/M3/M4)
- `pip` (Python package manager)

### Installation

```bash
# Clone the repository
git clone https://github.com/AngadKumar16/FSI-SLM.git
cd FSI-SLM

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import mlx.core; print('MLX:', mlx.core.__version__)"
python -c "import faiss; print('FAISS:', faiss.__version__)"
python -c "import torch; print('Torch:', torch.__version__)"
```

---

## Quick Start: End-to-End Pipeline

### 1. Generate synthetic data and build the dataset

```bash
# Generate 200 experiments (40 per regime), build SFT dataset
python cli.py generate-data --n-per-regime 40
python cli.py build-dataset --n-per-regime 40
```

This creates:
- `artifacts/dataset/train.jsonl` — 160 training examples
- `artifacts/dataset/val.jsonl` — 20 validation examples
- `artifacts/dataset/test.jsonl` — 20 test examples
- `artifacts/dataset/feature_records_{split}.jsonl` — raw feature records for evaluation

### 2. Index the paper corpus for RAG (optional)

```bash
python cli.py index-corpus
```

This extracts text from PDFs in `Data/Papers/`, embeds with `all-MiniLM-L6-v2`, builds a FAISS index, and saves to `artifacts/rag_index/`.

### 3. Train the model

```bash
python cli.py train --epochs 3 --batch-size 4
```

LoRA adapters are saved to `artifacts/adapters/`. Training can be resumed from checkpoints in `artifacts/checkpoints/`.

### 4. Evaluate

```bash
python cli.py eval
```

This runs the rule-based description generator as a baseline evaluator against the test set. Results are saved to `artifacts/eval_results/eval_results.json`.

### 5. Run inference

```bash
# Generate a test feature record and pass it for inference
python -c "
from data.synthetic import generate_experiment
from features.extract import extract_features
import numpy as np
rng = np.random.default_rng(42)
exp = generate_experiment('viv_lockin', rng, index=0)
fr = extract_features(exp)
import json
with open('/tmp/test_fr.json', 'w') as f:
    f.write(fr.model_dump_json())
"

# Run inference (rule-based fallback)
python cli.py infer --feature-record /tmp/test_fr.json

# With RAG
python cli.py infer --feature-record /tmp/test_fr.json --use-rag
```

### 6. Start the inference server

```bash
python cli.py serve --port 8000
```

Then send requests:

```bash
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{"feature_record": <YOUR_JSON>, "use_rag": false}'
```

### 7. Use it from Claude (MCP server)

The project ships an MCP (Model Context Protocol) server that exposes the
interpreter as tools for Claude Desktop, Claude Code, or any MCP client:

```bash
python cli.py mcp        # or: python -m serve.mcp_server
```

The repo's `.mcp.json` registers it automatically for Claude Code sessions
opened in this directory. For Claude Desktop, add to
`claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "fsi-slm": {
      "command": "python3",
      "args": ["-m", "serve.mcp_server"],
      "cwd": "/path/to/FSI-SLM"
    }
  }
}
```

**Tools:**

| Tool | What it does | Loads model? |
|------|--------------|--------------|
| `interpret_experiment` | Structured interpretation of a FeatureRecord (regime, lock-in, physics), optional RAG citations | Yes (first call, ~30s) |
| `ask_question` | Free-text Q&A grounded in a FeatureRecord + retrieved literature | Yes |
| `search_literature` | FAISS search over the indexed FSI paper corpus | No (embedder only) |
| `validate_interpretation` | Parse + physics-check any generated interpretation | No |
| `check_experiment_consistency` | Range + cross-field physics check of a FeatureRecord (units-bug detector) | No |
| `sweep_regime_map` | What-if sweep of U*/AoA/f_n → regime map + transition boundaries | No |
| `generate_synthetic_experiment` | Demo FeatureRecord for a chosen flow regime | No |

**Resources:** `fsi://schema/feature_record` — the FeatureRecord JSON schema.

### 8. Validate model output

```bash
# Pipe a model output through the validator
echo "Regime: VIV lock-in. The wake sheds at a dominant frequency of 10.00 Hz..." | \
  python -m eval.validate_output --pretty
```

---

## CLI Reference

All commands are accessible via the unified CLI:

```bash
python cli.py <command> [options]
```

| Command | Description |
|---------|-------------|
| `generate-data` | Generate synthetic RawExperiment data (JSONL) |
| `build-dataset` | Build SFT dataset from synthetic data |
| `index-corpus` | Build FAISS index over paper corpus |
| `train` | Run MLX-based LoRA training |
| `eval` | Evaluate the trained model |
| `infer` | Run inference on a single FeatureRecord |
| `serve` | Start the FastAPI inference server |
| `ui` | Launch the Gradio web UI |
| `mcp` | Start the MCP server (stdio) for Claude Desktop/Code |
| `demo` | Instant end-to-end terminal demo (no model): generate → extract → interpret → validate all 5 regimes; `--sweep` adds a U* regime map |
| `sensor-generate` | Solve the 107 × 18 = 1,926-case surrogate training set |
| `sensor-cache` | Flatten the generated `.npz` files into training memmaps (`--spec` picks the channel contract) |
| `sensor-rans` | Steady OpenFOAM sweep. Below stall only |
| `sensor-urans` | Unsteady OpenFOAM sweep: time-averaged fields + measured shedding frequency |
| `sensor-multifidelity` | Plan (and `--run`) high-fidelity coverage of *training* foils |
| `sensor-finalize` | Merge sweep chunks into the manifest, highest fidelity wins |
| `sensor-train` | Train the U-Net (fields + force coefficients) |
| `sensor-eval` | Score the surrogate on held-out sections |
| `sensor-predict` | Coefficients (and fields) for one section and incidence |
| `twin` | Geometry → fields → `FeatureRecord` → data-quality report → interpretation |

Each command accepts `--config-path` to specify a custom YAML configuration file (default: `configs/default.yaml`).

### The surrogate pipeline

```bash
python cli.py sensor-generate                  # 1,926 cases, ~2 min on 4 cores
python cli.py sensor-cache                     # 13-channel memmap cache
python cli.py sensor-train --epochs 80 --batch-size 16          # 128x128
python cli.py sensor-train --downsample 2 --base 16 --epochs 60 # fast 64x64
python cli.py sensor-eval --split test
python cli.py sensor-predict --airfoil Data/Airfoils/4/n0009sm.dat.txt --alpha 12
python -m sensor.figures --out poster/generated
```

Adding high fidelity — steady RANS below stall, unsteady above it, on foils the
network is allowed to see:

```bash
python cli.py sensor-multifidelity                  # print the plan first
python cli.py sensor-multifidelity --run --max-workers 4
python cli.py sensor-finalize && python cli.py sensor-cache
python cli.py sensor-train --lr 2e-4 --init-from artifacts/sensor/run/best.pt \
    --fidelity-weights "panel=1,rans=8,urans=8"
```

> The numbers previously quoted here — `Cl` R² = 0.996, relative field RMSE
> 0.025 (`u`), 0.16 (`v`), 0.17 (`p`) — were measured on the *previous*
> architecture and are stale. The network was rewritten and the old checkpoint
> does not load into it; `SurrogatePredictor` refuses it by name and tells you
> to retrain. See [docs/results.md](docs/results.md) §8 for the commands that
> regenerate every table.

### The full twin

```bash
# solver-driven, no network needed
python cli.py twin --airfoil Data/Airfoils/4/n0009sm.dat.txt --alpha 14 --use-surrogate False

# surrogate-driven, interpreted by the language model with RAG
python cli.py twin --airfoil Data/Airfoils/4/n0009sm.dat.txt --alpha 14 --interpret --use-rag

# exercise the instrumentation-fault detector end to end
python cli.py twin --airfoil Data/Airfoils/4/n0009sm.dat.txt --alpha 14 \
    --inject-faults "mains,dropout,clipping"
```

---

## Configuration

All hyperparameters are in `configs/default.yaml`. Key settings:

```yaml
model_name: "microsoft/Phi-4-mini-instruct"
seed: 42

dataset:
  n_per_regime: 40
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1

training:
  batch_size: 4
  learning_rate: 2.0e-4
  epochs: 3
  lora_rank: 8
  lora_alpha: 16
  lora_dropout: 0.05
  max_seq_length: 2048

rag:
  top_k: 3
  embedding_model: "all-MiniLM-L6-v2"
  chunk_size: 256
  chunk_overlap: 128

eval:
  freq_tolerance: 0.15
  regime_f1_threshold: 0.5
```

Override any setting via CLI: `python cli.py train --epochs 5 --learning-rate 1e-04`.

---

## Schema Contract

The `FeatureRecord` is the model-facing input and the source of truth for each experiment. It contains:

| Section | Key Fields | Units |
|---------|-----------|-------|
| **conditions** | Re, AoA, U∞, U*, chord | dimensionless, deg, m/s, m |
| **piv_derived** | f_shed, St, wake width, separation, POD, TKE | Hz, dimensionless, x/c |
| **structure** | A_rms/c, f_dom, f_n, mode, camber | dimensionless, Hz |
| **forces** | Cl mean/rms/freq, Cd mean/rms/freq, Cm mean/rms/freq | dimensionless, Hz |

See `docs/schema.md` for complete field documentation, valid ranges, and the PIV-to-FeatureRecord mapping.

The ground-truth regime is **never** included in the model prompt — it is carried alongside for evaluation only.

---

## Flow Regimes

The system distinguishes five mutually exclusive flow/aeroelastic regimes:

| Regime | Characteristics | Example Signals |
|--------|----------------|-----------------|
| **Attached** | Low unsteadiness, small structural response | Quiet wake, negligible vibration |
| **Separated** | Massive separation, broadband wake, no limit cycle | Energetic wake, moderate vibration |
| **VIV Lock-in** | Vortex shedding synchronized with structural frequency | Large saturated limit cycle |
| **Galloping** | Low-frequency, large-amplitude translational instability | f_n << f_shed, growing amplitude |
| **Stall Flutter** | SDOF torsional LCO driven by dynamic stall | Near/above stall, high AoA |

---

## Reproducibility

The project is fully seeded for reproducibility:

- Seeds can be set via the `seed` field in `configs/default.yaml` or overridden on the CLI
- `tools/utils/repro.py` seeds Python's `random`, NumPy, PyTorch, and MLX
- All results are deterministic given the same seed and configuration
- The `PYTHONHASHSEED` environment variable is set for dict iteration reproducibility

```python
from tools.utils.repro import set_seed
set_seed(42)  # seeds all RNGs
```

---

## Adding Real Experimental Data

To use real PIV/structural data instead of synthetic:

1. Convert your experimental measurements into `RawExperiment` objects matching `Data/schema.py`
2. Run feature extraction: `extract_features(raw_experiment)` → `FeatureRecord`
3. Add the resulting `FeatureRecord` to the dataset pipeline
4. The ground-truth regime should be `None` for real data — the rule-based classifier in `Data/describe.py` will attempt classification

The `range_warnings()` method on `FeatureRecord` will flag any values outside expected physical ranges, helping detect unit mismatches (e.g., Hz vs rad/s, mm vs m).

---

## Adding Papers to the Corpus

Place PDF files in `Data/Papers/`. Each PDF will be:

1. Extracted via `pdfminer.six`
2. Chunked into overlapping passages (256 tokens, 50% overlap)
3. Embedded with `all-MiniLM-L6-v2`
4. Indexed with FAISS (Flat or IVF)

```bash
# Add papers to Data/Papers/
cp my_paper.pdf Data/Papers/

# Rebuild the index
python cli.py index-corpus
```

---

## Training on Apple Silicon (M4 Pro)

The training system uses **MLX** (Apple's machine learning framework) for efficient on-device training:

- LoRA adapters (rank 8 by default) are applied to all linear layers in the transformer
- Training supports gradient accumulation, warmup, and periodic evaluation
- Checkpoints can be saved and resumed
- Adapter weights are saved in safetensors format

**Expected training time**: ~10-20 minutes per epoch on M4 Pro with 200 examples (batch size 4).

For inference after training, weights can be merged: `merge_weights(base_model, adapter_path, output_path)`.

---

## Evaluation Metrics

The evaluation harness (`eval/evaluate.py`) reports:

| Metric | Description |
|--------|-------------|
| **Regime Macro F1** | Macro-averaged F1 across all 5 flow regimes |
| **Lock-in F1** | Binary F1 for detecting locked vs. not-locked |
| **Per-field Accuracy** | Accuracy of numeric fields within tolerance |
| **Avg Field Accuracy** | Average across all numeric fields |

With-vs-without-RAG comparison is supported via `evaluate_with_rag()`.

Tolerance semantics are explicit per field: shedding frequency is judged at
15% *relative* tolerance, Strouhal number at ±0.05 *absolute*, RMS amplitude
at ±0.02 *absolute*, and mode number must match exactly.

`python cli.py eval` writes both `eval_results.json` and a
presentation-ready `eval_results.md` report (summary table, per-field
accuracy, regime confusion matrix, per-class precision/recall/F1, lock-in
detection) via `eval/report.py`.

### Regime score breakdown

`Data.describe.regime_scores(fr)` scores **all five** regimes for a record
(normalized to sum to 1; the winner keeps its calibrated confidence, the
rest share the remaining mass proportionally to raw evidence). Shown as a
bar breakdown in the Gradio Quick Summary.

### What-if parameter sweeps

`features/sweep.py` maps predicted regimes across a range of one operating
parameter — `reduced_velocity`, `angle_of_attack_deg`, or
`natural_frequency_hz` — rescaling physically coupled quantities (U* sweep
holds St and f_n fixed, so U, Re, and f_shed scale together). Detects regime
transition boundaries, e.g. the lock-in window. Available as a Gradio tab
("What-if sweep"), the MCP tool `sweep_regime_map`, and `cli.py demo --sweep`.

### Physics consistency checks

Beyond schema range checks (`FeatureRecord.range_warnings()`), the pipeline
cross-validates the record against itself via
`features.extract.check_physical_consistency()`: reported St vs. f·c/U,
lock-in label vs. actual frequency ratio, damping ratio vs. spectral
bandwidth, and coherence phase physics. These warnings surface in the Gradio
UI (Tab 2), the FastAPI `/infer` response, and the MCP
`interpret_experiment` tool as `record_consistency_warnings`.

---

## Project Structure

```
FSI-SLM/
├── Data/                    # Data generation and schemas
│   ├── schema.py           # Pydantic models (RawExperiment, FeatureRecord)
│   ├── synthetic.py        # Physically-informed experiment generator
│   ├── describe.py         # Deterministic description generator
│   ├── build_dataset.py    # SFT dataset builder
│   └── Papers/             # PDF corpus for RAG
├── features/
│   ├── extract.py          # Feature extraction (FFT, POD, St)
│   └── noise.py            # Instrumentation-fault detection (PIV, ADC, mains)
├── sensor/                  # Surrogate: geometry -> fields, loads, coefficients
│   ├── geometry.py         # .dat parsing, TE closure, rotation
│   ├── sdf.py              # Signed distance field + U-Net input tensor
│   ├── flow/
│   │   ├── panel.py        # Hess-Smith panel method
│   │   ├── viscous.py      # Beddoes-Leishman stall + drag
│   │   └── solve.py        # Fields: near-wall layer, bubble, wake, pressure
│   ├── structure/
│   │   └── beam.py         # Chordwise Hermite beam + spanwise modal model
│   ├── cfd/                # OpenFOAM backend (same tensor contract)
│   ├── simulate.py         # One case, end to end
│   ├── generate.py         # 107 x 18 batch driver, manifest, splits
│   ├── data.py             # Memmap cache, normalisation
│   ├── model.py            # U-Net (field decoder + scalar head)
│   ├── train.py            # Resumable, device-agnostic training
│   ├── predict.py          # Inference + held-out scoring
│   ├── bridge.py           # Steady fields -> time-resolved FeatureRecord
│   ├── api.py              # Shared entry point for CLI / REST / MCP
│   └── figures.py          # Figures generated from artefacts on disk
├── train/
│   ├── sft.py              # MLX-based LoRA training
│   └── tokenization/
│       └── prompt_template.py  # Shared Phi-4-mini chat template
├── rag/
│   ├── corpus_loader.py    # PDF text extraction and chunking
│   ├── index.py            # FAISS index builder/loader
│   ├── retrieve.py         # Semantic retrieval
│   └── prompt_assembly.py  # RAG prompt assembly
├── eval/
│   ├── validate_output.py  # Standalone output validator
│   └── evaluate.py         # Evaluation harness
├── serve/
│   └── fastapi_app.py      # FastAPI inference server
├── tools/utils/
│   └── repro.py            # Seed-setting utilities
├── tests/                  # 220 tests
│   ├── test_schema.py
│   ├── test_synthetic.py
│   ├── test_features.py
│   ├── test_describe.py
│   ├── test_validate_output.py
│   ├── test_panel.py       # thin-airfoil theory, d'Alembert, stagnation
│   ├── test_viscous.py     # polars vs Abbott & von Doenhoff
│   ├── test_structure.py   # beam vs closed-form, cantilever frequency
│   ├── test_noise.py       # detectors fire, and stay quiet
│   └── test_surrogate.py   # tensors, dataset, model, twin bridge
├── configs/
│   └── default.yaml        # Default configuration
├── docs/
│   ├── schema.md           # Schema documentation
│   ├── surrogate.md        # Surrogate physics, validation and limits
│   └── abstract_claims.md  # Abstract claims mapped to implementation
├── cli.py                  # Unified CLI entry point
├── requirements.txt        # Pinned dependencies
└── README.md               # This file
```

---

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_schema.py -v

# Run with coverage (install pytest-cov first)
python -m pytest tests/ --cov=Data --cov=features --cov=eval --cov=rag
```

---

## License

MIT

## Citation

If you use this work in research, please cite:

```bibtex
@software{fsi_slm,
  author = {Angad Kumar},
  title = {FSI-SLM: Fluid-Structure Interaction Interpretation via Small Language Models},
  year = {2026},
  url = {https://github.com/AngadKumar16/FSI-SLM}
}