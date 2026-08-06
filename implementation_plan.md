# Implementation Status

What is built, what is partially built, and what is not. This file replaces the
original pre-build plan, which described a system that has since been built and
then substantially outgrown — the plan called for 18 new files and a single
interpreter pipeline; the repository now has two halves, three CFD backends and
266 tests.

For measured numbers see [docs/results.md](docs/results.md). For the mapping
from the paper abstract to code see
[docs/abstract_claims.md](docs/abstract_claims.md).

Status key: **done** · **partial** (real but narrower than the name suggests) ·
**not started**

---

## 1. Interpreter (the original plan)

Everything in the original plan shipped.

| Component | Status | Where |
|---|---|---|
| Schema (`RawExperiment`, `FeatureRecord`) | **done** | `Data/schema.py` |
| Synthetic generator, 5 regimes | **done** | `Data/synthetic.py` |
| Feature extraction (FFT, POD/SVD, damping) | **done** | `features/extract.py` |
| Rule-based oracle | **done** | `Data/describe.py` |
| SFT dataset builder, split by `experiment_id` | **done** | `Data/build_dataset.py` |
| Shared Phi-4 chat template | **done** | `train/tokenization/prompt_template.py` |
| RAG: loader, FAISS index, retriever, assembly | **done** | `rag/` |
| MLX LoRA training, resumable | **done** | `train/sft.py` |
| Output validator (parse + physics checks) | **done** | `eval/validate_output.py` |
| Evaluation harness | **done** | `eval/evaluate.py`, `eval/report.py` |
| CLI | **done** | `cli.py`, 21 commands |
| FastAPI server | **done** | `serve/fastapi_app.py`, 8 routes |
| Reproducibility seeding | **done** | `tools/utils/repro.py` |

Built beyond the plan:

| Component | Where |
|---|---|
| Gradio web UI | `serve/gradio_app.py` |
| MCP server, 12 tools | `serve/mcp_server.py` |
| Instrumentation-fault detection, 11 detectors | `features/noise.py` |
| What-if regime sweeps | `features/sweep.py` |
| Self-consistency checks (`St` vs `f·c/U`, lock-in vs ratio, coherence phase) | `features/extract.py` |
| Streaming inference | `POST /infer/stream` |
| Markdown eval reports | `eval/report.py` |

## 2. Surrogate (`sensor/`) — not in the original plan at all

| Component | Status | Where |
|---|---|---|
| Geometry: `.dat` parsing, TE closure, rotation | **done** | `sensor/geometry.py` |
| Hess–Smith panel method | **done** | `sensor/flow/panel.py` |
| Beddoes–Leishman static stall + drag | **done** | `sensor/flow/viscous.py` |
| Field construction (near-wall, bubble, wake, pressure) | **done** | `sensor/flow/solve.py` |
| Chordwise Hermite beam + spanwise modal model | **done** | `sensor/structure/beam.py` |
| Steady RANS backend (gmsh + `simpleFoam`) | **done** | `sensor/cfd/` |
| Unsteady RANS backend (`pimpleFoam`) | **done** (code); sweep **partial** | `sensor/cfd/unsteady.py` |
| Versioned input contract, 13 channels + inviscid prior | **done** | `sensor/sdf.py` |
| Dataset driver, manifest, foil-disjoint splits | **done** | `sensor/generate.py` |
| Memmap cache with dataset fingerprinting | **done** | `sensor/data.py` |
| U-Net (FiLM, attention, wall-weighted scalar head) | **done** | `sensor/model.py` |
| Training: EMA, resumable, time-budgeted | **done** | `sensor/train.py` |
| Exact mirror symmetry: augmentation + TTA | **done** | `sensor/symmetry.py` |
| Per-fidelity scoring | **done** | `sensor/predict.py` |
| Twin bridge: steady fields → `FeatureRecord` | **done** | `sensor/bridge.py` |
| Shared API for CLI / REST / MCP | **done** | `sensor/api.py` |
| Multi-fidelity training (conditioning + weights + 2-stage) | **partial** — wired, tested, no released checkpoint | `sensor/train.py` |

## 3. What is genuinely incomplete

### 3.1 No PIV. **Not started.**

The rig is not connected. `Data/schema.py` defines the contract a rig would
populate and `sensor/bridge.py` fills it from the surrogate instead. Every
record self-labels `provenance.signals = "synthesised by a 1-DOF instrument
model, not measured"`.

The substitution point is exactly one function: `build_raw_experiment`. Nothing
downstream changes when it is replaced — that is the entire reason the
`RawExperiment` contract exists.

### 3.2 Multi-fidelity training. **Partial.**

All three mechanisms exist and are tested — a fidelity conditioning channel,
per-fidelity loss weights, and two-stage fine-tuning. What is missing is *legal
data*: the only RANS rows in the manifest are the 15 on NACA 0009, which is
pinned to the **test** split, so training on them would be training on test.

`scripts/verify_multifidelity.py data` is the gate that enforces this and it
currently exits non-zero, by design. Printing that failure is the point of it.

### 3.3 URANS sweep. **Partial — running now.**

The transient solver, its analysis and its two convergence tests are complete
and covered by `tests/test_unsteady.py`. The sweep that produces high-fidelity
rows on *training*-split foils is in flight: **8 of 12 cases solved**, rows
flushed per case to `artifacts/cfd/rows/urans_*.jsonl`, not yet merged into the
manifest. Only 3 of the 8 are `statistically_steady`.

This is the whole schedule risk in the project — cases cost hours apiece, and
finishing it is what turns §3.2 from "wired" into a reportable number.

### 3.4 Structural coupling is one-way and static. **By design, for now.**

Pressure drives deflection; deflection does not feed back into the flow. Two-way
static iteration is a short extension of `solve_structure` and is deliberately
not claimed anywhere.

### 3.5 Regime evaluation is saturated. **Needs a harder test set.**

The trained adapter scores 50/50 and 100/100 on regime classification. That is a
statement about the difficulty of the generated test set, not a result about the
model. The discriminating metric is lock-in F1 (0.857, precision 0.750 at recall
1.000), and the same over-firing bias is present in the rule-based labels the
adapter was trained on — so the fix is in `Data/describe.py`'s lock-in rule and a
regenerated dataset, not more training.

## 4. Ordered next steps

1. **Finish the URANS sweep** (4 cases outstanding), then `sensor-finalize`,
   then `scripts/verify_multifidelity.py data`. Extend `t_end` for the α = 12°
   cases first — they cover 0.5–2.4 shedding cycles against the 6 required.
2. **Fine-tune on the mix** once the gate passes, and score with
   `verify_multifidelity.py model`, which fails if RANS pressure error does not
   fall or the panel score degrades by more than 25%.
3. **Fix the lock-in rule** in `Data/describe.py`, regenerate the SFT dataset,
   retrain the adapter, and rebuild the test set to be discriminative.
4. **Run the two surrogate ablations** that are wired but unmeasured —
   `--spec v2_noprior` (what the inviscid prior is worth) and `--augment=False`
   (what mirror augmentation is worth).
5. **Connect the PIV rig**: implement the ingest that replaces
   `build_raw_experiment` and validate `range_warnings()` against real
   measurements.
6. **Close the structural loop** — two-way static iteration in
   `solve_structure`.
