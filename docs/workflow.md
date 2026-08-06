# FSI-SLM Project Workflow

How the repository fits together, end to end. Every box names the file that does
the work, so a box is greppable. Mermaid renders inline in GitHub, GitLab and
VSCode preview; `docs/workflow.txt` is the same content as ASCII for terminal
viewing.

The system has **two halves that meet at one contract**:

| Half | Question it answers | Entry point |
|---|---|---|
| **Surrogate** (`sensor/`) | *What does the flow do?* — geometry + incidence → fields, forces, deflection | `cli.py sensor-predict` |
| **Interpreter** (`Data/`, `features/`, `train/`, `rag/`, `inference/`) | *What does that mean?* — experiment record → regime, lock-in, caveats | `cli.py infer` |

The contract between them is `FeatureRecord` (`Data/schema.py`). The surrogate
produces one; the interpreter consumes one. Because the contract is the only
coupling, a real PIV rig can replace the surrogate without touching anything
downstream — that substitution point is `sensor/bridge.py:build_raw_experiment`.

`cli.py twin` runs both halves back to back, which is the full digital twin:
geometry in, interpreted experiment out, no human step.

---

## 1. The whole system

```mermaid
flowchart TB
    subgraph inputs["Inputs"]
        GEO["Airfoil section<br/>(Data/Airfoils/**.dat)"]
        AOA["Angle of attack, Re"]
        PDFs["Paper library<br/>(Data/Papers/, 16 PDFs)"]
        RIG["Real PIV rig<br/>(not connected — contract only)"]
        QRY["User question<br/>(free text)"]
    end

    subgraph solvers["Stage A — Solve (sensor/)"]
        PANEL["Panel method + viscous correction<br/>(sensor/flow/, ~0.13 s/case)"]
        RANS["Steady RANS, k-omega SST<br/>(sensor/cfd/, ~4 min/case)"]
        URANS["Unsteady RANS, pimpleFoam<br/>(sensor/cfd/unsteady.py, ~hours/case)"]
        BEAM["Chordwise Euler-Bernoulli beam<br/>(sensor/structure/beam.py)"]
        PACK["Shared tensor packer<br/>(sensor/cfd/adapt.py)"]
    end

    subgraph dataset["Stage B — Dataset"]
        GEN["Batch driver + manifest + splits<br/>(sensor/generate.py)"]
        MAN["manifest.jsonl<br/>1,928 rows, fidelity-tagged"]
        SPEC["Input contract v2, 13 channels<br/>(sensor/sdf.py InputSpec)"]
        CACHE["Memmap cache + norm stats<br/>(sensor/data.py)"]
    end

    subgraph unet["Stage C — Surrogate"]
        TRAIN["U-Net trainer<br/>(sensor/train.py, EMA, resumable)"]
        MODEL["U-Net, 11.59 M params<br/>(sensor/model.py)"]
        PRED["Predict + score<br/>(sensor/predict.py, mirror TTA)"]
    end

    subgraph bridge["Stage D — Bridge to an experiment record"]
        BR["Steady fields -> time series<br/>(sensor/bridge.py)"]
        RAW["RawExperiment<br/>(Data/schema.py)"]
        FE["FFT / POD / damping extraction<br/>(features/extract.py)"]
        NOISE["11 instrumentation detectors<br/>(features/noise.py)"]
        FR["FeatureRecord + [data_quality]<br/>THE CONTRACT"]
    end

    subgraph slm["Stage E — Interpreter"]
        SYN["Synthetic experiments, 5 regimes<br/>(Data/synthetic.py)"]
        ORACLE["Rule-based oracle<br/>(Data/describe.py)"]
        SFT["SFT dataset builder<br/>(Data/build_dataset.py)"]
        LORA["MLX LoRA trainer<br/>(train/sft.py, rank 8, 46 MB)"]
        BASE["Phi-4-mini-instruct 3.8 B<br/>(artifacts/models/)"]
        IDX["FAISS index, 1,840 chunks<br/>(rag/index.py, MiniLM-L6-v2)"]
        RUN["Inference runner<br/>(inference/runner.py)"]
    end

    subgraph out["Stage F — Surfaces"]
        CLI["CLI, 21 commands<br/>(cli.py)"]
        API["REST, 8 routes<br/>(serve/fastapi_app.py)"]
        MCP["MCP, 12 tools<br/>(serve/mcp_server.py)"]
        UI["Gradio UI<br/>(serve/gradio_app.py)"]
    end

    GEO --> PANEL & RANS & URANS
    AOA --> PANEL & RANS & URANS
    PANEL --> BEAM
    RANS --> BEAM
    URANS --> BEAM
    BEAM --> PACK
    PACK --> GEN --> MAN --> CACHE
    SPEC --> CACHE
    CACHE --> TRAIN --> MODEL --> PRED

    PRED --> BR
    RIG -.->|"replaces build_raw_experiment"| RAW
    BR --> RAW --> FE --> FR
    NOISE --> FR

    SYN --> FE
    FR --> SFT
    ORACLE --> SFT
    SFT --> LORA
    BASE --> LORA
    LORA --> RUN
    BASE --> RUN
    PDFs --> IDX --> RUN
    FR --> RUN
    QRY --> RUN

    RUN --> CLI & API & MCP & UI
    PRED --> CLI & API & MCP
```

Two things in that diagram are worth stating outright rather than reading off
the arrows:

- **The surrogate is trained on solver output, not on the rig.** The dashed
  arrow from `RIG` is the only unimplemented edge in the picture. Every record
  the system produces today carries
  `provenance.signals = "synthesised by a 1-DOF instrument model, not measured"`.
- **The prior enters at cache time, not at solve time.** `pot_u/pot_v/pot_p` in
  the 13-channel input are a panel solve on the same raster, so the network
  learns the *correction* to potential flow. It costs ~40 ms per case and is
  available at every fidelity, which is what keeps panel and RANS rows
  interchangeable.

---

## 2. Surrogate pipeline (geometry → fields)

```mermaid
flowchart LR
    A[".dat section"] --> B["Parse, close TE, rotate<br/>(sensor/geometry.py)"]
    B --> C{"Which solver?"}
    C -->|"any alpha"| D["Hess-Smith panel<br/>+ Beddoes-Leishman stall<br/>(sensor/flow/)"]
    C -->|"alpha <= 8 deg"| E["simpleFoam k-omega SST<br/>(sensor/cfd/run.py)"]
    C -->|"alpha >= 10 deg"| F["pimpleFoam + fieldAverage<br/>(sensor/cfd/unsteady.py)"]
    D --> G["Pressure -> dCp(x)"]
    E --> G
    F --> G
    F --> FS["Measured f_shed<br/>(FFT of Cl(t), parabolic peak)"]
    G --> H["Hermite beam solve<br/>(sensor/structure/beam.py)"]
    H --> I["5 field channels + 5 scalars<br/>u, v, p, load, disp &middot; Cl, Cd, Cm, w_max, twist_TE"]
    I --> J["128x128 npz + manifest row<br/>(sensor/cfd/adapt.py)"]
    FS --> J
```

Solver choice is a measurement, not a preference. Every steady case at α ≥ 10°
in the RANS sweep hit the iteration cap with initial residuals of order 10⁻² and
lift oscillating by up to ±0.43 — massively separated flow has no steady
solution to converge *to*. `finalize_manifest` ranks `urans > rans > panel` for
the same (foil, angle), so re-solving upgrades a case in place.

## 3. Surrogate training

```mermaid
flowchart TB
    M["manifest.jsonl<br/>1,928 rows"] --> S["Foil-disjoint splits<br/>76 train / 14 val / 17 test"]
    S --> C["sensor-cache<br/>13-channel memmap + norm stats"]
    C --> D["Mirror augmentation<br/>(sensor/symmetry.py, exact)"]
    D --> T["Train loop (sensor/train.py)"]
    T --> L1["L1 fluid + L1 solid + Huber scalars"]
    T --> L2["+ gradient 0.25, near-wall 3.0,<br/>divergence 0.02, circulation 0.05"]
    T --> E["EMA weights, decay 0.999"]
    E --> V["Validate on EMA weights"]
    V --> B["best.pt + norm stats + input spec<br/>+ dataset fingerprint"]
    B --> EV["sensor-eval<br/>per-fidelity, mirror TTA"]
```

Splits hold out **whole sections**, never individual angles: splitting by case
would put the same shape in train and test at neighbouring incidences and
inflate the score without measuring generalisation. NACA 0009 is pinned to test,
so every number reported for it is a generalisation result.

The checkpoint carries its own normalisation statistics, its input spec, and a
fingerprint of the dataset. A changed dataset forces a recompute rather than
silently rescaling the loss, and a mismatched cache is refused with a message
rather than a broadcast error.

## 4. The twin bridge (fields → experiment record)

```mermaid
flowchart TB
    F["Steady fields + scalars"] --> W["Wake width<br/>(half-deficit, 1.5c downstream)"]
    F --> SP["Separation point<br/>(first near-wall reversal)"]
    F --> DH["Den Hartog margin<br/>dCl/dalpha + Cd < 0<br/>(2 extra solver calls)"]
    F --> FQ{"Case has URANS row?"}
    FQ -->|"yes"| MEAS["Measured f_shed<br/>(lookup_unsteady, manifest)"]
    FQ -->|"no"| CORR["Roshko St on projected height<br/>c(sin &alpha; + t/c)"]
    MEAS --> SD["SDOF transfer function<br/>at r = f_shed / f_n"]
    CORR --> SD
    W --> RG["Regime by rule<br/>(sensor/bridge.py:classify_regime)"]
    SP --> RG
    DH --> RG
    SD --> SIG["Synthesised time series<br/>+ PIV snapshots"]
    RG --> SIG
    SIG --> RE["RawExperiment<br/>(+ optional injected faults)"]
    RE --> EX["extract_features<br/>FFT, POD/SVD, damping"]
    RE --> NZ["noise.py screen"]
    EX --> FRC["FeatureRecord"]
    NZ --> DQ["[data_quality] block"]
    FRC --> P["Prompt"]
    DQ --> P
```

`provenance.shedding_frequency` on every record says which of the two frequency
paths was used. That distinction matters: lock-in is decided by comparing
`f_shed` to `f_n` at the few-percent level, so a measured frequency makes the
lock-in call mean something, and a correlated one makes it restate an
assumption.

---

## 5. Interpreter: training the SLM

```mermaid
flowchart TB
    S["Simulator, 5 regimes<br/>(Data/synthetic.py)"] --> FE["extract_features"]
    FE --> FR["FeatureRecord"]
    FR --> DS["build_dataset"]
    OR["Rule-based oracle<br/>(Data/describe.py)"] --> DS
    PT["Phi-4 chat template<br/>(train/tokenization/prompt_template.py)"] --> DS
    DS --> J["train / val / test .jsonl<br/>split BY experiment_id"]
    J --> T["mlx_lm LoRA<br/>mask_prompt=True"]
    BASE["Phi-4-mini-instruct 3.8 B"] --> T
    T --> A["adapters.safetensors<br/>rank 8, scale 16, 46 MB"]
```

`mask_prompt=True` means the loss covers assistant tokens only — the model is
never scored on reproducing the record it was handed. The ground-truth regime is
excluded from `to_prompt_block()`, so the model must infer it rather than copy
it.

## 6. Interpreter: serving one record

```mermaid
flowchart LR
    A["CLI / REST / MCP / UI"] --> B["Parse request<br/>(Pydantic InferRequest)"]
    B --> C["InferenceRunner<br/>(lazy-loads model + adapter + FAISS)"]
    C --> D{"use_rag?"}
    D -->|"no"| E["render_chat + SYSTEM_PROMPT"]
    D -->|"yes"| F["Retrieve top-3<br/>(rag/retrieve.py)"]
    F --> G["Assemble<br/>(rag/prompt_assembly.py)"]
    G --> E
    E --> H["mlx_lm.generate<br/>Phi-4-mini + LoRA"]
    H --> I["Parse + physics-check<br/>(eval/validate_output.py)"]
    I --> J["Response + citations<br/>+ record_consistency_warnings"]
```

Two paths exist at serve time and they use **different models**:

| Path | Model | Why |
|---|---|---|
| `runner.generate` — structured interpretation | base **+ LoRA adapter** | the adapter is trained on exactly the fixed 6-part format |
| `runner.answer` — free-text Q&A | **base** instruct model | the adapter only knows the fixed format; open-ended questions go to the base model with the record pinned into context |

Neither path lets retrieved passages override the record. The system prompt
states the `FeatureRecord` is the only source of truth for measured numbers, and
`[data_quality]` flags carry an explicit instruction not to attribute an
instrumentation artefact to a flow mechanism.

## 7. Full twin (`cli.py twin`)

```mermaid
flowchart LR
    G["--airfoil, --alpha"] --> S{"--use-surrogate"}
    S -->|"true"| N["U-Net forward pass<br/>(~ms)"]
    S -->|"false"| P["Panel + beam solve<br/>(~0.13 s)"]
    N --> B["bridge.digital_twin"]
    P --> B
    B --> L["lookup_unsteady<br/>(manifest, foil stem + alpha +/- 0.25 deg)"]
    L --> R["FeatureRecord + [data_quality]"]
    R --> I{"--interpret"}
    I -->|"no"| O1["JSON record"]
    I -->|"yes"| O2["SLM interpretation<br/>(+ RAG citations)"]
```

---

## 8. Data flow summary

| Stage | Inputs | Outputs | Tool |
|---|---|---|---|
| Solve (panel) | section + α + Re | 5 fields, 5 scalars | `sensor/flow/`, `sensor/structure/` |
| Solve (steady RANS) | section + α ≤ 8° | same contract + convergence record | `sensor/cfd/` |
| Solve (URANS) | section + α ≥ 10° | same + **measured** `f_shed` | `sensor/cfd/unsteady.py` |
| Generate | 107 sections × 18 angles | 1,926 `.npz` + manifest + splits | `sensor/generate.py` |
| Cache | manifest + `InputSpec` | 13-channel memmaps + norm stats | `sensor/data.py` |
| Train surrogate | cache | `best.pt` (11.59 M params) | `sensor/train.py` |
| Score surrogate | `best.pt` + held-out split | per-fidelity JSON metrics | `sensor/predict.py` |
| Bridge | steady fields | `RawExperiment` | `sensor/bridge.py` |
| Extract | `RawExperiment` | `FeatureRecord` | `features/extract.py` |
| Screen | `RawExperiment` signals | `[data_quality]` flags | `features/noise.py` |
| Synthesize | regime distribution | `RawExperiment` list | `Data/synthetic.py` |
| Build SFT | `FeatureRecord` + oracle | train/val/test JSONL | `Data/build_dataset.py` |
| Index | 16 PDFs | 1,840 chunks in FAISS | `rag/index.py` |
| Train SLM | JSONL + Phi-4-mini | LoRA adapter | `train/sft.py` |
| Interpret | `FeatureRecord` (+ passages) | structured text | `inference/runner.py:generate` |
| Answer | record + question (+ passages) | free text + citations | `inference/runner.py:answer` |
| Score SLM | prediction vs ground truth | JSON + Markdown report | `eval/evaluate.py`, `eval/report.py` |

## 9. Command map

```mermaid
flowchart LR
    subgraph sur["Surrogate"]
        SG["sensor-generate"] --> SC["sensor-cache"]
        SR["sensor-rans"] --> SF["sensor-finalize"]
        SU["sensor-urans"] --> SF
        SM["sensor-multifidelity"] --> SF
        SF --> SC
        SC --> ST["sensor-train"] --> SE["sensor-eval"]
        ST --> SP["sensor-predict"]
    end
    subgraph int["Interpreter"]
        GD["generate-data"] --> BD["build-dataset"] --> TR["train"] --> EV["eval"]
        IC["index-corpus"] --> IN["infer"]
        TR --> IN
    end
    SP --> TW["twin"]
    IN --> TW
    IN --> SV["serve"] & UI["ui"] & MC["mcp"]
```

| Surface | Entry points |
|---|---|
| CLI | 21 commands; `python cli.py -- --help` lists them |
| REST | `GET /health`, `GET /schema`, `POST /infer`, `POST /infer/stream`, `GET /sensor/status`, `POST /sensor/predict`, `POST /sensor/polar`, `POST /twin` |
| MCP | 12 tools — `interpret_experiment`, `ask_question`, `search_literature`, `validate_interpretation`, `check_experiment_consistency`, `sweep_regime_map`, `generate_synthetic_experiment`, `predict_flow`, `predict_polar`, `digital_twin_record`, `check_data_quality`, `surrogate_status` |
| UI | Gradio: upload/enter a record, interpret, ask, what-if sweep |

All three of CLI, REST and MCP call `sensor/api.py` for surrogate work and
`inference/runner.py` for language work, so the surfaces cannot drift apart.
