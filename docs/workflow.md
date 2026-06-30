# FSI-SLM Project Workflow

End-to-end pipeline diagram for training and serving the Fluid-Structure
Interaction Small Language Model. Renders inline in any markdown viewer
that supports Mermaid (GitHub, GitLab, VSCode preview, etc.).

## High-level pipeline

```mermaid
flowchart TB
    subgraph sources["Inputs"]
        PDFs["PDF library<br/>(Data/Papers/*.pdf)"]
        Synth["Simulator<br/>(Data/synthetic.py)"]
    end

    subgraph build["Stage 1 - Build"]
        FE["Signal feature extractor<br/>(features/extract.py)"]
        DS["SFT dataset builder<br/>(Data/build_dataset.py)"]
        BR["Paper chunk loader<br/>(rag/corpus_loader.py)"]
        BI["Vector index builder<br/>(rag/index.py + FAISS + MiniLM-L6-v2)"]
        FD["Rule-based oracle<br/>(Data/describe.py)"]
    end

    subgraph train["Stage 2 - Train"]
        PT["Chat prompt template<br/>(train/tokenization/prompt_template.py)"]
        SFT["LoRA trainer<br/>(train/sft.py -> mlx_lm.lora.train_model)"]
        BASE["Base language model<br/>(artifacts/models/Phi-4-mini-instruct, 7.7 GB)"]
        ADAPT["Trained adapter<br/>(artifacts/adapters/, LoRA rank=8 alpha=16, 46 MB)"]
    end

    subgraph serve["Stage 3 - Serve"]
        INF["Inference runner<br/>(inference/runner.py)"]
        EVAL["Scorer<br/>(eval/evaluate.py + eval/validate_output.py)"]
        CLI["CLI commands<br/>(cli.py: eval, infer)"]
        API["HTTP API<br/>(serve/fastapi_app.py, /infer /health /schema)"]
    end

    Synth --> FE
    FE --> FR["FeatureRecord<br/>(Pydantic v2, Data/schema.py)"]
    FR --> DS
    FD --> DS
    PT --> DS
    DS --> JSONL["train.jsonl / val.jsonl / test.jsonl"]

    PDFs --> BR
    BR --> BI
    BI --> IDX["Paper index<br/>(artifacts/rag_index/index.faiss, 5,563 passages)"]

    BASE --> SFT
    JSONL --> SFT
    SFT --> ADAPT
    PT --> SFT

    ADAPT --> INF
    IDX --> INF
    FR --> INF
    INF --> EVAL
    INF --> CLI
    INF --> API
```

## CLI command flow

```mermaid
flowchart LR
    G["Generate dataset<br/>(cli.py generate-data)"] --> B["Build SFT data<br/>(cli.py build-dataset)"]
    I1["Index papers<br/>(cli.py index-corpus)"] --> R["RAG retriever"]
    B --> T["Train adapter<br/>(cli.py train)"]
    T --> A["Adapter<br/>(artifacts/adapters/)"]
    A --> E["Evaluate<br/>(cli.py eval)"]
    A --> IN["Run single record<br/>(cli.py infer)"]
    A --> S["HTTP server<br/>(cli.py serve)"]
    R --> IN
    R --> S
    R --> E
```

## Inference pipeline (single FeatureRecord -> text)

```mermaid
flowchart LR
    A[Client / CLI / HTTP] -->|POST| B[HTTP handler<br/>(serve/fastapi_app.py)]
    B --> C[Parse request<br/>(Pydantic InferRequest)]
    C --> D[Inference runner<br/>(inference/runner.py)]
    D -->|no RAG| E[Build prompt<br/>(render_chat + SYSTEM_PROMPT)]
    D -->|use_rag| F[Retrieve top-3<br/>(FAISS via rag/retrieve.py)]
    F --> G[Assemble RAG prompt<br/>(rag/prompt_assembly.py)]
    G --> E
    E --> H[Generate text<br/>(mlx_lm.generate)]
    H --> I[Phi-4-mini + LoRA adapter<br/>(artifacts/adapters/)]
    I --> J[Raw text]
    J --> K[Parse + validate<br/>(eval/validate_output.py)]
    K --> L[JSON response<br/>(InferResponse)]
    L -->|JSON| A
```

## Training loop (inside train/sft.py)

```mermaid
flowchart TB
    S[Configuration<br/>(TrainingConfig dataclass)] --> L["Load base model<br/>(mlx_lm.utils.load)"]
    L --> F["Inject LoRA<br/>(linear_to_lora_layers -> all linear)"]
    F --> D["Tokenize SFT data<br/>(ChatDataset + mask_prompt=True)"]
    D --> T["Train loop<br/>(mlx_lm.tuner.train)"]
    T -->|every save_every| C["Checkpoint<br/>(0000xxx_adapters.safetensors)"]
    T --> F1["Final adapter<br/>(artifacts/adapters/adapters.safetensors)"]
    F1 --> WD["Adapter + config<br/>(adapter_config.json)"]
```

## Data flow summary

| Stage | Inputs | Outputs | Tool |
|---|---|---|---|
| Synthesize | random + regime distribution | RawExperiment list | `Data/synthetic.py` |
| Feature extraction | RawExperiment | FeatureRecord | `features/extract.py` |
| Build dataset | FeatureRecord + (optional) FAISS hits | train/val/test JSONL | `Data/build_dataset.py` |
| Index corpus | PDFs in `Data/Papers/` | `index.faiss` + `metadata.json` | `rag/index.py` |
| Train | JSONL + Phi-4-mini base | LoRA adapter | `train/sft.py` |
| Inference | FeatureRecord (+ optional FAISS hits) + adapter | structured text | `inference/runner.py` |
| Score predictions | prediction text vs ground truth | JSON metrics | `eval/evaluate.py` |
| Serve | same as inference | JSON via HTTP | `serve/fastapi_app.py` |
