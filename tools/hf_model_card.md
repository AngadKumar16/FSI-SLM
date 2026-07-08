---
license: mit
base_model: microsoft/Phi-4-mini-instruct
library_name: mlx
tags:
  - lora
  - mlx
  - fluid-structure-interaction
  - physics
  - phi-4
pipeline_tag: text-generation
extra_gated_prompt: >-
  Access to this model is restricted. Please describe your intended use case.
  Requests are reviewed manually.
extra_gated_fields:
  Affiliation: text
  Intended use: text
---

# FSI-SLM — Phi-4-mini LoRA for Fluid-Structure Interaction Interpretation

LoRA adapters fine-tuning **microsoft/Phi-4-mini-instruct** (3.8B) to interpret
fluid-structure interaction (FSI) experiments: flow-regime classification
(attached / separated / VIV lock-in / galloping / stall flutter),
vortex-shedding frequency and Strouhal number reporting, lock-in detection,
amplitude trend, and mode-shape description.

Trained with MLX on Apple Silicon. Code: https://github.com/AngadKumar16/FSI-SLM

## Input format

The model consumes a compact `FeatureRecord` (flow conditions, PIV-derived
quantities, structural response, force coefficients) rendered through the
prompt template in `train/tokenization/prompt_template.py` of the repo, and
emits a fixed 6-part structured interpretation.

## Usage (MLX)

```python
from mlx_lm import load, generate

model, tokenizer = load(
    "microsoft/Phi-4-mini-instruct",
    adapter_path="adapters",  # this repo's adapters/ directory
)
```

See the GitHub repo for the full inference runner, RAG grounding over an FSI
paper corpus, and the Gradio UI.

## Training

- Method: LoRA (rank 8, alpha 16, dropout 0.05), all linear layers
- Data: 200 physically-consistent synthetic FSI experiments (160 train / 20 val / 20 test), seeded
- Framework: mlx-lm on Apple M4 Pro
- Eval: regime macro-F1, lock-in F1, per-field numeric accuracy (see below)

## Evaluation

<!-- EVAL_RESULTS -->

## Limitations

- Trained on synthetic data generated from canonical FSI physics (Strouhal
  scaling, lock-in bands, galloping/stall-flutter onset); real PIV data may
  distribution-shift.
- The structured adapter answers only the fixed interpretation format;
  free-text Q&A uses the base instruct model.

## Citation

```bibtex
@software{fsi_slm,
  author = {Angad Kumar},
  title = {FSI-SLM: Fluid-Structure Interaction Interpretation via Small Language Models},
  year = {2026},
  url = {https://github.com/AngadKumar16/FSI-SLM}
}
```
