---
license: mit
base_model: microsoft/Phi-4-mini-instruct
library_name: transformers
tags:
  - lora
  - merged
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

# FSI-SLM — Phi-4-mini fine-tuned for Fluid-Structure Interaction Interpretation

**microsoft/Phi-4-mini-instruct** (3.8B) with LoRA fine-tuning merged into the
base weights, for interpreting fluid-structure interaction (FSI) experiments:
flow-regime classification (attached / separated / VIV lock-in / galloping /
stall flutter), vortex-shedding frequency and Strouhal number reporting,
lock-in detection, amplitude trend, and mode-shape description.

Standard `transformers` format — runs on **CUDA, CPU, or Apple Silicon (MPS)**.
The original MLX LoRA adapters are included under `adapters/` for MLX users.
Code: https://github.com/AngadKumar16/FSI-SLM

## Usage (any platform)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "AngadKumar/fsi-slm-phi4-mini"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(
    repo, dtype=torch.bfloat16, device_map="auto"
)

SYSTEM_PROMPT = (
    "You are an expert in fluid-structure interaction (FSI) and aeroelasticity. "
    "You are given a serialized feature record from a single airfoil/hydrofoil/"
    "membrane experiment. The feature record is the ONLY source of truth for this "
    "experiment's measured numbers. Interpret it and respond with a structured "
    "description that states, in this order: (1) the flow regime (attached, "
    "separated, VIV lock-in, galloping, or stall flutter); (2) the dominant "
    "frequency in Hz and the Strouhal number St; (3) the lock-in status (compare "
    "shedding frequency to the structural natural frequency); (4) the amplitude "
    "trend; (5) the mode shape; and (6) a one-line mechanism note. If reference "
    "passages from the literature are provided, you may use them for domain "
    "grounding only -- never let them override the experiment's own measured values."
)

feature_record = """FSI_FEATURE_RECORD v1
experiment_id: my-experiment-001
[conditions]
reynolds_number: 9.775e+04
angle_of_attack_deg: 5.171
freestream_velocity_m_s: 0.3166
reduced_velocity_Ustar: 2.050
chord_m: 0.3088
[piv_derived]
shedding_frequency_hz: 0.5
strouhal_number_St: 0.4877
wake_width_over_c: 1.262
separation_location_x_over_c: 0.533
pod_mode_amplitudes: [0.115, 0.053, 0.035, 0.034]
turbulent_kinetic_energy: 0.1147
[structure]
rms_amplitude_over_c: 0.4909
dominant_frequency_hz: 0.5
natural_frequency_hz: 0.5
mode_number: 1
camber: 0.0275
[forces]
cl_mean: 0.5096
cl_rms: 0.5289
cl_dominant_frequency_hz: 0.5
"""

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": feature_record},
]
ids = tok.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to(model.device)
out = model.generate(ids, max_new_tokens=350, do_sample=False)
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
# -> "Regime: galloping (confidence: 0.97). The wake sheds at a dominant
#     frequency of 0.50 Hz, giving a Strouhal number St = 0.488. ..."
```

The full `FeatureRecord` schema (all fields, valid ranges, units) and the
feature-extraction pipeline (FFT, POD from raw PIV data) are in the GitHub
repo, along with a Gradio UI, RAG grounding over an FSI paper corpus, and an
evaluation harness.

## Usage (MLX, Apple Silicon)

```python
from mlx_lm import load, generate

# merged model
model, tokenizer = load("AngadKumar/fsi-slm-phi4-mini")
# or base + this repo's adapters/
model, tokenizer = load("microsoft/Phi-4-mini-instruct", adapter_path="adapters")
```

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
