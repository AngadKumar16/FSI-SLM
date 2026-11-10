# FSI-SLM Deployment Guide

## What you are deploying

The system has two independently-loadable models, and which one a request needs
decides its cost. Deploy accordingly.

| Model | Size | Loads when | Cost |
|---|---|---|---|
| Phi-4-mini-instruct + LoRA adapter | ~7.7 GB base + 46 MB adapter | first `/infer`, `/infer/stream`, or twin call with `--interpret` | ~30 s cold, ~3 s per request |
| U-Net surrogate | 11.6 M params, ~45 MB | first `/sensor/*` or `/twin` call | ~1 s cold, milliseconds per request |
| MiniLM-L6-v2 embedder + FAISS index | ~90 MB + 2.8 MB | first RAG-enabled request | ~2 s cold |

**The surrogate endpoints do not need the language model.** A deployment that
only serves `/sensor/predict` and `/sensor/polar` needs neither MLX nor the
7.7 GB base model and will run comfortably in 2 GB of RAM on any platform. Split
these into separate services if you can — it is the single biggest cost lever
here.

## Deployment Options

### Option 1: FastAPI REST API (Recommended for Production)

#### Local Deployment

**Basic launch:**
```bash
python cli.py serve --port 8000
```

**Production launch with Uvicorn:**
```bash
uvicorn serve.fastapi_app:create_app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1 \
  --log-level info
```

**With custom config:**
```bash
uvicorn serve.fastapi_app:create_app \
  --host 0.0.0.0 \
  --port 8000 \
  --app-dir . \
  --factory
```

#### Docker Deployment (Apple Silicon)

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run server
CMD ["uvicorn", "serve.fastapi_app:create_app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build and run:**
```bash
docker build -t fsi-slm .
docker run -p 8000:8000 fsi-slm
```

**With Docker Compose:**
```yaml
version: '3.8'
services:
  fsi-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MLX_LAZY_INIT=1
    volumes:
      - ./artifacts:/app/artifacts
    deploy:
      resources:
        limits:
          memory: 8g
```

Run: `docker-compose up -d`

---

### Option 2: MCP Server (Agentic Workflows)

The MCP server speaks stdio, so it is *launched by the client*, not run as a
long-lived service. There is no port and no reverse proxy — the client owns the
process lifetime.

```bash
python cli.py mcp          # or: python -m serve.mcp_server
```

Claude Code picks it up automatically from the repo's `.mcp.json` for sessions
opened in this directory. For Claude Desktop, add to
`claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "fsi-slm": {
      "command": "python3",
      "args": ["-m", "serve.mcp_server"],
      "cwd": "/absolute/path/to/FSI-SLM"
    }
  }
}
```

Three deployment notes specific to MCP:

- **`cwd` must be absolute and must be the repo root.** Every default path in
  `configs/default.yaml` is repo-relative; the server resolves them against the
  working directory it is launched in.
- **Model loading is lazy and per-tool.** `search_literature`,
  `validate_interpretation`, `check_experiment_consistency`, `sweep_regime_map`,
  `generate_synthetic_experiment`, `check_data_quality` and `surrogate_status`
  never load the language model. Only `interpret_experiment` and `ask_question`
  do, and the first such call costs ~30 s. Clients with short tool timeouts
  should call `surrogate_status` first to warm the process.
- **stdout belongs to the protocol.** Anything printed to stdout corrupts the
  JSON-RPC stream. Log to stderr or to a file.

---

### Option 3: Gradio Web UI (Interactive/Research)

#### Local Launch
```bash
# Default (http://127.0.0.1:7860)
python cli.py ui

# Custom port
python cli.py ui --port 8080

# Public share link (temporary)
python cli.py ui --share

# Bind to all interfaces
python cli.py ui --host 0.0.0.0 --port 7860
```

#### Gradio with Nginx Reverse Proxy

```nginx
# /etc/nginx/sites-available/fsi-slm
server {
    listen 80;
    server_name fsi.example.com;

    location / {
        proxy_pass http://127.0.0.1:7860;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }
}
```

Enable: `ln -s /etc/nginx/sites-available/fsi-slm /etc/nginx/sites-enabled/`

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MLX_LAZY_INIT` | Defer MLX initialization | `1` |
| `PYTHONHASHSEED` | Reproducibility | `42` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |

**Example:**
```bash
export MLX_LAZY_INIT=1
export PYTHONHASHSEED=42
python cli.py serve --port 8000
```

---

## Configuration

Override defaults in `configs/default.yaml` or via CLI:

**Custom adapter path:**
```bash
python cli.py serve --port 8000 --adapter-path /models/my_adapter
```

**Custom model:**
```bash
python cli.py serve --port 8000 --model-name ./artifacts/models/Phi-4-mini-instruct
```

**Enable RAG with custom index:**
```bash
python cli.py serve --port 8000 --rag-index-dir ./artifacts/rag_index --rag-top-k 5
```

---

## Testing Deployment

**Health check:**
```bash
curl http://localhost:8000/health
# Expected: {"status":"ok","model_loaded":true}
```

**Get schema:**
```bash
curl http://localhost:8000/schema | jq
```

**All 8 routes:**

| Route | Needs SLM | Needs U-Net |
|---|---|---|
| `GET /health` | no | no |
| `GET /schema` | no | no |
| `POST /infer` | **yes** | no |
| `POST /infer/stream` | **yes** | no |
| `GET /sensor/status` | no | no (reads checkpoint metadata) |
| `POST /sensor/predict` | no | **yes** |
| `POST /sensor/polar` | no | **yes** |
| `POST /twin` | only with `interpret: true` | **yes** |

**Surrogate prediction (no language model loaded):**
```bash
curl -X POST http://localhost:8000/sensor/predict \
  -H "Content-Type: application/json" \
  -d '{"airfoil": "Data/Airfoils/4/n0009sm.dat.txt",
       "alpha_deg": 12,
       "reynolds": 5.0e5,
       "return_fields": false}' | jq
```

**Polar sweep** (`alphas` defaults to −4…16 in 2° steps):
```bash
curl -X POST http://localhost:8000/sensor/polar \
  -H "Content-Type: application/json" \
  -d '{"airfoil": "Data/Airfoils/4/n0009sm.dat.txt", "alphas": [0, 4, 8, 12, 16]}' | jq
```

**Full twin — geometry to interpreted experiment:**
```bash
curl -X POST http://localhost:8000/twin \
  -H "Content-Type: application/json" \
  -d '{"airfoil": "Data/Airfoils/4/n0009sm.dat.txt",
       "alpha_deg": 14,
       "chord_m": 0.1,
       "span_m": 0.3,
       "u_inf_m_s": 5.0,
       "use_surrogate": true,
       "interpret": true,
       "use_rag": false}' | jq
```

Set `return_fields: true` on `/sensor/predict` only when you need the rasters —
the response then carries five 2-D arrays. `field_downsample` (default 4)
average-pools them before serialising, so leaving it at the default returns
32 × 32 rather than 128 × 128 and cuts the payload by 16×. Set it to 1 only if
you actually need full resolution over the wire.

**Run inference:**
```bash
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{
    "feature_record": {
      "experiment_id": "test_001",
      "conditions": {
        "reynolds_number": 150000,
        "angle_of_attack_deg": 8.0,
        "freestream_velocity_m_s": 2.0,
        "reduced_velocity": 6.0,
        "chord_m": 0.1
      },
      "piv": {
        "shedding_frequency_hz": 12.0,
        "strouhal_number": 0.6,
        "wake_width_over_c": 0.8,
        "separation_location_x_over_c": 0.6,
        "pod_mode_amplitudes": [0.6, 0.2, 0.1],
        "turbulent_kinetic_energy": 0.05
      },
      "structure": {
        "rms_amplitude_over_c": 0.3,
        "dominant_frequency_hz": 12.0,
        "natural_frequency_hz": 12.0,
        "mode_number": 1,
        "camber": 0.02
      },
      "forces": {
        "cl_mean": 0.8,
        "cl_rms": 0.4,
        "cl_dominant_frequency_hz": 12.0,
        "cd_mean": 0.2,
        "cd_rms": 0.1,
        "cd_dominant_frequency_hz": 24.0,
        "cm_mean": -0.05,
        "cm_rms": 0.05,
        "cm_dominant_frequency_hz": 12.0
      }
    },
    "use_rag": false
  }' | jq
```

---

## Cloud Deployment

### AWS/GCP/Azure (x86_64)

**Note:** MLX requires Apple Silicon. For cloud x86_64:
1. Train on Apple Silicon locally
2. Export adapters to safetensors
3. Deploy with PyTorch + `mlx-lm` CPU fallback

**Use `inference/runner.py` with PyTorch backend:**
```python
# Modify InferenceRunner to use transformers + PyTorch
from transformers import AutoModelForCausalLM, AutoTokenizer
```

### Hugging Face Spaces (Gradio)

Create `app.py`:
```python
from serve.gradio_app import create_ui
import yaml

with open("configs/default.yaml") as f:
    config = yaml.safe_load(f)

demo = create_ui(config=config)
demo.launch()
```

Deploy to HF Spaces with `Gradio` SDK.

---

## Performance Tuning

### Apple Silicon Optimization

```bash
# Enable Metal GPU
export PYTORCH_ENABLE_MPS_FALLBACK=1

# MLX optimization
export MLX_LAZY_INIT=1
export MLX_GPU_BUFFER_CACHE_SIZE=4096

# Launch
python cli.py serve --port 8000
```

### Resource Requirements

Scale to what the deployment actually serves:

| Deployment | RAM | Disk | Platform |
|---|---|---|---|
| Surrogate only (`/sensor/*`) | 2 GB | ~200 MB | any — CPU, CUDA or MPS |
| + RAG search (`search_literature`) | 3 GB | ~300 MB | any |
| Full stack (SLM interpretation) | 8 GB min, 16 GB recommended | ~10 GB | Apple Silicon for MLX |
| Training the surrogate | 8 GB | +1.1 GB cache, +14 GB `.npz` | any; ~2 h on M4 Pro CPU |
| Generating RANS/URANS data | 16 GB | +20 GB per sweep | needs OpenFOAM v2606 + gmsh |

The 7.7 GB base language model dominates the full-stack figure. If you are
deploying the surrogate as a service — which is the common case, since it is the
part that replaces a CFD solve — you do not need it, and you are not tied to
Apple Silicon.

---

## Monitoring

**Enable Prometheus metrics (optional):**
```bash
pip install prometheus-fastapi-instrumentator
```

Add to `serve/fastapi_app.py`:
```python
from prometheus_fastapi_instrumentator import Instrumentator

@app.on_event("startup")
async def startup():
    Instrumentator().instrument(app).expose(app)
```

Access metrics: `http://localhost:8000/metrics`

---

## Troubleshooting

### Model Loading Errors

```bash
# Verify model exists
ls -lh artifacts/models/Phi-4-mini-instruct/

# Verify adapters exist
ls -lh artifacts/adapters/
```

### Surrogate Refuses a Checkpoint

`SurrogatePredictor` validates a checkpoint by name and input spec rather than
failing obscurely later. Two common causes:

- **Architecture mismatch.** A checkpoint from an older architecture will not
  load into the current one. Retrain, or point `--checkpoint` at a current run
  directory.
- **Cache/spec mismatch.** The checkpoint records its `input_spec` and a
  fingerprint of the dataset it was normalised against. If the cache was built
  with a different channel contract (`v2` vs `v2_noprior`), training and eval
  refuse it with a message. Rebuild: `python cli.py sensor-cache`.

Check what a checkpoint expects without loading a model:

```bash
curl http://localhost:8000/sensor/status | jq
```

### Surrogate Returns an Extrapolation Warning

Asking for `fidelity="rans"` or `"urans"` from a checkpoint trained only on
panel rows returns the prediction *plus* a warning that the conditioning channel
is being extrapolated. This is by design, not a fault — the released checkpoint
is panel-trained. See `docs/implementation_plan.md` §3.2.

### Port Already in Use

```bash
# Find process
lsof -i :8000

# Kill process
kill -9 <PID>

# Or use different port
python cli.py serve --port 8001
```

### Memory Issues

```bash
# Reduce batch size in config
# configs/default.yaml
training:
  batch_size: 2  # Reduced from 4

# Or limit cache
export MLX_GPU_BUFFER_CACHE_SIZE=2048
```

---

## Production Checklist

- [ ] Set `MLX_LAZY_INIT=1` to defer initialization
- [ ] Use `uvicorn` with 1 worker (MLX is not fork-safe)
- [ ] Mount `artifacts/` directory for model persistence
- [ ] Set `PYTHONHASHSEED=42` for reproducibility
- [ ] Configure health check endpoint
- [ ] Enable logging to file
- [ ] Set up reverse proxy (Nginx/Caddy)
- [ ] Configure TLS/SSL termination
- [ ] Monitor memory usage (MLX models are large)
- [ ] Backup adapter weights regularly

---
 
## Quick Reference

| Task | Command |
|------|---------|
| Start API server | `python cli.py serve --port 8000` |
| Start Gradio UI | `python cli.py ui` |
| Start MCP server | `python cli.py mcp` |
| Run tests (266) | `pytest tests/ -q` |
| Generate SFT data | `python cli.py generate-data --n-per-regime 40` |
| Train interpreter | `python cli.py train --epochs 3 --batch-size 4` |
| Evaluate interpreter | `python cli.py eval --use-model` |
| Index papers | `python cli.py index-corpus` |
| Generate surrogate set | `python cli.py sensor-generate` |
| Cache for training | `python cli.py sensor-cache` |
| Train surrogate | `python cli.py sensor-train --epochs 80 --batch-size 16` |
| Score surrogate | `python cli.py sensor-eval --split test --fidelity panel` |
| One-shot prediction | `python cli.py sensor-predict --airfoil <path> --alpha 12` |
| Full twin | `python cli.py twin --airfoil <path> --alpha 12 --interpret` |
| No-install demo | `python cli.py demo` |

Two shell traps worth knowing, both silent:

- Interactive zsh does not treat `#` as a comment, so a trailing note on a
  `cli.py` command arrives as positional arguments. `cli.py` detects and rejects
  this rather than binding it to a parameter.
- Negative numbers need `--alphas="-8,-4,0"` — with the `=` and the quotes. A
  bare leading minus is parsed as a flag.