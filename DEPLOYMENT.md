# FSI-SLM Deployment Guide

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

### Option 2: Gradio Web UI (Interactive/Research)

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

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 32 GB |
| Disk | 5 GB (model + adapters) | 20 GB |
| CPU | Apple M1 | Apple M4 Pro |
| GPU | Metal-compatible | Metal GPU with 16 GB+ |

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
| Run tests | `pytest tests/ -v` |
| Generate data | `python cli.py generate-data --n-per-regime 40` |
| Train model | `python cli.py train --epochs 3 --batch-size 4` |
| Evaluate | `python cli.py eval` |
| Index papers | `python cli.py index-corpus` |