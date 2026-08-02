"""Check a training run in a minute instead of discovering it in an hour.

A full-resolution run is hours long. Almost everything that kills one is visible
in the first thirty seconds: a cache built under a different channel contract, a
batch size that does not fit, an op the backend does not implement, a device
that silently fell back to CPU. This runs the real model on the real data at the
real resolution for a handful of steps, then reports what the whole run will
cost.

    python cli.py sensor-preflight --batch-size 16 --downsample 1

It deliberately does *not* reuse the training loop. The point is to exercise the
same tensors through the same modules while being able to stop and report at
each stage, which a loop written for throughput cannot do.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import SensorCache, compute_norm_stats, make_torch_dataset
from .model import EMA, LossConfig, UNet, UNetConfig, pick_device
from .train import _Trainer, _downsample_batch


def _fmt_hms(seconds: float) -> str:
    s = int(max(seconds, 0))
    return f"{s // 3600}h{(s % 3600) // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def preflight(
    cache_dir: str = "artifacts/sensor/cache",
    epochs: int = 80,
    batch_size: int = 16,
    base: int = 32,
    depth: int = 4,
    downsample: int = 1,
    device: str = "auto",
    steps: int = 6,
    lr: float = 2.0e-3,
    augment: bool = True,
) -> dict:
    """Time a few real training steps and report what the full run will cost."""
    report: dict = {"checks": []}

    def ok(name, detail=""):
        report["checks"].append({"check": name, "status": "ok", "detail": detail})

    def warn(name, detail):
        report["checks"].append({"check": name, "status": "warning", "detail": detail})

    dev = pick_device(device)
    report["device"] = str(dev)
    if dev.type == "cpu":
        warn("device", "running on CPU. On Apple silicon expect --device auto to "
                       "resolve to 'mps'; if it did not, torch was built without it")
    else:
        ok("device", str(dev))

    # ---- data ---------------------------------------------------------------
    cache = SensorCache(cache_dir)
    spec = cache.spec
    n_done = int(np.asarray(cache.done).sum())
    if n_done < len(cache.cases):
        warn("cache complete",
             f"{n_done} of {len(cache.cases)} rows built; run sensor-cache again")
    else:
        ok("cache complete", f"{n_done} cases")
    if cache.inputs.shape[1] != spec.n_channels:
        raise SystemExit(
            f"cache inputs are {cache.inputs.shape[1]} channels but the index "
            f"records the {spec.name} contract ({spec.n_channels}). Rebuild: "
            f"python cli.py sensor-cache"
        )
    ok("input contract", f"{spec.name}, {spec.n_channels} channels")

    counts = {k: v for k, v in cache.fidelity_counts().items() if v}
    report["fidelity_mix"] = counts
    train_fid = sorted({cache.fidelity_of(int(j)) for j in cache.indices("train")})
    if len(counts) > 1 and train_fid == ["panel"]:
        warn("fidelity mix",
             f"the dataset has {counts} but the train split is {train_fid} only, so "
             f"the fidelity channel has nothing to learn from. See "
             f"`python cli.py sensor-multifidelity`")
    else:
        ok("fidelity mix", str(counts))

    for split in ("train", "val", "test"):
        n = int(cache.indices(split).size)
        report[f"n_{split}"] = n
        if n == 0:
            raise SystemExit(f"split '{split}' is empty -- rebuild the cache")

    t0 = time.time()
    stats = compute_norm_stats(cache, "train")
    ok("normalisation", f"{stats['n_samples']} cases in {time.time() - t0:.1f}s")
    bad = [
        spec.names[i] for i, s in enumerate(stats["x_std"])
        if not math.isfinite(s) or s <= 0
    ]
    if bad:
        raise SystemExit(f"non-positive input scale on channels {bad}")

    train_ds = make_torch_dataset(cache, "train", stats, augment=augment)
    dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    steps_per_epoch = len(dl)
    report["steps_per_epoch"] = steps_per_epoch

    # ---- model --------------------------------------------------------------
    cfg = UNetConfig.for_spec(spec, base=base, depth=depth)
    model = UNet(cfg).to(dev)
    report["params_m"] = round(model.n_params / 1e6, 3)
    res = 128 // downsample
    if res % (2 ** depth):
        warn("resolution",
             f"{res} is not divisible by 2^{depth}; the decoder will resize to "
             f"match the skips, which works but wastes a little")
    ok("model", f"{report['params_m']} M parameters at {res}x{res}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    trainer = _Trainer(cache, stats, LossConfig(), downsample)
    ema = EMA(model, 0.999)

    # ---- steps --------------------------------------------------------------
    it = iter(dl)
    times = []
    parts_last: dict = {}
    for i in range(steps):
        batch = _downsample_batch(next(it), downsample)
        t = time.time()
        x = batch["x"].to(dev)
        y = batch["y"].to(dev)
        m = batch["mask"].to(dev)
        s = batch["s"].to(dev)
        pred, sp = model(x)
        loss, parts_last = trainer(pred, y, m, sp, s, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model)
        if dev.type == "mps":
            torch.mps.synchronize()
        elif dev.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t)
        if i == 0:
            # First step carries kernel compilation and allocator warm-up, and
            # on MPS that is seconds. Reporting it as the step cost would triple
            # the projected run time.
            ok("forward/backward", f"first step {times[0]:.2f}s incl. warm-up")

    if not all(math.isfinite(v) for v in parts_last.values()):
        raise SystemExit(f"non-finite loss on the first steps: {parts_last}")
    ok("loss terms", ", ".join(f"{k} {v:.3f}" for k, v in parts_last.items()))

    step_s = float(np.median(times[1:] or times))
    report["seconds_per_step"] = round(step_s, 3)
    report["seconds_per_epoch"] = round(step_s * steps_per_epoch, 1)

    # One validation pass, through the averaged weights, exactly as training does.
    val_ds = make_torch_dataset(cache, "val", stats)
    val_dl = DataLoader(val_ds, batch_size=batch_size)
    ema_model = UNet(cfg).to(dev)
    ema.copy_to(ema_model)
    t = time.time()
    from .train import evaluate

    evaluate(ema_model, val_dl, dev, trainer, downsample)
    val_s = time.time() - t
    ok("validation pass", f"{val_s:.1f}s over {report['n_val']} cases")

    total = epochs * (report["seconds_per_epoch"] + val_s)
    report["projected"] = {
        "epochs": epochs,
        "per_epoch": _fmt_hms(report["seconds_per_epoch"] + val_s),
        "total": _fmt_hms(total),
        "total_seconds": round(total),
    }
    report["command"] = (
        f"python cli.py sensor-train --epochs {epochs} --batch-size {batch_size} "
        f"--base {base} --depth {depth} --downsample {downsample}"
    )
    return report
