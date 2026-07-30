"""Train the U-Net surrogate. Resumable, device-agnostic, wall-clock aware.

    python -m sensor.train --epochs 60 --batch-size 16

Every run writes ``checkpoint.pt`` after each validation pass and can be
restarted from it, so training survives an interrupted session and can be
spread across short windows. ``--time-budget-s`` stops cleanly at a deadline
rather than being killed mid-epoch.

``--downsample 2`` trains at 64 x 64 by average-pooling the cached 128 x 128
fields. Same physics, a quarter of the pixels, roughly four times the
throughput -- the right setting for a CPU-only smoke run or a quick sweep. The
released full-resolution model uses ``--downsample 1``.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import SensorCache, compute_norm_stats, make_torch_dataset
from .model import UNet, UNetConfig, masked_loss, pick_device

DEFAULT_RUN_DIR = "artifacts/sensor/run"


def _cache_fingerprint(cache_dir: str) -> str:
    """Cheap identity for the memmap cache: sizes and modification times."""
    import hashlib

    h = hashlib.sha256()
    for name in ("targets.npy", "scalars.npy", "inputs.npy", "index.json"):
        p = Path(cache_dir) / name
        if p.exists():
            st = p.stat()
            h.update(f"{name}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:16]


def _downsample_batch(batch: dict, factor: int) -> dict:
    if factor <= 1:
        return batch
    out = dict(batch)
    out["x"] = F.avg_pool2d(batch["x"], factor)
    out["y"] = F.avg_pool2d(batch["y"], factor)
    # Masks must stay binary and must not invent fluid inside the body: take the
    # strict minimum over each pooling window (a cell is fluid only if all of
    # its sub-cells were).
    out["mask"] = -F.max_pool2d(-batch["mask"], factor)
    return out


def evaluate(model, loader, device, downsample: int = 1) -> dict:
    model.eval()
    agg = {"loss": 0.0, "flow": 0.0, "struct": 0.0, "scalar": 0.0}
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = _downsample_batch(batch, downsample)
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            m = batch["mask"].to(device)
            s = batch["s"].to(device)
            pred, sp = model(x)
            _, parts = masked_loss(pred, y, m, sp, s)
            bs = x.shape[0]
            for k in agg:
                agg[k] += parts[k] * bs
            n += bs
    model.train()
    return {k: v / max(n, 1) for k, v in agg.items()}


def train(
    cache_dir: str = "artifacts/sensor/cache",
    run_dir: str = DEFAULT_RUN_DIR,
    epochs: int = 60,
    batch_size: int = 16,
    lr: float = 2.0e-3,
    weight_decay: float = 1.0e-5,
    base: int = 24,
    depth: int = 4,
    downsample: int = 1,
    device: str = "auto",
    num_workers: int = 0,
    time_budget_s: Optional[float] = None,
    resume: bool = True,
    seed: int = 0,
    log_every: int = 25,
    threads: Optional[int] = None,
    fidelity_weights: Optional[dict] = None,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if threads:
        # Default intra-op parallelism is one thread per *physical* core, which
        # a container's cgroup limit does not always report correctly.
        torch.set_num_threads(threads)
    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    dev = pick_device(device)

    cache = SensorCache(cache_dir)
    # Normalisation is cached in the run directory, but it must be invalidated
    # when the dataset underneath changes -- reusing statistics computed against
    # different targets silently rescales the loss (a stale set of scalar
    # standard deviations here sent the first epoch's loss to ~1000).
    stats_path = run / "norm_stats.json"
    fingerprint = _cache_fingerprint(cache_dir)
    stats = None
    if stats_path.exists():
        cached = json.loads(stats_path.read_text())
        if cached.get("fingerprint") == fingerprint:
            stats = cached
        else:
            print("dataset changed -- recomputing normalisation statistics", flush=True)
    if stats is None:
        stats = compute_norm_stats(cache, "train")
        stats["fingerprint"] = fingerprint
        stats_path.write_text(json.dumps(stats, indent=2))

    train_ds = make_torch_dataset(cache, "train", stats, fidelity_weights)
    # Validation is never re-weighted: a validation number has to stay
    # comparable across runs with different weighting.
    val_ds = make_torch_dataset(cache, "val", stats)
    counts = cache.fidelity_counts()
    if len(counts) > 1 or fidelity_weights:
        print(f"fidelity mix: {counts}  weights: {fidelity_weights or 'uniform'}",
              flush=True)
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers,
    )
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)

    cfg = UNetConfig(base=base, depth=depth)
    model = UNet(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = max(1, epochs * max(1, len(train_dl)))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=total_steps, pct_start=0.15
    )

    ckpt_path = run / "checkpoint.pt"
    start_epoch = 0
    history = []
    best = math.inf
    if resume and ckpt_path.exists():
        ck = torch.load(
            ckpt_path,
            map_location=dev,
            weights_only=False,
        )
        if ck.get("config") == cfg.to_dict() and ck.get("downsample") == downsample:
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["optimizer"])
            try:
                sched.load_state_dict(ck["scheduler"])
            except Exception:
                pass  # schedule length changed; keep the fresh one
            start_epoch = ck["epoch"] + 1
            history = ck.get("history", [])
            best = ck.get("best", math.inf)
            print(f"resumed from epoch {start_epoch}", flush=True)
        else:
            print("checkpoint config differs -- starting fresh", flush=True)

    t0 = time.time()
    stopped_early = False
    for epoch in range(start_epoch, epochs):
        run_agg = {"loss": 0.0, "flow": 0.0, "struct": 0.0, "scalar": 0.0}
        seen = 0
        for step, batch in enumerate(train_dl):
            if time_budget_s is not None and time.time() - t0 > time_budget_s:
                stopped_early = True
                break
            batch = _downsample_batch(batch, downsample)
            x = batch["x"].to(dev)
            y = batch["y"].to(dev)
            m = batch["mask"].to(dev)
            s = batch["s"].to(dev)

            w = batch["weight"].to(dev) if fidelity_weights else None
            pred, sp = model(x)
            loss, parts = masked_loss(pred, y, m, sp, s, sample_weight=w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched.last_epoch < total_steps - 1:
                sched.step()

            bs = x.shape[0]
            for k in run_agg:
                run_agg[k] += parts[k] * bs
            seen += bs
            if log_every and step % log_every == 0:
                print(
                    f"  e{epoch:03d} s{step:04d} loss {parts['loss']:.4f} "
                    f"flow {parts['flow']:.4f} struct {parts['struct']:.4f} "
                    f"scalar {parts['scalar']:.4f}",
                    flush=True,
                )

        train_metrics = {k: v / max(seen, 1) for k, v in run_agg.items()}
        val_metrics = evaluate(model, val_dl, dev, downsample)
        history.append({
            "epoch": epoch, "train": train_metrics, "val": val_metrics,
            "lr": sched.get_last_lr()[0],
        })
        print(
            f"epoch {epoch:03d} train {train_metrics['loss']:.4f} "
            f"val {val_metrics['loss']:.4f}",
            flush=True,
        )

        # Update `best` *before* writing the rolling checkpoint. Otherwise the
        # checkpoint stores the previous best, and a resumed run starts with a
        # stale (too high) bar -- so a worse epoch can overwrite best.pt.
        improved = val_metrics["loss"] < best
        if improved:
            best = val_metrics["loss"]

        state = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "epoch": epoch,
            "config": cfg.to_dict(),
            "downsample": downsample,
            "stats": stats,
            "history": history,
            "best": best,
            "scalar_names": cache.scalar_names,
        }
        torch.save(state, ckpt_path)
        if improved:
            torch.save(state, run / "best.pt")
        (run / "history.json").write_text(json.dumps(history, indent=2))

        if stopped_early:
            break

    return {
        "run_dir": str(run),
        "epochs_done": len(history),
        "best_val_loss": best,
        "params_m": round(model.n_params / 1e6, 3),
        "device": str(dev),
        "stopped_early": stopped_early,
        "seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", default="artifacts/sensor/cache")
    p.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--base", type=int, default=24)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--downsample", type=int, default=1)
    p.add_argument("--device", default="auto")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--time-budget-s", type=float, default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument(
        "--fidelity-weights", default=None,
        help='per-fidelity loss weights, e.g. "panel=1,rans=10"',
    )
    args = p.parse_args()
    weights = None
    if args.fidelity_weights:
        weights = {
            k.strip(): float(v)
            for k, v in (part.split("=") for part in args.fidelity_weights.split(","))
        }
    out = train(
        cache_dir=args.cache_dir, run_dir=args.run_dir, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, base=args.base, depth=args.depth,
        downsample=args.downsample, device=args.device, num_workers=args.num_workers,
        time_budget_s=args.time_budget_s, resume=not args.no_resume,
        threads=args.threads, log_every=args.log_every,
        fidelity_weights=weights,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
