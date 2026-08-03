"""Train the U-Net surrogate. Resumable, device-agnostic, wall-clock aware.

    python -m sensor.train --epochs 80 --batch-size 16

Every run writes ``checkpoint.pt`` after each validation pass and can be
restarted from it, so training survives an interrupted session and can be
spread across short windows. ``--time-budget-s`` stops cleanly at a deadline
rather than being killed mid-epoch.

``--downsample 2`` trains at 64 x 64 by average-pooling the cached 128 x 128
fields. Same physics, a quarter of the pixels, roughly four times the
throughput -- the right setting for a CPU-only smoke run or a quick sweep. The
released full-resolution model uses ``--downsample 1``.

Multi-fidelity
--------------

Two mechanisms, and they compose:

``--fidelity-weights "panel=1,rans=10"``
    Per-sample loss weight. A hundred RANS cases against 1,900 panel cases are
    5% of the gradient at equal weight, which does not move the network off the
    cheap targets.

*fidelity conditioning* (automatic with the ``v2`` input spec)
    The fidelity code is an input channel, so one network learns both solvers
    and can be *asked* which to emit. That is strictly more useful than a
    weighted average of the two, and it lets the 1,900 cheap cases teach the
    shared structure while the expensive ones only have to teach the difference.

``--init-from <checkpoint>`` starts from trained weights instead of noise, which
is how the two-stage recipe runs: pretrain on panel data, then fine-tune on the
mixed set at a tenth of the learning rate.
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
from .model import (
    EMA,
    LossConfig,
    UNet,
    UNetConfig,
    load_checkpoint,
    pick_device,
    surrogate_loss,
)
from .sdf import Grid

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


class _Trainer:
    """Holds the pieces the loss needs that are not in the batch itself."""

    def __init__(self, cache: SensorCache, stats: dict, loss_cfg: LossConfig,
                 downsample: int, grid: Optional[Grid] = None):
        grid = grid or Grid()
        self.loss_cfg = loss_cfg
        self.y_std = list(stats["y_std"])
        self.s_std = list(stats["s_std"])
        self.dx = grid.dx * downsample
        self.dy = grid.dy * downsample
        spec = cache.spec
        self.wall_idx = spec.index("wall") if "wall" in spec.names else -1
        if self.wall_idx >= 0:
            self.wall_mean = float(stats["x_mean"][self.wall_idx])
            self.wall_std = float(stats["x_std"][self.wall_idx])

    def wall_weight(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """The physical near-wall channel, recovered from the normalised input.

        The loss wants a weight in [0, 1]; the network is fed a standardised
        channel that is mostly negative. Denormalising is exact and costs two
        scalars, where clamping the normalised channel would silently change the
        band width with the dataset statistics.
        """
        if self.wall_idx < 0:
            return None
        w = x[:, self.wall_idx: self.wall_idx + 1] * self.wall_std + self.wall_mean
        return w.clamp(0.0, 1.0)

    def __call__(self, pred, y, mask, sp, s, x, sample_weight=None):
        return surrogate_loss(
            pred, y, mask, sp, s,
            cfg=self.loss_cfg,
            sample_weight=sample_weight,
            wall=self.wall_weight(x),
            y_std=self.y_std,
            s_std=self.s_std,
            dx=self.dx,
            dy=self.dy,
        )


def evaluate(model, loader, device, trainer: _Trainer, downsample: int = 1) -> dict:
    model.eval()
    agg: dict = {}
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = _downsample_batch(batch, downsample)
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            m = batch["mask"].to(device)
            s = batch["s"].to(device)
            pred, sp = model(x)
            _, parts = trainer(pred, y, m, sp, s, x)
            bs = x.shape[0]
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v * bs
            # Reported separately from the weighted loss: this is the number to
            # compare across runs with different loss weights.
            mae = float((sp - s).abs().mean().detach())
            agg["scalar_mae"] = agg.get("scalar_mae", 0.0) + mae * bs
            n += bs
    model.train()
    return {k: v / max(n, 1) for k, v in agg.items()}


def train(
    cache_dir: str = "artifacts/sensor/cache",
    run_dir: str = DEFAULT_RUN_DIR,
    epochs: int = 80,
    batch_size: int = 16,
    lr: float = 2.0e-3,
    weight_decay: float = 1.0e-5,
    base: int = 32,
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
    augment: bool = True,
    ema_decay: float = 0.999,
    attention: bool = True,
    film: bool = True,
    dropout: float = 0.0,
    accum_steps: int = 1,
    warmup_frac: float = 0.05,
    init_from: Optional[str] = None,
    loss_cfg: Optional[LossConfig] = None,
    preload: bool | str = "auto",
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
    loss_cfg = loss_cfg or LossConfig()

    # Shuffled access to a memmapped 821 MB cache is ~1.8 GB of random page
    # faults per epoch on one thread, and it was 95% of the wall clock: a
    # 64 x 64 epoch that computes in nine seconds took two hundred and
    # twenty-four. The arrays are 1.2 GB total.
    cache = SensorCache(cache_dir, preload=preload)
    spec = cache.spec
    # Normalisation is cached in the run directory, but it must be invalidated
    # when the dataset underneath changes -- reusing statistics computed against
    # different targets silently rescales the loss (a stale set of scalar
    # standard deviations here sent the first epoch's loss to ~1000).
    stats_path = run / "norm_stats.json"
    fingerprint = _cache_fingerprint(cache_dir)
    stats = None
    if stats_path.exists():
        cached = json.loads(stats_path.read_text())
        if cached.get("fingerprint") == fingerprint and \
                len(cached.get("x_mean", [])) == spec.n_channels:
            stats = cached
        else:
            print("dataset changed -- recomputing normalisation statistics", flush=True)
    if stats is None:
        stats = compute_norm_stats(cache, "train")
        stats["fingerprint"] = fingerprint
        stats_path.write_text(json.dumps(stats, indent=2))

    train_ds = make_torch_dataset(
        cache, "train", stats, fidelity_weights, augment=augment, seed=seed
    )
    # Validation is never re-weighted and never augmented: a validation number
    # has to stay comparable across runs.
    val_ds = make_torch_dataset(cache, "val", stats)
    # Which fidelities the network actually *sees*. Asking a checkpoint for RANS
    # output when no RANS row was in its training split is extrapolation on the
    # conditioning channel, and predict.py says so rather than guessing.
    train_fidelities = sorted(
        {cache.fidelity_of(int(j)) for j in cache.indices("train")}
    )
    counts = cache.fidelity_counts()
    if len(counts) > 1 or fidelity_weights:
        print(f"fidelity mix: {counts}  weights: {fidelity_weights or 'uniform'}",
              flush=True)
    if num_workers:
        # The dataset is a closure over the memmap cache and the normalisation
        # tensors, defined inside `make_torch_dataset`. macOS starts worker
        # processes with `spawn`, which pickles the dataset by reference, and a
        # class defined inside a function has no importable path -- the loader
        # dies with "Can't pickle local object". It would not buy much anyway:
        # a batch is a page-cache read and a normalisation, and the GPU is the
        # bottleneck.
        raise ValueError(
            "num_workers must be 0. The cache dataset is a local class and "
            "cannot be pickled for spawned workers; loading is page-cache reads "
            "and is not the bottleneck."
        )
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers,
    )
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)

    cfg = UNetConfig.for_spec(
        spec, base=base, depth=depth, attention=attention, film=film, dropout=dropout
    )
    model = UNet(cfg).to(dev)
    if init_from:
        loaded = _load_weights(model, init_from, dev)
        print(f"initialised {loaded} tensors from {init_from}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, math.ceil(max(1, len(train_dl)) / max(accum_steps, 1)))
    total_steps = max(1, epochs * steps_per_epoch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=total_steps, pct_start=warmup_frac
    )
    ema = EMA(model, ema_decay) if ema_decay and ema_decay > 0 else None
    trainer = _Trainer(cache, stats, loss_cfg, downsample)

    ckpt_path = run / "checkpoint.pt"
    start_epoch = 0
    history = []
    best = math.inf
    if resume and ckpt_path.exists():
        ck = load_checkpoint(ckpt_path, map_location=dev)
        if ck.get("config") == cfg.to_dict() and ck.get("downsample") == downsample:
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["optimizer"])
            if ema is not None and ck.get("ema"):
                ema.shadow = {k: v.to(dev).float() for k, v in ck["ema"].items()
                              if v.dtype.is_floating_point}
                ema.step = ck.get("ema_step", 0)
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

    # Allocated once. Rebuilding an 11 M-parameter module every epoch just to
    # hold the averaged weights churns memory for no reason.
    ema_model = UNet(cfg).to(dev) if ema is not None else None

    t0 = time.time()
    stopped_early = False
    for epoch in range(start_epoch, epochs):
        run_agg: dict = {}
        seen = 0
        opt.zero_grad(set_to_none=True)
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
            loss, parts = trainer(pred, y, m, sp, s, x, sample_weight=w)
            (loss / max(accum_steps, 1)).backward()

            if (step + 1) % max(accum_steps, 1) == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)
                if sched.last_epoch < total_steps - 1:
                    sched.step()

            bs = x.shape[0]
            for k, v in parts.items():
                run_agg[k] = run_agg.get(k, 0.0) + v * bs
            seen += bs
            if log_every and step % log_every == 0:
                bits = " ".join(f"{k} {v:.4f}" for k, v in parts.items())
                print(f"  e{epoch:03d} s{step:04d} {bits}", flush=True)

        train_metrics = {k: v / max(seen, 1) for k, v in run_agg.items()}

        # Validate the averaged weights, which is what will be released. A raw
        # SGD iterate on a 16-case batch is a noisier estimate of the same model
        # and picking "best" on it selects for lucky steps.
        eval_model = model
        if ema is not None:
            ema.copy_to(ema_model)
            eval_model = ema_model
        val_metrics = evaluate(eval_model, val_dl, dev, trainer, downsample)
        history.append({
            "epoch": epoch, "train": train_metrics, "val": val_metrics,
            "lr": sched.get_last_lr()[0],
        })
        print(
            f"epoch {epoch:03d} train {train_metrics['loss']:.4f} "
            f"val {val_metrics['loss']:.4f} "
            f"(flow {val_metrics['flow']:.4f} scalar_mae {val_metrics['scalar_mae']:.4f})",
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
            "ema": ema.state_dict() if ema is not None else None,
            "ema_step": ema.step if ema is not None else 0,
            # What predict.py should load: the averaged weights when they exist.
            "weights": (ema.state_dict() if ema is not None else model.state_dict()),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "epoch": epoch,
            "config": cfg.to_dict(),
            "input_spec": spec.to_dict(),
            "loss_config": loss_cfg.to_dict(),
            "fidelities": train_fidelities,
            "augment": augment,
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
        "input_spec": spec.name,
        "channels": spec.n_channels,
        "augment": augment,
        "device": str(dev),
        "stopped_early": stopped_early,
        "seconds": round(time.time() - t0, 1),
    }


def _load_weights(model: UNet, path: str, device) -> int:
    """Load whatever of ``path`` fits ``model``. Returns the number of tensors.

    Shapes legitimately differ between stages -- a fine-tune may widen the
    trunk, and the first convolution changes width whenever the input spec does
    -- so a strict load would refuse exactly the transfers this exists for.
    Mismatches are reported, not silently dropped.
    """
    ck = load_checkpoint(path, map_location=device)
    src = ck.get("weights") or ck.get("model") or ck
    own = model.state_dict()
    take, skip = {}, []
    for k, v in src.items():
        if k in own and own[k].shape == v.shape:
            take[k] = v
        else:
            skip.append(k)
    model.load_state_dict(take, strict=False)
    if skip:
        print(f"  skipped {len(skip)} tensor(s) with no shape match, e.g. {skip[:3]}",
              flush=True)
    return len(take)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", default="artifacts/sensor/cache")
    p.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--downsample", type=int, default=1)
    p.add_argument("--device", default="auto")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--time-budget-s", type=float, default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--no-augment", action="store_true",
                   help="disable the exact y-mirror symmetry augmentation")
    p.add_argument("--no-attention", action="store_true")
    p.add_argument("--no-film", action="store_true")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--init-from", default=None,
                   help="checkpoint to initialise from (two-stage fine-tuning)")
    p.add_argument("--no-preload", action="store_true",
                   help="memory-map the cache instead of loading it into RAM. "
                        "Costs roughly 25x on epoch time; only for a box that "
                        "cannot spare 1.2 GB")
    p.add_argument(
        "--fidelity-weights", default=None,
        help='per-fidelity loss weights, e.g. "panel=1,rans=10"',
    )
    for name, default in (
        ("gradient", LossConfig.gradient), ("divergence", LossConfig.divergence),
        ("circulation", LossConfig.circulation), ("near-wall", LossConfig.near_wall),
        ("scalar", LossConfig.scalar),
    ):
        p.add_argument(f"--w-{name}", type=float, default=default,
                       help=f"loss weight for the {name.replace('-', ' ')} term")
    p.add_argument("--scalar-loss", choices=("huber", "mse"),
                   default=LossConfig.scalar_loss,
                   help="'huber' clips outlier gradients; 'mse' does not. A "
                        "heavy-tailed target such as the trailing-edge twist "
                        "needs the tails")
    p.add_argument("--huber-delta", type=float, default=LossConfig.huber_delta)
    p.add_argument("--scalar-weights", default=None,
                   help="comma-separated weights over cl,cd,cm,w_max,twist_te_deg")
    p.add_argument("--channel-weights", default=None,
                   help="comma-separated weights over u,v,p,load,disp")
    args = p.parse_args()
    weights = None
    if args.fidelity_weights:
        weights = {
            k.strip(): float(v)
            for k, v in (part.split("=") for part in args.fidelity_weights.split(","))
        }
    _weights = lambda s, n: (
        tuple(float(v) for v in s.split(",")) if s else None
    )
    loss_cfg = LossConfig(
        gradient=args.w_gradient, divergence=args.w_divergence,
        circulation=args.w_circulation, near_wall=args.w_near_wall,
        scalar=args.w_scalar, scalar_loss=args.scalar_loss,
        huber_delta=args.huber_delta,
        **{k: v for k, v in (
            ("scalar_weights", _weights(args.scalar_weights, 5)),
            ("channel_weights", _weights(args.channel_weights, 5)),
        ) if v is not None},
    )
    for name, w in (("scalar_weights", loss_cfg.scalar_weights),
                    ("channel_weights", loss_cfg.channel_weights)):
        if len(w) != 5:
            raise SystemExit(f"--{name.replace('_', '-')} needs 5 values, got {len(w)}")
    out = train(
        cache_dir=args.cache_dir, run_dir=args.run_dir, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, base=args.base, depth=args.depth,
        downsample=args.downsample, device=args.device, num_workers=args.num_workers,
        time_budget_s=args.time_budget_s, resume=not args.no_resume,
        threads=args.threads, log_every=args.log_every,
        fidelity_weights=weights, augment=not args.no_augment,
        attention=not args.no_attention, film=not args.no_film,
        dropout=args.dropout, ema_decay=args.ema_decay,
        accum_steps=args.accum_steps, init_from=args.init_from,
        loss_cfg=loss_cfg, preload=not args.no_preload,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
