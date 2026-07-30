"""U-Net surrogate: geometry + flow condition -> fields and force coefficients.

Architecture follows the encoder-decoder used for airfoil flow prediction by
Thuerey et al. (2020) and Bhatnagar et al. (2019), with three changes that the
FSI task requires:

1. **Two output regions.** Channels 0-2 (u, v, p) live in the fluid, channels
   3-4 (load, displacement) live inside the solid. One decoder produces all
   five; the masked loss routes supervision to the right region. Sharing the
   trunk is deliberate -- the load field *is* the surface pressure, so the
   structural head should reuse the same features rather than relearn them.

2. **A scalar head.** Integrated Cl, Cd, Cm, peak deflection and trailing-edge
   twist are read from the bottleneck through a small MLP. Integrating them from
   the predicted pressure field instead would compound the field error and, at
   42 px/chord, quantise the surface integral badly. Predicting them directly
   also gives a training signal that is global rather than pixel-local.

3. **GroupNorm, not BatchNorm.** Batches are small (8-16 cases on a laptop GPU)
   and each case is a different geometry, so batch statistics are noisy.

Default width (``base=24``, four downsamples) is ~4.5 M parameters and trains on
an M-series laptop; ``base=48`` reproduces the usual ~18 M literature setting.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class UNetConfig:
    in_channels: int = 4
    out_channels: int = 5
    n_scalars: int = 5
    base: int = 24
    depth: int = 4  # number of down/up steps
    norm_groups: int = 8
    dropout: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _norm(ch: int, groups: int) -> nn.Module:
    g = groups
    while g > 1 and ch % g != 0:
        g //= 2
    return nn.GroupNorm(g, ch)


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with GroupNorm + SiLU, plus a residual shortcut."""

    def __init__(self, cin: int, cout: int, groups: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm1 = _norm(cout, groups)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.norm2 = _norm(cout, groups)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = F.silu(self.norm1(self.conv1(x)))
        h = self.drop(h)
        h = self.norm2(self.conv2(h))
        return F.silu(h + self.skip(x))


class UNet(nn.Module):
    """Field + scalar surrogate."""

    def __init__(self, cfg: UNetConfig | None = None, **kwargs):
        super().__init__()
        cfg = cfg or UNetConfig(**kwargs)
        self.cfg = cfg

        widths = [cfg.base * (2 ** i) for i in range(cfg.depth + 1)]
        self.stem = ConvBlock(cfg.in_channels, widths[0], cfg.norm_groups, cfg.dropout)

        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for i in range(cfg.depth):
            self.pools.append(nn.Conv2d(widths[i], widths[i], 3, stride=2, padding=1))
            self.downs.append(ConvBlock(widths[i], widths[i + 1], cfg.norm_groups, cfg.dropout))

        self.ups = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for i in range(cfg.depth, 0, -1):
            self.ups.append(nn.ConvTranspose2d(widths[i], widths[i - 1], 2, stride=2))
            self.up_blocks.append(
                ConvBlock(2 * widths[i - 1], widths[i - 1], cfg.norm_groups, cfg.dropout)
            )

        self.head = nn.Conv2d(widths[0], cfg.out_channels, 1)
        self.scalar_head = nn.Sequential(
            nn.Linear(widths[-1], 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, cfg.n_scalars),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        skips: List[torch.Tensor] = [h]
        for pool, block in zip(self.pools, self.downs):
            h = block(pool(h))
            skips.append(h)

        bottleneck = skips[-1]
        scalars = self.scalar_head(bottleneck.mean(dim=(2, 3)))

        h = bottleneck
        for up, block, skip in zip(self.ups, self.up_blocks, reversed(skips[:-1])):
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = block(torch.cat([h, skip], dim=1))
        return self.head(h), scalars

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def masked_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    scalar_pred: torch.Tensor,
    scalar_target: torch.Tensor,
    field_weight: float = 1.0,
    solid_weight: float = 1.0,
    scalar_weight: float = 0.5,
    sample_weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, dict]:
    """L1 in the fluid, L1 in the solid, MSE on the scalars.

    L1 rather than MSE on the fields: the targets contain a genuine suction peak
    and a genuine reverse-flow region, and a squared penalty lets those few
    extreme pixels dominate the gradient and wash out the wake, which is the
    part that carries the drag.

    ``sample_weight`` is a per-case multiplier, used to up-weight a small set of
    high-fidelity RANS rows against a large set of cheap panel rows. When it is
    ``None`` the loss is pooled over all pixels in the batch exactly as before;
    the weighted branch normalises per sample first, so a case with a slightly
    larger fluid region does not quietly count for more.
    """
    fluid = mask[:, 0:1]
    solid = mask[:, 1:2]

    if sample_weight is None:
        flow_err = (pred[:, :3] - target[:, :3]).abs() * fluid
        flow_loss = flow_err.sum() / (fluid.sum() * 3 + 1e-6)

        struct_err = (pred[:, 3:] - target[:, 3:]).abs() * solid
        struct_loss = struct_err.sum() / (solid.sum() * 2 + 1e-6)

        scalar_loss = F.mse_loss(scalar_pred, scalar_target)
    else:
        w = sample_weight.reshape(-1).to(pred.dtype)
        w_sum = w.sum() + 1e-8
        dims = (1, 2, 3)

        flow_num = ((pred[:, :3] - target[:, :3]).abs() * fluid).sum(dim=dims)
        flow_den = fluid.sum(dim=dims) * 3 + 1e-6
        flow_loss = (w * (flow_num / flow_den)).sum() / w_sum

        struct_num = ((pred[:, 3:] - target[:, 3:]).abs() * solid).sum(dim=dims)
        struct_den = solid.sum(dim=dims) * 2 + 1e-6
        struct_loss = (w * (struct_num / struct_den)).sum() / w_sum

        scalar_per = ((scalar_pred - scalar_target) ** 2).mean(dim=1)
        scalar_loss = (w * scalar_per).sum() / w_sum

    total = (
        field_weight * flow_loss
        + solid_weight * struct_loss
        + scalar_weight * scalar_loss
    )
    return total, {
        "loss": float(total.detach()),
        "flow": float(flow_loss.detach()),
        "struct": float(struct_loss.detach()),
        "scalar": float(scalar_loss.detach()),
    }


def pick_device(preference: str = "auto") -> torch.device:
    """``auto`` resolves to CUDA, then Apple MPS, then CPU."""
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
