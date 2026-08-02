"""U-Net surrogate: geometry + flow condition -> fields and force coefficients.

The backbone is the encoder-decoder used for airfoil flow prediction by Thuerey
et al. (2020) and Bhatnagar et al. (2019). Everything below is a deviation from
it, and each one exists because of something the FSI task or the measured error
made necessary.

1. **Two output regions.** Channels 0-2 (u, v, p) live in the fluid, channels
   3-4 (load, displacement) live inside the solid. One decoder produces all
   five; the masked loss routes supervision to the right region. Sharing the
   trunk is deliberate -- the load field *is* the surface pressure, so the
   structural head should reuse the same features rather than relearn them.

2. **FiLM conditioning, not constant image channels.** Incidence, Reynolds
   number and the fidelity code are scalars. Broadcasting them into full feature
   maps and asking 3x3 convolutions to carry them through four downsamples wastes
   capacity on reproducing a constant. They are read back out of the constant
   input channels as a global vector and injected as a per-channel scale and
   shift at every block, which is where a global condition belongs.

3. **A scalar head that pools where the forces are.** Integrated Cl, Cd, Cm,
   peak deflection and trailing-edge twist used to be read from a global average
   of the bottleneck. Global average pooling is the wrong reduction for a
   surface integral: it weights a pixel three chords downstream exactly like a
   pixel on the suction peak. The head now pools every encoder level with both
   mean and max, *and* takes a wall-weighted average of the finest decoder
   features using the near-wall input channel as the weight -- so the features
   that decide Cl are the ones near the surface.

4. **Bilinear upsampling, not transposed convolution.** A stride-2 transposed
   convolution with an even kernel lays down a checkerboard, and on a pressure
   field that shows up as a grid pattern in the wake, which is exactly where the
   drag is read.

5. **Replicate padding.** Zero padding tells the network the flow stops at the
   image border. It does not; the border is an interior slice of a much larger
   CFD domain.

6. **Self-attention at the bottleneck.** Circulation is global: the trailing-edge
   Kutta condition sets the lift, which sets the far field everywhere. At the
   bottleneck the map is 8 x 8, so full attention over 64 tokens is cheap.

7. **GroupNorm, not BatchNorm.** Batches are small (8-16 cases on a laptop GPU)
   and each case is a different geometry, so batch statistics are noisy.

Default width (``base=32``, four downsamples) is ~11 M parameters and still
trains on an M-series laptop; ``base=24`` reproduces the older ~2 M setting for
a quick sweep.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sdf import SPEC_V2, InputSpec


@dataclass
class UNetConfig:
    in_channels: int = SPEC_V2.n_channels
    out_channels: int = 5
    n_scalars: int = 5
    base: int = 32
    depth: int = 4  # number of down/up steps
    norm_groups: int = 8
    dropout: float = 0.0
    attention: bool = True
    film: bool = True
    padding_mode: str = "replicate"
    # Indices of the spatially-constant input channels, read back as the global
    # conditioning vector. Empty disables FiLM regardless of ``film``.
    global_idx: Tuple[int, ...] = SPEC_V2.constant_idx
    # Index of the near-wall channel, used to weight the scalar head's pooling.
    # ``-1`` falls back to unweighted pooling.
    wall_idx: int = SPEC_V2.index("wall")
    input_spec: str = SPEC_V2.name

    def to_dict(self) -> dict:
        d = asdict(self)
        d["global_idx"] = list(self.global_idx)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "UNetConfig":
        d = dict(d)
        d["global_idx"] = tuple(d.get("global_idx", ()))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def for_spec(cls, spec: InputSpec, **kwargs) -> "UNetConfig":
        """Config whose input contract matches ``spec``. The only safe route."""
        return cls(
            in_channels=spec.n_channels,
            global_idx=spec.constant_idx,
            wall_idx=spec.index("wall") if "wall" in spec.names else -1,
            input_spec=spec.name,
            **kwargs,
        )


def _norm(ch: int, groups: int) -> nn.Module:
    g = groups
    while g > 1 and ch % g != 0:
        g //= 2
    return nn.GroupNorm(g, ch)


class FiLM(nn.Module):
    """Per-channel scale and shift from the global flow condition.

    Initialised so that the scale starts at exactly 1 and the shift at 0: the
    conditioning begins as the identity and the network is never worse off for
    having it.
    """

    def __init__(self, cond_dim: int, ch: int):
        super().__init__()
        self.to_scale_shift = nn.Linear(cond_dim, 2 * ch)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, h, cond):
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=1)
        return h * (1.0 + scale[..., None, None]) + shift[..., None, None]


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with GroupNorm + SiLU, residual, optional FiLM."""

    def __init__(
        self,
        cin: int,
        cout: int,
        groups: int,
        dropout: float = 0.0,
        cond_dim: int = 0,
        padding_mode: str = "replicate",
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, padding_mode=padding_mode)
        self.norm1 = _norm(cout, groups)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, padding_mode=padding_mode)
        self.norm2 = _norm(cout, groups)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.film = FiLM(cond_dim, cout) if cond_dim else None

    def forward(self, x, cond=None):
        h = self.norm1(self.conv1(x))
        if self.film is not None and cond is not None:
            h = self.film(h, cond)
        h = self.drop(F.silu(h))
        h = self.norm2(self.conv2(h))
        return F.silu(h + self.skip(x))


class SelfAttention(nn.Module):
    """Multi-head self-attention over a feature map, flattened to tokens."""

    def __init__(self, ch: int, groups: int, heads: int = 4):
        super().__init__()
        while heads > 1 and ch % heads != 0:
            heads //= 2
        self.heads = heads
        self.norm = _norm(ch, groups)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        # Start as a no-op so attention can only help.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w).unbind(1)
        out = F.scaled_dot_product_attention(
            q.transpose(-2, -1), k.transpose(-2, -1), v.transpose(-2, -1)
        ).transpose(-2, -1)
        return x + self.proj(out.reshape(b, c, h, w))


class Up(nn.Module):
    """Bilinear upsample followed by a 3x3 convolution.

    Replaces ``ConvTranspose2d(k=2, stride=2)``, whose uneven kernel overlap
    stamps a checkerboard into every decoded field.
    """

    def __init__(self, cin: int, cout: int, padding_mode: str = "replicate"):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1, padding_mode=padding_mode)

    def forward(self, x, size):
        return self.conv(F.interpolate(x, size=size, mode="bilinear", align_corners=False))


class ScalarHead(nn.Module):
    """Force and deflection coefficients from pooled multi-scale features."""

    def __init__(self, widths: Sequence[int], fine_ch: int, n_out: int, cond_dim: int):
        super().__init__()
        # mean + max over every encoder level, plus mean + wall-weighted mean of
        # the finest decoder map, plus the raw global condition.
        dim = 2 * sum(widths) + 2 * fine_ch + cond_dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, 256), nn.SiLU(),
            nn.Linear(256, 128), nn.SiLU(),
            nn.Linear(128, n_out),
        )

    def forward(self, levels: List[torch.Tensor], fine: torch.Tensor,
                wall: Optional[torch.Tensor], cond: torch.Tensor):
        feats: List[torch.Tensor] = []
        for h in levels:
            feats.append(h.mean(dim=(2, 3)))
            feats.append(h.amax(dim=(2, 3)))
        feats.append(fine.mean(dim=(2, 3)))
        if wall is not None:
            w = wall / (wall.sum(dim=(2, 3), keepdim=True) + 1e-6)
            feats.append((fine * w).sum(dim=(2, 3)))
        else:
            feats.append(fine.amax(dim=(2, 3)))
        feats.append(cond)
        return self.mlp(torch.cat(feats, dim=1))


class UNet(nn.Module):
    """Field + scalar surrogate."""

    def __init__(self, cfg: UNetConfig | None = None, **kwargs):
        super().__init__()
        cfg = cfg or UNetConfig(**kwargs)
        self.cfg = cfg

        widths = [cfg.base * (2 ** i) for i in range(cfg.depth + 1)]
        self.widths = widths
        cond_dim = len(cfg.global_idx) if (cfg.film and cfg.global_idx) else 0
        self.cond_dim = cond_dim
        # A raw condition vector of four numbers is a thin thing to modulate
        # 11 M parameters with; lift it first.
        self.cond_mlp = (
            nn.Sequential(nn.Linear(cond_dim, 64), nn.SiLU(), nn.Linear(64, 64))
            if cond_dim else None
        )
        block_cond = 64 if cond_dim else 0

        mk = lambda cin, cout: ConvBlock(
            cin, cout, cfg.norm_groups, cfg.dropout, block_cond, cfg.padding_mode
        )

        self.stem = mk(cfg.in_channels, widths[0])
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for i in range(cfg.depth):
            self.pools.append(
                nn.Conv2d(widths[i], widths[i], 3, stride=2, padding=1,
                          padding_mode=cfg.padding_mode)
            )
            self.downs.append(mk(widths[i], widths[i + 1]))

        self.attn = (
            SelfAttention(widths[-1], cfg.norm_groups) if cfg.attention else None
        )

        self.ups = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for i in range(cfg.depth, 0, -1):
            self.ups.append(Up(widths[i], widths[i - 1], cfg.padding_mode))
            self.up_blocks.append(mk(2 * widths[i - 1], widths[i - 1]))

        self.head = nn.Sequential(
            nn.Conv2d(widths[0], widths[0], 3, padding=1, padding_mode=cfg.padding_mode),
            nn.SiLU(),
            nn.Conv2d(widths[0], cfg.out_channels, 1),
        )
        self.scalar_head = ScalarHead(
            widths, widths[0], cfg.n_scalars, max(block_cond, 1) if cond_dim else 1
        )

    # ------------------------------------------------------------------ pieces

    def _condition(self, x: torch.Tensor) -> torch.Tensor:
        """Global conditioning vector, read out of the constant input channels.

        The channels really are constant maps, so the spatial mean recovers the
        value exactly; this avoids a second input tensor and keeps the on-disk
        cache one array.
        """
        if not self.cfg.global_idx:
            return x.new_zeros((x.shape[0], 1))
        idx = torch.as_tensor(self.cfg.global_idx, device=x.device, dtype=torch.long)
        return x.index_select(1, idx).mean(dim=(2, 3))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_cond = self._condition(x)
        cond = self.cond_mlp(raw_cond) if self.cond_mlp is not None else None

        h = self.stem(x, cond)
        skips: List[torch.Tensor] = [h]
        for pool, block in zip(self.pools, self.downs):
            h = block(pool(h), cond)
            skips.append(h)

        h = skips[-1]
        if self.attn is not None:
            h = self.attn(h)

        for up, block, skip in zip(self.ups, self.up_blocks, reversed(skips[:-1])):
            h = up(h, skip.shape[-2:])
            h = block(torch.cat([h, skip], dim=1), cond)

        wall = (
            x[:, self.cfg.wall_idx: self.cfg.wall_idx + 1]
            if self.cfg.wall_idx >= 0 else None
        )
        # The wall channel arrives normalised, so it can be negative; the head
        # needs a nonnegative weight.
        if wall is not None:
            wall = wall - wall.amin(dim=(2, 3), keepdim=True)
        scalars = self.scalar_head(skips, h, wall, raw_cond)
        return self.head(h), scalars

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ----------------------------------------------------------------------- loss


def _erode(mask: torch.Tensor) -> torch.Tensor:
    """One-cell erosion, so a finite difference never straddles the boundary."""
    return -F.max_pool2d(-mask, 3, stride=1, padding=1)


def _dx(a: torch.Tensor) -> torch.Tensor:
    """Central difference along x (last axis), same shape, edges one-sided."""
    out = torch.zeros_like(a)
    out[..., 1:-1] = 0.5 * (a[..., 2:] - a[..., :-2])
    out[..., 0] = a[..., 1] - a[..., 0]
    out[..., -1] = a[..., -1] - a[..., -2]
    return out


def _dy(a: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(a)
    out[..., 1:-1, :] = 0.5 * (a[..., 2:, :] - a[..., :-2, :])
    out[..., 0, :] = a[..., 1, :] - a[..., 0, :]
    out[..., -1, :] = a[..., -1, :] - a[..., -2, :]
    return out


def border_circulation(u: torch.Tensor, v: torch.Tensor, dx: float, dy: float):
    """``-∮ u dx + v dy`` around the image border, per sample.

    Kelvin's theorem makes this the total circulation enclosed by the window,
    which for a steady 2-D section is the bound circulation and hence ``Cl / 2``
    by Kutta-Joukowski. The window boundary is over half a chord from the body
    on every side, so nothing singular is being integrated.

    The sign convention (counter-clockwise contour, negated) is the one that
    makes a positive-lift section return a positive number; the loss that uses
    this is a difference of differences, so it is insensitive to a global
    offset, but not to a flipped sign.
    """
    # bottom edge (+x), right edge (+y), top edge (-x), left edge (-y)
    bottom = u[..., 0, :].sum(-1) * dx
    top = u[..., -1, :].sum(-1) * dx
    right = v[..., :, -1].sum(-1) * dy
    left = v[..., :, 0].sum(-1) * dy
    return -(bottom + right - top - left)


@dataclass
class LossConfig:
    """Weights for :func:`surrogate_loss`. Every term is off at weight zero."""

    field: float = 1.0
    solid: float = 1.0
    scalar: float = 0.5
    gradient: float = 0.25       # match derivatives, not just values
    divergence: float = 0.02     # incompressibility of the predicted velocity
    circulation: float = 0.05    # Kutta-Joukowski tie between field and Cl
    near_wall: float = 3.0       # extra weight on pixels within the wall band
    wall_band_px: float = 3.0
    huber_delta: float = 1.0     # scalar loss: Huber, in normalised units
    channel_weights: Tuple[float, ...] = (1.0, 1.5, 1.5, 1.0, 1.0)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["channel_weights"] = list(self.channel_weights)
        return d


def _weighted_mean(err: torch.Tensor, w: torch.Tensor, sample_weight):
    """Mean of ``err`` over ``w``, per sample, then averaged with sample weights."""
    dims = (1, 2, 3)
    num = (err * w).sum(dim=dims)
    den = w.sum(dim=dims) + 1e-6
    per = num / den
    if sample_weight is None:
        return per.mean()
    sw = sample_weight.reshape(-1).to(per.dtype)
    return (sw * per).sum() / (sw.sum() + 1e-8)


def surrogate_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    scalar_pred: torch.Tensor,
    scalar_target: torch.Tensor,
    cfg: Optional[LossConfig] = None,
    sample_weight: Optional[torch.Tensor] = None,
    wall: Optional[torch.Tensor] = None,
    y_std: Optional[Sequence[float]] = None,
    s_std: Optional[Sequence[float]] = None,
    dx: float = 1.0,
    dy: float = 1.0,
) -> Tuple[torch.Tensor, dict]:
    """Masked L1 on the fields, Huber on the scalars, plus three physics terms.

    L1 rather than MSE on the fields: the targets contain a genuine suction peak
    and a genuine reverse-flow region, and a squared penalty lets those few
    extreme pixels dominate the gradient and wash out the wake, which is the
    part that carries the drag.

    The three additions each target a measured failure:

    ``gradient``
        A value-only L1 is minimised by a blurred field. Matching the finite
        differences as well is what keeps the shear layer and the wake edge
        sharp; ``v`` and ``p`` -- the two worst channels -- are the ones defined
        by gradients rather than by their own magnitude.

    ``divergence``
        The flow is incompressible, so ``du/dx + dv/dy = 0`` exactly. Penalising
        the predicted divergence costs nothing (two finite differences) and is a
        constraint the network otherwise has no way to know about.

    ``circulation``
        Ties the scalar head to the velocity field through Kutta-Joukowski. It
        is written as a difference of differences -- the *error* in the raster
        circulation must move with the *error* in Cl -- so any systematic offset
        between the two, which the clipped near-wall layer certainly introduces,
        cancels instead of being learned as a bias.

    ``sample_weight`` is a per-case multiplier, used to up-weight a small set of
    high-fidelity RANS rows against a large set of cheap panel rows. Every term
    normalises per sample before weighting, so a case with a slightly larger
    fluid region does not quietly count for more.
    """
    cfg = cfg or LossConfig()
    fluid = mask[:, 0:1]
    solid = mask[:, 1:2]
    cw = torch.as_tensor(cfg.channel_weights, dtype=pred.dtype, device=pred.device)
    cw = cw.view(1, -1, 1, 1)

    # Pixels near the wall carry the load and the suction peak, and there are
    # few of them; without this they are ~2% of the fluid area and are averaged
    # away.
    w_fluid = fluid
    if wall is not None and cfg.near_wall > 0:
        w_fluid = fluid * (1.0 + cfg.near_wall * wall.clamp(min=0.0))

    flow_err = (pred[:, :3] - target[:, :3]).abs() * cw[:, :3]
    flow_loss = _weighted_mean(flow_err, w_fluid.expand_as(flow_err), sample_weight)

    struct_err = (pred[:, 3:] - target[:, 3:]).abs() * cw[:, 3:]
    struct_loss = _weighted_mean(struct_err, solid.expand_as(struct_err), sample_weight)

    # Huber, not MSE: five normalised scalars, and a single badly-predicted
    # deep-stall case should not own the batch gradient.
    scalar_per = F.huber_loss(
        scalar_pred, scalar_target, reduction="none", delta=cfg.huber_delta
    ).mean(dim=1)
    if sample_weight is None:
        scalar_loss = scalar_per.mean()
    else:
        sw = sample_weight.reshape(-1).to(scalar_per.dtype)
        scalar_loss = (sw * scalar_per).sum() / (sw.sum() + 1e-8)

    parts = {
        "flow": flow_loss,
        "struct": struct_loss,
        "scalar": scalar_loss,
    }
    total = (
        cfg.field * flow_loss + cfg.solid * struct_loss + cfg.scalar * scalar_loss
    )

    inner_f = _erode(fluid)
    if cfg.gradient > 0:
        inner_s = _erode(solid)
        gx = (_dx(pred) - _dx(target)).abs() * cw
        gy = (_dy(pred) - _dy(target)).abs() * cw
        gw = torch.cat([inner_f.expand(-1, 3, -1, -1), inner_s.expand(-1, 2, -1, -1)], 1)
        grad_loss = _weighted_mean(gx + gy, gw, sample_weight)
        parts["grad"] = grad_loss
        total = total + cfg.gradient * grad_loss

    if cfg.divergence > 0 and y_std is not None:
        su = float(y_std[0])
        sv = float(y_std[1])
        # Fields arrive normalised; the constant means differentiate away, so
        # only the per-channel scale survives into the physical divergence.
        div = (
            su * _dx(pred[:, 0:1]) / max(dx, 1e-9)
            + sv * _dy(pred[:, 1:2]) / max(dy, 1e-9)
        )
        div_loss = _weighted_mean(div.abs(), inner_f, sample_weight)
        parts["div"] = div_loss
        total = total + cfg.divergence * div_loss

    if cfg.circulation > 0 and y_std is not None and s_std is not None:
        su, sv = float(y_std[0]), float(y_std[1])
        g_pred = border_circulation(su * pred[:, 0], sv * pred[:, 1], dx, dy)
        g_true = border_circulation(su * target[:, 0], sv * target[:, 1], dx, dy)
        cl_err = (scalar_pred[:, 0] - scalar_target[:, 0]) * float(s_std[0])
        circ_per = ((g_pred - g_true) - 0.5 * cl_err).abs()
        if sample_weight is None:
            circ_loss = circ_per.mean()
        else:
            sw = sample_weight.reshape(-1).to(circ_per.dtype)
            circ_loss = (sw * circ_per).sum() / (sw.sum() + 1e-8)
        parts["circ"] = circ_loss
        total = total + cfg.circulation * circ_loss

    log = {k: float(v.detach()) for k, v in parts.items()}
    log["loss"] = float(total.detach())
    return total, log


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
    """The original plain masked loss: L1 in each region, MSE on the scalars.

    Retained as the ablation baseline and so that older callers keep working.
    """
    cfg = LossConfig(
        field=field_weight, solid=solid_weight, scalar=scalar_weight,
        gradient=0.0, divergence=0.0, circulation=0.0, near_wall=0.0,
        channel_weights=(1.0, 1.0, 1.0, 1.0, 1.0),
    )
    total, log = surrogate_loss(
        pred, target, mask, scalar_pred, scalar_target, cfg, sample_weight
    )
    return total, log


# ------------------------------------------------------------------ utilities


class EMA:
    """Exponential moving average of the weights, evaluated instead of the raw ones.

    Nearly free and reliably worth a few percent on a noisy small-batch run: the
    averaged weights sit closer to the centre of the basin than any single SGD
    iterate. ``decay`` is warmed up so the average is not dominated by the
    random initialisation for the first few hundred steps.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.step = 0
        self.shadow = {
            k: v.detach().clone().float()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }
        self.buffers = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if not v.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.step += 1
        d = min(self.decay, (1.0 + self.step) / (10.0 + self.step))
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1.0 - d)
            else:
                self.buffers[k] = v.detach().clone()

    def state_dict(self) -> dict:
        return {**{k: v.clone() for k, v in self.shadow.items()}, **self.buffers}

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.state_dict(), strict=True)


def load_checkpoint(path, map_location=None):
    """``torch.load`` for our own checkpoints, across the 2.6 default change.

    PyTorch 2.6 flipped ``weights_only`` to ``True``. Our checkpoints are not
    bare tensors -- they carry the config dict, normalisation statistics and the
    training history -- so the safe unpickler rejects them::

        _pickle.UnpicklingError: Weights only load failed.
        WeightsUnpickler error: Unsupported global: GLOBAL getattr

    These files are written by this repository, so loading them with the full
    unpickler is the intended behaviour, not a security relaxation of untrusted
    input. Older torch has no such keyword, hence the fallback.
    """
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def pick_device(preference: str = "auto") -> torch.device:
    """``auto`` resolves to CUDA, then Apple MPS, then CPU."""
    if preference != "auto":
        try:
            return torch.device(preference)
        except RuntimeError as exc:
            raise ValueError(
                f"invalid device {preference!r}. Expected 'auto', 'cpu', 'mps' or "
                f"'cuda'. A stray shell word is the usual cause -- note that "
                f"interactive zsh does not treat '#' as a comment, so a trailing "
                f"'# note' on a command becomes positional arguments. ({exc})"
            ) from None
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
