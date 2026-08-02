"""The mirror symmetry of 2-D airfoil flow, as an exact data transform.

Reflect a case about ``y = 0`` and you get another physically valid case: the
mirrored section at the negated incidence, with the sign of every
transverse quantity flipped. Nothing is approximated, and no solver has to run.

    solve(mirror(foil), -alpha)  ==  mirror(solve(foil, +alpha))

Two uses.

**Augmentation.** The dataset is 107 sections. Applying the mirror at load time
doubles it at zero cost and, more usefully, stops the network learning that
positive incidence is somehow different in kind from negative -- with an
unbalanced angle sweep (-8 to +20 here) it otherwise will.

**Test-time averaging.** Predicting a case and its mirror and averaging the two
back removes the component of the error that is antisymmetric noise rather than
physics. It costs one extra forward pass and needs no retraining.

Correctness rests entirely on the parity tables below, so they are checked
against the solver itself rather than asserted: ``tests/test_symmetry.py`` runs
a symmetric section at +alpha and -alpha and requires the mirrored one to
reproduce the other to solver precision. A wrong sign anywhere in this file
fails that test.

The grid must be symmetric in ``y`` for a row flip to be the mirror; see
:attr:`sensor.sdf.Grid.y_symmetric`.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from .sdf import InputSpec

# Target channels (u, v, p, load, disp). ``load`` is the net chordwise pressure
# difference driving the beam and ``disp`` the resulting deflection: both are
# signed transverse quantities, so both negate.
TARGET_PARITY = np.array([+1.0, -1.0, +1.0, -1.0, -1.0], dtype=np.float32)

# Scalars (cl, cd, cm, w_max, twist_te_deg). ``cd`` is invariant. ``w_max`` is
# ``max |w|`` -- a magnitude, not a signed peak -- so it is invariant too, while
# ``twist_te_deg`` comes from the signed trailing-edge slope and negates. That
# asymmetry between the two structural scalars is the easiest thing in this file
# to get wrong.
SCALAR_PARITY = np.array([-1.0, +1.0, -1.0, +1.0, -1.0], dtype=np.float32)

# Masks (fluid, solid) are indicator fields: flipped, never negated.
MASK_PARITY = np.array([+1.0, +1.0], dtype=np.float32)


def _flip_np(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a[..., ::-1, :])


def mirror_np(a: np.ndarray, parity: np.ndarray) -> np.ndarray:
    """Flip rows and apply a per-channel sign. ``a`` is ``(C, H, W)``."""
    a = np.asarray(a)
    if a.shape[0] != parity.shape[0]:
        raise ValueError(
            f"parity has {parity.shape[0]} entries for a {a.shape[0]}-channel array"
        )
    return _flip_np(a) * parity[:, None, None].astype(a.dtype)


def mirror_scalars_np(s: np.ndarray, parity: np.ndarray = SCALAR_PARITY) -> np.ndarray:
    return np.asarray(s) * parity.astype(np.asarray(s).dtype)


def mirror_case_np(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    s: np.ndarray,
    spec: InputSpec,
) -> tuple:
    """Mirror a whole training row: input, target, masks, scalars."""
    return (
        mirror_np(x, spec.parity),
        mirror_np(y, TARGET_PARITY),
        mirror_np(mask, MASK_PARITY),
        mirror_scalars_np(s),
    )


# ------------------------------------------------------------------ torch side


def mirror_torch(a, parity):
    """``mirror_np`` for a batched torch tensor ``(B, C, H, W)``."""
    import torch

    if not isinstance(parity, torch.Tensor):
        parity = torch.as_tensor(parity, dtype=a.dtype, device=a.device)
    parity = parity.to(dtype=a.dtype, device=a.device)
    return torch.flip(a, dims=(-2,)) * parity.view(1, -1, 1, 1)


def mirror_batch(batch: Dict[str, "object"], spec: InputSpec) -> Dict[str, "object"]:
    """Mirror an entire batch dict in place of the original (not in-place)."""
    import torch

    out = dict(batch)
    out["x"] = mirror_torch(batch["x"], spec.parity)
    out["y"] = mirror_torch(batch["y"], TARGET_PARITY)
    out["mask"] = mirror_torch(batch["mask"], MASK_PARITY)
    sp = torch.as_tensor(SCALAR_PARITY, dtype=batch["s"].dtype, device=batch["s"].device)
    out["s"] = batch["s"] * sp.view(1, -1)
    return out


def normalised_scalar_parity(s_mean: Sequence[float], s_std: Sequence[float]):
    """Parity of the *normalised* scalars: an affine offset breaks the sign flip.

    Training normalises scalars as ``(s - mean) / std``. Negating the physical
    scalar is not the same as negating the normalised one unless the mean is
    zero, and ``cd`` and ``w_max`` have means far from zero. Mirroring in
    normalised space therefore needs both the sign and a shift:

        s'_norm = (parity * s - mean) / std
                = parity * s_norm + (parity - 1) * mean / std

    Returns ``(scale, shift)`` to apply as ``scale * s_norm + shift``.
    """
    mean = np.asarray(s_mean, dtype=np.float32)
    std = np.asarray(s_std, dtype=np.float32)
    scale = SCALAR_PARITY
    shift = (SCALAR_PARITY - 1.0) * mean / np.maximum(std, 1e-8)
    return scale, shift.astype(np.float32)


def normalised_field_parity(
    y_mean: Sequence[float],
    y_std: Sequence[float],
    parity: np.ndarray = TARGET_PARITY,
):
    """Same correction for the normalised field channels.

    ``p`` in particular has a nonzero mean (it is a pressure coefficient over a
    domain that is mostly freestream), so mirroring a normalised field by sign
    alone shifts the whole pressure channel. Returns ``(scale, shift)`` with
    shapes broadcastable over ``(C, 1, 1)``.
    """
    mean = np.asarray(y_mean, dtype=np.float32)
    std = np.asarray(y_std, dtype=np.float32)
    shift = (parity - 1.0) * mean / np.maximum(std, 1e-8)
    return parity.astype(np.float32), shift.astype(np.float32)


def normalised_input_parity(
    spec: InputSpec,
    x_mean: Sequence[float],
    x_std: Sequence[float],
):
    """And for the input channels."""
    mean = np.asarray(x_mean, dtype=np.float32)
    std = np.asarray(x_std, dtype=np.float32)
    parity = spec.parity
    shift = (parity - 1.0) * mean / np.maximum(std, 1e-8)
    return parity.astype(np.float32), shift.astype(np.float32)
