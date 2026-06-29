"""Reproducibility utilities: seed-setting at all levels.

Call ``set_seed(seed)`` once at CLI startup to ensure deterministic behaviour
across numpy, random, torch, and MLX random number generators.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Set the master seed for all random number generators used in the project.

    This function seeds:
    - Python's ``random`` module
    - ``numpy.random``
    - ``torch`` (CPU and CUDA, if available)
    - ``mlx`` (if importable)
    - Sets ``PYTHONHASHSEED`` for dict iteration reproducibility (note: this
      must be set before the Python interpreter starts to take full effect; we
      set it here as a best-effort measure).

    Parameters
    ----------
    seed : int
        The master seed value.
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (if available)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    # MLX (if available)
    try:
        import mlx.core as mx

        mx.random.seed(seed)
    except ImportError:
        pass

    # Environment
    os.environ["PYTHONHASHSEED"] = str(seed)


__all__ = ["set_seed"]