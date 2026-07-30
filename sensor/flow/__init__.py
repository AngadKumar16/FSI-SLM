"""Fast physics backend that produces U-Net training targets without a mesher.

Two interchangeable backends produce the *same* ``.npz`` contract
(see :mod:`sensor.cfd.sample`):

* ``sensor.flow``  -- Hess-Smith panel method + Beddoes-Leishman static-stall
  viscous correction + momentum-consistent wake model. Pure NumPy, ~0.1 s per
  case, runs anywhere. This is what generates the full 107 x 18 sweep.
* ``sensor.cfd``   -- gmsh + OpenFOAM steady RANS. Higher fidelity, external
  binaries, minutes per case. Used to regenerate a validation subset (NACA 0009
  is held out for exactly this) and to fine-tune a panel-pretrained U-Net.

Each dataset row carries a ``fidelity`` tag ("panel" | "rans") so the two can be
mixed, weighted, or split at training time.
"""

from .panel import PanelSolver, PanelSolution  # noqa: F401
from .viscous import ViscousModel, ViscousState  # noqa: F401
from .solve import FlowCase, solve_case  # noqa: F401

__all__ = [
    "PanelSolver",
    "PanelSolution",
    "ViscousModel",
    "ViscousState",
    "FlowCase",
    "solve_case",
]
