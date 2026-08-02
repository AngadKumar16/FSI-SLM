"""One dataset row, end to end: geometry -> flow -> structure -> tensors.

This is the single function the batch generator and the live API both call, so
a case produced offline for training and a case produced on demand at inference
time are byte-for-byte the same computation.

Tensor contract (identical to the OpenFOAM path in :mod:`sensor.cfd.sample`, so
panel-generated and RANS-generated rows are interchangeable):

``input``  (4, ny, nx)
    0 signed distance field, chord units, negative inside the section
    1 freestream x-component, cos(alpha)
    2 freestream y-component, sin(alpha)
    3 log10(Re) - 5

``target`` (5, ny, nx)
    0 u / Uinf          fluid only
    1 v / Uinf          fluid only
    2 p / (rho Uinf^2)  fluid only, kinematic pressure
    3 net load dCp      solid only
    4 displacement w/c  solid only

``mask`` (ny, nx) fluid indicator, and ``solid_mask`` its complement inside the
section. Channels 0-2 are zero where ``mask`` is 0 and channels 3-4 are zero
where it is 1, so a masked loss can supervise both regions without the network
having to learn where the body is (it already gets that from the SDF).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .flow.solve import N_PANELS, foil_constants, solve_case
from .geometry import Airfoil, close_trailing_edge
from .sdf import DEFAULT_SPEC, Grid, build_input, input_tensor
from .structure.beam import (
    DEFAULT_E_STAR,
    DEFAULT_SPAR_X,
    Material,
    ALUMINIUM,
    SectionProperties,
    modal_properties,
    rasterize_structure,
    section_properties,
    solve_structure,
    surface_dcp,
)

TARGET_CHANNELS = ("u", "v", "p", "load", "disp")
# The active input contract. Named channels and their mirror parities live in
# :mod:`sensor.sdf`; this is only a convenience re-export.
INPUT_CHANNELS = DEFAULT_SPEC.names


@dataclass
class CasePrep:
    """Per-foil quantities that do not depend on angle of attack.

    Building these once per foil instead of once per case cuts the 1,926-case
    sweep by roughly a third.
    """

    airfoil: Airfoil
    section: SectionProperties
    alpha_L0_deg: float
    cl_alpha_per_rad: float

    @classmethod
    def build(cls, af: Airfoil, n_panels: int = N_PANELS) -> "CasePrep":
        af = close_trailing_edge(af)
        a_l0, cl_a = foil_constants(af, n_panels)
        return cls(
            airfoil=af,
            section=section_properties(af.xy),
            alpha_L0_deg=a_l0,
            cl_alpha_per_rad=cl_a,
        )


def simulate_case(
    af: Airfoil | CasePrep,
    alpha_deg: float,
    reynolds: float = 5.0e5,
    grid: Optional[Grid] = None,
    e_star: float = DEFAULT_E_STAR,
    spar_x: float = DEFAULT_SPAR_X,
    n_panels: int = N_PANELS,
    material: Material = ALUMINIUM,
    chord_m: float = 0.1,
    span_m: float = 0.3,
    u_inf_m_s: float = 5.0,
    fluid_density: float = 998.0,
    with_modal: bool = True,
) -> dict:
    """Solve flow + structure for one (foil, alpha, Re) and pack the tensors."""
    grid = grid or Grid()
    prep = af if isinstance(af, CasePrep) else CasePrep.build(af, n_panels)

    flow = solve_case(
        prep.airfoil,
        alpha_deg,
        reynolds,
        grid,
        n_panels=n_panels,
        constants=(prep.alpha_L0_deg, prep.cl_alpha_per_rad),
    )

    # --- structure -----------------------------------------------------------
    surf = flow["surface"]
    dcp = surface_dcp(
        surf["xc"], surf["yc"], surf["nx"], surf["ny"], surf["cp"],
        flow["polygon"], prep.section.x,
    )
    struct = solve_structure(prep.section, dcp, e_star=e_star, spar_x=spar_x)

    xg, yg = grid.meshgrid()
    solid = flow["solid_mask"]
    load_field, disp_field = rasterize_structure(struct, flow["polygon"], xg, yg, solid)

    # --- tensors -------------------------------------------------------------
    # The inviscid raster is already in hand from the same panel solve that
    # produced the target, so the prior channels cost nothing here.
    inp = build_input(
        prep.airfoil, alpha_deg, reynolds, grid,
        sdf=flow["sdf"],
        prior={"u": flow["u_pot"], "v": flow["v_pot"], "p": flow["p_pot"]},
        fidelity="panel",
    )
    target = np.stack([
        flow["u"], flow["v"], flow["p"], load_field, disp_field
    ]).astype(np.float32)

    coeffs = dict(flow["coeffs"])
    coeffs.update({
        "w_max": struct.w_max,
        "w_te": struct.w_te,
        "twist_te_deg": struct.twist_te_deg,
        "stress_max": struct.stress_max,
        "e_star": struct.e_star,
        "spar_x": struct.spar_x,
        "max_thickness": prep.section.max_thickness,
        "max_thickness_x": prep.section.max_thickness_x,
        "section_area": prep.section.area,
        "section_i_zz": prep.section.i_zz,
    })

    out = {
        "input": inp,
        "target": target,
        "mask": flow["fluid_mask"].astype(np.float32),
        "solid_mask": solid.astype(np.float32),
        "alpha_deg": np.float32(alpha_deg),
        "reynolds": np.float32(reynolds),
        "coeffs": coeffs,
        "state": flow["state"],
        "structure": struct,
        "section": prep.section,
        "polygon": flow["polygon"],
        "surface_dcp": dcp,
        "fidelity": "panel",
    }

    if with_modal:
        out["modal"] = modal_properties(
            prep.section,
            chord_m=chord_m,
            span_m=span_m,
            material=material,
            fluid_density=fluid_density,
            u_inf_m_s=u_inf_m_s,
        )
    return out


def pack_npz(case: dict) -> dict:
    """Reduce a solved case to the arrays that go on disk.

    Targets are stored as float16: the fields live in [-3, 3] so half precision
    costs about 1e-3 of relative accuracy, an order of magnitude below the
    model's own error, and halves a 1,926-case dataset. The input tensor is
    *not* stored -- it is a pure function of geometry and flow condition, so the
    loader rebuilds it and the dataset shrinks by another 40%.
    """
    import json

    state = case.get("state")
    meta = {
        "coeffs": {k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v))
                   for k, v in case["coeffs"].items()},
        "modal": case.get("modal", {}),
        "fidelity": case.get("fidelity", "panel"),
        # Which route produced the surface pressure the structural channels were
        # built from: "openfoam_patch" is exact, "raster_probe" smooths the
        # leading-edge suction peak and biases the deflection targets low.
        "surface_source": case.get("surface_source", "solver"),
        # None for the panel backend, which is a closed-form solve with no
        # iteration to converge.
        "converged": case.get("converged"),
        "residual_max": case.get("residual_max"),
        "cl_std": case.get("cl_std"),
        "cd_std": case.get("cd_std"),
        # Present only on unsteady rows: shedding frequency, Strouhal numbers,
        # fluctuation levels and how many cycles the average covers.
        "unsteady": case.get("unsteady") or {},
        # Only the panel backend has a viscous-state model; a RANS row resolves
        # separation directly and has no equivalent scalars.
        "state": state.to_dict() if state is not None else {},
    }
    return {
        "target": case["target"].astype(np.float16),
        "mask": case["mask"].astype(np.uint8),
        "solid_mask": case["solid_mask"].astype(np.uint8),
        "alpha_deg": np.float32(case["alpha_deg"]),
        "reynolds": np.float32(case["reynolds"]),
        # Self-describing: everything the manifest reports is recoverable from
        # the file itself, so a resumed or partially cached sweep still yields a
        # homogeneous manifest.
        "meta": np.array(json.dumps(meta)),
        "scalars": np.array(
            [
                case["coeffs"]["cl"],
                case["coeffs"]["cd"],
                case["coeffs"]["cm"],
                case["coeffs"]["w_max"],
                case["coeffs"]["twist_te_deg"],
            ],
            dtype=np.float32,
        ),
    }


SCALAR_NAMES = ("cl", "cd", "cm", "w_max", "twist_te_deg")
