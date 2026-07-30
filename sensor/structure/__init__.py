"""Structural side of the surrogate: aerodynamic load and displacement fields.

The flow solver returns a surface pressure distribution; this package turns that
into the two structural quantities the abstract promises -- a **load field** and
a **displacement field** -- plus the modal properties the FSI interpreter needs
downstream (natural frequency, mass ratio, reduced velocity).

Coupling is one-way and static: pressure drives deflection, deflection does not
feed back into the flow. That is the honest description of what is implemented,
and it is the standard first rung for a steady surrogate. The two-way static
iteration is a small extension of :func:`sensor.structure.beam.solve_structure`
(recompute the flow at the deflected shape and repeat) and is deliberately not
claimed here.

Sign convention: displacement is positive along the chord normal pointing toward
the suction side at positive angle of attack (i.e. a lifting foil deflects with
positive ``w``).
"""

from .beam import (  # noqa: F401
    Material,
    SectionProperties,
    StructuralResult,
    ALUMINIUM,
    STEEL,
    PVC,
    POLYCARBONATE,
    section_properties,
    solve_structure,
    modal_properties,
)

__all__ = [
    "Material",
    "SectionProperties",
    "StructuralResult",
    "ALUMINIUM",
    "STEEL",
    "PVC",
    "POLYCARBONATE",
    "section_properties",
    "solve_structure",
    "modal_properties",
]
