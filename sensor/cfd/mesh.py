"""gmsh 2D airfoil mesh -> .msh ready for OpenFOAM ``gmshToFoam``.

Geometry (all in chord units, c = 1):
  * foil polygon (already rotated by AoA) punched as a hole in a far-field disk
    of radius ``R`` (default 20c) centred at the quarter chord.
  * near-wall refinement via a Distance + Threshold size field (isotropic).
    v1 targets wall functions (y+ ~ 30), not a resolved BL -- robust for
    batch meshing. Resolved-BL + transition model is a later upgrade.
  * the 2D surface is extruded one cell in +z; the two z-faces become an
    ``empty`` patch so OpenFOAM treats the case as 2D.

Physical patches written: ``airfoil`` (wall), ``farfield``, ``frontAndBack``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..geometry import Airfoil, rotate


@dataclass
class MeshParams:
    radius: float = 20.0        # far-field radius (chords)
    span: float = 0.1           # z-thickness of the single cell layer
    wall_size: float = 4.0e-3   # target cell size at the foil wall (chords)
    far_size: float = 1.5       # cell size at far field
    refine_dist: float = 1.0    # distance over which size grows wall->far
    wake_size: float = 3.0e-2   # size in the near wake box
    wake_xmax: float = 4.0      # downstream extent of the wake refinement box
    # Merge boundary panels shorter than this before meshing. A tabulated
    # section carries a blunt-TE base segment (~5e-4 c for the 6-series set)
    # and, on some foils, sub-1e-3 clustering at the nose. Each raw coordinate
    # is a *forced* mesh node, so one 5e-4 panel plants a 5e-4 cell whose
    # Courant limit then pins the URANS time step at ~6e-4 -- 50k+ steps to
    # cover the averaging window -- no matter how coarse ``wall_size`` is.
    # Collapsing panels below this floor sharpens the TE (a 5e-4 gap is 0.05%
    # of chord; negligible) and lets ``wall_size`` actually set the step.
    min_panel: float = 3.0e-3
    pivot: tuple[float, float] = (0.25, 0.0)


def _merge_short_panels(pts: np.ndarray, floor: float) -> np.ndarray:
    """Drop points that sit closer than ``floor`` to the previous kept point.

    Walks the open polygon (the closing point already dropped) greedily, then
    checks the wrap-around segment so the collapsed trailing edge does not leave
    a sub-floor panel between the last kept point and the first. Endpoints of the
    coarsest regions are preserved because the median panel is far above the
    floor; only the blunt-TE base and any nose slivers are actually removed.
    """
    if len(pts) < 4:
        return pts
    keep = [pts[0]]
    for p in pts[1:]:
        if float(np.hypot(*(p - keep[-1]))) >= floor:
            keep.append(p)
    # The wrap segment (last -> first) can still be short after a TE collapse.
    while len(keep) > 3 and float(np.hypot(*(keep[-1] - keep[0]))) < floor:
        keep.pop()
    return np.asarray(keep)


def build_mesh(af: Airfoil, alpha_deg: float, out_msh: str | Path,
               mp: MeshParams | None = None) -> Path:
    import gmsh

    mp = mp or MeshParams()
    out_msh = Path(out_msh)
    rot = rotate(af, alpha_deg, mp.pivot)
    pts = rot.xy[:-1]  # drop duplicated closing point; loop closed via lines
    pts = _merge_short_panels(pts, mp.min_panel)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(out_msh.stem)
    occ = gmsh.model.occ

    # --- foil curve loop (straight segments through the coordinate points) ---
    ptags = [occ.addPoint(x, y, 0.0) for x, y in pts]
    lines = [occ.addLine(ptags[i], ptags[(i + 1) % len(ptags)])
             for i in range(len(ptags))]
    foil_loop = occ.addCurveLoop(lines)
    foil_surf = occ.addPlaneSurface([foil_loop])

    # --- far-field disk, then cut the foil out of it ---
    cx, cy = mp.pivot
    disk = occ.addDisk(cx, cy, 0.0, mp.radius, mp.radius)
    domain, _ = occ.cut([(2, disk)], [(2, foil_surf)])
    occ.synchronize()

    surf_tag = domain[0][1]
    # boundary curves of the domain: the outer circle + the foil edges
    bnd = gmsh.model.getBoundary(domain, oriented=False)
    foil_curves, far_curves = [], []
    for dim, tag in bnd:
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        r = ((com[0] - cx) ** 2 + (com[1] - cy) ** 2) ** 0.5
        (far_curves if r > mp.radius * 0.5 else foil_curves).append(tag)

    # --- size field: fine at wall, grow to far field ---
    f_dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", foil_curves)
    gmsh.model.mesh.field.setNumber(f_dist, "Sampling", 400)
    f_thr = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_thr, "InField", f_dist)
    gmsh.model.mesh.field.setNumber(f_thr, "SizeMin", mp.wall_size)
    gmsh.model.mesh.field.setNumber(f_thr, "SizeMax", mp.far_size)
    gmsh.model.mesh.field.setNumber(f_thr, "DistMin", 0.02)
    gmsh.model.mesh.field.setNumber(f_thr, "DistMax", mp.refine_dist)
    # wake refinement box downstream of the foil
    f_wake = gmsh.model.mesh.field.add("Box")
    gmsh.model.mesh.field.setNumber(f_wake, "VIn", mp.wake_size)
    gmsh.model.mesh.field.setNumber(f_wake, "VOut", mp.far_size)
    gmsh.model.mesh.field.setNumber(f_wake, "XMin", 0.0)
    gmsh.model.mesh.field.setNumber(f_wake, "XMax", mp.wake_xmax)
    gmsh.model.mesh.field.setNumber(f_wake, "YMin", -1.2)
    gmsh.model.mesh.field.setNumber(f_wake, "YMax", 1.2)
    gmsh.model.mesh.field.setNumber(f_wake, "Thickness", 1.0)
    f_min = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(f_min, "FieldsList", [f_thr, f_wake])
    gmsh.model.mesh.field.setAsBackgroundMesh(f_min)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay

    gmsh.model.mesh.generate(2)

    # --- extrude one layer in z, recombine to prisms; z-faces = frontAndBack ---
    ext = occ.extrude([(2, surf_tag)], 0, 0, mp.span, numElements=[1],
                      recombine=True)
    occ.synchronize()

    vol = [t for (d, t) in ext if d == 3][0]
    # classify surfaces by bounding box (COM is unreliable for closed loops:
    # the outer cylinder's COM sits on the axis). A flat z-face has zero
    # z-extent -> frontAndBack; a sidewall spanning the full domain in xy is
    # the far field; a small-xy sidewall is the foil.
    front_back, foil_side, far_side = [], [], []
    for d, t in gmsh.model.getEntities(2):
        x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(d, t)
        if (z1 - z0) < mp.span * 0.5:                 # flat z-face (dz~0 vs span)
            front_back.append(t)
        elif max(abs(x0 - cx), abs(x1 - cx), abs(y0 - cy), abs(y1 - cy)) > mp.radius * 0.5:
            far_side.append(t)                        # spans out to far field
        else:
            foil_side.append(t)

    # gmshToFoam builds its patches purely from physical *surface* groups. An
    # empty group is not an error to gmsh, but it leaves OpenFOAM with zero
    # patches, and gmshToFoam then dies on UPtrList<polyPatch>::last() with
    # "Cannot dereference nullptr at index -1 in range [0,0)" -- a message that
    # says nothing about the real cause. Fail here instead, where the cause is
    # visible.
    groups = {"airfoil": foil_side, "farfield": far_side, "frontAndBack": front_back}
    empty = [name for name, tags in groups.items() if not tags]
    if empty:
        raise RuntimeError(
            f"surface classification produced empty physical group(s): {empty}. "
            f"Entities found: {len(gmsh.model.getEntities(2))} surfaces; "
            f"foil={len(foil_side)} far={len(far_side)} frontBack={len(front_back)}; "
            f"span={mp.span} radius={mp.radius} centre=({cx}, {cy}). "
            f"gmshToFoam needs all three."
        )

    gmsh.model.addPhysicalGroup(3, [vol], name="internal")
    gmsh.model.addPhysicalGroup(2, foil_side, name="airfoil")
    gmsh.model.addPhysicalGroup(2, far_side, name="farfield")
    gmsh.model.addPhysicalGroup(2, front_back, name="frontAndBack")

    gmsh.model.mesh.generate(3)
    # Must be set *before* write; gmshToFoam only parses MSH 2.2 ASCII, while
    # gmsh 4.x defaults to 4.1.
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.Binary", 0)
    gmsh.option.setNumber("Mesh.SaveAll", 0)  # physical groups only
    gmsh.write(str(out_msh))
    _verify_msh(out_msh)
    ncell = len(gmsh.model.mesh.getElements(3)[1][0]) if gmsh.model.mesh.getElements(3)[1] else 0
    gmsh.finalize()
    if ncell == 0:
        raise RuntimeError(f"{out_msh}: mesh has no 3-D cells")
    return out_msh


def _verify_msh(path) -> dict:
    """Check a written ``.msh`` is something gmshToFoam can actually consume.

    Catches the two failures that otherwise surface as an opaque OpenFOAM
    segfault-adjacent abort: wrong file format version, and missing physical
    names for the boundary patches.
    """
    from pathlib import Path

    text = Path(path).read_text(errors="ignore")
    lines = text.splitlines()

    version = None
    for i, line in enumerate(lines):
        if line.startswith("$MeshFormat") and i + 1 < len(lines):
            version = lines[i + 1].split()[0]
            break
    if version is None or not version.startswith("2"):
        raise RuntimeError(
            f"{path}: MSH format is {version!r}; gmshToFoam only reads 2.2. "
            f"Set Mesh.MshFileVersion = 2.2 before gmsh.write()."
        )

    if "$PhysicalNames" not in text:
        raise RuntimeError(
            f"{path}: no $PhysicalNames block -- gmshToFoam will create zero "
            f"patches and abort on UPtrList<polyPatch>::last()."
        )
    block = text.split("$PhysicalNames", 1)[1].split("$EndPhysicalNames", 1)[0]
    names = {
        part.split()[-1].strip('"')
        for part in block.strip().splitlines()[1:]
        if len(part.split()) >= 3
    }
    missing = {"airfoil", "farfield", "frontAndBack"} - names
    if missing:
        raise RuntimeError(f"{path}: physical names missing {sorted(missing)}; found {sorted(names)}")
    return {"version": version, "names": sorted(names)}
