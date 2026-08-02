"""OpenFOAM (ESI v2606) case generation + running for the airfoil surrogate.

Pipeline per case (shape, AoA, Re):
  mesh.py     : foil polygon -> gmsh 2D mesh (1-cell z-extrude for OpenFOAM 2D)
  case.py     : write incompressible simpleFoam case (kOmegaSST, wall functions)
  run.py      : gmshToFoam -> checkMesh -> simpleFoam, extract Cl/Cd
  unsteady.py : the same, transient (pimpleFoam) -- time-averaged fields plus
                the measured vortex-shedding frequency. This is the branch that
                works past stall, where steady SIMPLE has no fixed point to
                converge to.
  sample.py   : sample U,p onto the fixed Cartesian grid -> .npz for the UNet
  adapt.py    : the shared packer both backends write through

Which solver for which incidence:

  alpha <= 8   steady RANS  -- converges, and is ~100x cheaper
  alpha >= 10  URANS        -- steady residuals stall at 1e-2 and lift
                              limit-cycles by up to +-0.43

All OpenFOAM commands run through the gerlero wrapper ``openfoam2606 -c '...'``.
"""
