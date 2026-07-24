"""OpenFOAM (ESI v2606) case generation + running for the airfoil surrogate.

Pipeline per case (shape, AoA, Re):
  mesh.py   : foil polygon -> gmsh 2D mesh (1-cell z-extrude for OpenFOAM 2D)
  case.py    : write incompressible simpleFoam case (kOmegaSST, wall functions)
  run.py     : gmshToFoam -> checkMesh -> simpleFoam, extract Cl/Cd
  sample.py  : sample U,p onto the fixed Cartesian grid -> .npz for the UNet

All OpenFOAM commands run through the gerlero wrapper ``openfoam2606 -c '...'``.
"""
