"""Write an incompressible steady ``simpleFoam`` case for one foil condition.

Mesh is nondimensional (chord c = 1), so we set U = 1 and nu = 1/Re; forces
come out as coefficients directly. Geometry is pre-rotated by AoA in the mesh,
so the freestream is axial: Uinf = (1, 0, 0). Turbulence: k-omega SST with wall
functions (high-Re, y+ ~ 30). Far field uses OpenFOAM ``freestream`` BCs
(inlet/outlet decided by local flux) -- ideal for the circular external domain.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

HEADER = """/*--------------------------------*- C++ -*----------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    object      {obj};
}}
"""


@dataclass
class CaseParams:
    reynolds: float = 5.0e5
    span: float = 0.1          # must match MeshParams.span
    n_iter: int = 2000
    turb_intensity: float = 0.05
    turb_length: float = 0.1   # turbulence length scale (chords)


def _f(path: Path, cls: str, obj: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER.format(cls=cls, obj=obj) + body + "\n")


def write_case(case_dir: str | Path, cp: CaseParams | None = None) -> Path:
    cp = cp or CaseParams()
    d = Path(case_dir)
    nu = 1.0 / cp.reynolds
    U = 1.0
    k = 1.5 * (cp.turb_intensity * U) ** 2
    omega = k ** 0.5 / (0.09 ** 0.25 * cp.turb_length)
    nut = k / omega

    # ---- constant ----
    _f(d / "constant/transportProperties", "dictionary", "transportProperties",
       f"transportModel Newtonian;\nnu {nu:.8g};")
    _f(d / "constant/turbulenceProperties", "dictionary", "turbulenceProperties",
       "simulationType RAS;\nRAS {{ RASModel kOmegaSST; turbulence on; printCoeffs on; }}".replace("{{", "{").replace("}}", "}"))

    # ---- 0/ fields ----
    def field(obj, cls, dims, internal, bcs):
        _f(d / "0" / obj, cls, obj,
           f"dimensions {dims};\ninternalField uniform {internal};\nboundaryField {{\n{bcs}\n}}")

    fs_U = ("    farfield { type freestreamVelocity; freestreamValue uniform (1 0 0); }\n"
            "    airfoil { type noSlip; }\n"
            "    frontAndBack { type empty; }")
    fs_p = ("    farfield { type freestreamPressure; freestreamValue uniform 0; }\n"
            "    airfoil { type zeroGradient; }\n"
            "    frontAndBack { type empty; }")
    field("U", "volVectorField", "[0 1 -1 0 0 0 0]", "(1 0 0)", fs_U)
    field("p", "volScalarField", "[0 2 -2 0 0 0 0]", "0", fs_p)
    field("k", "volScalarField", "[0 2 -2 0 0 0 0]", f"{k:.6g}",
          f"    farfield {{ type inletOutlet; inletValue uniform {k:.6g}; value uniform {k:.6g}; }}\n"
          "    airfoil { type kqRWallFunction; value uniform 1e-12; }\n"
          "    frontAndBack { type empty; }")
    field("omega", "volScalarField", "[0 0 -1 0 0 0 0]", f"{omega:.6g}",
          f"    farfield {{ type inletOutlet; inletValue uniform {omega:.6g}; value uniform {omega:.6g}; }}\n"
          "    airfoil { type omegaWallFunction; value uniform 1e-12; }\n"
          "    frontAndBack { type empty; }")
    field("nut", "volScalarField", "[0 2 -1 0 0 0 0]", f"{nut:.6g}",
          "    farfield { type calculated; value uniform 0; }\n"
          "    airfoil { type nutkWallFunction; value uniform 0; }\n"
          "    frontAndBack { type empty; }")

    # ---- system ----
    _f(d / "system/controlDict", "dictionary", "controlDict",
       "application simpleFoam;\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\n"
       f"endTime {cp.n_iter};\ndeltaT 1;\nwriteControl timeStep;\nwriteInterval {cp.n_iter};\n"
       "purgeWrite 1;\nwriteFormat ascii;\nwritePrecision 8;\nrunTimeModifiable true;\n"
       "functions {\n"
       "  forceCoeffs {\n"
       "    type forceCoeffs; libs (forces); writeControl timeStep; writeInterval 1;\n"
       "    patches (airfoil); rho rhoInf; rhoInf 1; magUInf 1;\n"
       "    liftDir (0 1 0); dragDir (1 0 0); pitchAxis (0 0 1);\n"
       f"    CofR (0.25 0 0); lRef 1; Aref {cp.span};\n"
       "  }\n"
       "  yPlus { type yPlus; libs (fieldFunctionObjects); writeControl writeTime; }\n"
       "}")
    _f(d / "system/fvSchemes", "dictionary", "fvSchemes",
       "ddtSchemes { default steadyState; }\n"
       "gradSchemes { default Gauss linear; }\n"
       "divSchemes {\n"
       "  default none;\n"
       "  div(phi,U) bounded Gauss linearUpwind grad(U);\n"
       "  div(phi,k) bounded Gauss upwind;\n"
       "  div(phi,omega) bounded Gauss upwind;\n"
       "  div((nuEff*dev2(T(grad(U))))) Gauss linear;\n"
       "}\n"
       "laplacianSchemes { default Gauss linear corrected; }\n"
       "interpolationSchemes { default linear; }\n"
       "snGradSchemes { default corrected; }\n"
       "wallDist { method meshWave; }")
    _f(d / "system/fvSolution", "dictionary", "fvSolution",
       "solvers {\n"
       "  p { solver GAMG; tolerance 1e-7; relTol 0.01; smoother GaussSeidel; }\n"
       "  \"(U|k|omega)\" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.1; }\n"
       "}\n"
       "SIMPLE { nNonOrthogonalCorrectors 1; consistent yes;\n"
       "  residualControl { p 1e-5; U 1e-5; k 1e-5; omega 1e-5; } }\n"
       "relaxationFactors { equations { U 0.9; \".*\" 0.9; } }")
    return d
