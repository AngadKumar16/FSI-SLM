"""Build the CFD surrogate dataset for NACA 0009. Run from the repo root:

    python -m scripts.build_cfd_dataset
"""
from sensor.cfd.dataset import build_dataset

if __name__ == "__main__":
    build_dataset(
        ["Data/Airfoils/4/n0009sm.dat.txt"],
        out_dir="artifacts/cfd",
        max_workers=4,
    )