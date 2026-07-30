from sensor.cfd.dataset import build_dataset

if __name__ == "__main__":
    build_dataset(
        ["Data/Airfoils/4/n0009sm.dat.txt"],
        out_dir="artifacts/cfd",
        max_workers=4,
    )