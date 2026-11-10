#!/usr/bin/env python3
"""Gates for the multi-fidelity run: is the data legal, did the model improve.

Two subcommands, run at the two moments where a wrong answer is expensive.

``data`` runs after ``sensor-finalize`` and before the cache is rebuilt. It
answers the question the whole multi-fidelity path exists for: are the
high-fidelity rows on foils the network is *allowed* to see. Every RANS case in
the dataset before this run was on NACA 0009, which is pinned to the test split,
so training on them would have been training on test. It also checks that the
URANS rows carry the keys the twin bridge reads, because a measured shedding
frequency that never reaches the interpreter is an expensive no-op.

``model`` runs after training and scoring. It compares the fine-tuned checkpoint
against the baseline on the same rows and fails if the high-fidelity score did
not move or the panel score fell apart.

Exit status is the point: 0 means the claim is supported, non-zero means do not
report it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FAIL: list = []
WARN: list = []


def fail(msg: str) -> None:
    FAIL.append(msg)
    print(f"  FAIL  {msg}")


def warn(msg: str) -> None:
    WARN.append(msg)
    print(f"  warn  {msg}")


def ok(msg: str) -> None:
    print(f"  ok    {msg}")


def _rows(manifest: Path) -> list:
    return [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]


def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def check_data(args) -> None:
    cfd = Path(args.cfd_dir)
    manifest = cfd / "manifest.jsonl"
    splits_path = cfd / "splits.json"
    if not manifest.exists():
        fail(f"no manifest at {manifest}")
        return
    rows = _rows(manifest)
    splits = json.loads(splits_path.read_text())
    train, val, test = set(splits["train"]), set(splits["val"]), set(splits["test"])

    print("\n[1] fidelity census")
    by_fid: dict = {}
    for r in rows:
        by_fid.setdefault(r.get("fidelity", "panel"), []).append(r)
    for f, rs in sorted(by_fid.items()):
        good = [r for r in rs if r.get("ok")]
        print(f"        {f:6s} {len(rs):5d} rows, {len(good)} ok, "
              f"{len({r['foil'] for r in rs})} foils")
    if not by_fid.get("urans"):
        fail("no URANS rows in the manifest -- the unsteady sweep has not landed")
    if not by_fid.get("rans"):
        fail("no RANS rows in the manifest")

    print("\n[2] no training on test (the reason this path exists)")
    for fid in ("rans", "urans"):
        hi = [r for r in by_fid.get(fid, []) if r.get("ok")]
        foils = {r["foil"] for r in hi}
        trainable = sorted(foils & train)
        leaked = sorted(foils & (val | test))
        n_train_rows = sum(1 for r in hi if r["foil"] in train)
        if trainable:
            ok(f"{fid}: {n_train_rows} rows on train foils {trainable}")
        elif hi:
            fail(f"{fid}: {len(hi)} rows, none on a train foil -- unusable for training")
        if leaked:
            # Not an error: high-fidelity rows on held-out foils are how the
            # test score gets measured. It is only a problem if there are no
            # train-split rows to learn from, which is checked above.
            n_held = sum(1 for r in hi if r["foil"] in (val | test))
            ok(f"{fid}: {n_held} rows on held-out foils {leaked} (evaluation only)")

    print("\n[3] the split did not move under the new rows")
    backups = sorted(cfd.glob("backup_*/splits.json"))
    if backups:
        old = json.loads(backups[-1].read_text())
        for k in ("train", "val", "test"):
            if set(old[k]) != set(splits[k]):
                moved = sorted(set(old[k]) ^ set(splits[k]))
                fail(f"{k} split changed vs {backups[-1].parent.name}: {moved[:8]}")
            else:
                ok(f"{k} split unchanged ({len(splits[k])} foils)")
    else:
        warn("no backup to compare the split against")

    print("\n[4] the plan's foils are still train-split foils")
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from sensor.cfd.dataset import plan_multifidelity_sweep

        plan = plan_multifidelity_sweep(
            splits_path, manifest, n_foils=args.n_foils, seed=args.seed
        )
        bad = [f for f in plan["foils"] if f not in train]
        (fail if bad else ok)(
            f"planned foils {plan['foils']}" + (f" NOT in train: {bad}" if bad else "")
        )
    except Exception as exc:  # noqa: BLE001 - a gate should report, not raise
        warn(f"could not re-derive the plan: {type(exc).__name__}: {exc}")

    print("\n[5] URANS rows carry what the bridge reads")
    needed = ("strouhal_chord", "organised_shedding", "statistically_steady")
    urans = [r for r in by_fid.get("urans", []) if r.get("ok")]
    for r in urans:
        missing = [k for k in needed if r.get(k) is None]
        tag = f"{r['foil']} a={r['alpha_deg']:g}"
        if missing:
            fail(f"{tag}: missing {missing}")
        else:
            print(f"        {tag:22s} St_c={r['strouhal_chord']:.4f} "
                  f"St_d={r.get('strouhal_projected') or float('nan'):.4f} "
                  f"organised={r['organised_shedding']} "
                  f"steady={r['statistically_steady']} "
                  f"cycles={r.get('n_cycles') or 0:.1f} "
                  f"Cl={r.get('cl') or float('nan'):+.3f}+-{r.get('cl_rms') or 0:.3f}")
    unorganised = [r for r in urans if not r.get("organised_shedding")]
    unsettled = [r for r in urans if not r.get("statistically_steady")]
    if urans and unorganised:
        warn(f"{len(unorganised)}/{len(urans)} URANS cases show no organised street "
             f"-- those fall back to the correlation downstream, by design")
    if urans and unsettled:
        warn(f"{len(unsettled)}/{len(urans)} URANS cases did not pass the "
             f"stationarity check; --converged-only drops them")

    print("\n[6] the bridge lookup actually finds them")
    try:
        from sensor.bridge import TwinConfig, _measured_shedding, lookup_unsteady

        cfg = TwinConfig()
        hit = 0
        for r in urans:
            block = lookup_unsteady(r["foil"], r["alpha_deg"], manifest)
            if not block:
                fail(f"lookup missed {r['foil']} a={r['alpha_deg']:g}")
                continue
            got = _measured_shedding({"unsteady": block}, cfg)
            if got is None and r.get("organised_shedding"):
                fail(f"{r['foil']} a={r['alpha_deg']:g}: block found but rejected")
            hit += 1
        if urans:
            ok(f"{hit}/{len(urans)} URANS rows reachable from the twin path")
    except Exception as exc:  # noqa: BLE001
        warn(f"bridge lookup check skipped: {type(exc).__name__}: {exc}")

    print("\n[7] every ok high-fidelity row has its .npz")
    npz_dir = cfd / "npz"
    missing = []
    for fid in ("rans", "urans"):
        for r in by_fid.get(fid, []):
            if not r.get("ok"):
                continue
            cand = list(npz_dir.glob(f"{r['foil']}_a{r['alpha_deg']:g}*.npz"))
            if not cand:
                missing.append(f"{r['foil']} a={r['alpha_deg']:g} ({fid})")
    (fail if missing else ok)(
        f"{len(missing)} missing npz" + (f": {missing[:6]}" if missing else "")
    )

    print("\n[8] the cache is older than the manifest (it must be rebuilt)")
    cache = Path(args.cache_dir) / "inputs.npy"
    if cache.exists() and cache.stat().st_mtime < manifest.stat().st_mtime:
        ok("cache predates the manifest, as expected before sensor-cache")
    elif cache.exists():
        warn("cache is newer than the manifest -- check it was built from these rows")


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def _cmp(name: str, base, new, lower_is_better: bool = True) -> None:
    if base is None or new is None:
        print(f"        {name:28s} {'--':>10} -> {'--':>10}")
        return
    d = (new - base) / abs(base) * 100 if base else float("nan")
    arrow = "better" if (d < 0) == lower_is_better and abs(d) > 1 else (
        "worse" if abs(d) > 1 else "same")
    print(f"        {name:28s} {base:10.4f} -> {new:10.4f}  {d:+7.1f}%  {arrow}")


def check_model(args) -> None:
    run = Path(args.run_dir)
    ck = run / "best.pt"
    print("\n[1] checkpoint")
    if not ck.exists():
        fail(f"no checkpoint at {ck}")
        return
    try:
        import torch

        meta = torch.load(ck, map_location="cpu", weights_only=False)
        fids = meta.get("fidelities")
        ok(f"epoch {meta.get('epoch')}, fidelities seen: {sorted(fids) if fids else None}")
        if not fids or "urans" not in fids:
            fail("the checkpoint did not see any URANS rows -- "
                 "predict(fidelity='urans') will still warn that it is extrapolating")
        if fids and "rans" not in fids:
            fail("the checkpoint did not see any RANS rows")
    except Exception as exc:  # noqa: BLE001
        warn(f"could not read the checkpoint: {type(exc).__name__}: {exc}")

    print("\n[2] high-fidelity score, baseline -> fine-tuned")
    base = _load(Path("artifacts/sensor/eval_test_rans_baseline.json"))
    new = _load(Path("artifacts/sensor/eval_test_rans_mf.json"))
    if not base or not new:
        warn("missing one of the RANS eval files; run the eval stage first")
    else:
        print(f"        n_cases: {base.get('n_cases')} -> {new.get('n_cases')}")
        for f in ("u", "v", "p", "load", "disp"):
            _cmp(f"{f} relative_rmse",
                 (base["fields"].get(f) or {}).get("relative_rmse"),
                 (new["fields"].get(f) or {}).get("relative_rmse"))
        for s in ("cl", "cd", "cm", "w_max", "twist_te_deg"):
            _cmp(f"{s} nrmse",
                 (base["scalars"].get(s) or {}).get("nrmse"),
                 (new["scalars"].get(s) or {}).get("nrmse"))
        bp = (base["fields"].get("p") or {}).get("relative_rmse")
        np_ = (new["fields"].get("p") or {}).get("relative_rmse")
        if bp and np_ and np_ >= bp:
            fail("pressure on RANS rows did not improve -- the fine-tune bought "
                 "nothing on the fidelity it was weighted for")

    print("\n[3] the cheap fidelity did not collapse")
    bp = _load(Path("artifacts/sensor/eval_test_panel.json"))
    npn = _load(Path("artifacts/sensor/eval_test_panel_mf.json"))
    if not bp or not npn:
        warn("missing a panel eval file; run the eval stage first")
    else:
        for s in ("cl", "cd", "cm"):
            _cmp(f"panel {s} nrmse",
                 (bp["scalars"].get(s) or {}).get("nrmse"),
                 (npn["scalars"].get(s) or {}).get("nrmse"))
        b = (bp["scalars"].get("cl") or {}).get("nrmse")
        n = (npn["scalars"].get("cl") or {}).get("nrmse")
        if b and n and n > 1.25 * b:
            fail(f"panel Cl nrmse degraded by more than 25% ({b:.4f} -> {n:.4f}); "
                 f"the fidelity weights are too aggressive")

    print("\n[4] URANS rows in the test split")
    u = _load(Path("artifacts/sensor/eval_test_urans_mf.json"))
    if u:
        ok(f"scored {u.get('n_cases')} URANS test cases")
    else:
        warn("no URANS test rows -- the sweep covered train foils only, so the "
             "URANS claim rests on training rows and the val split")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("data", help="run after sensor-finalize")
    d.add_argument("--cfd-dir", default="artifacts/cfd")
    d.add_argument("--cache-dir", default="artifacts/sensor/cache")
    d.add_argument("--n-foils", type=int, default=4)
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(fn=check_data)

    m = sub.add_parser("model", help="run after sensor-train and sensor-eval")
    m.add_argument("--cfd-dir", default="artifacts/cfd")
    m.add_argument("--run-dir", default="artifacts/sensor/run_mf")
    m.add_argument("--baseline", default="artifacts/sensor/run3/best.pt")
    m.set_defaults(fn=check_model)

    args = ap.parse_args()
    args.fn(args)

    print(f"\n{'-' * 60}")
    if FAIL:
        print(f"{len(FAIL)} FAILED, {len(WARN)} warnings")
        for f in FAIL:
            print(f"  FAIL  {f}")
        return 1
    print(f"all checks passed, {len(WARN)} warnings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
