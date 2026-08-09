#!/usr/bin/env python3
"""Reproducible driver: real DaVis PIV -> FeatureRecord -> figures + report.

Runs the real-data pipeline end to end and writes every artifact needed to
defend the result:

  * ``feature_record.json``   -- model-facing record (features.extract output)
  * ``analysis.json``         -- provenance + all measured PIV diagnostics
  * ``wake_probe_signal.json``-- the wake v'(t) probe (for data-quality screening)
  * ``figures/*.png``         -- TKE field, wake spectrum, POD energy, POD modes
  * a printed summary incl. the rule-based baseline interpretation

The heavy lifting lives in :mod:`features.piv_ingest`; this script only
orchestrates, plots, and reports so the numbers in the report are exactly the
numbers in the record.

Usage::

    python scripts/ingest_real_piv.py --export-dir Exports --out-dir artifacts/real_piv
    python scripts/ingest_real_piv.py --freestream 1.0 --chord 0.1   # sensitivity
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("USE_TF", "0")
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Data.schema import FlowRegime  # noqa: E402
from features.extract import check_physical_consistency  # noqa: E402
from features.piv_ingest import (  # noqa: E402
    DEFAULT_CHORD_M,
    DEFAULT_FREESTREAM_M_S,
    ingest_piv_directory,
    tke_field,
    wake_geometry,
    wake_mask,
)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 160,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return plt


def fig_tke_field(series, chord_m, geom, wake, out_path):
    plt = _style()
    tke = tke_field(series)
    xr = (series.X.min() / chord_m, series.X.max() / chord_m)
    yr = (series.Y.min() / chord_m, series.Y.max() / chord_m)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    im = ax.imshow(
        tke, origin="upper", extent=[xr[0], xr[1], yr[0], yr[1]],
        aspect="equal", cmap="viridis",
    )
    # wake region contour
    ax.contour(
        np.linspace(xr[0], xr[1], series.X.shape[1]),
        np.linspace(yr[1], yr[0], series.X.shape[0]),
        wake.astype(float), levels=[0.5], colors="white", linewidths=1.2, alpha=0.8,
    )
    ax.plot(geom["centroid_x"] / chord_m, geom["centroid_y"] / chord_m,
            "o", color="crimson", ms=7, label="wake centroid")
    ax.set_xlabel("x / c"); ax.set_ylabel("y / c")
    ax.set_title(f"Wake turbulent kinetic energy  (AoA {series.aoa_deg:g}°, "
                 f"{series.n_frames} PIV frames)")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label(r"TKE $= \frac{1}{2}(\overline{u'^2}+\overline{v'^2})$  [m$^2$/s$^2$]")
    ax.legend(loc="upper right", framealpha=0.85)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)


def fig_wake_spectrum(diag, out_path):
    plt = _style()
    f = np.asarray(diag["spectrum"]["freqs"])
    P = np.asarray(diag["spectrum"]["power"])
    fpk = diag["shedding"]["shedding_frequency_hz"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.semilogy(f[1:], P[1:], color="#2b6cb0", lw=1.6)
    ax.axvline(fpk, color="crimson", ls="--", lw=1.4,
               label=f"$f_{{shed}}$ = {fpk:.2f} Hz  (peak/median = {diag['shedding']['peak_factor']:.1f})")
    ax.set_xlabel("frequency [Hz]"); ax.set_ylabel(r"wake-averaged $v'$ power [a.u.]")
    ax.set_title("Wake vortex-shedding spectrum  (broadband → separated flow)")
    ax.set_xlim(0, f.max())
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)


def fig_pod_energy(frac, out_path):
    plt = _style()
    k = np.arange(1, len(frac) + 1)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars = ax.bar(k, 100 * np.asarray(frac), color="#2b6cb0", width=0.7)
    bars[0].set_color("crimson")
    ax.set_xlabel("POD mode"); ax.set_ylabel("energy fraction [%]")
    ax.set_title(f"POD spectrum — flat & low (mode 1 = {100*frac[0]:.1f}%): "
                 "no coherent shedding pair")
    ax.set_xticks(k)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)


def fig_pod_modes(series, chord_m, out_path, n_modes=4):
    """Spatial POD mode shapes (streamwise component) on the common-valid grid."""
    plt = _style()
    T, H, W = series.U.shape
    common = series.common_valid_mask(min_frac=1.0)
    if common.sum() == 0:
        common = series.common_valid_mask(min_frac=0.95)
    idx = np.where(common.ravel())[0]
    U = series.U.reshape(T, -1)[:, idx].astype(float)
    V = series.V.reshape(T, -1)[:, idx].astype(float)
    snap = np.hstack([U, V])
    snap -= snap.mean(0, keepdims=True)
    _, s, Vt = np.linalg.svd(snap, full_matrices=False)
    frac = (s ** 2) / (s ** 2).sum()
    N = idx.size
    xr = (series.X.min() / chord_m, series.X.max() / chord_m)
    yr = (series.Y.min() / chord_m, series.Y.max() / chord_m)
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.0))
    for m, ax in enumerate(axes.ravel()):
        mode_u = np.full(H * W, np.nan)
        mode_u[idx] = Vt[m, :N]
        grid = mode_u.reshape(H, W)
        vmax = np.nanmax(np.abs(grid)) or 1.0
        im = ax.imshow(grid, origin="upper", extent=[xr[0], xr[1], yr[0], yr[1]],
                       aspect="equal", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"POD mode {m+1}  ({100*frac[m]:.1f}% energy)")
        ax.set_xlabel("x / c"); ax.set_ylabel("y / c"); ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Leading POD spatial modes (streamwise fluctuation u′)", y=0.99)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
    return [float(x) for x in frac[:12]]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run(
    export_dir: str = "Exports",
    out_dir: str = "results/real_piv",
    chord: float = DEFAULT_CHORD_M,
    freestream: float = DEFAULT_FREESTREAM_M_S,
    max_frames=None,
    no_figures: bool = False,
) -> dict:
    """Ingest a real PIV directory and write all artifacts. Returns a summary dict."""
    out = Path(out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    print(f"[piv] loading + ingesting {export_dir} ...")
    fr, raw, series, diag = ingest_piv_directory(
        export_dir, chord_m=chord, freestream_m_s=freestream, max_frames=max_frames,
    )

    # --- persist record + provenance ---
    (out / "feature_record.json").write_text(fr.model_dump_json(indent=2))
    (out / "wake_probe_signal.json").write_text(
        json.dumps({"wake_velocity_probe_signal_m_s": raw.wake_velocity_probe_signal,
                    "sampling_rate_hz": raw.sampling_rate_hz})
    )

    range_warn = fr.range_warnings()
    phys_warn = check_physical_consistency(fr)

    # --- instrumentation screening (trust layer, no model) ---
    from features.noise import analyse_raw_experiment
    noise_report = analyse_raw_experiment(raw).to_dict()

    # --- rule-based baseline (independent of the LoRA model) ---
    from Data.describe import classify_regime_confidence, describe
    regime, confidence, weights = classify_regime_confidence(fr)
    baseline = {
        "regime": regime.value,
        "confidence": round(float(confidence), 3),
        "top_features": dict(sorted(weights.items(), key=lambda kv: -abs(kv[1]))[:5]),
        "text": describe(fr),
    }

    frac12 = None
    if not no_figures:
        print("[piv] rendering figures ...")
        wake = wake_mask(series)
        geom = wake_geometry(series, chord, wake=wake)
        fig_tke_field(series, chord, geom, wake, out / "figures" / "wake_tke_field.png")
        fig_wake_spectrum(diag, out / "figures" / "wake_spectrum.png")
        frac12 = fig_pod_modes(series, chord, out / "figures" / "pod_modes.png")
        fig_pod_energy(frac12, out / "figures" / "pod_energy.png")

    analysis = {
        "provenance": {
            "source_dir": series.source_dir,
            "run_id": series.run_id,
            "n_frames": diag["n_frames"],
            "fs_hz": diag["fs_hz"],
            "duration_s": diag["duration_s"],
            "grid_wxh": diag["grid"],
            "dx_mm": diag["dx_mm"],
            "davis_note": "mean-subtracted fluctuation fields (Originator 'Vector Arithmetic')",
        },
        "rig_inputs": {
            "angle_of_attack_deg": diag["aoa_deg"],
            "freestream_velocity_m_s": freestream,
            "chord_m": chord,
            "reynolds_number": diag["reynolds_number"],
            "note": "U and c are rig inputs; not recoverable from mean-subtracted PIV.",
        },
        "measured_from_piv": {
            "shedding_frequency_hz": diag["shedding"]["shedding_frequency_hz"],
            "shedding_peak_over_median": diag["shedding"]["peak_factor"],
            "probe_xy_m": diag["shedding"]["probe_xy"],
            "strouhal_number": fr.piv.strouhal_number,
            "wake_width_over_c": diag["wake_geometry"]["wake_width_over_c"],
            "wake_length_over_c": diag["wake_geometry"]["wake_length_over_c"],
            "tke_dimensional_m2_s2": diag["tke"]["tke_dimensional_m2_s2"],
            "turbulent_kinetic_energy_norm": fr.piv.turbulent_kinetic_energy,
            "turbulence_intensity": diag["tke"]["turbulence_intensity"],
            "pod_mode_amplitudes": fr.piv.pod_mode_amplitudes,
            "pod_energy_fractions_top12": frac12,
        },
        "estimated": {"separation_location_x_over_c": fr.piv.separation_location_x_over_c,
                      "note": diag["separation_note"]},
        "absent_channels": {
            "reason": "PIV-only export; structural displacement and Cl/Cd/Cm come from "
                      "the load-cell and video-strain subsystems, not this DaVis export.",
            "fields": ["structural_displacement", "natural_frequency", "cl", "cd", "cm"],
        },
        "validation": {
            "range_warnings": range_warn,
            "physics_consistency_warnings": phys_warn,
            "lock_in_status": fr.lock_in_status().value,
            "instrumentation_screen": {
                "verdict": noise_report["verdict"],
                "quality_score": noise_report["quality_score"],
                "flags": noise_report["flags"],
                "note": "flatline flags on structural/Cl/Cd/Cm channels are expected: "
                        "those channels are not in a PIV-only export.",
            },
        },
        "baseline_rule_based_interpretation": baseline,
    }
    (out / "analysis.json").write_text(json.dumps(analysis, indent=2))

    # --- summary ---
    print("\n" + "=" * 70)
    print(f"REAL PIV INGEST  —  {fr.experiment_id}")
    print("=" * 70)
    print(f"frames={diag['n_frames']}  fs={diag['fs_hz']:.2f} Hz  dur={diag['duration_s']:.1f} s"
          f"  grid={diag['grid'][0]}x{diag['grid'][1]}  AoA={diag['aoa_deg']:g}°")
    print(f"Re={diag['reynolds_number']:.2e}  U={freestream:.3f} m/s  c={chord:g} m (rig inputs)")
    print(f"f_shed={fr.piv.shedding_frequency_hz:.2f} Hz  St={fr.piv.strouhal_number:.3f}"
          f"  wake_width={fr.piv.wake_width_over_c:.2f}c  TKE*={fr.piv.turbulent_kinetic_energy:.4f}")
    print(f"POD mode-1 energy={100*fr.piv.pod_mode_amplitudes[0]:.1f}%  lock-in={fr.lock_in_status().value}")
    print(f"range warnings: {range_warn or 'none'}")
    print(f"physics warnings: {phys_warn or 'none'}")
    print(f"instrumentation screen: {noise_report['verdict']} (score {noise_report['quality_score']}, "
          f"{len(noise_report['flags'])} flags; flatlines = absent channels)")
    print(f"rule-based baseline regime: {baseline['regime']} (conf {baseline['confidence']})")
    print(f"\nartifacts -> {out}/  (feature_record.json, analysis.json, figures/)")
    return {
        "experiment_id": fr.experiment_id,
        "regime_baseline": baseline["regime"],
        "shedding_frequency_hz": fr.piv.shedding_frequency_hz,
        "strouhal_number": fr.piv.strouhal_number,
        "out_dir": str(out),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export-dir", default="Exports")
    ap.add_argument("--out-dir", default="results/real_piv")
    ap.add_argument("--chord", type=float, default=DEFAULT_CHORD_M)
    ap.add_argument("--freestream", type=float, default=DEFAULT_FREESTREAM_M_S)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    run(
        export_dir=args.export_dir, out_dir=args.out_dir, chord=args.chord,
        freestream=args.freestream, max_frames=args.max_frames, no_figures=args.no_figures,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
