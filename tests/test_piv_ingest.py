"""Hermetic tests for the real-PIV ingestion pipeline (features/piv_ingest.py).

These synthesize a handful of DaVis ``DSExport`` CSV frames with a *known* wake
oscillation and assert that the loader + feature extraction recover it and emit a
schema-valid FeatureRecord through the unchanged features.extract path. No
dependency on the real ``Exports/`` directory.
"""

from __future__ import annotations

import numpy as np
import pytest

from Data.schema import FeatureRecord, LockInStatus
from features.piv_ingest import (
    ingest_piv_directory,
    load_piv_series,
    parse_frame_header,
)

# Synthetic acquisition / geometry
FS_HZ = 50.0
N_FRAMES = 100          # df = fs/N = 0.5 Hz -> f_shed lands on a bin
F_SHED_HZ = 5.0
WIDTH, HEIGHT = 12, 10
DX = 0.01               # metres
AOA_DEG = 5.0
CHORD_M = 0.1
FREESTREAM = 1.0        # -> target St = f*c/U = 0.5


def _write_frame(path, frame_idx, t_seconds):
    """Write one DaVis-format fluctuation frame with a travelling wake wave."""
    ys = (HEIGHT - 1 - np.arange(HEIGHT)) * DX  # row 0 is largest Y (as in DaVis)
    xs = np.arange(WIDTH) * DX
    XX, YY = np.meshgrid(xs, ys)                 # (H, W)
    rng = np.random.default_rng(1000 + frame_idx)
    # Energetic wake band in the middle rows; travelling transverse wave at F_SHED.
    band = np.exp(-((YY - ys.mean()) ** 2) / (2 * (2 * DX) ** 2))  # (H,W) weight
    phase = 2 * np.pi * F_SHED_HZ * t_seconds - 2 * np.pi * XX / (6 * DX)
    v = band * np.sin(phase) + 0.02 * rng.standard_normal((HEIGHT, WIDTH))
    u = 0.02 * rng.standard_normal((HEIGHT, WIDTH))
    lines = [
        ">>*HEADER*<<",
        "FileID:DSExport.CSV",
        "Version:2",
        f"GridSize:{{Width={WIDTH}, Height={HEIGHT}}}",
        f"Originator:AOA{AOA_DEG:.3f} TEST.Run 00:00:00.Vector Arithmetic",
        "CameraInfo:Test",
        f"TimeStamp:#{frame_idx + 1}, {t_seconds} s",
        ">>*DATA*<<",
        '"X (m)[m]","Y (m)[m]","Z (m)[m]","Status[-]","U[m/s]","V[m/s]","Length[m/s]"',
    ]
    for j in range(HEIGHT):
        for i in range(WIDTH):
            uu, vv = float(u[j, i]), float(v[j, i])
            lines.append(
                f"{XX[j, i]:.6f},{YY[j, i]:.6f},0,\"Valid\","
                f"{uu:.6f},{vv:.6f},{np.hypot(uu, vv):.6f}"
            )
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def piv_dir(tmp_path):
    d = tmp_path / "Exports"
    d.mkdir()
    for i in range(N_FRAMES):
        t = i / FS_HZ
        _write_frame(d / f"Export.test.{i:06d}-{t * 1000:.4f}.csv", i, t)
    return d


def test_header_parse(piv_dir):
    f0 = sorted(piv_dir.glob("Export.*.csv"))[0]
    hdr = parse_frame_header(str(f0))
    assert (hdr["width"], hdr["height"]) == (WIDTH, HEIGHT)
    assert hdr["aoa_deg"] == pytest.approx(AOA_DEG)


def test_load_series_shapes_and_timebase(piv_dir):
    s = load_piv_series(str(piv_dir))
    assert s.U.shape == (N_FRAMES, HEIGHT, WIDTH)
    assert s.aoa_deg == pytest.approx(AOA_DEG)
    assert s.fs_hz == pytest.approx(FS_HZ, rel=1e-3)
    # DaVis ordering: row 0 is the largest Y.
    assert s.Y[0, 0] > s.Y[-1, 0]


def test_ingest_recovers_shedding_and_is_schema_valid(piv_dir):
    fr, raw, series, diag = ingest_piv_directory(
        str(piv_dir), chord_m=CHORD_M, freestream_m_s=FREESTREAM
    )
    assert isinstance(fr, FeatureRecord)

    # Shedding frequency recovered within one spectral bin (df = 0.5 Hz).
    assert fr.piv.shedding_frequency_hz == pytest.approx(F_SHED_HZ, abs=1.0)

    # Strouhal is internally consistent with the recovered f_shed and the rig inputs.
    st_expected = fr.piv.shedding_frequency_hz * CHORD_M / FREESTREAM
    assert fr.piv.strouhal_number == pytest.approx(st_expected, rel=1e-6)

    # POD ran on the real fluctuation stack and produced valid energy fractions.
    assert len(fr.piv.pod_mode_amplitudes) >= 1
    assert all(0.0 <= a <= 1.0 for a in fr.piv.pod_mode_amplitudes)

    # Wake geometry is finite and positive.
    assert fr.piv.wake_width_over_c > 0.0
    assert diag["wake_geometry"]["wake_length_over_c"] > 0.0


def test_absent_structural_channel_is_flagged_not_faked(piv_dir):
    fr, raw, *_ = ingest_piv_directory(
        str(piv_dir), chord_m=CHORD_M, freestream_m_s=FREESTREAM
    )
    # No structural channel in a PIV-only export -> lock-in cannot be asserted.
    assert fr.lock_in_status() == LockInStatus.UNLOCKED
    assert fr.structure.natural_frequency_hz == 0.0
    assert fr.structure.rms_amplitude_over_c == 0.0
    assert fr.forces.cl_mean == 0.0
    # The record honestly flags the un-set reduced velocity rather than inventing it.
    assert any("reduced_velocity" in w for w in fr.range_warnings())
    # camber is a real measurement for the symmetric NACA0009 (0), not a placeholder.
    assert fr.structure.camber == 0.0


def test_subsampling_preserves_pod_leading_fraction(piv_dir):
    """POD leading energy fraction is stable under the point-count cap."""
    from features.piv_ingest import pod_snapshot_stack

    series = load_piv_series(str(piv_dir))
    full, _ = pod_snapshot_stack(series, max_points=10_000)
    sub, _ = pod_snapshot_stack(series, max_points=90)  # stride 2 (still above Nyquist)

    def lead(stack):
        s = stack - stack.mean(0, keepdims=True)
        sv = np.linalg.svd(s, full_matrices=False, compute_uv=False)
        return (sv[0] ** 2) / (sv ** 2).sum()

    assert lead(sub) == pytest.approx(lead(full), abs=0.15)
