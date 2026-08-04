"""The sweep driver must not lose or repeat hours of CFD.

Both properties are cheap to state and were both wrong before: rows were only
written after the last case finished, and a re-run re-solved everything. These
tests use a fake worker, so they exercise the driver without OpenFOAM.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sensor.cfd.sweep import FIDELITY_RANK, foil_name, run_sweep, solved_keys


def _row(foil: str, alpha: float, fidelity: str = "rans", ok: bool = True) -> dict:
    return {"foil": foil, "alpha_deg": alpha, "fidelity": fidelity, "ok": ok,
            "error": None}


def _worker(fidelity: str, seen: list):
    def fn(path, alpha):
        seen.append((foil_name(path), alpha))
        return _row(foil_name(path), alpha, fidelity)
    return fn


def test_rows_are_on_disk_before_the_sweep_finishes(tmp_path: Path):
    rows_dir = tmp_path / "rows"

    def fn(path, alpha):
        if alpha == 20.0:
            raise RuntimeError("solver died on the last case")
        return _row(foil_name(path), alpha, "urans")

    jobs = [(Path("n0012.dat.txt"), a) for a in (12.0, 16.0, 20.0)]
    with pytest.raises(RuntimeError):
        run_sweep(jobs, fn, rows_dir, prefix="urans", fidelity="urans")

    # The two cases that succeeded survived the crash of the third.
    written = [json.loads(l) for f in rows_dir.glob("*.jsonl")
               for l in f.read_text().splitlines() if l.strip()]
    assert sorted(r["alpha_deg"] for r in written) == [12.0, 16.0]


def test_resume_skips_cases_already_solved_at_this_fidelity(tmp_path: Path):
    rows_dir = tmp_path / "rows"
    rows_dir.mkdir(parents=True)
    (rows_dir / "urans_1.jsonl").write_text(
        json.dumps(_row("n0012", 12.0, "urans")) + "\n"
    )

    seen: list = []
    jobs = [(Path("n0012.dat.txt"), a) for a in (12.0, 16.0)]
    rows, _chunk, skipped = run_sweep(
        jobs, _worker("urans", seen), rows_dir, prefix="urans", fidelity="urans"
    )
    assert seen == [("n0012", 16.0)]
    assert len(skipped) == 1
    assert [r["alpha_deg"] for r in rows] == [16.0]


def test_urans_does_not_treat_a_steady_row_as_solved(tmp_path: Path):
    """A RANS row at 12 degrees is exactly what the unsteady sweep is for."""
    rows_dir = tmp_path / "rows"
    rows_dir.mkdir(parents=True)
    (rows_dir / "rans_1.jsonl").write_text(
        json.dumps(_row("n0012", 12.0, "rans")) + "\n"
    )

    seen: list = []
    run_sweep([(Path("n0012.dat.txt"), 12.0)], _worker("urans", seen), rows_dir,
              prefix="urans", fidelity="urans")
    assert seen == [("n0012", 12.0)]


def test_steady_does_not_clobber_an_unsteady_row(tmp_path: Path):
    rows_dir = tmp_path / "rows"
    rows_dir.mkdir(parents=True)
    (rows_dir / "urans_1.jsonl").write_text(
        json.dumps(_row("n0012", 8.0, "urans")) + "\n"
    )

    seen: list = []
    run_sweep([(Path("n0012.dat.txt"), 8.0)], _worker("rans", seen), rows_dir,
              prefix="rans", fidelity="rans")
    assert seen == []


def test_a_failed_row_is_retried(tmp_path: Path):
    rows_dir = tmp_path / "rows"
    rows_dir.mkdir(parents=True)
    (rows_dir / "urans_1.jsonl").write_text(
        json.dumps(_row("n0012", 12.0, "urans", ok=False)) + "\n"
    )

    seen: list = []
    run_sweep([(Path("n0012.dat.txt"), 12.0)], _worker("urans", seen), rows_dir,
              prefix="urans", fidelity="urans")
    assert seen == [("n0012", 12.0)]


def test_a_half_written_line_does_not_break_the_scan(tmp_path: Path):
    """A process killed mid-write leaves a truncated JSON line behind."""
    rows_dir = tmp_path / "rows"
    rows_dir.mkdir(parents=True)
    (rows_dir / "urans_1.jsonl").write_text(
        json.dumps(_row("n0012", 12.0, "urans")) + "\n" + '{"foil": "n0012", "alph'
    )
    assert solved_keys(rows_dir, min_rank=FIDELITY_RANK["urans"]) == {("n0012", 12.0)}


def test_resume_false_re_solves_everything(tmp_path: Path):
    rows_dir = tmp_path / "rows"
    rows_dir.mkdir(parents=True)
    (rows_dir / "urans_1.jsonl").write_text(
        json.dumps(_row("n0012", 12.0, "urans")) + "\n"
    )

    seen: list = []
    run_sweep([(Path("n0012.dat.txt"), 12.0)], _worker("urans", seen), rows_dir,
              prefix="urans", fidelity="urans", resume=False)
    assert seen == [("n0012", 12.0)]


def test_foil_name_matches_what_the_drivers_write():
    # Both run_one implementations derive the key this way; if that changes,
    # resume silently stops matching and every case is re-solved.
    assert foil_name(Path("Data/Airfoils/4/n0012.dat.txt")) == "n0012"
    assert foil_name(Path("Data/Airfoils/4/n0009sm.dat")) == "n0009sm"
