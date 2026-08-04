"""The measured shedding frequency has to *reach* the interpreter.

``_measured_shedding`` was already tested, but nothing fed it: both twin paths
build the flow from geometry alone and carry no time series, so the block it
reads was always absent and every record fell back on the Fage-Johansen
correlation -- including for the cases that had been solved transiently at hours
apiece. These tests pin the lookup that closes that gap.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sensor.bridge import (
    TwinConfig,
    _measured_shedding,
    attach_measured_unsteady,
    foil_key,
    lookup_unsteady,
)


def _manifest(tmp_path: Path, rows) -> Path:
    p = tmp_path / "manifest.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def _urans_row(foil="n0012", alpha=12.0, **kw):
    row = {
        "foil": foil, "alpha_deg": alpha, "fidelity": "urans", "ok": True,
        "f_shed": 0.24, "strouhal_chord": 0.24, "strouhal_projected": 0.29,
        "organised_shedding": True, "statistically_steady": True,
        "cl_rms": 0.21, "n_cycles": 12.0,
    }
    row.update(kw)
    return row


def test_lookup_returns_the_measured_block(tmp_path):
    m = _manifest(tmp_path, [_urans_row()])
    block = lookup_unsteady("n0012", 12.0, m)
    assert block["strouhal_chord"] == 0.24
    assert block["organised_shedding"] is True


def test_lookup_ignores_steady_and_failed_rows(tmp_path):
    m = _manifest(tmp_path, [
        _urans_row(fidelity="rans"),
        _urans_row(alpha=16.0, ok=False),
    ])
    assert lookup_unsteady("n0012", 12.0, m) is None
    assert lookup_unsteady("n0012", 16.0, m) is None


def test_lookup_will_not_borrow_a_neighbouring_angle(tmp_path):
    """Shedding frequency varies with incidence; 16 is not a measurement of 12."""
    m = _manifest(tmp_path, [_urans_row(alpha=16.0)])
    assert lookup_unsteady("n0012", 12.0, m) is None
    assert lookup_unsteady("n0012", 16.0, m) is not None
    assert lookup_unsteady("n0012", 16.2, m) is not None  # within tolerance


def test_lookup_survives_a_missing_manifest(tmp_path):
    assert lookup_unsteady("n0012", 12.0, tmp_path / "nope.jsonl") is None


def test_foil_key_is_the_dat_stem_not_the_title_in_the_file():
    # load_dat names the section from the header line ("NACA 0012"); the
    # manifest keys on the file stem ("n0012"). Matching on the wrong one
    # silently finds nothing.
    assert foil_key("Data/Airfoils/4/n0012.dat.txt") == "n0012"
    assert foil_key(Path("Data/Airfoils/4/n0009sm.dat")) == "n0009sm"


def test_attach_then_measured_shedding_agrees_with_the_row(tmp_path):
    m = _manifest(tmp_path, [_urans_row()])
    case = {"airfoil": "NACA 0012", "alpha_deg": 12.0}
    attach_measured_unsteady(case, "n0012", m)

    cfg = TwinConfig()
    f, note = _measured_shedding(case, cfg)
    assert f == pytest.approx(0.24 * cfg.u_inf_m_s / cfg.chord_m)
    assert "measured" in note
    assert case["unsteady_source"] == str(m)


def test_attach_does_not_overwrite_a_case_that_already_has_one(tmp_path):
    m = _manifest(tmp_path, [_urans_row()])
    case = {"alpha_deg": 12.0, "unsteady": {"strouhal_chord": 0.19}}
    attach_measured_unsteady(case, "n0012", m)
    assert case["unsteady"]["strouhal_chord"] == 0.19


def test_a_provisional_case_is_used_but_flagged(tmp_path):
    """Statistics that never settled still beat a correlation -- if they say so."""
    m = _manifest(tmp_path, [_urans_row(statistically_steady=False)])
    case = {"alpha_deg": 12.0}
    attach_measured_unsteady(case, "n0012", m)
    _f, note = _measured_shedding(case, TwinConfig())
    assert "provisional" in note


def test_an_unorganised_peak_is_not_promoted_to_a_measurement(tmp_path):
    m = _manifest(tmp_path, [_urans_row(organised_shedding=False)])
    case = {"alpha_deg": 12.0}
    attach_measured_unsteady(case, "n0012", m)
    assert _measured_shedding(case, TwinConfig()) is None


def test_the_manifest_row_carries_every_key_the_bridge_reads(tmp_path):
    """_row_from_meta must not drop organised_shedding on the way to the row."""
    from sensor.generate import _row_from_meta

    meta = {
        "coeffs": {}, "modal": {}, "fidelity": "urans",
        "unsteady": {
            "f_shed": 0.24, "strouhal_chord": 0.24, "strouhal_projected": 0.29,
            "cl_rms": 0.2, "cd_rms": 0.05, "n_cycles": 12.0,
            "statistically_steady": True, "organised_shedding": True,
            "shedding_prominence": 9.1, "shedding_note": "peak at bin 31",
            "stationarity_note": "halves agree to 2% of RMS",
        },
    }
    row = _row_from_meta(meta)
    for k in ("strouhal_chord", "organised_shedding", "statistically_steady"):
        assert row[k] == meta["unsteady"][k], k
