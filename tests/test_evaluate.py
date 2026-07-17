"""Tests for the evaluation harness (eval/evaluate.py).

Covers the confusion matrix, per-field accuracy tolerance semantics
(relative vs absolute vs exact), lock-in metrics, and the end-to-end
``evaluate`` path using the rule-based describer as a perfect oracle.
"""

import numpy as np
import pytest

from Data.describe import describe
from Data.schema import FlowRegime
from Data.synthetic import generate_experiment
from eval.evaluate import (
    compute_accuracy,
    compute_confusion_matrix,
    evaluate,
    parse_numeric_claims,
)
from features.extract import extract_features


def _record(regime=FlowRegime.VIV_LOCKIN, seed=0):
    raw = generate_experiment(regime, rng=np.random.default_rng(seed))
    return extract_features(raw)


class TestConfusionMatrix:
    def test_perfect_predictions(self):
        labels = ["a", "b", "c"]
        preds = ["a", "b", "c", "a"]
        truth = ["a", "b", "c", "a"]
        cm = compute_confusion_matrix(preds, truth, labels)
        assert cm["macro_f1"] == 1.0
        assert cm["matrix"]["a"]["a"] == 2

    def test_all_wrong(self):
        preds = ["b", "b"]
        truth = ["a", "a"]
        cm = compute_confusion_matrix(preds, truth, ["a", "b"])
        assert cm["macro_f1"] == 0.0
        assert cm["class_metrics"]["a"]["false_negatives"] == 2
        assert cm["class_metrics"]["b"]["false_positives"] == 2

    def test_labels_inferred_when_none(self):
        cm = compute_confusion_matrix(["x"], ["y"])
        assert set(cm["labels"]) == {"x", "y"}


class TestAccuracyTolerances:
    def test_absolute_tolerance_is_absolute(self):
        """St within +/-0.05 absolute must count as correct even when the
        relative error is far above 5% (the old semantics bug)."""
        fr = _record()
        gt_st = fr.piv.strouhal_number
        pred = {"strouhal_number": gt_st + 0.04}  # inside +/-0.05 absolute
        res = compute_accuracy([pred], [fr])
        assert res["strouhal_number"]["accuracy"] == 1.0

    def test_absolute_tolerance_rejects_outside(self):
        fr = _record()
        pred = {"strouhal_number": fr.piv.strouhal_number + 0.2}
        res = compute_accuracy([pred], [fr])
        assert res["strouhal_number"]["accuracy"] == 0.0

    def test_relative_tolerance_frequency(self):
        fr = _record()
        f = fr.piv.shedding_frequency_hz
        good = {"shedding_frequency_hz": f * 1.10}   # within 15% rel
        bad = {"shedding_frequency_hz": f * 1.30}    # outside
        res = compute_accuracy([good, bad], [fr, fr])
        assert res["shedding_frequency_hz"]["correct"] == 1
        assert res["shedding_frequency_hz"]["total"] == 2

    def test_exact_match_mode_number(self):
        fr = _record()
        good = {"mode_number": fr.structure.mode_number}
        bad = {"mode_number": fr.structure.mode_number + 1}
        res = compute_accuracy([good, bad], [fr, fr])
        assert res["mode_number"]["correct"] == 1

    def test_missing_field_not_counted(self):
        fr = _record()
        res = compute_accuracy([{}], [fr])
        assert res["strouhal_number"]["total"] == 0
        assert res["strouhal_number"]["accuracy"] == 0.0


class TestEvaluateEndToEnd:
    def test_oracle_predictions_score_high(self):
        """Feeding the rule-based describe() output back through the parser
        should recover regime and fields with near-perfect accuracy."""
        records = [
            _record(regime, seed)
            for seed, regime in enumerate(FlowRegime)
        ]
        preds = [describe(fr) for fr in records]
        results = evaluate(preds, records)
        assert results["summary"]["num_examples"] == len(records)
        assert results["summary"]["num_parsed"] == len(records)
        # The describer states the extracted values verbatim, so field
        # accuracy vs the same records must be perfect.
        assert results["per_field_accuracy"]["strouhal_number"]["accuracy"] == 1.0
        assert results["per_field_accuracy"]["mode_number"]["accuracy"] == 1.0

    def test_parse_numeric_claims_roundtrip(self):
        fr = _record()
        parsed = parse_numeric_claims(describe(fr))
        assert "regime" in parsed
        assert parsed["strouhal_number"] == pytest.approx(
            fr.piv.strouhal_number, abs=5e-4
        )
        assert parsed["mode_number"] == fr.structure.mode_number

    def test_empty_predictions(self):
        results = evaluate([], [])
        assert results["summary"]["num_examples"] == 0
        assert results["summary"]["regime_macro_f1"] == 0.0
