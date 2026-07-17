"""Tests for the markdown evaluation report (eval/report.py)."""

import numpy as np

from Data.describe import describe
from Data.schema import FlowRegime
from Data.synthetic import generate_experiment
from eval.evaluate import evaluate
from eval.report import render_markdown_report
from features.extract import extract_features


def _results():
    records = []
    for seed, regime in enumerate(FlowRegime):
        raw = generate_experiment(regime, rng=np.random.default_rng(seed))
        records.append(extract_features(raw))
    preds = [describe(fr) for fr in records]
    return evaluate(preds, records)


class TestMarkdownReport:
    def test_report_contains_all_sections(self):
        md = render_markdown_report(_results())
        assert "# FSI-SLM Evaluation Report" in md
        assert "## Summary" in md
        assert "## Per-Field Accuracy" in md
        assert "## Regime Confusion Matrix" in md
        assert "## Per-Class Metrics" in md
        assert "## Lock-In Detection" in md

    def test_report_reflects_metrics(self):
        results = _results()
        md = render_markdown_report(results)
        assert f"{results['summary']['regime_macro_f1']:.4f}" in md
        assert "strouhal_number" in md

    def test_meta_section_when_present(self):
        results = _results()
        results["meta"] = {"predictor": "rule-based", "num_records": 5}
        md = render_markdown_report(results)
        assert "**Predictor:** rule-based" in md

    def test_empty_results_no_crash(self):
        md = render_markdown_report(
            {"summary": {}, "per_field_accuracy": {},
             "regime_confusion_matrix": {}, "lock_in_metrics": {}}
        )
        assert md.startswith("# ")
