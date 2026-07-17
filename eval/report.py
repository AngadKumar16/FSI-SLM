"""Render evaluation results as a human-readable markdown report.

Takes the results dict produced by :func:`eval.evaluate.evaluate` and writes
a presentation-ready markdown document: summary metrics, per-field accuracy
table, regime confusion matrix, and per-class precision/recall/F1.
"""

from __future__ import annotations

from typing import Optional


def render_markdown_report(results: dict, title: str = "FSI-SLM Evaluation Report") -> str:
    """Return a markdown report string for an ``evaluate()`` results dict."""
    summary = results.get("summary", {})
    meta = results.get("meta", {})
    lines = [f"# {title}", ""]

    if meta:
        lines += [
            f"- **Predictor:** {meta.get('predictor', 'unknown')}",
            f"- **Records:** {meta.get('num_records', summary.get('num_examples', 0))}",
            "",
        ]

    lines += [
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Examples | {summary.get('num_examples', 0)} |",
        f"| Outputs parsed | {summary.get('num_parsed', 0)} |",
        f"| Regime macro F1 | {summary.get('regime_macro_f1', 0.0):.4f} |",
        f"| Lock-in F1 | {summary.get('lock_in_f1', 0.0):.4f} |",
        f"| Avg field accuracy | {summary.get('avg_field_accuracy', 0.0):.4f} |",
        "",
    ]

    fields = results.get("per_field_accuracy", {})
    if fields:
        lines += [
            "## Per-Field Accuracy",
            "",
            "| Field | Accuracy | Correct / Total | Mean rel. error |",
            "|---|---|---|---|",
        ]
        for name, m in fields.items():
            lines.append(
                f"| {name} | {m['accuracy']:.2%} | {m['correct']}/{m['total']} "
                f"| {m.get('mean_relative_error', 0.0):.4f} |"
            )
        lines.append("")

    cm = results.get("regime_confusion_matrix", {})
    matrix = cm.get("matrix", {})
    labels = cm.get("labels", sorted(matrix.keys()))
    if matrix:
        lines += ["## Regime Confusion Matrix", "", "true \\ predicted", ""]
        header = "| | " + " | ".join(labels) + " |"
        sep = "|---" * (len(labels) + 1) + "|"
        lines += [header, sep]
        for t in labels:
            row = matrix.get(t, {})
            lines.append(
                f"| **{t}** | " + " | ".join(str(row.get(p, 0)) for p in labels) + " |"
            )
        lines.append("")

    class_metrics = cm.get("class_metrics", {})
    if class_metrics:
        lines += [
            "## Per-Class Metrics",
            "",
            "| Regime | Precision | Recall | F1 |",
            "|---|---|---|---|",
        ]
        for label, m in class_metrics.items():
            lines.append(
                f"| {label} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |"
            )
        lines.append("")

    lock = results.get("lock_in_metrics", {})
    if lock:
        lines += [
            "## Lock-In Detection",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Precision | {lock.get('precision', 0.0):.4f} |",
            f"| Recall | {lock.get('recall', 0.0):.4f} |",
            f"| F1 | {lock.get('f1', 0.0):.4f} |",
            f"| TP / FP / FN / TN | {lock.get('true_positives', 0)} / "
            f"{lock.get('false_positives', 0)} / {lock.get('false_negatives', 0)} / "
            f"{lock.get('true_negatives', 0)} |",
            "",
        ]

    return "\n".join(lines)


__all__ = ["render_markdown_report"]
