"""Evaluation harness for the trained FSI interpreter.

Runs the model on a held-out test set with and without RAG augmentation.
Extracts numeric claims from generated text via regex parsing, compares to
ground-truth ``FeatureRecord`` values within tolerances, and reports:

- Per-field accuracy (within tolerance)
- Regime classification F1 score
- Lock-in precision/recall
- With-vs-without-RAG improvement

Outputs a JSON results file.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from Data.schema import FeatureRecord, FlowRegime, LockInStatus
from eval.validate_output import parse_structured_fields


def parse_numeric_claims(text: str) -> dict:
    """Parse numeric claims from generated text.

    Delegates to ``eval.validate_output.parse_structured_fields`` for the
    actual parsing logic.

    Parameters
    ----------
    text : str
        The generated model output text.

    Returns
    -------
    dict
        Parsed numeric claims with keys like ``regime``, ``shedding_frequency_hz``,
        ``strouhal_number``, etc.
    """
    return parse_structured_fields(text)


def compute_confusion_matrix(
    predictions: List[str],
    ground_truth: List[str],
    labels: Optional[List[str]] = None,
) -> dict:
    """Compute a confusion matrix for categorical predictions.

    Parameters
    ----------
    predictions : list[str]
        Predicted labels.
    ground_truth : list[str]
        Ground-truth labels.
    labels : list[str], optional
        Complete list of label names. If None, inferred from union of predictions
        and ground truth.

    Returns
    -------
    dict
        Confusion matrix as a dict of ``{true_label: {predicted_label: count}}``
        along with per-class metrics (precision, recall, F1).
    """
    if labels is None:
        labels = sorted(set(ground_truth) | set(predictions))

    # Build matrix
    matrix = {t: {p: 0 for p in labels} for t in labels}
    for pred, true in zip(predictions, ground_truth):
        if true in matrix and pred in matrix[true]:
            matrix[true][pred] += 1

    # Compute per-class metrics
    class_metrics = {}
    for label in labels:
        tp = matrix[label].get(label, 0)
        fp = sum(matrix[t].get(label, 0) for t in labels if t != label)
        fn = sum(matrix[label].get(p, 0) for p in labels if p != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        class_metrics[label] = {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    # Macro-averaged F1
    macro_f1 = np.mean([m["f1"] for m in class_metrics.values()])

    return {
        "matrix": matrix,
        "class_metrics": class_metrics,
        "macro_f1": round(macro_f1, 4),
        "labels": labels,
    }


def compute_accuracy(
    predictions: List[dict],
    ground_truth: List[FeatureRecord],
    tolerances: Optional[dict] = None,
) -> dict:
    """Compute per-field accuracy comparing predicted claims to ground truth.

    Parameters
    ----------
    predictions : list[dict]
        List of parsed prediction dicts (from ``parse_numeric_claims``).
    ground_truth : list[FeatureRecord]
        List of ground-truth ``FeatureRecord`` objects.
    tolerances : dict, optional
        Dict mapping field name to absolute tolerance. Default tolerances:
        ``shedding_frequency_hz``: 15% relative, ``strouhal_number``: 0.05,
        ``rms_amplitude_over_c``: 0.02, ``mode_number``: 0 (exact).

    Returns
    -------
    dict
        Per-field accuracy metrics.
    """
    if tolerances is None:
        tolerances = {
            "shedding_frequency_hz": 0.15,  # 15% relative
            "strouhal_number": 0.05,
            "rms_amplitude_over_c": 0.02,
            "mode_number": 0,  # exact
        }

    # Map ground truth fields
    field_map = {
        "shedding_frequency_hz": lambda fr: fr.piv.shedding_frequency_hz,
        "strouhal_number": lambda fr: fr.piv.strouhal_number,
        "rms_amplitude_over_c": lambda fr: fr.structure.rms_amplitude_over_c,
        "mode_number": lambda fr: fr.structure.mode_number,
    }

    results = {}
    for field, gt_fn in field_map.items():
        correct = 0
        total = 0
        errors: List[float] = []
        for pred, gt in zip(predictions, ground_truth):
            if field not in pred:
                continue
            pred_val = pred[field]
            gt_val = gt_fn(gt)
            total += 1

            if field in tolerances:
                tol = tolerances[field]
                if tol == 0:
                    is_correct = pred_val == gt_val
                elif tol < 1.0:  # relative tolerance
                    if abs(gt_val) > 1e-10:
                        err = abs(pred_val - gt_val) / abs(gt_val)
                        is_correct = err <= tol
                    else:
                        is_correct = abs(pred_val) <= tol
                else:  # absolute tolerance
                    err = abs(pred_val - gt_val)
                    is_correct = err <= tol
            else:
                is_correct = False

            if is_correct:
                correct += 1
            errors.append(abs(pred_val - gt_val) / max(abs(gt_val), 1e-10) if abs(gt_val) > 1e-10 else abs(pred_val))

        acc = correct / max(total, 1)
        results[field] = {
            "accuracy": round(acc, 4),
            "correct": correct,
            "total": total,
            "mean_relative_error": round(float(np.mean(errors)), 4) if errors else 0.0,
        }

    return results


def _regime_label_to_enum(label: str) -> Optional[FlowRegime]:
    """Convert a regime label string to a FlowRegime enum."""
    mapping = {
        "attached flow": FlowRegime.ATTACHED,
        "separated flow": FlowRegime.SEPARATED,
        "viv lock-in": FlowRegime.VIV_LOCKIN,
        "galloping": FlowRegime.GALLOPING,
        "stall flutter": FlowRegime.STALL_FLUTTER,
    }
    return mapping.get(label.lower().strip())


def evaluate(
    predictions: List[str],
    test_dataset: List[FeatureRecord],
    config: Optional[dict] = None,
) -> dict:
    """Run evaluation on test set predictions.

    Parameters
    ----------
    predictions : list[str]
        List of generated text outputs (one per test example).
    test_dataset : list[FeatureRecord]
        List of ground-truth ``FeatureRecord`` objects for the test set.
    config : dict, optional
        Evaluation configuration dict with keys like ``freq_tolerance``,
        ``regime_f1_threshold``.

    Returns
    -------
    dict
        Evaluation results with keys: ``per_field_accuracy``,
        ``regime_confusion_matrix``, ``lock_in_metrics``, ``summary``.
    """
    if config is None:
        config = {}

    freq_tol = config.get("freq_tolerance", 0.15)
    tolerances = {
        "shedding_frequency_hz": freq_tol,
        "strouhal_number": 0.05,
        "rms_amplitude_over_c": 0.02,
        "mode_number": 0,
    }

    # Parse all predictions
    parsed = [parse_numeric_claims(t) for t in predictions]

    # Per-field accuracy
    field_accuracy = compute_accuracy(parsed, test_dataset, tolerances)

    # Regime classification
    predicted_regimes = []
    true_regimes = []
    for p, gt in zip(parsed, test_dataset):
        regime_str = p.get("regime", "")
        pred_enum = _regime_label_to_enum(regime_str)
        if pred_enum is not None and gt.ground_truth_regime is not None:
            predicted_regimes.append(pred_enum.value)
            true_regimes.append(gt.ground_truth_regime.value)

    regime_cm = {}
    if predicted_regimes:
        labels = [r.value for r in FlowRegime]
        regime_cm = compute_confusion_matrix(predicted_regimes, true_regimes, labels)
    else:
        regime_cm = {"macro_f1": 0.0, "class_metrics": {}, "matrix": {}}

    # Lock-in metrics (precision/recall for detecting locked vs not-locked)
    lock_in_tp = 0
    lock_in_fp = 0
    lock_in_fn = 0
    lock_in_tn = 0

    for p, gt in zip(parsed, test_dataset):
        pred_lock = p.get("lock_in_status", "").strip().lower()
        true_lock = gt.lock_in_status().value

        pred_locked = pred_lock == "locked"
        true_locked = true_lock == "locked"

        if pred_locked and true_locked:
            lock_in_tp += 1
        elif pred_locked and not true_locked:
            lock_in_fp += 1
        elif not pred_locked and true_locked:
            lock_in_fn += 1
        else:
            lock_in_tn += 1

    lock_in_precision = lock_in_tp / max(lock_in_tp + lock_in_fp, 1)
    lock_in_recall = lock_in_tp / max(lock_in_tp + lock_in_fn, 1)
    lock_in_f1 = (
        2 * lock_in_precision * lock_in_recall / max(lock_in_precision + lock_in_recall, 1e-10)
    )

    lock_in_metrics = {
        "precision": round(lock_in_precision, 4),
        "recall": round(lock_in_recall, 4),
        "f1": round(lock_in_f1, 4),
        "true_positives": lock_in_tp,
        "false_positives": lock_in_fp,
        "false_negatives": lock_in_fn,
        "true_negatives": lock_in_tn,
    }

    # Summary
    regime_f1 = regime_cm.get("macro_f1", 0.0)
    summary = {
        "num_examples": len(test_dataset),
        "num_parsed": len([p for p in parsed if p.get("regime")]),
        "regime_macro_f1": regime_f1,
        "lock_in_f1": lock_in_f1,
        "regime_f1_threshold_met": bool(regime_f1 >= config.get("regime_f1_threshold", 0.5)),
        "avg_field_accuracy": round(
            np.mean([v["accuracy"] for v in field_accuracy.values()]), 4
        ) if field_accuracy else 0.0,
    }

    return {
        "per_field_accuracy": field_accuracy,
        "regime_confusion_matrix": regime_cm,
        "lock_in_metrics": lock_in_metrics,
        "summary": summary,
    }


def evaluate_with_rag(
    predictions_no_rag: List[str],
    predictions_with_rag: List[str],
    test_dataset: List[FeatureRecord],
    config: Optional[dict] = None,
) -> dict:
    """Run evaluation comparing without-RAG and with-RAG performance.

    Parameters
    ----------
    predictions_no_rag : list[str]
        Model outputs without RAG augmentation.
    predictions_with_rag : list[str]
        Model outputs with RAG augmentation.
    test_dataset : list[FeatureRecord]
        Ground-truth test set.
    config : dict, optional
        Evaluation configuration.

    Returns
    -------
    dict
        Comparison results including both evaluations and the improvement delta.
    """
    eval_no_rag = evaluate(predictions_no_rag, test_dataset, config)
    eval_with_rag = evaluate(predictions_with_rag, test_dataset, config)

    # Compute deltas
    delta = {
        "regime_f1_delta": round(
            eval_with_rag["summary"]["regime_macro_f1"]
            - eval_no_rag["summary"]["regime_macro_f1"],
            4,
        ),
        "lock_in_f1_delta": round(
            eval_with_rag["summary"]["lock_in_f1"]
            - eval_no_rag["summary"]["lock_in_f1"],
            4,
        ),
        "avg_field_accuracy_delta": round(
            eval_with_rag["summary"]["avg_field_accuracy"]
            - eval_no_rag["summary"]["avg_field_accuracy"],
            4,
        ),
    }

    return {
        "without_rag": eval_no_rag,
        "with_rag": eval_with_rag,
        "improvement": delta,
    }


__all__ = [
    "parse_numeric_claims",
    "compute_accuracy",
    "compute_confusion_matrix",
    "evaluate",
    "evaluate_with_rag",
]