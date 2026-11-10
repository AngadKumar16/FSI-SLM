# FSI-SLM Evaluation Report

- **Predictor:** rule-based
- **Records:** 20

## Summary

| Metric | Value |
|---|---|
| Examples | 20 |
| Outputs parsed | 20 |
| Regime macro F1 | 0.7833 |
| Lock-in F1 | 1.0000 |
| Avg field accuracy | 1.0000 |

## Per-Field Accuracy

| Field | Accuracy | Correct / Total | Mean rel. error |
|---|---|---|---|
| shedding_frequency_hz | 100.00% | 20/20 | 0.0007 |
| strouhal_number | 100.00% | 20/20 | 0.0007 |
| rms_amplitude_over_c | 100.00% | 20/20 | 0.0305 |
| mode_number | 100.00% | 20/20 | 0.0000 |

## Regime Confusion Matrix

true \ predicted

| | attached | separated | viv_lockin | galloping | stall_flutter |
|---|---|---|---|---|---|
| **attached** | 4 | 0 | 0 | 0 | 0 |
| **separated** | 0 | 3 | 0 | 0 | 0 |
| **viv_lockin** | 0 | 0 | 6 | 0 | 0 |
| **galloping** | 0 | 0 | 2 | 1 | 0 |
| **stall_flutter** | 0 | 0 | 2 | 0 | 2 |

## Per-Class Metrics

| Regime | Precision | Recall | F1 |
|---|---|---|---|
| attached | 1.000 | 1.000 | 1.000 |
| separated | 1.000 | 1.000 | 1.000 |
| viv_lockin | 0.600 | 1.000 | 0.750 |
| galloping | 1.000 | 0.333 | 0.500 |
| stall_flutter | 1.000 | 0.500 | 0.667 |

## Lock-In Detection

| Metric | Value |
|---|---|
| Precision | 1.0000 |
| Recall | 1.0000 |
| F1 | 1.0000 |
| TP / FP / FN / TN | 10 / 0 / 0 / 10 |
