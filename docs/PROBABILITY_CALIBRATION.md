# Probability Calibration

## Objective

Evaluate whether probability calibration improves the probabilistic reliability of the tuned XGBoost model.

## Calibration Design

- Base model: tuned XGBoost
- Calibration methods: sigmoid and isotonic
- Calibration folds: 5
- Calibration fitted using training data only
- Validation set used only for method comparison
- Test set not accessed

## Selection Policy

The primary calibration metric is Brier score, followed by log loss. Lower values are better.

ROC-AUC and PR-AUC are retained to ensure calibration does not materially damage discrimination.

## Validation Comparison

| Variant | ROC-AUC | PR-AUC | Log Loss | Brier | ECE | MCE |
|---|---:|---:|---:|---:|---:|---:|
| Uncalibrated tuned XGBoost | 0.7889 | 0.5629 | 0.4243 | 0.1332 | 0.0063 | 0.0355 |
| Isotonic calibrated XGBoost | 0.7875 | 0.5598 | 0.4258 | 0.1335 | 0.0069 | 0.0342 |
| Sigmoid calibrated XGBoost | 0.7881 | 0.5612 | 0.4284 | 0.1342 | 0.0228 | 0.0772 |

## Selected Probability Variant

**Uncalibrated tuned XGBoost**

- Brier score: **0.1332**
- Log loss: **0.4243**
- Expected calibration error: **0.0063**

## Governance

No validation or test observations are used to fit the calibration models.

The selected probability model will be passed to the decision-threshold optimisation stage.
