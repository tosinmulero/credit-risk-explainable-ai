# Exploratory Data Analysis Report

## Dataset Overview

- Observations: **30,000**
- Predictive features: **23**
- Target: **DEFAULT_NEXT_MONTH**
- Default observations: **6,636**
- Non-default observations: **23,364**
- Default rate: **22.12%**

## Data Quality

- Missing predictive values: **0**
- Duplicate predictive profiles: **56**

Duplicate predictive profiles are not automatically removed. Different clients can legitimately share the same model features.

## Strongest Linear Associations with Default

| Feature | Correlation |
|---|---:|
| PAY_0 | 0.3248 |
| PAY_2 | 0.2636 |
| PAY_3 | 0.2353 |
| PAY_4 | 0.2166 |
| PAY_5 | 0.2041 |
| PAY_6 | 0.1869 |
| LIMIT_BAL | -0.1535 |
| PAY_AMT1 | -0.0729 |

Correlation is descriptive only and does not establish causality or feature importance in the final model.

## Encoded Category Review

The following source values fall outside the documented category definitions and require explicit preprocessing:

| Feature | Code | Count | Percentage |
|---|---:|---:|---:|
| EDUCATION | 0 | 14 | 0.05% |
| EDUCATION | 5 | 280 | 0.93% |
| EDUCATION | 6 | 51 | 0.17% |
| MARRIAGE | 0 | 54 | 0.18% |
| PAY_0 | -2 | 2,759 | 9.20% |
| PAY_0 | 0 | 14,737 | 49.12% |
| PAY_2 | -2 | 3,782 | 12.61% |
| PAY_2 | 0 | 15,730 | 52.43% |
| PAY_3 | -2 | 4,085 | 13.62% |
| PAY_3 | 0 | 15,764 | 52.55% |
| PAY_4 | -2 | 4,348 | 14.49% |
| PAY_4 | 0 | 16,455 | 54.85% |
| PAY_5 | -2 | 4,546 | 15.15% |
| PAY_5 | 0 | 16,947 | 56.49% |
| PAY_6 | -2 | 4,895 | 16.32% |
| PAY_6 | 0 | 16,286 | 54.29% |

These values are not silently deleted or relabelled during EDA.

## Repayment Behaviour

Repayment-status variables show material variation in observed default rates across status codes.
The highest observed grouped default rate in this summary occurs for `PAY_5` status `nan` at **100.00%**.

## Modelling Implications

1. The target is imbalanced, so plain accuracy will not be treated as the primary evaluation metric.
2. Average precision / PR-AUC, ROC-AUC, recall, precision, F1, log loss and Brier score will be considered.
3. Probability calibration will be evaluated because credit risk decisions depend on estimated probability, not only class.
4. Decision thresholds will be separated from model training and optimised using explicit business-cost assumptions.
5. Encoded categorical variables require deliberate treatment.
6. Demographic variables will receive subgroup/fairness analysis.
7. Predictive associations will not be presented as causal effects.

## Leakage Review

The modelling pipeline will use only variables available in the source feature matrix and will keep the target isolated throughout training. Train/validation/test splitting will occur before any learned preprocessing.

## Next Stage

Feature engineering and reproducible train/validation/test construction.
