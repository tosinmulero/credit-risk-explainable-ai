# Advanced Model Benchmarking

## Objective

Compare multiple supervised-learning model families under the same train/validation design before hyperparameter tuning.

## Models

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

## Evaluation Policy

All advanced models are trained on the same 18,000-row training set and evaluated on the same 6,000-row validation set.

The 6,000-row test set remains untouched.

Model selection at this stage prioritises Average Precision (PR-AUC), followed by ROC-AUC. Probability-quality metrics are also retained for later calibration analysis.

## Validation Comparison

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Log Loss | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.7867 | 0.5547 | 0.6667 | 0.3632 | 0.4702 | 0.4278 | 0.1340 |
| LightGBM | 0.7787 | 0.5481 | 0.6523 | 0.3534 | 0.4585 | 0.4373 | 0.1361 |
| Random Forest | 0.7794 | 0.5424 | 0.6487 | 0.3715 | 0.4724 | 0.4330 | 0.1359 |
| Logistic Regression | 0.7689 | 0.5266 | 0.6797 | 0.3406 | 0.4538 | 0.4391 | 0.1371 |

## Provisional Validation Leader

**XGBoost**

Validation PR-AUC: **0.5547**
Validation ROC-AUC: **0.7867**

This is not the final production model. The leading candidate will undergo hyperparameter optimisation, probability calibration, threshold optimisation, explainability and final test-set evaluation.
