# XGBoost Hyperparameter Optimisation

## Objective

Optimise the provisional XGBoost leader using Optuna while preserving the untouched validation and test sets.

## Optimisation Design

- Optuna trials: **30**
- Cross-validation folds: **5**
- Cross-validation: stratified and shuffled
- Primary optimisation metric: Average Precision / PR-AUC
- Hyperparameter search data: training set only
- Validation data excluded from Optuna search
- Test data not accessed

## Best Cross-Validated Result

- Best training CV PR-AUC: **0.5666**

## Best Hyperparameters

| Parameter | Value |
|---|---:|
| n_estimators | 300 |
| learning_rate | 0.015056138788764583 |
| max_depth | 5 |
| min_child_weight | 1 |
| subsample | 0.6313325999942377 |
| colsample_bytree | 0.7033387080163427 |
| gamma | 3.950218790438706 |
| reg_alpha | 0.2239432027975281 |
| reg_lambda | 0.10565963013430783 |

## Validation Comparison

| Metric | Benchmark XGBoost | Tuned XGBoost |
|---|---:|---:|
| ROC-AUC | 0.7867 | 0.7889 |
| PR-AUC | 0.5547 | 0.5629 |
| Precision | 0.6667 | 0.6799 |
| Recall | 0.3632 | 0.3602 |
| F1 | 0.4702 | 0.4709 |
| Log Loss | 0.4278 | 0.4243 |
| Brier Score | 0.1340 | 0.1332 |

## Selected XGBoost Variant

**Tuned XGBoost**

The selection uses validation PR-AUC first and ROC-AUC second. Threshold-specific metrics are not used as the primary selection criterion because the operating threshold will be optimised separately.

## Governance

The test set remains sealed. Hyperparameter optimisation does not use validation observations.
