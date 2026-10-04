# Logistic Regression Benchmark

## Purpose

This model establishes the first supervised-learning benchmark for the credit-risk project.

It is intentionally interpretable and will be used as a reference when comparing more complex tree-based models.

## Data Usage

- Training data: 18,000 observations
- Validation data: 6,000 observations
- Test data: not accessed
- Decision threshold: 0.50

## Preprocessing

- Numeric features are standardised using statistics fitted only on the training set.
- Categorical variables are one-hot encoded using categories learned only from the training set.
- Unknown validation categories are ignored safely.

## Candidate Features

- Total candidate features before encoding: **40**
- Categorical features: **2**

## Validation Performance

- ROC-AUC: **0.7689**
- Average Precision / PR-AUC: **0.5266**
- Accuracy: **0.8187**
- Precision: **0.6797**
- Recall: **0.3406**
- F1: **0.4538**
- Log Loss: **0.4391**
- Brier Score: **0.1371**

## Interpretation

Accuracy is not treated as the primary metric because the target is imbalanced.

ROC-AUC evaluates ranking discrimination across thresholds, while Average Precision gives greater visibility into performance on the default class.

Log loss and Brier score evaluate probability quality and will be important when probability calibration is assessed later.

## Governance

The validation set is used for benchmark assessment. The test set remains sealed for final model evaluation.

Coefficient magnitude is descriptive of this fitted model and should not be interpreted as causal evidence.
