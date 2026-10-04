# Model Card — Credit Risk Intelligence

## Model overview

- Model family: XGBoost
- Probability variant: Uncalibrated tuned XGBoost
- Optuna trials: 30
- Operating threshold: 0.145
- Final holdout observations: 6,000

## Final holdout performance

| Metric | Value |
|---|---:|
| ROC-AUC | 0.7834 |
| PR-AUC / Average Precision | 0.5622 |
| Precision | 0.3491 |
| Recall | 0.8071 |
| F1 | 0.4874 |
| Log loss | 0.4270 |
| Brier score | 0.1341 |
| ECE | 0.0133 |

## Intended use

Portfolio demonstration of a complete credit-risk machine-learning workflow.

## Limitations

This model is built on a public historical benchmark dataset and is not suitable for real lending or underwriting decisions. A production system would require representative current data, formal model-risk governance, legal and regulatory review, bias testing, security controls, and institution-specific cost assumptions.

## Responsible use

`SEX` and `AGE` are excluded from the primary model and retained for audit analysis. This design choice does not by itself establish fairness or legal compliance. SHAP explanations describe model behaviour and are not causal.
