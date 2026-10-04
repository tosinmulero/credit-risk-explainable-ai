# SHAP Explainability

TreeSHAP is used to explain the tuned XGBoost model.

Positive SHAP values push predictions toward higher estimated default risk; negative values push them lower.

SHAP explains model behaviour, not causality.

## Top global features

| Rank | Feature | Mean absolute SHAP |
|---:|---|---:|
| 1 | PAY_0 | 0.299118 |
| 2 | SEVERE_DELINQUENT_MONTHS_6M | 0.188404 |
| 3 | DELINQUENT_MONTHS_6M | 0.142038 |
| 4 | LATEST_UTILIZATION | 0.066702 |
| 5 | RECENT_DELAY_MONTHS | 0.066251 |
| 6 | LIMIT_BAL | 0.064186 |
| 7 | MAX_UTILIZATION_6M | 0.063319 |
| 8 | MEAN_DELAY_MONTHS_6M | 0.063048 |
| 9 | PAYMENT_MEAN_6M | 0.057774 |
| 10 | ZERO_PAYMENT_MONTHS_6M | 0.055489 |
| 11 | MAX_DELAY_MONTHS_6M | 0.054047 |
| 12 | MEAN_UTILIZATION_6M | 0.053721 |
| 13 | BILL_CHANGE_RECENT_VS_OLDEST | 0.052632 |
| 14 | BILL_AMT1 | 0.052350 |
| 15 | BILL_MAX_6M | 0.049486 |

Selected threshold: **0.145**.

The test set was not accessed in this stage.
