# Data Dictionary

## Predictive Features

| Feature | Data type | Unique values | Description |
|---|---|---:|---|
| LIMIT_BAL | int64 | 81 | Amount of given credit. |
| SEX | int64 | 2 | Sex code supplied by the source dataset. |
| EDUCATION | int64 | 7 | Education category code supplied by the source dataset. |
| MARRIAGE | int64 | 4 | Marital-status category code supplied by the source dataset. |
| AGE | int64 | 56 | Age in years. |
| PAY_0 | int64 | 11 | Most recent repayment-status code. |
| PAY_2 | int64 | 11 | Repayment-status code for the prior month. |
| PAY_3 | int64 | 11 | Repayment-status code. |
| PAY_4 | int64 | 11 | Repayment-status code. |
| PAY_5 | int64 | 10 | Repayment-status code. |
| PAY_6 | int64 | 10 | Repayment-status code. |
| BILL_AMT1 | int64 | 22723 | Historical bill-statement amount. |
| BILL_AMT2 | int64 | 22346 | Historical bill-statement amount. |
| BILL_AMT3 | int64 | 22026 | Historical bill-statement amount. |
| BILL_AMT4 | int64 | 21548 | Historical bill-statement amount. |
| BILL_AMT5 | int64 | 21010 | Historical bill-statement amount. |
| BILL_AMT6 | int64 | 20604 | Historical bill-statement amount. |
| PAY_AMT1 | int64 | 7943 | Historical payment amount. |
| PAY_AMT2 | int64 | 7899 | Historical payment amount. |
| PAY_AMT3 | int64 | 7518 | Historical payment amount. |
| PAY_AMT4 | int64 | 6937 | Historical payment amount. |
| PAY_AMT5 | int64 | 6897 | Historical payment amount. |
| PAY_AMT6 | int64 | 6939 | Historical payment amount. |

## Target

`DEFAULT_NEXT_MONTH` is the binary credit-default response.

## Important coding note

Categorical variables are encoded numerically. Some observed category codes may fall outside the documented source definitions. These are treated as undocumented source values rather than silently corrected.

Repayment-status codes are ordinal operational codes and must not be interpreted as continuous monetary quantities.
