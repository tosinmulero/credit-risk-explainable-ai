# UCI Feature Schema Mapping

The raw UCI data is preserved unchanged.

Semantic names are applied only within the analytical and modelling layers.

| UCI name | Semantic name | Role |
|---|---|---|
| ID | CLIENT_ID | Identifier |
| X1 | LIMIT_BAL | Feature |
| X2 | SEX | Feature |
| X3 | EDUCATION | Feature |
| X4 | MARRIAGE | Feature |
| X5 | AGE | Feature |
| X6 | PAY_0 | Feature |
| X7 | PAY_2 | Feature |
| X8 | PAY_3 | Feature |
| X9 | PAY_4 | Feature |
| X10 | PAY_5 | Feature |
| X11 | PAY_6 | Feature |
| X12 | BILL_AMT1 | Feature |
| X13 | BILL_AMT2 | Feature |
| X14 | BILL_AMT3 | Feature |
| X15 | BILL_AMT4 | Feature |
| X16 | BILL_AMT5 | Feature |
| X17 | BILL_AMT6 | Feature |
| X18 | PAY_AMT1 | Feature |
| X19 | PAY_AMT2 | Feature |
| X20 | PAY_AMT3 | Feature |
| X21 | PAY_AMT4 | Feature |
| X22 | PAY_AMT5 | Feature |
| X23 | PAY_AMT6 | Feature |
| Y | DEFAULT_NEXT_MONTH | Target |

## Governance Principle

Raw source columns are never overwritten. The mapping is explicit, version-controlled and reusable across EDA, feature engineering, modelling and deployment.

## Category Handling

Values outside the category definitions documented by the source dataset are retained and flagged rather than silently deleted, recoded or assigned an unsupported interpretation.
