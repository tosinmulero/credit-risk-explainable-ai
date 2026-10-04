# Feature Engineering and Leakage-Safe Splitting

## Split Strategy

- Training: 60%
- Validation: 20%
- Test: 20%
- Stratified by `DEFAULT_NEXT_MONTH`
- Random state: 42

The test set is sealed from model selection.

## Leakage Control

All features created in this stage are deterministic row-level transformations. No encoder, scaler, resampling method or model is fitted before the train/validation/test split.

Learned preprocessing will be fitted on training data only.

## Responsible Feature Policy

`SEX` and `AGE` are retained for audit and subgroup analysis but excluded from the primary modelling feature set.

Raw education and marital-status codes are preserved for lineage. Grouped categorical representations are used as model candidates.

## Engineered Feature Families

- delinquency frequency and severity
- bill-level summaries
- payment summaries
- utilisation ratios
- payment-to-limit behaviour
- zero-payment frequency

## Split Summary

| Split | Rows | Defaults | Default rate |
|---|---:|---:|---:|
| train | 18,000 | 3,982 | 22.12% |
| validation | 6,000 | 1,327 | 22.12% |
| test | 6,000 | 1,327 | 22.12% |
