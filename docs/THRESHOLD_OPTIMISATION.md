# Decision Threshold Optimisation

## Objective

Replace the arbitrary 0.50 classification cutoff with an explicit, validation-derived operating threshold.

## Primary Cost Scenario

- False-positive cost: **1 illustrative cost unit**
- False-negative cost: **5 illustrative cost units**

These are scenario weights for portfolio modelling only. They are not observed financial losses and do not represent a real lender's underwriting policy.

## Selected Threshold

- Cost-optimised validation threshold: **0.145**

## Comparison with 0.50

| Metric | Threshold 0.50 | Optimised threshold |
|---|---:|---:|
| Precision | 0.6799 | 0.3550 |
| Recall | 0.3602 | 0.8206 |
| F1 | 0.4709 | 0.4956 |
| False positives | 225 | 1979 |
| False negatives | 849 | 238 |
| Cost units | 4470 | 3169 |

- Illustrative cost reduction: **1301 units (29.11%)**

## Sensitivity Analysis

| FN:FP Ratio | Optimal threshold | Precision | Recall | False positives | False negatives |
|---:|---:|---:|---:|---:|---:|
| 2:1 | 0.330 | 0.5732 | 0.5222 | 516 | 634 |
| 5:1 | 0.145 | 0.3550 | 0.8206 | 1979 | 238 |
| 10:1 | 0.090 | 0.2796 | 0.9337 | 3192 | 88 |

## Governance

Threshold optimisation uses the validation set only.

The test set remains sealed and will be used only once the modelling, calibration and operating-policy choices have been frozen.

A production credit decision system would require institution-specific financial costs, regulatory review, fair-lending analysis and policy approval.
