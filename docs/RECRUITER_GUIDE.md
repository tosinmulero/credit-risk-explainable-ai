# Recruiter evidence guide — Credit Risk Intelligence & Explainable Default Prediction

[**Full case study on Mulero Analytics**](https://tosinmulero.github.io/data-analytics-portfolio/case-studies/credit-risk-explainable-ai.html) · [Repository README](../README.md) · [Main portfolio](https://tosinmulero.github.io/data-analytics-portfolio/projects.html)

> **Data provenance:** Public historical UCI Default of Credit Card Clients benchmark dataset; not operational underwriting data.

## The question

How can a credit-default model be evaluated responsibly across discrimination, probability quality, operating thresholds, explainability and subgroup risks?

## Technical approach

- Develop and tune benchmarked classifiers with Optuna.
- Evaluate probability calibration and threshold sensitivity on held-out data.
- Inspect SHAP contributions, subgroup performance, model governance and production-style API/monitoring assets.

## Documented results to review

- A 6,000-record sealed holdout reports ROC-AUC 0.7834 and PR-AUC 0.5622.
- Recall is 80.71% and precision 34.91% at the frozen threshold of 0.145.
- The 29.1% illustrative cost reduction is modelled under assumed decision costs; it is not realised savings.

## Evidence map

| Review item | Source | What it supports |
|---|---|---|
| Model card | [Open](../docs/MODEL_CARD.md) | Final holdout, scope and limitations |
| Final evaluation | [Open](../docs/FINAL_TEST_EVALUATION.md) | Locked evaluation and operating metrics |
| Fairness audit | [Open](../docs/FAIRNESS_AUDIT.md) | Subgroup checks and boundaries |
| Global SHAP importance | [Open](../reports/figures/21_shap_global_importance.png) | Descriptive model attributions |

## Reproducibility and limitations

Use the documented README setup and project pipelines for technical reproduction. This model is not suitable for real lending decisions, and SHAP does not imply causality. Excluding protected attributes is not proof of fairness.

The repository README and linked source files remain the primary technical record. Figures in this guide summarise the documented project assets, not verified third-party business outcomes.

## Suggested review path

1. Start with the [standalone case study](https://tosinmulero.github.io/data-analytics-portfolio/case-studies/credit-risk-explainable-ai.html) to understand the problem.
2. Inspect the evidence-map figures and documentation.
3. Read the repository README for tools, workflow and limitations.
4. Review code and tests for deeper technical assurance.

---

*Independent portfolio work by Oluwatosin Mulero. Evidence and source code are maintained in this repository.*
