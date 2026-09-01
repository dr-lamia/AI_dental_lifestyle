# Dental AI Coach / Elham's Index – Q1 revision

This branch contains the leakage-safe research implementation for the Dental AI Coach project.

## Why this branch exists
The earlier Streamlit implementation allowed tooth-level clinical variables that mathematically compose Elham's Index to enter the machine-learning predictor matrix. Those results measure partial reconstruction of the index and must not be interpreted as independent predictive validity.

The revised workflow separates:

1. **Outcome calculation and quality control** – direct Elham component counts are used to verify the stored outcome.
2. **Machine-learning estimation** – only independently collected demographic, socioeconomic, behavioral, dietary and salivary variables are predictors.
3. **Rule-based clinical support** – tooth-level findings are used here, not as ML predictors.

## Primary files
- `analysis_pipeline.py` – outcome audit, leakage-safe predictors, five-fold out-of-fold validation, baselines and fairness screening.
- `streamlit_app_q1.py` – revised research interface with Data QC, study design, performance, XAI, scenario, fairness and rule-based care modules.
- `METHODOLOGY_Q1.md` – manuscript-ready methods specification.
- `data/DATA_DICTIONARY_Q1.md` – predictor/outcome/clinical-rule variable roles.
- `scripts/prepare_dataset.py` – derived-dataset/QC utility; the source file is not overwritten.
- `.github/workflows/q1-validation.yml` – reproducible validation job when GitHub Actions is available for the repository.

## Dataset used by the revised app
The Q1 app intentionally reads the richer root-level file:

`no_recommendation_dental_dataset_cleaned_keep_including_wisdom.csv`

The older `data/` CSV is a reduced deployment copy and should not be treated as the definitive modeling source.

## Outcome-QC rule
A record is eligible for supervised validation only when the stored `elham_s_index_including_wisdom` is present, all documented direct component counts are present, and the stored index equals their sum. Failed records are preserved and reported; they are not silently repaired.

## Validation
- shuffled five-fold cross-validation, random seed 42;
- pooled out-of-fold R², MAE and RMSE;
- mean and median outcome baselines;
- Random Forest, XGBoost and an equal-weight blend when XGBoost is available;
- final full-cohort refitting is for prototype deployment only and is not reported as validation performance.

## Interpretability and safeguards
Grouped SHAP is descriptive of model behavior and is explicitly non-causal. The what-if simulator is associational, not an intervention-effect estimator. Fairness screening uses out-of-fold predictions and a prespecified subgroup-MAE flag. The rule-based care layer is clinician-editable decision support and does not replace examination, diagnosis, professional judgment or local guidelines.

## Current data-status warning
Accessible GitHub/Drive copies do not currently document a complete 500-participant modeling cohort with valid Elham outcomes. The manuscript sample-flow statement must therefore be reconciled with the original study source before final submission. Do not merge this branch or replace the production Streamlit app until the final source cohort and audited metrics are confirmed.

A local audit of the accessible Drive/GitHub-aligned copies identified 205 participant IDs, of which 159 records satisfied the documented Elham arithmetic; 46 failed the arithmetic check. Exploratory five-fold validation after excluding target-derived predictors showed performance close to simple baselines rather than the previously reported very high R². These exploratory values should not be treated as final manuscript results until the intended complete study cohort is located and reconciled.
