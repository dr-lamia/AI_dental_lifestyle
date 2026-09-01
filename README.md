# Dental AI Coach / Elham's Index – Q1 revision

This branch contains the scientifically revised implementation for the Dental AI Coach project.

## Revised study concept
Elham's Index is no longer treated primarily as one machine-learning target. Its individual clinical components are retained as a detailed oral-health profile. The AI layer evaluates sufficiently prevalent components separately using independently collected demographic, socioeconomic, behavioral, dietary and salivary predictors.

The intended workflow is:

**Detailed Elham clinical profile + independent patient factors → component-specific explainable AI → patient-specific risk profile → personalized preventive and clinical action plan.**

This is a cross-sectional decision-support study. It can evaluate associations and internal component-specific prediction/risk stratification, but it cannot yet claim causal effects or future oral-health forecasting. Genuine forecasting requires longitudinal follow-up with repeat clinical assessment.

## Why the older near-perfect results are not used
The earlier Streamlit implementation allowed tooth-level clinical variables that mathematically compose Elham's Index to enter the predictor matrix. Those results measure partial reconstruction of the index and must not be interpreted as independent predictive validity.

## Raw-data audit
The uploaded raw workbooks identify:

- 205 participants with socioeconomic, behavioral, dietary, other-factor and salivary data;
- 160 participants with original tooth-level clinical assessment;
- 160 matched participants for the intended primary analysis;
- 45 non-clinical records without a corresponding raw clinical assessment;
- all 160 raw clinical records pass the documented Elham component-sum arithmetic check.

The audited 160-case reconstruction is now embedded in the research branch and is the default source for `streamlit_app_audited.py`. The older processed CSV is retained only for historical/reproducibility comparison and should not be used for final manuscript claims.

## Component-specific modeling
The current cohort supports separate exploratory/internal models for the components with adequate prevalence, principally:

- missing teeth including wisdom teeth;
- decayed teeth;
- filled teeth;
- hypocalcified teeth.

The missing-tooth variable includes wisdom teeth and must therefore be interpreted cautiously in this adolescent cohort. It cannot automatically be labeled disease-related tooth loss; eruption/developmental status requires clinical verification.

Rare findings such as fluorosis, erosion, abrasion, attrition, abfraction, sealants, fractures, pontics, implant crowns and veneers remain part of the detailed clinical profile but should not receive separate ML models when event counts are too small.

## Current component-model findings
The leakage-safe analyses show only modest predictive signal from the independent non-clinical variables. In the audited analysis, Random Forest explained approximately 13% of the variation in filled teeth, 8% in missing teeth and 7.5% in decayed teeth, while hypocalcification was not meaningfully predicted. These results support risk profiling and explanation rather than replacement of the clinical examination.

## Primary files
- `analysis_pipeline.py` – leakage-safe preprocessing, five-fold validation, baselines and fairness screening.
- `component_pipeline.py` – component-specific Elham modeling.
- `streamlit_app_audited.py` – preferred audited 160-case research dashboard.
- `streamlit_app_component.py` – component-specific dashboard that can accept an uploaded audited dataset.
- `streamlit_app_q1.py` – previous total-score research interface retained for comparison only.
- `METHODOLOGY_Q1.md` – manuscript-ready component-specific methods specification.
- `data/master160_embedded.py` – embedded deidentified 160-case audited analysis cohort used by the preferred research app.
- `data/raw_reconstructed_160.csv` – deidentified reconstructed analysis table retained for transparent inspection.
- `data/DATA_DICTIONARY_Q1.md` – predictor/outcome/clinical-rule variable roles.
- `scripts/prepare_dataset.py` – derived-dataset/QC utility; the source file is not overwritten.

## Personalized recommendation layer
The preferred audited app separates three outputs:

1. direct clinical priorities from the detailed Elham examination;
2. modifiable behavioral/dietary/salivary factors that should be reviewed;
3. clinician-reviewable preventive and clinical recommendations.

Patient-level SHAP explanations are displayed separately as model-attribution signals. They are not treated as causal effects and are not automatically converted into treatment instructions.

The recommendation engine is not an autonomous prescription system. Diagnosis, treatment selection, medication/therapeutic dosing and definitive recall intervals remain the responsibility of the treating clinician and applicable guidance.

## Next research phase
A longitudinal follow-up study should repeat the detailed Elham assessment after a prespecified interval, such as 12 months. Baseline clinical profile, lifestyle, diet, socioeconomic and salivary factors could then be tested as predictors of change in individual oral-health components, allowing genuine future-risk forecasting.

## Deployment status
The audited component-specific research app is now implemented on the revision branch, but the public production Streamlit deployment has not been replaced. The branch should undergo a final runtime check before merge/deployment. The manuscript should report the audited 160-case analysis and should not use the previous leakage-affected near-perfect metrics.
