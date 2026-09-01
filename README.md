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

The repository's older processed CSV remains only a temporary branch default. It should not be treated as the definitive manuscript dataset. The audited 160-case reconstruction should be used for final manuscript analysis and production deployment.

## Component-specific modeling
The current cohort supports separate exploratory/internal models for the components with adequate prevalence, principally:

- missing teeth including wisdom teeth;
- decayed teeth;
- filled teeth;
- hypocalcified teeth.

Rare findings such as fluorosis, erosion, abrasion, attrition, abfraction, sealants, fractures, pontics, implant crowns and veneers remain part of the detailed clinical profile but should not receive separate ML models when event counts are too small.

## Current component-model findings
The leakage-safe analyses show only modest predictive signal from the independent non-clinical variables. In the audited analysis, Random Forest explained approximately 13% of the variation in filled teeth, 8% in missing teeth and 7.5% in decayed teeth, while hypocalcification was not meaningfully predicted. These results support risk profiling and explanation rather than replacement of the clinical examination.

## Primary files
- `analysis_pipeline.py` – leakage-safe preprocessing, five-fold validation, baselines and fairness screening.
- `component_pipeline.py` – component-specific Elham modeling.
- `streamlit_app_component.py` – redesigned dashboard with detailed oral-health profile, component-specific AI, patient-level explanations and personalized action plan.
- `streamlit_app_q1.py` – previous total-score research interface retained for comparison only.
- `METHODOLOGY_Q1.md` – methods specification; should be updated to the component-specific manuscript framing before submission.
- `data/DATA_DICTIONARY_Q1.md` – predictor/outcome/clinical-rule variable roles.
- `scripts/prepare_dataset.py` – derived-dataset/QC utility; the source file is not overwritten.

## Clinical recommendation layer
The recommendation engine combines direct clinical findings with modifiable patient factors to generate a clinician-reviewable personalized oral-health action plan. It is not an autonomous prescription system. Diagnosis, treatment selection, medication/therapeutic dosing and definitive recall intervals remain the responsibility of the treating clinician and applicable guidance.

## Next research phase
A longitudinal follow-up study should repeat the detailed Elham assessment after a prespecified interval, such as 12 months. Baseline clinical profile, lifestyle, diet, socioeconomic and salivary factors could then be tested as predictors of change in individual oral-health components, allowing genuine future-risk forecasting.

## Deployment status
This branch remains a research revision and has not replaced the production Streamlit app. Do not merge or deploy as the final clinical/research version until the audited 160-case dataset is used directly and the component-specific analyses are rerun reproducibly on that exact source.
