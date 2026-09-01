# Revised methodology for Q1 manuscript

## Study objective
The primary objective is to develop and internally evaluate an explainable artificial intelligence–based personalized oral-health decision-support framework that examines demographic, socioeconomic, behavioral, dietary and salivary factors in relation to individual clinical components recorded by Elham's Index, constructs patient-specific oral-health risk profiles, and supports tailored preventive and clinical recommendations.

Elham's Index is therefore treated primarily as a detailed clinical oral-health profile rather than as a single machine-learning target. Each sufficiently prevalent component is analyzed separately. The current cross-sectional study evaluates associations and internal predictive/risk-stratification performance; it does not claim causal effects or future disease forecasting.

## Raw-data reconstruction and participant flow
The analysis is based on the two original raw workbooks supplied for the study. The non-clinical workbook contains socioeconomic, behavioral, dietary, other-factor and salivary records for 205 participant IDs. The raw clinical workbook contains original tooth-level clinical assessment for 160 participant IDs. Linkage is performed by participant ID, yielding 160 participants with both clinical and non-clinical data for the intended primary analysis. The remaining 45 non-clinical records do not have a corresponding raw clinical assessment in the supplied clinical workbook and are not assigned fabricated or propagated clinical outcomes.

The source workbooks are preserved unchanged. Cleaning and linkage are performed programmatically in a derived analysis copy.

## Detailed Elham clinical profile and quality control
Elham's Index records the following direct clinical component counts: missing teeth including wisdom teeth, decayed teeth, filled teeth, hypoplasia, hypocalcification, fluorosis, erosion, abrasion, attrition, abfraction, sealants, fractures, pontics, abutment crowns, implant crowns and veneers.

For quality control, the stored overall Elham total is compared with the arithmetic sum of these direct components. In the audited raw clinical cohort, all 160 matched clinical records passed this arithmetic check. The total score is retained as a descriptive summary and quality-control variable, but the component-specific models use the individual clinical findings as separate outcomes.

Direct Elham components, their normalized copies, DMF, sound-teeth counts, treatment indices, composite clinical scores, derived risk scores/bands and downstream treatment-phase variables are excluded from the independent predictor matrix. This prevents target leakage and circular prediction.

## Component-specific outcomes
Separate predictive models are considered only for clinical components with sufficient event frequency and variation in the 160-participant cohort. In the current data, the principal modelable outcomes are:

- number of missing teeth including wisdom teeth;
- number of decayed teeth;
- number of filled teeth;
- number of hypocalcified teeth.

Rare findings such as fluorosis, erosion, abrasion, attrition, abfraction, sealants, fractures, pontics, implant crowns and veneers remain part of the detailed clinical profile and clinical recommendation layer but are not assigned separate machine-learning models when the cohort contains too few events for reliable estimation.

## Predictor domains
The primary independent predictor matrix contains variables collected independently of the Elham clinical component counts. These include:

- demographic variables: age and gender;
- socioeconomic/access variables: grade, household/living variables, pocket money, parental education and occupation, reported income category, insurance, access to oral health care, visit frequency and affordability;
- oral-health behavior variables: brushing history and frequency, brush/toothpaste use, interdental cleaning and mouthrinse;
- diet/lifestyle variables: meal and snack patterns, snack content, sugar and sticky-food exposure, carbonated/acidic exposures, hydration, food-group variables, exercise, smoking, supplements and medications;
- salivary variables: hydration/flow-related fields, consistency, pH, quantity, buffering capacity, mutans streptococci category and lactobacilli category.

School, current residence, place of birth and nationality are excluded from the primary prediction matrix to reduce site/geographic memorization and improve transportability. They may be retained for descriptive analysis. Periodontal status and other contemporaneous clinical variables should be analyzed separately as sensitivity variables rather than mixed into the primary non-clinical predictor set.

## Preprocessing
All learned preprocessing is fitted inside each validation training fold. Numeric predictors are imputed using the training-fold median. Categorical predictors are imputed using the most frequent training-fold category and one-hot encoded with unknown-category handling. When supported by the software version, very rare categorical levels are grouped to reduce sparse one-off categories.

Free-text values are not aggressively recoded into investigator-invented clinical classes. Processing is limited to whitespace/missing-value standardization and conservative harmonization of explicit binary responses. Reported income and pocket-money variables remain questionnaire categories unless a prespecified reproducible numeric conversion is available from the original instrument.

## Model development
Two prespecified tree-based regression algorithms are evaluated for each sufficiently prevalent component:

1. Random Forest regressor with 450 trees, random seed 42, minimum terminal leaf size 2 and `max_features=1.0`.
2. XGBoost regressor with 600 boosting iterations, learning rate 0.05, maximum depth 5, subsample 0.90, column subsample 0.90, L2 regularization 1.0, histogram tree method and random seed 42.

An equal-weight ensemble is calculated as the arithmetic mean of Random Forest and XGBoost predictions. Hyperparameters are prespecified from the development workflow and are not selected from the validation outcomes.

Because clinical component counts cannot be negative, displayed prototype predictions are constrained to a minimum of zero. This display constraint does not change the observed clinical outcome values.

## Internal validation and baseline comparison
Primary performance estimation uses shuffled five-fold cross-validation with random seed 42. Every analyzed participant is predicted by a model that was not trained on that participant. Performance is calculated from pooled out-of-fold predictions.

For each component, R-squared (R²), mean absolute error (MAE) and root mean squared error (RMSE) are reported. Mean-outcome and median-outcome baselines are shown alongside the machine-learning models so that any predictive gain is interpreted relative to simple non-ML rules.

The currently observed component-specific performance is modest. This supports interpreting the AI as a risk-profiling and explanatory aid rather than a replacement for direct dental examination.

After validation, models may be refitted to the complete eligible cohort for use in the research prototype. Full-cohort refit performance is not reported as validation performance.

## Patient-specific oral-health profile
For each participant, the interface displays the detailed observed Elham clinical profile rather than only one total score. Observed component counts remain the clinical reference. Model-estimated values are shown separately and are explicitly labeled as model estimates derived from independent patient factors.

The clinical profile also retains rare findings that are not modeled separately, allowing the clinician to view the complete documented oral status even when a machine-learning estimate is inappropriate.

## Explainable artificial intelligence
SHapley Additive exPlanations (SHAP) are used to describe how the fitted component-specific models use the independent predictors. One-hot encoded variables are regrouped to their parent variables. Patient-level SHAP values indicate whether a variable pushes the fitted component estimate upward or downward relative to the model baseline.

SHAP values are model-attribution measures and not causal estimates. They do not prove that a factor caused the observed disease, that changing the factor will reverse the disease, or that the magnitude of a SHAP value represents a treatment effect.

## Personalized recommendation layer
The recommendation engine combines two information sources:

1. the directly observed detailed clinical Elham profile; and
2. modifiable patient factors identified from the behavioral, dietary, socioeconomic/access and salivary data.

The system then produces a clinician-reviewable personalized oral-health action plan. Examples include prioritization of caries control, review of existing restorations, verification of causes of missing teeth, assessment of developmental enamel defects, oral-hygiene reinforcement, interdental-cleaning advice, reduction in free-sugar/snack frequency, reduction in acidic/carbonated beverage exposure, salivary-risk review and appropriate professional preventive or restorative assessment.

The output is decision support rather than an autonomous prescription. Diagnosis, treatment selection, therapeutic agent choice/dose/frequency and definitive recall intervals remain subject to the treating clinician's judgment, medical history, age, local regulations and applicable clinical guidance.

## Streamlit research prototype
The redesigned component-specific Streamlit interface contains six modules:

1. detailed oral-health profile;
2. component-specific AI estimates;
3. patient-level model explanation;
4. personalized oral-health action plan;
5. internal validation results;
6. research-design and limitation statements.

The prototype distinguishes observed clinical findings from model estimates and explicitly states that the current study is cross-sectional.

## Future forecasting phase
Prediction of future oral-health change requires longitudinal data. A future follow-up study should repeat the detailed Elham assessment after a prespecified interval, for example 12 months. Baseline clinical components together with baseline demographic, behavioral, dietary, socioeconomic and salivary variables could then be evaluated as predictors of change in individual oral-health components, such as newly decayed teeth or worsening/restorative treatment burden.

Only after such longitudinal validation should the system be described as forecasting future oral-health outcomes.

## Reporting implication
The previously reported near-perfect R² values were generated when variables mathematically defining the Elham total were allowed into the predictor matrix and should not be reported as independent predictive validity. The manuscript should report the raw reconstructed participant flow, component-specific leakage-safe analyses, baseline comparisons, modest current predictive performance, and the role of the system as an explainable personalized decision-support framework.
