# Revised methodology for Q1 manuscript

## Study objective
The primary modeling objective is to estimate Elham's Index from variables available independently of the index calculation. The primary predictor set therefore contains demographic, socioeconomic, oral-hygiene, dietary and salivary variables only. Tooth-level findings that mathematically compose Elham's Index are excluded from machine-learning prediction and retained for outcome quality control and the clinician-editable rule-based care layer.

This framing deliberately separates two functions of the prototype: (1) machine-learning estimation of dental-status burden from independently collected non-index variables and (2) deterministic/rule-based clinical support using recorded tooth findings.

## Dataset governance and outcome quality control
The source dataset is preserved unchanged. Cleaning is performed programmatically in memory or in a derived analysis copy. Missing textual values are standardized conservatively, true numeric variables are coerced to numeric form, and explicit binary fields are harmonized without overwriting the source record.

The application documents Elham's Index as the sum of the following direct component counts: missing teeth including wisdom teeth, decayed teeth, filled teeth, hypoplasia, hypocalcification, fluorosis, erosion, abrasion, attrition, abfraction, sealants, fractures, pontics, abutment crowns, implant crowns and veneers. Before supervised modeling, every row is audited against this documented arithmetic. A row is eligible for model validation only when the stored Elham outcome is present, all required direct components are present, and the stored outcome equals their component sum within numerical tolerance. Inconsistent rows are retained in the source dataset and exported in the quality-control report but are not silently corrected or used to estimate model performance.

All direct components, normalized copies of those variables, DMF, sound-teeth counts, treatment indices, composite clinical scores, derived risk scores/bands and downstream treatment-phase variables are excluded from machine-learning inputs. This prevents mathematical target leakage and circular prediction.

## Predictor domains
The primary independent predictor matrix includes variables available in the dataset from the following domains:

- demographic variables: age and gender;
- socioeconomic/access variables: grade, household/living variables, pocket money, parental education and occupation, reported income category, insurance, access to oral health care, visit frequency and affordability;
- oral-health behavior variables: brushing history and frequency, brush/toothpaste use, interdental cleaning and mouthrinse;
- diet/lifestyle variables: meal and snack patterns, snack content, sugar and sticky-food exposure, carbonated/acidic exposures, hydration, food-group variables, exercise, smoking, supplements and medications;
- salivary variables: hydration/flow-related fields, consistency, pH, quantity, buffering capacity, mutans streptococci category and lactobacilli category.

School, current residence, place of birth and nationality are excluded from the primary prediction matrix to reduce site/geographic memorization and improve transportability. They may be retained for descriptive analysis. Periodontal status, occlusion and other contemporaneous clinical variables are not included in the primary non-index model; if evaluated, they should be reported separately as sensitivity analyses rather than mixed into the main model.

## Preprocessing
All learned preprocessing is fitted within each validation training fold. Numeric predictors are imputed using the training-fold median. Categorical predictors are imputed using the most frequent training-fold category and one-hot encoded with unknown-category handling. When supported by the software version, categorical levels occurring fewer than two times are grouped by the encoder to reduce one-off sparse categories. Free-text values are not aggressively recoded into investigator-invented clinical classes; processing is limited to whitespace/missing-value standardization and conservative harmonization of explicit binary responses.

Reported income is treated as a questionnaire category rather than converted to a continuous monetary value unless a prespecified and reproducible conversion is justified from the original questionnaire. Pocket money remains categorical.

## Model development
Two prespecified tree-based regression algorithms are evaluated:

1. Random Forest regressor: 450 trees, random seed 42, minimum terminal leaf size 2 and all available candidate predictors considered according to the implementation's `max_features=1.0` setting.
2. XGBoost regressor: 600 boosting iterations, learning rate 0.05, maximum depth 5, subsample 0.90, column subsample 0.90, L2 regularization 1.0, histogram tree method and random seed 42.

An equal-weight ensemble is calculated as the arithmetic mean of Random Forest and XGBoost predictions. Hyperparameters are prespecified from development work and are not selected using the validation outcomes.

## Internal validation and baseline comparison
Primary performance estimation uses shuffled five-fold cross-validation with random seed 42. All performance estimates are calculated from pooled out-of-fold predictions, ensuring that every eligible participant is predicted by a model not trained on that participant. R-squared (R²), mean absolute error (MAE) and root mean squared error (RMSE) are reported.

Simple mean-outcome and median-outcome baselines are reported alongside the machine-learning models. This is necessary because a complex model should demonstrate improvement over trivial prediction rules before predictive usefulness is claimed.

After validation, each model is refitted to all quality-control-eligible observations for research-prototype deployment. Performance from this full-data refit is not reported as validation performance.

## Risk stratification
Low, moderate and high descriptive strata are defined using the 34th and 67th percentiles of the observed Elham Index among quality-control-eligible observations. These strata are used only for interface organization and the rule-based prevention layer; they are not model predictors. The thresholds are empirical and dataset-specific and require recalibration in external populations.

## Explainable artificial intelligence
SHapley Additive exPlanations (SHAP) are applied to a refitted tree model for descriptive model interpretation. One-hot encoded variables are regrouped to their original parent variables. Global importance is summarized as mean absolute grouped SHAP magnitude. Patient-level grouped SHAP values show whether a variable pushes a specific prediction upward or downward relative to the model baseline.

SHAP values are model-attribution measures rather than causal estimates. Mean absolute SHAP does not indicate direction, and neither global nor local SHAP establishes protection, harm, mechanism or expected treatment effect.

## What-if simulation
The research prototype permits modification of selected behavioral or salivary inputs and recalculates the model prediction. The displayed difference is explicitly described as a model-based association under a hypothetical input combination and must not be interpreted as the expected effect of an intervention.

## Fairness screening
Fairness screening is calculated from out-of-fold predictions. Subgroup MAE is evaluated for predefined demographic, socioeconomic and access variables; school is excluded. Only subgroups with at least 20 quality-control-eligible participants are assessed. A subgroup is flagged for review when its MAE is at least 1.5 times the overall out-of-fold MAE. This is a screening criterion only and does not establish either fairness or bias.

## Rule-based clinical decision-support layer
The machine-learning predictor is separated from the clinical rule layer. Recorded tooth-level findings are mapped to clinician-editable preventive and treatment considerations. The revised app avoids hard-coded prescription-strength fluoride or drug regimens and instead defers dose-specific decisions to age, medical history, local guidelines and clinician judgment. Recommendations are decision-support suggestions rather than autonomous prescriptions.

## Streamlit research prototype
The revised Streamlit interface contains seven modules: Data QC, Study design, Validated performance, Explainable AI, Patient scenario/what-if, Fairness audit and Rule-based care. It displays the source row count, number of outcome-QC-eligible rows and exclusions; distinguishes out-of-fold validation from final deployment refitting; and includes explicit research-use and non-causal interpretation statements.

## Reporting implication
The previously reported very high R²/low MAE values were generated from a predictor matrix that included variables mathematically defining the outcome and should not be reported as independent predictive validity. The revised manuscript must use only quality-controlled, leakage-safe out-of-fold results. The Orange workflow must be rebuilt using the same quality-control cohort, independent predictor set and validation design before any cross-platform comparison is retained.

The current repository/Drive copies should not be described as a 500-participant modeling cohort unless the complete 500-participant source dataset and valid Elham outcome labels can be documented. The manuscript sample flow should distinguish recruited/assessed participants, records available in the modeling file, outcome-QC exclusions and the final analyzed cohort.
