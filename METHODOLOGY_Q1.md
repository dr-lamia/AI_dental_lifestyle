# Revised methodology for Q1 manuscript

## Study objective
The primary modeling objective is to estimate Elham's Index from variables that are available independently of the index calculation, including demographic, behavioral, salivary, socioeconomic, and periodontal-status variables. Tooth-level variables that mathematically compose Elham's Index are not used as machine-learning predictors. They are retained only for index verification, descriptive analysis, and the clinician-editable rule-based care layer.

## Dataset governance and quality control
The source dataset is preserved unchanged. A reproducible preparation script standardizes missing categorical values as `Unknown`, coerces true numeric variables to numeric form, harmonizes explicit Yes/No variables conservatively, and produces a separate analysis copy. No participant value is imputed before data partitioning. The arithmetic relationship between the recorded Elham's Index and its component counts is audited and reported rather than silently corrected.

The following variables are treated as target-derived and excluded from machine-learning inputs: missing-tooth counts, decayed and filled tooth counts, hypoplasia, hypocalcification, fluorosis, erosion, abrasion, attrition, abfraction, sealants, fractures, crown/pontic/abutment/implant counts, veneers, sound-teeth count, DMF, and index of treatment. This exclusion is necessary to prevent target leakage and circular prediction.

School, place of birth, and nationality are excluded from predictive modeling to reduce site- and geography-specific memorization and improve transportability. School is also excluded from the fairness analysis.

## Predictor domains
Predictors are restricted to variables present in the dataset and available independently of the Elham Index calculation. They include age and gender; oral-hygiene behaviors; dietary exposures; hydration; salivary characteristics; mutans streptococci and lactobacilli categories; selected socioeconomic variables; medication/supplement information; and periodontal status where available.

## Preprocessing
Preprocessing is performed within each validation fold. Numeric variables are imputed using the median of the training fold. Categorical variables are imputed using the most frequent training-fold category and one-hot encoded with unknown-category handling. Free-text values are not aggressively reclassified into invented clinical categories; cleaning is limited to whitespace, missing-value standardization, and conservative harmonization of explicit Yes/No fields. This approach reduces investigator-induced recoding bias.

## Model development
Two tree-based regression algorithms are evaluated:

1. Random Forest regressor: 450 trees, random seed 42, minimum terminal leaf size 2.
2. XGBoost regressor: 600 boosting iterations, learning rate 0.05, maximum depth 5, subsample 0.90, column subsample 0.90, L2 regularization 1.0, histogram tree method, and random seed 42.

An equal-weight ensemble is calculated as the arithmetic mean of Random Forest and XGBoost predictions. Hyperparameters are prespecified from the development work and are not tuned on validation folds.

## Internal validation
Primary performance estimation uses shuffled 5-fold cross-validation with random seed 42. All reported performance estimates are calculated from pooled out-of-fold predictions, so every participant is predicted by a model that was not trained on that participant. R-squared (R²), mean absolute error (MAE), and root mean squared error (RMSE) are reported.

After internal validation, each model is refitted to the complete analysis cohort for prototype deployment. Performance from this full-data refit is not reported as validation performance.

## Risk stratification
Low, moderate, and high descriptive risk strata are defined from the 34th and 67th percentiles of the observed Elham Index distribution. Risk strata are used for user-interface organization and rule-based prevention output and are not model predictors. These thresholds are dataset-specific and require recalibration before use in another population.

## Explainable artificial intelligence
SHapley Additive exPlanations (SHAP) are applied to the refitted tree model for model interpretation. One-hot encoded variables are regrouped to their original parent variables. Global importance is summarized as mean absolute grouped SHAP magnitude. Patient-level grouped SHAP values show whether a variable pushes a specific prediction upward or downward relative to the model baseline.

SHAP values are interpreted as model-attribution measures only. Mean absolute SHAP values do not indicate effect direction, and neither global nor local SHAP results establish causality, protection, harm, or treatment effect.

## What-if simulation
The prototype permits modification of selected behavioral or salivary inputs and recalculates the predicted Elham Index. The displayed difference represents a model-based association under the modified input combination. It is explicitly labeled as non-causal and must not be interpreted as the expected clinical benefit of an intervention.

## Fairness screening
Fairness is assessed from out-of-fold predictions, not predictions from models fitted to the same observations. Subgroup MAE is calculated for predefined demographic and socioeconomic variables, excluding school. Only subgroups with at least 20 participants are evaluated. A subgroup is flagged for review when its MAE is at least 1.5 times the overall out-of-fold MAE. This threshold is a screening criterion and does not establish the presence or absence of algorithmic bias.

## Rule-based clinical decision-support layer
The machine-learning prediction is separated from the clinical rule layer. Recorded tooth-level findings are mapped to clinician-editable preventive and treatment considerations. Recommendations are phrased as decision-support suggestions rather than autonomous prescriptions and require adaptation to clinical examination, radiographic findings, patient preferences, contraindications, local guidelines, and regulatory requirements.

## Streamlit research prototype
The revised Streamlit prototype contains six modules: study design, validated performance, explainable AI, patient-level prediction and what-if simulation, fairness screening, and rule-based care. The interface clearly distinguishes validated out-of-fold performance from final full-cohort models used only for deployment. It also displays a research-use disclaimer and states that the application is not a substitute for professional clinical judgment.

## Reporting implication
The previously reported very high performance values obtained when target-component clinical counts were allowed to enter the predictor matrix should not be used as evidence of predictive validity. The revised manuscript should report only leakage-safe out-of-fold metrics generated by this pipeline. The Orange workflow should likewise be rerun after removal of all target-derived predictors before any cross-platform comparison is retained in the paper.
