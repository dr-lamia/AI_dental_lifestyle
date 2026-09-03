# Dental AI Coach dataset roles

This document defines variable roles for the leakage-safe Q1 analysis. It does not alter the original observations.

## Outcome
- `elham_s_index_including_wisdom`: stored continuous Elham outcome.

## Identifier
- `id`: participant identifier; never used as a predictor.

## Direct Elham components used for arithmetic QC and rule-based care
The documented app calculation sums these counts. They must never enter the ML predictor matrix:

- `missing_0_including_wisdom_`
- `decayed_1`
- `filled_2`
- `hypoplasia_3`
- `hypocalcification_4`
- `fluorosis_5`
- `erosion_6`
- `abrasion_7`
- `attrition_8`
- `abfraction_9`
- `sealant_a`
- `fractured_h`
- `crown_pontic`
- `crown_abutment`
- `crown_implant`
- `veneer_f`

## Other target-derived variables excluded from ML
These include alternate/normalized component columns, composite clinical scores, or downstream outputs. The exact set is enforced in `analysis_pipeline.py` and includes, when present:

- `missing_0_excluding_wisdom_`, `missing_0_excluding_ortho_`
- normalized `__2` component columns
- `sound_teeth`, `sound_teeth__2`
- `dmf`, `dmft_auto`
- `elham_s_index_excluding_wisdom`
- `index_of_treatment`
- decay/filling/enamel-defect composite fields
- `teeth_total_est`
- `caries_risk_score`, `periodontal_risk_score`, `erosion_risk_score`
- derived risk bands
- derived treatment-phase indicators and `maintenance_recall_months`

## Primary independent predictor domains
Only columns present in the source file are used.

### Demographic
- `age`
- `gender`

### Socioeconomic and access
- `grade`
- `functional_status`
- `house_owned_or_rent`
- `i_live_with_my_parents`
- `of_family_members`
- `pocket_money`
- `father_s_education`
- `mother_s_education`
- `average_income`
- `father_s_job`
- `mother_s_job`
- `insurance`
- `access_to_oral_health_care`
- `frequency_of_visits`
- `affordability`

`pocket_money` remains categorical. Reported income is treated as a questionnaire category unless a separate prespecified conversion is justified from the original questionnaire.

### Oral-health behavior and lifestyle
- `start_of_brushing_when_i_was`
- `tooth_brushing_frequency`
- `time_of_tooth_brushing`
- `take_tooth_brush_to_school`
- `tooth_brush`
- `tooth_paste`
- `interdental_cleaning`
- `mouth_rinse`
- `habits`
- `diet`
- `snacks`
- `acidity`
- `hydration`
- `sugar`
- `sticky_food`
- `carbonated_beverages`
- `exercise_sports`
- `supplements`
- `medications`
- `smoking`
- `breakfast`, `lunch`, `dinner`
- `snacks_frequency`, `snack_content`
- `dairy_products`, `proteins`, `vegetables`, `fruits`, `spices`, `sweets`, `nuts`
- `cho_content`, `sugar_rich`, `vitamin_rich`, `fat_rich`
- `carbonated_beverages_diet`, `acidic_food_or_drinks`, `retention_in_mouth`
- `type_of_diet`

### Salivary / biological
- `level_of_hydration`
- `salivary_consistency`
- `salivary_ph`
- `salivary_quantity`
- `buffering_capacity`
- `mutans_load_in_saliva`
- `lactobacilli_load_in_saliva`

## Variables excluded from the primary prediction model for transportability
- `school`
- `current_residence`
- `place_of_birth`
- `nationality`

These may be retained for descriptive analyses. `school` is not included in the fairness screening.

## Contemporaneous clinical variables reserved for sensitivity analyses
The following are not direct Elham components but are excluded from the primary non-index model to keep its interpretation clear:

- `periodontal_status`
- `occlusion`
- `masseter_muscle_status`
- `orthodontic_treatment`
- `tmj`

If investigated, they should be reported as a separate sensitivity/expanded model rather than silently added to the primary predictor set.

## Outcome-QC eligibility
A row is eligible for supervised validation only when:

1. the stored `elham_s_index_including_wisdom` is available;
2. all 16 documented direct component counts are available; and
3. the stored outcome equals the sum of those components within numerical tolerance.

Failed rows remain in the source data and are exported in `target_arithmetic_qc.csv`; they are not silently corrected.

## Analysis safeguards
- Source files are preserved unchanged.
- Missing textual values are standardized only in the analysis layer.
- Numeric and categorical imputation is fitted within validation folds.
- One-hot encoding is fitted within validation folds with unknown-category handling.
- Rare one-off categorical levels are grouped by the encoder when supported.
- No target-derived feature may pass the leakage check in `analysis_pipeline.py`.
- Validation metrics are derived from pooled out-of-fold predictions, not predictions on training data.
- Mean and median baselines are reported alongside ML models.
