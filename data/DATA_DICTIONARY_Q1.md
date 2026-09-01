# Dental AI Coach dataset roles

This file documents how the existing study dataset is used in the revised Q1 analysis. It does not change the original observations.

## Outcome
- `elham_s_index_including_wisdom`: continuous target outcome.

## Identifier
- `id`: participant identifier; never used as a predictor.

## Excluded target-derived variables
These fields are retained for audit and rule-based clinical output but must not enter the machine-learning predictor matrix because they directly compose, reproduce, or are downstream derivatives of the target:

- `missing_0_including_wisdom_`
- `missing_0_excluding_ortho_`
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
- `sound_teeth`
- `dmf`
- `index_of_treatment`

## Candidate predictor domains
### Demographic
- `age`
- `gender`

### Oral-health behavior and lifestyle
- `start_of_brushing_when_i_was`
- `tooth_brushing_frequency`
- `time_of_tooth_brushing`
- `take_tooth_brush_to_school`
- `tooth_brush`
- `tooth_paste`
- `interdental_cleaning`
- `mouth_rinse`
- `diet`
- `acidity`
- `hydration`
- `sugar`
- `carbonated_beverages`
- `exercise_sports`
- `supplements`
- `medications`
- `breakfast`
- `lunch`
- `dinner`
- `snacks_frequency`
- `snack_content`

### Salivary / biological
- `level_of_hydration`
- `salivary_consistency`
- `salivary_ph`
- `salivary_quantity`
- `buffering_capacity`
- `mutans_load_in_saliva`
- `lactobacilli_load_in_saliva`

### Socioeconomic/contextual
- `current_residence`
- `functional_status`
- `of_family_members`
- `pocket_money`
- `father_s_education`
- `mother_s_education`
- `average_income`
- `father_s_job`
- `mother_s_job`
- `insurance`

### Other clinical variable independent of the Elham score calculation
- `periodontal_status`

## Context variables deliberately excluded from the prediction model
- `school`: excluded to reduce site-specific memorization; also excluded from fairness tables.
- `place_of_birth`: excluded to improve transportability and avoid unnecessary geographic proxies.
- `nationality`: excluded to avoid unnecessary population-proxy effects in this single-cohort model.

## Data-quality rules
- The raw CSV is preserved unchanged.
- Missing categorical values are represented as `Unknown` during analysis.
- Numeric imputation occurs within validation folds only.
- Categorical encoding is fitted within validation folds only.
- No target-derived feature may pass the leakage check in `analysis_pipeline.py`.
- The arithmetic relationship between the target and its direct component counts is audited rather than silently corrected.
