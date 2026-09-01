"""Prepare a reproducible analysis copy of the Dental AI Coach dataset.

The script NEVER overwrites the source file. It standardizes missing values,
checks the Elham Index arithmetic, and writes a cleaned analysis CSV plus a QC
summary. Direct target components remain in the output for auditing and the
rule-based clinical layer, but analysis_pipeline.py excludes them from ML inputs.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

from analysis_pipeline import TARGET, TARGET_COMPONENTS, canonicalize, predictor_columns, validate_no_leakage

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "no_recommendation_dental_dataset_cleaned_keep_including_wisdom.csv"
OUT = ROOT / "data" / "analysis_dataset_clean.csv"
QC = ROOT / "data" / "dataset_qc.json"


def main():
    raw = pd.read_csv(SOURCE)
    df = canonicalize(raw)

    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    present_components = [c for c in TARGET_COMPONENTS if c in df.columns]
    direct_components = [
        c for c in present_components
        if c not in {"missing_0_excluding_ortho_", "sound_teeth", "dmf", "index_of_treatment"}
    ]

    # Arithmetic audit only. A mismatch is reported, never silently corrected.
    component_sum = df[direct_components].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
    target = pd.to_numeric(df[TARGET], errors="coerce")
    diff = target - component_sum
    exact_match = np.isclose(diff.fillna(np.inf), 0.0)

    predictors = predictor_columns(df)
    validate_no_leakage(predictors)

    qc = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "target_missing": int(target.isna().sum()),
        "target_component_columns_present": present_components,
        "direct_component_columns_used_for_arithmetic_audit": direct_components,
        "rows_where_target_equals_component_sum": int(exact_match.sum()),
        "rows_where_target_differs_from_component_sum": int((~exact_match & target.notna()).sum()),
        "candidate_predictors": predictors,
        "leakage_exclusions": [c for c in TARGET_COMPONENTS if c in df.columns],
        "notes": [
            "Raw source file was not modified.",
            "Target-derived clinical variables are retained for audit/rule-based care only.",
            "School, place of birth, and nationality are excluded from predictive modeling.",
            "Missing categorical values are represented as Unknown; numeric missingness is handled inside each validation fold."
        ]
    }

    df.to_csv(OUT, index=False)
    QC.write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(f"Wrote: {OUT}")
    print(f"Wrote: {QC}")
    print(json.dumps(qc, indent=2))


if __name__ == "__main__":
    main()
