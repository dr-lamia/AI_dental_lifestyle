"""Create a derived, audited analysis copy for Dental AI Coach.

The source CSV is never overwritten. Outcome arithmetic is checked using the
same rules as the validation pipeline. Inconsistent records remain traceable in
the QC file and are marked as ineligible rather than silently corrected.
"""
from pathlib import Path
import json
import pandas as pd

from analysis_pipeline import (
    TARGET, TARGET_COMPONENTS, canonicalize, predictor_columns,
    validate_no_leakage, target_audit,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "no_recommendation_dental_dataset_cleaned_keep_including_wisdom.csv"
OUT = ROOT / "data" / "analysis_dataset_clean.csv"
AUDIT_OUT = ROOT / "data" / "target_arithmetic_qc.csv"
QC = ROOT / "data" / "dataset_qc.json"


def main():
    raw = pd.read_csv(SOURCE)
    df = canonicalize(raw)
    audit = target_audit(df)
    predictors = predictor_columns(df)
    validate_no_leakage(predictors)

    df_out = df.copy()
    df_out["q1_target_consistent"] = audit["target_consistent"].to_numpy()

    qc = {
        "n_source_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "target_missing": int(pd.to_numeric(df[TARGET], errors="coerce").isna().sum()),
        "n_qc_eligible": int(audit["target_consistent"].sum()),
        "n_qc_excluded": int((~audit["target_consistent"]).sum()),
        "candidate_predictors": predictors,
        "leakage_exclusions_present": [c for c in TARGET_COMPONENTS if c in df.columns],
        "notes": [
            "The source file was not modified.",
            "Eligibility requires the stored Elham target to equal the documented direct-component sum.",
            "Failed rows are retained and flagged rather than corrected.",
            "Target-derived variables remain available for QC and rule-based care only.",
            "School, current residence, place of birth and nationality are excluded from the primary prediction model.",
            "Learned imputation and categorical encoding are fitted inside validation folds."
        ],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT, index=False)
    audit.to_csv(AUDIT_OUT, index=False)
    QC.write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps(qc, indent=2))


if __name__ == "__main__":
    main()
