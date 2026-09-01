"""Leakage-safe analysis pipeline for Dental AI Coach.

Scientific design:
- Outcome: elham_s_index_including_wisdom
- Predictors are restricted to variables that are NOT mathematical components or
  downstream derivatives of Elham's Index.
- Clinical tooth-count fields are retained only for the rule-based treatment layer.
- Performance is estimated from out-of-fold predictions using 5-fold CV.
- Final models are refit on all participants only after validation for deployment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import re
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

RANDOM_STATE = 42
N_SPLITS = 5
TARGET = "elham_s_index_including_wisdom"
ID_COL = "id"

# Variables that mathematically define, reproduce, or are downstream derivatives
# of the target. They MUST NOT enter the predictive model.
TARGET_COMPONENTS = [
    "missing_0_including_wisdom_", "missing_0_excluding_ortho_", "decayed_1",
    "filled_2", "hypoplasia_3", "hypocalcification_4", "fluorosis_5",
    "erosion_6", "abrasion_7", "attrition_8", "abfraction_9", "sealant_a",
    "fractured_h", "crown_pontic", "crown_abutment", "crown_implant",
    "veneer_f", "sound_teeth", "dmf", "index_of_treatment"
]

# Candidate predictors available independently of the Elham Index calculation.
# Only columns actually present in the dataset are used.
DEMOGRAPHIC_COLS = ["age", "gender"]
BEHAVIOR_COLS = [
    "start_of_brushing_when_i_was", "tooth_brushing_frequency",
    "time_of_tooth_brushing", "take_tooth_brush_to_school", "tooth_brush",
    "tooth_paste", "interdental_cleaning", "mouth_rinse", "diet", "acidity",
    "hydration", "sugar", "carbonated_beverages", "exercise_sports",
    "supplements", "medications", "breakfast", "lunch", "dinner",
    "snacks_frequency", "snack_content"
]
SALIVARY_COLS = [
    "level_of_hydration", "salivary_consistency", "salivary_ph",
    "salivary_quantity", "buffering_capacity", "mutans_load_in_saliva",
    "lactobacilli_load_in_saliva"
]
SES_COLS = [
    "current_residence", "functional_status", "of_family_members",
    "pocket_money", "father_s_education", "mother_s_education",
    "average_income", "father_s_job", "mother_s_job", "insurance"
]
OTHER_CLINICAL_COLS = ["periodontal_status"]

# School and place of birth are deliberately excluded from predictive modeling to
# reduce site-specific/geographic memorization and improve transportability.
EXCLUDED_CONTEXT_COLS = ["school", "place_of_birth", "nationality"]

FAIRNESS_COLS = [
    "gender", "current_residence", "functional_status", "of_family_members",
    "pocket_money", "father_s_education", "mother_s_education",
    "average_income", "father_s_job", "mother_s_job", "insurance"
]


def _clean_text(x):
    if pd.isna(x):
        return "Unknown"
    s = str(x).strip()
    if s.lower() in {"", "nan", "none", "n/a", "na", "unknown", "unk"}:
        return "Unknown"
    return s


def _yes_no_unknown(x):
    s = _clean_text(x).lower()
    if s in {"yes", "y", "1", "true"} or s.startswith("yes "):
        return "Yes"
    if s in {"no", "n", "0", "false"} or s.startswith("no "):
        return "No"
    return "Unknown" if s == "unknown" else _clean_text(x).title()


def canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    """Conservative cleaning; raw data are never overwritten."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    # numeric fields
    if "age" in out:
        out["age"] = pd.to_numeric(out["age"], errors="coerce")
    if TARGET in out:
        out[TARGET] = pd.to_numeric(out[TARGET], errors="coerce")
    for c in TARGET_COMPONENTS:
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # explicit yes/no variables
    for c in ["pocket_money", "insurance", "take_tooth_brush_to_school",
              "interdental_cleaning", "mouth_rinse", "acidity"]:
        if c in out:
            out[c] = out[c].map(_yes_no_unknown)

    # all other strings: trim and standardize missingness only; do not invent classes
    for c in out.select_dtypes(include="object").columns:
        if c not in {"pocket_money", "insurance", "take_tooth_brush_to_school",
                     "interdental_cleaning", "mouth_rinse", "acidity"}:
            out[c] = out[c].map(_clean_text)

    # obvious case-only harmonization
    if "gender" in out:
        out["gender"] = out["gender"].astype(str).str.strip().str.title().replace({"Unknown": "Unknown"})
    return out


def predictor_columns(df: pd.DataFrame) -> List[str]:
    ordered = DEMOGRAPHIC_COLS + BEHAVIOR_COLS + SALIVARY_COLS + SES_COLS + OTHER_CLINICAL_COLS
    cols = [c for c in ordered if c in df.columns]
    forbidden = set(TARGET_COMPONENTS + [TARGET, ID_COL] + EXCLUDED_CONTEXT_COLS)
    return [c for c in cols if c not in forbidden]


def validate_no_leakage(cols: List[str]) -> None:
    forbidden = set(TARGET_COMPONENTS + [TARGET])
    overlap = sorted(forbidden.intersection(cols))
    if overlap:
        raise ValueError(f"Target leakage detected: {overlap}")


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    transformers = []
    if num_cols:
        transformers.append(("num", SimpleImputer(strategy="median"), num_cols))
    if cat_cols:
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", make_ohe()),
        ])
        transformers.append(("cat", cat_pipe, cat_cols))
    return ColumnTransformer(transformers, remainder="drop"), num_cols, cat_cols


def build_rf(X: pd.DataFrame) -> Pipeline:
    pre, _, _ = build_preprocessor(X)
    model = RandomForestRegressor(
        n_estimators=450,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    return Pipeline([("pre", pre), ("model", model)])


def build_xgb(X: pd.DataFrame) -> Pipeline:
    if not XGB_AVAILABLE:
        raise RuntimeError("xgboost is not installed")
    pre, _, _ = build_preprocessor(X)
    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("pre", pre), ("model", model)])


def metric_dict(y_true, y_pred) -> Dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
    }


@dataclass
class ValidationResult:
    metrics: Dict[str, Dict[str, float]]
    oof: pd.DataFrame
    predictors: List[str]
    rf_final: Pipeline
    xgb_final: Pipeline | None
    risk_bins: Tuple[float, float]


def fit_validate(df: pd.DataFrame) -> ValidationResult:
    df = canonicalize(df)
    df = df.loc[df[TARGET].notna()].reset_index(drop=True)
    cols = predictor_columns(df)
    validate_no_leakage(cols)
    X = df[cols].copy()
    y = df[TARGET].astype(float).to_numpy()

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    pred_rf = np.full(len(df), np.nan)
    pred_xgb = np.full(len(df), np.nan) if XGB_AVAILABLE else None

    for train_idx, test_idx in kf.split(X):
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        ytr = y[train_idx]
        rf = build_rf(Xtr)
        rf.fit(Xtr, ytr)
        pred_rf[test_idx] = rf.predict(Xte)
        if XGB_AVAILABLE:
            xgb = build_xgb(Xtr)
            xgb.fit(Xtr, ytr)
            pred_xgb[test_idx] = xgb.predict(Xte)

    metrics = {"Random Forest": metric_dict(y, pred_rf)}
    oof = pd.DataFrame({"observed": y, "rf": pred_rf}, index=df.index)
    if XGB_AVAILABLE:
        pred_blend = 0.5 * (pred_rf + pred_xgb)
        metrics["XGBoost"] = metric_dict(y, pred_xgb)
        metrics["Blend"] = metric_dict(y, pred_blend)
        oof["xgb"] = pred_xgb
        oof["blend"] = pred_blend

    # Risk thresholds are empirical outcome quantiles. They are used only for
    # descriptive/app stratification, not as model inputs.
    q34, q67 = np.quantile(y, [0.34, 0.67])

    rf_final = build_rf(X)
    rf_final.fit(X, y)
    xgb_final = None
    if XGB_AVAILABLE:
        xgb_final = build_xgb(X)
        xgb_final.fit(X, y)

    return ValidationResult(
        metrics=metrics,
        oof=oof,
        predictors=cols,
        rf_final=rf_final,
        xgb_final=xgb_final,
        risk_bins=(float(q34), float(q67)),
    )


def fairness_table(df: pd.DataFrame, oof: pd.DataFrame, pred_col="blend",
                   min_n=20, multiplier=1.5) -> pd.DataFrame:
    """Screen subgroup MAE using OOF predictions; this is not proof of fairness."""
    data = canonicalize(df).loc[df[TARGET].notna()].reset_index(drop=True)
    pred_col = pred_col if pred_col in oof.columns else "rf"
    overall = mean_absolute_error(oof["observed"], oof[pred_col])
    rows = []
    for col in [c for c in FAIRNESS_COLS if c in data.columns and c != "school"]:
        for level, idx in data.groupby(col, dropna=False).groups.items():
            idx = list(idx)
            if len(idx) < min_n:
                continue
            mae = mean_absolute_error(oof.loc[idx, "observed"], oof.loc[idx, pred_col])
            rows.append({
                "variable": col,
                "group": str(level),
                "n": len(idx),
                "MAE": float(mae),
                "overall_MAE": float(overall),
                "ratio": float(mae / overall) if overall > 0 else np.nan,
                "flag_for_review": bool(mae >= multiplier * overall),
            })
    return pd.DataFrame(rows).sort_values(["flag_for_review", "ratio"], ascending=[False, False])
