"""Leakage-safe validation pipeline for Dental AI Coach / Elham's Index.

The Elham outcome is a deterministic sum of tooth-level component counts in the
current application. Those components are therefore never used as ML predictors.
Rows whose stored Elham outcome is inconsistent with the documented component
sum are flagged and excluded from supervised model validation rather than being
silently corrected.

Primary ML question: how much of the cross-sectional Elham dental-status burden
can be estimated from independently collected demographic, socioeconomic,
behavioural, dietary and salivary variables?
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

# Direct mathematical components of the Elham Index used by the app.
ELHAM_DIRECT_COMPONENTS = [
    "missing_0_including_wisdom_", "decayed_1", "filled_2", "hypoplasia_3",
    "hypocalcification_4", "fluorosis_5", "erosion_6", "abrasion_7",
    "attrition_8", "abfraction_9", "sealant_a", "fractured_h",
    "crown_pontic", "crown_abutment", "crown_implant", "veneer_f",
]

# Broader set of direct components, normalized duplicates, composite scores and
# downstream care variables that must not enter the ML predictor matrix.
TARGET_COMPONENTS = ELHAM_DIRECT_COMPONENTS + [
    "missing_0_excluding_wisdom_", "missing_0_excluding_ortho_", "sound_teeth",
    "dmf", "index_of_treatment", "elham_s_index_excluding_wisdom",
    "missing_0__2", "missing_0_2__2", "decayed_1__2", "filled_2__2",
    "hypoplasia_3__2", "hypocalcification_4__2", "fluorosis_5__2",
    "erosion_6__2", "abrasion_7__2", "attrition_8__2", "abfraction_9__2",
    "sealant_a__2", "fractured_h__2", "crown_pontic__2",
    "crown_abutment__2", "crown_implant__2", "veneer_f__2", "sound_teeth__2",
    "decayed_1_filled_2", "decayed(1),filled(2)&hypoplasia(3)",
    "decayed_1_filled_2_hypocalcified_4", "decayed_1_filled_2__2",
    "decayed_1_filled_2_hypoplasia_3", "decayed_1_filled_2_hypocalcified_4__2",
    "dmft_auto", "teeth_total_est", "caries_risk_score", "periodontal_risk_score",
    "erosion_risk_score", "caries_risk_band", "periodontal_risk_band",
    "erosion_risk_band", "need_emergency_phase", "need_disease_control_phase",
    "need_restorative_phase", "maintenance_recall_months",
]

DEMOGRAPHIC_COLS = ["age", "gender"]
SES_COLS = [
    "grade", "functional_status", "house_owned_or_rent", "i_live_with_my_parents",
    "of_family_members", "pocket_money", "father_s_education",
    "mother_s_education", "average_income", "father_s_job", "mother_s_job",
    "insurance", "access_to_oral_health_care", "frequency_of_visits",
    "affordability",
]
BEHAVIOR_COLS = [
    "start_of_brushing_when_i_was", "tooth_brushing_frequency",
    "time_of_tooth_brushing", "take_tooth_brush_to_school", "tooth_brush",
    "tooth_paste", "interdental_cleaning", "mouth_rinse", "habits", "diet",
    "snacks", "acidity", "hydration", "sugar", "sticky_food",
    "carbonated_beverages", "exercise_sports", "supplements", "medications",
    "smoking", "breakfast", "lunch", "dinner", "snacks_frequency",
    "snack_content", "dairy_products", "proteins", "vegetables", "fruits",
    "spices", "sweets", "nuts", "cho_content", "sugar_rich", "vitamin_rich",
    "fat_rich", "carbonated_beverages_diet", "acidic_food_or_drinks",
    "retention_in_mouth", "type_of_diet",
]
SALIVARY_COLS = [
    "level_of_hydration", "salivary_consistency", "salivary_ph",
    "salivary_quantity", "buffering_capacity", "mutans_load_in_saliva",
    "lactobacilli_load_in_saliva",
]

# Current residence, school, nationality and place of birth are retained for
# descriptive/fairness work but excluded from the primary prediction model to
# limit geographic/site memorization.
EXCLUDED_CONTEXT_COLS = ["school", "current_residence", "place_of_birth", "nationality"]

FAIRNESS_COLS = [
    "gender", "current_residence", "grade", "functional_status",
    "house_owned_or_rent", "i_live_with_my_parents", "of_family_members",
    "pocket_money", "father_s_education", "mother_s_education", "average_income",
    "father_s_job", "mother_s_job", "insurance", "access_to_oral_health_care",
    "frequency_of_visits", "affordability",
]

MISSING_TOKENS = {"", "nan", "none", "n/a", "na", "unknown", "unk", "-"}


def _clean_text(x):
    if pd.isna(x):
        return "Unknown"
    s = str(x).strip()
    return "Unknown" if s.lower() in MISSING_TOKENS else s


def _yes_no_unknown(x):
    s = _clean_text(x).lower()
    if s in {"yes", "y", "1", "true"} or s.startswith("yes "):
        return "Yes"
    if s in {"no", "n", "0", "false"} or s.startswith("no "):
        return "No"
    return "Unknown" if s == "unknown" else _clean_text(x).title()


def canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    """Conservative, reproducible cleaning; the source file is never overwritten."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    numeric_cols = ["age", TARGET] + TARGET_COMPONENTS
    for c in numeric_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    yn_cols = [
        "pocket_money", "take_tooth_brush_to_school", "interdental_cleaning",
        "mouth_rinse", "acidity", "sticky_food", "smoking",
    ]
    for c in yn_cols:
        if c in out.columns:
            out[c] = out[c].map(_yes_no_unknown)

    for c in out.select_dtypes(include="object").columns:
        if c not in yn_cols:
            out[c] = out[c].map(_clean_text)
    if "gender" in out.columns:
        out["gender"] = out["gender"].map(lambda x: _clean_text(x).title())
    return out


def target_audit(df: pd.DataFrame, tolerance: float = 1e-8) -> pd.DataFrame:
    """Return row-level QC for the documented Elham arithmetic.

    A row is validation-eligible only when the stored target and every required
    direct component are present and the stored target equals their sum.
    """
    data = canonicalize(df)
    required = [c for c in ELHAM_DIRECT_COMPONENTS if c in data.columns]
    if TARGET not in data.columns:
        raise ValueError(f"Missing target column: {TARGET}")
    if len(required) != len(ELHAM_DIRECT_COMPONENTS):
        missing = sorted(set(ELHAM_DIRECT_COMPONENTS) - set(required))
        raise ValueError(f"Cannot audit Elham Index because components are missing: {missing}")

    complete_components = data[required].notna().all(axis=1)
    component_sum = data[required].sum(axis=1, min_count=len(required))
    difference = data[TARGET] - component_sum
    consistent = data[TARGET].notna() & complete_components & difference.abs().le(tolerance)
    return pd.DataFrame({
        "id": data[ID_COL] if ID_COL in data.columns else np.arange(len(data)),
        "stored_target": data[TARGET],
        "component_sum": component_sum,
        "difference": difference,
        "complete_components": complete_components,
        "target_consistent": consistent,
    }, index=data.index)


def analysis_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = canonicalize(df)
    audit = target_audit(data)
    eligible = audit["target_consistent"].fillna(False)
    return data.loc[eligible].copy().reset_index(drop=True), audit.reset_index(drop=True)


def predictor_columns(df: pd.DataFrame) -> List[str]:
    ordered = DEMOGRAPHIC_COLS + SES_COLS + BEHAVIOR_COLS + SALIVARY_COLS
    forbidden = set(TARGET_COMPONENTS + [TARGET, ID_COL] + EXCLUDED_CONTEXT_COLS)
    return [c for c in ordered if c in df.columns and c not in forbidden]


def validate_no_leakage(cols: List[str]) -> None:
    forbidden = set(TARGET_COMPONENTS + [TARGET, ID_COL])
    overlap = sorted(forbidden.intersection(cols))
    if overlap:
        raise ValueError(f"Target leakage detected: {overlap}")


def make_ohe():
    # Rare categories are grouped when supported, reducing sparse one-off levels.
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    transformers = []
    if num_cols:
        transformers.append(("num", SimpleImputer(strategy="median"), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", make_ohe()),
        ]), cat_cols))
    return ColumnTransformer(transformers, remainder="drop"), num_cols, cat_cols


def build_rf(X: pd.DataFrame) -> Pipeline:
    pre, _, _ = build_preprocessor(X)
    model = RandomForestRegressor(
        n_estimators=450, random_state=RANDOM_STATE, n_jobs=-1,
        min_samples_leaf=2, max_features=1.0,
    )
    return Pipeline([("pre", pre), ("model", model)])


def build_xgb(X: pd.DataFrame) -> Pipeline:
    if not XGB_AVAILABLE:
        raise RuntimeError("xgboost is not installed")
    pre, _, _ = build_preprocessor(X)
    model = XGBRegressor(
        n_estimators=600, learning_rate=0.05, max_depth=5,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        objective="reg:squarederror", tree_method="hist",
        random_state=RANDOM_STATE, n_jobs=-1,
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
    audit: pd.DataFrame
    n_source: int
    n_eligible: int


def fit_validate(df: pd.DataFrame) -> ValidationResult:
    data, audit = analysis_data(df)
    if len(data) < N_SPLITS * 5:
        raise ValueError(f"Too few QC-eligible rows for {N_SPLITS}-fold validation: n={len(data)}")
    cols = predictor_columns(data)
    validate_no_leakage(cols)
    if not cols:
        raise ValueError("No independent predictor columns were found")

    X = data[cols].copy()
    y = data[TARGET].astype(float).to_numpy()
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    pred_rf = np.full(len(data), np.nan)
    pred_xgb = np.full(len(data), np.nan) if XGB_AVAILABLE else None

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
    # Baselines are included so a high-capacity model is not judged in isolation.
    metrics["Mean baseline"] = metric_dict(y, np.repeat(np.mean(y), len(y)))
    metrics["Median baseline"] = metric_dict(y, np.repeat(np.median(y), len(y)))

    oof = pd.DataFrame({
        "id": data[ID_COL].to_numpy() if ID_COL in data.columns else np.arange(len(data)),
        "observed": y,
        "rf": pred_rf,
    })
    if XGB_AVAILABLE:
        pred_blend = 0.5 * (pred_rf + pred_xgb)
        metrics["XGBoost"] = metric_dict(y, pred_xgb)
        metrics["Blend"] = metric_dict(y, pred_blend)
        oof["xgb"] = pred_xgb
        oof["blend"] = pred_blend

    q34, q67 = np.quantile(y, [0.34, 0.67])

    rf_final = build_rf(X)
    rf_final.fit(X, y)
    xgb_final = None
    if XGB_AVAILABLE:
        xgb_final = build_xgb(X)
        xgb_final.fit(X, y)

    return ValidationResult(
        metrics=metrics, oof=oof, predictors=cols, rf_final=rf_final,
        xgb_final=xgb_final, risk_bins=(float(q34), float(q67)), audit=audit,
        n_source=int(len(df)), n_eligible=int(len(data)),
    )


def fairness_table(df: pd.DataFrame, oof: pd.DataFrame, pred_col="rf",
                   min_n=20, multiplier=1.5) -> pd.DataFrame:
    """Screen OOF subgroup MAE; flags are signals for review, not proof of bias."""
    data, _ = analysis_data(df)
    pred_col = pred_col if pred_col in oof.columns else "rf"
    merged = data.merge(oof[["id", "observed", pred_col]], on="id", how="inner")
    overall = mean_absolute_error(merged["observed"], merged[pred_col])
    rows = []
    for col in [c for c in FAIRNESS_COLS if c in merged.columns and c != "school"]:
        for level, sub in merged.groupby(col, dropna=False):
            if len(sub) < min_n:
                continue
            mae = mean_absolute_error(sub["observed"], sub[pred_col])
            rows.append({
                "variable": col, "group": str(level), "n": int(len(sub)),
                "MAE": float(mae), "overall_MAE": float(overall),
                "ratio": float(mae / overall) if overall > 0 else np.nan,
                "flag_for_review": bool(mae >= multiplier * overall),
            })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["flag_for_review", "ratio"], ascending=[False, False]).reset_index(drop=True)
