"""Component-specific modeling for Dental AI Coach.

Scientific purpose
------------------
Elham's Index is retained as a detailed clinical oral-health profile rather than
collapsed into a single machine-learning target. Each sufficiently prevalent
clinical component is modeled separately from independently collected
demographic, socioeconomic, behavioral, dietary and salivary predictors.

These models describe cross-sectional predictive associations. They do not
establish causality and they do not forecast future disease without longitudinal
follow-up data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from analysis_pipeline import (
    RANDOM_STATE, N_SPLITS, XGB_AVAILABLE, analysis_data, predictor_columns,
    validate_no_leakage, build_rf, build_xgb,
)

# Components sufficiently represented in the audited raw clinical cohort to
# justify component-specific exploratory/internal modeling.
# IMPORTANT: the missing-tooth field includes wisdom teeth. In this adolescent
# cohort it must not automatically be interpreted as disease-related tooth loss.
MODELED_COMPONENTS = {
    "missing_0_including_wisdom_": "Teeth recorded as missing (including wisdom teeth)",
    "decayed_1": "Decayed teeth",
    "filled_2": "Filled teeth",
    "hypocalcification_4": "Hypocalcified teeth",
}

# Components retained in the detailed clinical profile but too sparse for a
# separate predictive model in the current cohort.
DESCRIPTIVE_COMPONENTS = {
    "hypoplasia_3": "Hypoplasia",
    "fluorosis_5": "Fluorosis",
    "erosion_6": "Erosion",
    "abrasion_7": "Abrasion",
    "attrition_8": "Attrition",
    "abfraction_9": "Abfraction",
    "sealant_a": "Sealants",
    "fractured_h": "Fractured teeth",
    "crown_pontic": "Crown pontics",
    "crown_abutment": "Crown abutments",
    "crown_implant": "Implant crowns",
    "veneer_f": "Veneers",
}


def metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
    }


@dataclass
class ComponentResult:
    target: str
    label: str
    predictors: List[str]
    metrics: Dict[str, Dict[str, float]]
    oof: pd.DataFrame
    rf_final: object
    xgb_final: object | None
    n: int
    prevalence: float
    mean_count: float
    median_count: float
    max_count: float


def fit_component(df: pd.DataFrame, target: str) -> ComponentResult:
    if target not in MODELED_COMPONENTS:
        raise ValueError(f"Component is not configured for modeling: {target}")

    data, _ = analysis_data(df)
    if target not in data.columns:
        raise ValueError(f"Missing component column: {target}")

    data[target] = pd.to_numeric(data[target], errors="coerce")
    data = data.loc[data[target].notna()].reset_index(drop=True)
    if len(data) < N_SPLITS * 5:
        raise ValueError(f"Too few records for {N_SPLITS}-fold validation: n={len(data)}")

    cols = predictor_columns(data)
    validate_no_leakage(cols)
    X = data[cols].copy()
    y = data[target].astype(float).to_numpy()

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    pred_rf = np.full(len(data), np.nan)
    pred_xgb = np.full(len(data), np.nan) if XGB_AVAILABLE else None

    for train_idx, test_idx in kf.split(X):
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        ytr = y[train_idx]

        rf = build_rf(Xtr)
        rf.fit(Xtr, ytr)
        pred_rf[test_idx] = np.maximum(0.0, rf.predict(Xte))

        if XGB_AVAILABLE:
            xgb = build_xgb(Xtr)
            xgb.fit(Xtr, ytr)
            pred_xgb[test_idx] = np.maximum(0.0, xgb.predict(Xte))

    mean_pred = np.repeat(np.mean(y), len(y))
    median_pred = np.repeat(np.median(y), len(y))
    score = {
        "Mean baseline": metrics(y, mean_pred),
        "Median baseline": metrics(y, median_pred),
        "Random Forest": metrics(y, pred_rf),
    }

    oof = pd.DataFrame({
        "id": data["id"].to_numpy() if "id" in data.columns else np.arange(len(data)),
        "observed": y,
        "rf": pred_rf,
    })

    if XGB_AVAILABLE:
        pred_blend = 0.5 * (pred_rf + pred_xgb)
        score["XGBoost"] = metrics(y, pred_xgb)
        score["Blend"] = metrics(y, pred_blend)
        oof["xgb"] = pred_xgb
        oof["blend"] = pred_blend

    rf_final = build_rf(X)
    rf_final.fit(X, y)
    xgb_final = None
    if XGB_AVAILABLE:
        xgb_final = build_xgb(X)
        xgb_final.fit(X, y)

    return ComponentResult(
        target=target,
        label=MODELED_COMPONENTS[target],
        predictors=cols,
        metrics=score,
        oof=oof,
        rf_final=rf_final,
        xgb_final=xgb_final,
        n=len(data),
        prevalence=float((y > 0).mean()),
        mean_count=float(np.mean(y)),
        median_count=float(np.median(y)),
        max_count=float(np.max(y)),
    )


def fit_all_components(df: pd.DataFrame) -> Dict[str, ComponentResult]:
    return {target: fit_component(df, target) for target in MODELED_COMPONENTS}


def clinical_profile(row: pd.Series | dict) -> pd.DataFrame:
    """Return the detailed Elham component profile for one participant."""
    d = dict(row)
    labels = {**MODELED_COMPONENTS, **DESCRIPTIVE_COMPONENTS}
    records = []
    for col, label in labels.items():
        value = pd.to_numeric(d.get(col, 0), errors="coerce")
        value = 0.0 if pd.isna(value) else float(value)
        records.append({
            "component": label,
            "count": value,
            "modeled": col in MODELED_COMPONENTS,
        })
    return pd.DataFrame(records)
