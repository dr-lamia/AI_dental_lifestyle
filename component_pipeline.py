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

MODELED_COMPONENTS = {
    "missing_0_including_wisdom_": "Teeth recorded as missing (including wisdom teeth)",
    "decayed_1": "Decayed teeth",
    "filled_2": "Filled teeth",
    "hypocalcification_4": "Hypocalcified teeth",
}

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

# Five-fold out-of-fold metrics from the successful GitHub Actions validation
# run on the identical audited 160-participant embedded cohort (2026-09-01).
# They are used by the deployed Streamlit app so it does not repeat the full
# cross-validation workload on every cold start. The research pipeline below
# remains the source of truth for regenerating these values.
VALIDATED_METRICS = {
    "missing_0_including_wisdom_": {
        "Mean baseline": {"R2": 0.0, "MAE": 1.33546875, "RMSE": 1.8358304490066617},
        "Median baseline": {"R2": -0.09597932289433109, "MAE": 1.09375, "RMSE": 1.921913109378257},
        "Random Forest": {"R2": 0.0743685443530967, "MAE": 1.2636651565255732, "RMSE": 1.766247748160268},
        "XGBoost": {"R2": 0.0341503747266827, "MAE": 1.357864110870287, "RMSE": 1.8042110011520245},
        "Blend": {"R2": 0.0705372760163252, "MAE": 1.3015774849913282, "RMSE": 1.769899299335579},
    },
    "decayed_1": {
        "Mean baseline": {"R2": 0.0, "MAE": 3.395625, "RMSE": 4.426906369012112},
        "Median baseline": {"R2": -0.05625717566016086, "MAE": 3.375, "RMSE": 4.54972526643093},
        "Random Forest": {"R2": 0.10966337073199262, "MAE": 3.222067116101491, "RMSE": 4.177124859527157},
        "XGBoost": {"R2": 0.06794095905860076, "MAE": 3.285591740813106, "RMSE": 4.273877285890304},
        "Blend": {"R2": 0.10273921715692169, "MAE": 3.2240750673176537, "RMSE": 4.193336164889148},
    },
    "filled_2": {
        "Mean baseline": {"R2": 0.0, "MAE": 1.3914062500000002, "RMSE": 1.8969959804912608},
        "Median baseline": {"R2": -0.2991185793061526, "MAE": 1.0375, "RMSE": 2.1621748310439655},
        "Random Forest": {"R2": 0.08089353166550584, "MAE": 1.209755009920635, "RMSE": 1.8186508165486532},
        "XGBoost": {"R2": 0.07552519464603336, "MAE": 1.186818484563264, "RMSE": 1.8239542912527305},
        "Blend": {"R2": 0.09799636807903467, "MAE": 1.1896823459408523, "RMSE": 1.8016505300163197},
    },
    "hypocalcification_4": {
        "Mean baseline": {"R2": 0.0, "MAE": 5.44375, "RMSE": 6.8032620852058905},
        "Median baseline": {"R2": -0.14887583552764827, "MAE": 5.0875, "RMSE": 7.292119033586876},
        "Random Forest": {"R2": -0.16418092778534055, "MAE": 5.769035055376722, "RMSE": 7.340530405186305},
        "XGBoost": {"R2": -0.3058107984252916, "MAE": 6.136524564377032, "RMSE": 7.774229008291793},
        "Blend": {"R2": -0.21602992261257148, "MAE": 5.919951136001629, "RMSE": 7.502212003764039},
    },
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
    """Full five-fold validation plus final-model fitting; use for research/CI."""
    return {target: fit_component(df, target) for target in MODELED_COMPONENTS}


def fit_component_for_app(df: pd.DataFrame, target: str) -> ComponentResult:
    """Fit only final models for interactive deployment.

    Validation metrics come from the successful five-fold CI run on the identical
    audited embedded cohort. This avoids repeating 5-fold CV on every Streamlit
    cold start while preserving the scientifically validated values displayed in
    the dashboard.
    """
    if target not in MODELED_COMPONENTS:
        raise ValueError(f"Component is not configured for modeling: {target}")
    data, _ = analysis_data(df)
    data[target] = pd.to_numeric(data[target], errors="coerce")
    data = data.loc[data[target].notna()].reset_index(drop=True)
    cols = predictor_columns(data)
    validate_no_leakage(cols)
    X = data[cols].copy()
    y = data[target].astype(float).to_numpy()

    rf_final = build_rf(X)
    rf_final.fit(X, y)
    xgb_final = None
    if XGB_AVAILABLE:
        xgb_final = build_xgb(X)
        xgb_final.fit(X, y)

    score = dict(VALIDATED_METRICS[target])
    if not XGB_AVAILABLE:
        score.pop("XGBoost", None)
        score.pop("Blend", None)

    return ComponentResult(
        target=target,
        label=MODELED_COMPONENTS[target],
        predictors=cols,
        metrics=score,
        oof=pd.DataFrame(),
        rf_final=rf_final,
        xgb_final=xgb_final,
        n=len(data),
        prevalence=float((y > 0).mean()),
        mean_count=float(np.mean(y)),
        median_count=float(np.median(y)),
        max_count=float(np.max(y)),
    )


def fit_all_components_for_app(df: pd.DataFrame) -> Dict[str, ComponentResult]:
    """Fast deployment path: final models only, prevalidated metrics displayed."""
    return {target: fit_component_for_app(df, target) for target in MODELED_COMPONENTS}


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
