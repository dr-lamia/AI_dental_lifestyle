# streamlit_app_behaviors.py — Dental AI Coach (Behaviours + SES + Explainability)
# with RandomForest + XGBoost + Blend + Final Insights + PocketMoney(Yes/No/Unknown)

import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay

# Try XGBoost (optional)
try:
    from xgboost import XGBRegressor
    XGB_OK = True
except Exception:
    XGB_OK = False

# ============================ CONFIG =============================
st.set_page_config(page_title="AI tool for personalized awareness and education", page_icon="🦷", layout="wide")

DATA_PATH = "data/no_recommendation_dental_dataset_cleaned_keep_including_wisdom.csv"
TARGET_COL = "elham_s_index_including_wisdom"
ID_COLS = ["id"]

# Behaviour feature list (column names present in your dataset)
BEHAVIOR_COLS = [
    "tooth_brushing_frequency", "time_of_tooth_brushing", "interdental_cleaning", "mouth_rinse",
    "snacks_frequency", "snack_content", "sugar", "sticky_food", "carbonated_beverages",
    "type_of_diet", "hydration", "salivary_ph", "salivary_consistency", "buffering_capacity",
    "mutans_load_in_saliva", "lactobacilli_load_in_saliva"
]

# ======== ELHAM (subset) FIELDS YOU HAVE NOW ========
ELHAM_FIELDS_PRESENT = [
    "missing_0_including_wisdom_", "decayed_1", "filled_2",
    "hypoplasia_3", "hypocalcification_4", "fluorosis_5",
    "erosion_6", "abrasion_7", "attrition_8", "abfraction_9",
    "sealant_a", "fractured_h",
    "crown_pontic", "crown_abutment", "crown_implant",
    "veneer_f"
]


def compute_elham_from_inputs(values_dict: dict):
    per_item = {}
    total = 0.0
    for k in ELHAM_FIELDS_PRESENT:
        v = float(values_dict.get(k, 0) or 0)
        per_item[k] = v
        total += v
    return total, per_item


# ====================== BEHAVIOUR NORMALIZERS =====================
def _title(v): return str(v).strip().title()


def norm_yes_no(v: str) -> str:
    s = str(v).strip().lower()
    if s in {"yes", "y", "true", "1"}:
        return "Yes"
    if s in {"no", "n", "false", "0"}:
        return "No"
    if s in {"", "unknown", "unk", "na", "nan", "none"}:
        return "Unknown"
    return _title(v)


def norm_brushing_freq(v: str) -> str:
    s = str(v).strip().lower()
    if any(k in s for k in ["twice", "two", "2/day", "2 per", " 2 "]): return "2/day"
    if any(k in s for k in ["once", "one", "1/day", "1 per", " 1 "]): return "1/day"
    if "irreg" in s or "sometimes" in s: return "Irregular"
    return _title(v)


def norm_snack_freq(v: str) -> str:
    s = str(v).strip().lower()
    if s in {"never", "none", "no", "0", "0/day"}: return "0/day"
    if any(k in s for k in ["1-2", "1–2", "sometimes", "occasional", "few"]): return "1–2/day"
    if any(k in s for k in [">2", "3+", "many", "often", "frequent", "a lot"]): return "3+/day"
    return _title(v)


def norm_risk_freq(v: str) -> str:
    s = str(v).strip().lower()
    if s in {"none", "no", "never", "0"}: return "None"
    if any(k in s for k in ["occas", "some", "rare"]): return "Occasional"
    if any(k in s for k in ["freq", "often", "daily", "many"]): return "Frequent"
    return _title(v)


def norm_saliva_level(v: str) -> str:
    s = str(v).strip().lower()
    if "low" in s or "acid" in s: return "Low"
    if "high" in s: return "High"
    if "mod" in s: return "Moderate"
    return "Normal" if "normal" in s else _title(v)


def norm_mutans_lacto(v: str) -> str:
    s = str(v).strip().lower()
    if "more" in s or ">" in s or "10^5" in s or "10)5" in s: return "High"
    if "less" in s or "<" in s: return "Low"
    return "Normal" if "normal" in s else _title(v)


NORMALIZERS = {
    "tooth_brushing_frequency": norm_brushing_freq,
    "time_of_tooth_brushing": _title,
    "interdental_cleaning": norm_yes_no,
    "mouth_rinse": norm_yes_no,
    "snacks_frequency": norm_snack_freq,
    "snack_content": _title,
    "sugar": norm_risk_freq,
    "sticky_food": norm_yes_no,
    "carbonated_beverages": norm_risk_freq,
    "type_of_diet": _title,
    "hydration": _title,
    "salivary_ph": norm_saliva_level,
    "salivary_consistency": norm_saliva_level,
    "buffering_capacity": norm_saliva_level,
    "mutans_load_in_saliva": norm_mutans_lacto,
    "lactobacilli_load_in_saliva": norm_mutans_lacto,
}


def normalize_cats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c, f in NORMALIZERS.items():
        if c in df.columns:
            df[c] = df[c].astype(str).map(f)
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()
    return df


# ====================== SES HELPERS (inline) ======================
def _t(x) -> str:
    return str(x).strip().lower()


def _keep(v, default="Unknown"):
    s = str(v).strip()
    return s.title() if s else default


def _num(x):
    s = str(x).strip().lower()
    if not s or s in {"nan", "none", "unknown"}:
        return None
    if s.endswith("k"):
        try:
            return float(s[:-1]) * 1000.0
        except Exception:
            pass
    s = re.sub(r"[^\d.,-]", "", s).replace(",", "")
    try:
        return float(s)
    except Exception:
        return None


SES_FUZZY = {
    "school": ["school"],
    "grade": ["grade"],
    "house_ownership": ["house_owned_or_rent", "house_owi", "house_ow", "ownership", "own", "rent"],
    "i_live_with": ["i_live_with_my_parents", "i_live_with", "live_with"],
    "average_income": ["average_in", "avg_income", "family_in", "income", "household_income"],
    "pocket_money": ["pocket_m", "pocket_money", "allowance"],
    "father_s_education": ["father_s_e", "father edu", "father_edu"],
    "mother_s_education": ["mother_s_education", "mother_s_", "mother edu", "mother_edu"],
    "father_s_job": ["father_s_jc", "father_s_j", "father job", "father occ"],
    "mother_s_job": ["mother_s_j", "mother_s_job", "mother_s_", "mother occ", "mother job"],
    "insurance": ["insurance", "health_insurance", "dental_insurance"],
    "access_to": ["access_to_oral_health_care", "access_to", "access"],
    "frequency": ["frequency_of_visits", "frequency,", "frequency", "visit"],
    "affordability": ["affordability", "afford", "affordabilit"],
}


def find_cols(df, fuzzy=SES_FUZZY):
    out = {}
    lower = {c: c.lower() for c in df.columns}
    for canon, keys in fuzzy.items():
        hit = None
        for c in df.columns:
            if lower[c] in keys:
                hit = c
                break
        if hit is None:
            for c in df.columns:
                if any(k in lower[c] for k in keys):
                    hit = c
                    break
        out[canon] = hit
    return out


def norm_school(v):
    s = _t(v)
    if any(k in s for k in ["government", "gov", "public", "experimental"]): return "Public"
    if "language" in s and any(k in s for k in ["experimental", "gov", "public"]): return "Public"
    if any(k in s for k in ["international", "american", "british", "german", "french", "canadian", "igcse", "ib"]): return "International"
    if any(k in s for k in ["private", "national", "language"]): return "Private"
    return _keep(v)


def norm_grade(v):
    s = _t(v)
    if any(k in s for k in ["kg", "nursery", "primary", "grade 1", "grade 2", "grade 3", "grade 4", "grade 5", "grade 6"]): return "Primary"
    if any(k in s for k in ["prep", "preparatory", "grade 7", "grade 8", "grade 9"]): return "Preparatory"
    if any(k in s for k in ["sec", "secondary", "grade 10", "grade 11", "grade 12"]): return "Secondary"
    return _keep(v)


def norm_house_ownership(v):
    s = _t(v)
    if "own" in s: return "Owned"
    if "rent" in s: return "Rented"
    return _keep(v)


def norm_live_with(v):
    s = _t(v)
    if any(k in s for k in ["father and mother", "both parents", "two parents", "with my parents"]): return "Two Parents"
    if any(k in s for k in ["single", "mother only", "father only", "one parent"]): return "Single Parent"
    if any(k in s for k in ["relative", "grand", "aunt", "uncle", "guardian", "care"]): return "Relatives/Other"
    return _keep(v)


def norm_parent_edu(v):
    s = _t(v)
    if any(k in s for k in ["phd", "master", "msc", "postgrad", "doctor"]): return "Postgrad"
    if any(k in s for k in ["uni", "college", "bsc", "ba", "license", "licence"]): return "University"
    if any(k in s for k in ["secondary", "high school", "prep"]): return "Secondary"
    if "primary" in s or "elementary" in s: return "Primary"
    return _keep(v)


def norm_job(v):
    s = _t(v)
    if any(k in s for k in ["not working", "no job", "housewife", "unemployed", "homemaker"]): return "Not Working"
    if any(k in s for k in ["manager", "engineer", "doctor", "dentist", "pharmacist", "teacher", "accountant", "lawyer", "architect", "nurse"]): return "Professional/Manager"
    if s and s != "unknown": return _keep(v)
    return "Unknown"


def norm_insurance(v):
    s = _t(v)
    if s in {"yes", "insured", "y", "1", "covered", "have"}: return "Insured"
    if s in {"no", "uninsured", "n", "0", "don’t have", "dont have", "do not have"}: return "Uninsured"
    return _keep(v)


def norm_access(v):
    s = _t(v)
    if any(k in s for k in ["easy", "available", "near", "good"]): return "Easy"
    if any(k in s for k in ["hard", "difficult", "far", "limited", "poor"]): return "Difficult"
    if s in {"moderate", "average", "ok"}: return "Moderate"
    return _keep(v)


def norm_afford(v):
    s = _t(v)
    if any(k in s for k in ["cannot", "can't", "no", "unaffordable"]): return "No"
    if any(k in s for k in ["hard", "difficult", "sometimes", "partial", "struggle"]): return "Hard"
    if any(k in s for k in ["yes", "afford", "can", "affordable"]): return "Yes"
    return _keep(v)


def norm_visit_freq(v):
    s = _t(v)
    if any(k in s for k in ["6", "12", "regular", "check", "year", "every"]): return "Regular"
    if any(k in s for k in ["pain", "emergency", "only when"]): return "Pain-Only"
    if "never" in s: return "Never"
    return _keep(v)


def norm_pocket_money(v):
    s = str(v).strip().lower()
    if s in {"yes", "y", "1", "true"}: return "Yes"
    if s in {"no", "n", "0", "false"}: return "No"
    return "Unknown"


SES_NORMALIZERS = {
    "school": norm_school,
    "grade": norm_grade,
    "house_ownership": norm_house_ownership,
    "i_live_with": norm_live_with,
    "father_s_education": norm_parent_edu,
    "mother_s_education": norm_parent_edu,
    "father_s_job": norm_job,
    "mother_s_job": norm_job,
    "insurance": norm_insurance,
    "access_to": norm_access,
    "affordability": norm_afford,
    "frequency": norm_visit_freq,
    "pocket_money": norm_pocket_money,  # categorical Yes/No/Unknown
}


def prepare_ses(df: pd.DataFrame, train_idx=None):
    df2 = df.copy()
    ses_map = find_cols(df2)

    # normalize text SES (incl. pocket_money categorical)
    for key, col in ses_map.items():
        if col is not None and key in SES_NORMALIZERS and col in df2.columns:
            df2[col] = df2[col].apply(SES_NORMALIZERS[key])

    # band only for average_income (NOT pocket money)
    def _bands(col_key, new_name):
        col = ses_map.get(col_key)
        if col is None or col not in df2.columns:
            return None
        idx = train_idx if train_idx is not None else df2.index
        s = df2.loc[idx, col].apply(_num).dropna()
        if len(s) >= 10 and s.nunique() >= 3:
            q1, q2 = s.quantile([0.34, 0.67])
        elif len(s) >= 2:
            q1 = q2 = s.median()
        else:
            df2[new_name] = "Unknown"
            return None

        def lab(v):
            x = _num(v)
            if x is None: return "Unknown"
            if x < q1: return "Low"
            if x < q2: return "Medium"
            return "High"

        df2[new_name] = df2[col].apply(lab)
        return (float(q1), float(q2))

    inc_qs = _bands("average_income", "income_band")
    pok_qs = None  # pocket money kept categorical

    # collect SES categorical columns (include pocket_money)
    ses_cat_cols = []
    for key in [
        "school", "grade", "house_ownership", "i_live_with",
        "father_s_education", "mother_s_education",
        "father_s_job", "mother_s_job",
        "insurance", "access_to", "frequency", "affordability",
        "pocket_money"
    ]:
        col = ses_map.get(key)
        if col is not None and col in df2.columns:
            ses_cat_cols.append(col)
    if "income_band" in df2.columns:
        ses_cat_cols.append("income_band")

    # only average_income considered numeric (drop raw)
    raw_numeric_drop = []
    for key in ["average_income"]:
        col = ses_map.get(key)
        if col is not None and col in df2.columns:
            raw_numeric_drop.append(col)

    meta = {"ses_map": ses_map, "income_quantiles": inc_qs}
    return df2, ses_cat_cols, raw_numeric_drop, meta


# ========================= RISK TIERS =============================
def build_risk_bins(df, target_col=TARGET_COL):
    y = df[target_col].dropna().values
    q1, q2 = np.quantile(y, [0.34, 0.67])
    return (float(q1), float(q2))


def index_tier(y_hat, bins):
    low_u, mod_u = bins
    if y_hat < low_u: return "low"
    if y_hat < mod_u: return "moderate"
    return "high"


def tier_plan(tier):
    if tier == "high":
        return dict(
            recall="1–3 months",
            toothpaste="High-fluoride (2800–5000 ppm) twice daily",
            rinse="0.05% NaF daily (or 0.2% weekly) + consider CHX as indicated",
            varnish="Professional fluoride varnish every 3 months",
            diet_focus="Strong sugar reduction + avoid acidic drinks",
        )
    if tier == "moderate":
        return dict(
            recall="3–6 months",
            toothpaste="1450 ppm fluoride twice daily",
            rinse="0.05% NaF daily",
            varnish="Fluoride varnish every 6 months",
            diet_focus="Reduce between-meal snacks, especially sticky sweets",
        )
    return dict(
        recall="6–12 months",
        toothpaste="1450 ppm fluoride twice daily",
        rinse="Optional fluoride rinse if enamel defects or ortho",
        varnish="Varnish at routine intervals if indicated",
        diet_focus="Maintain current habits; keep sweets with meals",
    )


# ======== RULE-BASED TREATMENT PLAN FROM ELHAM FINDINGS ==========
def treatment_plan_from_elham(counts: dict, tier: str):
    n = lambda k: int(counts.get(k, 0) or 0)
    out = []
    plan = tier_plan(tier)
    out += [
        f"**Overall plan for {tier.title()} risk:**",
        f"- Recall: **{plan['recall']}**",
        f"- Toothpaste: **{plan['toothpaste']}**",
        f"- Mouthrinse: **{plan['rinse']}**",
        f"- Varnish: **{plan['varnish']}**",
        f"- Diet focus: **{plan['diet_focus']}**",
        "—"
    ]
    if n("decayed_1") > 0:
        out += [
            f"• Caries on **{n('decayed_1')}** tooth/teeth → **restore** (GIC/composite); pulpal diagnosis if deep.",
            "  - Add fluoride varnish and sugar frequency counseling."
        ]
    if n("filled_2") > 0:
        out += [f"• **{n('filled_2')}** restoration(s) → check margins; repair/polish if needed."]
    if n("hypoplasia_3") > 0 or n("hypocalcification_4") > 0:
        out += [
            f"• Enamel defects (hypoplasia: {n('hypoplasia_3')}, hypocalcification: {n('hypocalcification_4')}) →",
            "  - **Sealants / resin infiltration**, fluoride varnish for sensitivity."
        ]
    if n("fluorosis_5") > 0:
        out += [f"• **Fluorosis** ({n('fluorosis_5')}) → microabrasion ± external bleaching."]
    if n("erosion_6") > 0:
        out += [f"• **Erosion** ({n('erosion_6')}) → acid control, straw use, rinse water; high-fluoride paste; restore if dentin exposed."]
    if n("abrasion_7") > 0:
        out += [f"• **Abrasion** ({n('abrasion_7')}) → brushing technique coaching; soft brush; desensitizing paste; restore if needed."]
    if n("attrition_8") > 0:
        out += [f"• **Attrition** ({n('attrition_8')}) → assess parafunction; **night guard** consideration; restore worn facets if indicated."]
    if n("abfraction_9") > 0:
        out += [f"• **Abfraction** ({n('abfraction_9')}) → manage occlusal load; restore if sensitive or deep."]
    if n("fractured_h") > 0:
        out += [f"• **Fracture** ({n('fractured_h')}) → immediate protection; definitive onlay/crown after assessment."]
    if n("sealant_a") > 0:
        out += [f"• **Sealants** present ({n('sealant_a')}) → check retention; re-seal where partial loss."]
    if n("missing_0_including_wisdom_") > 0:
        out += [f"• **Missing teeth** (incl. wisdom): {n('missing_0_including_wisdom_')} → discuss replacement needs."]
    if n("crown_pontic") > 0 or n("crown_abutment") > 0:
        out += [f"• **Bridge** (pontic {n('crown_pontic')}, abutment {n('crown_abutment')}) → super-floss/interdental brushes; margin review."]
    if n("crown_implant") > 0:
        out += [f"• **Implant crowns** ({n('crown_implant')}) → implant maintenance; peri-implant screening."]
    if n("veneer_f") > 0:
        out += [f"• **Veneers** ({n('veneer_f')}) → hygiene at margins; composite repair if chipping."]
    out += ["—", "• Reinforce personalized diet & hygiene education (see behaviour section).", f"• Recall per tier: **{plan['recall']}**."]
    return out


# ==================== PREPROCESS / TRAINING =======================
def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"Dataset not found at '{path}'. Put the CSV in a 'data' folder with this exact name.")
        st.stop()
    df = pd.read_csv(path)
    return normalize_cats(df)


def split_feature_types(df: pd.DataFrame):
    cat_cols = [c for c in BEHAVIOR_COLS if c in df.columns]
    num_cols = df.select_dtypes(exclude="object").columns.tolist()
    num_cols = [c for c in num_cols if c not in ID_COLS + [TARGET_COL]]
    if TARGET_COL not in df.columns:
        st.error(f"Target column '{TARGET_COL}' not found.")
        st.stop()
    return num_cols, cat_cols


def build_rf_pipeline(num_cols, cat_cols):
    pre = ColumnTransformer(
        [("num", SimpleImputer(strategy="median"), num_cols)] +
        ([("cat", make_ohe(), cat_cols)] if len(cat_cols) > 0 else []),
        remainder="drop", verbose_feature_names_out=True
    )
    reg = RandomForestRegressor(n_estimators=450, random_state=42, n_jobs=-1)
    return Pipeline([("pre", pre), ("reg", reg)])


def build_xgb_pipeline(num_cols, cat_cols):
    pre = ColumnTransformer(
        [("num", SimpleImputer(strategy="median"), num_cols)] +
        ([("cat", make_ohe(), cat_cols)] if len(cat_cols) > 0 else []),
        remainder="drop", verbose_feature_names_out=True
    )
    reg = XGBRegressor(
        n_estimators=600, learning_rate=0.05, max_depth=5,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        random_state=42, n_jobs=-1, tree_method="hist"
    )
    return Pipeline([("pre", pre), ("reg", reg)])


@st.cache_resource(show_spinner=False)
def train_models(df: pd.DataFrame, cat_cols_override=None, drop_num_cols=None):
    num_cols, beh_cat_cols = split_feature_types(df)
    if drop_num_cols:
        num_cols = [c for c in num_cols if c not in set(drop_num_cols)]
    cat_cols = (cat_cols_override[:] if cat_cols_override else beh_cat_cols[:])

    X = df[num_cols + cat_cols].copy()
    y = df[TARGET_COL].astype(float).values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_pipe = build_rf_pipeline(num_cols, cat_cols)
    rf_pipe.fit(X_tr, y_tr)
    y_rf = rf_pipe.predict(X_te)
    metrics_rf = {"R2": float(r2_score(y_te, y_rf)), "MAE": float(mean_absolute_error(y_te, y_rf))}

    if XGB_OK:
        xgb_pipe = build_xgb_pipeline(num_cols, cat_cols)
        xgb_pipe.fit(X_tr, y_tr)
        y_xgb = xgb_pipe.predict(X_te)
        metrics_xgb = {"R2": float(r2_score(y_te, y_xgb)), "MAE": float(mean_absolute_error(y_te, y_xgb))}
        y_blend = 0.5 * (y_rf + y_xgb)
        metrics_blend = {"R2": float(r2_score(y_te, y_blend)), "MAE": float(mean_absolute_error(y_te, y_blend))}
    else:
        xgb_pipe = None
        metrics_xgb = None
        metrics_blend = None

    feat_names = rf_pipe.named_steps["pre"].get_feature_names_out().tolist()
    num_medians = X[num_cols].median(numeric_only=True)
    cat_modes = {c: (X[c].mode(dropna=True).iloc[0] if X[c].notna().any() else "Unknown") for c in cat_cols}
    cat_values = {c: sorted(X[c].dropna().astype(str).unique().tolist(), key=lambda s: (s == "Unknown", s)) for c in cat_cols}
    risk_bins = build_risk_bins(df, TARGET_COL)

    metrics_all = {"RandomForest": metrics_rf}
    if XGB_OK:
        metrics_all["XGBoost"] = metrics_xgb
        metrics_all["Blend (avg RF+XGB)"] = metrics_blend

    return rf_pipe, xgb_pipe, metrics_all, num_cols, cat_cols, feat_names, num_medians, cat_modes, cat_values, risk_bins


# ======== SHAP GROUPING + SMALL PLOTTING HELPERS ==================
def build_group_map(feature_names, num_cols, cat_cols):
    group_map = {}
    for i, name in enumerate(feature_names):
        if name.startswith("num__"):
            orig = name[len("num__"):]
        elif name.startswith("cat__"):
            orig = None
            for c in cat_cols:
                prefix = f"cat__{c}_"
                if name.startswith(prefix):
                    orig = c
                    break
            if orig is None:
                orig = name
        else:
            orig = name
        group_map.setdefault(orig, []).append(i)
    return group_map


def group_shap_by_original(feature_names, shap_vals, num_cols, cat_cols):
    group_map = build_group_map(feature_names, num_cols, cat_cols)
    arr = shap_vals[0] if isinstance(shap_vals, list) else shap_vals
    row = arr[0] if getattr(arr, "ndim", 1) == 2 else arr
    grouped = {orig: float(np.sum(row[idxs])) for orig, idxs in group_map.items()}
    return sorted(grouped.items(), key=lambda kv: abs(kv[1]), reverse=True)


def plot_bar(items, title):
    if not items:
        st.info("Nothing to show.")
        return
    fig, ax = plt.subplots()
    ax.bar([k for k, _ in items], [v for _, v in items])
    ax.set_xticklabels([k for k, _ in items], rotation=45, ha="right")
    ax.set_ylabel("Grouped value")
    ax.set_title(title)
    fig.tight_layout()
    st.pyplot(fig)


# =================== DETAILED ADVICE (ALL BEHAVIOURS) =============
def tier_plan_text(tier):
    plan = tier_plan(tier)
    return [
        f"**Overall plan for {tier.title()} risk:**",
        f"- Recall: **{plan['recall']}**",
        f"- Toothpaste: **{plan['toothpaste']}**",
        f"- Mouthrinse: **{plan['rinse']}**",
        f"- Varnish: **{plan['varnish']}**",
        f"- Diet focus: **{plan['diet_focus']}**",
    ]


def lines_for_behavior(name, val, tier):
    v = (str(val) if val is not None else "Unknown").strip().lower()
    plan = tier_plan(tier)
    out = []
    if name == "tooth_brushing_frequency":
        if v in {"1/day", "once", "1"}:
            out += ["Brush **twice daily** with fluoride toothpaste.", plan["toothpaste"],
                    "Use a **pea-sized** amount; spit, don’t rinse after brushing.",
                    "Consider an **electric brush** if plaque control is sub-optimal."]
        elif v in {"irregular", "sometimes"}:
            out += ["Set **fixed times** for brushing (morning & before bed).", plan["toothpaste"]]
        else:
            out += [plan["toothpaste"]]
    if name == "interdental_cleaning":
        out += ["Start **daily interdental cleaning** (floss/interproximal brushes)." if v in {"no", "unknown"}
                else "Keep **daily interdental cleaning**; focus on back teeth."]
    if name == "mouth_rinse":
        out += [f"Add a **fluoride mouthrinse**: {plan['rinse']}." if v in {"no", "unknown"}
                else f"Continue **fluoride mouthrinse**: {plan['rinse']}."]
    if name == "snacks_frequency":
        if any(k in v for k in ["3+", ">2", "many", "often", "frequent"]):
            out += ["**Cut snacks to ≤1–2/day**; keep sweets **with meals**.", plan["diet_focus"]]
        elif any(k in v for k in ["1–2", "1-2", "sometimes", "occasional", "few"]):
            out += ["Keep snacks **≤1–2/day**; choose tooth-friendly options."]
        else:
            out += ["Great job limiting snacks; keep sweets **with meals** only."]
    if name == "snack_content":
        if any(k in v for k in ["sweet", "candy", "dessert", "cake", "chocolate"]):
            out += ["Swap sweets for **cheese, nuts, raw veg, yogurt**."]
        if "sticky" in v:
            out += ["Avoid **sticky, retentive sweets** (caramels, toffees)."]
    if name == "sugar":
        if any(k in v for k in ["frequent", "daily", "often"]):
            out += ["**Reduce added sugars**; replace sugary drinks with **water or milk**."]
        elif any(k in v for k in ["occasional", "some"]):
            out += ["Keep sugars to **meals only** and limit portion size."]
        else:
            out += ["Maintain **low sugar** intake."]
    if name == "carbonated_beverages":
        if any(k in v for k in ["frequent", "daily", "often"]):
            out += ["Limit **carbonated/acidic drinks**; use a **straw**, avoid **sipping over time**, and **rinse with water** afterwards."]
        elif any(k in v for k in ["occasional", "some"]):
            out += ["Keep acidic drinks **occasional**; prefer water."]
        else:
            out += ["Great—**avoid acidic drinks** as a habit."]
    if name == "type_of_diet":
        out += ["Shift toward **balanced meals** rich in **proteins, dairy, vegetables**; cut ultra-processed foods."
                if any(k in v for k in ["junk", "fast"]) else
                "Keep a **balanced diet**; pair carbs with protein/dairy."]
    if name == "hydration":
        out += ["Increase **water intake** throughout the day (carry a bottle)." if "low" in v else "Maintain **good hydration**."]
    if name == "sticky_food":
        if v in {"yes", "y"}:
            out += ["Avoid **sticky foods**; if eaten, **rinse with water** and avoid bedtime intake."]
    if name == "salivary_ph":
        if "low" in v:
            out += ["**Low pH**: sugar-free gum (xylitol), avoid acids between meals, increase hydration.", plan["rinse"]]
        else:
            out += ["Maintain **neutral pH** habits: water instead of acidic drinks."]
    if name == "salivary_consistency":
        if any(k in v for k in ["high", "thick", "visc"]):
            out += ["**Thick saliva**: hydrate, review meds causing dryness, consider sugar-free lozenges/gum."]
        elif "low" in v:
            out += ["Ensure **adequate hydration**; monitor for dryness symptoms."]
    if name == "buffering_capacity":
        out += ["**Low buffering**: minimize acids; consider **varnish** and sugar-free gum after meals."
                if "low" in v else "Maintain habits that support saliva buffering (water, balanced meals)."]
    if name == "mutans_load_in_saliva":
        out += ["**High mutans**: tighten hygiene + fluoride, reduce sugars, consider **CHX** if indicated."
                if "high" in v else "Maintain good control of **mutans streptococci**." if "low" in v else ""]
    if name == "lactobacilli_load_in_saliva":
        out += ["**High lactobacilli**: focus on **fermentable carbs** reduction and **retentive snacks**."
                if "high" in v else "Maintain low **lactobacilli** through diet control." if "low" in v else ""]
    return [x for x in out if x]


def detailed_behavior_recommendations(all_behaviors: dict, tier: str):
    recs, seen = tier_plan_text(tier), set()
    for name, val in all_behaviors.items():
        for line in lines_for_behavior(name, val, tier):
            if line and line not in seen:
                recs.append(f"- {line}")
                seen.add(line)
    return recs


# =============== DATASET-DRIVEN UI OPTIONS ========================
def _mode_or_unknown(series: pd.Series) -> str:
    m = series.dropna().astype(str)
    return m.mode().iloc[0] if not m.empty and not m.mode().empty else "Unknown"


PREFERRED_ORDER = {
    "tooth_brushing_frequency": ["Irregular", "1/day", "2/day"],
    "interdental_cleaning": ["No", "Yes"],
    "mouth_rinse": ["No", "Yes"],
    "snacks_frequency": ["0/day", "1–2/day", "3+/day"],
    "sugar": ["None", "Occasional", "Frequent"],
    "carbonated_beverages": ["None", "Occasional", "Frequent"],
    "sticky_food": ["No", "Yes"],
    "hydration": ["Low", "Normal", "High"],
    "salivary_ph": ["Low", "Normal", "High"],
    "salivary_consistency": ["Low", "Normal", "High"],
    "buffering_capacity": ["Low", "Moderate", "Normal", "High"],
    "mutans_load_in_saliva": ["Low", "Normal", "High"],
    "lactobacilli_load_in_saliva": ["Low", "Normal", "High"],
    "pocket_money": ["Yes", "No", "Unknown"],  # ordered categorical choices
}


def build_options_from_df(df: pd.DataFrame, cols):
    out = {}
    for c in cols:
        if c in df.columns:
            vals = df[c].dropna().astype(str).unique().tolist()
            vals = sorted(vals, key=lambda s: (s == "Unknown", s))
            pref = PREFERRED_ORDER.get(c)
            if pref:
                seen = set()
                ordered = [v for v in pref if v in vals and not (v in seen or seen.add(v))]
                ordered += [v for v in vals if v not in set(pref)]
                vals = ordered
            out[c] = vals if vals else ["Unknown"]
        else:
            out[c] = ["Unknown"]
    return out


def default_from_df(df: pd.DataFrame, col: str) -> str:
    return _mode_or_unknown(df[col]) if col in df.columns else "Unknown"


# =============================== UI ===============================
st.title("🦷 AI tool for personalized awareness and education")

if not XGB_OK:
    st.caption("ℹ️ XGBoost not installed — using RandomForest only. To enable blend, `pip install xgboost`.")

# Dev helper
if st.button("🔁 Force clear cache (use after changing SES rules)"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.experimental_rerun()

df = load_data(DATA_PATH)

# SES prep (bands computed on TRAIN indices)
idx_train, idx_test = train_test_split(df.index, test_size=0.2, random_state=42)
df, ses_cat_cols, raw_numeric_ses, ses_meta = prepare_ses(df, train_idx=idx_train)

# UI lists and dataset-driven options
beh_cols = [c for c in BEHAVIOR_COLS if c in df.columns]
cat_cols_all = beh_cols + [c for c in ses_cat_cols if c not in beh_cols]
beh_options = build_options_from_df(df, beh_cols)
ses_cols_ui = [c for c in cat_cols_all if c not in beh_cols]
ses_options = build_options_from_df(df, ses_cols_ui)

# Train models (RF, XGB, Blend)
rf_pipe, xgb_pipe, metrics_all, num_cols, all_cat_cols, feat_names, num_medians, cat_modes, cat_values, risk_bins = \
    train_models(df, cat_cols_override=cat_cols_all, drop_num_cols=raw_numeric_ses)

# Model selector
model_choices = ["RandomForest"]
if XGB_OK:
    model_choices += ["XGBoost", "Blend (avg RF+XGB)"]
default_choice = "Blend (avg RF+XGB)" if XGB_OK else "RandomForest"
model_choice = st.selectbox("Model to use", options=model_choices, index=model_choices.index(default_choice))

# Show metrics
mcols = st.columns(len(model_choices))
for i, name in enumerate(model_choices):
    m = metrics_all[name]
    with mcols[i]:
        st.metric(name, f"R² {m['R2']:.3f}", delta=f"MAE {m['MAE']:.2f}")


def predict_with_choice(X):
    if model_choice == "RandomForest":
        return rf_pipe.predict(X)
    elif model_choice == "XGBoost" and XGB_OK:
        return xgb_pipe.predict(X)
    else:  # Blend
        if XGB_OK:
            return 0.5 * (rf_pipe.predict(X) + xgb_pipe.predict(X))
        return rf_pipe.predict(X)


def active_pipe_for_explain():
    # Use the chosen model; for Blend, pick the better single model (higher R²) for SHAP/PDP
    if model_choice == "RandomForest" or not XGB_OK:
        return rf_pipe, "RandomForest"
    if model_choice == "XGBoost":
        return xgb_pipe, "XGBoost"
    best = max(("RandomForest", "XGBoost"), key=lambda k: metrics_all[k]["R2"])
    return (rf_pipe if best == "RandomForest" else xgb_pipe), best


# ========================= MODEL VISUALIZATIONS ===================
st.subheader("Model visualizations")

X_all = df[num_cols + all_cat_cols].copy()
y_all = df[TARGET_COL].astype(float).values
X_tr_v, X_te_v, y_tr_v, y_te_v = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

VIS_SAMPLE_MAX = 400
PDP_GRID = 15
if len(X_te_v) > VIS_SAMPLE_MAX:
    X_te_s = X_te_v.sample(VIS_SAMPLE_MAX, random_state=42)
    idx = X_te_s.index
    y_te_s = y_te_v[idx]
else:
    X_te_s, y_te_s = X_te_v, y_te_v

y_pred_s = predict_with_choice(X_te_s)
resid_s = y_te_s - y_pred_s

tab_perf, tab_imp, tab_beh, tab_pdp = st.tabs(["📈 Performance", "⭐ Global importance", "🧠 Behaviour effects", "🧩 PDP / ICE"])

with tab_perf:
    with st.spinner("Drawing performance plots..."):
        fig1, ax1 = plt.subplots()
        ax1.scatter(y_te_s, y_pred_s, alpha=0.65)
        lo = float(min(y_te_s.min(), y_pred_s.min()))
        hi = float(max(y_te_s.max(), y_pred_s.max()))
        ax1.plot([lo, hi], [lo, hi])
        ax1.set_xlabel("Actual Elham Index"); ax1.set_ylabel("Predicted Elham Index")
        ax1.set_title("Predicted vs Actual (sampled hold-out)")
        fig1.tight_layout(); st.pyplot(fig1); plt.close(fig1)

        fig2, ax2 = plt.subplots()
        ax2.hist(resid_s, bins=30); ax2.set_xlabel("Residual (Actual − Predicted)")
        ax2.set_title("Residuals (sampled hold-out)")
        fig2.tight_layout(); st.pyplot(fig2); plt.close(fig2)

        fig3, ax3 = plt.subplots()
        ax3.scatter(y_pred_s, resid_s, alpha=0.6); ax3.axhline(0, linestyle="--")
        ax3.set_xlabel("Predicted"); ax3.set_ylabel("Residual")
        ax3.set_title("Residuals vs Predicted (sampled hold-out)")
        fig3.tight_layout(); st.pyplot(fig3); plt.close(fig3)

with tab_imp:
    try:
        if model_choice == "XGBoost" and XGB_OK:
            importances = xgb_pipe.named_steps["reg"].feature_importances_
            trans_names = xgb_pipe.named_steps["pre"].get_feature_names_out().tolist()
        elif model_choice == "Blend (avg RF+XGB)" and XGB_OK:
            imp_rf = rf_pipe.named_steps["reg"].feature_importances_
            imp_xgb = xgb_pipe.named_steps["reg"].feature_importances_
            imp_rf = imp_rf / (imp_rf.sum() + 1e-12)
            imp_xgb = imp_xgb / (imp_xgb.sum() + 1e-12)
            importances = 0.5 * (imp_rf + imp_xgb)
            trans_names = rf_pipe.named_steps["pre"].get_feature_names_out().tolist()
        else:
            importances = rf_pipe.named_steps["reg"].feature_importances_
            trans_names = rf_pipe.named_steps["pre"].get_feature_names_out().tolist()

        def _group_importances(trans_names, all_cat_cols, importances):
            grouped = {}
            for i, name in enumerate(trans_names):
                if name.startswith("num__"):
                    orig = name[len("num__"):]
                elif name.startswith("cat__"):
                    orig = None
                    for c in all_cat_cols:
                        pref = f"cat__{c}_"
                        if name.startswith(pref):
                            orig = c
                            break
                    if orig is None: orig = name
                else:
                    orig = name
                grouped.setdefault(orig, 0.0)
                grouped[orig] += float(importances[i])
            return sorted(grouped.items(), key=lambda kv: kv[1], reverse=True)

        gitems = _group_importances(trans_names, all_cat_cols, importances)[:20]
        fig5, ax5 = plt.subplots()
        ax5.bar([k for k, _ in gitems], [v for _, v in gitems])
        ax5.set_xticklabels([k for k, _ in gitems], rotation=45, ha="right")
        ax5.set_ylabel("Grouped importance (sum)"); ax5.set_title("Model importances (top 20)")
        fig5.tight_layout(); st.pyplot(fig5); plt.close(fig5)
    except Exception as e:
        st.info(f"Importances not available: {e}")

with tab_beh:
    st.caption("Pick a behaviour to see mean predicted index by category (on sampled hold-out).")
    if len(beh_cols) == 0:
        st.info("No behaviour (categorical) columns detected.")
    else:
        beh_choice = st.selectbox("Behaviour", options=beh_cols)
        if beh_choice not in X_te_s.columns:
            st.info("Selected behaviour not found in the sampled set.")
        else:
            df_te = X_te_s.copy()
            df_te["_yhat_"] = y_pred_s
            levels = df_te[beh_choice].astype(str).value_counts().head(10).index.tolist()
            means = []
            for lv in levels:
                m = float(df_te.loc[df_te[beh_choice].astype(str) == lv, "_yhat_"].mean())
                if not np.isnan(m): means.append((lv, m))
            means = sorted(means, key=lambda kv: kv[1], reverse=True)
            fig6, ax6 = plt.subplots()
            ax6.bar([k for k, _ in means], [v for _, v in means])
            ax6.set_xticklabels([k for k, _ in means], rotation=45, ha="right")
            ax6.set_ylabel("Mean predicted index"); ax6.set_title(f"{beh_choice}: mean predicted index by category")
            fig6.tight_layout(); st.pyplot(fig6); plt.close(fig6)
            st.write("**Levels shown (top by frequency):**", ", ".join(levels))

with tab_pdp:
    st.caption("Partial dependence can be expensive. Toggle to compute.")
    do_pdp = st.toggle("Compute PDP / ICE (fast mode)", value=False)
    if do_pdp:
        try:
            active_pipe, active_name = active_pipe_for_explain()
            pick = st.selectbox("Numeric feature", options=(num_cols[:3] if len(num_cols) >= 3 else num_cols))
            fig7, ax7 = plt.subplots()
            PartialDependenceDisplay.from_estimator(active_pipe, X_te_s, features=[pick], kind="average", grid_resolution=PDP_GRID, ax=ax7)
            ax7.set_title(f"PDP · {pick} (sampled) · {active_name}")
            fig7.tight_layout(); st.pyplot(fig7); plt.close(fig7)
        except Exception as e:
            st.info(f"PDP unavailable: {e}")

with st.expander("See features used for training & current selections"):
    st.caption(f"Numeric features ({len(num_cols)}): {', '.join(num_cols[:30])}{' ...' if len(num_cols) > 30 else ''}")
    st.caption(f"Behaviour (categorical) features ({len(beh_cols)}): {', '.join(beh_cols)}")
    other = [c for c in all_cat_cols if c not in beh_cols]
    st.caption(f"SES (categorical) features ({len(other)}): {', '.join(other)}")
    st.caption(f"SES column map: {ses_meta['ses_map']}")
    st.caption(f"Income q34/q67: {ses_meta['income_quantiles']}")

# ------------------------ INPUT UI (patient) -----------------------
st.subheader("Enter Elham Index (counts)")
left, right = st.columns(2)
elham_core = {}
present_elham_fields = [c for c in ELHAM_FIELDS_PRESENT if c in df.columns]
mid = len(present_elham_fields) // 2
for k in present_elham_fields[:mid]:
    with left:
        elham_core[k] = st.number_input(k, min_value=0, step=1, value=int(df[k].median() if k in df.columns else 0))
for k in present_elham_fields[mid:]:
    with right:
        elham_core[k] = st.number_input(k, min_value=0, step=1, value=int(df[k].median() if k in df.columns else 0))

# Behaviours UI (dataset-driven)
st.subheader("Behavior & lifestyle inputs")
beh_vals, cols = {}, st.columns(2)
beh_options = build_options_from_df(df, beh_cols)
for i, c in enumerate(beh_cols):
    opts = beh_options.get(c, ["Unknown"])
    default = default_from_df(df, c)
    with cols[i % 2]:
        beh_vals[c] = st.selectbox(c, options=opts, index=(opts.index(default) if default in opts else 0))

# SES UI (dataset-driven)
ses_vals = {}
if len(ses_cols_ui):
    st.subheader("Socio-economic inputs")
    grid = st.columns(2)
    for i, c in enumerate(ses_cols_ui):
        opts = ses_options.get(c, ["Unknown"])
        default = default_from_df(df, c)
        with grid[i % 2]:
            ses_vals[c] = st.selectbox(c, options=opts, index=(opts.index(default) if default in opts else 0))

with st.expander("Your current selections"):
    st.json({"behaviours": beh_vals, "ses": ses_vals})

# ------------------------- PREDICT & EXPLAIN -----------------------
if st.button("Predict + Explain"):
    # Build one-row input for model
    num_cols_current = [c for c in num_cols if c in df.columns]
    X_row = {c: float(elham_core.get(c, df[c].median() if c in df.columns else 0)) for c in num_cols_current}
    for c in beh_cols:
        X_row[c] = beh_vals.get(c, default_from_df(df, c))
    for c in ses_cols_ui:
        X_row[c] = ses_vals.get(c, default_from_df(df, c))
    X_df = pd.DataFrame([X_row])

    # 1) Compute Elham index from entered counts (for tiering)
    entered_index, per_item = compute_elham_from_inputs(elham_core)
    st.info(f"**Computed Elham’s Index (from your inputs): {entered_index:.0f}**")

    # 2) ML prediction (chosen model)
    y_hat = float(predict_with_choice(X_df)[0])
    st.success(f"Model prediction of Elham Index ({model_choice}): **{y_hat:.2f}**")

    # 3) Risk tier from computed index
    tier = index_tier(entered_index, risk_bins)
    st.info(f"Risk tier based on computed Elham Index: **{tier.title()}**")

    # 4) Rule-based treatment plan from findings
    st.subheader("Treatment plan (rule-based from Elham findings)")
    for line in treatment_plan_from_elham(per_item, tier):
        st.markdown(line)

    # ---------------------- WHAT-IF SIMULATOR ----------------------
    st.subheader("🧪 What-if simulator (behaviours)")
    st.caption("Adjust behaviours (e.g., 3+/day snacks → 1–2/day) and see model re-prediction.")
    sim_cols = st.columns(2)
    sim_beh = {}
    for i, c in enumerate(beh_cols):
        opts = beh_options.get(c, ["Unknown"])
        current = beh_vals.get(c, opts[0])
        with sim_cols[i % 2]:
            try:
                sim_beh[c] = st.select_slider(f"What if **{c}** becomes", options=opts, value=current)
            except Exception:
                idx = opts.index(current) if current in opts else 0
                sim_beh[c] = st.selectbox(f"What if {c} becomes", options=opts, index=idx)

    X_row_sim = X_row.copy()
    for c in beh_cols:
        X_row_sim[c] = sim_beh[c]
    X_df_sim = pd.DataFrame([X_row_sim])
    y_sim = float(predict_with_choice(X_df_sim)[0])
    delta = y_sim - y_hat

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Simulated Elham Index (model)", f"{y_sim:.2f}", delta=f"{delta:+.2f}")
    with c2:
        changed = {k: (beh_vals.get(k, ""), sim_beh[k]) for k in beh_cols if sim_beh[k] != beh_vals.get(k, "")}
        st.write("**Changed behaviours**")
        st.json(changed if changed else {"(none)": "No behaviour changed"})

    st.divider()

    # ----------------- SHAP explanations (grouped) ----------------
    try:
        import shap
        active_pipe, active_name = active_pipe_for_explain()
        pre = active_pipe.named_steps["pre"]
        X_trans = pre.transform(X_df)
        explainer = shap.TreeExplainer(active_pipe.named_steps["reg"])
        shap_vals = explainer.shap_values(X_trans)

        feat_names_active = pre.get_feature_names_out().tolist()
        grouped = group_shap_by_original(feat_names_active, shap_vals, num_cols, all_cat_cols)

        st.subheader(f"Top drivers of prediction (SHAP · {active_name})")
        top_all = grouped[:14]
        plot_bar(top_all, "Top drivers of prediction")

        st.subheader("Behaviour drivers of the predicted index")
        beh_only = [(k, v) for k, v in grouped if k in beh_cols]
        beh_only = sorted(beh_only, key=lambda kv: abs(kv[1]), reverse=True)[:12]
        plot_bar(beh_only, "Behaviour features (grouped SHAP)")

        if len(ses_cols_ui):
            st.subheader("Socio-economic drivers of the predicted index")
            ses_only = [(k, v) for k, v in grouped if k in ses_cols_ui]
            ses_only = sorted(ses_only, key=lambda kv: abs(kv[1]), reverse=True)[:10]
            plot_bar(ses_only, "SES features (grouped SHAP)")
    except Exception as e:
        st.warning(f"Explanation step failed: {e}")

    # 5) Personalized preventive recommendations (from behaviours)
    st.subheader("Personalized preventive recommendations")
    for line in detailed_behavior_recommendations({c: beh_vals.get(c, "") for c in beh_cols}, tier):
        st.markdown(line)

# ----------------------- Fairness quick audit -----------------------
with st.expander("Fairness check by SES (hold-out)"):
    if len(ses_cols_ui):
        try:
            X_all_f = df[num_cols + all_cat_cols].copy()
            y_all_f = df[TARGET_COL].astype(float).values
            X_tr_a, X_te_a, y_tr_a, y_te_a = train_test_split(X_all_f, y_all_f, test_size=0.2, random_state=42)
            yhat_a = predict_with_choice(X_te_a)
            te = X_te_a.copy(); te["_y"] = y_te_a; te["_yhat"] = yhat_a
            for c in ses_cols_ui:
                mae = te.groupby(te[c].astype(str)).apply(lambda g: float(np.mean(np.abs(g["_y"] - g["_yhat"])))).rename("MAE")
                st.markdown(f"**MAE by {c}**"); st.dataframe(mae.sort_values().to_frame())
        except Exception as e:
            st.caption(f"Subgroup MAE not available: {e}")
    else:
        st.caption("No SES columns detected.")

# ----------------------- Final insights (hold-out) -----------------------
st.subheader("Final insights")
try:
    X_all_i = df[num_cols + all_cat_cols].copy()
    y_all_i = df[TARGET_COL].astype(float).values
    _, X_te_i, _, y_te_i = train_test_split(X_all_i, y_all_i, test_size=0.2, random_state=42)
    yhat_i = predict_with_choice(X_te_i)

    overall_mae = float(np.mean(np.abs(y_te_i - yhat_i)))
    st.markdown(f"**Overall MAE (chosen model): {overall_mae:.3f}**")

    # Importances for drivers
    if XGB_OK and model_choice in {"XGBoost", "Blend (avg RF+XGB)"}:
        imp_rf = rf_pipe.named_steps["reg"].feature_importances_
        trans_rf = rf_pipe.named_steps["pre"].get_feature_names_out().tolist()
        if model_choice == "XGBoost":
            importances = xgb_pipe.named_steps["reg"].feature_importances_
            trans_names = xgb_pipe.named_steps["pre"].get_feature_names_out().tolist()
        else:
            imp_xgb = xgb_pipe.named_steps["reg"].feature_importances_
            imp_rf = imp_rf / (imp_rf.sum() + 1e-12)
            imp_xgb = imp_xgb / (imp_xgb.sum() + 1e-12)
            importances = 0.5 * (imp_rf + imp_xgb)
            trans_names = trans_rf
    else:
        importances = rf_pipe.named_steps["reg"].feature_importances_
        trans_names = rf_pipe.named_steps["pre"].get_feature_names_out().tolist()

    def _group_imp(trans_names, all_cat_cols, importances):
        grouped = {}
        for i, name in enumerate(trans_names):
            if name.startswith("num__"):
                orig = name[len("num__"):]
            elif name.startswith("cat__"):
                orig = None
                for c in all_cat_cols:
                    pref = f"cat__{c}_"
                    if name.startswith(pref):
                        orig = c
                        break
                if orig is None: orig = name
            else:
                orig = name
            grouped.setdefault(orig, 0.0)
            grouped[orig] += float(importances[i])
        return grouped

    g = _group_imp(trans_names, all_cat_cols, importances)

    top_beh = sorted([(k, v) for k, v in g.items() if k in beh_cols], key=lambda kv: kv[1], reverse=True)[:5]
    top_ses = sorted([(k, v) for k, v in g.items() if (k in ses_cols_ui and k.lower() != "school")],
                     key=lambda kv: kv[1], reverse=True)[:5]

    st.write("**Top behaviour drivers:** " + (", ".join([k for k, _ in top_beh]) or "—"))
    st.write("**Top SES drivers (excluding school):** " + (", ".join([k for k, _ in top_ses]) or "—"))

    # Fairness gaps (exclude school). Flag when n≥20 and MAE ≥ 1.5× overall MAE
    te_i = X_te_i.copy(); te_i["_y"] = y_te_i; te_i["_yhat"] = yhat_i
    flagged_any = False
    for c in [c for c in ses_cols_ui if c.lower() != "school"]:
        grp = te_i.groupby(te_i[c].astype(str)).apply(
            lambda g: pd.Series({"n": len(g), "MAE": float(np.mean(np.abs(g["_y"] - g["_yhat"])))})
        ).sort_values("MAE", ascending=False)

        flagged = grp[(grp["n"] >= 20) & (grp["MAE"] >= (1.5 * overall_mae))]
        if len(flagged):
            flagged_any = True
            st.markdown(f"**Fairness gaps for `{c}`** (n≥20 and MAE ≥ 1.5× overall)")
            st.dataframe(flagged)

    if not flagged_any:
        st.caption("No fairness gaps flagged under the current thresholds (excluding school).")

except Exception as e:
    st.caption(f"Final insights unavailable: {e}")
