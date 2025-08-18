import os, re
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

# ----------------------------- CONFIG -----------------------------
st.set_page_config(page_title="Elham AI · Behaviors + Explainability", page_icon="🦷", layout="wide")

DATA_PATH  = "data/no_recommendation_dental_dataset_cleaned_keep_including_wisdom.csv"
TARGET_COL = "elham_s_index_including_wisdom"
ID_COLS    = ["id"]

# Curated behavior columns from your sheet (adjust/extend if you add more)
BEHAVIOR_COLS = [
    "tooth_brushing_frequency","time_of_tooth_brushing","interdental_cleaning","mouth_rinse",
    "snacks_frequency","snack_content","sugar","sticky_food","carbonated_beverages",
    "type_of_diet","hydration","salivary_ph","salivary_consistency","buffering_capacity",
    "mutans_load_in_saliva","lactobacilli_load_in_saliva"
]

# ----------------------- NORMALIZATION HELPERS --------------------
def norm_yes_no(v: str) -> str:
    s = str(v).strip().lower()
    if s in {"yes","y","true","1"}:  return "Yes"
    if s in {"no","n","false","0"}:  return "No"
    if s in {"", "unknown","unk","na","nan","none"}: return "Unknown"
    return v

def norm_brushing_freq(v: str) -> str:
    s = str(v).strip().lower()
    if any(k in s for k in ["twice","two","2/day","2 per","2 "]): return "2/day"
    if any(k in s for k in ["once","one","1/day","1 per","1 "]):  return "1/day"
    if "irreg" in s or "sometimes" in s:                            return "Irregular"
    return v

def norm_snack_freq(v: str) -> str:
    s = str(v).strip().lower()
    if s in {"never","none","no","0","0/day"}:                      return "0/day"
    if any(k in s for k in ["1-2","1–2","sometimes","occasional","few"]): return "1–2/day"
    if any(k in s for k in [">2","3+","many","often","frequent","a lot"]): return "3+/day"
    return v

def norm_risk_freq(v: str) -> str:
    s = str(v).strip().lower()
    if s in {"none","no","never","0"}:         return "None"
    if any(k in s for k in ["occas","some","rare"]): return "Occasional"
    if any(k in s for k in ["freq","often","daily","many"]): return "Frequent"
    return v

def norm_saliva_level(v: str) -> str:
    s = str(v).strip().lower()
    if "low" in s or "acid" in s:              return "Low"
    if "high" in s:                            return "High"
    if "mod" in s:                             return "Moderate"
    return "Normal" if "normal" in s else v

def norm_mutans_lacto(v: str) -> str:
    s = str(v).strip().lower()
    if "more" in s or ">" in s or "10^5" in s or "10)5" in s: return "High"
    if "less" in s or "<" in s:                               return "Low"
    return "Normal" if "normal" in s else v

NORMALIZERS = {
    "tooth_brushing_frequency": norm_brushing_freq,
    "time_of_tooth_brushing":   lambda x: str(x).strip().title(),
    "interdental_cleaning":     norm_yes_no,
    "mouth_rinse":              norm_yes_no,
    "snacks_frequency":         norm_snack_freq,
    "snack_content":            lambda x: str(x).strip().title(),
    "sugar":                    norm_risk_freq,
    "sticky_food":              norm_yes_no,
    "carbonated_beverages":     norm_risk_freq,
    "type_of_diet":             lambda x: str(x).strip().title(),
    "hydration":                lambda x: str(x).strip().title(),
    "salivary_ph":              norm_saliva_level,
    "salivary_consistency":     norm_saliva_level,
    "buffering_capacity":       norm_saliva_level,
    "mutans_load_in_saliva":    norm_mutans_lacto,
    "lactobacilli_load_in_saliva": norm_mutans_lacto,
}

def normalize_cats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c, f in NORMALIZERS.items():
        if c in df.columns:
            df[c] = df[c].astype(str).map(f)
    # strip spaces uniformly
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()
    return df

# ----------------------- PREPROCESSING / MODEL --------------------
def make_ohe() -> OneHotEncoder:
    # sklearn >=1.2 uses sparse_output; older uses sparse
    try:    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError: return OneHotEncoder(handle_unknown="ignore", sparse=False)

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
        st.error(f"Target column '{TARGET_COL}' not found in dataset.")
        st.stop()
    return num_cols, cat_cols

def build_pipeline(num_cols, cat_cols):
    transformers = [("num", SimpleImputer(strategy="median"), num_cols)]
    if len(cat_cols) > 0:
        transformers.append(("cat", make_ohe(), cat_cols))
    pre = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)
    reg = RandomForestRegressor(n_estimators=450, random_state=42, n_jobs=-1)
    return Pipeline([("pre", pre), ("reg", reg)])

@st.cache_resource(show_spinner=False)
def train_model(df: pd.DataFrame):
    num_cols, cat_cols = split_feature_types(df)
    X = df[num_cols + cat_cols].copy()
    y = df[TARGET_COL].astype(float).values

    pipe = build_pipeline(num_cols, cat_cols)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    metrics = {"R2": float(r2_score(y_te, y_pred)), "MAE": float(mean_absolute_error(y_te, y_pred))}

    pre = pipe.named_steps["pre"]
    feat_names = pre.get_feature_names_out().tolist()

    num_medians = X[num_cols].median(numeric_only=True)
    cat_modes   = {c: (X[c].mode(dropna=True).iloc[0] if X[c].notna().any() else "Unknown") for c in cat_cols}
    cat_values  = {c: sorted(X[c].dropna().unique().tolist()) for c in cat_cols}   # for UI options

    return pipe, metrics, num_cols, cat_cols, feat_names, num_medians, cat_modes, cat_values

def group_shap_by_original(feature_names, shap_vals):
    group_map = {}
    for i, name in enumerate(feature_names):
        if name.startswith("num__"):
            orig = name.split("num__")[1]
        elif name.startswith("cat__"):
            orig = name.split("cat__")[1].split("_", 1)[0]
        else:
            orig = name
        group_map.setdefault(orig, []).append(i)

    arr = shap_vals[0] if isinstance(shap_vals, list) else shap_vals
    row = arr[0] if getattr(arr, "ndim", 1) == 2 else arr
    grouped = {orig: float(np.sum(row[idxs])) for orig, idxs in group_map.items()}
    return sorted(grouped.items(), key=lambda kv: abs(kv[1]), reverse=True)

# --------------------- RECOMMENDATION RULES -----------------------
RECO_MAP = {
    "tooth_brushing_frequency": "Brush twice daily with fluoride toothpaste; consider an electric brush if plaque control is suboptimal.",
    "interdental_cleaning": "Do daily interdental cleaning (floss/interdental brushes).",
    "mouth_rinse": "Use an alcohol-free fluoride mouthrinse once daily.",
    "snacks_frequency": "Reduce between-meal snacks; keep sweets with meals only.",
    "snack_content": "Prefer tooth-friendly snacks (cheese, nuts, raw veggies) instead of sticky sweets.",
    "sugar": "Cut down on added sugars; replace sugary drinks with water or milk.",
    "sticky_food": "Avoid sticky, retentive sweets; rinse with water afterwards.",
    "carbonated_beverages": "Limit carbonated/acidic drinks; use a straw and avoid sipping over time.",
    "type_of_diet": "Shift toward balanced meals rich in proteins, dairy, and vegetables; limit ultra-processed foods.",
    "hydration": "Increase water intake throughout the day.",
    "salivary_ph": "If pH is low: sugar-free gum; avoid acids; consider saliva substitutes if symptomatic.",
    "salivary_consistency": "If saliva is viscous: hydrate; review xerostomic meds; sugar-free gum/lozenges.",
    "buffering_capacity": "If buffering is low: limit acids; fluoride varnish as indicated; sugar-free gum may help.",
    "mutans_load_in_saliva": "High mutans: reinforce hygiene/fluoride; reduce sugars; consider chlorhexidine if indicated.",
    "lactobacilli_load_in_saliva": "High lactobacilli: review sugar frequency and retentive snacks."
}

def make_behavior_recommendations(top_features, patient_values):
    recs = []
    for feat, _impact in top_features[:12]:
        if feat not in RECO_MAP: 
            continue
        text = RECO_MAP[feat]
        val = str(patient_values.get(feat, "")).lower()
        if feat == "tooth_brushing_frequency" and ("once" in val or val.strip()=="1/day"):
            text += " (current: once daily)."
        if feat == "snacks_frequency" and any(t in val for t in ["many","often","frequent","3+"]):
            text += " (current: frequent snacks)."
        if feat == "salivary_ph" and "low" in val:
            text += " (low pH)."
        recs.append(text)
    if not recs:
        recs.append("Maintain twice-daily brushing with fluoride toothpaste, daily interdental cleaning, and regular checkups.")
    out, seen = [], set()
    for r in recs:
        if r not in seen:
            out.append(r); seen.add(r)
    return out

# ------------------------------- UI --------------------------------
st.title("🦷 Elham AI: Behaviors → Explainable Index + Advice")

df = load_data(DATA_PATH)
pipe, metrics, num_cols, cat_cols, feat_names, num_medians, cat_modes, cat_values = train_model(df)
st.success(f"Model ready · R² = {metrics['R2']:.3f} · MAE = {metrics['MAE']:.2f}")

# Debug/visibility: what features are used
with st.expander("See features used for training & current selections"):
    st.caption(f"Numeric features ({len(num_cols)}): {', '.join(num_cols[:30])}{' ...' if len(num_cols)>30 else ''}")
    st.caption(f"Behavior (categorical) features ({len(cat_cols)}): {', '.join(cat_cols)}")

# --- Inputs
st.subheader("Enter Elham features (counts)")
left, right = st.columns(2)
elham_core = {}

primary_elham_fields = [
    "missing_0_excluding_wisdom","missing_0_including_wisdom","decayed_1","filled_2",
    "hypoplasia_3","hypocalcification_4","fluorosis_5","erosion_6","abrasion_7",
    "attrition_8","abfraction","fractured_","sealant_a","crown_por","crown_abu","crown_imp","veneer_f","sound_te"
]
present_elham_fields = [c for c in primary_elham_fields if c in df.columns]
mid = len(present_elham_fields)//2

with left:
    for k in present_elham_fields[:mid]:
        elham_core[k] = st.number_input(k, min_value=0, step=1, value=int(num_medians.get(k, 0)))
with right:
    for k in present_elham_fields[mid:]:
        elham_core[k] = st.number_input(k, min_value=0, step=1, value=int(num_medians.get(k, 0)))

st.subheader("Behavior & lifestyle inputs")
beh_vals = {}
cols = st.columns(2)
for i, c in enumerate(cat_cols):
    opts = cat_values.get(c, [])
    default = cat_modes.get(c, opts[0] if opts else "Unknown")
    with cols[i % 2]:
        beh_vals[c] = st.selectbox(c, options=opts or ["Unknown"], index=(opts.index(default) if default in opts else 0))

with st.expander("Your current selections"):
    st.json(beh_vals)

if st.button("Predict + Explain"):
    # Build single-row input (apply same normalization to be safe)
    X_row = {}
    for c in num_cols:
        X_row[c] = float(elham_core.get(c, num_medians.get(c, 0)))
    for c in cat_cols:
        X_row[c] = beh_vals.get(c, cat_modes.get(c, "Unknown"))
    X_df = pd.DataFrame([X_row])
    X_df = normalize_cats(X_df)   # keep categories aligned

    # Predict
    y_hat = float(pipe.predict(X_df)[0])
    st.success(f"Predicted Elham’s Index (including wisdom): **{y_hat:.2f}**")

    # Explain
    try:
        import shap
        pre = pipe.named_steps["pre"]
        X_trans = pre.transform(X_df)
        explainer = shap.TreeExplainer(pipe.named_steps["reg"])
        shap_vals = explainer.shap_values(X_trans)
        grouped = group_shap_by_original(feat_names, shap_vals)
        top = grouped[:14]

        fig, ax = plt.subplots()
        ax.bar([k for k, _ in top], [v for _, v in top])
        ax.set_xticklabels([k for k, _ in top], rotation=45, ha="right")
        ax.set_ylabel("Grouped SHAP value")
        ax.set_title("Top drivers of prediction")
        fig.tight_layout()
        st.pyplot(fig)

        st.subheader("Personalized preventive recommendations")
        for r in make_behavior_recommendations(top, {**elham_core, **beh_vals}):
            st.markdown(f"- {r}")
    except Exception as e:
        st.warning(f"Explanation step failed: {e}")
