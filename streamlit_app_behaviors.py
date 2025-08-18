
import os, io
import numpy as np
import pandas as pd
import streamlit as st
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Elham AI – Behaviors + SHAP", page_icon="🦷", layout="wide")

DATA_PATH = "data/no_recommendation_dental_dataset_cleaned_keep_including_wisdom.csv"
TARGET_COL = "elham_s_index_including_wisdom"
ID_COLS = ["id"]

def detect_behavior_cols(df):
    pats = ["tooth","brush","interdental","mouth","rinse","diet","acid","hydration","sugar","sticky",
            "carbonated","exercise","supplement","medication","smoking","breakfast","lunch","dinner",
            "snack","dairy","protein","vegetable","fruit","spice","sweet","retention","salivary",
            "buffer","mutans","lactobacilli","ph","consistency","quality","orthodont","occlusion"]
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    return [c for c in obj_cols if any(p in c for p in pats)]

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()
    return df

def split_feature_types(df):
    cat_cols = detect_behavior_cols(df)
    num_cols = df.select_dtypes(exclude="object").columns.tolist()
    num_cols = [c for c in num_cols if c not in ID_COLS + [TARGET_COL]]
    assert TARGET_COL in df.columns, f"Target '{TARGET_COL}' missing."
    return num_cols, cat_cols

def build_pipeline(num_cols, cat_cols):
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols)
    ], remainder="drop", verbose_feature_names_out=True)
    model = RandomForestRegressor(n_estimators=450, random_state=42, n_jobs=-1)
    from sklearn.pipeline import Pipeline
    return Pipeline([("pre", pre), ("reg", model)])

@st.cache_resource(show_spinner=False)
def train_model(df):
    num_cols, cat_cols = split_feature_types(df)
    X = df[num_cols + cat_cols].copy()
    y = df[TARGET_COL].astype(float).values

    pipe = build_pipeline(num_cols, cat_cols)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(X_tr, y_tr)

    y_pred = pipe.predict(X_te)
    metrics = {"R2": r2_score(y_te, y_pred), "MAE": float(mean_absolute_error(y_te, y_pred))}

    pre = pipe.named_steps["pre"]
    feat_names = pre.get_feature_names_out().tolist()
    num_medians = X[num_cols].median(numeric_only=True)
    cat_modes = {c: X[c].mode(dropna=True).iloc[0] if X[c].notna().any() else "Unknown" for c in cat_cols}

    return pipe, metrics, num_cols, cat_cols, feat_names, num_medians, cat_modes

def group_shap_by_original(feature_names, shap_vals):
    group_map = {}
    for i,name in enumerate(feature_names):
        if name.startswith("num__"):
            orig = name.split("num__")[1]
        elif name.startswith("cat__"):
            orig = name.split("cat__")[1].split("_",1)[0]
        else:
            orig = name
        group_map.setdefault(orig, []).append(i)

    grouped = {}
    arr = shap_vals[0] if isinstance(shap_vals, list) else shap_vals
    row = arr[0] if getattr(arr, "ndim", 1) == 2 else arr
    for orig, idxs in group_map.items():
        grouped[orig] = float(np.sum(row[idxs]))
    sorted_items = sorted(grouped.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return sorted_items

def explain_single(pipe, X_row, feat_names, max_display=12):
    reg = pipe.named_steps["reg"]
    pre = pipe.named_steps["pre"]
    X_trans = pre.transform(X_row)
    import shap
    explainer = shap.TreeExplainer(reg)
    shap_vals = explainer.shap_values(X_trans)
    sorted_items = group_shap_by_original(feat_names, shap_vals)
    top = sorted_items[:max_display]
    fig, ax = plt.subplots()
    ax.bar([k for k,_ in top], [v for _,v in top])
    ax.set_xticklabels([k for k,_ in top], rotation=45, ha="right")
    ax.set_ylabel("Grouped SHAP value")
    ax.set_title("Top drivers (grouped by feature)")
    fig.tight_layout()
    return sorted_items, fig

RECO_MAP = {
  "tooth_brushing_frequency": "Brush twice daily with fluoride toothpaste; consider an electric brush.",
  "interdental_cleaning": "Do daily interdental cleaning (floss/interdental brushes).",
  "mouth_rinse": "Use an alcohol‑free fluoride mouthrinse once daily.",
  "snacks_frequency": "Reduce between‑meal snacks; keep sweets with meals only.",
  "snack_content": "Choose tooth‑friendly snacks (cheese, nuts, raw veggies) instead of sticky sweets.",
  "sugar": "Cut down on added sugars; replace sugary drinks with water or milk.",
  "sticky_food": "Avoid sticky, retentive sweets; rinse with water afterwards.",
  "carbonated_beverages": "Limit carbonated/acidic drinks; use a straw and avoid sipping over time.",
  "type_of_diet": "Prefer balanced meals rich in proteins, dairy and vegetables; limit ultra‑processed foods.",
  "hydration": "Increase water intake throughout the day.",
  "salivary_ph": "If pH is low: sugar‑free gum; avoid acids; consider saliva substitutes if symptomatic.",
  "salivary_consistency": "If saliva is viscous: hydrate; review xerostomic meds; sugar‑free gum/lozenges.",
  "buffering_capacity": "If buffering is low: limit acids; fluoride varnish as indicated; sugar‑free gum may help.",
  "mutans_load_in_saliva": "High mutans: reinforce hygiene/fluoride; reduce sugars; consider chlorhexidine if indicated.",
  "lactobacilli_load_in_saliva": "High lactobacilli: review sugar frequency and retentive snacks."
}

def make_behavior_recommendations(top_features, patient_values):
    recs = []
    for feat, impact in top_features[:12]:
        if feat in RECO_MAP:
            text = RECO_MAP[feat]
            val = str(patient_values.get(feat, ""))
            if feat == "tooth_brushing_frequency" and val and ("once" in val.lower() or val.strip()=="1"):
                text += " (current: once daily)."
            if feat == "snacks_frequency" and val and any(s in val.lower() for s in ["many","often","frequent",">"]):
                text += " (current: frequent snacks)."
            if feat == "salivary_ph" and val and "low" in val.lower():
                text += " (low pH)."
            recs.append(text)
    if not recs:
        recs.append("Maintain twice‑daily brushing with fluoride toothpaste, daily interdental cleaning, and regular checkups.")
    # deduplicate
    out, seen = [], set()
    for r in recs:
        if r not in seen:
            out.append(r); seen.add(r)
    return out

# -------- UI --------
st.title("🦷 Elham AI: Behaviors → Explainable Index + Advice")

# Load data & train
if not os.path.exists(DATA_PATH):
    st.error(f"Dataset not found at '{DATA_PATH}'. Create a 'data' folder and put the CSV there.")
    st.stop()

df = load_data(DATA_PATH)
pipe, metrics, num_cols, cat_cols, feat_names, num_medians, cat_modes = train_model(df)

st.success(f"Model ready · R² = {metrics['R2']:.3f}, MAE = {metrics['MAE']:.2f}")

# Numeric elham inputs
st.subheader("Enter Elham features")
left,right = st.columns(2)
elham_core = {}
with left:
    for k in ["missing_0_excluding_wisdom","missing_0_including_wisdom","decayed_1","filled_2",
              "hypoplasia_3","hypocalcification_4","fluorosis_5","erosion_6","abrasion_7"]:
        if k in df.columns:
            elham_core[k] = st.number_input(k, min_value=0, step=1, value=int(num_medians.get(k,0)))
with right:
    for k in ["attrition_8","abfraction","fractured_","sealant_a","crown_por","crown_abu","crown_imp","veneer_f","sound_te"]:
        if k in df.columns:
            elham_core[k] = st.number_input(k, min_value=0, step=1, value=int(num_medians.get(k,0)))

# Behavioral inputs (categorical)
st.subheader("Behavior & lifestyle inputs")
beh_vals = {}
cols = st.columns(2)
for i,c in enumerate(cat_cols):
    opts = sorted([str(x) for x in df[c].dropna().unique().tolist()])
    default = cat_modes.get(c, opts[0] if opts else "")
    with cols[i%2]:
        beh_vals[c] = st.selectbox(c, options=opts, index=opts.index(default) if default in opts else 0)

if st.button("Predict + Explain"):
    X_row = {}
    for c in num_cols:
        X_row[c] = float(elham_core.get(c, num_medians.get(c, 0)))
    for c in cat_cols:
        X_row[c] = beh_vals.get(c, cat_modes.get(c, "Unknown"))
    X_df = pd.DataFrame([X_row])

    y_hat = float(pipe.predict(X_df)[0])
    st.success(f"Predicted Elham’s Index (including wisdom): **{y_hat:.2f}**")

    sorted_items, fig = explain_single(pipe, X_df, feat_names, max_display=14)
    st.pyplot(fig)

    recs = make_behavior_recommendations(sorted_items, {**elham_core, **beh_vals})
    st.subheader("Personalized preventive recommendations")
    for r in recs:
        st.markdown(f"- {r}")
