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
    return str(v).strip().title()

def norm_brushing_freq(v: str) -> str:
    s = str(v).strip().lower()
    if any(k in s for k in ["twice","two","2/day","2 per"," 2 "]): return "2/day"
    if any(k in s for k in ["once","one","1/day","1 per"," 1 "]):  return "1/day"
    if "irreg" in s or "sometimes" in s:                            return "Irregular"
    return str(v).strip().title()

def norm_snack_freq(v: str) -> str:
    s = str(v).strip().lower()
    if s in {"never","none","no","0","0/day"}:                      return "0/day"
    if any(k in s for k in ["1-2","1–2","sometimes","occasional","few"]): return "1–2/day"
    if any(k in s for k in [">2","3+","many","often","frequent","a lot"]): return "3+/day"
    return str(v).strip().title()

def norm_risk_freq(v: str) -> str:
    s = str(v).strip().lower()
    if s in {"none","no","never","0"}:         return "None"
    if any(k in s for k in ["occas","some","rare"]): return "Occasional"
    if any(k in s for k in ["freq","often","daily","many"]): return "Frequent"
    return str(v).strip().title()

def norm_saliva_level(v: str) -> str:
    s = str(v).strip().lower()
    if "low" in s or "acid" in s:              return "Low"
    if "high" in s:                            return "High"
    if "mod" in s:                             return "Moderate"
    return "Normal" if "normal" in s else str(v).strip().title()

def norm_mutans_lacto(v: str) -> str:
    s = str(v).strip().lower()
    if "more" in s or ">" in s or "10^5" in s or "10)5" in s: return "High"
    if "less" in s or "<" in s:                               return "Low"
    return "Normal" if "normal" in s else str(v).strip().title()

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

# ----------------------- RISK TIERS (INDEX) -----------------------
def build_risk_bins(df, target_col=TARGET_COL):
    """Compute terciles to split low/moderate/high from your data."""
    y = df[target_col].dropna().values
    q1, q2 = np.quantile(y, [0.34, 0.67])
    return (float(q1), float(q2))  # (low_upper, mod_upper)

def index_tier(y_hat, bins):
    low_u, mod_u = bins
    if y_hat < low_u:   return "low"
    if y_hat < mod_u:   return "moderate"
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
    return dict(  # low
        recall="6–12 months",
        toothpaste="1450 ppm fluoride twice daily",
        rinse="Optional fluoride rinse if enamel defects or ortho",
        varnish="Varnish at routine intervals if indicated",
        diet_focus="Maintain current habits; keep sweets with meals",
    )

# ----------------------- PREPROCESSING / MODEL --------------------
def make_ohe() -> OneHotEncoder:
    try:    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # sklearn >=1.2
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
    cat_values  = {c: sorted(X[c].dropna().unique().tolist()) for c in cat_cols}

    risk_bins = build_risk_bins(df, TARGET_COL)
    return pipe, metrics, num_cols, cat_cols, feat_names, num_medians, cat_modes, cat_values, risk_bins

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

# ------------------ DETAILED BEHAVIOUR ADVICE ENGINE ---------------
def lines_for_behavior(name, val, tier):
    v = (str(val) if val is not None else "Unknown").strip().lower()
    plan = tier_plan(tier)
    out = []

    # Oral hygiene
    if name == "tooth_brushing_frequency":
        if v in {"1/day","once","1"}:
            out += [
                "Brush **twice daily** with fluoride toothpaste.",
                plan["toothpaste"],
                "Use a **pea-sized** amount; spit, don’t rinse after brushing.",
                "Consider an **electric brush** if plaque control is sub-optimal."
            ]
        elif v in {"irregular","sometimes"}:
            out += ["Set **fixed times** for brushing (morning & before bed).", plan["toothpaste"]]
        else:
            out += [plan["toothpaste"]]

    if name == "interdental_cleaning":
        out += ["Start **daily interdental cleaning** (floss/interproximal brushes)." if v in {"no","unknown"} 
                else "Keep **daily interdental cleaning**; focus on back teeth."]

    if name == "mouth_rinse":
        out += [f"Add a **fluoride mouthrinse**: {plan['rinse']}." if v in {"no","unknown"} 
                else f"Continue **fluoride mouthrinse**: {plan['rinse']}."]
    
    # Diet / sugars / snacks
    if name == "snacks_frequency":
        if any(k in v for k in ["3+",">2","many","often","frequent"]):
            out += ["**Cut snacks to ≤1–2/day**; keep sweets **with meals**.", plan["diet_focus"]]
        elif any(k in v for k in ["1–2","1-2","sometimes","occasional","few"]):
            out += ["Keep snacks **≤1–2/day**; choose tooth-friendly options."]
        else:
            out += ["Great job limiting snacks; keep sweets **with meals** only."]

    if name == "snack_content":
        if any(k in v for k in ["sweet","candy","dessert","cake","chocolate"]):
            out += ["Swap sweets for **cheese, nuts, raw veg, yogurt**."]
        if "sticky" in v:
            out += ["Avoid **sticky, retentive sweets** (caramels, toffees)."]

    if name == "sugar":
        if any(k in v for k in ["frequent","daily","often"]):
            out += ["**Reduce added sugars**; replace sugary drinks with **water or milk**."]
        elif any(k in v for k in ["occasional","some"]):
            out += ["Keep sugars to **meals only** and limit portion size."]
        else:
            out += ["Maintain **low sugar** intake."]

    if name == "carbonated_beverages":
        if any(k in v for k in ["frequent","daily","often"]):
            out += ["Limit **carbonated/acidic drinks**; use a **straw**, avoid **sipping over time**, and **rinse with water** afterwards."]
        elif any(k in v for k in ["occasional","some"]):
            out += ["Keep acidic drinks **occasional**; prefer water."]
        else:
            out += ["Great—**avoid acidic drinks** as a habit."]

    if name == "type_of_diet":
        out += ["Shift toward **balanced meals** rich in **proteins, dairy, vegetables**; cut ultra-processed foods."
                if any(k in v for k in ["junk","fast"]) else
                "Keep a **balanced diet**; pair carbs with protein/dairy."]

    if name == "hydration":
        out += ["Increase **water intake** throughout the day (carry a bottle)." if "low" in v else "Maintain **good hydration**."]

    if name == "sticky_food":
        if v in {"yes","y"}:
            out += ["Avoid **sticky foods**; if eaten, **rinse with water** and avoid bedtime intake."]

    # Saliva / biology
    if name == "salivary_ph":
        if "low" in v:
            out += ["**Low pH**: use **sugar-free gum** (xylitol), avoid acids between meals, increase hydration.", plan["rinse"]]
        else:
            out += ["Maintain **neutral pH** habits: water instead of acidic drinks."]

    if name == "salivary_consistency":
        if "high" in v or "thick" in v or "visc" in v:
            out += ["**Thick saliva**: hydrate, review meds causing dryness, consider sugar-free lozenges/gum."]
        elif "low" in v:
            out += ["Ensure **adequate hydration**; monitor for dryness symptoms."]

    if name == "buffering_capacity":
        out += ["**Low buffering**: minimize acids; consider **varnish** and sugar-free gum after meals."
                if "low" in v else
                "Maintain habits that support saliva buffering (water, balanced meals)."]

    if name == "mutans_load_in_saliva":
        out += ["**High mutans**: tighten hygiene + fluoride, reduce sugars, consider **CHX** if clinically indicated."
                if "high" in v else
                "Maintain good control of **mutans streptococci**." if "low" in v else ""]

    if name == "lactobacilli_load_in_saliva":
        out += ["**High lactobacilli**: focus on **fermentable carbs** reduction and **retentive snacks**."
                if "high" in v else
                "Maintain low **lactobacilli** through diet control." if "low" in v else ""]

    return [x for x in out if x]

def detailed_behavior_recommendations(all_behaviors: dict, tier: str):
    recs = []
    for name, val in all_behaviors.items():
        recs += lines_for_behavior(name, val, tier)

    plan = tier_plan(tier)
    header = [
        f"**Overall plan for {tier.title()} risk**:",
        f"- Recall: **{plan['recall']}**",
        f"- Toothpaste: **{plan['toothpaste']}**",
        f"- Mouthrinse: **{plan['rinse']}**",
        f"- Varnish: **{plan['varnish']}**",
        f"- Diet focus: **{plan['diet_focus']}**",
    ]

    out, seen = header[:], set()
    for r in recs:
        if r and r not in seen:
            out.append(f"- {r}")
            seen.add(r)
    return out

# ------------------------------- UI --------------------------------
st.title("🦷 Elham AI: Behaviors → Explainable Index + Advice")

df = load_data(DATA_PATH)
pipe, metrics, num_cols, cat_cols, feat_names, num_medians, cat_modes, cat_values, risk_bins = train_model(df)
st.success(f"Model ready · R² = {metrics['R2']:.3f} · MAE = {metrics['MAE']:.2f}")

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
present_elham_fields = [c for c in primary_elham_fields if c in df.columns]; mid = len(present_elham_fields)//2
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
    X_row = {c: float(elham_core.get(c, num_medians.get(c, 0))) for c in num_cols}
    for c in cat_cols:
        X_row[c] = beh_vals.get(c, cat_modes.get(c, "Unknown"))
    X_df = normalize_cats(pd.DataFrame([X_row]))

    y_hat = float(pipe.predict(X_df)[0])
    st.success(f"Predicted Elham’s Index (including wisdom): **{y_hat:.2f}**")

    # Risk tier + summary
    tier = index_tier(y_hat, risk_bins)
    st.info(f"Risk tier based on predicted Elham Index: **{tier.title()}**")

    # SHAP explanation
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
    except Exception as e:
        st.warning(f"Explanation step failed: {e}")

    # Detailed tiered advice
    st.subheader("Personalized preventive recommendations")
    for line in detailed_behavior_recommendations({c: beh_vals.get(c, "") for c in cat_cols}, tier):
        st.markdown(line)
