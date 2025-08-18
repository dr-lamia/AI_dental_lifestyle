import os
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

# ============================ CONFIG =============================
st.set_page_config(page_title="Elham AI · Behaviors + Explainability",
                   page_icon="🦷", layout="wide")

DATA_PATH  = "data/no_recommendation_dental_dataset_cleaned_keep_including_wisdom.csv"
TARGET_COL = "elham_s_index_including_wisdom"
ID_COLS    = ["id"]

# Curated behaviour columns taken from your sheet
BEHAVIOR_COLS = [
    "tooth_brushing_frequency","time_of_tooth_brushing","interdental_cleaning","mouth_rinse",
    "snacks_frequency","snack_content","sugar","sticky_food","carbonated_beverages",
    "type_of_diet","hydration","salivary_ph","salivary_consistency","buffering_capacity",
    "mutans_load_in_saliva","lactobacilli_load_in_saliva"
]

# ====================== NORMALIZATION HELPERS =====================
def _title(v): return str(v).strip().title()

def norm_yes_no(v: str) -> str:
    s = str(v).strip().lower()
    if s in {"yes","y","true","1"}:  return "Yes"
    if s in {"no","n","false","0"}:  return "No"
    if s in {"", "unknown","unk","na","nan","none"}: return "Unknown"
    return _title(v)

def norm_brushing_freq(v: str) -> str:
    s = str(v).strip().lower()
    if any(k in s for k in ["twice","two","2/day","2 per"," 2 "]): return "2/day"
    if any(k in s for k in ["once","one","1/day","1 per"," 1 "]):  return "1/day"
    if "irreg" in s or "sometimes" in s:                            return "Irregular"
    return _title(v)

def norm_snack_freq(v: str) -> str:
    s = str(v).strip().lower()
    if s in {"never","none","no","0","0/day"}:                           return "0/day"
    if any(k in s for k in ["1-2","1–2","sometimes","occasional","few"]): return "1–2/day"
    if any(k in s for k in [">2","3+","many","often","frequent","a lot"]):return "3+/day"
    return _title(v)

def norm_risk_freq(v: str) -> str:
    s = str(v).strip().lower()
    if s in {"none","no","never","0"}:                 return "None"
    if any(k in s for k in ["occas","some","rare"]):   return "Occasional"
    if any(k in s for k in ["freq","often","daily","many"]): return "Frequent"
    return _title(v)

def norm_saliva_level(v: str) -> str:
    s = str(v).strip().lower()
    if "low" in s or "acid" in s: return "Low"
    if "high" in s:               return "High"
    if "mod" in s:                return "Moderate"
    return "Normal" if "normal" in s else _title(v)

def norm_mutans_lacto(v: str) -> str:
    s = str(v).strip().lower()
    if "more" in s or ">" in s or "10^5" in s or "10)5" in s: return "High"
    if "less" in s or "<" in s:                               return "Low"
    return "Normal" if "normal" in s else _title(v)

NORMALIZERS = {
    "tooth_brushing_frequency": norm_brushing_freq,
    "time_of_tooth_brushing":   _title,
    "interdental_cleaning":     norm_yes_no,
    "mouth_rinse":              norm_yes_no,
    "snacks_frequency":         norm_snack_freq,
    "snack_content":            _title,
    "sugar":                    norm_risk_freq,
    "sticky_food":              norm_yes_no,
    "carbonated_beverages":     norm_risk_freq,
    "type_of_diet":             _title,
    "hydration":                _title,
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
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()
    return df

# ========================= RISK TIERS =============================
def build_risk_bins(df, target_col=TARGET_COL):
    y = df[target_col].dropna().values
    q1, q2 = np.quantile(y, [0.34, 0.67])   # tercile-like cutpoints
    return (float(q1), float(q2))

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

# ==================== PREPROCESS / TRAINING =======================
def make_ohe() -> OneHotEncoder:
    # sklearn >=1.2 uses 'sparse_output'; older uses 'sparse'
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
        st.error(f"Target column '{TARGET_COL}' not found.")
        st.stop()
    return num_cols, cat_cols

def build_pipeline(num_cols, cat_cols):
    transformers = [("num", SimpleImputer(strategy="median"), num_cols)]
    if len(cat_cols) > 0:
        transformers.append(("cat", make_ohe(), cat_cols))
    pre = ColumnTransformer(transformers, remainder="drop",
                            verbose_feature_names_out=True)
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
    metrics = {"R2": float(r2_score(y_te, y_pred)),
               "MAE": float(mean_absolute_error(y_te, y_pred))}

    pre = pipe.named_steps["pre"]
    feat_names = pre.get_feature_names_out().tolist()
    num_medians = X[num_cols].median(numeric_only=True)
    cat_modes   = {c: (X[c].mode(dropna=True).iloc[0] if X[c].notna().any() else "Unknown") for c in cat_cols}
    cat_values  = {c: sorted(X[c].dropna().unique().tolist()) for c in cat_cols}
    risk_bins   = build_risk_bins(df, TARGET_COL)

    return pipe, metrics, num_cols, cat_cols, feat_names, num_medians, cat_modes, cat_values, risk_bins

# ======== SHAP GROUPING + SMALL PLOTTING HELPERS ==================
def build_group_map(feature_names, num_cols, cat_cols):
    """
    Map transformed feature indices back to original column names.
    Works with ColumnTransformer prefixes: 'num__' and 'cat__<col>_'.
    """
    group_map = {}
    for i, name in enumerate(feature_names):
        if name.startswith("num__"):
            orig = name[len("num__"):]
        elif name.startswith("cat__"):
            orig = None
            for c in cat_cols:
                prefix = f"cat__{c}_"
                if name.startswith(prefix):
                    orig = c; break
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
    ax.set_ylabel("Grouped SHAP value")
    ax.set_title(title)
    fig.tight_layout()
    st.pyplot(fig)

# =================== DETAILED ADVICE (ALL BEHAVIOURS) =============
def tiered_plan_header(tier):
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
            out += ["**Low pH**: use **sugar-free gum** (xylitol), avoid acids between meals, increase hydration.",
                    plan["rinse"]]
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
    header = tiered_plan_header(tier)
    recs, seen = header[:], set()
    for name, val in all_behaviors.items():
        for line in lines_for_behavior(name, val, tier):
            if line and line not in seen:
                recs.append(f"- {line}")
                seen.add(line)
    return recs

# =============================== UI ===============================
st.title("🦷 Elham AI: Behaviors → Explainable Index + Advice")

df = load_data(DATA_PATH)
pipe, metrics, num_cols, cat_cols, feat_names, num_medians, cat_modes, cat_values, risk_bins = train_model(df)
st.success(f"Model ready · R² = {metrics['R2']:.3f} · MAE = {metrics['MAE']:.2f}")

with st.expander("See features used for training & current selections"):
    st.caption(f"Numeric features ({len(num_cols)}): {', '.join(num_cols[:30])}{' ...' if len(num_cols)>30 else ''}")
    st.caption(f"Behaviour (categorical) features ({len(cat_cols)}): {', '.join(cat_cols)}")

# Elham counts
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

# Behaviours
st.subheader("Behavior & lifestyle inputs")
beh_vals = {}
cols = st.columns(2)
for i, c in enumerate(cat_cols):
    opts = cat_values.get(c, [])
    default = cat_modes.get(c, opts[0] if opts else "Unknown")
    with cols[i % 2]:
        beh_vals[c] = st.selectbox(c, options=opts or ["Unknown"],
                                   index=(opts.index(default) if default in opts else 0))

with st.expander("Your current selections"):
    st.json(beh_vals)

if st.button("Predict + Explain"):
    # Assemble one-row input; normalize for safety
    X_row = {c: float(elham_core.get(c, num_medians.get(c, 0))) for c in num_cols}
    for c in cat_cols:
        X_row[c] = beh_vals.get(c, cat_modes.get(c, "Unknown"))
    X_df = normalize_cats(pd.DataFrame([X_row]))

    # Predict
    y_hat = float(pipe.predict(X_df)[0])
    st.success(f"Predicted Elham’s Index (including wisdom): **{y_hat:.2f}**")
    tier = index_tier(y_hat, risk_bins)
    st.info(f"Risk tier based on predicted Elham Index: **{tier.title()}**")

    # SHAP explanations (overall + behaviours-only) + table
    try:
        import shap
        pre = pipe.named_steps["pre"]
        X_trans = pre.transform(X_df)
        explainer = shap.TreeExplainer(pipe.named_steps["reg"])
        shap_vals = explainer.shap_values(X_trans)

        grouped = group_shap_by_original(feat_names, shap_vals, num_cols, cat_cols)

        st.subheader("Top drivers of prediction (all features)")
        top_all = grouped[:14]
        plot_bar(top_all, "Top drivers of prediction")

        st.subheader("Behaviour drivers of the predicted index")
        beh_only = [(k, v) for k, v in grouped if k in cat_cols]
        beh_only = sorted(beh_only, key=lambda kv: abs(kv[1]), reverse=True)
        top_beh = beh_only[:12]
        plot_bar(top_beh, "Behaviour features (grouped SHAP)")

        if top_beh:
            rows = []
            for name, impact in top_beh:
                pv = beh_vals.get(name, cat_modes.get(name, "Unknown"))
                direction = "↑ increases index" if impact > 0 else "↓ decreases index"
                try:
                    adv_lines = lines_for_behavior(name, pv, tier)
                    first_line = adv_lines[0] if adv_lines else ""
                except Exception:
                    first_line = ""
                rows.append({
                    "Behaviour": name,
                    "Patient value": str(pv),
                    "Impact (SHAP)": round(float(impact), 4),
                    "Direction": direction,
                    "Advice (first line)": first_line
                })
            st.markdown("**Most influential behaviours for this patient**")
            st.dataframe(pd.DataFrame(rows))
        else:
            st.info("No behaviour features appeared in the top drivers.")

    except Exception as e:
        st.warning(f"Explanation step failed: {e}")

    # Detailed tiered advice for all behaviours (not just top)
    st.subheader("Personalized preventive recommendations")
    for line in detailed_behavior_recommendations({c: beh_vals.get(c, "") for c in cat_cols}, tier):
        st.markdown(line)
