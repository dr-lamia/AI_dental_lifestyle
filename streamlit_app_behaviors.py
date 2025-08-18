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
from sklearn.inspection import permutation_importance, PartialDependenceDisplay


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
    header = [
        f"**Overall plan for {tier.title()} risk:**",
        f"- Recall: **{tier_plan(tier)['recall']}**",
        f"- Toothpaste: **{tier_plan(tier)['toothpaste']}**",
        f"- Mouthrinse: **{tier_plan(tier)['rinse']}**",
        f"- Varnish: **{tier_plan(tier)['varnish']}**",
        f"- Diet focus: **{tier_plan(tier)['diet_focus']}**",
    ]
    recs, seen = header[:], set()
    for name, val in all_behaviors.items():
        for line in lines_for_behavior(name, val, tier):
            if line and line not in seen:
                recs.append(f"- {line}")
                seen.add(line)
    return recs

# =============== ORDERED OPTIONS FOR WHAT-IF SLIDERS ==============
ORDERED_CHOICES = {
    "tooth_brushing_frequency": ["Irregular","1/day","2/day"],
    "interdental_cleaning": ["No","Yes"],
    "mouth_rinse": ["No","Yes"],
    "snacks_frequency": ["0/day","1–2/day","3+/day"],
    "sugar": ["None","Occasional","Frequent"],
    "carbonated_beverages": ["None","Occasional","Frequent"],
    "sticky_food": ["No","Yes"],
    "hydration": ["Low","Normal","High"],
    "salivary_ph": ["Low","Normal","High"],
    "salivary_consistency": ["Low","Normal","High"],
    "buffering_capacity": ["Low","Moderate","Normal","High"],
    "mutans_load_in_saliva": ["Low","Normal","High"],
    "lactobacilli_load_in_saliva": ["Low","Normal","High"],
    # type_of_diet & snack_content vary a lot → we’ll use dataset values
}

def ordered_options(col, cat_values, current):
    """Return an ordered list for a behaviour, ensuring current value exists."""
    if col in ORDERED_CHOICES:
        opts = ORDERED_CHOICES[col][:]
    else:
        opts = cat_values.get(col, [])
    if current not in opts:
        opts = [current] + [o for o in opts if o != current]
    return opts

# =============================== UI ===============================
st.title("🦷 Dental AI Lifestyle: Behaviors → Explainable Index + Advice")

df = load_data(DATA_PATH)
pipe, metrics, num_cols, cat_cols, feat_names, num_medians, cat_modes, cat_values, risk_bins = train_model(df)
st.success(f"Model ready · R² = {metrics['R2']:.3f} · MAE = {metrics['MAE']:.2f}")
# ========================= MODEL VISUALIZATIONS (FAST) =========================
st.subheader("Model visualizations")

# Reconstruct the same train/test split (same random_state)
X_all = df[num_cols + cat_cols].copy()
y_all = df[TARGET_COL].astype(float).values
X_tr_v, X_te_v, y_tr_v, y_te_v = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42
)

# ---- SPEED SETTINGS (tune here) ----
VIS_SAMPLE_MAX = 400     # cap hold-out rows used by visualizers
PI_REPEATS      = 3      # permutation importance repeats (15 -> 3)
PDP_GRID        = 15     # PDP grid resolution (20 -> 15)

# Sample the hold-out set to keep things snappy
if len(X_te_v) > VIS_SAMPLE_MAX:
    X_te_s = X_te_v.sample(VIS_SAMPLE_MAX, random_state=42)
    idx = X_te_s.index
    y_te_s = y_te_v[idx]
else:
    X_te_s, y_te_s = X_te_v, y_te_v

# Reuse predictions repeatedly
y_pred_s = pipe.predict(X_te_s)
resid_s  = y_te_s - y_pred_s

tab_perf, tab_imp, tab_beh, tab_pdp = st.tabs(
    ["📈 Performance", "⭐ Global importance", "🧠 Behaviour effects", "🧩 PDP / ICE"]
)

# ---------- 📈 Performance ----------
with tab_perf:
    import matplotlib.pyplot as plt

    with st.spinner("Drawing performance plots..."):
        # Predicted vs Actual
        fig1, ax1 = plt.subplots()
        ax1.scatter(y_te_s, y_pred_s, alpha=0.65)
        lo = float(min(y_te_s.min(), y_pred_s.min()))
        hi = float(max(y_te_s.max(), y_pred_s.max()))
        ax1.plot([lo, hi], [lo, hi])
        ax1.set_xlabel("Actual Elham Index")
        ax1.set_ylabel("Predicted Elham Index")
        ax1.set_title("Predicted vs Actual (sampled hold-out)")
        fig1.tight_layout()
        st.pyplot(fig1); plt.close(fig1)

        # Residuals histogram
        fig2, ax2 = plt.subplots()
        ax2.hist(resid_s, bins=30)
        ax2.set_xlabel("Residual (Actual − Predicted)")
        ax2.set_title("Residuals (sampled hold-out)")
        fig2.tight_layout()
        st.pyplot(fig2); plt.close(fig2)

        # Residuals vs Prediction
        fig3, ax3 = plt.subplots()
        ax3.scatter(y_pred_s, resid_s, alpha=0.6)
        ax3.axhline(0, linestyle="--")
        ax3.set_xlabel("Predicted")
        ax3.set_ylabel("Residual")
        ax3.set_title("Residuals vs Predicted (sampled hold-out)")
        fig3.tight_layout()
        st.pyplot(fig3); plt.close(fig3)

# ---------- ⭐ Global importance ----------
with tab_imp:
    st.caption("These can be slow. Toggle to compute.")
    do_pi  = st.toggle("Compute permutation importance (fast mode)", value=False)
    do_tree = st.toggle("Show RandomForest grouped importances", value=True)

    if do_pi:
        from sklearn.inspection import permutation_importance
        with st.spinner("Computing permutation importance on a sample..."):
            pi = permutation_importance(
                pipe, X_te_s, y_te_s, n_repeats=PI_REPEATS, random_state=42, scoring="r2"
            )
            base_names = num_cols + cat_cols
            pi_items = sorted(
                [(name, float(m)) for name, m in zip(base_names, pi.importances_mean)],
                key=lambda kv: kv[1], reverse=True
            )[:20]

            fig4, ax4 = plt.subplots()
            ax4.bar([k for k, _ in pi_items], [v for _, v in pi_items])
            ax4.set_xticklabels([k for k, _ in pi_items], rotation=45, ha="right")
            ax4.set_ylabel("Importance (mean R² drop)")
            ax4.set_title("Permutation importance (top 20, sampled)")
            fig4.tight_layout()
            st.pyplot(fig4); plt.close(fig4)

    if do_tree:
        try:
            reg = pipe.named_steps["reg"]
            pre = pipe.named_steps["pre"]
            trans_names = pre.get_feature_names_out().tolist()

            def _group_tree_importances(trans_names, num_cols, cat_cols, importances):
                grouped = {}
                for i, name in enumerate(trans_names):
                    if name.startswith("num__"):
                        orig = name[len("num__"):]
                    elif name.startswith("cat__"):
                        orig = None
                        for c in cat_cols:
                            pref = f"cat__{c}_"
                            if name.startswith(pref): orig = c; break
                        if orig is None: orig = name
                    else:
                        orig = name
                    grouped.setdefault(orig, 0.0)
                    grouped[orig] += float(importances[i])
                return sorted(grouped.items(), key=lambda kv: kv[1], reverse=True)

            gitems = _group_tree_importances(trans_names, num_cols, cat_cols, reg.feature_importances_)[:20]

            fig5, ax5 = plt.subplots()
            ax5.bar([k for k, _ in gitems], [v for _, v in gitems])
            ax5.set_xticklabels([k for k, _ in gitems], rotation=45, ha="right")
            ax5.set_ylabel("Grouped importance (sum)")
            ax5.set_title("Model internal importances (top 20)")
            fig5.tight_layout()
            st.pyplot(fig5); plt.close(fig5)
        except Exception as e:
            st.info(f"Tree importances not available: {e}")

# ---------- 🧠 Behaviour effects ----------
with tab_beh:
    st.caption("Pick a behaviour to see mean predicted index by category (on sampled hold-out).")
    if len(cat_cols) == 0:
        st.info("No behaviour (categorical) columns detected.")
    else:
        beh_choice = st.selectbox("Behaviour", options=cat_cols)
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
            ax6.set_ylabel("Mean predicted index")
            ax6.set_title(f"{beh_choice}: mean predicted index by category")
            fig6.tight_layout()
            st.pyplot(fig6); plt.close(fig6)
            st.write("**Levels shown (top by frequency):**", ", ".join(levels))

# ---------- 🧩 PDP / ICE ----------
with tab_pdp:
    st.caption("Partial dependence can be expensive. Toggle to compute.")
    do_pdp = st.toggle("Compute PDP / ICE (fast mode)", value=False)
    if do_pdp:
        from sklearn.inspection import PartialDependenceDisplay
        # Light scoring to pick interesting numerics
        try:
            # If permutation importance already computed above, reuse; else compute quick importances on numerics only
            if 'base_names' in locals():
                pi_map = {name: val for name, val in pi_items}  # quick reuse if available
            else:
                pi_map = {c: 0.0 for c in num_cols}
            top_num = sorted(num_cols, key=lambda c: pi_map.get(c, 0.0), reverse=True)[:3]
        except Exception:
            top_num = num_cols[:3]

        pick = st.selectbox("Numeric feature", options=top_num if len(top_num) else num_cols)
        with st.spinner("Computing PDP (fast)…"):
            try:
                fig7, ax7 = plt.subplots()
                PartialDependenceDisplay.from_estimator(
                    pipe, X_te_s, features=[pick], kind="average", grid_resolution=PDP_GRID, ax=ax7
                )
                ax7.set_title(f"PDP · {pick} (sampled)")
                fig7.tight_layout()
                st.pyplot(fig7); plt.close(fig7)
            except Exception as e:
                # Manual fallback
                def manual_pdp(pipe, X_df, feature, q_low=0.05, q_high=0.95, grid=PDP_GRID):
                    vals = np.linspace(float(X_df[feature].quantile(q_low)),
                                       float(X_df[feature].quantile(q_high)), grid)
                    means = []
                    for v in vals:
                        X_tmp = X_df.copy()
                        X_tmp[feature] = v
                        means.append(float(pipe.predict(X_tmp).mean()))
                    return vals, means

                xs, ys = manual_pdp(pipe, X_te_s, pick)
                fig8, ax8 = plt.subplots()
                ax8.plot(xs, ys)
                ax8.set_xlabel(pick)
                ax8.set_ylabel("Mean predicted Elham index")
                ax8.set_title(f"Manual PDP (sampled) · {pick}")
                fig8.tight_layout()
                st.pyplot(fig8); plt.close(fig8)

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

    # Baseline prediction
    y_hat = float(pipe.predict(X_df)[0])
    st.success(f"Predicted Elham’s Index (including wisdom): **{y_hat:.2f}**")
    tier = index_tier(y_hat, risk_bins)
    st.info(f"Risk tier based on predicted Elham Index: **{tier.title()}**")

    # -------------------------- WHAT-IF SIMULATOR --------------------------
    st.subheader("🧪 What-if simulator (behaviours)")
    st.caption("Adjust behaviours below (e.g., reduce snacks from 3+/day → 1–2/day) and see the new predicted index.")
    sim_cols = st.columns(2)
    sim_beh = {}

    for i, c in enumerate(cat_cols):
        current = beh_vals.get(c, cat_modes.get(c, "Unknown"))
        opts = ordered_options(c, cat_values, current)
        with sim_cols[i % 2]:
            if len(opts) <= 1:
                sim_beh[c] = current
            else:
                # Use select_slider for ordered choices, fallback to selectbox if needed
                try:
                    sim_beh[c] = st.select_slider(f"What if **{c}** becomes", options=opts, value=current)
                except Exception:
                    sim_beh[c] = st.selectbox(f"What if {c} becomes", options=opts, index=opts.index(current))

    # Build simulated row and predict
    X_row_sim = X_row.copy()
    for c in cat_cols:
        X_row_sim[c] = sim_beh[c]
    X_df_sim = normalize_cats(pd.DataFrame([X_row_sim]))
    y_sim = float(pipe.predict(X_df_sim)[0])
    delta = y_sim - y_hat

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Simulated Elham’s Index", f"{y_sim:.2f}", delta=f"{delta:+.2f}")
    with c2:
        # Show which behaviours changed (before → after)
        changed = {k: (beh_vals.get(k, ""), sim_beh[k]) for k in cat_cols if sim_beh[k] != beh_vals.get(k, "")}
        st.write("**Changed behaviours**")
        st.json(changed if changed else {"(none)": "No behaviour changed"})

    st.divider()

    # ------------------ SHAP explanations (overall + behaviours) ------------------
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
