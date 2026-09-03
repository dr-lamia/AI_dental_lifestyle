import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from analysis_pipeline import (
    TARGET, TARGET_COMPONENTS, ELHAM_DIRECT_COMPONENTS, BEHAVIOR_COLS,
    SALIVARY_COLS, SES_COLS, canonicalize, analysis_data, fit_validate,
    fairness_table, XGB_AVAILABLE,
)

st.set_page_config(page_title="Dental AI Coach – Research Prototype", layout="wide")
st.title("Dental AI Coach")
st.caption("Explainable machine-learning and rule-based oral-health decision-support research prototype")

# The root-level CSV contains the richer independent predictor set. The older
# data/ CSV is a reduced deployment copy and should not be used for the Q1 model.
DATA_PATH = "no_recommendation_dental_dataset_cleaned_keep_including_wisdom.csv"


@st.cache_data(show_spinner=False)
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset not found: {DATA_PATH}")
        st.stop()
    return canonicalize(pd.read_csv(DATA_PATH))


@st.cache_resource(show_spinner="Running audited five-fold validation...")
def train_cached(signature: str):
    return fit_validate(load_data())


def risk_tier(value, bins):
    q34, q67 = bins
    if value < q34:
        return "Low"
    if value < q67:
        return "Moderate"
    return "High"


def safe_int(value):
    x = pd.to_numeric(value, errors="coerce")
    return int(x) if pd.notna(x) else 0


def preventive_rules(row: dict, tier: str):
    """Clinician-editable, guideline-dependent prevention support.

    The prototype intentionally avoids hard-coding prescription-strength fluoride
    doses or drug regimens because those require age, jurisdiction, medical history
    and clinician judgment.
    """
    lines = []
    if tier == "High":
        lines.append("Consider a shorter risk-based recall interval and intensified professional prevention according to local guidelines.")
    elif tier == "Moderate":
        lines.append("Consider moderate-frequency recall with reinforcement of preventive measures according to local guidelines.")
    else:
        lines.append("Maintain routine risk-based recall unless examination findings indicate a shorter interval.")

    def txt(name):
        return str(row.get(name, "Unknown")).strip().lower()

    if any(k in txt("tooth_brushing_frequency") for k in ["never", "once", "1/day", "1-3"]):
        lines.append("Reinforce twice-daily brushing with age-appropriate fluoride toothpaste and individualized technique coaching.")
    if txt("interdental_cleaning") in {"no", "unknown"}:
        lines.append("Introduce an appropriate daily interdental-cleaning method where clinically suitable.")
    if any(k in txt("snacks_frequency") for k in ["daily", "3+", "often", "frequent", "more than once"]):
        lines.append("Reduce the frequency of between-meal cariogenic snacks and favor lower-cariogenic alternatives.")
    if any(k in txt("sugar") for k in ["daily", "twice", "frequent", "once a day"]):
        lines.append("Reduce the frequency of free-sugar exposure, particularly between meals.")
    if any(k in txt("carbonated_beverages") for k in ["daily", "twice", "frequent", "once/day"]):
        lines.append("Reduce frequent acidic/carbonated beverage exposure and favor water as the routine drink.")
    if "viscos" in txt("salivary_consistency") or "sticky" in txt("salivary_consistency"):
        lines.append("Review hydration and possible contributors to reduced salivary flow or increased viscosity.")
    if "acid" in txt("salivary_ph") or "low" in txt("buffering_capacity"):
        lines.append("Reinforce dietary acid control and fluoride-based prevention according to clinical indication.")
    if "more" in txt("mutans_load_in_saliva") or "more" in txt("lactobacilli_load_in_saliva"):
        lines.append("Reinforce plaque control and reduction in fermentable-carbohydrate frequency; adjunctive measures require clinical judgment.")
    return lines


def treatment_rules(row: dict):
    out = []
    n = lambda c: safe_int(row.get(c, 0))
    if n("decayed_1"):
        out.append(f"Caries recorded on {n('decayed_1')} tooth/teeth: perform lesion and pulpal assessment and provide appropriate caries management/restoration where indicated.")
    if n("filled_2"):
        out.append(f"Existing restorations: {n('filled_2')}; review integrity, margins, recurrent disease and maintenance needs.")
    if n("hypoplasia_3") or n("hypocalcification_4"):
        out.append("Developmental enamel defects are present; assess sensitivity, plaque retention, esthetic concerns and preventive/restorative needs.")
    if n("fluorosis_5"):
        out.append("Fluorosis is recorded; management should be based on severity, symptoms and esthetic concern.")
    if any(n(c) for c in ["erosion_6", "abrasion_7", "attrition_8", "abfraction_9"]):
        out.append("Non-carious tooth-surface loss is present; investigate etiologic factors before selecting preventive or restorative management.")
    if n("fractured_h"):
        out.append("Fractured tooth structure is recorded; perform clinical/radiographic assessment and provide protection or definitive restoration as indicated.")
    if n("missing_0_including_wisdom_"):
        out.append("Missing teeth are recorded; determine replacement need after considering third-molar status and the clinical context.")
    if n("crown_pontic") or n("crown_abutment"):
        out.append("Fixed dental prosthesis components are present; reinforce hygiene and assess abutment/pontic maintenance needs.")
    if n("crown_implant"):
        out.append("Implant-supported restoration is present; include peri-implant tissue assessment and maintenance.")
    if n("veneer_f"):
        out.append("Veneers are present; review margins, surface integrity, hygiene and maintenance.")
    return out or ["No treatment action was generated from the recorded component counts; clinical examination remains required."]


def transformed_feature_groups(pipe, original_cols):
    names = list(pipe.named_steps["pre"].get_feature_names_out())
    groups = {}
    for i, name in enumerate(names):
        if name.startswith("num__"):
            parent = name[len("num__"):]
        elif name.startswith("cat__"):
            raw = name[len("cat__"):]
            matches = [c for c in original_cols if raw == c or raw.startswith(c + "_")]
            parent = max(matches, key=len) if matches else raw
        else:
            parent = name
        groups.setdefault(parent, []).append(i)
    return groups


def grouped_shap(pipe, X, local=False, max_rows=250):
    import shap
    sample = X.iloc[[0]].copy() if local else (X.sample(min(max_rows, len(X)), random_state=42) if len(X) > max_rows else X.copy())
    Xt = pipe.named_steps["pre"].transform(sample)
    explainer = shap.TreeExplainer(pipe.named_steps["model"])
    sv = explainer.shap_values(Xt)
    if isinstance(sv, list):
        sv = sv[0]
    groups = transformed_feature_groups(pipe, list(X.columns))
    items = []
    for parent, idxs in groups.items():
        grouped = np.sum(sv[:, idxs], axis=1)
        value = float(grouped[0]) if local else float(np.mean(np.abs(grouped)))
        items.append((parent, value))
    return sorted(items, key=lambda x: abs(x[1]) if local else x[1], reverse=True)


df = load_data()
eligible_df, audit = analysis_data(df)
signature = f"{df.shape}-{','.join(df.columns)}-{audit['target_consistent'].sum()}"
res = train_cached(signature)
X = eligible_df[res.predictors].copy()
y = eligible_df[TARGET].astype(float).reset_index(drop=True)

st.warning(
    "Research prototype only. Elham's Index components are excluded from ML prediction because they mathematically define the outcome. "
    "Rows that fail the documented Elham arithmetic are excluded from supervised validation rather than silently corrected."
)

model_options = ["Random Forest"]
if XGB_AVAILABLE:
    model_options += ["XGBoost", "Blend"]
model_name = st.sidebar.selectbox("Model used for scenario prediction", model_options, index=0)


def predict(Xnew):
    if model_name == "Random Forest" or not XGB_AVAILABLE:
        return res.rf_final.predict(Xnew)
    if model_name == "XGBoost":
        return res.xgb_final.predict(Xnew)
    return 0.5 * (res.rf_final.predict(Xnew) + res.xgb_final.predict(Xnew))


tabs = st.tabs([
    "Data QC", "Study design", "Validated performance", "Explainable AI",
    "Patient scenario", "Fairness audit", "Rule-based care"
])

with tabs[0]:
    st.subheader("Outcome and dataset quality control")
    c1, c2, c3 = st.columns(3)
    c1.metric("Source rows", res.n_source)
    c2.metric("QC-eligible rows", res.n_eligible)
    c3.metric("Excluded rows", res.n_source - res.n_eligible)
    st.write(
        "Eligibility requires a stored Elham's Index, complete direct component counts, and equality between the stored index and the documented component sum."
    )
    failed = audit.loc[~audit["target_consistent"]].copy()
    if failed.empty:
        st.success("All rows passed the Elham arithmetic audit.")
    else:
        st.dataframe(failed, use_container_width=True)
        st.caption("These rows are retained in the source dataset for traceability but are not used to estimate model performance.")

with tabs[1]:
    st.subheader("Leakage-safe study design")
    st.write(f"Independent predictors used: {len(res.predictors)}")
    st.write("Validation: shuffled five-fold cross-validation with random seed 42; performance is calculated from pooled out-of-fold predictions.")
    st.write("Final deployment models are refitted only after validation on all QC-eligible observations.")
    st.write("School, current residence, place of birth and nationality are excluded from the primary prediction matrix to reduce site/geographic memorization.")
    with st.expander("Independent predictor list"):
        st.write(res.predictors)
    with st.expander("Target-derived variables excluded from ML"):
        st.write([c for c in TARGET_COMPONENTS if c in df.columns])

with tabs[2]:
    st.subheader("Out-of-fold predictive performance")
    metric_df = pd.DataFrame(res.metrics).T.reset_index().rename(columns={"index": "Model"})
    st.dataframe(metric_df.style.format({"R2": "{:.3f}", "MAE": "{:.3f}", "RMSE": "{:.3f}"}), use_container_width=True)
    st.caption("Mean and median baselines are displayed so ML performance is interpreted relative to simple non-ML prediction rules.")
    pred_col = "blend" if model_name == "Blend" and "blend" in res.oof else ("xgb" if model_name == "XGBoost" and "xgb" in res.oof else "rf")
    fig, ax = plt.subplots()
    ax.scatter(res.oof["observed"], res.oof[pred_col], alpha=0.65)
    lo = float(min(res.oof["observed"].min(), res.oof[pred_col].min()))
    hi = float(max(res.oof["observed"].max(), res.oof[pred_col].max()))
    ax.plot([lo, hi], [lo, hi])
    ax.set_xlabel("Observed Elham's Index")
    ax.set_ylabel("Out-of-fold prediction")
    ax.set_title(f"Observed vs predicted: {model_name}")
    st.pyplot(fig)
    st.caption("The previous R² values obtained when target components were allowed into the predictor matrix are not valid estimates of independent predictive performance.")

with tabs[3]:
    st.subheader("Explainable AI")
    st.write("SHAP describes how the fitted model uses predictors. It is an associational model explanation and does not establish causality.")
    explain_pipe = res.xgb_final if XGB_AVAILABLE and res.xgb_final is not None else res.rf_final
    try:
        top = grouped_shap(explain_pipe, X)[:15]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh([a for a, _ in top][::-1], [b for _, b in top][::-1])
        ax.set_xlabel("Mean absolute grouped SHAP value")
        ax.set_title("Global predictor importance")
        st.pyplot(fig)
        st.caption("Magnitude only. Larger bars indicate greater contribution to model predictions, not a harmful, protective or causal effect.")
    except Exception as exc:
        st.warning(f"SHAP could not be calculated: {exc}")

with tabs[4]:
    st.subheader("Patient-level model scenario")
    row_no = st.number_input("QC-eligible row", 0, max(0, len(X)-1), 0, 1)
    base = X.iloc[[int(row_no)]].copy()
    base_pred = float(predict(base)[0])
    st.metric("Observed Elham's Index", f"{float(y.iloc[int(row_no)]):.2f}")
    st.metric("Model prediction", f"{base_pred:.2f}")
    st.write(f"Model-estimated tier: **{risk_tier(base_pred, res.risk_bins)}**")

    explain_pipe = res.xgb_final if XGB_AVAILABLE and res.xgb_final is not None else res.rf_final
    try:
        loc = grouped_shap(explain_pipe, base, local=True)[:12]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh([a for a, _ in loc][::-1], [b for _, b in loc][::-1])
        ax.axvline(0)
        ax.set_xlabel("Grouped SHAP contribution")
        ax.set_title("Patient-level model explanation")
        st.pyplot(fig)
        st.caption("Positive values increase this model prediction relative to its baseline; negative values decrease it. This is not causal inference.")
    except Exception as exc:
        st.warning(f"Local SHAP could not be calculated: {exc}")

    st.markdown("#### What-if simulator")
    scenario = base.copy()
    mutable = [c for c in BEHAVIOR_COLS + SALIVARY_COLS if c in X.columns and not pd.api.types.is_numeric_dtype(X[c])]
    for c in mutable:
        choices = sorted(X[c].astype(str).dropna().unique().tolist())
        current = str(base.iloc[0][c])
        idx = choices.index(current) if current in choices else 0
        scenario.loc[scenario.index[0], c] = st.selectbox(c.replace("_", " ").title(), choices, index=idx, key=f"whatif_{c}")
    scenario_pred = float(predict(scenario)[0])
    st.metric("Scenario prediction", f"{scenario_pred:.2f}", delta=f"{scenario_pred-base_pred:+.2f}")
    st.caption("Scenario differences are model-based associations, not estimates of treatment effect or expected clinical benefit.")

with tabs[5]:
    st.subheader("Fairness screening")
    pred_col = "blend" if "blend" in res.oof.columns else "rf"
    fair = fairness_table(df, res.oof, pred_col=pred_col)
    if fair.empty:
        st.info("No predefined subgroup met the minimum sample size for this screen.")
    else:
        st.dataframe(fair, use_container_width=True)
    st.caption("A group is flagged only when n ≥20 and subgroup MAE is ≥1.5× overall out-of-fold MAE. This screen does not prove fairness or bias.")

with tabs[6]:
    st.subheader("Rule-based personalized prevention and treatment support")
    care_row = st.number_input("Source dataset row", 0, max(0, len(df)-1), 0, 1, key="care_row")
    row = df.iloc[int(care_row)].to_dict()
    model_row = pd.DataFrame([{c: df.iloc[int(care_row)].get(c, "Unknown") for c in res.predictors}])
    score = float(predict(model_row)[0])
    tier = risk_tier(score, res.risk_bins)
    st.write(f"Model-estimated tier: **{tier}**")
    st.markdown("#### Prevention support")
    for line in preventive_rules(row, tier):
        st.write("- " + line)
    st.markdown("#### Clinical rule layer")
    for line in treatment_rules(row):
        st.write("- " + line)
    st.caption("Recommendations are clinician-editable decision support. They do not replace examination, diagnosis, professional judgment or local clinical guidelines.")
