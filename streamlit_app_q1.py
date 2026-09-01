import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from analysis_pipeline import (
    TARGET, TARGET_COMPONENTS, BEHAVIOR_COLS, SALIVARY_COLS, SES_COLS,
    canonicalize, fit_validate, fairness_table, XGB_AVAILABLE,
)

st.set_page_config(page_title="Dental AI Coach", layout="wide")
st.title("Dental AI Coach")
st.caption("Leakage-safe explainable AI research prototype for individualized oral-health risk assessment")

DATA_PATH = "data/no_recommendation_dental_dataset_cleaned_keep_including_wisdom.csv"

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_PATH)
    return canonicalize(df)

@st.cache_resource(show_spinner="Training leakage-safe validation models...")
def train_cached(csv_signature: str):
    df = load_data()
    return fit_validate(df)


def risk_tier(value, bins):
    q34, q67 = bins
    if value < q34:
        return "Low"
    if value < q67:
        return "Moderate"
    return "High"


def preventive_rules(row: dict, tier: str):
    lines = []
    if tier == "High":
        lines.append("Shorter risk-based recall and intensified professional prevention should be considered according to local clinical guidelines.")
    elif tier == "Moderate":
        lines.append("Moderate-frequency recall and reinforcement of preventive measures should be considered according to local clinical guidelines.")
    else:
        lines.append("Routine risk-based recall and maintenance of effective preventive habits are appropriate unless clinical findings indicate otherwise.")

    def txt(name):
        return str(row.get(name, "Unknown")).lower()

    if "never" in txt("tooth_brushing_frequency") or "once" in txt("tooth_brushing_frequency") or "1/day" in txt("tooth_brushing_frequency"):
        lines.append("Reinforce twice-daily toothbrushing with an age-appropriate fluoride toothpaste and individualized technique coaching.")
    if txt("interdental_cleaning") in {"no", "unknown"}:
        lines.append("Introduce an appropriate daily interdental-cleaning method where clinically suitable.")
    if any(k in txt("snacks_frequency") for k in ["daily", "3+", "often", "frequent"]):
        lines.append("Reduce the frequency of between-meal cariogenic snacks and favor lower-cariogenic alternatives.")
    if any(k in txt("sugar") for k in ["daily", "twice", "frequent", "once a day"]):
        lines.append("Reduce the frequency of free-sugar exposure, particularly between meals.")
    if any(k in txt("carbonated_beverages") for k in ["daily", "twice", "frequent", "once/day"]):
        lines.append("Reduce frequent acidic/carbonated beverage exposure and favor water as the routine drink.")
    if "increased" in txt("salivary_consistency") or "viscos" in txt("salivary_consistency"):
        lines.append("Review hydration and factors associated with reduced salivary flow or increased viscosity.")
    if "acid" in txt("salivary_ph") or "low" in txt("buffering_capacity"):
        lines.append("Reinforce dietary acid control and fluoride-based prevention according to clinical indication.")
    if "more" in txt("mutans_load_in_saliva") or "more" in txt("lactobacilli_load_in_saliva"):
        lines.append("Reinforce plaque control and reduction in fermentable-carbohydrate frequency; adjunctive measures require clinical judgment.")
    return lines


def treatment_rules(row: dict):
    out = []
    n = lambda c: int(pd.to_numeric(row.get(c, 0), errors="coerce") or 0)
    if n("decayed_1"):
        out.append(f"Caries recorded on {n('decayed_1')} tooth/teeth: perform lesion and pulpal assessment and provide appropriate caries management/restoration where indicated.")
    if n("filled_2"):
        out.append(f"Existing restorations: {n('filled_2')}; review integrity, margins, recurrent disease, and maintenance needs.")
    if n("hypoplasia_3") or n("hypocalcification_4"):
        out.append("Enamel developmental defects are present; assess sensitivity, plaque retention, esthetic concerns, and need for preventive/restorative management.")
    if n("fluorosis_5"):
        out.append("Fluorosis is recorded; management should be based on severity, symptoms, and esthetic concern.")
    if n("erosion_6") or n("abrasion_7") or n("attrition_8") or n("abfraction_9"):
        out.append("Non-carious tooth surface loss is present; investigate etiologic factors before selecting preventive or restorative management.")
    if n("fractured_h"):
        out.append("Fractured tooth structure is recorded; perform clinical/radiographic assessment and provide protection or definitive restoration as indicated.")
    if n("missing_0_including_wisdom_"):
        out.append("Missing teeth are recorded; determine whether replacement is required after excluding clinically irrelevant third-molar absence.")
    if n("crown_pontic") or n("crown_abutment"):
        out.append("Fixed dental prosthesis components are present; reinforce hygiene and assess abutment/pontic maintenance needs.")
    if n("crown_implant"):
        out.append("Implant-supported restoration is present; include peri-implant tissue assessment and maintenance.")
    if n("veneer_f"):
        out.append("Veneers are present; review margins, surface integrity, hygiene, and maintenance.")
    return out or ["No treatment action was generated from the recorded Elham component counts; clinical examination remains required."]


def transformed_feature_groups(pipe):
    pre = pipe.named_steps["pre"]
    names = list(pre.get_feature_names_out())
    groups = {}
    for i, name in enumerate(names):
        if name.startswith("num__"):
            parent = name.replace("num__", "", 1)
        elif name.startswith("cat__"):
            raw = name.replace("cat__", "", 1)
            parent = raw
            # recover the longest matching original categorical column name
            candidates = [c for c in pipe.feature_names_in_ if raw == c or raw.startswith(c + "_")]
            if candidates:
                parent = max(candidates, key=len)
        else:
            parent = name
        groups.setdefault(parent, []).append(i)
    return names, groups


def global_shap(pipe, X, max_rows=250):
    import shap
    sample = X.sample(min(max_rows, len(X)), random_state=42) if len(X) > max_rows else X.copy()
    Xt = pipe.named_steps["pre"].transform(sample)
    explainer = shap.TreeExplainer(pipe.named_steps["model"])
    sv = explainer.shap_values(Xt)
    if isinstance(sv, list):
        sv = sv[0]
    names, groups = transformed_feature_groups(pipe)
    values = []
    for parent, idxs in groups.items():
        grouped = np.sum(sv[:, idxs], axis=1)
        values.append((parent, float(np.mean(np.abs(grouped)))))
    return sorted(values, key=lambda x: x[1], reverse=True)


def local_shap(pipe, row):
    import shap
    Xt = pipe.named_steps["pre"].transform(row)
    explainer = shap.TreeExplainer(pipe.named_steps["model"])
    sv = explainer.shap_values(Xt)
    if isinstance(sv, list):
        sv = sv[0]
    names, groups = transformed_feature_groups(pipe)
    values = []
    for parent, idxs in groups.items():
        values.append((parent, float(np.sum(sv[0, idxs]))))
    return sorted(values, key=lambda x: abs(x[1]), reverse=True)


df = load_data()
signature = f"{df.shape}-{','.join(df.columns)}"
res = train_cached(signature)
X = df.loc[df[TARGET].notna(), res.predictors].reset_index(drop=True)
y = df.loc[df[TARGET].notna(), TARGET].astype(float).reset_index(drop=True)

st.info(
    "Scientific safeguard: variables that mathematically compose Elham's Index (decay, fillings, missing teeth, enamel defects, wear, fractures, prosthetic components, DMF, treatment index, and sound-teeth count) are excluded from the machine-learning predictor set. They are used only in the rule-based clinical layer."
)

model_name = st.sidebar.selectbox("Deployment model", ["Blend", "XGBoost", "Random Forest"] if XGB_AVAILABLE else ["Random Forest"])

if model_name == "Random Forest":
    deploy_pipe = res.rf_final
elif model_name == "XGBoost":
    deploy_pipe = res.xgb_final
else:
    deploy_pipe = None


def predict(Xnew):
    if model_name == "Random Forest" or not XGB_AVAILABLE:
        return res.rf_final.predict(Xnew)
    if model_name == "XGBoost":
        return res.xgb_final.predict(Xnew)
    return 0.5 * (res.rf_final.predict(Xnew) + res.xgb_final.predict(Xnew))


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Study design", "Validated performance", "Explainable AI", "Patient scenario", "Fairness audit", "Rule-based care"
])

with tab1:
    st.subheader("Leakage-safe study design")
    st.write(f"Participants available for modeling: {len(y)}")
    st.write(f"Predictors used: {len(res.predictors)}")
    st.write("Validation: five-fold cross-validation with shuffling and random seed 42; all reported performance values are based on out-of-fold predictions.")
    st.write("Deployment: models are refitted on the complete dataset only after validation.")
    st.write("School, place of birth, and nationality are excluded from the prediction model to reduce site/geographic memorization.")
    with st.expander("Predictor list"):
        st.write(res.predictors)
    with st.expander("Excluded target-derived variables"):
        st.write([c for c in TARGET_COMPONENTS if c in df.columns])

with tab2:
    st.subheader("Out-of-fold predictive performance")
    metric_df = pd.DataFrame(res.metrics).T.reset_index().rename(columns={"index": "Model"})
    st.dataframe(metric_df.style.format({"R2": "{:.3f}", "MAE": "{:.3f}", "RMSE": "{:.3f}"}), use_container_width=True)
    st.caption("These values intentionally replace the earlier leakage-prone results. They should be used in the revised manuscript only after this branch is executed on the full study dataset.")
    pred_col = "blend" if model_name == "Blend" and "blend" in res.oof else ("xgb" if model_name == "XGBoost" and "xgb" in res.oof else "rf")
    fig, ax = plt.subplots()
    ax.scatter(res.oof["observed"], res.oof[pred_col], alpha=0.65)
    lo = min(res.oof["observed"].min(), res.oof[pred_col].min())
    hi = max(res.oof["observed"].max(), res.oof[pred_col].max())
    ax.plot([lo, hi], [lo, hi])
    ax.set_xlabel("Observed Elham's Index")
    ax.set_ylabel("Out-of-fold prediction")
    ax.set_title(f"Observed vs predicted: {model_name}")
    st.pyplot(fig)

with tab3:
    st.subheader("Explainable AI")
    st.write("SHAP is used to describe model behavior. SHAP values are associational model explanations and must not be interpreted as causal effects.")
    if not XGB_AVAILABLE:
        st.warning("Install xgboost and shap to enable the preferred XAI view.")
    else:
        explain_pipe = res.xgb_final
        try:
            imp = global_shap(explain_pipe, X)
            top = imp[:15]
            fig, ax = plt.subplots(figsize=(8, 5))
            labels = [a for a, _ in top][::-1]
            vals = [b for _, b in top][::-1]
            ax.barh(labels, vals)
            ax.set_xlabel("Mean absolute grouped SHAP value")
            ax.set_title("Global predictor importance")
            st.pyplot(fig)
            st.caption("Magnitude only: this plot does not indicate whether a factor is harmful, protective, or causal.")
        except Exception as e:
            st.warning(f"SHAP visualization could not be calculated: {e}")

with tab4:
    st.subheader("Patient-level prediction and what-if analysis")
    selected = st.number_input("Dataset row number", min_value=0, max_value=max(0, len(X)-1), value=0, step=1)
    base = X.iloc[[int(selected)]].copy()
    observed = float(y.iloc[int(selected)])
    base_pred = float(predict(base)[0])
    st.metric("Observed Elham's Index", f"{observed:.2f}")
    st.metric("Predicted Elham's Index", f"{base_pred:.2f}")
    st.write(f"Predicted risk tier: **{risk_tier(base_pred, res.risk_bins)}**")

    st.markdown("#### Local explanation")
    if XGB_AVAILABLE:
        try:
            loc = local_shap(res.xgb_final, base)[:12]
            fig, ax = plt.subplots(figsize=(8, 5))
            labels = [a for a, _ in loc][::-1]
            vals = [b for _, b in loc][::-1]
            ax.barh(labels, vals)
            ax.axvline(0)
            ax.set_xlabel("Grouped SHAP contribution")
            ax.set_title("Patient-level model explanation")
            st.pyplot(fig)
            st.caption("Positive values push this model prediction upward; negative values push it downward. They do not prove causation.")
        except Exception as e:
            st.warning(f"Local SHAP could not be calculated: {e}")

    st.markdown("#### What-if simulator")
    mutable = [c for c in BEHAVIOR_COLS + SALIVARY_COLS if c in X.columns and not pd.api.types.is_numeric_dtype(X[c])]
    scenario = base.copy()
    for c in mutable:
        vals = sorted(X[c].astype(str).dropna().unique().tolist())
        current = str(base.iloc[0][c])
        idx = vals.index(current) if current in vals else 0
        scenario.loc[scenario.index[0], c] = st.selectbox(c.replace("_", " ").title(), vals, index=idx, key=f"whatif_{c}")
    new_pred = float(predict(scenario)[0])
    st.metric("Scenario prediction", f"{new_pred:.2f}", delta=f"{new_pred-base_pred:+.2f}")
    st.caption("The what-if difference is a model-based association, not an estimate of treatment effect or causal benefit.")

with tab5:
    st.subheader("Fairness screening")
    pred_col = "blend" if "blend" in res.oof.columns else "rf"
    fair = fairness_table(df, res.oof, pred_col=pred_col)
    if fair.empty:
        st.info("No eligible socioeconomic subgroups met the minimum sample size for this screen.")
    else:
        st.dataframe(fair, use_container_width=True)
        st.caption("A subgroup is flagged for review when n ≥ 20 and subgroup MAE is ≥1.5 times the overall out-of-fold MAE. This is a screening rule and does not establish absence or presence of algorithmic bias.")

with tab6:
    st.subheader("Rule-based personalized prevention and treatment")
    row_idx = st.number_input("Clinical row number", min_value=0, max_value=max(0, len(df)-1), value=0, step=1, key="care_row")
    row = df.iloc[int(row_idx)].to_dict()
    model_row = pd.DataFrame([{c: df.iloc[int(row_idx)].get(c, "Unknown") for c in res.predictors}])
    score = float(predict(model_row)[0])
    tier = risk_tier(score, res.risk_bins)
    st.write(f"Model-estimated tier: **{tier}**")
    st.markdown("#### Prevention plan")
    for line in preventive_rules(row, tier):
        st.write("- " + line)
    st.markdown("#### Clinical rule layer")
    for line in treatment_rules(row):
        st.write("- " + line)
    st.warning("Research decision-support prototype. Recommendations require clinician review and adaptation to examination findings, patient needs, local guidelines, contraindications, and regulatory requirements.")
