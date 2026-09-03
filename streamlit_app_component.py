import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from analysis_pipeline import canonicalize, analysis_data, BEHAVIOR_COLS, SALIVARY_COLS
from component_pipeline import (
    MODELED_COMPONENTS, DESCRIPTIVE_COMPONENTS, fit_all_components,
    clinical_profile,
)

st.set_page_config(page_title="Dental AI Coach – Personalized Oral Health", layout="wide")
st.title("Dental AI Coach")
st.caption("Detailed Elham clinical profile + explainable component-specific risk analysis + personalized action plan")

DEFAULT_DATA = "no_recommendation_dental_dataset_cleaned_keep_including_wisdom.csv"

# Allows the audited raw-data reconstruction to be supplied without changing
# the modeling code.  The branch default remains the older repository CSV until
# the deidentified 160-case raw reconstruction is intentionally added.
MASTER_TO_REPO = {
    "participant_id": "id",
    "missing_including_wisdom": "missing_0_including_wisdom_",
    "decayed": "decayed_1",
    "filled": "filled_2",
    "hypoplasia": "hypoplasia_3",
    "hypocalcification": "hypocalcification_4",
    "fluorosis": "fluorosis_5",
    "erosion": "erosion_6",
    "abrasion": "abrasion_7",
    "attrition": "attrition_8",
    "abfraction": "abfraction_9",
    "sealant": "sealant_a",
    "fractured": "fractured_h",
    "veneer": "veneer_f",
    "elham_index_including_wisdom": "elham_s_index_including_wisdom",
    "salivary_level_of_hydration": "level_of_hydration",
}


def normalize_uploaded(df):
    out = df.rename(columns={k: v for k, v in MASTER_TO_REPO.items() if k in df.columns}).copy()
    # Reconstruct the legacy target name only for QC compatibility.  The total
    # Elham score is not used as the component-model target.
    if "elham_s_index_including_wisdom" not in out.columns:
        direct = list(MODELED_COMPONENTS) + list(DESCRIPTIVE_COMPONENTS)
        if all(c in out.columns for c in direct):
            out["elham_s_index_including_wisdom"] = out[direct].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    return canonicalize(out)


@st.cache_data(show_spinner=False)
def load_default():
    if not os.path.exists(DEFAULT_DATA):
        st.error(f"Default dataset not found: {DEFAULT_DATA}")
        st.stop()
    return canonicalize(pd.read_csv(DEFAULT_DATA))


uploaded = st.sidebar.file_uploader("Optional audited 160-case CSV", type=["csv"])
if uploaded is not None:
    df = normalize_uploaded(pd.read_csv(uploaded))
    data_label = "Uploaded audited dataset"
else:
    df = load_default()
    data_label = "Repository dataset"

eligible, audit = analysis_data(df)

@st.cache_resource(show_spinner="Validating component-specific models...")
def train_components(signature, payload):
    return fit_all_components(payload)

signature = f"{df.shape}-{','.join(df.columns)}-{int(audit['target_consistent'].sum())}"
results = train_components(signature, df)

st.warning(
    "Research decision-support prototype. The current study is cross-sectional: model outputs describe internal predictive associations and risk patterns, not causal effects or future disease. Future forecasting requires longitudinal follow-up."
)

if uploaded is None:
    st.info(
        "The raw-data audit identified 160 genuine matched clinical/non-clinical participants. The repository's older processed CSV is used only as a temporary branch default. For manuscript analysis or production deployment, use the audited 160-case reconstruction."
    )

# ---------- helpers ----------
def safe_text(v):
    if pd.isna(v):
        return "Unknown"
    return str(v).strip()


def safe_num(v):
    x = pd.to_numeric(v, errors="coerce")
    return 0.0 if pd.isna(x) else float(x)


def model_prediction(result, row_df, model="Random Forest"):
    Xnew = row_df[result.predictors].copy()
    if model == "XGBoost" and result.xgb_final is not None:
        return max(0.0, float(result.xgb_final.predict(Xnew)[0]))
    if model == "Blend" and result.xgb_final is not None:
        a = float(result.rf_final.predict(Xnew)[0])
        b = float(result.xgb_final.predict(Xnew)[0])
        return max(0.0, 0.5 * (a + b))
    return max(0.0, float(result.rf_final.predict(Xnew)[0]))


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


def grouped_local_shap(pipe, Xrow, original_cols):
    import shap
    Xt = pipe.named_steps["pre"].transform(Xrow)
    explainer = shap.TreeExplainer(pipe.named_steps["model"])
    sv = explainer.shap_values(Xt)
    if isinstance(sv, list):
        sv = sv[0]
    groups = transformed_feature_groups(pipe, original_cols)
    items = []
    for parent, idxs in groups.items():
        contribution = float(np.sum(sv[0, idxs]))
        items.append((parent, contribution))
    return sorted(items, key=lambda x: abs(x[1]), reverse=True)


def recommendation_engine(row):
    advice = []
    priority = []

    decay = safe_num(row.get("decayed_1"))
    missing = safe_num(row.get("missing_0_including_wisdom_"))
    filled = safe_num(row.get("filled_2"))
    hypo = safe_num(row.get("hypocalcification_4"))

    if decay > 0:
        priority.append(f"Caries burden: {int(decay)} decayed tooth/teeth")
        advice.append("Complete lesion activity, cavitation and pulpal assessment; provide caries control and restorative care according to clinical findings.")
    if missing > 0:
        priority.append(f"Missing teeth: {int(missing)} including wisdom teeth")
        advice.append("Verify the reason for each missing tooth and separate developmental/third-molar status from disease-related tooth loss before planning replacement or prevention.")
    if filled > 0:
        priority.append(f"Existing restorations: {int(filled)}")
        advice.append("Review existing restorations for integrity, margins, recurrent disease and maintenance needs.")
    if hypo > 0:
        priority.append(f"Hypocalcified teeth: {int(hypo)}")
        advice.append("Assess developmental enamel defects clinically for sensitivity, plaque retention, esthetic concern and preventive/restorative need.")

    def txt(c):
        return safe_text(row.get(c, "Unknown")).lower()

    if any(k in txt("tooth_brushing_frequency") for k in ["never", "once", "1/day", "1-3"]):
        advice.append("Oral hygiene priority: reinforce twice-daily toothbrushing with age-appropriate fluoride toothpaste and individualized technique coaching.")
    if txt("interdental_cleaning") in {"no", "unknown"}:
        advice.append("Interdental care: introduce a suitable daily interdental-cleaning method when clinically appropriate.")
    if any(k in txt("sugar") for k in ["daily", "frequent", "twice", "once a day"]):
        advice.append("Diet priority: reduce the frequency of free-sugar exposure, particularly between meals.")
    if any(k in txt("snacks_frequency") for k in ["daily", "often", "frequent", "3+"]):
        advice.append("Snack pattern: reduce frequent cariogenic between-meal snacks and favor lower-cariogenic alternatives.")
    carb = txt("carbonated_beverages") + " " + txt("carbonated_beverages_diet")
    if any(k in carb for k in ["daily", "frequent", "once/day", "twice"]):
        advice.append("Acid exposure: reduce frequent carbonated/acidic beverages and favor water as the routine drink.")
    if "low" in txt("buffering_capacity") or "acid" in txt("salivary_ph"):
        advice.append("Salivary risk: review hydration, diet and clinically indicated preventive measures because low buffering/acidic saliva may increase vulnerability.")
    if "more" in txt("mutans_load_in_saliva") or "more" in txt("lactobacilli_load_in_saliva"):
        advice.append("Microbial risk: intensify plaque control and reduce fermentable-carbohydrate frequency; adjunctive measures require clinician judgment.")
    if not advice:
        advice.append("Maintain routine risk-based prevention and recall, modified according to direct clinical findings.")
    return priority, advice


def overall_component_table(row):
    profile = clinical_profile(row)
    profile["status"] = np.where(profile["count"] > 0, "Present", "Not recorded")
    return profile


# ---------- patient selector ----------
id_col = "id" if "id" in eligible.columns else None
selector_values = eligible[id_col].tolist() if id_col else list(range(len(eligible)))
selected = st.sidebar.selectbox("Patient / participant", selector_values)
if id_col:
    patient = eligible.loc[eligible[id_col] == selected].iloc[0]
else:
    patient = eligible.iloc[int(selected)]
patient_df = pd.DataFrame([patient])

model_choice = st.sidebar.selectbox("Component model", ["Random Forest", "XGBoost", "Blend"])

# ---------- tabs ----------
tabs = st.tabs([
    "Oral-health profile", "Component-specific AI", "Why this patient?",
    "Personalized action plan", "Validation", "Research design"
])

with tabs[0]:
    st.subheader("Detailed Elham oral-health profile")
    st.write("The clinical examination is displayed as separate findings rather than one total score.")
    profile = overall_component_table(patient)
    st.dataframe(profile, use_container_width=True, hide_index=True)

    present = profile.loc[profile["count"] > 0].copy()
    if not present.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(present["component"], present["count"])
        ax.set_xlabel("Number of teeth / recorded units")
        ax.set_title("Patient clinical profile")
        st.pyplot(fig)

with tabs[1]:
    st.subheader("Component-specific risk estimation")
    st.write("Each sufficiently common Elham component is analyzed separately from independent patient factors.")
    rows = []
    for target, result in results.items():
        observed = safe_num(patient.get(target))
        pred = model_prediction(result, patient_df, model_choice)
        rows.append({
            "Clinical component": result.label,
            "Observed count": observed,
            "Model-estimated count": pred,
            "Cohort prevalence": result.prevalence,
        })
    comp = pd.DataFrame(rows)
    st.dataframe(comp.style.format({"Observed count":"{:.0f}", "Model-estimated count":"{:.2f}", "Cohort prevalence":"{:.1%}"}), use_container_width=True)
    st.caption("Observed counts come from the dental examination. Model estimates are not a substitute for examination and are not forecasts of future disease.")

with tabs[2]:
    st.subheader("Model explanation for this patient")
    selected_target = st.selectbox("Clinical component to explain", list(MODELED_COMPONENTS), format_func=lambda x: MODELED_COMPONENTS[x])
    result = results[selected_target]
    pipe = result.xgb_final if model_choice in {"XGBoost", "Blend"} and result.xgb_final is not None else result.rf_final
    try:
        local = grouped_local_shap(pipe, patient_df[result.predictors], result.predictors)[:12]
        exp = pd.DataFrame(local, columns=["Patient factor", "Model contribution"])
        st.dataframe(exp, use_container_width=True, hide_index=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(exp["Patient factor"][::-1], exp["Model contribution"][::-1])
        ax.axvline(0)
        ax.set_xlabel("Grouped SHAP contribution")
        ax.set_title(f"Factors influencing the model estimate: {result.label}")
        st.pyplot(fig)
        st.caption("Positive/negative SHAP values show how the fitted model uses this patient's variables. They do not prove that changing a factor will cause the clinical outcome to change.")
    except Exception as exc:
        st.warning(f"Model explanation unavailable: {exc}")

with tabs[3]:
    st.subheader("Personalized oral-health action plan")
    priority, advice = recommendation_engine(patient.to_dict())
    if priority:
        st.markdown("#### Clinical priorities")
        for item in priority:
            st.write(f"• {item}")
    st.markdown("#### Tailored preventive and clinical recommendations")
    for item in advice:
        st.write(f"• {item}")
    st.info("The action plan is clinician-reviewable decision support, not an autonomous prescription. Diagnosis, treatment selection, dose/frequency of any therapeutic agent, and recall interval require professional judgment and applicable clinical guidance.")

with tabs[4]:
    st.subheader("Internal validation by clinical component")
    tables = []
    for target, result in results.items():
        for model, md in result.metrics.items():
            tables.append({"Component":result.label, "Model":model, **md})
    perf = pd.DataFrame(tables)
    st.dataframe(perf.style.format({"R2":"{:.3f}", "MAE":"{:.3f}", "RMSE":"{:.3f}"}), use_container_width=True, hide_index=True)
    st.caption("Five-fold shuffled cross-validation is compared with simple mean/median baselines. Modest or negative R² values must not be described as high predictive accuracy.")

with tabs[5]:
    st.subheader("What the current study is designed to answer")
    st.markdown(
        "**Current study:** Which demographic, socioeconomic, behavioral, dietary and salivary factors are associated with individual components of the detailed Elham oral-health profile, and how can these data support a personalized preventive and clinical action plan?"
    )
    st.markdown(
        "**Not yet answered:** What will happen to this patient's mouth in 6–12 months? Genuine forecasting requires longitudinal follow-up with repeat clinical assessment."
    )
    st.write(f"Dataset shown: {data_label}. Source rows: {len(df)}. Arithmetic-QC eligible rows: {len(eligible)}.")
    st.write("Rare Elham findings remain visible in the patient's clinical profile but are not given separate ML models when the cohort contains too few events.")
