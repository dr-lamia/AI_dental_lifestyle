import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from analysis_pipeline import canonicalize, analysis_data
from component_pipeline import MODELED_COMPONENTS, fit_all_components, clinical_profile
from data.master160_embedded import load_master160

st.set_page_config(page_title="Dental AI Coach – Audited Research Prototype", layout="wide")
st.title("Dental AI Coach")
st.caption("Detailed Elham oral-health profile, component-specific AI analysis, and personalized action planning")

@st.cache_data(show_spinner=False)
def load_data():
    return canonicalize(load_master160())

@st.cache_resource(show_spinner="Validating component-specific models on the audited cohort...")
def train(signature, data):
    return fit_all_components(data)


def safe_num(v):
    x = pd.to_numeric(v, errors="coerce")
    return 0.0 if pd.isna(x) else float(x)


def safe_text(v):
    return "Unknown" if pd.isna(v) else str(v).strip()


def predict_component(result, row, model_name):
    X = pd.DataFrame([row])[result.predictors]
    if model_name == "XGBoost" and result.xgb_final is not None:
        return max(0.0, float(result.xgb_final.predict(X)[0]))
    if model_name == "Blend" and result.xgb_final is not None:
        a = float(result.rf_final.predict(X)[0])
        b = float(result.xgb_final.predict(X)[0])
        return max(0.0, 0.5 * (a + b))
    return max(0.0, float(result.rf_final.predict(X)[0]))


def action_plan(row):
    priorities, advice = [], []
    decay = safe_num(row.get("decayed_1"))
    missing = safe_num(row.get("missing_0_including_wisdom_"))
    filled = safe_num(row.get("filled_2"))
    hypo = safe_num(row.get("hypocalcification_4"))

    if decay > 0:
        priorities.append(f"Caries burden: {int(decay)} decayed tooth/teeth")
        advice.append("Assess lesion activity, cavitation and pulpal status; provide caries control and restorative care according to the examination.")
    if missing > 0:
        priorities.append(f"Missing teeth including wisdom teeth: {int(missing)}")
        advice.append("Verify why each tooth is missing and distinguish third-molar/developmental absence from disease-related tooth loss before planning treatment.")
    if filled > 0:
        priorities.append(f"Existing restorations: {int(filled)}")
        advice.append("Review restorations for integrity, margins, recurrent disease and maintenance needs.")
    if hypo > 0:
        priorities.append(f"Hypocalcified teeth: {int(hypo)}")
        advice.append("Assess developmental enamel defects for sensitivity, plaque retention, esthetic concern and preventive/restorative need.")

    def txt(c):
        return safe_text(row.get(c, "Unknown")).lower()

    if any(k in txt("tooth_brushing_frequency") for k in ["never", "once", "1/day", "1-3"]):
        advice.append("Oral-hygiene priority: reinforce twice-daily toothbrushing with age-appropriate fluoride toothpaste and individualized technique coaching.")
    if txt("interdental_cleaning") in {"no", "unknown"}:
        advice.append("Introduce a suitable daily interdental-cleaning method where clinically appropriate.")
    if any(k in txt("sugar") for k in ["daily", "frequent", "twice", "once a day"]):
        advice.append("Reduce the frequency of free-sugar exposure, especially between meals.")
    if any(k in txt("snacks_frequency") for k in ["daily", "often", "frequent", "3+"]):
        advice.append("Reduce frequent cariogenic between-meal snacks and favor lower-cariogenic alternatives.")
    carbonated = txt("carbonated_beverages") + " " + txt("carbonated_beverages_diet")
    if any(k in carbonated for k in ["daily", "frequent", "once/day", "twice"]):
        advice.append("Reduce frequent carbonated/acidic beverage exposure and favor water as the routine drink.")
    if "low" in txt("buffering_capacity") or "acid" in txt("salivary_ph"):
        advice.append("Review hydration, dietary acid exposure and clinically indicated preventive measures because the salivary profile may increase vulnerability.")
    if "more" in txt("mutans_load_in_saliva") or "more" in txt("lactobacilli_load_in_saliva"):
        advice.append("Intensify plaque control and reduce fermentable-carbohydrate frequency; adjunctive measures require clinician judgment.")
    if not advice:
        advice.append("Maintain routine risk-based prevention and recall, modified according to the direct clinical examination.")
    return priorities, advice


df = load_data()
eligible, audit = analysis_data(df)
results = train(f"{df.shape}-{int(audit['target_consistent'].sum())}", df)

st.success(f"Audited raw-data cohort loaded: {len(df)} matched participants; {int(audit['target_consistent'].sum())} passed Elham arithmetic QC.")
st.warning("Research decision-support prototype. Current data are cross-sectional: results describe internal predictive associations and do not prove causation or forecast future disease. Longitudinal follow-up is required for genuine future prediction.")

patient_id = st.sidebar.selectbox("Participant", eligible["id"].tolist())
model_name = st.sidebar.selectbox("Model", ["Random Forest", "XGBoost", "Blend"])
patient = eligible.loc[eligible["id"] == patient_id].iloc[0]

profile_tab, ai_tab, plan_tab, validation_tab, design_tab = st.tabs([
    "Detailed oral-health profile", "Component-specific AI", "Personalized action plan", "Validation", "Study meaning"
])

with profile_tab:
    st.subheader("Detailed Elham clinical profile")
    profile = clinical_profile(patient)
    profile["status"] = np.where(profile["count"] > 0, "Present", "Not recorded")
    st.dataframe(profile, use_container_width=True, hide_index=True)
    shown = profile.loc[profile["count"] > 0]
    if not shown.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(shown["component"], shown["count"])
        ax.set_xlabel("Number of teeth / recorded units")
        ax.set_title("Clinical oral-health profile")
        st.pyplot(fig)
    st.caption("The total Elham score is not used as the main AI target here. The detailed component profile is retained because it contains more clinically useful information.")

with ai_tab:
    st.subheader("Component-specific AI estimates")
    rows = []
    for target, result in results.items():
        rows.append({
            "Clinical component": result.label,
            "Observed clinical count": safe_num(patient.get(target)),
            "AI-estimated count": predict_component(result, patient.to_dict(), model_name),
            "Cohort prevalence": result.prevalence,
        })
    table = pd.DataFrame(rows)
    st.dataframe(table.style.format({"Observed clinical count":"{:.0f}", "AI-estimated count":"{:.2f}", "Cohort prevalence":"{:.1%}"}), use_container_width=True, hide_index=True)
    st.caption("The observed value comes from the dental examination. The AI estimate shows how much information the non-index factors contain about that component; it does not replace examination.")

with plan_tab:
    st.subheader("Personalized oral-health action plan")
    priorities, advice = action_plan(patient.to_dict())
    if priorities:
        st.markdown("#### Clinical priorities")
        for x in priorities:
            st.write(f"• {x}")
    st.markdown("#### Tailored recommendations")
    for x in advice:
        st.write(f"• {x}")
    st.info("This is clinician-reviewable decision support, not an autonomous prescription. Diagnosis, treatment choice, therapeutic dosing and recall intervals require professional judgment and applicable guidelines.")

with validation_tab:
    st.subheader("Internal validation")
    rows = []
    for _, result in results.items():
        for model, md in result.metrics.items():
            rows.append({"Component": result.label, "Model": model, **md})
    perf = pd.DataFrame(rows)
    st.dataframe(perf.style.format({"R2":"{:.3f}", "MAE":"{:.3f}", "RMSE":"{:.3f}"}), use_container_width=True, hide_index=True)
    st.caption("Five-fold out-of-fold results are displayed with mean and median baselines. Modest or negative R² must not be described as high predictive accuracy.")

with design_tab:
    st.subheader("What this study is trying to do")
    st.write("The dentist records the patient's detailed Elham oral-health profile. The AI then examines demographic, socioeconomic, behavioral, dietary and salivary information in relation to individual clinical components that are common enough to analyze.")
    st.write("The aim is to identify patient-specific risk patterns and support a tailored preventive and clinical action plan, rather than to reconstruct one total Elham score from pieces of that same score.")
    st.write("Rare clinical findings remain part of the patient's Elham profile but are not given separate ML models when too few participants have the condition.")
    st.write("Future forecasting is a second phase: repeat the clinical examination at follow-up and model change in each component from the baseline profile and baseline risk factors.")
