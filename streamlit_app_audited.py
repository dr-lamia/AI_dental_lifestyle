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


def local_model_factors(result, row, model_name, top_n=6):
    """Return patient-level grouped SHAP factors for model review, not causal effects."""
    try:
        import shap
        pipe = result.xgb_final if model_name in {"XGBoost", "Blend"} and result.xgb_final is not None else result.rf_final
        Xrow = pd.DataFrame([row])[result.predictors]
        Xt = pipe.named_steps["pre"].transform(Xrow)
        explainer = shap.TreeExplainer(pipe.named_steps["model"])
        sv = explainer.shap_values(Xt)
        if isinstance(sv, list):
            sv = sv[0]
        groups = transformed_feature_groups(pipe, result.predictors)
        items = []
        for parent, idxs in groups.items():
            contribution = float(np.sum(sv[0, idxs]))
            items.append((parent, contribution))
        return sorted(items, key=lambda x: abs(x[1]), reverse=True)[:top_n]
    except Exception:
        return []


def action_plan(row):
    """Clinician-reviewable plan combining direct findings with modifiable inputs.

    This layer does not infer causality from the ML model and intentionally avoids
    drug doses or fixed recall intervals.
    """
    clinical_priorities, modifiable_factors, recommendations = [], [], []

    decay = safe_num(row.get("decayed_1"))
    missing = safe_num(row.get("missing_0_including_wisdom_"))
    filled = safe_num(row.get("filled_2"))
    hypo = safe_num(row.get("hypocalcification_4"))
    fractured = safe_num(row.get("fractured_h"))
    erosion = safe_num(row.get("erosion_6"))
    attrition = safe_num(row.get("attrition_8"))
    abrasion = safe_num(row.get("abrasion_7"))
    abfraction = safe_num(row.get("abfraction_9"))

    if decay > 0:
        clinical_priorities.append(f"Caries burden: {int(decay)} decayed tooth/teeth")
        recommendations.append("Assess lesion activity, cavitation and pulpal status; provide caries control and restorative care according to the examination.")
    if filled > 0:
        clinical_priorities.append(f"Existing restorations: {int(filled)}")
        recommendations.append("Review restorations for integrity, margins, recurrent disease and maintenance needs.")
    if hypo > 0:
        clinical_priorities.append(f"Hypocalcified teeth: {int(hypo)}")
        recommendations.append("Assess developmental enamel defects for sensitivity, plaque retention, esthetic concern and preventive/restorative need.")
    if fractured > 0:
        clinical_priorities.append(f"Fractured teeth: {int(fractured)}")
        recommendations.append("Assess fractured teeth clinically and radiographically as indicated and determine protection or definitive treatment needs.")
    if any(v > 0 for v in [erosion, attrition, abrasion, abfraction]):
        clinical_priorities.append("Non-carious tooth-surface loss recorded")
        recommendations.append("Investigate the likely etiology of tooth-surface loss before selecting preventive or restorative management.")
    if missing > 0:
        clinical_priorities.append(f"Teeth recorded as missing including wisdom teeth: {int(missing)}")
        recommendations.append("Interpret the missing-tooth count cautiously: this variable includes wisdom teeth. Verify eruption/developmental status and distinguish third molars from disease-related tooth loss before assigning treatment need.")

    def txt(c):
        return safe_text(row.get(c, "Unknown")).lower()

    brushing = txt("tooth_brushing_frequency")
    if any(k in brushing for k in ["never", "once/day", "once a day", "once"]):
        modifiable_factors.append("Suboptimal brushing frequency")
        recommendations.append("Oral-hygiene priority: reinforce twice-daily toothbrushing with age-appropriate fluoride toothpaste and individualized technique coaching.")

    interdental = txt("interdental_cleaning")
    if interdental.startswith("no"):
        modifiable_factors.append("No reported interdental cleaning")
        recommendations.append("Introduce a suitable daily interdental-cleaning method where clinically appropriate.")

    sugar = txt("sugar")
    if any(k in sugar for k in ["daily", "frequent", "twice", "once a day"]):
        modifiable_factors.append("Frequent free-sugar exposure")
        recommendations.append("Reduce the frequency of free-sugar exposure, especially between meals.")

    snacks = txt("snacks_frequency")
    if any(k in snacks for k in ["daily", "often", "frequent", "3+"]):
        modifiable_factors.append("Frequent between-meal snacking")
        recommendations.append("Reduce frequent cariogenic between-meal snacks and favor lower-cariogenic alternatives.")

    carbonated = txt("carbonated_beverages") + " " + txt("carbonated_beverages_diet")
    if any(k in carbonated for k in ["daily", "frequent", "once/day", "twice"]):
        modifiable_factors.append("Frequent carbonated/acidic beverage exposure")
        recommendations.append("Reduce frequent carbonated/acidic beverage exposure and favor water as the routine drink.")

    saliva_flags = []
    if "low" in txt("buffering_capacity"):
        saliva_flags.append("low buffering capacity")
    if "acid" in txt("salivary_ph"):
        saliva_flags.append("acidic salivary pH")
    if saliva_flags:
        modifiable_factors.append("Salivary vulnerability: " + ", ".join(saliva_flags))
        recommendations.append("Review hydration, dietary acid exposure and clinically indicated preventive measures in light of the recorded salivary findings.")

    microbial_flags = []
    if "more" in txt("mutans_load_in_saliva"):
        microbial_flags.append("higher mutans category")
    if "more" in txt("lactobacilli_load_in_saliva"):
        microbial_flags.append("higher lactobacilli category")
    if microbial_flags:
        modifiable_factors.append("Microbial/salivary profile: " + ", ".join(microbial_flags))
        recommendations.append("Intensify plaque control and reduce fermentable-carbohydrate frequency; any adjunctive measure requires clinician judgment.")

    # Deduplicate while preserving order.
    recommendations = list(dict.fromkeys(recommendations))
    if not recommendations:
        recommendations.append("Maintain routine risk-based prevention and recall, modified according to the direct clinical examination.")
    return clinical_priorities, modifiable_factors, recommendations


df = load_data()
eligible, audit = analysis_data(df)
results = train(f"{df.shape}-{int(audit['target_consistent'].sum())}", df)

st.success(f"Audited raw-data cohort loaded: {len(df)} matched participants; {int(audit['target_consistent'].sum())} passed Elham arithmetic QC.")
st.warning("Research decision-support prototype. Current data are cross-sectional: results describe internal predictive associations and do not prove causation or forecast future disease. Longitudinal follow-up is required for genuine future prediction.")

patient_id = st.sidebar.selectbox("Participant", eligible["id"].tolist())
model_name = st.sidebar.selectbox("Model", ["Random Forest", "XGBoost", "Blend"])
patient = eligible.loc[eligible["id"] == patient_id].iloc[0]

profile_tab, ai_tab, explain_tab, plan_tab, validation_tab, design_tab = st.tabs([
    "Detailed oral-health profile", "Component-specific AI", "Factors to review",
    "Personalized action plan", "Validation", "Study meaning"
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
    st.caption("The observed value comes from the dental examination. The AI estimate shows how much information the non-index factors contain about that component; it does not replace examination and is not a future forecast.")

with explain_tab:
    st.subheader("Patient factors the model uses")
    st.write("Select a clinical component to see the strongest patient-level model contributions. These are model-attribution signals for review, not evidence that a factor causes disease.")
    selected_target = st.selectbox("Clinical component", list(MODELED_COMPONENTS), format_func=lambda x: MODELED_COMPONENTS[x])
    result = results[selected_target]
    factors = local_model_factors(result, patient.to_dict(), model_name)
    if factors:
        explain_df = pd.DataFrame(factors, columns=["Patient factor", "Model contribution"])
        explain_df["Direction in model"] = np.where(explain_df["Model contribution"] > 0, "Pushes estimate upward", "Pushes estimate downward")
        st.dataframe(explain_df, use_container_width=True, hide_index=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(explain_df["Patient factor"][::-1], explain_df["Model contribution"][::-1])
        ax.axvline(0)
        ax.set_xlabel("Grouped SHAP contribution")
        ax.set_title(f"Patient-level model factors: {result.label}")
        st.pyplot(fig)
    else:
        st.info("Patient-level model explanation is unavailable in this environment.")
    st.caption("Changing one of these inputs cannot be assumed to change the clinical outcome by the displayed amount. The current study is observational and cross-sectional.")

with plan_tab:
    st.subheader("Personalized oral-health action plan")
    priorities, modifiable, advice = action_plan(patient.to_dict())

    st.markdown("#### 1. Clinical priorities from the examination")
    if priorities:
        for x in priorities:
            st.write(f"• {x}")
    else:
        st.write("No major modeled clinical component was recorded for this participant.")

    st.markdown("#### 2. Modifiable factors to review")
    if modifiable:
        for x in modifiable:
            st.write(f"• {x}")
    else:
        st.write("No prespecified modifiable trigger was detected from the recorded questionnaire/salivary fields.")

    st.markdown("#### 3. Tailored preventive and clinical recommendations")
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
    st.write("The recorded missing-tooth component includes wisdom teeth; therefore it must not automatically be interpreted as disease-related tooth loss. Third-molar eruption/developmental status requires clinical verification.")
    st.write("Rare clinical findings remain part of the patient's Elham profile but are not given separate ML models when too few participants have the condition.")
    st.write("Future forecasting is a second phase: repeat the clinical examination at follow-up and model change in each component from the baseline profile and baseline risk factors.")
