import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from analysis_pipeline import (
    canonicalize, analysis_data, DEMOGRAPHIC_COLS, SES_COLS, BEHAVIOR_COLS,
    SALIVARY_COLS, ELHAM_DIRECT_COMPONENTS,
)
from component_pipeline import (
    MODELED_COMPONENTS, DESCRIPTIVE_COMPONENTS, fit_all_components_for_app,
    clinical_profile,
)
from data.master160_embedded import load_master160

st.set_page_config(page_title="Dental AI Coach – Audited Research Prototype", layout="wide")
st.title("Dental AI Coach")
st.caption("Detailed Elham oral-health profile, explainable component-specific AI, and personalized action planning")


@st.cache_data(show_spinner=False)
def load_data():
    return canonicalize(load_master160())


@st.cache_resource(show_spinner="Loading validated component-specific models...")
def train(signature, data):
    return fit_all_components_for_app(data)


def safe_num(v):
    x = pd.to_numeric(v, errors="coerce")
    return 0.0 if pd.isna(x) else float(x)


def safe_text(v):
    return "Unknown" if pd.isna(v) else str(v).strip()


def pretty_label(name):
    replacements = {
        "cho": "carbohydrate",
        "ph": "pH",
        "of": "number of",
    }
    words = str(name).replace("_", " ").split()
    words = [replacements.get(w.lower(), w) for w in words]
    return " ".join(words).capitalize()


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
    """Patient-level grouped SHAP factors for model review, not causal effects."""
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


def overall_model_factors(results, row, model_name, top_n=10):
    """Aggregate absolute local attribution across modeled Elham components."""
    combined = {}
    signed = {}
    for result in results.values():
        for factor, contribution in local_model_factors(result, row, model_name, top_n=999):
            combined[factor] = combined.get(factor, 0.0) + abs(contribution)
            signed[factor] = signed.get(factor, 0.0) + contribution
    ranked = sorted(combined, key=combined.get, reverse=True)[:top_n]
    return pd.DataFrame([
        {
            "Patient factor": pretty_label(f),
            "Relative model influence": combined[f],
            "Net direction across modeled findings": "Upward" if signed[f] > 0 else "Downward",
            "field": f,
        }
        for f in ranked
    ])


def action_plan(row, prioritized_fields=None):
    """Clinician-reviewable plan combining findings and recorded modifiable inputs.

    Model attribution is used only to prioritize review. Recommendations remain
    rule/guideline based and do not treat SHAP values as causal treatment effects.
    """
    clinical_priorities, modifiable_factors, recommendations = [], [], []
    prioritized_fields = prioritized_fields or []

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
        recommendations.append("Interpret the missing-tooth count cautiously because it includes wisdom teeth. Verify eruption/developmental status before assigning disease-related tooth loss or treatment need.")

    def txt(c):
        return safe_text(row.get(c, "Unknown")).lower()

    triggers = []
    brushing = txt("tooth_brushing_frequency")
    if any(k in brushing for k in ["never", "once/day", "once a day", "once"]):
        triggers.append(("tooth_brushing_frequency", "Suboptimal brushing frequency", "Reinforce twice-daily toothbrushing with age-appropriate fluoride toothpaste and individualized technique coaching."))

    interdental = txt("interdental_cleaning")
    if interdental.startswith("no"):
        triggers.append(("interdental_cleaning", "No reported interdental cleaning", "Introduce a suitable daily interdental-cleaning method where clinically appropriate."))

    sugar = txt("sugar")
    if any(k in sugar for k in ["daily", "frequent", "twice", "once a day"]):
        triggers.append(("sugar", "Frequent free-sugar exposure", "Reduce the frequency of free-sugar exposure, especially between meals."))

    snacks = txt("snacks_frequency")
    if any(k in snacks for k in ["daily", "often", "frequent", "3+"]):
        triggers.append(("snacks_frequency", "Frequent between-meal snacking", "Reduce frequent cariogenic between-meal snacks and favor lower-cariogenic alternatives."))

    carbonated = txt("carbonated_beverages") + " " + txt("carbonated_beverages_diet")
    if any(k in carbonated for k in ["daily", "frequent", "once/day", "twice"]):
        triggers.append(("carbonated_beverages", "Frequent carbonated/acidic beverage exposure", "Reduce frequent carbonated/acidic beverage exposure and favor water as the routine drink."))

    saliva_flags = []
    if "low" in txt("buffering_capacity"):
        saliva_flags.append("low buffering capacity")
    if "acid" in txt("salivary_ph"):
        saliva_flags.append("acidic salivary pH")
    if saliva_flags:
        triggers.append(("buffering_capacity", "Salivary vulnerability: " + ", ".join(saliva_flags), "Review hydration, dietary acid exposure and clinically indicated preventive measures in light of the salivary findings."))

    microbial_flags = []
    if "more" in txt("mutans_load_in_saliva"):
        microbial_flags.append("higher mutans category")
    if "more" in txt("lactobacilli_load_in_saliva"):
        microbial_flags.append("higher lactobacilli category")
    if microbial_flags:
        triggers.append(("mutans_load_in_saliva", "Microbial/salivary profile: " + ", ".join(microbial_flags), "Intensify plaque control and reduce fermentable-carbohydrate frequency; adjunctive measures require clinician judgment."))

    priority_order = {field: i for i, field in enumerate(prioritized_fields)}
    triggers.sort(key=lambda x: priority_order.get(x[0], 999))
    for _, label, recommendation in triggers:
        modifiable_factors.append(label)
        recommendations.append(recommendation)

    recommendations = list(dict.fromkeys(recommendations))
    if not recommendations:
        recommendations.append("Maintain routine risk-based prevention and recall, modified according to the direct clinical examination.")
    return clinical_priorities, modifiable_factors, recommendations


def _base_choice_text(value):
    """Normalize superficial spelling/case/punctuation differences for every dropdown."""
    s = safe_text(value)
    s = s.replace("’", "'").replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s).strip().lower()
    s = re.sub(r"\s*[/_-]\s*", " ", s)
    s = re.sub(r"[^a-z0-9%+.' ]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _choice_key(value, col):
    """Collapse equivalent questionnaire categories while preserving a raw model value."""
    s = _base_choice_text(value)
    if not s or s in {"nan", "none", "n a", "na", "unknown", "unk", "not known", "-"}:
        return "unknown"

    # Universal yes/no variants.
    if s in {"yes", "y", "yeah", "true", "1"}:
        return "yes"
    if s in {"no", "n", "false", "0"}:
        return "no"

    # Common frequency wording used across brushing, diet, drinks and snacks.
    frequency_aliases = {
        "once daily": "once daily", "once a day": "once daily", "1 time daily": "once daily",
        "1 time a day": "once daily", "one time daily": "once daily", "daily once": "once daily",
        "twice daily": "twice daily", "twice a day": "twice daily", "2 times daily": "twice daily",
        "2 times a day": "twice daily", "two times daily": "twice daily", "daily twice": "twice daily",
        "three times daily": "three times daily", "3 times daily": "three times daily",
        "3 times a day": "three times daily", "three times a day": "three times daily",
        "every day": "daily", "everyday": "daily", "daily": "daily",
        "never": "never", "not at all": "never",
        "occasionally": "occasionally", "occasional": "occasionally", "sometimes": "sometimes",
        "frequently": "frequent", "frequent": "frequent", "often": "often",
    }
    if s in frequency_aliases:
        return frequency_aliases[s]

    # Categories that recur in several socioeconomic fields.
    universal_aliases = {
        "own": "owned", "owned": "owned", "owner": "owned",
        "rent": "rented", "rented": "rented", "rental": "rented",
        "post graduate": "postgraduate level", "post graduate level": "postgraduate level",
        "postgraduate": "postgraduate level", "postgraduate level": "postgraduate level",
        "university": "university level", "university education": "university level",
        "university level": "university level", "college": "university level",
        "school": "school level", "school education": "school level", "school level": "school level",
        "male": "male", "m": "male", "female": "female", "f": "female",
        "dont know": "unknown", "don't know": "unknown", "do not know": "unknown",
        "not applicable": "not applicable", "n a applicable": "not applicable",
    }
    if s in universal_aliases:
        return universal_aliases[s]

    # Field-specific semantic aliases. These are deliberately conservative: only
    # clearly equivalent source responses are merged.
    aliases = {
        "house_owned_or_rent": {
            "family owned": "owned", "owned house": "owned", "rented house": "rented",
        },
        "father_s_education": {
            "primary school": "school level", "secondary school": "school level",
            "high school": "school level",
        },
        "mother_s_education": {
            "primary school": "school level", "secondary school": "school level",
            "high school": "school level",
        },
        "tooth_brushing_frequency": {
            "one time": "once daily", "once": "once daily", "two times": "twice daily",
            "twice": "twice daily", "three times": "three times daily",
        },
        "frequency_of_visits": {
            "when needed": "when needed", "only when needed": "when needed",
            "when i have pain": "when symptomatic", "when pain": "when symptomatic",
        },
        "smoking": {
            "non smoker": "no", "nonsmoker": "no", "non smoking": "no",
            "smoker": "yes", "smoking": "yes",
        },
        "interdental_cleaning": {
            "not using": "no", "none": "no",
        },
        "mouth_rinse": {
            "not using": "no", "none": "no",
        },
    }
    return aliases.get(col, {}).get(s, s)


def _choice_display(key):
    display = {
        "unknown": "Unknown",
        "owned": "Owned",
        "rented": "Rented",
        "postgraduate level": "Postgraduate level",
        "school level": "School level",
        "university level": "University level",
        "once daily": "Once daily",
        "twice daily": "Twice daily",
        "three times daily": "Three times daily",
        "not applicable": "Not applicable",
        "when symptomatic": "When symptomatic",
    }
    if key in display:
        return display[key]
    if key in {"yes", "no", "male", "female", "daily", "never", "occasionally", "sometimes", "frequent", "often"}:
        return key.title()
    return key[:1].upper() + key[1:]


def categorical_options(data, col):
    """Return a clean, deduplicated option list for every categorical field.

    Each displayed category maps back to the most frequent equivalent raw value
    from the training cohort. Thus the UI is standardized without inventing new
    model categories or changing the underlying audited dataset.
    """
    if col not in data.columns:
        return [("Unknown", "Unknown")]

    counts = data[col].map(safe_text).value_counts()
    grouped = {}
    for raw, count in counts.items():
        key = _choice_key(raw, col)
        if key not in grouped or count > grouped[key][1]:
            grouped[key] = (raw, int(count))

    grouped.setdefault("unknown", ("Unknown", 0))

    non_unknown = [(k, v) for k, v in grouped.items() if k != "unknown"]
    non_unknown.sort(key=lambda item: (-item[1][1], _choice_display(item[0]).lower()))
    ordered = non_unknown + [("unknown", grouped["unknown"])]
    return [(raw_count[0], _choice_display(key)) for key, raw_count in ordered]


def render_predictor_inputs(data, columns, prefix):
    values = {}
    visible = [c for c in columns if c in data.columns]
    grid = st.columns(2)
    for i, col in enumerate(visible):
        with grid[i % 2]:
            series = data[col]
            label = pretty_label(col)
            if pd.api.types.is_numeric_dtype(series):
                med = pd.to_numeric(series, errors="coerce").median()
                med = 0.0 if pd.isna(med) else float(med)
                values[col] = st.number_input(label, value=med, step=1.0, key=f"{prefix}_{col}")
            else:
                choices = categorical_options(data, col)
                raw_options = [raw for raw, _ in choices]
                display_map = {raw: display for raw, display in choices}
                values[col] = st.selectbox(
                    label,
                    raw_options,
                    format_func=lambda x, m=display_map: m.get(x, safe_text(x)),
                    key=f"{prefix}_{col}",
                )
    return values


def new_patient_form(data):
    all_components = {**MODELED_COMPONENTS, **DESCRIPTIVE_COMPONENTS}
    with st.form("new_patient_form"):
        st.subheader("Enter a new patient's data")
        st.caption("Enter the detailed Elham clinical findings and the independently collected patient factors. No existing participant record is used for the new patient.")

        new = {"id": "NEW_PATIENT"}
        with st.expander("A. Detailed Elham clinical findings", expanded=True):
            st.write("Enter the number of teeth/units recorded for each finding.")
            cols = st.columns(4)
            for i, (field, label) in enumerate(all_components.items()):
                with cols[i % 4]:
                    new[field] = st.number_input(label, min_value=0, max_value=32, value=0, step=1, key=f"new_{field}")

        with st.expander("B. Demographic factors", expanded=True):
            new.update(render_predictor_inputs(data, DEMOGRAPHIC_COLS, "new_demo"))
        with st.expander("C. Socioeconomic and access factors"):
            new.update(render_predictor_inputs(data, SES_COLS, "new_ses"))
        with st.expander("D. Behavioral and dietary factors"):
            new.update(render_predictor_inputs(data, BEHAVIOR_COLS, "new_beh"))
        with st.expander("E. Salivary factors"):
            new.update(render_predictor_inputs(data, SALIVARY_COLS, "new_saliva"))

        submitted = st.form_submit_button("Analyze new patient", type="primary", use_container_width=True)

    if submitted:
        new["elham_s_index_including_wisdom"] = float(sum(safe_num(new.get(c, 0)) for c in ELHAM_DIRECT_COMPONENTS))
        st.session_state["new_patient_data"] = new
    return st.session_state.get("new_patient_data")


df = load_data()
eligible, audit = analysis_data(df)
results = train(f"{df.shape}-{int(audit['target_consistent'].sum())}", df)

st.success(f"Audited reference cohort loaded: {len(df)} matched participants; {int(audit['target_consistent'].sum())} passed Elham arithmetic QC.")
st.warning("Research decision-support prototype. Current data are cross-sectional: model attributions are associations, not proof of causation or future disease forecasting.")

source_mode = st.sidebar.radio("Patient source", ["New patient", "Existing study participant"], index=0)
available_models = ["Random Forest"]
if all(r.xgb_final is not None for r in results.values()):
    available_models += ["XGBoost", "Blend"]
model_name = st.sidebar.selectbox("AI model", available_models)

if source_mode == "Existing study participant":
    patient_id = st.sidebar.selectbox("Participant", eligible["id"].tolist())
    patient = eligible.loc[eligible["id"] == patient_id].iloc[0]
    patient_ready = True
    st.info("Viewing a participant from the audited study cohort. Select 'New patient' in the sidebar to enter a completely new case.")
else:
    entered = new_patient_form(df)
    if entered is None:
        st.info("Complete the new-patient form and click 'Analyze new patient' to generate the Elham profile, model-attribution factors, and personalized recommendations.")
        st.stop()
    patient = pd.Series(entered)
    patient_ready = True
    st.success("New patient data loaded for analysis. The patient is not added to the research training cohort.")

profile_tab, ai_tab, explain_tab, plan_tab, validation_tab, design_tab = st.tabs([
    "Detailed oral-health profile", "Component-specific AI", "Most affecting factors",
    "Personalized action plan", "Validation", "Study meaning"
])

with profile_tab:
    st.subheader("Detailed Elham clinical profile")
    profile = clinical_profile(patient)
    profile["status"] = np.where(profile["count"] > 0, "Present", "Not recorded")
    total_elham = float(profile["count"].sum())
    c1, c2 = st.columns(2)
    c1.metric("Calculated Elham Index total", f"{total_elham:.0f}")
    c2.metric("Clinical findings present", int((profile["count"] > 0).sum()))
    st.dataframe(profile, use_container_width=True, hide_index=True)
    shown = profile.loc[profile["count"] > 0]
    if not shown.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(shown["component"], shown["count"])
        ax.set_xlabel("Number of teeth / recorded units")
        ax.set_title("Clinical oral-health profile")
        st.pyplot(fig)
    st.caption("The total is displayed descriptively. The leakage-safe AI models analyze sufficiently prevalent Elham components separately rather than predicting the total from its own component counts.")

with ai_tab:
    st.subheader("Component-specific AI estimates from nonclinical factors")
    rows = []
    for target, result in results.items():
        rows.append({
            "Clinical component": result.label,
            "Observed clinical count": safe_num(patient.get(target)),
            "AI-estimated count from patient factors": predict_component(result, patient.to_dict(), model_name),
            "Cohort prevalence": result.prevalence,
        })
    table = pd.DataFrame(rows)
    st.dataframe(table.style.format({"Observed clinical count":"{:.0f}", "AI-estimated count from patient factors":"{:.2f}", "Cohort prevalence":"{:.1%}"}), use_container_width=True, hide_index=True)
    st.caption("The observed count comes from the entered examination. The AI estimate uses demographic, socioeconomic, behavioral, dietary and salivary inputs only. The difference is not a diagnostic error score and the estimate does not replace examination.")

with explain_tab:
    st.subheader("Most affecting patient factors for the recorded Elham findings")
    st.write("The first table combines patient-level model attribution across the four modeled Elham findings. A larger value means the factor had greater influence on this patient's model outputs. It does not mean the factor caused the clinical finding.")
    overall = overall_model_factors(results, patient.to_dict(), model_name, top_n=10)
    if not overall.empty:
        display_overall = overall.drop(columns=["field"]).copy()
        st.dataframe(display_overall.style.format({"Relative model influence": "{:.3f}"}), use_container_width=True, hide_index=True)
        fig, ax = plt.subplots(figsize=(9, 5))
        plot_df = display_overall.iloc[::-1]
        ax.barh(plot_df["Patient factor"], plot_df["Relative model influence"])
        ax.set_xlabel("Aggregated absolute SHAP influence")
        ax.set_title("Most influential patient factors across modeled Elham findings")
        st.pyplot(fig)
    else:
        st.info("Patient-level model explanation is unavailable in this environment.")

    st.markdown("#### Explain one clinical finding")
    selected_target = st.selectbox("Elham clinical component", list(MODELED_COMPONENTS), format_func=lambda x: MODELED_COMPONENTS[x])
    result = results[selected_target]
    factors = local_model_factors(result, patient.to_dict(), model_name)
    if factors:
        explain_df = pd.DataFrame(factors, columns=["field", "Model contribution"])
        explain_df["Patient factor"] = explain_df["field"].map(pretty_label)
        explain_df["Direction in model"] = np.where(explain_df["Model contribution"] > 0, "Pushes estimated count upward", "Pushes estimated count downward")
        st.dataframe(explain_df[["Patient factor", "Model contribution", "Direction in model"]], use_container_width=True, hide_index=True)
    st.caption("SHAP direction describes the fitted model, not a harmful/protective causal effect. Recommendations below are therefore based on the entered clinical findings and modifiable risk information, with model attribution used only to prioritize what should be reviewed.")

with plan_tab:
    st.subheader("Personalized oral-health action plan")
    overall_for_plan = overall_model_factors(results, patient.to_dict(), model_name, top_n=15)
    prioritized_fields = overall_for_plan["field"].tolist() if not overall_for_plan.empty else []
    priorities, modifiable, advice = action_plan(patient.to_dict(), prioritized_fields)

    st.markdown("#### 1. Clinical priorities from the entered Elham examination")
    if priorities:
        for x in priorities:
            st.write(f"• {x}")
    else:
        st.write("No major modeled clinical component was recorded for this patient.")

    st.markdown("#### 2. Modifiable factors to review")
    if modifiable:
        for x in modifiable:
            st.write(f"• {x}")
    else:
        st.write("No prespecified modifiable trigger was detected from the entered behavioral, dietary or salivary fields.")

    if prioritized_fields:
        st.markdown("#### 3. Model-prioritized factors for clinician review")
        st.write(", ".join(pretty_label(x) for x in prioritized_fields[:6]))
        st.caption("These factors are prioritized because they influenced this patient's model outputs. They are not automatically treatment targets and should be interpreted with the clinical history.")

    st.markdown("#### 4. Tailored preventive and clinical recommendations")
    for x in advice:
        st.write(f"• {x}")

    st.info("This is clinician-reviewable decision support, not an autonomous prescription. Diagnosis, treatment choice, therapeutic dosing and recall intervals require professional judgment and applicable guidelines.")

with validation_tab:
    st.subheader("Internal validation of the component models")
    rows = []
    for _, result in results.items():
        for model, md in result.metrics.items():
            rows.append({"Component": result.label, "Model": model, **md})
    perf = pd.DataFrame(rows)
    st.dataframe(perf.style.format({"R2":"{:.3f}", "MAE":"{:.3f}", "RMSE":"{:.3f}"}), use_container_width=True, hide_index=True)
    st.caption("Five-fold out-of-fold results are shown with mean and median baselines. These internally validated results are modest and are reported transparently; the app is a research decision-support prototype.")

with design_tab:
    st.subheader("What this app is designed to do")
    st.write("For a new patient, the dentist enters the detailed Elham clinical findings plus demographic, socioeconomic, behavioral, dietary and salivary information. The app then retains the full Elham profile, estimates the sufficiently prevalent components using independent nonclinical factors, shows the patient-specific factors that most influenced those estimates, and generates a clinician-reviewable personalized action plan.")
    st.write("The calculated Elham total is retained as a descriptive summary of oral status. It is not used as the main machine-learning target because using its own component findings as predictors would create target leakage.")
    st.write("The recorded missing-tooth component includes wisdom teeth, so third-molar eruption/developmental status must be verified before interpreting it as disease-related tooth loss.")
    st.write("Rare findings remain visible in the patient's Elham profile even when event counts in the current 160-participant cohort are too small for a reliable separate ML model.")
    st.write("Future forecasting requires longitudinal follow-up with repeat Elham examination. The present cross-sectional app supports current risk profiling and explainable associations, not future disease prediction.")