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
from clinical_guidelines import build_guideline_action_plan, GUIDELINE_REFERENCES
from decision_support_display import (
    model_reliability, clinical_concern_map, recommendation_trigger_summary,
)
from data.master160_embedded import load_master160
from patient_pdf_report import build_patient_pdf_report

st.set_page_config(page_title="Dental AI Coach – Integrated Research Prototype", layout="wide")
st.title("Dental AI Coach")
st.caption(
    "Integrated explainable framework for detailed oral-health profiling, personalized education, "
    "prevention, clinician-reviewed treatment planning, and maintenance support"
)


@st.cache_data(show_spinner=False)
def load_data():
    return canonicalize(load_master160())


@st.cache_resource(show_spinner="Loading internally evaluated component-specific models...")
def train(signature, data):
    return fit_all_components_for_app(data)


def safe_num(v):
    x = pd.to_numeric(v, errors="coerce")
    return 0.0 if pd.isna(x) else float(x)


def safe_text(v):
    return "Unknown" if pd.isna(v) else str(v).strip()


def pretty_label(name):
    replacements = {"cho": "carbohydrate", "ph": "pH", "of": "number of"}
    words = str(name).replace("_", " ").split()
    return " ".join(replacements.get(w.lower(), w) for w in words).capitalize()


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
    """Patient-level grouped SHAP factors; attribution is not a causal effect."""
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
            items.append((parent, float(np.sum(sv[0, idxs]))))
        return sorted(items, key=lambda x: abs(x[1]), reverse=True)[:top_n]
    except Exception:
        return []


def overall_model_factors(results, row, model_name, top_n=10):
    combined, signed = {}, {}
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


def _base_choice_text(value):
    s = safe_text(value).replace("’", "'").replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s).strip().lower()
    s = re.sub(r"\s*[/_-]\s*", " ", s)
    s = re.sub(r"[^a-z0-9%+.' ]+", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _choice_key(value):
    s = _base_choice_text(value)
    if not s or s in {"nan", "none", "n a", "na", "unknown", "unk", "not known", "-"}:
        return "unknown"
    if s in {"yes", "y", "yeah", "true", "1"}:
        return "yes"
    if s in {"no", "n", "false", "0"}:
        return "no"
    aliases = {
        "own": "owned", "owner": "owned", "rent": "rented", "rental": "rented",
        "post graduate": "postgraduate level", "post graduate level": "postgraduate level",
        "postgraduate": "postgraduate level", "university": "university level",
        "university education": "university level", "school": "school level",
        "school education": "school level", "once/day": "once daily",
        "once a day": "once daily", "twice/day": "twice daily",
        "twice a day": "twice daily", "greater than 5": "more than 5", "3 to5": "3 to 5",
    }
    return aliases.get(s, s)


def _choice_display(key):
    display = {
        "unknown": "Unknown", "owned": "Owned", "rented": "Rented",
        "postgraduate level": "Postgraduate level", "university level": "University level",
        "school level": "School level", "once daily": "Once daily", "twice daily": "Twice daily",
        "more than 5": "More than 5", "3 to 5": "3 to 5",
    }
    return display.get(key, key[:1].upper() + key[1:])


def categorical_options(data, col):
    if col not in data.columns:
        return [("Unknown", "Unknown")]
    counts = data[col].map(safe_text).value_counts()
    grouped = {}
    for raw, count in counts.items():
        key = _choice_key(raw)
        if key not in grouped or count > grouped[key][1]:
            grouped[key] = (raw, int(count))
    grouped.setdefault("unknown", ("Unknown", 0))
    ordered = [(k, v) for k, v in grouped.items() if k != "unknown"]
    ordered.sort(key=lambda item: (-item[1][1], _choice_display(item[0]).lower()))
    ordered.append(("unknown", grouped["unknown"]))
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
                values[col] = st.number_input(label, value=0.0 if pd.isna(med) else float(med), step=1.0, key=f"{prefix}_{col}")
            else:
                choices = categorical_options(data, col)
                raw_options = [raw for raw, _ in choices]
                display_map = {raw: display for raw, display in choices}
                values[col] = st.selectbox(label, raw_options, format_func=lambda x, m=display_map: m.get(x, safe_text(x)), key=f"{prefix}_{col}")
    return values


def new_patient_form(data):
    all_components = {**MODELED_COMPONENTS, **DESCRIPTIVE_COMPONENTS}
    with st.form("new_patient_form"):
        st.subheader("Enter a new patient's data")
        st.caption("Enter detailed Elham findings and independently collected patient factors. The patient is analyzed only and is not added to the research cohort.")
        new = {"id": "NEW_PATIENT"}
        with st.expander("A. Detailed Elham clinical findings", expanded=True):
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


CARE_PHASES = ["Awareness & education", "Prevention", "Disease-control / urgent needs", "Restorative considerations", "Maintenance & recall"]


def recommendation_phase(domain):
    mapping = {
        "Caries": "Disease-control / urgent needs", "Dental trauma": "Disease-control / urgent needs",
        "Restorations": "Restorative considerations", "Developmental enamel defects": "Restorative considerations",
        "Missing/developing dentition": "Restorative considerations", "Tooth surface loss": "Prevention",
        "Diet / caries prevention": "Prevention", "Diet / tooth surface loss": "Prevention",
        "Salivary risk": "Prevention", "Caries risk": "Prevention",
        "Oral hygiene / fluoride": "Awareness & education", "Plaque control": "Awareness & education",
        "Tobacco / periodontal prevention": "Awareness & education", "Periodontal": "Maintenance & recall",
        "Periodontal health": "Maintenance & recall", "Sealants": "Prevention", "Prevention": "Maintenance & recall",
    }
    return mapping.get(str(domain), "Maintenance & recall")


df = load_data()
eligible, audit = analysis_data(df)
results = train(f"{df.shape}-{int(audit['target_consistent'].sum())}", df)

st.info("Purpose: integrate a detailed oral-health profile with demographic, socioeconomic, behavioral, dietary, and salivary information to support personalized patient awareness and education, preventive planning, clinician-reviewed treatment planning, and maintenance support.")
st.success(f"Real-world Egyptian adolescent dataset: {len(df)} matched participants in the current analytical cohort; {int(audit['target_consistent'].sum())} passed Elham arithmetic QC.")
st.warning("Research decision-support prototype. Current data are cross-sectional: model attributions are associations, not proof of causation or future disease forecasting. Observed clinical findings remain the primary reference.")

cols = st.columns(4)
for col, label in zip(cols, ["Personalized awareness & education", "Preventive plan", "Clinician-reviewable treatment planning", "Maintenance & recall support"]):
    col.markdown(f"**{label}**")

source_mode = st.sidebar.radio("Patient source", ["New patient", "Existing study participant"], index=0)
available_models = ["Random Forest"]
if all(r.xgb_final is not None for r in results.values()):
    available_models += ["XGBoost", "Blend"]
model_name = st.sidebar.selectbox("AI model", available_models)

if source_mode == "Existing study participant":
    patient_id = st.sidebar.selectbox("Participant", eligible["id"].tolist())
    patient = eligible.loc[eligible["id"] == patient_id].iloc[0]
    st.info("Viewing a participant from the audited study cohort. Select 'New patient' in the sidebar to enter a new case.")
else:
    entered = new_patient_form(df)
    if entered is None:
        st.info("Complete the new-patient form and click 'Analyze new patient' to generate the detailed Elham profile, clinical concern map, explainable AI outputs, phased action plan, and PDF report.")
        st.stop()
    patient = pd.Series(entered)
    patient_id = safe_text(patient.get("id", "NEW_PATIENT"))
    st.success("New patient data loaded for analysis. The patient is not added to the research training cohort.")

profile = clinical_profile(patient)
profile["status"] = np.where(profile["count"] > 0, "Present", "Not recorded")
concern = clinical_concern_map(patient.to_dict())

ai_rows = []
for target, result in results.items():
    rel = model_reliability(result, model_name)
    ai_rows.append({
        "Clinical component": result.label,
        "Observed clinical count": safe_num(patient.get(target)),
        "AI-estimated count from patient factors": predict_component(result, patient.to_dict(), model_name),
        "Cohort prevalence": result.prevalence,
        "Model reliability": rel["label"],
        "Clinical use": rel["level"],
    })
ai_table = pd.DataFrame(ai_rows)
overall = overall_model_factors(results, patient.to_dict(), model_name, top_n=15)
prioritized_fields = overall["field"].tolist() if not overall.empty else []
priorities, modifiable, rec_df = build_guideline_action_plan(patient.to_dict(), prioritized_fields)
rec_for_report = rec_df.copy()
if not rec_for_report.empty:
    rec_for_report["care_phase"] = rec_for_report["domain"].map(recommendation_phase)

profile_tab, concern_tab, ai_tab, explain_tab, plan_tab, validation_tab, design_tab = st.tabs([
    "Detailed oral-health profile", "Clinical concern map", "Component-specific AI",
    "Most affecting factors", "Personalized action plan", "Validation", "Study meaning"
])

with profile_tab:
    st.subheader("Detailed Elham clinical profile")
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
    st.caption("The total is descriptive. Leakage-safe AI models analyze sufficiently prevalent Elham components separately rather than predicting the total from its own component counts.")

with concern_tab:
    st.subheader("Clinical concern map")
    st.write("This organizes the entered examination and modifiable factors into domains for clinician review. The categories are heuristic decision-support labels, not validated diagnostic or prognostic scores.")
    st.dataframe(concern, use_container_width=True, hide_index=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("High-concern domains", int((concern["Concern"] == "High").sum()))
    c2.metric("Moderate-concern domains", int((concern["Concern"] == "Moderate").sum()))
    c3.metric("Low-concern domains", int((concern["Concern"] == "Low").sum()))
    st.caption("Concern level helps structure review only. It must not be interpreted as disease probability, severity staging, or a treatment indication without direct clinical assessment.")

with ai_tab:
    st.subheader("Component-specific AI estimates from nonclinical factors")
    st.dataframe(ai_table.style.format({"Observed clinical count": "{:.0f}", "AI-estimated count from patient factors": "{:.2f}", "Cohort prevalence": "{:.1%}"}), use_container_width=True, hide_index=True)
    st.caption("Reliability labels are derived from internal five-fold validation against the mean baseline. They are not probabilities of correctness. Observed counts from examination remain the clinical reference.")
    for _, result in results.items():
        rel = model_reliability(result, model_name)
        if rel["label"] == "No demonstrated predictive advantage":
            st.warning(f"{result.label}: {rel['note']} The AI estimate should not be used for individual clinical prediction.")

with explain_tab:
    st.subheader("Most affecting patient factors for the recorded Elham findings")
    st.write("Larger attribution values indicate greater influence on this patient's model outputs; they do not establish that a factor caused the clinical finding.")
    if not overall.empty:
        display_overall = overall.drop(columns=["field"]).head(10).copy()
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
    factors = local_model_factors(results[selected_target], patient.to_dict(), model_name)
    if factors:
        explain_df = pd.DataFrame(factors, columns=["field", "Model contribution"])
        explain_df["Patient factor"] = explain_df["field"].map(pretty_label)
        explain_df["Direction in model"] = np.where(explain_df["Model contribution"] > 0, "Pushes estimated count upward", "Pushes estimated count downward")
        st.dataframe(explain_df[["Patient factor", "Model contribution", "Direction in model"]], use_container_width=True, hide_index=True)
    rel = model_reliability(results[selected_target], model_name)
    st.info(f"Reliability for this model/component: {rel['label']} — {rel['note']}")
    st.caption("SHAP direction describes fitted-model behavior, not a harmful/protective causal effect. It prioritizes clinician review but does not generate treatment indications.")

with plan_tab:
    st.subheader("Personalized oral-health action plan")
    st.caption("Recommendations are generated by a separate guideline-based rule layer from recorded clinical findings and modifiable factors. AI attribution is used only to prioritize review.")

    factor_report = overall.drop(columns=["field"]).copy() if not overall.empty else pd.DataFrame()
    pdf_bytes = build_patient_pdf_report(
        patient_id=patient_id,
        model_name=model_name,
        profile_df=profile,
        concern_df=concern,
        ai_df=ai_table,
        factor_df=factor_report,
        priorities=priorities,
        modifiable=modifiable,
        recommendations_df=rec_for_report,
        care_phases=CARE_PHASES,
        guideline_references=GUIDELINE_REFERENCES,
    )
    st.download_button(
        "Download detailed PDF patient report",
        data=pdf_bytes,
        file_name=f"Dental_AI_Coach_Report_{re.sub(r'[^A-Za-z0-9_-]+', '_', patient_id)}.pdf",
        mime="application/pdf",
        type="primary",
        use_container_width=True,
    )
    st.caption("The PDF includes the clinical profile, concern map, AI estimates and reliability, model-prioritized factors, clinical priorities, modifiable factors, phased recommendations, guideline basis, and safety statement.")

    st.markdown("#### 1. Clinical priorities from the entered Elham examination")
    for x in priorities or ["No major entered Elham finding triggered a clinical-priority rule."]:
        st.write(f"• {x}")

    st.markdown("#### 2. Modifiable factors to review")
    for x in modifiable or ["No prespecified modifiable trigger was detected from the entered behavioral, dietary or salivary fields."]:
        st.write(f"• {x}")

    if prioritized_fields:
        st.markdown("#### 3. Model-prioritized factors for clinician review")
        st.write(", ".join(pretty_label(x) for x in prioritized_fields[:6]))
        st.caption("These are model-attribution priorities, not causal treatment targets.")

    st.markdown("#### 4. Guideline-based personalized recommendations by care phase")
    if rec_for_report.empty:
        st.write("No guideline-based recommendation rule was triggered by the entered data.")
    else:
        for phase in CARE_PHASES:
            phase_df = rec_for_report.loc[rec_for_report["care_phase"] == phase]
            if phase_df.empty:
                continue
            st.markdown(f"### {phase}")
            for _, r in phase_df.iterrows():
                with st.container(border=True):
                    st.markdown(f"**{r['priority']} priority — {r['domain']}**")
                    st.write(r["recommendation"])
                    st.caption(f"Triggered by: {recommendation_trigger_summary(patient.to_dict(), r['domain'])}")
                    st.caption(f"Why this recommendation: {r['rationale']}")
                    st.caption(f"Guideline basis: {r['evidence_source']}")
                    st.caption("Clinical confirmation required before acting.")

    with st.expander("Guidelines used by the recommendation engine"):
        for ref in GUIDELINE_REFERENCES:
            st.markdown(f"**{ref['short']}**")
            st.write(ref["scope"])

    st.info("This is clinician-reviewable decision support, not an autonomous prescription. Definitive diagnosis, radiographs, fluoride concentration/application, medications, operative technique, treatment timing, and recall interval require professional judgment, patient-specific assessment, and local guidance.")

with validation_tab:
    st.subheader("Internal validation of the component models")
    rows = []
    for _, result in results.items():
        for model, md in result.metrics.items():
            rel = model_reliability(result, model) if model not in {"Mean baseline", "Median baseline"} else None
            rows.append({"Component": result.label, "Model": model, **md, "Reliability interpretation": rel["label"] if rel else "Reference baseline"})
    perf = pd.DataFrame(rows)
    st.dataframe(perf.style.format({"R2": "{:.3f}", "MAE": "{:.3f}", "RMSE": "{:.3f}"}), use_container_width=True, hide_index=True)
    st.caption("Five-fold out-of-fold results are shown with mean and median baselines. Reliability interpretation is intentionally conservative. The predictive models are internally evaluated components of a broader research decision-support framework.")

with design_tab:
    st.subheader("What this framework delivers")
    deliver = [
        "Detailed 16-finding Elham oral-health phenotype", "Component-specific AI estimates with explicit reliability",
        "Domain-based explainability (non-causal)", "Personalized awareness and education",
        "Preventive recommendations", "Clinician-reviewable treatment-planning support",
        "Maintenance and recall considerations", "Downloadable detailed PDF patient report",
    ]
    cols = st.columns(2)
    for i, item in enumerate(deliver):
        cols[i % 2].markdown(f"**{item}**")
    st.markdown("#### Dataset")
    st.write("The framework was developed from original real-world oral-health data collected from adolescents in the Egyptian community. The current analytical cohort contains 160 matched participants with clinical and nonclinical data.")
    st.markdown("#### Design meaning")
    st.write("The interface separates three layers: (1) entered clinical findings, (2) explainable AI associations with explicit reliability indicators, and (3) guideline-based personalized actions. This separation prevents model attribution from being mistaken for diagnosis or treatment advice.")
    st.write("The clinical concern map organizes caries/restorative, periodontal/plaque-control, developmental enamel, tooth-surface-loss, trauma, and missing/developing-dentition domains. These concern labels are heuristic review aids, not validated risk scores.")
    st.write("The recommendation engine is deliberately separated from the ML models. Recorded clinical findings and modifiable factors trigger recommendations; SHAP only changes which factors are reviewed first.")
    st.write("The calculated Elham total is retained as a descriptive summary and is not predicted from its own component findings, preventing target leakage.")
    st.write("The recorded missing-tooth component includes wisdom teeth, so eruption/developmental status must be verified before interpreting it as disease-related tooth loss.")
    st.write("Rare findings remain visible even when the current 160-participant cohort is too small for reliable separate ML models. Future disease forecasting requires longitudinal follow-up.")
