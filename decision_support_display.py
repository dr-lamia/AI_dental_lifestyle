"""Presentation helpers for transparent clinical decision support.

These functions do not diagnose disease and do not modify treatment rules. They
summarize entered findings into clinician-reviewable concern domains and translate
cross-validated model performance into plain-language reliability indicators.
"""
from __future__ import annotations

import re
import pandas as pd

from clinical_guidelines import build_guideline_action_plan
from patient_report_pdf import build_detailed_patient_pdf


def _num(row, field):
    v = pd.to_numeric(row.get(field, 0), errors="coerce")
    return 0.0 if pd.isna(v) else float(v)


def _txt(row, field):
    v = row.get(field, "Unknown")
    return "unknown" if pd.isna(v) else str(v).strip().lower()


def model_reliability(result, model_name: str):
    """Return a conservative reliability label from internal CV performance.

    This is deliberately not a probability of correctness. It compares the chosen
    model with the mean baseline using R² and MAE and is intended to prevent users
    from overinterpreting weak predictions.
    """
    metrics = result.metrics
    chosen = metrics.get(model_name) or metrics.get("Random Forest") or {}
    baseline = metrics.get("Mean baseline", {})
    r2 = float(chosen.get("R2", 0.0))
    mae = float(chosen.get("MAE", float("nan")))
    base_mae = float(baseline.get("MAE", float("nan")))

    if r2 <= 0 or (pd.notna(mae) and pd.notna(base_mae) and mae >= base_mae):
        return {
            "label": "No demonstrated predictive advantage",
            "level": "Do not use for individual prediction",
            "note": "The selected model does not outperform the mean baseline on internal validation.",
        }
    if r2 < 0.05:
        return {
            "label": "Very limited predictive signal",
            "level": "Interpret with substantial caution",
            "note": "Internal validation shows only a very small improvement over baseline.",
        }
    if r2 < 0.15:
        return {
            "label": "Limited predictive signal",
            "level": "Supportive only",
            "note": "The model captures a modest amount of variation and should not replace examination.",
        }
    return {
        "label": "Moderate internal predictive signal",
        "level": "Still requires clinical confirmation",
        "note": "Performance is stronger but remains internally validated only.",
    }


def _render_pdf_download(row: dict, concern_df: pd.DataFrame):
    """Render one detailed PDF download button when running inside Streamlit."""
    try:
        import streamlit as st
        from streamlit.runtime import exists as streamlit_runtime_exists

        if not streamlit_runtime_exists():
            return
        priorities, modifiable, rec_df = build_guideline_action_plan(row, [])
        pdf_bytes = build_detailed_patient_pdf(row, concern_df, priorities, modifiable, rec_df)
        raw_id = str(row.get("id", "patient"))
        file_id = re.sub(r"[^A-Za-z0-9_-]+", "_", raw_id).strip("_") or "patient"
        st.download_button(
            "Download detailed PDF report",
            data=pdf_bytes,
            file_name=f"Dental_AI_Coach_Report_{file_id}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True,
            key=f"download_pdf_report_{file_id}",
            help="Includes the detailed Elham profile, concern map, relevant patient factors, clinical priorities, modifiable factors, phased guideline-based recommendations, and safety notes.",
        )
        st.caption(
            "The PDF is a clinician-reviewable report based on recorded findings and guideline rules. "
            "AI attribution remains supportive and non-causal."
        )
    except Exception as exc:
        try:
            import streamlit as st
            st.warning(f"PDF report could not be generated: {exc}")
        except Exception:
            pass


def clinical_concern_map(row: dict):
    """Create non-diagnostic concern domains from entered findings and risk factors.

    Levels are heuristic decision-support categories (not validated risk scores).
    They are intended to organize review, not diagnose or determine treatment.
    """
    domains = []

    decay = _num(row, "decayed_1")
    filled = _num(row, "filled_2")
    sugar = " ".join([_txt(row, "sugar"), _txt(row, "snacks_frequency"), _txt(row, "snack_content")])
    brushing = _txt(row, "tooth_brushing_frequency")
    saliva = " ".join([_txt(row, "buffering_capacity"), _txt(row, "salivary_ph"), _txt(row, "salivary_quantity")])
    caries_flags = sum([
        decay > 0,
        any(k in sugar for k in ["daily", "more than once", "frequent", "sweet", "cake", "junk", "chips"]),
        any(k in brushing for k in ["never", "rare", "once/day", "once daily", "1-3 times/week"]),
        any(k in saliva for k in ["very low", "low", "acid"]),
    ])
    caries_level = "High" if decay > 0 or caries_flags >= 3 else "Moderate" if caries_flags >= 1 or filled > 0 else "Low"
    domains.append({"Domain": "Caries / restorative", "Concern": caries_level, "Why": f"{int(decay)} decayed and {int(filled)} filled teeth; behavioral/salivary modifiers considered."})

    periodontal = _txt(row, "periodontal_status")
    interdental = _txt(row, "interdental_cleaning")
    smoking = _txt(row, "smoking")
    perio_flags = sum([
        periodontal not in {"unknown", "normal", "healthy", "none", "0"},
        interdental.startswith("no") or interdental in {"none", "never"},
        smoking.startswith("yes") or "smoker" in smoking,
    ])
    perio_level = "High" if periodontal not in {"unknown", "normal", "healthy", "none", "0"} else "Moderate" if perio_flags >= 1 else "Low"
    domains.append({"Domain": "Periodontal / plaque control", "Concern": perio_level, "Why": "Entered periodontal status plus plaque-control and tobacco factors."})

    hypo = _num(row, "hypocalcification_4") + _num(row, "hypoplasia_3") + _num(row, "fluorosis_5")
    enamel_level = "High" if hypo >= 6 else "Moderate" if hypo > 0 else "Low"
    domains.append({"Domain": "Developmental enamel defects", "Concern": enamel_level, "Why": f"{int(hypo)} teeth/units with recorded developmental enamel findings."})

    wear = _num(row, "erosion_6") + _num(row, "attrition_8") + _num(row, "abrasion_7") + _num(row, "abfraction_9")
    acidic = " ".join([_txt(row, "carbonated_beverages"), _txt(row, "carbonated_beverages_diet"), _txt(row, "acidic_food_or_drinks")])
    wear_level = "High" if wear >= 4 else "Moderate" if wear > 0 or any(k in acidic for k in ["daily", "frequent", "yes", "more than"]) else "Low"
    domains.append({"Domain": "Tooth surface loss / erosion", "Concern": wear_level, "Why": f"{int(wear)} recorded wear/erosion units plus acidic-exposure history."})

    fractured = _num(row, "fractured_h")
    trauma_level = "High" if fractured > 0 else "Low"
    domains.append({"Domain": "Dental trauma", "Concern": trauma_level, "Why": f"{int(fractured)} fractured tooth/teeth recorded."})

    missing = _num(row, "missing_0_including_wisdom_")
    missing_level = "Moderate" if missing > 0 else "Low"
    domains.append({"Domain": "Missing / developing dentition", "Concern": missing_level, "Why": f"{int(missing)} teeth recorded missing including wisdom teeth; eruption/development must be verified."})

    order = {"High": 0, "Moderate": 1, "Low": 2}
    out = pd.DataFrame(domains)
    out = out.sort_values(["Concern", "Domain"], key=lambda s: s.map(order) if s.name == "Concern" else s).reset_index(drop=True)
    _render_pdf_download(row, out)
    return out


def recommendation_trigger_summary(row: dict, domain: str):
    """Return the patient data that most directly triggered a recommendation domain."""
    mapping = {
        "Caries": f"Entered decayed teeth: {int(_num(row, 'decayed_1'))}",
        "Restorations": f"Entered filled teeth: {int(_num(row, 'filled_2'))}",
        "Developmental enamel defects": f"Hypocalcified/hypoplastic findings recorded: {int(_num(row, 'hypocalcification_4') + _num(row, 'hypoplasia_3'))}",
        "Dental trauma": f"Fractured teeth recorded: {int(_num(row, 'fractured_h'))}",
        "Tooth surface loss": f"Recorded erosion/attrition/abrasion/abfraction units: {int(_num(row, 'erosion_6') + _num(row, 'attrition_8') + _num(row, 'abrasion_7') + _num(row, 'abfraction_9'))}",
        "Missing/developing dentition": f"Missing teeth including wisdom teeth: {int(_num(row, 'missing_0_including_wisdom_'))}",
        "Oral hygiene / fluoride": f"Reported brushing frequency: {_txt(row, 'tooth_brushing_frequency')}",
        "Plaque control": f"Reported interdental cleaning: {_txt(row, 'interdental_cleaning')}",
        "Diet / caries prevention": f"Sugar: {_txt(row, 'sugar')}; snacks: {_txt(row, 'snacks_frequency')}",
        "Diet / tooth surface loss": f"Carbonated/acidic exposure: {_txt(row, 'carbonated_beverages')} / {_txt(row, 'acidic_food_or_drinks')}",
        "Tobacco / periodontal prevention": f"Smoking status: {_txt(row, 'smoking')}",
        "Salivary risk": f"Buffering: {_txt(row, 'buffering_capacity')}; pH: {_txt(row, 'salivary_ph')}; quantity: {_txt(row, 'salivary_quantity')}",
        "Caries risk": f"Mutans: {_txt(row, 'mutans_load_in_saliva')}; lactobacilli: {_txt(row, 'lactobacilli_load_in_saliva')}",
    }
    return mapping.get(domain, "Triggered by the entered clinical or modifiable-risk information relevant to this domain.")
