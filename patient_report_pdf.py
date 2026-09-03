from __future__ import annotations

from io import BytesIO
from datetime import datetime
from html import escape
import re
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

CLINICAL_COMPONENTS = [
    ("missing_0_including_wisdom_", "Missing teeth including wisdom teeth"),
    ("decayed_1", "Decayed teeth"),
    ("filled_2", "Filled teeth"),
    ("hypoplasia_3", "Hypoplasia"),
    ("hypocalcification_4", "Hypocalcification"),
    ("fluorosis_5", "Fluorosis"),
    ("erosion_6", "Erosion"),
    ("abrasion_7", "Abrasion"),
    ("attrition_8", "Attrition"),
    ("abfraction_9", "Abfraction"),
    ("sealant_a", "Sealants"),
    ("fractured_h", "Fractured teeth"),
    ("crown_pontic", "Crown pontics"),
    ("crown_abutment", "Crown abutments"),
    ("crown_implant", "Implant crowns"),
    ("veneer", "Veneers"),
]

REPORT_FACTOR_FIELDS = [
    ("age", "Age"), ("gender", "Gender"), ("grade", "Grade"),
    ("functional_status", "Functional status"), ("of_family_members", "Family size"),
    ("average_income", "Average income"), ("insurance", "Insurance"),
    ("access_to_oral_health_care", "Access to oral health care"),
    ("frequency_of_visits", "Dental visit frequency"), ("affordability", "Affordability"),
    ("tooth_brushing_frequency", "Toothbrushing frequency"),
    ("interdental_cleaning", "Interdental cleaning"), ("mouth_rinse", "Mouth rinse"),
    ("sugar", "Sugar exposure"), ("snacks_frequency", "Snack frequency"),
    ("snack_content", "Snack content"), ("carbonated_beverages", "Carbonated beverages"),
    ("acidic_food_or_drinks", "Acidic food/drinks"), ("smoking", "Smoking"),
    ("level_of_hydration", "Hydration"), ("salivary_consistency", "Salivary consistency"),
    ("salivary_ph", "Salivary pH"), ("salivary_quantity", "Salivary quantity"),
    ("buffering_capacity", "Buffering capacity"),
    ("mutans_load_in_saliva", "Mutans streptococci load"),
    ("lactobacilli_load_in_saliva", "Lactobacilli load"),
]

CARE_PHASE_ORDER = [
    "Awareness & education", "Prevention", "Disease-control / urgent needs",
    "Restorative considerations", "Maintenance & recall",
]


def _ascii_text(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "Unknown"
    s = str(value)
    s = s.replace("–", "-").replace("—", "-").replace("’", "'").replace("•", "-")
    s = re.sub(r"\s+", " ", s).strip()
    return s or "Unknown"


def _num(row, field):
    v = pd.to_numeric(row.get(field, 0), errors="coerce")
    return 0.0 if pd.isna(v) else float(v)


def _phase(domain: str) -> str:
    mapping = {
        "Caries": "Disease-control / urgent needs",
        "Dental trauma": "Disease-control / urgent needs",
        "Restorations": "Restorative considerations",
        "Developmental enamel defects": "Restorative considerations",
        "Missing/developing dentition": "Restorative considerations",
        "Tooth surface loss": "Prevention",
        "Diet / caries prevention": "Prevention",
        "Diet / tooth surface loss": "Prevention",
        "Salivary risk": "Prevention",
        "Caries risk": "Prevention",
        "Sealants": "Prevention",
        "Prevention": "Prevention",
        "Oral hygiene / fluoride": "Awareness & education",
        "Plaque control": "Awareness & education",
        "Tobacco / periodontal prevention": "Awareness & education",
        "Periodontal health": "Maintenance & recall",
    }
    return mapping.get(str(domain), "Maintenance & recall")


def _p(text, style):
    return Paragraph(escape(_ascii_text(text)), style)


def build_detailed_patient_pdf(row: dict, concern_df, clinical_priorities, modifiable_factors, rec_df) -> bytes:
    """Build a clinician-reviewable PDF from directly entered patient data and guideline rules."""
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, rightMargin=15*mm, leftMargin=15*mm,
        topMargin=14*mm, bottomMargin=14*mm,
        title="Dental AI Coach - Detailed Patient Report",
        author="Dental AI Coach",
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], alignment=TA_CENTER, fontSize=17, leading=21, spaceAfter=8))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], fontSize=12, leading=15, spaceBefore=8, spaceAfter=5, textColor=colors.HexColor("#17324D")))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8.5, leading=11))
    styles.add(ParagraphStyle(name="Body2", parent=styles["BodyText"], fontSize=9.5, leading=13))
    styles.add(ParagraphStyle(name="Warning", parent=styles["BodyText"], fontSize=8.5, leading=11, textColor=colors.HexColor("#7A3E00")))

    story = []
    story.append(_p("Dental AI Coach - Detailed Patient Report", styles["ReportTitle"]))
    story.append(_p("Clinician-reviewable oral-health profile and personalized action plan", styles["Body2"]))
    story.append(Spacer(1, 5))

    pid = _ascii_text(row.get("id", "New patient"))
    meta = [
        ["Patient/record ID", pid],
        ["Age", _ascii_text(row.get("age", "Unknown"))],
        ["Gender", _ascii_text(row.get("gender", "Unknown"))],
        ["Report generated", datetime.now().strftime("%Y-%m-%d %H:%M")],
    ]
    t = Table(meta, colWidths=[45*mm, 125*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#EAF1F7")),
        ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#B7C4D1")),
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 8.5),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 5), ("RIGHTPADDING", (0,0), (-1,-1), 5),
        ("TOPPADDING", (0,0), (-1,-1), 4), ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story += [t, Spacer(1, 7)]

    story.append(_p("1. Detailed Elham oral-health profile", styles["Section"]))
    total = sum(_num(row, f) for f, _ in CLINICAL_COMPONENTS)
    clinical_rows = [["Clinical finding", "Count", "Status"]]
    for field, label in CLINICAL_COMPONENTS:
        count = _num(row, field)
        clinical_rows.append([label, f"{count:.0f}", "Present" if count > 0 else "Not recorded"])
    clinical_rows.append(["Descriptive Elham total", f"{total:.0f}", "Descriptive summary"])
    ct = Table(clinical_rows, colWidths=[95*mm, 25*mm, 50*mm], repeatRows=1)
    ct.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#17324D")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#C6D0D9")),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("ROWBACKGROUNDS", (0,1), (-1,-2), [colors.white, colors.HexColor("#F7F9FB")]),
        ("BACKGROUND", (0,-1), (-1,-1), colors.HexColor("#EAF1F7")),
        ("FONTNAME", (0,-1), (-1,-1), "Helvetica-Bold"),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
    ]))
    story += [ct, Spacer(1, 7)]

    story.append(_p("2. Clinical concern map", styles["Section"]))
    if concern_df is not None and not concern_df.empty:
        rows = [["Domain", "Concern", "Why"]]
        for _, r in concern_df.iterrows():
            rows.append([_ascii_text(r.get("Domain")), _ascii_text(r.get("Concern")), _ascii_text(r.get("Why"))])
        tb = Table(rows, colWidths=[47*mm, 25*mm, 98*mm], repeatRows=1)
        tb.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#244E6F")), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#C6D0D9")),
            ("FONTSIZE", (0,0), (-1,-1), 7.5), ("VALIGN", (0,0), (-1,-1), "TOP"),
        ]))
        story += [tb, Spacer(1, 6)]
    story.append(_p("Concern levels are heuristic review aids, not validated diagnostic, prognostic, or risk scores.", styles["Warning"]))

    story.append(_p("3. Patient factors relevant to personalized support", styles["Section"]))
    factor_rows = [["Factor", "Recorded value"]]
    for field, label in REPORT_FACTOR_FIELDS:
        if field in row:
            value = _ascii_text(row.get(field))
            if value.lower() not in {"unknown", "nan", "none", ""}:
                factor_rows.append([label, value])
    ft = Table(factor_rows, colWidths=[75*mm, 95*mm], repeatRows=1)
    ft.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#244E6F")), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#C6D0D9")),
        ("FONTSIZE", (0,0), (-1,-1), 8), ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F7F9FB")]),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
    ]))
    story += [ft, Spacer(1, 7)]

    story.append(_p("4. Clinical priorities", styles["Section"]))
    if clinical_priorities:
        for x in clinical_priorities:
            story.append(_p("- " + x, styles["Body2"]))
    else:
        story.append(_p("No major clinical-priority rule was triggered by the entered findings.", styles["Body2"]))

    story.append(_p("5. Modifiable factors to review", styles["Section"]))
    if modifiable_factors:
        for x in modifiable_factors:
            story.append(_p("- " + x, styles["Body2"]))
    else:
        story.append(_p("No prespecified modifiable-factor rule was triggered.", styles["Body2"]))

    story.append(PageBreak())
    story.append(_p("6. Personalized guideline-based action plan", styles["Section"]))
    if rec_df is None or rec_df.empty:
        story.append(_p("No guideline-based recommendation rule was triggered.", styles["Body2"]))
    else:
        work = rec_df.copy()
        work["care_phase"] = work["domain"].map(_phase)
        for phase in CARE_PHASE_ORDER:
            phase_df = work.loc[work["care_phase"] == phase]
            if phase_df.empty:
                continue
            story.append(_p(phase, styles["Heading3"]))
            for _, r in phase_df.iterrows():
                heading = f"{_ascii_text(r.get('priority'))} priority - {_ascii_text(r.get('domain'))}"
                story.append(_p(heading, styles["Body2"]))
                story.append(_p(_ascii_text(r.get("recommendation")), styles["Body2"]))
                story.append(_p("Rationale: " + _ascii_text(r.get("rationale")), styles["Small"]))
                story.append(_p("Guideline basis: " + _ascii_text(r.get("evidence_source")), styles["Small"]))
                story.append(_p("Clinical confirmation required before acting.", styles["Warning"]))
                story.append(Spacer(1, 5))

    story.append(_p("7. Interpretation and safety", styles["Section"]))
    safety = [
        "Observed clinical findings remain the primary clinical reference.",
        "AI model attributions shown in the application are supportive and non-causal; they are not treatment indications.",
        "This report does not provide an autonomous diagnosis or prescription.",
        "Definitive diagnosis, radiographs, medications, fluoride concentration/application, operative technique, treatment timing, and recall interval require professional clinical judgment and applicable local guidance.",
        "The current analytical framework is cross-sectional and does not forecast future disease.",
    ]
    for s in safety:
        story.append(_p("- " + s, styles["Body2"]))

    story.append(Spacer(1, 8))
    story.append(_p("Dental AI Coach - clinician-reviewable research decision support", styles["Small"]))

    doc.build(story)
    return buf.getvalue()
