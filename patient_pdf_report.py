from __future__ import annotations

from io import BytesIO
from datetime import datetime
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
)


def _txt(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "Unknown"
    return str(value)


def _p(text, style):
    return Paragraph(escape(_txt(text)).replace("\n", "<br/>"), style)


def _table(data, widths=None, header=True):
    tbl = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D4DCE6")),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        commands += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0E5A8A")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
    tbl.setStyle(TableStyle(commands))
    return tbl


def build_patient_pdf_report(
    *,
    patient_id,
    model_name,
    profile_df,
    concern_df,
    ai_df,
    factor_df,
    priorities,
    modifiable,
    recommendations_df,
    care_phases,
    guideline_references,
):
    """Return a detailed clinician-reviewable patient report as PDF bytes.

    The report documents the recorded clinical profile, current-state AI estimates,
    non-causal model attributions, and guideline-based recommendations. It does not
    diagnose, prescribe, or forecast future disease.
    """
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=14 * mm,
        leftMargin=14 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title="Dental AI Coach - Detailed Patient Report",
        author="Dental AI Coach",
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle", parent=styles["Title"], fontSize=18, leading=22,
        textColor=colors.HexColor("#123B63"), alignment=TA_CENTER, spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="Section", parent=styles["Heading2"], fontSize=12, leading=15,
        textColor=colors.HexColor("#0E5A8A"), spaceBefore=8, spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="Small", parent=styles["BodyText"], fontSize=8.2, leading=10.5,
        textColor=colors.HexColor("#394B5A"),
    ))
    styles.add(ParagraphStyle(
        name="BodyCompact", parent=styles["BodyText"], fontSize=9, leading=12,
    ))

    story = []
    story.append(Paragraph("Dental AI Coach", styles["ReportTitle"]))
    story.append(Paragraph("Detailed Personalized Oral-Health Decision-Support Report", styles["Heading2"]))
    story.append(Spacer(1, 4))

    meta = [
        [_p("Patient / case", styles["Small"]), _p(patient_id, styles["Small"]),
         _p("Selected AI model", styles["Small"]), _p(model_name, styles["Small"])],
        [_p("Generated", styles["Small"]), _p(datetime.now().strftime("%Y-%m-%d %H:%M"), styles["Small"]),
         _p("Report type", styles["Small"]), _p("Clinician-reviewable research decision support", styles["Small"])],
    ]
    story.append(_table(meta, widths=[28*mm, 54*mm, 32*mm, 60*mm], header=False))
    story.append(Spacer(1, 7))
    story.append(_p(
        "Important: observed clinical findings remain the clinical reference. AI estimates describe current-state patterns from nonclinical factors. SHAP-style attributions explain fitted-model behavior and are not causal. Recommendations require clinician confirmation.",
        styles["Small"],
    ))

    # 1. Clinical profile
    story.append(Paragraph("1. Detailed Elham oral-health profile", styles["Section"]))
    total = float(pd.to_numeric(profile_df.get("count", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    present = int((pd.to_numeric(profile_df.get("count", pd.Series(dtype=float)), errors="coerce").fillna(0) > 0).sum())
    story.append(_p(f"Calculated descriptive Elham total: {total:.0f} | Recorded findings present: {present}", styles["BodyCompact"]))
    pdata = [[_p("Clinical component", styles["Small"]), _p("Count", styles["Small"]), _p("Status", styles["Small"])]]
    for _, r in profile_df.iterrows():
        pdata.append([
            _p(r.get("component", ""), styles["Small"]),
            _p(f"{float(r.get('count', 0)):.0f}", styles["Small"]),
            _p(r.get("status", ""), styles["Small"]),
        ])
    story.append(_table(pdata, widths=[110*mm, 25*mm, 40*mm]))

    # 2. Concern map
    story.append(Paragraph("2. Clinical concern map", styles["Section"]))
    cdata = [[_p("Domain", styles["Small"]), _p("Concern", styles["Small"]), _p("Why", styles["Small"])]]
    for _, r in concern_df.iterrows():
        cdata.append([_p(r.get("Domain", ""), styles["Small"]), _p(r.get("Concern", ""), styles["Small"]), _p(r.get("Why", ""), styles["Small"])])
    story.append(_table(cdata, widths=[48*mm, 25*mm, 102*mm]))
    story.append(_p("Concern categories are heuristic review aids, not validated diagnostic or prognostic scores.", styles["Small"]))

    # 3. AI estimates
    story.append(Paragraph("3. Component-specific AI estimates", styles["Section"]))
    adata = [[_p("Component", styles["Small"]), _p("Observed", styles["Small"]), _p("AI estimate", styles["Small"]), _p("Reliability", styles["Small"]), _p("Use", styles["Small"])]]
    for _, r in ai_df.iterrows():
        adata.append([
            _p(r.get("Clinical component", ""), styles["Small"]),
            _p(f"{float(r.get('Observed clinical count', 0)):.0f}", styles["Small"]),
            _p(f"{float(r.get('AI-estimated count from patient factors', 0)):.2f}", styles["Small"]),
            _p(r.get("Model reliability", ""), styles["Small"]),
            _p(r.get("Clinical use", ""), styles["Small"]),
        ])
    story.append(_table(adata, widths=[50*mm, 22*mm, 27*mm, 42*mm, 34*mm]))

    # 4. Factors
    story.append(Paragraph("4. Model-prioritized patient factors", styles["Section"]))
    if factor_df is not None and not factor_df.empty:
        fdata = [[_p("Patient factor", styles["Small"]), _p("Relative model influence", styles["Small"]), _p("Net model direction", styles["Small"])]]
        for _, r in factor_df.head(10).iterrows():
            fdata.append([
                _p(r.get("Patient factor", ""), styles["Small"]),
                _p(f"{float(r.get('Relative model influence', 0)):.3f}", styles["Small"]),
                _p(r.get("Net direction across modeled findings", ""), styles["Small"]),
            ])
        story.append(_table(fdata, widths=[75*mm, 45*mm, 55*mm]))
    else:
        story.append(_p("Patient-level model attribution was unavailable.", styles["BodyCompact"]))
    story.append(_p("These factors are model-attribution priorities and must not be interpreted as causal treatment targets.", styles["Small"]))

    # 5. Clinical priorities/modifiable factors
    story.append(Paragraph("5. Clinical priorities and modifiable factors", styles["Section"]))
    story.append(Paragraph("Clinical priorities", styles["Heading3"]))
    for item in priorities or ["No major entered Elham finding triggered a clinical-priority rule."]:
        story.append(_p("• " + item, styles["BodyCompact"]))
    story.append(Paragraph("Modifiable factors for review", styles["Heading3"]))
    for item in modifiable or ["No prespecified modifiable trigger was detected."]:
        story.append(_p("• " + item, styles["BodyCompact"]))

    # 6. Recommendations
    story.append(Paragraph("6. Guideline-based personalized action plan", styles["Section"]))
    rec = recommendations_df.copy() if recommendations_df is not None else pd.DataFrame()
    if not rec.empty and "care_phase" not in rec.columns:
        rec["care_phase"] = "Clinical review"
    if rec.empty:
        story.append(_p("No guideline-based recommendation rule was triggered by the entered data.", styles["BodyCompact"]))
    else:
        for phase in care_phases:
            subset = rec.loc[rec["care_phase"] == phase]
            if subset.empty:
                continue
            story.append(Paragraph(escape(str(phase)), styles["Heading3"]))
            for _, r in subset.iterrows():
                block = [
                    _p(f"{r.get('priority', '')} priority — {r.get('domain', '')}", styles["BodyCompact"]),
                    _p(r.get("recommendation", ""), styles["BodyCompact"]),
                    _p("Rationale: " + _txt(r.get("rationale", "")), styles["Small"]),
                    _p("Guideline basis: " + _txt(r.get("evidence_source", "")), styles["Small"]),
                    _p("Clinical confirmation required before acting.", styles["Small"]),
                    Spacer(1, 4),
                ]
                story.append(KeepTogether(block))

    # 7. Guideline set
    story.append(PageBreak())
    story.append(Paragraph("7. Guideline sources used by the recommendation engine", styles["Section"]))
    for ref in guideline_references:
        story.append(_p(f"• {ref.get('short', '')}: {ref.get('scope', '')}", styles["Small"]))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Interpretation and safety statement", styles["Section"]))
    story.append(_p(
        "This report is generated by a research decision-support prototype. It does not replace history taking, direct clinical examination, radiographic assessment when indicated, diagnosis, treatment planning, informed consent, or professional judgment. The current models are internally evaluated on cross-sectional data and do not forecast future disease. Medication, fluoride concentration/application, operative technique, treatment timing, and definitive recall interval must be selected by the treating clinician according to the individual patient and applicable guidance.",
        styles["Small"],
    ))

    doc.build(story)
    return buf.getvalue()
