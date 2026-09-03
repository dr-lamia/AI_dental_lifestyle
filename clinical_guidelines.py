"""Guideline-based personalized recommendation engine for Dental AI Coach.

The engine is deliberately rule based. AI/SHAP may prioritize which recorded
factors deserve review, but model attribution is never treated as a causal
indication for treatment. Recommendations are anchored to current authoritative
dental guidance relevant to adolescents and permanent teeth.

Guideline set reviewed September 2026:
- AAPD Adolescent Oral Health Care, Reference Manual 2026-2027, latest revision 2025.
- AAPD Periodontal Conditions in Pediatric Dental Patients, revision 2024.
- AAPD Fluoride Therapy / Policy on Use of Fluoride, latest revision 2023.
- AAPD Caries-risk Assessment and Management for Infants, Children, and Adolescents,
  Reference Manual 2026-2027 (latest revision 2022).
- ADA Evidence-based Clinical Practice Guideline on Restorative Treatments for
  Caries Lesions, 2023.
- ADA/AAPD Evidence-based Guideline for Pit-and-Fissure Sealants (current guideline).
- AAPD Molar-Incisor Hypomineralization best practice (current Reference Manual).
- AAE Recommended Guidelines for Treatment of Traumatic Dental Injuries, 2026.

This module does not prescribe drugs, fluoride concentrations, radiographic
intervals, recall intervals, or definitive procedures without clinician review.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable
import pandas as pd


GUIDELINE_REFERENCES = [
    {
        "short": "AAPD Adolescent Oral Health Care (2025)",
        "scope": "Adolescent history, prevention, caries, fluoride, hygiene, diet, sealants, periodontal care, third molars, trauma and transition of care.",
    },
    {
        "short": "AAPD Periodontal Conditions in Pediatric Dental Patients (2024)",
        "scope": "Periodontal diagnosis, risk assessment, contributing-factor control, treatment and referral in children/adolescents.",
    },
    {
        "short": "AAPD Fluoride Therapy / Policy on Use of Fluoride (2023)",
        "scope": "Twice-daily fluoridated toothpaste and risk-based professionally applied fluoride.",
    },
    {
        "short": "AAPD Caries-risk Assessment and Management (Reference Manual 2026-2027)",
        "scope": "Risk-based diagnostics, prevention, diet counseling, sealants, fluoride and restorative/nonrestorative care.",
    },
    {
        "short": "ADA Restorative Treatments for Caries Lesions CPG (2023)",
        "scope": "Conservative/selective carious tissue removal and evidence-based restorative management of moderate/advanced lesions.",
    },
    {
        "short": "ADA/AAPD Pit-and-Fissure Sealants CPG (current)",
        "scope": "Sealants for sound or noncavitated occlusal surfaces of at-risk primary/permanent molars in children and adolescents.",
    },
    {
        "short": "AAPD Molar-Incisor Hypomineralization best practice (current)",
        "scope": "Diagnosis, prevention, sensitivity control, sealants and staged restorative/extraction considerations for hypomineralized teeth.",
    },
    {
        "short": "AAE Traumatic Dental Injuries Guidelines (2026)",
        "scope": "Contemporary evaluation and management pathways for permanent-tooth fractures and traumatic dental injuries.",
    },
]


@dataclass
class Recommendation:
    domain: str
    priority: str
    recommendation: str
    rationale: str
    evidence_source: str
    clinician_action: str = "Clinical confirmation required"

    def record(self):
        return asdict(self)


def _num(row, field):
    value = pd.to_numeric(row.get(field, 0), errors="coerce")
    return 0.0 if pd.isna(value) else float(value)


def _txt(row, field):
    value = row.get(field, "Unknown")
    if pd.isna(value):
        return "unknown"
    return str(value).strip().lower()


def _contains_any(text, terms):
    return any(term in text for term in terms)


def _add(items, domain, priority, recommendation, rationale, source):
    items.append(Recommendation(domain, priority, recommendation, rationale, source))


def build_guideline_action_plan(row: dict, prioritized_fields: Iterable[str] | None = None):
    """Return clinical priorities, modifiable factors, and guideline-based advice.

    `prioritized_fields` may come from SHAP and only changes display ordering. It
    does not create treatment indications.
    """
    prioritized_fields = list(prioritized_fields or [])
    items: list[Recommendation] = []
    clinical_priorities: list[str] = []
    modifiable_factors: list[str] = []

    decay = _num(row, "decayed_1")
    missing = _num(row, "missing_0_including_wisdom_")
    filled = _num(row, "filled_2")
    hypo = _num(row, "hypocalcification_4")
    fractured = _num(row, "fractured_h")
    erosion = _num(row, "erosion_6")
    attrition = _num(row, "attrition_8")
    abrasion = _num(row, "abrasion_7")
    abfraction = _num(row, "abfraction_9")
    sealants = _num(row, "sealant_a")

    # Clinical findings: examination drives treatment need, not the model.
    if decay > 0:
        clinical_priorities.append(f"Active/untreated caries burden to assess: {int(decay)} decayed tooth/teeth")
        _add(
            items, "Caries", "High",
            "Perform tooth- and surface-level caries assessment (activity, cavitation, cleansability, depth and pulpal/periapical status) before selecting nonrestorative or restorative care.",
            "Contemporary caries management is lesion- and risk-based rather than based only on a total caries count.",
            "AAPD Caries-risk Assessment and Management; ADA Caries CPG 2023",
        )
        _add(
            items, "Caries", "High",
            "For moderate/advanced permanent-tooth lesions requiring restoration, preserve sound tooth structure and consider conservative/selective carious-tissue removal when clinically appropriate rather than routine complete excavation.",
            "The 2023 ADA restorative guideline favors conservative caries removal approaches in appropriate primary and permanent teeth.",
            "ADA Restorative Treatments for Caries Lesions CPG (2023)",
        )

    if filled > 0:
        clinical_priorities.append(f"Existing restorations requiring routine review: {int(filled)}")
        _add(
            items, "Restorations", "Moderate",
            "Assess existing restorations clinically for symptoms, recurrent caries, marginal integrity, fracture and function; repair or replace only when findings justify intervention.",
            "Restorative decisions should be individualized and tooth-preserving rather than triggered by the presence of a restoration alone.",
            "ADA restorative evidence-based framework; AAPD Adolescent Oral Health Care (2025)",
        )

    if hypo > 0:
        clinical_priorities.append(f"Hypocalcified/hypomineralized teeth to characterize: {int(hypo)}")
        _add(
            items, "Developmental enamel defects", "Moderate",
            "Differentiate the developmental defect clinically (including whether the pattern is compatible with molar-incisor hypomineralization), record sensitivity/post-eruptive breakdown and caries susceptibility, and tailor prevention/restoration to severity.",
            "Management of hypomineralized teeth depends on defect severity, sensitivity, breakdown, age and restorability.",
            "AAPD Molar-Incisor Hypomineralization best practice",
        )
        _add(
            items, "Developmental enamel defects", "Moderate",
            "Use intensified preventive care for susceptible hypomineralized surfaces and consider sealant/protective restorative options when indicated; definitive treatment should follow direct examination and tooth prognosis.",
            "AAPD guidance includes fluoride, sealants, hypersensitivity management and staged restorative options.",
            "AAPD Molar-Incisor Hypomineralization best practice",
        )

    if fractured > 0:
        clinical_priorities.append(f"Traumatic/fractured teeth requiring injury-specific assessment: {int(fractured)}")
        _add(
            items, "Dental trauma", "High",
            "Classify each fracture and evaluate associated luxation/root or alveolar injury, pulp status and radiographic findings as indicated; manage according to the injury-specific permanent-tooth trauma pathway and schedule appropriate follow-up.",
            "Traumatic dental injuries require injury-specific diagnosis, timely treatment and follow-up rather than a generic restorative recommendation.",
            "AAE Recommended Guidelines for Traumatic Dental Injuries (2026)",
        )

    if any(v > 0 for v in [erosion, attrition, abrasion, abfraction]):
        clinical_priorities.append("Non-carious tooth-surface loss recorded")
        _add(
            items, "Tooth surface loss", "Moderate",
            "Document distribution and severity and investigate likely erosive, mechanical and/or functional contributors before restorative intervention; prioritize control of the suspected cause and monitor progression.",
            "Adolescent management should be diagnosis- and risk-factor based; irreversible restorative treatment should not precede etiologic assessment when avoidable.",
            "AAPD Adolescent Oral Health Care (2025)",
        )

    if missing > 0:
        clinical_priorities.append(f"Teeth recorded as missing (including wisdom teeth): {int(missing)}")
        _add(
            items, "Missing/developing dentition", "Moderate",
            "Verify which teeth are truly absent versus unerupted/developing third molars, previous extractions or congenitally missing teeth. Assess eruption, space and occlusion before assigning treatment need.",
            "Adolescent third molars and congenitally missing teeth require developmental/occlusal assessment; the current Elham variable includes wisdom teeth.",
            "AAPD Adolescent Oral Health Care (2025); AAPD Developing Dentition and Occlusion (2024)",
        )

    # Behavior and prevention.
    brushing = _txt(row, "tooth_brushing_frequency")
    if _contains_any(brushing, ["never", "rare", "once/day", "once daily", "once a day", "1-3 times/week"]):
        modifiable_factors.append("Toothbrushing frequency below the recommended twice-daily pattern")
        _add(
            items, "Oral hygiene / fluoride", "High",
            "Support twice-daily toothbrushing with fluoridated toothpaste and individualized brushing instruction; select any additional fluoride intervention according to age, caries risk and clinician assessment.",
            "AAPD identifies twice-daily fluoridated toothpaste as a core caries-preventive measure and recommends professional fluoride for individuals at caries risk.",
            "AAPD Fluoride Therapy / Policy on Use of Fluoride (2023)",
        )

    interdental = _txt(row, "interdental_cleaning")
    if interdental.startswith("no") or interdental in {"none", "never"}:
        modifiable_factors.append("No reported interdental cleaning")
        _add(
            items, "Plaque control", "Moderate",
            "Assess interdental plaque/gingival inflammation and teach a patient-appropriate interdental method where indicated, integrated with twice-daily toothbrushing.",
            "Plaque-control advice should be individualized to periodontal findings and the patient's ability to use the selected method.",
            "AAPD Periodontal Conditions in Pediatric Dental Patients (2024); AAPD Adolescent Oral Health Care (2025)",
        )

    sugar = _txt(row, "sugar")
    snacks = _txt(row, "snacks_frequency")
    snack_content = _txt(row, "snack_content")
    if _contains_any(sugar, ["daily", "once", "twice", "more than", "frequent"]) or _contains_any(snacks, ["daily", "once daily", "more than once", "frequent", "often"]) or _contains_any(snack_content, ["sweet", "cake", "carbohydrate", "junk", "chips"]):
        modifiable_factors.append("Frequent fermentable-carbohydrate/free-sugar exposure")
        _add(
            items, "Diet / caries prevention", "High",
            "Review a typical-day diet with emphasis on frequency and timing of free-sugar/fermentable-carbohydrate exposures; reduce repeated between-meal exposures and replace them with lower-cariogenic choices where feasible.",
            "AAPD caries-risk pathways include diet counseling and emphasize individualized management of behavioral disease indicators.",
            "AAPD Caries-risk Assessment and Management; AAPD Adolescent Oral Health Care (2025)",
        )

    carbonated = _txt(row, "carbonated_beverages") + " " + _txt(row, "carbonated_beverages_diet")
    acidic = _txt(row, "acidic_food_or_drinks")
    if _contains_any(carbonated, ["daily", "once/day", "twice", "more than", "frequent", "yes"]) or _contains_any(acidic, ["daily", "frequent", "yes"]):
        modifiable_factors.append("Frequent acidic/carbonated beverage or food exposure")
        _add(
            items, "Diet / tooth surface loss", "Moderate",
            "Reduce the frequency and prolonged oral contact of acidic/carbonated drinks, favor water for routine hydration, and assess whether erosive tooth wear is clinically present before escalating treatment.",
            "Dietary history and prevention are integral to adolescent oral-health and tooth-surface-loss management.",
            "AAPD Adolescent Oral Health Care (2025)",
        )

    smoking = _txt(row, "smoking")
    if smoking.startswith("yes") or "smoker" in smoking:
        modifiable_factors.append("Tobacco exposure")
        _add(
            items, "Tobacco / periodontal prevention", "High",
            "Provide developmentally appropriate brief tobacco-use counseling and cessation support/referral, and assess periodontal and oral mucosal status.",
            "Adolescent dental care should address tobacco/risk-taking behaviors and periodontal risk factors.",
            "AAPD Adolescent Oral Health Care (2025); AAPD Periodontal Conditions (2024)",
        )

    saliva_flags = []
    if "low" in _txt(row, "buffering_capacity") or "very low" in _txt(row, "buffering_capacity"):
        saliva_flags.append("reduced buffering")
    if "acid" in _txt(row, "salivary_ph"):
        saliva_flags.append("acidic salivary pH")
    if "low" in _txt(row, "salivary_quantity") or "low" in _txt(row, "level_of_hydration"):
        saliva_flags.append("low salivary flow/hydration category")
    if saliva_flags:
        modifiable_factors.append("Salivary vulnerability: " + ", ".join(dict.fromkeys(saliva_flags)))
        _add(
            items, "Salivary risk", "Moderate",
            "Confirm salivary findings clinically when relevant, review hydration, medications/medical causes and dietary acid/sugar exposure, and integrate the findings into the overall caries-risk assessment rather than treating a salivary value in isolation.",
            "Caries-risk management is multifactorial; salivary findings should modify the overall risk profile and preventive plan.",
            "AAPD Caries-risk Assessment and Management; AAPD Adolescent Oral Health Care (2025)",
        )

    mutans = _txt(row, "mutans_load_in_saliva")
    lacto = _txt(row, "lactobacilli_load_in_saliva")
    if "more" in mutans or "more" in lacto or "high" in mutans or "high" in lacto:
        modifiable_factors.append("Elevated microbial/salivary risk category")
        _add(
            items, "Caries risk", "Moderate",
            "Use the microbial result as one element of a multifactorial caries-risk assessment; prioritize plaque control, fluoride exposure and reduction of frequent fermentable-carbohydrate intake rather than treating the test result alone.",
            "Validated caries-risk frameworks combine disease indicators, protective factors and social/behavioral/clinical risk factors.",
            "AAPD Caries-risk Assessment and Management",
        )

    # Sealant opportunity is phrased as an assessment because the dataset does not
    # identify individual sound/noncavitated pits and fissures.
    if decay > 0 or _contains_any(sugar + " " + snacks, ["daily", "frequent", "more than", "twice"]):
        _add(
            items, "Sealants", "Moderate",
            "Assess erupted permanent molar pits and fissures for sealant eligibility, especially when surfaces are sound but at elevated caries risk or have noncavitated occlusal lesions; review existing sealants for retention and repair need.",
            "Sealants are recommended for at-risk sound or noncavitated occlusal molar surfaces in children/adolescents.",
            "ADA/AAPD Pit-and-Fissure Sealants CPG",
        )
    elif sealants > 0:
        _add(
            items, "Sealants", "Routine",
            "Review existing sealants at periodic examination for retention and repair/replacement when clinically indicated.",
            "Sealants require periodic review after placement.",
            "AAPD sealant policy / ADA-AAPD Sealants CPG",
        )

    # Periodontal variable may be available in existing cohort but is not required
    # for new-patient modeling. If present and abnormal, make the recommendation.
    periodontal = _txt(row, "periodontal_status")
    if periodontal not in {"unknown", "normal", "healthy", "none", ""}:
        _add(
            items, "Periodontal health", "High",
            "Confirm periodontal diagnosis using an age-appropriate periodontal examination, identify contributing factors, reinforce plaque control and provide treatment or specialist referral according to severity and response.",
            "AAPD 2024 guidance emphasizes periodontal diagnosis, risk assessment, contributing-factor control and care coordination/referral when needed.",
            "AAPD Periodontal Conditions in Pediatric Dental Patients (2024)",
        )

    if not items:
        _add(
            items, "Prevention", "Routine",
            "Continue individualized preventive care based on direct examination and updated caries/periodontal risk assessment, including twice-daily fluoridated toothpaste, diet review and appropriate professional preventive services.",
            "Preventive frequency and interventions should be individualized rather than assigned by a fixed app rule.",
            "AAPD Adolescent Oral Health Care (2025); AAPD Caries-risk Assessment and Management",
        )

    # SHAP ranking only prioritizes review order for modifiable factors; clinical
    # priority remains dominant and evidence source remains explicit.
    rank = {field: i for i, field in enumerate(prioritized_fields)}
    field_domain = {
        "tooth_brushing_frequency": "Oral hygiene / fluoride",
        "interdental_cleaning": "Plaque control",
        "sugar": "Diet / caries prevention",
        "snacks_frequency": "Diet / caries prevention",
        "snack_content": "Diet / caries prevention",
        "carbonated_beverages": "Diet / tooth surface loss",
        "carbonated_beverages_diet": "Diet / tooth surface loss",
        "acidic_food_or_drinks": "Diet / tooth surface loss",
        "smoking": "Tobacco / periodontal prevention",
        "buffering_capacity": "Salivary risk",
        "salivary_ph": "Salivary risk",
        "salivary_quantity": "Salivary risk",
        "level_of_hydration": "Salivary risk",
        "mutans_load_in_saliva": "Caries risk",
        "lactobacilli_load_in_saliva": "Caries risk",
    }
    domain_rank = {}
    for f, i in rank.items():
        d = field_domain.get(f)
        if d is not None:
            domain_rank[d] = min(i, domain_rank.get(d, 10_000))

    priority_weight = {"High": 0, "Moderate": 1, "Routine": 2}
    items.sort(key=lambda x: (priority_weight.get(x.priority, 9), domain_rank.get(x.domain, 10_000)))

    rec_df = pd.DataFrame([x.record() for x in items])
    return clinical_priorities, list(dict.fromkeys(modifiable_factors)), rec_df
