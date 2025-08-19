# ses_utils.py
# Lightweight helpers to integrate socio-economic (SES) factors in your Streamlit app.

import re
import numpy as np
import pandas as pd

# -------- fuzzy header map: canonical -> substrings to look for ----------
SES_FUZZY = {
    "school":              ["school"],
    "grade":               ["grade"],
    "house_ownership":     ["house_ow", "ownership"],
    "i_live_with":         ["i_live_with", "live_with"],
    "average_income":      ["average_in", "family_in", "income"],
    "pocket_money":        ["pocket_m", "allowance"],
    "father_s_education":  ["father_s_e", "father edu"],
    "mother_s_education":  ["mother_s_e", "mother edu"],
    "father_s_job":        ["father_s_j", "father job", "father occ"],
    "mother_s_job":        ["mother_s_j", "mother job", "mother occ"],
    "insurance":           ["insurance"],
    "access_to":           ["access_to", "access"],
    "frequency":           ["frequency", "visit"],
    "affordability":       ["afford"],
}

# -------------------- normalizers (compact categories) -------------------
def _t(x): return str(x).strip().lower()

def norm_school(v):
    s = _t(v)
    if any(k in s for k in ["public","government","gov","experimental"]): return "Public"
    if any(k in s for k in ["international","american","british","german","french","canadian","igcse","ib","intl"]): return "International"
    if "private" in s or "independent" in s: return "Private"
    return "Unknown"

def norm_grade(v):
    s = _t(v)
    if any(k in s for k in ["kg","nursery","primary","grade 1","grade 2","grade 3","grade 4","grade 5","grade 6"]): return "Primary"
    if any(k in s for k in ["prep","grade 7","grade 8","grade 9"]): return "Preparatory"
    if any(k in s for k in ["sec","grade 10","grade 11","grade 12"]): return "Secondary"
    return "Unknown"

def norm_house_ownership(v):
    s = _t(v)
    if "own" in s: return "Owned"
    if "rent" in s: return "Rented"
    return "Other/Unknown"

def norm_live_with(v):
    s = _t(v)
    if any(k in s for k in ["father and mother","both parents","two parents"]): return "Two parents"
    if any(k in s for k in ["single","mother only","father only","one parent"]): return "Single parent"
    if any(k in s for k in ["relative","grand","aunt","uncle","guardian","care"]): return "Relatives/Other"
    return "Unknown"

def _num(x):
    s = re.sub(r"[^\d.]", "", str(x))
    try: return float(s) if s else None
    except: return None

def norm_parent_edu(v):
    s = _t(v)
    if any(k in s for k in ["phd","master","msc","postgrad","doctor"]): return "Postgrad"
    if any(k in s for k in ["uni","college","bsc","ba","license","licence"]): return "University"
    if any(k in s for k in ["secondary","high school","prep"]): return "Secondary"
    if "primary" in s or "elementary" in s: return "Primary"
    return "Unknown"

def norm_job(v):
    s = _t(v)
    if any(k in s for k in ["not working","no job","housewife","unemployed"]): return "Not working"
    if any(k in s for k in ["manager","engineer","doctor","dentist","pharmacist","teacher","accountant","lawyer"]): return "Professional/Manager"
    if s and s != "unknown": return "Worker/Clerk"
    return "Unknown"

def norm_insurance(v):
    s = _t(v)
    if s in {"yes","insured","y","1","covered"}: return "Insured"
    if s in {"no","uninsured","n","0"}:          return "Uninsured"
    return "Unknown"

def norm_access(v):
    s = _t(v)
    if any(k in s for k in ["easy","available","near"]): return "Easy"
    if any(k in s for k in ["hard","difficult","far","limited"]): return "Difficult"
    if s in {"moderate","average"}: return "Moderate"
    return "Unknown"

def norm_afford(v):
    s = _t(v)
    if any(k in s for k in ["cannot","can't","no","unaffordable"]): return "No"
    if any(k in s for k in ["hard","difficult","sometimes","partial"]): return "Hard"
    if any(k in s for k in ["yes","afford","can"]): return "Yes"
    return "Unknown"

def norm_visit_freq(v):
    s = _t(v)
    if any(k in s for k in ["6","12","regular","check","year","every"]): return "Regular"
    if any(k in s for k in ["pain","emergency","only when"]):           return "Pain-only"
    if "never" in s:                                                    return "Never"
    return "Occasional"

NORMALIZERS = {
    "school": norm_school,
    "grade": norm_grade,
    "house_ownership": norm_house_ownership,
    "i_live_with": norm_live_with,
    "father_s_education": norm_parent_edu,
    "mother_s_education": norm_parent_edu,
    "father_s_job": norm_job,
    "mother_s_job": norm_job,
    "insurance": norm_insurance,
    "access_to": norm_access,
    "affordability": norm_afford,
    "frequency": norm_visit_freq,
}

# ---------------------- column detection + prep --------------------------
def find_cols(df: pd.DataFrame, fuzzy_map=SES_FUZZY) -> dict:
    """Return {canonical_name: actual_column or None} using substring search."""
    low = {c: c.lower() for c in df.columns}
    out = {}
    for canon, keys in fuzzy_map.items():
        hit = None
        for c, lc in low.items():
            if any(k in lc for k in keys):
                hit = c; break
        out[canon] = hit
    return out

def prepare_ses(df: pd.DataFrame, train_idx=None):
    """
    Normalize SES text columns and create tertile bands for income and pocket_money.
    Returns:
        df2               : updated DataFrame
        ses_cat_cols      : list of categorical SES columns to add to cat_cols
        raw_numeric_drop  : list of raw numeric SES columns to remove from num_cols
    """
    df2 = df.copy()
    ses_map = find_cols(df2)
    # normalize text SES columns if present
    for k, col in ses_map.items():
        if col is None: 
            continue
        if k in NORMALIZERS:
            df2[col] = df2[col].apply(NORMALIZERS[k])

    # compute income/pocket tertile bands on training subset (if provided), else on all rows
    def _bands(col_key, new_name):
        col = ses_map.get(col_key)
        if col is None:
            return None
        idx = train_idx if train_idx is not None else df2.index
        s = df2.loc[idx, col].apply(_num).dropna()
        if len(s) >= 10:
            q1, q2 = s.quantile([0.34, 0.67])
            df2[new_name] = df2[col].apply(lambda v: "Unknown" if _num(v) is None else ("Low" if _num(v) < q1 else ("Medium" if _num(v) < q2 else "High")))
            return (q1, q2)
        else:
            df2[new_name] = "Unknown"
            return None

    inc_qs = _bands("average_income", "income_band")
    pocket_qs = _bands("pocket_money", "pocket_band")

    # SES categorical columns to use in the model
    ses_cat_cols = []
    for key in ["school","grade","house_ownership","i_live_with",
                "father_s_education","mother_s_education",
                "father_s_job","mother_s_job",
                "insurance","access_to","frequency","affordability"]:
        col = ses_map.get(key)
        if col is not None and col in df2.columns:
            ses_cat_cols.append(col)
    # add derived bands
    if "income_band" in df2.columns: ses_cat_cols.append("income_band")
    if "pocket_band" in df2.columns: ses_cat_cols.append("pocket_band")

    # raw numeric SES to drop from num_cols if you’re using bands
    raw_numeric_drop = []
    for key in ["average_income","pocket_money"]:
        col = ses_map.get(key)
        if col is not None and col in df2.columns:
            raw_numeric_drop.append(col)

    meta = {
        "ses_map": ses_map,
        "income_quantiles": inc_qs,      # (q1,q2) or None
        "pocket_quantiles": pocket_qs,
    }
    return df2, ses_cat_cols, raw_numeric_drop, meta

# ------------------------- UI + explainability ---------------------------
def build_ses_ui(df: pd.DataFrame, ses_cat_cols):
    """Render SES select boxes; return (ses_vals: dict, used_cols: list)."""
    import streamlit as st
    ses_cols = [c for c in ses_cat_cols if c in df.columns]
    ses_vals = {}
    if not ses_cols:
        return ses_vals, []
    st.subheader("Socio-economic inputs")
    cols = st.columns(2)
    for i, c in enumerate(ses_cols):
        opts = sorted(df[c].astype(str).dropna().unique().tolist()) or ["Unknown"]
        default = df[c].mode(dropna=True).iloc[0] if not df[c].dropna().empty else opts[0]
        with cols[i % 2]:
            ses_vals[c] = st.selectbox(c, options=opts, index=(opts.index(default) if default in opts else 0))
    return ses_vals, ses_cols

def include_ses_in_row(X_row: pd.DataFrame, ses_vals: dict, ses_cols_ui: list, df: pd.DataFrame):
    """Add SES selections to the single-row DataFrame used for prediction."""
    for c in ses_cols_ui:
        fallback = df[c].mode(dropna=True).iloc[0] if not df[c].dropna().empty else "Unknown"
        X_row[c] = ses_vals.get(c, fallback)
    return X_row

def ses_shap_panel(grouped, ses_cols_ui, title="Socio-economic drivers (grouped SHAP)"):
    """Plot a simple bar chart for SES SHAP contributions. 'grouped' can be dict or list[(name,val)]."""
    import streamlit as st
    import matplotlib.pyplot as plt

    if isinstance(grouped, dict):
        data = [(k, grouped[k]) for k in ses_cols_ui if k in grouped]
    else:
        gdict = dict(grouped)
        data = [(k, gdict[k]) for k in ses_cols_ui if k in gdict]

    if not data:
        return

    data = sorted(data, key=lambda kv: abs(kv[1]), reverse=True)[:10]
    st.subheader(title)
    labels = [k for k,_ in data]
    vals   = [v for _,v in data]
    fig, ax = plt.subplots()
    ax.bar(labels, vals)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("SHAP contribution")
    ax.set_title("Top SES drivers")
    fig.tight_layout()
    st.pyplot(fig)

def fairness_audit(pipe, df, num_cols, cat_cols, target_col, ses_cols_ui):
    """MAE by SES subgroup on the hold-out split (80/20, rs=42)."""
    import streamlit as st
    from sklearn.model_selection import train_test_split
    import numpy as np
    if not ses_cols_ui:
        return
    with st.expander("Fairness check by SES (hold-out)"):
        X_all = df[num_cols + cat_cols].copy()
        y_all = df[target_col].astype(float).values
        X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
        yhat = pipe.predict(X_te)
        te = X_te.copy()
        te["_y"], te["_yhat"] = y_te, yhat
        for c in ses_cols_ui:
            try:
                g = te.groupby(te[c].astype(str)).apply(lambda g: float(np.mean(np.abs(g["_y"]-g["_yhat"])))).rename("MAE")
                st.markdown(f"**MAE by {c}**")
                st.dataframe(g.sort_values().to_frame())
            except Exception as e:
                st.caption(f"{c}: subgroup MAE not available ({e})")
