"""
Onco-Palliative Prognosis — Precision Survival Explorer
=======================================================

A Streamlit user interface that combines:
  1. Validated bedside prognostic scales, computed exactly from their published rules
       - Palliative Prognostic Index (PPI)            Morita et al. 1999
       - Palliative Prognostic Score (PaP)            Pirovano et al. 1999
       - Delirium PaP (D-PaP)                         Scarpi et al. 2011
       - modified Glasgow Prognostic Score (mGPS)     McMillan 2007/2013
  2. Data-driven survival models trained on a cohort dataset
       - Penalised Cox proportional-hazards model (survival curve, 30/90/180-day survival)
       - Random-forest classifier for 90-day mortality (with permutation importance)
     using ASA grade, age/sex, vital signs, TNM stage and the prognostic-scale variables.

Data
----
* "Synthetic demo cohort": simulated patients, generated in code, for demonstration only.
  Its numbers carry NO clinical meaning.
* "Upload CSV": your own (ethically approved, de-identified) cohort. Download the template
  from the sidebar to see the required columns.

Run
---
    pip install -r requirements.txt
    streamlit run app.py

Educational / research prototype. Not a medical device. Any model trained here must be
externally validated (TRIPOD+AI, PROBAST) before clinical use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------------------
REQUIRED_COLUMNS = {
    "age": "years",
    "sex": "M or F",
    "asa": "ASA physical status 1-5",
    "hr": "heart rate, beats/min",
    "sbp": "systolic BP, mmHg",
    "dbp": "diastolic BP, mmHg",
    "rr": "respiratory rate, breaths/min",
    "spo2": "SpO2, %",
    "temp": "temperature, °C",
    "t_stage": "T category 0-4 (Tis/T0 = 0)",
    "n_stage": "N category 0-3",
    "m_stage": "M category 0 or 1",
    "stage_group": "overall stage group 1-4 (cancer-specific AJCC/UICC)",
    "ecog": "ECOG 0-4",
    "kps": "Karnofsky 10-100",
    "pps": "Palliative Performance Scale 10-100",
    "oral_intake": "0 normal, 1 moderately reduced, 2 mouthfuls or less",
    "oedema": "0/1",
    "dyspnoea": "any dyspnoea 0/1 (PaP)",
    "dyspnoea_rest": "dyspnoea at rest 0/1 (PPI)",
    "delirium": "0/1",
    "anorexia": "0/1",
    "cps_weeks": "clinician prediction of survival, weeks",
    "wbc": "total WBC ×10^3/µL",
    "lymph_pct": "lymphocytes, %",
    "albumin": "serum albumin, g/L",
    "crp": "C-reactive protein, mg/L",
    "time_days": "follow-up time from assessment, days",
    "event": "1 = died, 0 = censored (alive at last follow-up)",
}

MODEL_FEATURES = [
    "age", "sex_male", "asa", "hr", "sbp", "rr", "spo2", "temp",
    "t_stage", "n_stage", "m_stage", "stage_group", "ecog", "pps",
    "ppi", "pap", "mgps",
]

FEATURE_LABELS = {
    "age": "Age", "sex_male": "Male sex", "asa": "ASA grade", "hr": "Heart rate",
    "sbp": "Systolic BP", "rr": "Respiratory rate", "spo2": "SpO₂", "temp": "Temperature",
    "t_stage": "T category", "n_stage": "N category", "m_stage": "Distant metastasis (M1)",
    "stage_group": "Stage group", "ecog": "ECOG", "pps": "PPS", "ppi": "PPI score",
    "pap": "PaP score", "mgps": "mGPS",
}

HORIZONS = [30, 90, 180]

# --------------------------------------------------------------------------------------
# Validated prognostic scales (vectorised: work on scalars or pandas Series)
# --------------------------------------------------------------------------------------

def ppi_score(pps, oral_intake, oedema, dyspnoea_rest, delirium):
    """Palliative Prognostic Index (Morita 1999). Range 0-15."""
    pps = np.asarray(pps, dtype=float)
    oi = np.asarray(oral_intake, dtype=float)
    pts = np.where(pps <= 20, 4.0, np.where(pps <= 50, 2.5, 0.0))
    pts = pts + np.select([oi >= 2, oi == 1], [2.5, 1.0], 0.0)
    pts = pts + 1.0 * np.asarray(oedema, float) + 3.5 * np.asarray(dyspnoea_rest, float) \
        + 4.0 * np.asarray(delirium, float)
    return pts


def cps_points(weeks):
    """PaP points for the clinical prediction of survival (weeks)."""
    w = np.asarray(weeks, dtype=float)
    return np.select([w > 12, w >= 11, w >= 7, w >= 5, w >= 3], [0.0, 2.0, 2.5, 4.5, 6.0], 8.5)


def pap_score(dyspnoea, anorexia, kps, cps_weeks, wbc, lymph_pct):
    """Palliative Prognostic Score (Pirovano 1999). Range 0-17.5."""
    wbc = np.asarray(wbc, float)
    lym = np.asarray(lymph_pct, float)
    kps = np.asarray(kps, float)
    pts = 1.0 * np.asarray(dyspnoea, float) + 1.5 * np.asarray(anorexia, float)
    pts = pts + np.where(kps <= 20, 2.5, 0.0) + cps_points(cps_weeks)
    pts = pts + np.select([wbc > 11, wbc > 8.5], [1.5, 0.5], 0.0)
    pts = pts + np.select([lym < 12, lym < 20], [2.5, 1.0], 0.0)
    return pts


def dpap_score(pap, delirium):
    """Delirium-PaP (Scarpi 2011): PaP + 2 points for delirium. Range 0-19.5."""
    return np.asarray(pap, float) + 2.0 * np.asarray(delirium, float)


def mgps_score(crp, albumin):
    """modified Glasgow Prognostic Score: CRP mg/L, albumin g/L."""
    crp = np.asarray(crp, float)
    alb = np.asarray(albumin, float)
    return np.where(crp <= 10, 0, np.where(alb < 35, 2, 1)).astype(float)


def interpret_ppi(s):
    if s > 6:
        return "Survival likely < 3 weeks", "high"
    if s > 4:
        return "Survival likely < 6 weeks", "mid"
    return "Survival likely > 6 weeks", "low"


def interpret_pap(s, delirium_version=False):
    a, b = (7.0, 12.0) if delirium_version else (5.5, 11.0)
    if s <= a:
        return "Group A · 30-day survival > 70%", "low"
    if s <= b:
        return "Group B · 30-day survival 30-70%", "mid"
    return "Group C · 30-day survival < 30%", "high"


def interpret_mgps(s):
    return {0: ("mGPS 0 · no systemic inflammation", "low"),
            1: ("mGPS 1 · raised CRP", "mid"),
            2: ("mGPS 2 · raised CRP + low albumin", "high")}[int(s)]


def add_derived_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ppi"] = ppi_score(df["pps"], df["oral_intake"], df["oedema"], df["dyspnoea_rest"], df["delirium"])
    df["pap"] = pap_score(df["dyspnoea"], df["anorexia"], df["kps"], df["cps_weeks"], df["wbc"], df["lymph_pct"])
    df["dpap"] = dpap_score(df["pap"], df["delirium"])
    df["mgps"] = mgps_score(df["crp"], df["albumin"])
    df["sex_male"] = (df["sex"].astype(str).str.upper().str[0] == "M").astype(int)
    return df

# --------------------------------------------------------------------------------------
# Synthetic demonstration cohort
# --------------------------------------------------------------------------------------

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@st.cache_data(show_spinner=False)
def make_synthetic_cohort(n: int = 1500, seed: int = 7) -> pd.DataFrame:
    """Simulated advanced-cancer cohort. FOR DEMONSTRATION ONLY — not real patients."""
    rng = np.random.default_rng(seed)
    stage = rng.choice([1, 2, 3, 4], n, p=[0.08, 0.15, 0.27, 0.50])
    frailty = rng.normal(0, 1, n) + 0.6 * (stage - 2.5)          # latent illness burden
    age = rng.normal(60, 12, n).clip(18, 95).round()
    sex = rng.choice(["M", "F"], n)

    m = ((stage == 4) & (rng.random(n) < 0.85)).astype(int)
    t = np.clip(np.round(1 + stage * 0.6 + rng.normal(0, 0.8, n)), 1, 4)
    nn = np.clip(np.round(stage * 0.6 + rng.normal(0, 0.8, n) - 0.5), 0, 3)

    pps = np.clip(np.round((70 - 15 * frailty + rng.normal(0, 8, n)) / 10) * 10, 10, 100)
    kps = np.clip(pps + rng.choice([-10, 0, 0, 10], n), 10, 100)
    ecog = np.select([kps >= 90, kps >= 70, kps >= 50, kps >= 30], [0, 1, 2, 3], 4)
    asa = np.clip(np.round(2 + 0.5 * frailty + 0.02 * (age - 60) + rng.normal(0, 0.5, n)), 1, 5)

    hr = (85 + 8 * frailty + rng.normal(0, 10, n)).round()
    sbp = (122 - 6 * frailty + rng.normal(0, 14, n)).round()
    dbp = (75 - 3 * frailty + rng.normal(0, 8, n)).round()
    rr = (16 + 2.5 * frailty + rng.normal(0, 2, n)).clip(10, 40).round()
    spo2 = (97 - 1.5 * frailty + rng.normal(0, 1.5, n)).clip(80, 100).round()
    temp = (37.0 + 0.15 * frailty + rng.normal(0, 0.4, n)).round(1)

    oral = np.select([frailty > 1.4, frailty > 0.5], [2, 1], 0)
    oedema = (rng.random(n) < _sigmoid(frailty - 1.0)).astype(int)
    dysp = (rng.random(n) < _sigmoid(frailty - 0.6)).astype(int)
    dysp_rest = (dysp & (rng.random(n) < _sigmoid(frailty - 0.5))).astype(int)
    delir = (rng.random(n) < _sigmoid(frailty - 1.9)).astype(int)
    anorex = (rng.random(n) < _sigmoid(frailty - 0.2)).astype(int)

    wbc = np.exp(rng.normal(np.log(7.5) + 0.15 * frailty, 0.3, n)).round(1)
    lymph = (22 - 5 * frailty + rng.normal(0, 5, n)).clip(2, 50).round(1)
    albumin = (38 - 4 * frailty + rng.normal(0, 3, n)).clip(15, 50).round()
    crp = np.exp(rng.normal(1.6 + 0.8 * frailty, 0.8, n)).round(1)
    mg = mgps_score(crp, albumin)

    lin = (0.9 * frailty + 0.015 * (age - 60) + 0.15 * (asa - 2) + 0.3 * m + 0.35 * delir
           + 0.25 * dysp_rest + 0.25 * mg + 0.03 * (97 - spo2))
    base_rate = np.log(2) / 120.0                       # median ~120 days when lin = 0
    true_t = rng.exponential(1.0 / (base_rate * np.exp(lin)))
    censor = rng.uniform(30, 450, n)
    time = np.maximum(np.minimum(true_t, censor), 1).round()
    event = (true_t <= censor).astype(int)

    # clinicians see the expected (not the actual) course and are optimistic (~1.3×), with wide error
    expected_median = np.log(2) / (base_rate * np.exp(lin))
    cps = np.clip(np.round(expected_median / 7 * 1.3 * np.exp(rng.normal(0, 0.6, n))), 1, 26)

    return pd.DataFrame(dict(
        age=age, sex=sex, asa=asa, hr=hr, sbp=sbp, dbp=dbp, rr=rr, spo2=spo2, temp=temp,
        t_stage=t, n_stage=nn, m_stage=m, stage_group=stage, ecog=ecog, kps=kps, pps=pps,
        oral_intake=oral, oedema=oedema, dyspnoea=dysp, dyspnoea_rest=dysp_rest,
        delirium=delir, anorexia=anorex, cps_weeks=cps, wbc=wbc, lymph_pct=lymph,
        albumin=albumin, crp=crp, time_days=time, event=event,
    ))


def template_csv() -> bytes:
    demo = make_synthetic_cohort(5, seed=1)
    return demo.to_csv(index=False).encode()

# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------

@st.cache_resource(show_spinner="Training survival models…")
def train_models(df: pd.DataFrame, seed: int = 42):
    data = add_derived_scores(df).dropna(subset=MODEL_FEATURES + ["time_days", "event"])
    train, test = train_test_split(data, test_size=0.3, random_state=seed, stratify=data["event"])

    # --- Cox proportional hazards (penalised) ---
    cph = CoxPHFitter(penalizer=0.05)
    cph.fit(train[MODEL_FEATURES + ["time_days", "event"]], duration_col="time_days", event_col="event")
    risk = cph.predict_partial_hazard(test[MODEL_FEATURES])
    c_index = concordance_index(test["time_days"], -risk, test["event"])

    # calibration at 90 days: predicted vs Kaplan-Meier observed, by risk decile
    s90 = cph.predict_survival_function(test[MODEL_FEATURES], times=[90]).T.iloc[:, 0].values
    calib = test[["time_days", "event"]].copy()
    calib["pred_death_90"] = 1 - s90
    calib["bin"] = pd.qcut(calib["pred_death_90"], q=10, labels=False, duplicates="drop")
    rows = []
    for b, g in calib.groupby("bin"):
        km = KaplanMeierFitter().fit(g["time_days"], g["event"])
        rows.append({"predicted": g["pred_death_90"].mean(),
                     "observed": 1 - float(km.survival_function_at_times(90).iloc[0]),
                     "n": len(g)})
    calibration = pd.DataFrame(rows)

    # --- Random forest: 90-day mortality (exclude patients censored before 90 days) ---
    def label90(d):
        keep = ~((d["event"] == 0) & (d["time_days"] < 90))
        d = d[keep]
        y = ((d["event"] == 1) & (d["time_days"] <= 90)).astype(int)
        return d, y

    tr90, ytr = label90(train)
    te90, yte = label90(test)
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=10, random_state=seed, n_jobs=-1)
    rf.fit(tr90[MODEL_FEATURES], ytr)
    p_te = rf.predict_proba(te90[MODEL_FEATURES])[:, 1]
    rf_auc = roc_auc_score(yte, p_te) if yte.nunique() > 1 else float("nan")
    rf_brier = brier_score_loss(yte, p_te)
    imp = permutation_importance(rf, te90[MODEL_FEATURES], yte, n_repeats=5, random_state=seed,
                                 scoring="roc_auc")
    importance = (pd.DataFrame({"feature": MODEL_FEATURES, "importance": imp.importances_mean})
                  .sort_values("importance", ascending=True))

    # bedside-score benchmarks on the same test set (higher score = higher risk)
    bench = {}
    for col in ["ppi", "pap", "dpap", "mgps"]:
        bench[col.upper().replace("DPAP", "D-PaP").replace("PAP", "PaP")] = \
            concordance_index(test["time_days"], -test[col], test["event"])

    return dict(cph=cph, c_index=c_index, calibration=calibration, rf=rf, rf_auc=rf_auc,
                rf_brier=rf_brier, importance=importance, bench=bench,
                train_means=train[MODEL_FEATURES].mean(), n_train=len(train), n_test=len(test),
                data=data)

# --------------------------------------------------------------------------------------
# UI helpers
# --------------------------------------------------------------------------------------
COLORS = {"low": "#2e7d55", "mid": "#c7741c", "high": "#b4432f"}


def score_card(col, title, value, max_value, label, level, help_text=""):
    with col:
        st.markdown(
            f"""
            <div style="border:1px solid #ddd6ca;border-radius:12px;padding:16px;background:#fff">
              <div style="font-size:13px;color:#5b6676;font-weight:600">{title}</div>
              <div style="font-size:40px;font-family:Georgia,serif;color:#13233a">{value:g}
                   <span style="font-size:14px;color:#5b6676">/ {max_value:g}</span></div>
              <div style="font-weight:600;color:{COLORS[level]}">{label}</div>
              <div style="font-size:12px;color:#5b6676;margin-top:6px">{help_text}</div>
            </div>""",
            unsafe_allow_html=True,
        )


def vital_alerts(p):
    alerts = []
    if p["spo2"] < 92: alerts.append(f"SpO₂ {p['spo2']}% (< 92%)")
    if p["rr"] > 24: alerts.append(f"Respiratory rate {p['rr']}/min (> 24)")
    if p["sbp"] < 90: alerts.append(f"Systolic BP {p['sbp']} mmHg (< 90)")
    if p["hr"] > 130: alerts.append(f"Heart rate {p['hr']}/min (> 130)")
    if p["temp"] >= 38.3 or p["temp"] < 35.5: alerts.append(f"Temperature {p['temp']} °C")
    return alerts

# --------------------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Onco-Palliative Prognosis", page_icon="📉", layout="wide")
    st.markdown(
        "<h1 style='font-family:Georgia,serif;font-weight:500;margin-bottom:0'>Onco-Palliative Prognosis</h1>"
        "<p style='color:#5b6676;margin-top:4px'>Precision survival explorer — validated bedside scales "
        "alongside data-driven survival models</p>", unsafe_allow_html=True)
    st.warning("Educational / research prototype. Not a medical device and not for clinical decisions "
               "without local validation.", icon="⚠️")

    # ---------------- Sidebar: data ----------------
    with st.sidebar:
        st.header("Cohort data")
        source = st.radio("Training data", ["Synthetic demo cohort", "Upload CSV"])
        if source == "Upload CSV":
            up = st.file_uploader("De-identified cohort (.csv)", type="csv")
            if up is None:
                st.info("Upload a CSV, or switch back to the synthetic demo cohort.")
                df = None
            else:
                df = pd.read_csv(up)
                missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
                if missing:
                    st.error("Missing columns: " + ", ".join(missing))
                    df = None
        else:
            n = st.slider("Synthetic cohort size", 300, 5000, 1500, step=100)
            df = make_synthetic_cohort(n)
            st.caption("🧪 Simulated patients. Model outputs illustrate the workflow only.")
        st.download_button("Download CSV template", template_csv(), "prognosis_template.csv", "text/csv")
        with st.expander("Column dictionary"):
            st.table(pd.DataFrame({"column": REQUIRED_COLUMNS.keys(), "meaning": REQUIRED_COLUMNS.values()}))

    # ---------------- Patient inputs ----------------
    st.subheader("Patient assessment")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**Demographics & fitness**")
        age = st.number_input("Age (years)", 18, 105, 66)
        sex = st.selectbox("Sex", ["M", "F"])
        asa = st.select_slider("ASA physical status", options=[1, 2, 3, 4, 5], value=3)
        ecog = st.select_slider("ECOG", options=[0, 1, 2, 3, 4], value=3)
        kps = st.select_slider("Karnofsky (KPS)", options=list(range(10, 101, 10)), value=40)
        pps = st.select_slider("Palliative Performance Scale (PPS)", options=list(range(10, 101, 10)), value=40)
    with c2:
        st.markdown("**Vital signs**")
        hr = st.number_input("Heart rate (/min)", 30, 220, 104)
        sbp = st.number_input("Systolic BP (mmHg)", 50, 250, 108)
        dbp = st.number_input("Diastolic BP (mmHg)", 30, 150, 66)
        rr = st.number_input("Respiratory rate (/min)", 6, 60, 24)
        spo2 = st.number_input("SpO₂ (%)", 60, 100, 93)
        temp = st.number_input("Temperature (°C)", 33.0, 42.0, 37.4, step=0.1)
    with c3:
        st.markdown("**Tumour (TNM)**")
        site = st.text_input("Primary site", "Non-small cell lung cancer")
        t_lbl = st.selectbox("T", ["Tis/T0", "T1", "T2", "T3", "T4"], index=3)
        n_lbl = st.selectbox("N", ["N0", "N1", "N2", "N3"], index=2)
        m_lbl = st.selectbox("M", ["M0", "M1"], index=1)
        default_stage = 3 if m_lbl == "M1" else 2
        stage_group = st.selectbox("Stage group (site-specific AJCC/UICC)", ["I", "II", "III", "IV"],
                                   index=default_stage, help="Stage grouping depends on the cancer site — enter it from the site-specific table.")
        if m_lbl == "M1" and stage_group != "IV":
            st.caption("⚠️ M1 is usually stage IV — please check.")
    with c4:
        st.markdown("**Symptoms, labs & clinician estimate**")
        oral = st.radio("Oral intake", ["Normal", "Moderately reduced", "Mouthfuls or less"], index=1, horizontal=False)
        s1, s2 = st.columns(2)
        dysp = s1.checkbox("Dyspnoea", True)
        dysp_rest = s2.checkbox("Dyspnoea at rest", True)
        oed = s1.checkbox("Oedema", True)
        delir = s2.checkbox("Delirium", False)
        anor = s1.checkbox("Anorexia", True)
        cps = st.number_input("Clinician predicted survival (weeks)", 1, 52, 6)
        l1, l2 = st.columns(2)
        wbc = l1.number_input("WBC ×10³/µL", 0.5, 60.0, 10.4, step=0.1)
        lym = l2.number_input("Lymphocytes %", 0.0, 90.0, 14.0, step=0.5)
        alb = l1.number_input("Albumin g/L", 10.0, 55.0, 31.0, step=0.5)
        crp = l2.number_input("CRP mg/L", 0.0, 400.0, 42.0, step=1.0)

    patient = dict(
        age=age, sex=sex, asa=asa, hr=hr, sbp=sbp, dbp=dbp, rr=rr, spo2=spo2, temp=temp,
        t_stage=["Tis/T0", "T1", "T2", "T3", "T4"].index(t_lbl), n_stage=int(n_lbl[1]),
        m_stage=int(m_lbl[1]), stage_group=["I", "II", "III", "IV"].index(stage_group) + 1,
        ecog=ecog, kps=kps, pps=pps,
        oral_intake=["Normal", "Moderately reduced", "Mouthfuls or less"].index(oral),
        oedema=int(oed), dyspnoea=int(dysp), dyspnoea_rest=int(dysp_rest), delirium=int(delir),
        anorexia=int(anor), cps_weeks=cps, wbc=wbc, lymph_pct=lym, albumin=alb, crp=crp,
    )
    pdf = add_derived_scores(pd.DataFrame([patient]))
    prow = pdf.iloc[0]

    alerts = vital_alerts(patient)
    if alerts:
        st.error("Vital-sign alerts: " + " · ".join(alerts))

    tab1, tab2, tab3, tab4 = st.tabs(["Bedside scores", "Survival model", "Cohort & validation", "About"])

    # ---------------- Tab 1: validated scores ----------------
    with tab1:
        k1, k2, k3, k4 = st.columns(4)
        lbl, lev = interpret_ppi(prow.ppi)
        score_card(k1, "Palliative Prognostic Index", prow.ppi, 15, lbl, lev, "PPS, intake, oedema, dyspnoea at rest, delirium")
        lbl, lev = interpret_pap(prow.pap)
        score_card(k2, "Palliative Prognostic Score", prow.pap, 17.5, lbl, lev, "Dyspnoea, anorexia, KPS, clinician estimate, WBC, lymphocytes")
        lbl, lev = interpret_pap(prow.dpap, delirium_version=True)
        score_card(k3, "Delirium-PaP", prow.dpap, 19.5, lbl, lev, "PaP + 2 points for delirium")
        lbl, lev = interpret_mgps(prow.mgps)
        score_card(k4, "modified Glasgow Prognostic Score", prow.mgps, 2, lbl, lev, "CRP > 10 mg/L and albumin < 35 g/L")
        st.caption(f"{site} · T{patient['t_stage']} N{patient['n_stage']} M{patient['m_stage']} · stage {stage_group} · "
                   f"ASA {asa} · ECOG {ecog}")

        horizons = []
        for name, (_, lev) in {"PPI": interpret_ppi(prow.ppi), "PaP": interpret_pap(prow.pap),
                               "D-PaP": interpret_pap(prow.dpap, True)}.items():
            horizons.append({"low": "months", "mid": "weeks", "high": "days"}[lev])
        agree = len(set(horizons)) == 1
        st.info(f"Bedside horizon: **{max(set(horizons), key=horizons.count)}** "
                + ("— scales agree." if agree else "— scales disagree; re-assess and weigh clinical judgement."))

    # ---------------- Model training ----------------
    if df is None:
        for t in (tab2, tab3):
            with t:
                st.info("Provide cohort data in the sidebar to train the survival models.")
        models = None
    else:
        models = train_models(df)

    # ---------------- Tab 2: individual prediction ----------------
    if models is not None:
        with tab2:
            if source == "Synthetic demo cohort":
                st.caption("🧪 Trained on SIMULATED data — numbers below demonstrate the method only.")
            cph = models["cph"]
            X = pdf[MODEL_FEATURES]
            times = np.arange(0, 366)
            sf = cph.predict_survival_function(X, times=times).iloc[:, 0]
            surv = {h: float(sf.loc[h]) for h in HORIZONS}
            median = float(np.asarray(cph.predict_median(X), dtype=float).ravel()[0])
            p90_rf = float(models["rf"].predict_proba(X)[:, 1][0])

            m1, m2, m3, m4, m5 = st.columns(5)
            for col, h in zip((m1, m2, m3), HORIZONS):
                col.metric(f"{h}-day survival (Cox)", f"{surv[h]*100:.0f}%")
            m4.metric("Median survival (Cox)", "> 1 year" if np.isinf(median) else f"{median:.0f} days")
            m5.metric("90-day mortality (RF)", f"{p90_rf*100:.0f}%")

            left, right = st.columns([3, 2])
            with left:
                fig = go.Figure()
                km = KaplanMeierFitter().fit(models["data"]["time_days"], models["data"]["event"])
                kmdf = km.survival_function_.reset_index()
                fig.add_trace(go.Scatter(x=kmdf["timeline"], y=kmdf["KM_estimate"], name="Cohort (Kaplan-Meier)",
                                         line=dict(color="#9aa6b6", shape="hv")))
                fig.add_trace(go.Scatter(x=times, y=sf.values, name="This patient (Cox)",
                                         line=dict(color="#1f6f73", width=3)))
                for h in HORIZONS:
                    fig.add_vline(x=h, line_dash="dot", line_color="#ddd6ca")
                fig.update_layout(title="Predicted survival curve", xaxis_title="Days from assessment",
                                  yaxis_title="Survival probability", yaxis_range=[0, 1.02], xaxis_range=[0, 365],
                                  height=420, legend=dict(orientation="h", y=-0.2))
                st.plotly_chart(fig, use_container_width=True)
            with right:
                coefs = cph.params_
                contrib = ((X.iloc[0] - models["train_means"]) * coefs).rename("log_hr").reset_index()
                contrib.columns = ["feature", "log_hr"]
                contrib["label"] = contrib["feature"].map(FEATURE_LABELS)
                contrib = contrib.reindex(contrib["log_hr"].abs().sort_values().index).tail(10)
                figc = px.bar(contrib, x="log_hr", y="label", orientation="h",
                              color=contrib["log_hr"] > 0, color_discrete_map={True: "#b4432f", False: "#1f6f73"},
                              title="What drives this patient's risk (vs cohort average)")
                figc.update_layout(showlegend=False, xaxis_title="Contribution to log-hazard", yaxis_title="", height=420)
                st.plotly_chart(figc, use_container_width=True)

            with st.expander("Cox model coefficients (hazard ratios)"):
                summ = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].copy()
                summ.index = [FEATURE_LABELS[i] for i in summ.index]
                st.dataframe(summ.style.format("{:.3f}"), use_container_width=True)

        # ---------------- Tab 3: validation ----------------
        with tab3:
            st.markdown(f"Training set **{models['n_train']}** · held-out test set **{models['n_test']}**")
            v1, v2, v3 = st.columns(3)
            v1.metric("Cox C-index (test)", f"{models['c_index']:.3f}")
            v2.metric("RF 90-day AUC (test)", f"{models['rf_auc']:.3f}")
            v3.metric("RF 90-day Brier score", f"{models['rf_brier']:.3f}")

            b = pd.DataFrame({"model": ["Cox (all features)"] + list(models["bench"].keys()),
                              "c_index": [models["c_index"]] + list(models["bench"].values())})
            figb = px.bar(b, x="model", y="c_index", text=b["c_index"].round(3), title="Discrimination on the test set (C-index)",
                          color=b["model"].str.startswith("Cox"), color_discrete_map={True: "#1f6f73", False: "#9aa6b6"})
            figb.update_layout(showlegend=False, yaxis_range=[0.5, 1], height=360, xaxis_title="", yaxis_title="C-index")

            cal = models["calibration"]
            figcal = go.Figure()
            figcal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect", line=dict(dash="dash", color="#9aa6b6")))
            figcal.add_trace(go.Scatter(x=cal["predicted"], y=cal["observed"], mode="markers+lines", name="Cox, by risk decile",
                                        marker=dict(size=9, color="#1f6f73")))
            figcal.update_layout(title="Calibration: 90-day mortality", xaxis_title="Predicted", yaxis_title="Observed (Kaplan-Meier)",
                                 xaxis_range=[0, 1], yaxis_range=[0, 1], height=360)

            g1, g2 = st.columns(2)
            g1.plotly_chart(figb, use_container_width=True)
            g2.plotly_chart(figcal, use_container_width=True)

            imp = models["importance"].copy()
            imp["label"] = imp["feature"].map(FEATURE_LABELS)
            figi = px.bar(imp, x="importance", y="label", orientation="h", title="Random forest: permutation importance (Δ AUC)",
                          color_discrete_sequence=["#13233a"])
            figi.update_layout(height=480, yaxis_title="", xaxis_title="Mean decrease in AUC")
            st.plotly_chart(figi, use_container_width=True)

            st.markdown("**Kaplan-Meier by PPI band**")
            d = models["data"].copy()
            d["PPI band"] = pd.cut(d["ppi"], [-0.1, 4, 6, 15], labels=["≤ 4", "4.5–6", "> 6"])
            figkm = go.Figure()
            for band, g in d.groupby("PPI band", observed=True):
                k = KaplanMeierFitter().fit(g["time_days"], g["event"]).survival_function_.reset_index()
                figkm.add_trace(go.Scatter(x=k["timeline"], y=k["KM_estimate"], name=f"PPI {band} (n={len(g)})", line=dict(shape="hv")))
            figkm.update_layout(xaxis_title="Days", yaxis_title="Survival", xaxis_range=[0, 365], height=380)
            st.plotly_chart(figkm, use_container_width=True)

    # ---------------- Tab 4: about ----------------
    with tab4:
        st.markdown("""
**Validated scales (computed exactly)**
- **PPI** (Morita 1999): PPS 10–20 = 4, 30–50 = 2.5; oral intake severely reduced = 2.5, moderately = 1; oedema 1; dyspnoea at rest 3.5; delirium 4. > 6 → < 3 weeks; > 4 → < 6 weeks.
- **PaP** (Pirovano 1999): dyspnoea 1, anorexia 1.5, KPS ≤ 20 = 2.5, clinician estimate 0–8.5, WBC 0–1.5, lymphocytes 0–2.5. Groups A ≤ 5.5, B 5.6–11, C > 11.
- **D-PaP** (Scarpi 2011): PaP + 2 for delirium. Groups A ≤ 7, B 7.1–12, C > 12.
- **mGPS**: CRP ≤ 10 → 0; CRP > 10 with albumin ≥ 35 → 1; CRP > 10 with albumin < 35 → 2.

**Data-driven models**
- Penalised Cox proportional-hazards model on ASA, age, sex, vitals, TNM, stage group, ECOG, PPS, PPI, PaP and mGPS.
- Random forest for 90-day mortality (patients censored before day 90 excluded from that label).
- 70/30 train-test split; discrimination (C-index, AUC), calibration and Brier score reported.

**Before any clinical use**
- Train on real, representative Indian cohorts with ethics approval; report per TRIPOD+AI and assess bias with PROBAST.
- Validate externally (other centres, later time periods); check calibration, not just discrimination.
- Check performance across subgroups (sex, age, cancer site, socioeconomic setting).
- Link predictions to an action pathway (palliative referral, advance care planning) and keep the clinician-patient conversation central.
""")


if __name__ == "__main__":
    main()
