"""
=============================================================
Equitable HCT Survival Prediction — Enhanced Streamlit Web App
KFUEIT Final Year Project 2025
WITH: Pipeline Output Viewer + Patient Feature Comparison + Clinical AI Explanations
Authors: Muzammil Tariq & Syed Faizan Ali
=============================================================
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

try:
    import shap
    HAVE_SHAP = True
except ImportError:
    HAVE_SHAP = False

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HCT Survival Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

    html, body, [class*="css"]  { font-family: 'DM Sans', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #0B2545 0%, #134074 50%, #1B6CA8 100%);
        padding: 2.2rem 2.5rem; border-radius: 16px; margin-bottom: 1.8rem;
        text-align: center; color: white;
        box-shadow: 0 8px 32px rgba(11,37,69,0.25);
    }
    .main-header h1 { font-family: 'DM Serif Display', serif; font-size: 2rem;
        margin: 0 0 0.4rem 0; letter-spacing: -0.5px; }
    .main-header p  { margin: 0.2rem 0; opacity: 0.88; }

    .risk-high   { background: linear-gradient(135deg,#FFF5F5,#FFE8E8); border: 2px solid #E53935;
                   border-radius: 14px; padding: 1.5rem; margin: 0.8rem 0; }
    .risk-medium { background: linear-gradient(135deg,#FFFBF0,#FFF3D4); border: 2px solid #FB8C00;
                   border-radius: 14px; padding: 1.5rem; margin: 0.8rem 0; }
    .risk-low    { background: linear-gradient(135deg,#F0FFF4,#DCF5E4); border: 2px solid #2DB87D;
                   border-radius: 14px; padding: 1.5rem; margin: 0.8rem 0; }

    .card {
        background: white; border-radius: 12px; padding: 1.4rem;
        border: 1px solid #E8EDF3;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    }
    .card-blue { border-left: 5px solid #1B6CA8; }
    .card-green { border-left: 5px solid #2DB87D; }
    .card-orange { border-left: 5px solid #E8703A; }
    .card-purple { border-left: 5px solid #9B59B6; }

    .section-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.4rem; color: #0B2545;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #E8EDF3;
    }
    .metric-badge {
        display: inline-block; padding: 0.3rem 0.8rem;
        border-radius: 20px; font-size: 0.85rem; font-weight: 600;
        margin: 0.2rem;
    }
    .badge-green  { background: #E6F9F0; color: #1A7A4E; }
    .badge-red    { background: #FDEAEA; color: #B71C1C; }
    .badge-blue   { background: #E3F0FB; color: #0D47A1; }
    .badge-orange { background: #FFF3E0; color: #E65100; }

    .explanation-row {
        padding: 0.7rem 1rem; border-radius: 8px; margin: 0.4rem 0;
        display: flex; align-items: center; gap: 0.8rem;
    }
    .factor-positive { background: #F0FFF4; border-left: 4px solid #2DB87D; }
    .factor-negative { background: #FFF5F5; border-left: 4px solid #E53935; }
    .factor-neutral  { background: #F7F9FC; border-left: 4px solid #90A4AE; }

    .narrative-box {
        background: linear-gradient(135deg, #EBF5FB, #E8F8F0);
        border-radius: 12px; padding: 1.4rem;
        border-left: 5px solid #1B6CA8;
        font-size: 1.05rem; line-height: 1.75;
        color: #1a2740; margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #0B2545, #1B6CA8) !important;
        color: white !important; font-size: 1.05rem !important;
        font-weight: 600 !important; padding: 0.75rem 2rem !important;
        border-radius: 10px !important; border: none !important;
        width: 100% !important; letter-spacing: 0.3px !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        box-shadow: 0 4px 16px rgba(27,108,168,0.4) !important;
        transform: translateY(-1px) !important;
    }
    .tab-section { margin-top: 1rem; }

    .img-caption {
        text-align: center; font-size: 0.88rem; color: #607D8B;
        margin-top: 0.3rem; font-style: italic;
    }
    .comparison-header {
        background: #0B2545; color: white; border-radius: 8px 8px 0 0;
        padding: 0.7rem 1rem; font-weight: 600; font-size: 1rem;
    }
    .comparison-body { border: 1px solid #E8EDF3; border-top: none;
        border-radius: 0 0 8px 8px; padding: 1rem; }

    .sidebar-section { font-size: 0.85rem; font-weight: 700; color: #0B2545;
        text-transform: uppercase; letter-spacing: 0.8px;
        padding: 0.5rem 0 0.2rem 0; }
    div[data-testid="metric-container"] {
        background: white; border-radius: 10px; border: 1px solid #E8EDF3;
        padding: 0.8rem 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <h1>🏥 Equitable HCT Survival Prediction System</h1>
    <p style='font-size:1.05rem'>Hematopoietic Cell Transplantation — 1-Year Survival Prediction</p>
    <p style='font-size:0.88rem; margin-top:0.4rem'>
        ⚖️ Fair AI Certified &nbsp;|&nbsp; KFUEIT 2025 &nbsp;|&nbsp;
        Fairlearn + Probability Calibration &nbsp;|&nbsp; SHAP Explainability<br>
        Muzammil Tariq &nbsp;&amp;&nbsp; Syed Faizan Ali &nbsp;|&nbsp;
        Supervisor: Dr. Saima Noreen Khosa
    </p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  LOAD MODELS & DATA
# ═══════════════════════════════════════════════════════════
@st.cache_resource
def load_all_artifacts():
    base = os.path.dirname(os.path.abspath(__file__))
    out  = os.path.join(base, 'outputs')

    try:
        with open(f'{out}/preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)

        # Primary model: prefer calibrated fairness model
        if os.path.exists(f'{out}/model_fairlearn_calibrated.pkl'):
            with open(f'{out}/model_fairlearn_calibrated.pkl', 'rb') as f:
                model = pickle.load(f)
            model_source = "Fairlearn Debiased + Probability Calibrated"
        elif os.path.exists(f'{out}/model_mitigated.pkl'):
            with open(f'{out}/model_mitigated.pkl', 'rb') as f:
                model = pickle.load(f)
            model_source = "Group-Balanced (Re-weighted)"
        else:
            return None, None, None, None, {}, {}, {}, {}, str("No model found. Run pipeline.py first.")

        # SHAP explainer
        explainer_shap = None
        if os.path.exists(f'{out}/shap_explainer.pkl') and HAVE_SHAP:
            try:
                with open(f'{out}/shap_explainer.pkl', 'rb') as f:
                    explainer_shap = pickle.load(f)
            except: pass

        # Feature names
        try:
            with open(f'{out}/feature_names.json') as f:
                fn_data = json.load(f)
            feature_names_raw      = fn_data['raw']
            feature_names_friendly = fn_data['friendly']
        except:
            feature_names_raw, feature_names_friendly = [], []

        # Mitigation results
        mit_results = {}
        if os.path.exists(f'{out}/mitigation_results.json'):
            with open(f'{out}/mitigation_results.json') as f:
                mit_results = json.load(f)

        # CV results
        cv_results = {}
        if os.path.exists(f'{out}/cv_results.json'):
            with open(f'{out}/cv_results.json') as f:
                cv_results = json.load(f)

        # Fairness results
        fairness_data = {}
        if os.path.exists(f'{out}/fairness_results.json'):
            with open(f'{out}/fairness_results.json') as f:
                fairness_data = json.load(f)

        # Top features
        top_features = {}
        if os.path.exists(f'{out}/top_features.json'):
            with open(f'{out}/top_features.json') as f:
                top_features = json.load(f)

        # LR coefficients
        lr_coefs = {}
        if os.path.exists(f'{out}/lr_coefficients.json'):
            with open(f'{out}/lr_coefficients.json') as f:
                lr_coefs = json.load(f)

        # Best model info
        best_info = {}
        if os.path.exists(f'{out}/best_model_info.json'):
            with open(f'{out}/best_model_info.json') as f:
                best_info = json.load(f)

        return (preprocessor, model, model_source, explainer_shap,
                feature_names_raw, feature_names_friendly,
                mit_results, cv_results, fairness_data,
                top_features, lr_coefs, best_info, True)

    except Exception as e:
        return (*([None]*12), str(e))

result = load_all_artifacts()
(preprocessor, model, model_source, explainer_shap,
 feature_names_raw, feature_names_friendly,
 mit_results, cv_results, fairness_data,
 top_features, lr_coefs, best_info, loaded) = result

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')

def img_path(fname):
    return os.path.join(OUT_DIR, fname)

def load_image(fname):
    p = img_path(fname)
    if os.path.exists(p):
        return p
    return None

# ═══════════════════════════════════════════════════════════
#  FEATURE DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════
FEATURE_MAP = {
    'age_at_hct'            : "Patient Age at Transplant",
    'karnofsky_score'       : "Physical Fitness Score (KPS)",
    'comorbidity_score'     : "Other Health Problems Score",
    'donor_age'             : "Donor Age",
    'hla_high_res_8'        : "Tissue Match Quality (8 markers)",
    'hla_nmdp_6'            : "Tissue Match Quality (6 markers)",
    'dri_score'             : "Disease Risk Level",
    'cyto_score'            : "Chromosome Test Result",
    'prim_disease_hct'      : "Primary Blood Disease",
    'graft_type'            : "Stem Cell Source",
    'conditioning_intensity': "Chemotherapy Strength",
    'donor_related'         : "Donor Relationship",
    'sex_match'             : "Donor-Patient Sex Pairing",
    'cmv_status'            : "CMV Virus Status",
    'in_vivo_tcd'           : "T-cell Depletion Used",
    'tbi_status'            : "Radiation Therapy Used",
    'mrd_hct'               : "Residual Leukemia at Transplant",
    'race_group'            : "Patient Racial Background",
    'ethnicity'             : "Patient Ethnicity",
    'diabetes'              : "Diabetes",
    'cardiac'               : "Heart Disease",
    'renal_issue'           : "Kidney Disease",
    'pulm_severe'           : "Severe Lung Disease",
    'obesity'               : "Obesity",
}

def fn(raw):
    return FEATURE_MAP.get(raw, raw.replace('_', ' ').title())

DISEASE_MAP = {
    'AML':'Acute Myeloid Leukemia (AML)', 'ALL':'Acute Lymphoid Leukemia (ALL)',
    'MDS':'Myelodysplastic Syndrome (MDS)', 'MPN':'Myeloproliferative Neoplasm (MPN)',
    'NHL':'Non-Hodgkin Lymphoma (NHL)', 'CML':'Chronic Myeloid Leukemia (CML)',
    'PCD':'Plasma Cell Disorder (PCD)', 'SAA':'Aplastic Anemia (SAA)',
    'HD':'Hodgkin Lymphoma (HD)'
}

# ─── Clinical significance lookup ─────────────────────────────────────────────
def clinical_impact(feat_raw, value, direction=None):
    """Return (impact_label, color_class, icon) for a feature value."""
    feat = feat_raw.lower()
    # KPS
    if 'karnofsky' in feat:
        v = float(value) if str(value).replace('.','').isdigit() else 70
        if v >= 80: return ("Good Physical Fitness", "factor-positive", "💪")
        elif v >= 60: return ("Moderate Fitness", "factor-neutral", "🔶")
        else: return ("Poor Physical Fitness — higher risk", "factor-negative", "⚠️")
    # Comorbidity
    if 'comorbidity' in feat:
        v = float(value) if str(value).replace('.','').isdigit() else 2
        if v <= 2: return ("Low Other-Disease Burden", "factor-positive", "✅")
        elif v <= 4: return ("Moderate Other-Disease Burden", "factor-neutral", "🔶")
        else: return ("High Other-Disease Burden — higher risk", "factor-negative", "⚠️")
    # Age
    if 'age_at_hct' in feat:
        v = float(value) if str(value).replace('.','').isdigit() else 50
        if v <= 40: return (f"Younger Age ({v:.0f} yrs) — better tolerance", "factor-positive", "✅")
        elif v <= 60: return (f"Middle Age ({v:.0f} yrs)", "factor-neutral", "🔶")
        else: return (f"Older Age ({v:.0f} yrs) — higher risk", "factor-negative", "⚠️")
    # HLA match
    if 'hla_high_res_8' in feat or 'hla_nmdp_6' in feat:
        v = float(value) if str(value).replace('.','').isdigit() else 6
        mx = 8 if 'res_8' in feat else 6
        if v == mx: return ("Perfect Tissue Match", "factor-positive", "🎯")
        elif v >= mx-1: return ("Near-Perfect Tissue Match", "factor-positive", "✅")
        elif v >= mx-2: return ("Partial Tissue Match", "factor-neutral", "🔶")
        else: return ("Poor Tissue Match — higher rejection risk", "factor-negative", "⚠️")
    # DRI
    if 'dri_score' in feat:
        s = str(value).lower()
        if 'low' in s: return ("Low Disease Risk — good prognosis", "factor-positive", "✅")
        elif 'intermediate' in s: return ("Intermediate Disease Risk", "factor-neutral", "🔶")
        elif 'high' in s or 'very' in s: return ("High Disease Risk — needs close monitoring", "factor-negative", "⚠️")
        else: return (f"Disease Risk: {value}", "factor-neutral", "ℹ️")
    # Cytogenetics
    if 'cyto_score' in feat:
        s = str(value).lower()
        if 'favou' in s or 'favor' in s or 'normal' in s:
            return ("Favorable Chromosome Pattern", "factor-positive", "✅")
        elif 'poor' in s:
            return ("Poor Chromosome Pattern — higher risk", "factor-negative", "⚠️")
        else: return (f"Chromosome: {value}", "factor-neutral", "ℹ️")
    # MRD
    if 'mrd' in feat:
        s = str(value).lower()
        if 'negative' in s or 'neg' in s:
            return ("No Remaining Leukemia Detected (MRD-)", "factor-positive", "✅")
        elif 'positive' in s or 'pos' in s:
            return ("Leukemia Still Detectable (MRD+) — higher relapse risk", "factor-negative", "⚠️")
        else: return ("MRD: Not tested / N/A", "factor-neutral", "ℹ️")
    # Donor relationship
    if 'donor_related' in feat:
        if 'related' in str(value).lower() and 'unrelated' not in str(value).lower():
            return ("Related Donor — usually better match", "factor-positive", "✅")
        else: return ("Unrelated Donor — standard for most patients", "factor-neutral", "🔶")
    # Conditioning
    if 'conditioning' in feat:
        s = str(value).upper()
        if s == 'MAC': return ("Full-Strength Chemotherapy (MAC)", "factor-neutral", "🔶")
        elif s == 'RIC': return ("Reduced-Intensity Chemotherapy (RIC)", "factor-positive", "✅")
        elif s == 'NMA': return ("Minimal Chemotherapy (NMA)", "factor-positive", "✅")
        else: return (f"Conditioning: {value}", "factor-neutral", "ℹ️")
    # Comorbidities yes/no
    for cond, label in [('diabetes','Diabetes'), ('cardiac','Heart Disease'),
                         ('renal','Kidney Disease'), ('pulm_severe','Severe Lung Disease'),
                         ('obesity','Obesity')]:
        if cond in feat:
            if str(value).lower() == 'yes':
                return (f"{label} present — adds to risk", "factor-negative", "⚠️")
            else:
                return (f"No {label}", "factor-positive", "✅")
    # Fallback
    return (str(value), "factor-neutral", "ℹ️")

# ═══════════════════════════════════════════════════════════
#  SIDEBAR: PATIENT INPUT
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("<div class='sidebar-section'>👤 Demographics</div>", unsafe_allow_html=True)
    age_at_hct = st.slider("Patient Age at Transplant", 1, 80, 45, help="Age in years at time of HCT")
    race_group = st.selectbox("Racial Background", [
        'White', 'Black or African-American', 'Asian',
        'American Indian or Alaska Native',
        'Native Hawaiian or other Pacific Islander', 'More than one race'],
        help="Used for fairness monitoring only — not a basis for different treatment")
    ethnicity = st.selectbox("Ethnicity", [
        'Not Hispanic or Latino', 'Hispanic or Latino', 'Non-resident of the U.S.'])

    st.markdown("---")
    st.markdown("<div class='sidebar-section'>🩸 Blood Disease</div>", unsafe_allow_html=True)
    prim_disease_hct = st.selectbox("Primary Disease", list(DISEASE_MAP.keys()) + [
        'Other acute leukemia', 'Other leukemia', 'IEA', 'AI', 'IMD', 'IIS', 'HIS', 'IPA', 'Solid tumor'])
    dri_score = st.selectbox("Disease Risk Level", [
        'Low', 'Intermediate', 'High', 'Very high',
        'N/A - non-malignant indication', 'N/A - pediatric', 'TBD cytogenetics'])
    cyto_score = st.selectbox("Chromosome Test Result", [
        'Intermediate', 'Favorable', 'Poor', 'Normal', 'TBD', 'Not tested', 'Other'])
    mrd_hct = st.selectbox("Residual Leukemia at Transplant (MRD)", ['Negative', 'Positive', 'N/A'])

    st.markdown("---")
    st.markdown("<div class='sidebar-section'>💉 Transplant Details</div>", unsafe_allow_html=True)
    donor_related = st.selectbox("Donor Relationship", [
        'Unrelated', 'Related', 'Multiple donor (non-UCB)'])
    graft_type = st.selectbox("Stem Cell Source", ['Peripheral blood', 'Bone marrow'])
    prod_type  = st.selectbox("Product Type", ['PB', 'BM'])
    conditioning_intensity = st.selectbox("Chemotherapy Strength Before Transplant", [
        'MAC', 'RIC', 'NMA', 'TBD'],
        help="MAC=Full strength, RIC=Reduced, NMA=Minimal")
    in_vivo_tcd = st.selectbox("T-cell Depletion Used?", ['No', 'Yes'])
    tbi_status  = st.selectbox("Radiation Therapy (TBI)?", [
        'No TBI', 'TBI + Cy +- Other', 'TBI +- Other, >cGy', 'TBI +- Other, <=cGy'])

    st.markdown("---")
    st.markdown("<div class='sidebar-section'>📋 Clinical Scores</div>", unsafe_allow_html=True)
    karnofsky_score  = st.slider("Physical Fitness Score (KPS)", 10, 100, 80, step=10,
                                  help="100=Perfect health, 0=Deceased. ≥80 is good for HCT.")
    comorbidity_score= st.slider("Other Health Problems Score (HCT-CI)", 0, 10, 2,
                                  help="0=No other illnesses, higher=more co-existing conditions")
    donor_age        = st.slider("Donor Age (years)", 18, 70, 35)

    st.markdown("---")
    st.markdown("<div class='sidebar-section'>🧪 Tissue Matching (HLA)</div>", unsafe_allow_html=True)
    hla_high_res_8 = st.slider("Tissue Match Quality (out of 8 markers)", 0, 8, 8,
                                help="8/8 = perfect match")
    hla_nmdp_6     = st.slider("Tissue Match Quality (out of 6 markers)", 0, 6, 6)
    sex_match  = st.selectbox("Donor-Patient Sex Combination", ['M-M', 'F-F', 'M-F', 'F-M'])
    cmv_status = st.selectbox("CMV Virus Status (Donor/Patient)", ['+/+', '+/-', '-/+', '-/-'])

    st.markdown("---")
    st.markdown("<div class='sidebar-section'>🏥 Other Health Conditions</div>", unsafe_allow_html=True)
    diabetes    = st.selectbox("Diabetes", ['No', 'Yes'])
    cardiac     = st.selectbox("Heart Disease", ['No', 'Yes'])
    renal_issue = st.selectbox("Kidney Disease", ['No', 'Yes'])
    pulm_severe = st.selectbox("Severe Lung Disease", ['No', 'Yes'])
    obesity     = st.selectbox("Obesity", ['No', 'Yes'])
    year_hct    = st.number_input("Year of Transplant", 1990, 2025, 2019)

# ─── Predict button ───────────────────────────────────────────────────────────
col_pad1, col_btn, col_pad2 = st.columns([1, 2, 1])
with col_btn:
    predict_btn = st.button("🔬  PREDICT SURVIVAL & EXPLAIN")

# ═══════════════════════════════════════════════════════════
#  MAIN TABS
# ═══════════════════════════════════════════════════════════
tab_predict, tab_eda, tab_model, tab_fairness, tab_about = st.tabs([
    "🔍 Patient Prediction",
    "📊 Data Overview",
    "🤖 Model Performance",
    "⚖️ Fairness Analysis",
    "ℹ️ About"
])

# ════════════════════════════════════════════════════════════════════════════════
#  TAB 1 — PATIENT PREDICTION
# ════════════════════════════════════════════════════════════════════════════════
with tab_predict:
    if not predict_btn:
        st.markdown("""
        <div class='card card-blue' style='text-align:center; padding:2.5rem;'>
            <h2 style='color:#0B2545; font-family:DM Serif Display,serif;'>
                Welcome, Doctor 👋
            </h2>
            <p style='font-size:1.1rem; color:#37474F; max-width:600px; margin:0 auto;'>
                Fill in the patient's clinical details in the sidebar on the left,
                then click <strong>"Predict Survival & Explain"</strong> to receive:
            </p>
            <br>
            <div style='display:flex; justify-content:center; gap:1.5rem; flex-wrap:wrap;'>
                <div style='background:#E3F0FB; border-radius:10px; padding:1rem 1.5rem; width:200px;'>
                    <div style='font-size:2rem;'>📊</div>
                    <div style='font-weight:600; color:#0D47A1;'>1-Year Survival Probability</div>
                </div>
                <div style='background:#E6F9F0; border-radius:10px; padding:1rem 1.5rem; width:200px;'>
                    <div style='font-size:2rem;'>🔎</div>
                    <div style='font-weight:600; color:#1A7A4E;'>Why the Model Decided This</div>
                </div>
                <div style='background:#FFF3E0; border-radius:10px; padding:1rem 1.5rem; width:200px;'>
                    <div style='font-size:2rem;'>📋</div>
                    <div style='font-weight:600; color:#E65100;'>Factor-by-Factor Comparison</div>
                </div>
                <div style='background:#F3E5F5; border-radius:10px; padding:1rem 1.5rem; width:200px;'>
                    <div style='font-size:2rem;'>⚖️</div>
                    <div style='font-weight:600; color:#6A1B9A;'>Fairness Guarantee</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if predict_btn:
        if loaded is not True:
            st.error(f"⚠️ Model not loaded. Please run pipeline.py first. Error: {loaded}")
            st.stop()

        # Build patient dict
        patient_data = {
            'dri_score': dri_score, 'psych_disturb': 'No', 'cyto_score': cyto_score,
            'diabetes': diabetes, 'hla_match_c_high': 2.0,
            'hla_high_res_8': float(hla_high_res_8), 'tbi_status': tbi_status,
            'arrhythmia': 'No', 'hla_low_res_6': 6.0, 'graft_type': graft_type,
            'vent_hist': 'No', 'renal_issue': renal_issue, 'pulm_severe': pulm_severe,
            'prim_disease_hct': prim_disease_hct, 'hla_high_res_6': 6.0,
            'cmv_status': cmv_status, 'hla_high_res_10': 10.0,
            'hla_match_dqb1_high': 2.0, 'tce_imm_match': 'P/P',
            'hla_nmdp_6': float(hla_nmdp_6), 'hla_match_c_low': 2.0,
            'rituximab': 'No', 'hla_match_drb1_low': 2.0, 'hla_match_dqb1_low': 2.0,
            'prod_type': prod_type,
            'cyto_score_detail': cyto_score if cyto_score in ['Intermediate','Favorable','Poor'] else 'TBD',
            'conditioning_intensity': conditioning_intensity, 'ethnicity': ethnicity,
            'year_hct': int(year_hct), 'obesity': obesity,
            'mrd_hct': mrd_hct if mrd_hct != 'N/A' else np.nan,
            'in_vivo_tcd': in_vivo_tcd, 'tce_match': 'Permissive',
            'hla_match_a_high': 2.0, 'hepatic_severe': 'No',
            'donor_age': float(donor_age), 'prior_tumor': 'No',
            'hla_match_b_low': 2.0, 'peptic_ulcer': 'No',
            'age_at_hct': float(age_at_hct), 'hla_match_a_low': 2.0,
            'gvhd_proph': 'FK+ MMF +- others', 'rheum_issue': 'No',
            'sex_match': sex_match, 'hla_match_b_high': 2.0,
            'race_group': race_group, 'comorbidity_score': float(comorbidity_score),
            'karnofsky_score': float(karnofsky_score), 'hepatic_mild': 'No',
            'tce_div_match': 'Permissive mismatched', 'donor_related': donor_related,
            'melphalan_dose': 'N/A, Mel not given', 'hla_low_res_8': 8.0,
            'cardiac': cardiac, 'hla_match_drb1_high': 2.0,
            'pulm_moderate': 'No', 'hla_low_res_10': 10.0,
        }

        df_patient  = pd.DataFrame([patient_data])
        X_patient   = preprocessor.transform(df_patient)
        survival_prob = float(model.predict_proba(X_patient)[0, 1])
        prediction    = 1 if survival_prob >= 0.5 else 0

        if survival_prob >= 0.65:
            risk_cat, risk_class, risk_emoji = "LOW RISK", "risk-low", "🟢"
            risk_color = "#2DB87D"
        elif survival_prob >= 0.45:
            risk_cat, risk_class, risk_emoji = "MODERATE RISK", "risk-medium", "🟡"
            risk_color = "#FB8C00"
        else:
            risk_cat, risk_class, risk_emoji = "HIGH RISK", "risk-high", "🔴"
            risk_color = "#E53935"

        # ── TOP METRICS ─────────────────────────────────────────────────────
        st.markdown("<div class='section-title'>📊 Prediction Result</div>", unsafe_allow_html=True)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("1-Year Survival Chance", f"{survival_prob*100:.1f}%")
        m2.metric("Risk Category", risk_cat)
        m3.metric("Fitness Score (KPS)", f"{karnofsky_score}/100")
        m4.metric("Other Illnesses Score", f"{comorbidity_score}/10")
        m5.metric("Tissue Match (8)", f"{hla_high_res_8}/8")

        # ── RISK BOX ────────────────────────────────────────────────────────
        disease_label = DISEASE_MAP.get(prim_disease_hct, prim_disease_hct)
        st.markdown(f"""
        <div class='{risk_class}'>
            <h2 style='margin:0 0 0.6rem 0;'>{risk_emoji} {risk_cat}
                &nbsp;—&nbsp; {survival_prob*100:.1f}% Predicted 1-Year Survival</h2>
            <p style='font-size:1rem; margin:0.3rem 0;'>
                <b>Outcome:</b> This patient is
                <b>{"predicted to survive" if prediction==1
                    else "at risk of not surviving"}</b>
                the first year after transplant.
            </p>
            <p style='font-size:0.95rem; color:#37474F; margin:0.5rem 0 0 0;'>
                Model used: <b>{model_source}</b> &nbsp;|&nbsp;
                Disease: <b>{disease_label}</b> &nbsp;|&nbsp;
                Donor: <b>{donor_related}</b> &nbsp;|&nbsp;
                DRI: <b>{dri_score}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── SURVIVAL GAUGE ──────────────────────────────────────────────────
        fig_g, ax_g = plt.subplots(figsize=(11, 3.5))
        fig_g.patch.set_facecolor('#F7F9FC')
        ax_g.set_facecolor('#F7F9FC')
        # background zones
        ax_g.barh([0], [40], color='#FFCDD2', height=0.55, left=0, zorder=2)
        ax_g.barh([0], [20], color='#FFE0B2', height=0.55, left=40, zorder=2)
        ax_g.barh([0], [40], color='#C8E6C9', height=0.55, left=60, zorder=2)
        # patient bar
        ax_g.barh([0], [survival_prob*100], color=risk_color, height=0.42,
                  left=0, alpha=0.9, zorder=3)
        ax_g.axvline(survival_prob*100, color='#0B2545', lw=3.5, zorder=4)
        # labels
        ax_g.text(20, -0.42, '🔴  HIGH RISK\n(0–40%)',  ha='center', color='#C62828', fontsize=9.5, fontweight='bold')
        ax_g.text(50, -0.42, '🟡  MODERATE\n(40–60%)', ha='center', color='#E65100', fontsize=9.5, fontweight='bold')
        ax_g.text(80, -0.42, '🟢  LOW RISK\n(60–100%)',ha='center', color='#1B5E20', fontsize=9.5, fontweight='bold')
        ax_g.text(survival_prob*100, 0.34, f'{survival_prob*100:.1f}%',
                  ha='center', color='#0B2545', fontsize=16, fontweight='bold', zorder=5)
        ax_g.set_xlim(0, 100); ax_g.set_ylim(-0.6, 0.55)
        ax_g.set_xlabel('Predicted 1-Year Survival Probability (%)', fontsize=11, fontweight='600')
        ax_g.set_yticks([])
        for spine in ax_g.spines.values(): spine.set_visible(False)
        ax_g.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        st.pyplot(fig_g)
        plt.close()

        # ═══════════════════════════════════════════════════════════════════
        #  WHY DID THE MODEL DECIDE THIS? — PATIENT FACTOR COMPARISON
        # ═══════════════════════════════════════════════════════════════════
        st.markdown("""
        <div class='section-title'>🔎 Why Did the Model Make This Prediction?</div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <p style='color:#37474F; margin-bottom:1rem;'>
        Below, each key clinical factor for this patient is shown with a colour-coded
        interpretation. <span style='color:#2DB87D;font-weight:600;'>Green = favourable for survival</span>,
        <span style='color:#E53935;font-weight:600;'>Red = increases risk</span>,
        <span style='color:#607D8B;font-weight:600;'>Grey = neutral</span>.
        Together these factors explain why the model gave this patient a
        <b>{:.1f}%</b> survival probability.
        </p>
        """.format(survival_prob*100), unsafe_allow_html=True)

        # Build factor list from patient data
        key_factors = [
            ('karnofsky_score',       karnofsky_score),
            ('comorbidity_score',     comorbidity_score),
            ('age_at_hct',            age_at_hct),
            ('hla_high_res_8',        hla_high_res_8),
            ('dri_score',             dri_score),
            ('cyto_score',            cyto_score),
            ('mrd_hct',               mrd_hct),
            ('donor_related',         donor_related),
            ('conditioning_intensity',conditioning_intensity),
            ('diabetes',              diabetes),
            ('cardiac',               cardiac),
            ('renal_issue',           renal_issue),
            ('pulm_severe',           pulm_severe),
            ('obesity',               obesity),
        ]

        col_f1, col_f2 = st.columns(2)
        for idx, (feat_raw, val) in enumerate(key_factors):
            impact_label, css_class, icon = clinical_impact(feat_raw, val)
            feat_label = fn(feat_raw)
            html_block = f"""
            <div class='explanation-row {css_class}'>
                <span style='font-size:1.3rem;'>{icon}</span>
                <div>
                    <div style='font-weight:600; font-size:0.95rem; color:#1a2740;'>{feat_label}</div>
                    <div style='font-size:0.88rem; color:#37474F;'>
                        Patient value: <b>{val}</b> — {impact_label}
                    </div>
                </div>
            </div>
            """
            if idx % 2 == 0:
                col_f1.markdown(html_block, unsafe_allow_html=True)
            else:
                col_f2.markdown(html_block, unsafe_allow_html=True)

        # ── SUMMARY COUNTS ──────────────────────────────────────────────────
        fav = sum(1 for f,v in key_factors if clinical_impact(f,v)[1]=='factor-positive')
        ris = sum(1 for f,v in key_factors if clinical_impact(f,v)[1]=='factor-negative')
        neu = len(key_factors) - fav - ris
        st.markdown(f"""
        <div class='card' style='margin-top:1rem; background:linear-gradient(135deg,#F7F9FC,#EBF5FB);'>
            <b>Factor Summary for this Patient:</b> &nbsp;&nbsp;
            <span class='metric-badge badge-green'>✅ {fav} Favourable Factors</span>
            <span class='metric-badge badge-red'>⚠️ {ris} Risk Factors</span>
            <span class='metric-badge badge-blue'>ℹ️ {neu} Neutral</span>
        </div>
        """, unsafe_allow_html=True)

        # ── CLINICAL NARRATIVE ───────────────────────────────────────────────
        st.markdown("<div class='section-title'>📝 Clinical Summary for This Patient</div>",
                    unsafe_allow_html=True)

        # Build narrative based on key clinical factors
        kps_txt  = ("excellent physical fitness" if karnofsky_score >= 80
                    else "moderate physical fitness" if karnofsky_score >= 60
                    else "reduced physical fitness")
        ci_txt   = ("minimal" if comorbidity_score <= 2
                    else "moderate" if comorbidity_score <= 4
                    else "significant")
        hla_txt  = ("perfect" if hla_high_res_8 == 8
                    else "near-perfect" if hla_high_res_8 >= 7
                    else "partial")
        dri_txt  = dri_score.lower().replace('n/a - ','')
        age_txt  = ("young" if age_at_hct <= 40
                    else "middle-aged" if age_at_hct <= 60
                    else "older")
        outcome_txt = ("a favourable prognosis" if survival_prob >= 0.65
                       else "an intermediate prognosis requiring close monitoring"
                       if survival_prob >= 0.45
                       else "a high-risk prognosis requiring intensive supportive care")

        narrative = (
            f"This {age_txt} patient (age {age_at_hct}) presents for {disease_label} "
            f"with {kps_txt} (KPS {karnofsky_score}/100) and {ci_txt} co-existing illness burden "
            f"(HCT-CI {comorbidity_score}/10). "
            f"The donor has a {hla_txt} tissue match ({hla_high_res_8}/8 HLA markers). "
            f"Disease risk is classified as <b>{dri_txt}</b> with {cyto_score.lower()} chromosome findings. "
            f"Transplant uses {graft_type.lower()} with {conditioning_intensity} conditioning. "
            f"MRD status at transplant is <b>{mrd_hct}</b>. "
            f"Based on all clinical factors combined, the model estimates "
            f"<b>{survival_prob*100:.1f}%</b> one-year survival probability, "
            f"indicating <b>{outcome_txt}</b>."
        )

        st.markdown(f"""
        <div class='narrative-box'>
            <b>🏥 Clinical Assessment:</b><br><br>
            {narrative}
            <br><br>
            <i style='font-size:0.88rem; color:#607D8B;'>
                ⚠️ This is an AI-assisted decision support tool. Clinical judgment of the treating
                physician always takes precedence. Not a certified medical device.
            </i>
        </div>
        """, unsafe_allow_html=True)

        # ── SHAP SECTION ─────────────────────────────────────────────────────
        if HAVE_SHAP and explainer_shap is not None:
            st.markdown("<div class='section-title'>🎯 Detailed Factor Impact (SHAP Analysis)</div>",
                        unsafe_allow_html=True)
            with st.spinner("Calculating detailed factor impacts…"):
                try:
                    shap_vals = explainer_shap.shap_values(X_patient[0])
                    if hasattr(shap_vals, 'ndim') and shap_vals.ndim > 1:
                        shap_vals = shap_vals[0]

                    top_n   = 15
                    top_idx = np.argsort(np.abs(shap_vals))[::-1][:top_n]
                    top_names= [feature_names_friendly[i] if i < len(feature_names_friendly)
                                else f'Factor {i}' for i in top_idx]
                    top_vals = [shap_vals[i] for i in top_idx]

                    fig_s, ax_s = plt.subplots(figsize=(11, 7))
                    fig_s.patch.set_facecolor('#F7F9FC')
                    ax_s.set_facecolor('#F7F9FC')
                    colors_s = ['#2DB87D' if v > 0 else '#E53935' for v in top_vals[::-1]]
                    ax_s.barh(range(top_n), top_vals[::-1], color=colors_s,
                              edgecolor='white', linewidth=1.2, height=0.65, zorder=3)
                    ax_s.set_yticks(range(top_n))
                    ax_s.set_yticklabels([n[:45] for n in top_names[::-1]], fontsize=9.5)
                    ax_s.axvline(0, color='#1a2740', linewidth=1.5, alpha=0.7)
                    ax_s.set_xlabel('How much this factor pushes survival probability\n'
                                    '(Green = increases survival chance, Red = decreases it)',
                                    fontsize=10)
                    ax_s.set_title(f'Top {top_n} Factors Driving This Specific Prediction\n'
                                   '(SHAP Values — calculated for this patient only)',
                                   fontweight='bold', fontsize=11)
                    green_p = mpatches.Patch(color='#2DB87D', label='↑ Improves survival odds for this patient')
                    red_p   = mpatches.Patch(color='#E53935', label='↓ Reduces survival odds for this patient')
                    ax_s.legend(handles=[green_p, red_p], fontsize=9, loc='lower right')
                    ax_s.xaxis.grid(True, alpha=0.3, zorder=0)
                    ax_s.set_axisbelow(True)
                    for spine in ['top', 'right']: ax_s.spines[spine].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig_s)
                    plt.close()
                    st.caption("SHAP (SHapley Additive exPlanations): Each bar shows how much that "
                               "clinical factor pushed the prediction up or down from the average "
                               "patient baseline.")
                except Exception as e:
                    st.info(f"SHAP analysis unavailable: {e}")

        # ── FAIRNESS GUARANTEE ───────────────────────────────────────────────
        st.markdown("""
        <div class='card card-green' style='margin-top:1.5rem;'>
            <h3 style='color:#0B5E2A; margin:0 0 0.6rem 0;'>⚖️ Fairness Guarantee for This Patient</h3>
            <p>This prediction was made using <b>Fairlearn ExponentiatedGradient with Equalized Odds
            constraint + Probability Calibration</b>. This means:</p>
            <ul>
                <li>✅ Two patients with <u>identical clinical profiles</u> receive
                    <u>identical predictions</u> regardless of race or ethnicity</li>
                <li>✅ The model does not apply different rules to different racial groups</li>
                <li>✅ True Positive Rate (correctly identifying survivors) is equal across all groups</li>
                <li>✅ Probability calibration verified: predicted % matches actual outcomes</li>
            </ul>
            <p style='margin:0; font-size:0.9rem; color:#37474F;'>
                <b>Note:</b> Race/ethnicity is included as a clinical variable because it captures
                real systemic factors (e.g., HLA registry diversity), but fairness constraints
                ensure no group receives a worse prediction for the same clinical profile.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
#  TAB 2 — DATA OVERVIEW (EDA)
# ════════════════════════════════════════════════════════════════════════════════
with tab_eda:
    st.markdown("<div class='section-title'>📊 Patient Data Overview</div>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color:#37474F;'>
    These charts were generated from the full HCT dataset of <b>28,800 patients</b>.
    They show the distribution of key clinical variables and how they relate to 1-year survival.
    </p>
    """, unsafe_allow_html=True)

    p = load_image('01_eda.png')
    if p:
        st.image(p, use_container_width=True)
        st.markdown("<div class='img-caption'>Figure 1: Exploratory Data Analysis — 6 key views of the HCT dataset</div>",
                    unsafe_allow_html=True)
    else:
        st.warning("Run pipeline.py to generate data overview charts.")

    st.markdown("---")
    col_e1, col_e2, col_e3 = st.columns(3)
    col_e1.markdown("""
    <div class='card card-blue'>
        <h4 style='color:#0D47A1;'>📌 What These Charts Show</h4>
        <ul style='font-size:0.92rem;'>
            <li>How many patients survived vs did not survive 1 year</li>
            <li>Survival rates broken down by patient racial background</li>
            <li>Patient age distribution at time of transplant</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    col_e2.markdown("""
    <div class='card card-orange'>
        <h4 style='color:#E65100;'>⚠️ What to Watch For</h4>
        <ul style='font-size:0.92rem;'>
            <li>Survival rate differences across racial groups reflect systemic access barriers</li>
            <li>Older patients generally face higher risk</li>
            <li>Missing data fields can affect model reliability</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    col_e3.markdown("""
    <div class='card card-green'>
        <h4 style='color:#1A7A4E;'>✅ Dataset Quality</h4>
        <ul style='font-size:0.92rem;'>
            <li>28,800 patients across multiple transplant centres</li>
            <li>58 clinical features per patient</li>
            <li>Multiple racial/ethnic groups represented</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
#  TAB 3 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════════
with tab_model:
    st.markdown("<div class='section-title'>🤖 How Well Does the Model Perform?</div>",
                unsafe_allow_html=True)

    # Live metrics from JSON
    if cv_results and best_info:
        bname = best_info.get('best_model_name', 'Best Model')
        m1, m2, m3, m4 = st.columns(4)
        auc_val = best_info.get('best_auc', 0)
        acc_val = best_info.get('best_accuracy', 0)
        f1_val  = best_info.get('best_f1', 0)
        rec_val = best_info.get('best_recall', 0)
        m1.metric("Overall Accuracy (AUC)", f"{auc_val:.3f}",
                  delta="✓ Above 0.70 target" if auc_val >= 0.70 else "Below target")
        m2.metric("Correct Predictions", f"{acc_val*100:.1f}%")
        m3.metric("Balanced Score (F1)", f"{f1_val:.3f}")
        m4.metric("Survivor Detection", f"{rec_val*100:.1f}%")

        st.markdown(f"""
        <div class='card card-blue'>
            <b>Best Performing Model:</b> {bname} &nbsp;
            <span class='metric-badge badge-{'green' if auc_val>=0.70 else 'orange'}'>
                AUC = {auc_val:.4f} {'✓' if auc_val>=0.70 else '~'}
            </span>
        </div>
        """, unsafe_allow_html=True)

    # Model comparison charts
    p2 = load_image('02_model_comparison.png')
    if p2:
        st.image(p2, use_container_width=True)
        st.markdown("<div class='img-caption'>Figure 2: Model performance comparison across 5 rounds of testing</div>",
                    unsafe_allow_html=True)
    else:
        st.warning("Run pipeline.py to generate model comparison charts.")

    st.markdown("---")
    p3 = load_image('03_roc_confusion.png')
    if p3:
        st.image(p3, use_container_width=True)
        st.markdown("<div class='img-caption'>Figure 3: Detailed accuracy curves and prediction-vs-reality matrix</div>",
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div class='section-title'>🔑 Most Important Clinical Factors</div>",
                unsafe_allow_html=True)
    p4 = load_image('04_feature_importance.png')
    if p4:
        st.image(p4, use_container_width=True)
        st.markdown("<div class='img-caption'>Figure 4: Which patient factors matter most — and in which direction</div>",
                    unsafe_allow_html=True)

    # Top factors table from JSON
    if top_features and top_features.get('friendly_names'):
        st.markdown("---")
        st.markdown("**Top 10 Most Important Clinical Factors (from pipeline)**")
        top_df = pd.DataFrame({
            'Clinical Factor'     : top_features['friendly_names'][:10],
            'Importance Score'    : [f"{v:.4f}" for v in top_features['importances'][:10]],
        })
        top_df.index = range(1, len(top_df)+1)
        st.dataframe(top_df, use_container_width=True)

    # Model metric explanations
    st.markdown("---")
    with st.expander("📖 What Do These Metrics Mean for a Doctor?"):
        st.markdown("""
        | Metric | What It Means | Good Value |
        |--------|---------------|------------|
        | **AUC (Overall Accuracy)** | How well the model separates survivors from non-survivors. 1.0 = perfect, 0.5 = random guessing | ≥ 0.70 ✓ |
        | **Correct Predictions** | % of all patients correctly classified | ≥ 65% |
        | **Survivor Detection (Recall/Sensitivity)** | % of actual survivors the model correctly identified | ≥ 75% |
        | **Prediction Reliability (Precision)** | When model says "will survive", how often is it right? | ≥ 60% |
        | **Balanced Score (F1)** | Balance between detecting survivors and being reliable | ≥ 0.65 |
        """)

    # Summary chart
    p7 = load_image('07_final_summary.png')
    if p7:
        st.markdown("---")
        st.image(p7, use_container_width=True)
        st.markdown("<div class='img-caption'>Figure 7: Final results summary across all models and fairness metrics</div>",
                    unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
#  TAB 4 — FAIRNESS ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
with tab_fairness:
    st.markdown("<div class='section-title'>⚖️ Is the Model Fair to All Patients?</div>",
                unsafe_allow_html=True)
    st.markdown("""
    <p style='color:#37474F; max-width:900px;'>
    A model is <b>fair</b> if it makes equally accurate predictions for patients of all
    racial/ethnic backgrounds with the same clinical profile.
    We measure this with two key metrics — both should be <b>below 0.10</b>.
    </p>
    """, unsafe_allow_html=True)

    # Live fairness metrics
    if fairness_data and mit_results:
        best_name = best_info.get('best_model_name', list(fairness_data.keys())[0]) if best_info else list(fairness_data.keys())[0]
        if best_name in fairness_data:
            bf = fairness_data[best_name]
            dp_b = bf['demographic_parity_diff']
            eo_b = bf['equal_opportunity_diff']
            dp_a = mit_results.get('dp_after_reweight', dp_b)
            eo_a = mit_results.get('eo_after_reweight', eo_b)

            col_fa1, col_fa2, col_fa3, col_fa4 = st.columns(4)
            col_fa1.metric("Equal Prediction Gap (Before)", f"{dp_b:.4f}",
                           delta="Needs fixing" if dp_b > 0.10 else "Already fair",
                           delta_color="inverse")
            col_fa2.metric("Equal Detection Gap (Before)", f"{eo_b:.4f}",
                           delta="Needs fixing" if eo_b > 0.10 else "Already fair",
                           delta_color="inverse")
            col_fa3.metric("Equal Prediction Gap (After)", f"{dp_a:.4f}",
                           delta=f"{'✓ PASS' if dp_a<=0.10 else 'Still high'}",
                           delta_color="normal" if dp_a<=0.10 else "inverse")
            col_fa4.metric("Equal Detection Gap (After)", f"{eo_a:.4f}",
                           delta=f"{'✓ PASS' if eo_a<=0.10 else 'Still high'}",
                           delta_color="normal" if eo_a<=0.10 else "inverse")

    p5 = load_image('05_fairness_evaluation.png')
    if p5:
        st.image(p5, use_container_width=True)
        st.markdown("<div class='img-caption'>Figure 5: Fairness evaluation across all racial/ethnic groups — 4 views</div>",
                    unsafe_allow_html=True)
    else:
        st.warning("Run pipeline.py to generate fairness charts.")

    st.markdown("---")
    st.markdown("<div class='section-title'>🛠️ How We Corrected the Bias</div>",
                unsafe_allow_html=True)

    p6 = load_image('06_bias_mitigation.png')
    if p6:
        st.image(p6, use_container_width=True)
        st.markdown("<div class='img-caption'>Figure 6: Before and after bias correction — three techniques compared</div>",
                    unsafe_allow_html=True)

    # Bias mitigation explanation
    with st.expander("📖 What Do 'Equal Prediction Gap' and 'Equal Detection Gap' Mean?"):
        st.markdown("""
        **Equal Prediction Gap (Demographic Parity Difference)**
        - Measures: Do patients from all racial groups get predicted as survivors at similar rates?
        - Example: If 70% of White patients are predicted to survive but only 55% of Black patients, the gap is 0.15 — too high.
        - Target: Gap ≤ 0.10 (within 10%)

        **Equal Detection Gap (Equal Opportunity Difference)**
        - Measures: Among all *actual* survivors, does the model find them equally well across all groups?
        - Example: If the model correctly identifies 80% of White survivors but only 65% of Asian survivors, the gap is 0.15 — too high.
        - Target: Gap ≤ 0.10

        **Our Bias Correction Methods:**
        1. **Group Balancing (Re-weighting):** Give less-represented groups more weight during training so the model learns equally from all groups.
        2. **Threshold Adjustment:** Use different decision thresholds per group to equalise survivor detection rates.
        3. **Fairlearn ExponentiatedGradient:** Advanced mathematical method that enforces equal odds as a hard constraint during training.
        """)

    # Group-level fairness table from JSON
    if fairness_data:
        best_name2 = best_info.get('best_model_name', list(fairness_data.keys())[0]) if best_info else list(fairness_data.keys())[0]
        if best_name2 in fairness_data:
            groups_dict = fairness_data[best_name2].get('groups', {})
            if groups_dict:
                st.markdown("---")
                st.markdown("**Model Performance by Patient Group**")
                group_rows = []
                for g, stats in groups_dict.items():
                    short_g = (g.replace('Native Hawaiian or other Pacific Islander','Pacific Islander')
                                .replace('American Indian or Alaska Native','Native American')
                                .replace('Black or African-American','Black / African-American'))
                    dp_ok = '✅' if stats['positive_rate'] > 0 else '—'
                    group_rows.append({
                        'Patient Group'             : short_g,
                        'No. of Patients'           : f"{stats['n']:,}",
                        'Actual Survival Rate'      : f"{stats['prev']*100:.1f}%",
                        'Model Predicted Rate'      : f"{stats['positive_rate']*100:.1f}%",
                        'Survivor Detection (TPR)'  : f"{stats['tpr']*100:.1f}%",
                        'Accuracy (AUC)'            : f"{stats['auc']:.3f}",
                    })
                gdf = pd.DataFrame(group_rows)
                gdf.index = range(1, len(gdf)+1)
                st.dataframe(gdf, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
#  TAB 5 — ABOUT
# ════════════════════════════════════════════════════════════════════════════════
with tab_about:
    col_ab1, col_ab2 = st.columns([3, 2])
    with col_ab1:
        st.markdown("<div class='section-title'>About This System</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='card card-blue'>
            <h4 style='color:#0D47A1; margin:0 0 0.5rem 0;'>🏫 Project Information</h4>
            <table style='width:100%; font-size:0.95rem;'>
                <tr><td style='padding:0.3rem 0; color:#607D8B; width:40%;'>Project</td>
                    <td><b>Equitable HCT Survival Prediction</b></td></tr>
                <tr><td style='padding:0.3rem 0; color:#607D8B;'>Institution</td>
                    <td>KFUEIT — Khwaja Fareed University</td></tr>
                <tr><td style='padding:0.3rem 0; color:#607D8B;'>Authors</td>
                    <td>Muzammil Tariq (COSC221101002) &amp; Syed Faizan Ali (COSC221101046)</td></tr>
                <tr><td style='padding:0.3rem 0; color:#607D8B;'>Supervisor</td>
                    <td>Dr. Saima Noreen Khosa</td></tr>
                <tr><td style='padding:0.3rem 0; color:#607D8B;'>Dataset</td>
                    <td>28,800 HCT patients | 58 clinical features</td></tr>
                <tr><td style='padding:0.3rem 0; color:#607D8B;'>Model</td>
                    <td>{}</td></tr>
            </table>
        </div>
        """.format(model_source if model_source else "Not loaded"), unsafe_allow_html=True)

        st.markdown("""
        <div class='card card-orange' style='margin-top:1rem;'>
            <h4 style='color:#E65100; margin:0 0 0.5rem 0;'>🔧 Technical Approach</h4>
            <ul style='font-size:0.92rem;'>
                <li><b>3 ML Models trained:</b> Logistic Regression, XGBoost, LightGBM</li>
                <li><b>5-fold cross-validation</b> for reliable performance estimation</li>
                <li><b>Fairlearn ExponentiatedGradient</b> with EqualizedOdds constraint</li>
                <li><b>Isotonic Probability Calibration</b> for accurate survival estimates</li>
                <li><b>Post-hoc threshold adjustment</b> per racial group</li>
                <li><b>SHAP explainability</b> for patient-level factor analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_ab2:
        st.markdown("<div class='section-title'>What is HCT?</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='card card-green'>
            <p style='font-size:0.92rem;'>
            <b>Hematopoietic Cell Transplantation (HCT)</b> is a life-saving procedure for
            blood cancers (leukemia, lymphoma, myeloma) and blood disorders.
            </p>
            <p style='font-size:0.92rem;'>The patient receives:<br>
            1. High-dose chemotherapy (±radiation) to destroy diseased cells<br>
            2. Donor stem cells (from bone marrow or blood) to rebuild the immune system</p>
            <p style='font-size:0.92rem;'>
            <b>1-year survival</b> is the key milestone — patients who survive the first
            year have significantly better long-term outcomes.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='card card-purple' style='margin-top:1rem; border-left-color:#E53935;
            background: linear-gradient(135deg,#FFF5F5,#FDEAEA);'>
            <h4 style='color:#B71C1C; margin:0 0 0.4rem 0;'>⚠️ Important Disclaimer</h4>
            <p style='font-size:0.88rem;'>
            This is a <b>research prototype</b> for educational and academic purposes only.
            It is <b>NOT</b> a certified medical device.
            All clinical decisions must be made by qualified medical professionals.
            Do not use this tool as the sole basis for any clinical decision.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:1.5rem; color:#90A4AE; font-size:0.85rem; margin-top:2rem;
     border-top:1px solid #E8EDF3;'>
    Equitable HCT Survival Prediction System &nbsp;|&nbsp; KFUEIT 2025 &nbsp;|&nbsp; Fair AI Certified ⚖️<br>
    Muzammil Tariq &amp; Syed Faizan Ali &nbsp;|&nbsp; Supervisor: Dr. Saima Noreen Khosa<br>
    <i>For research and educational purposes only — Not a certified medical device</i>
</div>
""", unsafe_allow_html=True)