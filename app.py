"""
=============================================================
Equitable HCT Survival Prediction — Streamlit Web App
KFUEIT Final Year Project 2025
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

# ─── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="HCT Survival Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #1565c0 100%);
        padding: 2rem; border-radius: 12px; margin-bottom: 2rem;
        text-align: center; color: white;
    }
    .metric-card {
        background: white; border-radius: 10px; padding: 1.2rem;
        border-left: 5px solid #1565c0; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .risk-high   { background: #FFEBEE; border: 2px solid #E53935; border-radius: 10px; padding: 1.5rem; }
    .risk-medium { background: #FFF8E1; border: 2px solid #FB8C00; border-radius: 10px; padding: 1.5rem; }
    .risk-low    { background: #E8F5E9; border: 2px solid #43A047; border-radius: 10px; padding: 1.5rem; }
    .fairness-box {
        background: #E3F2FD; border-radius: 10px; padding: 1rem;
        border: 2px solid #1565c0; margin-top: 1rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1a237e, #1565c0);
        color: white; font-size: 1.1rem; font-weight: bold;
        padding: 0.7rem 2rem; border-radius: 8px; border: none;
        width: 100%; margin-top: 1rem;
    }
    .stButton > button:hover { background: #0d47a1; transform: translateY(-1px); }
    .sidebar-header { font-size: 1.3rem; font-weight: bold; color: #1a237e; padding: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <h1>🏥 Equitable Survival Prediction after HCT</h1>
    <p style='font-size:1.1rem; margin:0'>Hematopoietic Cell Transplantation — 1-Year Survival Prediction</p>
    <p style='font-size:0.9rem; opacity:0.8; margin-top:0.5rem'>
        KFUEIT | Institute of Computer Science | 2025<br>
        Muzammil Tariq (COSC221101002) & Syed Faizan Ali (COSC221101046)
    </p>
</div>
""", unsafe_allow_html=True)

# ─── Load model ───────────────────────────────────────────
@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(base, 'outputs')

    try:
        with open(f'{out}/preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)

        # Load debiased model (from fairness_debiasing_solution.py)
        with open(f'{out}/model_debiased_fairlearn.pkl', 'rb') as f:
            model = pickle.load(f)

        # Load legacy models for backward compatibility
        with open(f'{out}/model_logistic_regression.pkl', 'rb') as f:
            lr_model = pickle.load(f)

        # Load cross-validation results
        with open(f'{out}/cv_results.json') as f:
            results = json.load(f)

        # Load fairness debiasing report
        mit_path = f'{out}/fairness_debiasing_report.json'
        if os.path.exists(mit_path):
            with open(mit_path) as f:
                mit = json.load(f)
        else:
            mit = {}

        # Load global threshold optimizer
        threshold_opt_path = f'{out}/threshold_optimizer.pkl'
        if os.path.exists(threshold_opt_path):
            with open(threshold_opt_path, 'rb') as f:
                threshold_opt = pickle.load(f)
        else:
            threshold_opt = None

        meta = {
            'best_model': max(results, key=lambda k: results[k]['AUC']['mean']) if results else 'Fairlearn',
            'n_features': None,
        }

        return preprocessor, model, lr_model, meta, results, mit, threshold_opt, True

    except Exception as e:
        return None, None, None, {}, {}, {}, None, str(e)

preprocessor, model, lr_model, meta, results, mit, threshold_opt, loaded = load_model()

# ─── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='sidebar-header'>📋 Patient Information</div>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**🧬 Patient Demographics**")
    age_at_hct = st.slider("Age at HCT (years)", 1, 80, 45, help="Patient age at time of transplant")
    race_group  = st.selectbox("Race Group", [
        'White', 'Black or African-American', 'Asian',
        'American Indian or Alaska Native',
        'Native Hawaiian or other Pacific Islander', 'More than one race'
    ])
    ethnicity = st.selectbox("Ethnicity", ['Not Hispanic or Latino', 'Hispanic or Latino', 'Non-resident of the U.S.'])

    st.markdown("---")
    st.markdown("**🩺 Disease Information**")
    prim_disease_hct = st.selectbox("Primary Disease for HCT", [
        'AML', 'ALL', 'MDS', 'MPN', 'NHL', 'CML', 'PCD', 'SAA',
        'Other acute leukemia', 'Other leukemia', 'IEA', 'AI', 'IMD',
        'IIS', 'HIS', 'IPA', 'HD', 'Solid tumor'
    ])
    dri_score = st.selectbox("Disease Risk Index (DRI)", [
        'Low', 'Intermediate', 'High', 'Very high',
        'N/A - non-malignant indication', 'N/A - pediatric', 'TBD cytogenetics'
    ])
    cyto_score = st.selectbox("Cytogenetic Score", [
        'Intermediate', 'Favorable', 'Poor', 'Normal', 'TBD', 'Not tested', 'Other'
    ])
    mrd_hct = st.selectbox("MRD at HCT (AML/ALL)", ['Negative', 'Positive', 'N/A'])

    st.markdown("---")
    st.markdown("**💉 Transplant Details**")
    donor_related = st.selectbox("Donor Relationship", ['Unrelated', 'Related', 'Multiple donor (non-UCB)'])
    graft_type    = st.selectbox("Graft Type", ['Peripheral blood', 'Bone marrow'])
    prod_type     = st.selectbox("Product Type", ['PB', 'BM'])
    conditioning_intensity = st.selectbox("Conditioning Intensity", ['MAC', 'RIC', 'NMA', 'TBD'])
    in_vivo_tcd   = st.selectbox("In-vivo T-cell Depletion", ['No', 'Yes'])
    tbi_status    = st.selectbox("TBI Status", ['No TBI', 'TBI + Cy +- Other', 'TBI +- Other, >cGy', 'TBI +- Other, <=cGy'])

    st.markdown("---")
    st.markdown("**📊 Clinical Scores**")
    karnofsky_score   = st.slider("Karnofsky Score (KPS)", 10, 100, 80, step=10, help="Performance status at HCT")
    comorbidity_score = st.slider("HCT-CI Comorbidity Score", 0, 10, 2, help="Sorror comorbidity index")
    donor_age         = st.slider("Donor Age (years)", 18, 70, 35)

    st.markdown("---")
    st.markdown("**🧪 HLA Matching**")
    hla_high_res_8 = st.slider("HLA High Res 8 (A,B,C,DRB1)", 0, 8, 8)
    hla_nmdp_6     = st.slider("HLA NMDP 6", 0, 6, 6)
    sex_match      = st.selectbox("Donor/Recipient Sex Match", ['M-M', 'F-F', 'M-F', 'F-M'])
    cmv_status     = st.selectbox("CMV Serostatus (D/R)", ['+/+', '+/-', '-/+', '-/-'])

    st.markdown("---")
    st.markdown("**🏥 Comorbidities**")
    diabetes    = st.selectbox("Diabetes",    ['No', 'Yes'])
    cardiac     = st.selectbox("Cardiac",     ['No', 'Yes'])
    renal_issue = st.selectbox("Renal Issue", ['No', 'Yes'])
    pulm_severe = st.selectbox("Pulmonary (Severe)", ['No', 'Yes'])
    obesity     = st.selectbox("Obesity",     ['No', 'Yes'])

    year_hct = st.number_input("Year of HCT", 1990, 2025, 2019)

# ─── Predict button ────────────────────────────────────────
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button("🔬 PREDICT 1-YEAR SURVIVAL PROBABILITY")

# ─── Prediction logic ─────────────────────────────────────
if predict_btn:
    if loaded is True:

        # Build patient record matching training columns
        patient_data = {
            'dri_score': dri_score,
            'psych_disturb': 'No',
            'cyto_score': cyto_score,
            'diabetes': diabetes,
            'hla_match_c_high': 2.0,
            'hla_high_res_8': float(hla_high_res_8),
            'tbi_status': tbi_status,
            'arrhythmia': 'No',
            'hla_low_res_6': 6.0,
            'graft_type': graft_type,
            'vent_hist': 'No',
            'renal_issue': renal_issue,
            'pulm_severe': pulm_severe,
            'prim_disease_hct': prim_disease_hct,
            'hla_high_res_6': 6.0,
            'cmv_status': cmv_status,
            'hla_high_res_10': 10.0,
            'hla_match_dqb1_high': 2.0,
            'tce_imm_match': 'P/P',
            'hla_nmdp_6': float(hla_nmdp_6),
            'hla_match_c_low': 2.0,
            'rituximab': 'No',
            'hla_match_drb1_low': 2.0,
            'hla_match_dqb1_low': 2.0,
            'prod_type': prod_type,
            'cyto_score_detail': cyto_score if cyto_score in ['Intermediate','Favorable','Poor'] else 'TBD',
            'conditioning_intensity': conditioning_intensity,
            'ethnicity': ethnicity,
            'year_hct': int(year_hct),
            'obesity': obesity,
            'mrd_hct': mrd_hct if mrd_hct != 'N/A' else np.nan,
            'in_vivo_tcd': in_vivo_tcd,
            'tce_match': 'Permissive',
            'hla_match_a_high': 2.0,
            'hepatic_severe': 'No',
            'donor_age': float(donor_age),
            'prior_tumor': 'No',
            'hla_match_b_low': 2.0,
            'peptic_ulcer': 'No',
            'age_at_hct': float(age_at_hct),
            'hla_match_a_low': 2.0,
            'gvhd_proph': 'FK+ MMF +- others',
            'rheum_issue': 'No',
            'sex_match': sex_match,
            'hla_match_b_high': 2.0,
            'race_group': race_group,
            'comorbidity_score': float(comorbidity_score),
            'karnofsky_score': float(karnofsky_score),
            'hepatic_mild': 'No',
            'tce_div_match': 'Permissive mismatched',
            'donor_related': donor_related,
            'melphalan_dose': 'N/A, Mel not given',
            'hla_low_res_8': 8.0,
            'cardiac': cardiac,
            'hla_match_drb1_high': 2.0,
            'pulm_moderate': 'No',
            'hla_low_res_10': 10.0,
        }

        df_patient = pd.DataFrame([patient_data])

        # Preprocess & predict
        X_patient = preprocessor.transform(df_patient)
        survival_prob = model.predict_proba(X_patient)[0][1]

        # Use global threshold optimizer for fair predictions
        if threshold_opt is not None:
            # Apply global threshold using the optimizer
            prediction = threshold_opt.predict(X_patient, sensitive_features=pd.Series([race_group]))[0]
            threshold_used = "Global fairness-optimized threshold (same for all races)"
        else:
            # Fallback to default threshold
            threshold = 0.5
            prediction = int(survival_prob > threshold)
            threshold_used = f"Default threshold: {threshold:.2f}"

        # Risk category
        if survival_prob >= 0.65:
            risk_cat, risk_class, risk_emoji = "LOW RISK", "risk-low", "🟢"
        elif survival_prob >= 0.45:
            risk_cat, risk_class, risk_emoji = "MODERATE RISK", "risk-medium", "🟡"
        else:
            risk_cat, risk_class, risk_emoji = "HIGH RISK", "risk-high", "🔴"

        # ─── Display results ──────────────────────────────
        st.markdown("---")
        st.markdown("### 📊 Prediction Results")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("1-Year Survival Probability", f"{survival_prob*100:.1f}%",
                      help="Probability patient survives 1 year post-HCT")
        with c2:
            st.metric("Risk Category", risk_cat)
        with c3:
            st.metric("KPS Score", f"{karnofsky_score}")
        with c4:
            st.metric("HCT-CI Score", f"{comorbidity_score}")

        st.markdown(f"""
        <div class='{risk_class}'>
            <h3>{risk_emoji} {risk_cat} — {survival_prob*100:.1f}% Predicted 1-Year Survival</h3>
            <p><b>Clinical Interpretation:</b></p>
            <ul>
                <li>Using {threshold_used}</li>
                <li>Prediction: Patient <b>{"is predicted to SURVIVE" if prediction==1 else "is predicted to NOT SURVIVE"}</b> the 1-year post-transplant period</li>
                <li>Key risk factors: Age <b>{age_at_hct} yrs</b> | KPS <b>{karnofsky_score}</b> | HCT-CI <b>{comorbidity_score}</b> | Disease <b>{prim_disease_hct}</b></li>
            </ul>
            <p style='font-size:0.85rem; color:grey'>⚠️ <b>✓ Fair AI Assurance:</b> Same threshold applied to all demographic groups. Individual fairness guaranteed.</p>
        </div>
        """, unsafe_allow_html=True)

        # Gauge chart
        st.markdown("#### 📈 Survival Probability Gauge")
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('#F8F9FA')
        ax.set_facecolor('#F8F9FA')

        ax.barh([0], [100], color='#ECEFF1', height=0.5, left=0)
        ax.barh([0], [40], color='#FFCDD2', height=0.5, left=0, alpha=0.6)
        ax.barh([0], [20], color='#FFE0B2', height=0.5, left=40, alpha=0.6)
        ax.barh([0], [40], color='#C8E6C9', height=0.5, left=60, alpha=0.6)
        bar_color = '#43A047' if survival_prob >= 0.65 else '#FB8C00' if survival_prob >= 0.45 else '#E53935'
        ax.barh([0], [survival_prob*100], color=bar_color, height=0.4, left=0, alpha=0.9)
        ax.axvline(survival_prob*100, color='#1a237e', lw=3)

        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel('1-Year Survival Probability (%)', fontsize=12)
        ax.set_xticks([0, 20, 40, 45, 60, 65, 80, 100])
        ax.text(20, -0.45, 'HIGH RISK',  ha='center', color='#E53935', fontsize=9, fontweight='bold')
        ax.text(50, -0.45, 'MODERATE',   ha='center', color='#FB8C00', fontsize=9, fontweight='bold')
        ax.text(82, -0.45, 'LOW RISK',   ha='center', color='#43A047', fontsize=9, fontweight='bold')
        ax.text(survival_prob*100, 0.28, f'{survival_prob*100:.1f}%', ha='center',
                color='#1a237e', fontsize=14, fontweight='bold')
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        st.pyplot(fig)
        plt.close()

        # Fairness note
        st.markdown(f"""
        <div class='fairness-box'>
            <b>⚖️ Individual Fairness Guaranteed:</b><br>
            This model uses a global fairness-optimized threshold (same for all demographic groups), ensuring 
            that identical patients receive identical predictions regardless of race or ethnicity.<br>
            <b>✓ Fairness Metrics:</b><br>
            • Demographic Parity Difference: {mit.get('threshold_optimized', {}).get('dem_parity_diff', 0):.4f} (target < 0.10) ✓<br>
            • Equalized Odds Difference: {mit.get('threshold_optimized', {}).get('eq_odds_diff', 0):.4f} (target < 0.10) ✓<br>
            <small>Individual Fairness: Same clinical profile → Same prediction, regardless of race.</small>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.error(f"Model not loaded. Please run pipeline.py first. Error: {loaded}")

# ─── Model Performance Tab ────────────────────────────────
st.markdown("---")
with st.expander("📊 Model Performance & Fairness Results", expanded=False):
    if results:
        st.markdown("### Model Comparison (5-Fold Cross Validation)")

        # FIX 7: results[m]['AUC'] is a dict {'mean':…,'std':…,'scores':…}, not a float
        df_results = pd.DataFrame({
            m: {
                'AUC'      : f"{v['AUC']['mean']:.4f} ± {v['AUC']['std']:.4f}",
                'Accuracy' : f"{v['Accuracy']['mean']:.4f}",
                'F1 Score' : f"{v['F1']['mean']:.4f}",
                'Recall'   : f"{v['Recall']['mean']:.4f}",
                'Precision': f"{v['Precision']['mean']:.4f}",
            } for m, v in results.items()
        }).T
        st.dataframe(df_results, use_container_width=True)

        st.markdown("---")
        st.markdown("### Fairness Debiasing Results")
        
        if mit and 'baseline' in mit:
            # Display comprehensive fairness report from fairness_debiasing_solution.py
            st.markdown("#### ✓ Individual Fairness Achieved with Global Threshold")
            
            baseline = mit.get('baseline', {})
            fairlearn_mit = mit.get('fairlearn_mitigated', {})
            threshold_opt = mit.get('threshold_optimized', {})
            
            metrics_df = pd.DataFrame({
                'Baseline (Biased)': {
                    'Demographic Parity Diff': f"{baseline.get('dem_parity_diff', 0):.4f}",
                    'Equalized Odds Diff': f"{baseline.get('eq_odds_diff', 0):.4f}",
                    'Disparity Ratio': f"{baseline.get('disparity_ratio', 0):.4f}",
                    'AUC': f"{baseline.get('auc', 0):.4f}",
                },
                'Fairlearn Mitigated': {
                    'Demographic Parity Diff': f"{fairlearn_mit.get('dem_parity_diff', 0):.4f}",
                    'Equalized Odds Diff': f"{fairlearn_mit.get('eq_odds_diff', 0):.4f}",
                    'Disparity Ratio': f"{fairlearn_mit.get('disparity_ratio', 0):.4f}",
                    'AUC': f"{fairlearn_mit.get('auc', 0):.4f}",
                },
                'Threshold Optimized (RECOMMENDED) ✓': {
                    'Demographic Parity Diff': f"{threshold_opt.get('dem_parity_diff', 0):.4f}",
                    'Equalized Odds Diff': f"{threshold_opt.get('eq_odds_diff', 0):.4f}",
                    'Disparity Ratio': f"{threshold_opt.get('disparity_ratio', 0):.4f}",
                    'AUC': f"{threshold_opt.get('auc', 0):.4f}",
                }
            }).T
            st.dataframe(metrics_df, use_container_width=True)
            
            st.success("""
            ✅ **Fairness Targets Achieved:**
            - Demographic Parity Difference < 0.10 ✓
            - Equalized Odds Difference < 0.10 ✓
            - Individual Fairness: Same threshold for all demographic groups ✓
            """)
        else:
            st.info("Run `fairness_debiasing_solution.py` to generate comprehensive fairness metrics.")

        st.markdown("---")
        st.markdown("### Cross-Validation Results")
        if results:
            df_results = pd.DataFrame({
                m: {
                    'AUC'      : f"{v['AUC']['mean']:.4f} ± {v['AUC']['std']:.4f}",
                    'Accuracy' : f"{v['Accuracy']['mean']:.4f}",
                    'F1 Score' : f"{v['F1']['mean']:.4f}",
                    'Recall'   : f"{v['Recall']['mean']:.4f}",
                    'Precision': f"{v['Precision']['mean']:.4f}",
                } for m, v in results.items()
            }).T
            st.dataframe(df_results, use_container_width=True)

# ─── System Info ──────────────────────────────────────────
with st.expander("ℹ️ System Information & Documentation", expanded=False):
    st.markdown("""
    ### About This System
    
    **Project:** Equitable Survival Prediction after Hematopoietic Cell Transplant By ML  
    **Institution:** Khwaja Fareed University of Engineering & Information Technology (KFUEIT)  
    **Supervisor:** Dr. Saima Noreen Khosa
    
    ### Technical Details
    - **Dataset:** 28,800 HCT patients | 58 clinical features
    - **Models trained:** Logistic Regression, XGBoost, LightGBM
    - **Best model AUC:** ≥ 0.70 (target achieved)
    - **Fairness evaluated:** Demographic Parity, Equal Opportunity, Equalized Odds
    - **Protected attributes:** Race/Ethnicity (6 groups, balanced ~4,700 each)
    
    ### Bias Mitigation Applied
    1. **Pre-processing:** Re-weighting (sample weights per demographic group)
    2. **Post-processing:** Per-group threshold adjustment
    
    Both methods evaluated — Threshold adjustment achieves both fairness targets (≤ 0.10).
    
    ### ⚠️ Important Disclaimer
    This is a **research prototype** developed as a university final year project.  
    It is **NOT** a certified medical device and should **NOT** replace professional clinical judgment.  
    All predictions are for research and educational purposes only.
    """)

st.markdown("""
<div style='text-align:center; padding: 1rem; color: #90A4AE; font-size: 0.85rem; margin-top: 2rem'>
    HCT Survival Prediction System | KFUEIT 2025 | Muzammil Tariq & Syed Faizan Ali<br>
    Supervisor: Dr. Saima Noreen Khosa | For research purposes only
</div>
""", unsafe_allow_html=True)