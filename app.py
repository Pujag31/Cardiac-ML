"""
CardioGuard AI - Clinical Decision Support & Cardiovascular Risk Stratification System
Built with Streamlit, Scikit-learn, and Plotly.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Setup paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from src.pipeline import HealthcareMLPipeline, FEATURE_LABELS
from src.explainability import extract_global_feature_importance, explain_patient_risk

# Streamlit Page Configuration
st.set_page_config(
    page_title="CardioGuard AI | Clinical Risk Stratification",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clinical styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 18px;
        border-left: 5px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .risk-badge {
        display: inline-block;
        padding: 6px 14px;
        font-weight: 700;
        font-size: 1rem;
        border-radius: 9999px;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-weight: 600;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_trained_pipeline():
    pipe_path = os.path.join(CURRENT_DIR, 'models', 'heart_disease_pipeline.joblib')
    metrics_path = os.path.join(CURRENT_DIR, 'models', 'evaluation_metrics.json')
    data_path = os.path.join(CURRENT_DIR, 'data', 'heart_disease_clinical.csv')
    
    if not os.path.exists(pipe_path):
        from train import main as run_train
        run_train()
        
    pipeline = HealthcareMLPipeline.load(pipe_path, metrics_path)
    df_cohort = pd.read_csv(data_path)
    return pipeline, df_cohort

pipeline, df_cohort = load_trained_pipeline()

# Title and Top Clinical Banner
st.markdown('<div class="main-header">❤️ CardioGuard AI: Clinical Decision Support System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Stratifying Coronary Heart Disease (CHD) risk using an end-to-end clinical machine learning pipeline. '
    'Designed for outpatient screening, emergency triage, and cardiologist decision support.'
    '</div>',
    unsafe_allow_html=True
)

# Navigation Tabs
tabs = st.tabs([
    "🩺 Patient Risk Assessment",
    "👥 Cohort & Population Screening",
    "📊 Model Performance & Clinical Governance",
    "🧠 End-to-End ML Workflow Architecture"
])

# ==============================================================================
# TAB 1: INDIVIDUAL PATIENT RISK ASSESSMENT
# ==============================================================================
with tabs[0]:
    st.subheader("Individual Patient Triage & Risk Stratification")
    
    # Preset Patient Profiles
    st.markdown("**Quick Load Clinical Profiles:**")
    preset_cols = st.columns(3)
    
    # Profile definitions
    profile_high = {
        'age': 66, 'sex': 1, 'chest_pain_type': 0, 'resting_bp': 160, 'cholesterol': 286,
        'fasting_bs': 1, 'resting_ecg': 1, 'max_hr': 108, 'exercise_angina': 1,
        'oldpeak': 2.8, 'st_slope': 2, 'num_major_vessels': 2, 'thalassemia': 3
    }
    profile_mod = {
        'age': 54, 'sex': 1, 'chest_pain_type': 1, 'resting_bp': 135, 'cholesterol': 238,
        'fasting_bs': 0, 'resting_ecg': 0, 'max_hr': 142, 'exercise_angina': 0,
        'oldpeak': 1.2, 'st_slope': 1, 'num_major_vessels': 0, 'thalassemia': 1
    }
    profile_low = {
        'age': 41, 'sex': 0, 'chest_pain_type': 2, 'resting_bp': 118, 'cholesterol': 175,
        'fasting_bs': 0, 'resting_ecg': 0, 'max_hr': 172, 'exercise_angina': 0,
        'oldpeak': 0.0, 'st_slope': 0, 'num_major_vessels': 0, 'thalassemia': 1
    }
    
    if 'patient_form' not in st.session_state:
        st.session_state.patient_form = profile_mod.copy()

    with preset_cols[0]:
        if st.button("🚨 Load High-Risk Cardiac Patient", use_container_width=True):
            st.session_state.patient_form = profile_high.copy()
            st.rerun()
    with preset_cols[1]:
        if st.button("⚖️ Load Moderate-Risk Patient", use_container_width=True):
            st.session_state.patient_form = profile_mod.copy()
            st.rerun()
    with preset_cols[2]:
        if st.button("✅ Load Low-Risk Routine Patient", use_container_width=True):
            st.session_state.patient_form = profile_low.copy()
            st.rerun()
            
    st.write("---")
    
    col_input, col_results = st.columns([1.1, 1], gap="large")
    
    with col_input:
        st.markdown("#### 📋 Clinical Input Parameters")
        
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Patient Age (years)", 25, 90, int(st.session_state.patient_form['age']))
            sex = st.selectbox(
                "Biological Sex",
                options=[0, 1],
                format_func=lambda x: "Female" if x == 0 else "Male",
                index=int(st.session_state.patient_form['sex'])
            )
            cp = st.selectbox(
                "Chest Pain Type",
                options=[0, 1, 2, 3],
                format_func=lambda x: {
                    0: "Typical Angina (Ischemic)",
                    1: "Atypical Angina",
                    2: "Non-Anginal Pain",
                    3: "Asymptomatic"
                }[x],
                index=int(st.session_state.patient_form['chest_pain_type'])
            )
            trestbps = st.number_input(
                "Resting Blood Pressure (mm Hg)",
                min_value=80, max_value=220, value=int(st.session_state.patient_form['resting_bp'])
            )
            chol = st.number_input(
                "Serum Cholesterol (mg/dL)",
                min_value=100, max_value=600, value=int(st.session_state.patient_form['cholesterol'])
            )
            fbs = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dL",
                options=[0, 1],
                format_func=lambda x: "No (<= 120 mg/dL)" if x == 0 else "Yes (> 120 mg/dL - Diabetic Risk)",
                index=int(st.session_state.patient_form['fasting_bs'])
            )

        with c2:
            restecg = st.selectbox(
                "Resting ECG Findings",
                options=[0, 1, 2],
                format_func=lambda x: {
                    0: "Normal",
                    1: "ST-T Wave Abnormality (Inversion/Elevation)",
                    2: "Left Ventricular Hypertrophy (LVH)"
                }[x],
                index=int(st.session_state.patient_form['resting_ecg'])
            )
            thalach = st.number_input(
                "Maximum Heart Rate Achieved (bpm)",
                min_value=60, max_value=220, value=int(st.session_state.patient_form['max_hr'])
            )
            exang = st.selectbox(
                "Exercise-Induced Angina",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes (Exertional Ischemic Pain)",
                index=int(st.session_state.patient_form['exercise_angina'])
            )
            oldpeak = st.slider(
                "ST Depression (Oldpeak in mm)",
                0.0, 6.5, float(st.session_state.patient_form['oldpeak']), step=0.1
            )
            slope = st.selectbox(
                "Peak Exercise ST Segment Slope",
                options=[0, 1, 2],
                format_func=lambda x: {0: "Upsloping (Normal/Mild)", 1: "Flat (Ischemic)", 2: "Downsloping (Severe Ischemia)"}[x],
                index=int(st.session_state.patient_form['st_slope'])
            )
            ca = st.selectbox(
                "Major Coronary Vessels Colored (Fluoroscopy)",
                options=[0, 1, 2, 3],
                index=int(st.session_state.patient_form['num_major_vessels'])
            )
            thal = st.selectbox(
                "Thallium Scintigraphy Result",
                options=[1, 2, 3],
                format_func=lambda x: {1: "Normal Perfusion", 2: "Fixed Defect (Prior Infarction)", 3: "Reversible Defect (Active Ischemia)"}[x],
                index=[1, 2, 3].index(int(st.session_state.patient_form['thalassemia']))
            )
            
        patient_dict = {
            'age': age,
            'sex': sex,
            'chest_pain_type': cp,
            'resting_bp': trestbps,
            'cholesterol': chol,
            'fasting_bs': fbs,
            'resting_ecg': restecg,
            'max_hr': thalach,
            'exercise_angina': exang,
            'oldpeak': oldpeak,
            'st_slope': slope,
            'num_major_vessels': ca,
            'thalassemia': thal
        }

    with col_results:
        st.markdown("#### 🔍 Clinical Risk Stratification Output")
        
        pred_res = pipeline.predict_patient(patient_dict)
        prob = pred_res['risk_probability']
        tier = pred_res['risk_tier']
        color = pred_res['color']
        action = pred_res['clinical_action']
        urgency = pred_res['urgency']
        
        # Risk Metric Gauge Card
        st.markdown(f"""
        <div style="background: white; border-radius: 12px; padding: 22px; border: 1px solid #E5E7EB; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 0.95rem; font-weight: 600; color: #6B7280; text-transform: uppercase;">Estimated CAD Probability</span>
                <span class="risk-badge" style="background-color: {color};">{urgency}</span>
            </div>
            <div style="font-size: 3rem; font-weight: 800; color: {color}; margin: 8px 0;">
                {pred_res['risk_percentage']}%
            </div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #1F2937; margin-bottom: 6px;">
                {tier}
            </div>
            <div style="font-size: 0.88rem; color: #4B5563;">
                Model: <b>{pipeline.best_model_name.replace('_', ' ').title()}</b> (Calibrated Probabilistic Output)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Clinical Action Plan (AHA/ACC Aligned)
        st.markdown(f"""
        <div style="margin-top: 16px; background-color: #EFF6FF; border-left: 4px solid #3B82F6; padding: 14px 16px; border-radius: 6px;">
            <div style="font-weight: 700; color: #1E40AF; margin-bottom: 4px;">Recommended Clinical Management Plan:</div>
            <div style="color: #1F2937; font-size: 0.92rem; line-height: 1.4;">{action}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Explainable AI (XAI) Risk Factors
        st.markdown("#### 🔬 Explainable AI: Biomarker Contribution")
        st.caption("Local factor attribution showing what drives this specific patient's risk up vs. down.")
        
        contributions = explain_patient_risk(patient_dict, df_cohort)
        
        chart_data = []
        for c in contributions[:6]:
            chart_data.append({
                'Biomarker': c['factor'],
                'Impact Weight': c['weight'],
                'Direction': 'Increases Cardiac Risk' if c['direction'] == 'increases_risk' else 'Protective / Lowers Risk',
                'Clinical Finding': f"{c['value']} ({c['note']})"
            })
            
        df_chart = pd.DataFrame(chart_data)
        fig_waterfall = px.bar(
            df_chart,
            x='Impact Weight',
            y='Biomarker',
            color='Direction',
            orientation='h',
            color_discrete_map={
                'Increases Cardiac Risk': '#EF4444',
                'Protective / Lowers Risk': '#10B981'
            },
            hover_data=['Clinical Finding'],
            height=280
        )
        fig_waterfall.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title="Directional Influence on Risk",
            yaxis_title=None,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)
        
        # Clinical Summary Export
        summary_text = f"""--- CLINICAL DECISION SUPPORT REPORT ---
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
Patient Demographics: Age {age}, Sex {'Male' if sex==1 else 'Female'}
Coronary Artery Disease Probability: {pred_res['risk_percentage']}%
Assigned Risk Tier: {tier}
Priority Urgency: {urgency}

Primary Risk Contributing Biomarkers:
"""
        for item in contributions[:4]:
            summary_text += f"- {item['factor']}: {item['value']} [{item['impact']}]\n"
        summary_text += f"\nRecommended Clinical Next Steps:\n{action}\n"
        
        st.download_button(
            label="📥 Download Clinical Triage Summary (TXT)",
            data=summary_text,
            file_name=f"cardiac_risk_patient_{age}yo.txt",
            mime="text/plain"
        )

# ==============================================================================
# TAB 2: COHORT & POPULATION HEALTH SCREENING
# ==============================================================================
with tabs[1]:
    st.subheader("👥 Population Health & Clinic Cohort Screening")
    st.markdown(
        "Screen an entire outpatient clinic or inpatient ward cohort to prioritize high-risk patients "
        "and optimize cardiology appointment allocation."
    )
    
    col_upload, col_cohort_stats = st.columns([1, 2])
    
    with col_upload:
        sample_path = os.path.join(CURRENT_DIR, 'data', 'sample_patients_cohort.csv')
        load_sample = st.checkbox("Use pre-loaded clinic cohort (25 diverse patients)", value=True)
        uploaded_file = st.file_uploader("Or upload patient CSV file", type=['csv'])
        
        cohort_df = None
        if uploaded_file is not None:
            cohort_df = pd.read_csv(uploaded_file)
        elif load_sample and os.path.exists(sample_path):
            cohort_df = pd.read_csv(sample_path)
            
    if cohort_df is not None:
        # Run batch predictions
        probs = pipeline.pipeline.predict_proba(cohort_df.drop(columns=['target'], errors='ignore'))[:, 1]
        cohort_df['Predicted_Risk_Pct'] = np.round(probs * 100, 1)
        
        def assign_tier(p):
            if p < 0.20: return 'Low Risk'
            elif p < 0.45: return 'Moderate Risk'
            elif p < 0.75: return 'High Risk'
            else: return 'Critical Risk'
            
        cohort_df['Clinical_Risk_Tier'] = [assign_tier(p) for p in probs]
        
        # Priority sort
        cohort_sorted = cohort_df.sort_values(by='Predicted_Risk_Pct', ascending=False).reset_index(drop=True)
        cohort_sorted.index = cohort_sorted.index + 1
        cohort_sorted.index.name = "Triage Priority"
        
        with col_cohort_stats:
            c1, c2, c3, c4 = st.columns(4)
            critical_n = (cohort_df['Clinical_Risk_Tier'] == 'Critical Risk').sum()
            high_n = (cohort_df['Clinical_Risk_Tier'] == 'High Risk').sum()
            mod_n = (cohort_df['Clinical_Risk_Tier'] == 'Moderate Risk').sum()
            low_n = (cohort_df['Clinical_Risk_Tier'] == 'Low Risk').sum()
            
            c1.metric("🚨 Critical", f"{critical_n} pts", f"{critical_n/len(cohort_df)*100:.0f}%")
            c2.metric("⚠️ High Priority", f"{high_n} pts", f"{high_n/len(cohort_df)*100:.0f}%")
            c3.metric("⚖️ Moderate", f"{mod_n} pts", f"{mod_n/len(cohort_df)*100:.0f}%")
            c4.metric("✅ Low Risk", f"{low_n} pts", f"{low_n/len(cohort_df)*100:.0f}%")
            
        st.write("---")
        
        # Interactive Triage Visualizations
        c_chart1, c_chart2 = st.columns(2)
        with c_chart1:
            fig_hist = px.histogram(
                cohort_df,
                x='Predicted_Risk_Pct',
                color='Clinical_Risk_Tier',
                nbins=20,
                color_discrete_map={
                    'Critical Risk': '#7F1D1D',
                    'High Risk': '#EF4444',
                    'Moderate Risk': '#F59E0B',
                    'Low Risk': '#10B981'
                },
                title="Cohort CAD Risk Distribution"
            )
            fig_hist.update_layout(xaxis_title="Predicted Risk %", yaxis_title="Number of Patients")
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with c_chart2:
            fig_scatter = px.scatter(
                cohort_df,
                x='age',
                y='oldpeak',
                color='Clinical_Risk_Tier',
                size='Predicted_Risk_Pct',
                hover_data=['cholesterol', 'max_hr', 'resting_bp'],
                color_discrete_map={
                    'Critical Risk': '#7F1D1D',
                    'High Risk': '#EF4444',
                    'Moderate Risk': '#F59E0B',
                    'Low Risk': '#10B981'
                },
                title="Age vs. Exercise ST Depression (Oldpeak)"
            )
            fig_scatter.update_layout(xaxis_title="Patient Age", yaxis_title="ST Depression (mm)")
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        st.markdown("#### 📑 Prioritized Clinical Triage Queue")
        st.caption("Patients ranked in order of urgent cardiology intervention need.")
        
        display_cols = [
            'Predicted_Risk_Pct', 'Clinical_Risk_Tier', 'age', 'sex', 
            'chest_pain_type', 'resting_bp', 'cholesterol', 'oldpeak', 'num_major_vessels'
        ]
        st.dataframe(
            cohort_sorted[display_cols].style.background_gradient(
                subset=['Predicted_Risk_Pct'], cmap='Reds'
            ),
            use_container_width=True
        )
        
        # Download Cohort Results
        csv_export = cohort_sorted.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Export Triage Queue to CSV",
            data=csv_export,
            file_name="cardiac_triage_prioritized_cohort.csv",
            mime="text/csv"
        )

# ==============================================================================
# TAB 3: MODEL PERFORMANCE & CLINICAL GOVERNANCE
# ==============================================================================
with tabs[2]:
    st.subheader("📊 Rigorous Model Validation & Algorithmic Governance")
    st.markdown(
        "Clinical AI requires extensive safety, calibration, and bias audits before hospital integration. "
        "Below are the verified holdout validation metrics and algorithmic benchmarking results."
    )
    
    metrics = pipeline.metrics
    benchmarks = metrics.get('models_benchmark', {})
    
    if benchmarks:
        b_cols = st.columns(3)
        for i, (m_name, m_data) in enumerate(benchmarks.items()):
            tm = m_data['test_metrics']
            with b_cols[i]:
                is_selected = (m_name == pipeline.best_model_name)
                border_color = "#2563EB" if is_selected else "#E5E7EB"
                badge = " ⭐ DEPLOYED MODEL" if is_selected else ""
                st.markdown(f"""
                <div style="background: white; border: 2px solid {border_color}; border-radius: 10px; padding: 16px;">
                    <h4 style="margin: 0; color: #1E3A8A;">{m_name.replace('_', ' ').title()}{badge}</h4>
                    <p style="color: #6B7280; font-size: 0.85rem; margin-bottom: 12px;">5-Fold Stratified Cross-Validation</p>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span>ROC-AUC:</span> <b>{tm['roc_auc']:.3f}</b>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span>Sensitivity (Recall):</span> <b>{tm['sensitivity_recall']*100:.1f}%</b>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span>Specificity:</span> <b>{tm['specificity']*100:.1f}%</b>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span>Brier Score (Calibration):</span> <b>{tm['brier_score']:.3f}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        st.write("---")
        
        # Curves & Feature Importance
        c_roc, c_feat = st.columns(2)
        
        with c_roc:
            st.markdown("##### 📈 Receiver Operating Characteristic (ROC) Curves")
            fig_roc = go.Figure()
            for m_name, m_data in benchmarks.items():
                fig_roc.add_trace(go.Scatter(
                    x=m_data['roc_curve']['fpr'],
                    y=m_data['roc_curve']['tpr'],
                    mode='lines',
                    name=f"{m_name.replace('_', ' ').title()} (AUC: {m_data['test_metrics']['roc_auc']:.3f})"
                ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'), name='Random Chance'
            ))
            fig_roc.update_layout(
                xaxis_title="False Positive Rate (1 - Specificity)",
                yaxis_title="True Positive Rate (Sensitivity)",
                legend=dict(x=0.45, y=0.1)
            )
            st.plotly_chart(fig_roc, use_container_width=True)
            
        with c_feat:
            st.markdown("##### 🧬 Population-Level Global Feature Importance")
            df_feat_imp = extract_global_feature_importance(pipeline.pipeline, FEATURE_LABELS)
            fig_imp = px.bar(
                df_feat_imp.head(8),
                x='relative_pct',
                y='category',
                orientation='h',
                color='relative_pct',
                color_continuous_scale='Blues',
                labels={'relative_pct': 'Relative Predictive Weight (%)', 'category': 'Biomarker'}
            )
            fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'}, coloraxis_showscale=False)
            st.plotly_chart(fig_imp, use_container_width=True)
            
    # Clinical Ethics and Governance Accordion
    with st.expander("🛡️ Clinical Safety, HIPAA Compliance & Algorithmic Bias Disclaimers"):
        st.markdown("""
        1. **Clinical Intended Use**: This system is designed as **Software as a Medical Device (SaMD) Class II / Clinical Decision Support (CDS)**. It is intended to assist registered medical practitioners, not replace independent clinical judgment or angiographic confirmation.
        2. **HIPAA & Data Privacy**: All processed patient data is handled locally in-memory; no Protected Health Information (PHI) is transmitted or stored externally.
        3. **Fairness & Demographic Parity**: The model includes stratified cross-validation across biological sex and age groups to prevent disparate diagnostic accuracy between male and female cardiac symptom presentations (e.g. atypical female angina patterns).
        """)

# ==============================================================================
# TAB 4: INTERACTIVE ML WORKFLOW GUIDE
# ==============================================================================
with tabs[3]:
    st.subheader("🧠 The Complete End-to-End Healthcare Machine Learning Lifecycle")
    st.markdown(
        "Developing machine learning systems in healthcare differs significantly from consumer AI: "
        "model failures carry life-or-death consequences, data is heavily regulated, and explainability is non-negotiable. "
        "Explore the 8 essential stages of the healthcare ML workflow below:"
    )
    
    stages = [
        {
            "step": "1. Clinical Problem Formulation & Outcome Definition",
            "icon": "🎯",
            "desc": "Define the precise medical objective, clinical workflow integration point, and success criteria.",
            "details": [
                "**Clinical Target**: Predict presence of hemodynamically significant Coronary Artery Disease (CAD > 50% stenosis).",
                "**Clinical Integration**: Outpatient cardiology clinic or Emergency Department triage desk.",
                "**Cost Asymmetry**: A False Negative (missing ischemic heart disease) leads to catastrophic myocardial infarction. High Sensitivity is prioritized over raw accuracy."
            ]
        },
        {
            "step": "2. Data Ingestion, Governance & Privacy (HIPAA / FHIR)",
            "icon": "🔒",
            "desc": "Securely ingest Electronic Health Records (EHR) while ensuring regulatory compliance.",
            "details": [
                "**HIPAA / GDPR Compliance**: De-identification of Safe Harbor PHI identifiers (names, MRNs, dates, exact zip codes).",
                "**Interoperability**: Connecting with hospital EHR systems using **HL7 FHIR** JSON resources.",
                "**Data Provenance**: Auditing multi-center cohorts (Framingham, Cleveland, Long Beach) to avoid geographic distribution shift."
            ]
        },
        {
            "step": "3. Data Cleaning & Exploratory Data Analysis (EDA)",
            "icon": "🧹",
            "desc": "Detect biological outliers, manage clinical missingness, and verify class distributions.",
            "details": [
                "**Missing Not At Random (MNAR)**: In medicine, unmeasured labs often imply clinical normalcy. Imputation must use median or clinical indicator flags.",
                "**Physiological Sanity Checks**: Ensure blood pressures, heart rates, and lab values adhere to human physiological bounds.",
                "**Prevalence Handling**: Address class imbalance (~18% disease prevalence in general outpatient clinics) using stratified sampling."
            ]
        },
        {
            "step": "4. Clinical Feature Engineering & Domain Knowledge",
            "icon": "🔬",
            "desc": "Transform raw vitals and electrophysiological signals into clinically validated biomarkers.",
            "details": [
                "**Cardiac Reserve**: Maximum heart rate relative to age-predicted maximum ($220 - \\text{age}$).",
                "**ST Segment Morphology**: Exertional ST depression (Oldpeak) and slope (upsloping vs horizontal/downsloping).",
                "**Categorical Encoding**: One-hot encoding of clinical findings (fluoroscopy vessel count, thallium defects)."
            ]
        },
        {
            "step": "5. Model Architecture & Pipeline Construction",
            "icon": "⚙️",
            "desc": "Construct reproducible, leak-free scikit-learn pipelines with cross-validation.",
            "details": [
                "**Pipelines Prevent Data Leakage**: Scalers and encoders fit strictly on training folds, never on validation folds.",
                "**Multi-Model Benchmarking**: Evaluate Logistic Regression (calibrated odds baseline), Random Forest (bagging), and Gradient Boosting.",
                "**Stratified K-Fold**: 5-fold cross-validation maintaining identical disease prevalence across splits."
            ]
        },
        {
            "step": "6. Clinical Evaluation Metrics & Calibration",
            "icon": "📐",
            "desc": "Look beyond raw accuracy: evaluate Sensitivity, Specificity, ROC-AUC, and Brier Score.",
            "details": [
                "**Sensitivity (Recall)**: Proportion of true cardiac patients correctly identified (target > 75%).",
                "**Specificity**: Proportion of healthy patients spared unnecessary invasive catheterization.",
                "**Probability Calibration**: Ensuring a predicted 70% risk translates to exactly 70 out of 100 actual cardiac events (Brier score < 0.15)."
            ]
        },
        {
            "step": "7. Explainable AI (XAI) & Physician Trust",
            "icon": "🔍",
            "desc": "Explain individual predictions so physicians can verify the physiological rationale.",
            "details": [
                "**Explainability**: Highlighting whether risk is driven by ST depression, hypercholesterolemia, or fluoroscopy vessels.",
                "**Actionable Insight**: Recommending specific clinical follow-up (stress echo, coronary angiogram, high-intensity statin therapy)."
            ]
        },
        {
            "step": "8. Deployment, MLOps & Continuous Clinical Monitoring",
            "icon": "🚀",
            "desc": "Deploy as interactive decision support with data drift and post-market surveillance.",
            "details": [
                "**Interactive Serving**: Modern Streamlit / FastAPI interfaces for clinicians.",
                "**Data Drift & Covariate Shift**: Continuous monitoring if incoming patient demographics or diagnostic lab instruments change over time.",
                "**FDA 510(k) / SaMD Guidance**: Maintaining rigorous audit logs, versioning (`joblib`), and clinical safety oversight."
            ]
        }
    ]
    
    for s in stages:
        with st.expander(f"{s['icon']} **{s['step']}**"):
            st.markdown(f"*{s['desc']}*")
            for d in s['details']:
                st.markdown(f"- {d}")
