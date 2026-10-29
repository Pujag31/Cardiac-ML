# ❤️ CardioGuard AI: Clinical Decision Support & Risk Stratification System

An end-to-end Machine Learning pipeline and interactive clinical decision support web application for **Cardiovascular Disease (CVD) & Coronary Heart Disease (CHD)** risk stratification.

Built with **Scikit-learn**, **Pandas**, **Streamlit**, and **Plotly**.

---

## 🏥 Clinical Problem Formulation

- **Disease Focus**: Coronary Artery Disease (CAD) / Significant Stenosis (>50% diameter narrowing)
- **Objective**: Stratify incoming patients into calibrated clinical risk categories (Low, Moderate, High, Critical) to guide physician referral, urgent cardiology triage, and preventative pharmacotherapy.
- **Why ML is uniquely suited**: Cardiac ischemia presents through multifaceted clinical, electrophysiological (ST depression, angina), and metabolic biomarkers (cholesterol, blood pressure). Clinicians require both **calibrated probabilities** and **interpretable rationale (Explainable AI)** to take life-saving action.

---

## 📁 Project Structure

```
healthcare_ml_app/
│
├── data/
│   ├── generate_dataset.py       # Realistic clinical cohort generator (UCI/Framingham calibrated)
│   ├── heart_disease_clinical.csv # 1,500 patient cohort
│   └── sample_patients_cohort.csv # 25 patient sample for batch clinical screening
│
├── src/
│   ├── pipeline.py               # Complete preprocessing, cross-validation & ML pipeline
│   └── explainability.py         # Global feature importance & patient-level XAI waterfall
│
├── models/
│   ├── heart_disease_pipeline.joblib # Serialized scikit-learn best model pipeline
│   └── evaluation_metrics.json       # Holdout validation metrics & ROC curves
│
├── app.py                        # Interactive Streamlit Clinical Web Dashboard
├── train.py                      # Training & 5-fold cross-validation benchmarking runner
├── run_app.bat                   # 1-click Windows launcher
└── README.md                     # Documentation & Clinical Guide
```

---

## 🚀 How to Run the Application

### 1. Launch the Web Application
Open your terminal or PowerShell and run:
```powershell
cd C:\Users\pujag\.gemini\antigravity\scratch\healthcare_ml_app
streamlit run app.py
```
Or double-click `run_app.bat`.

The application will launch at: `http://localhost:8501`

---

## 🩺 Application Capabilities

1. **Patient Risk Assessment (CDSS)**:
   - Interactive diagnostic inputs (demographics, ECG findings, ST depression, fluoroscopy vessels, blood pressure, cholesterol).
   - Instant 1-click preset clinical profiles (High Risk, Moderate Risk, Low Risk).
   - Real-time probability prediction with risk triage badge (Critical, High, Moderate, Low).
   - **Explainable AI (XAI)**: Visualizes biomarker contributions (e.g. ST depression pushing risk up vs high heart rate reserve protective).
   - **Clinical Decision Support (CDS)**: Actionable clinical recommendations aligned with AHA/ACC guidelines.
   - One-click downloadable patient summary report (`.txt`).

2. **Population Health & Cohort Screening**:
   - Screen whole clinics or hospital wards simultaneously.
   - Automatic triage queue sorted by risk urgency.
   - Interactive population risk histograms and scatter distributions.
   - Export prioritized triage list as CSV.

3. **Rigorous Clinical Validation & Governance**:
   - 5-Fold Stratified Cross-Validation benchmarking between Logistic Regression, Random Forest, and Gradient Boosting.
   - ROC curves, Sensitivity/Recall (prioritizing patient safety against false negatives), Specificity, and Brier score calibration.
   - Global feature importance analysis.
   - HIPAA and SaMD regulatory compliance disclaimers.

4. **Interactive ML Lifecycle Educational Guide**:
   - Step-by-step breakdown of the 8 pillars of healthcare machine learning.
