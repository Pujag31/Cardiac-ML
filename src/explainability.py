"""
Explainable AI (XAI) Module for Clinical Decision Support
Extracts global feature importances and computes patient-level local risk factor attributions.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np

def extract_global_feature_importance(pipeline, feature_labels: Dict[str, str]) -> pd.DataFrame:
    """
    Extracts normalized feature importances from the fitted pipeline.
    """
    clf = pipeline.named_steps['classifier']
    preprocessor = pipeline.named_steps['preprocessor']
    
    # Extract feature names from preprocessor
    num_cols = preprocessor.transformers_[0][2]
    cat_encoder = preprocessor.transformers_[1][1].named_steps['encoder']
    cat_cols = cat_encoder.get_feature_names_out(preprocessor.transformers_[1][2])
    all_feature_names = list(num_cols) + list(cat_cols)
    
    importances = None
    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
    elif hasattr(clf, 'coef_'):
        importances = np.abs(clf.coef_[0])
    else:
        importances = np.ones(len(all_feature_names))
        
    df_imp = pd.DataFrame({
        'feature': all_feature_names,
        'importance': importances
    })
    
    # Map back to readable parent feature category
    def map_parent(f):
        for orig in feature_labels.keys():
            if f.startswith(orig):
                return feature_labels[orig]
        return f

    df_imp['category'] = df_imp['feature'].apply(map_parent)
    # Aggregate one-hot categories back to parent clinical biomarker
    summary_df = df_imp.groupby('category', as_index=False)['importance'].sum()
    summary_df['relative_pct'] = (summary_df['importance'] / summary_df['importance'].sum()) * 100
    summary_df = summary_df.sort_values(by='relative_pct', ascending=False).reset_index(drop=True)
    return summary_df

def explain_patient_risk(patient_dict: Dict[str, Any], df_baseline: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Calculates patient-level risk factor attributions compared to clinical population norms.
    Identifies what factors push the risk UP versus what factors are PROTECTIVE.
    """
    contributions = []
    
    # 1. Oldpeak (ST depression)
    oldpeak = patient_dict.get('oldpeak', 0.0)
    if oldpeak >= 1.5:
        contributions.append({
            'factor': 'Exercise ST Depression (Oldpeak)',
            'value': f"{oldpeak:.1f} mm",
            'impact': 'High Risk Driver',
            'direction': 'increases_risk',
            'weight': 0.85 * (oldpeak / 2.0),
            'note': 'Significant ischemic myocardial response during exertion'
        })
    elif oldpeak > 0.5:
        contributions.append({
            'factor': 'Exercise ST Depression (Oldpeak)',
            'value': f"{oldpeak:.1f} mm",
            'impact': 'Moderate Risk Driver',
            'direction': 'increases_risk',
            'weight': 0.45 * oldpeak,
            'note': 'Mild ischemic ST segment depression noted'
        })
    else:
        contributions.append({
            'factor': 'Exercise ST Depression (Oldpeak)',
            'value': f"{oldpeak:.1f} mm",
            'impact': 'Protective / Normal',
            'direction': 'lowers_risk',
            'weight': -0.30,
            'note': 'Normal ST recovery without ischemic changes'
        })
        
    # 2. Number of Major Vessels colored by fluoroscopy
    vessels = patient_dict.get('num_major_vessels', 0)
    if vessels > 0:
        contributions.append({
            'factor': 'Fluoroscopy Occluded Vessels',
            'value': f"{vessels} vessel(s)",
            'impact': 'Critical Risk Driver',
            'direction': 'increases_risk',
            'weight': 0.90 * vessels,
            'note': f"Evidence of severe stenosis in {vessels} major coronary arteries"
        })
    else:
        contributions.append({
            'factor': 'Fluoroscopy Occluded Vessels',
            'value': "0 vessels",
            'impact': 'Protective',
            'direction': 'lowers_risk',
            'weight': -0.40,
            'note': 'Clear coronary vasculature on fluoroscopy'
        })
        
    # 3. Exercise Angina
    ex_ang = patient_dict.get('exercise_angina', 0)
    if ex_ang == 1:
        contributions.append({
            'factor': 'Exercise-Induced Angina',
            'value': 'Present (Positive)',
            'impact': 'High Risk Driver',
            'direction': 'increases_risk',
            'weight': 0.75,
            'note': 'Chest pain triggered by cardiac workload strongly correlates with CAD'
        })
    else:
        contributions.append({
            'factor': 'Exercise-Induced Angina',
            'value': 'Absent (Negative)',
            'impact': 'Protective',
            'direction': 'lowers_risk',
            'weight': -0.35,
            'note': 'No exertional angina experienced during stress testing'
        })
        
    # 4. Maximum Heart Rate achieved
    max_hr = patient_dict.get('max_hr', 150)
    age = patient_dict.get('age', 55)
    expected_max = 220 - age
    achieved_pct = (max_hr / expected_max) * 100 if expected_max > 0 else 100
    if achieved_pct < 75:
        contributions.append({
            'factor': 'Chronotropic Incompetence (Max HR)',
            'value': f"{max_hr} bpm ({achieved_pct:.0f}% of max)",
            'impact': 'Moderate Risk Driver',
            'direction': 'increases_risk',
            'weight': 0.50,
            'note': 'Sub-optimal target heart rate achieved; suggests impaired cardiac reserve'
        })
    elif achieved_pct >= 90:
        contributions.append({
            'factor': 'Cardiac Reserve (Max HR)',
            'value': f"{max_hr} bpm ({achieved_pct:.0f}% of max)",
            'impact': 'Protective',
            'direction': 'lowers_risk',
            'weight': -0.35,
            'note': 'Excellent workload tolerance and cardiovascular performance'
        })
        
    # 5. Serum Cholesterol
    chol = patient_dict.get('cholesterol', 200)
    if chol > 260:
        contributions.append({
            'factor': 'Serum Cholesterol',
            'value': f"{chol} mg/dL",
            'impact': 'Moderate Risk Driver',
            'direction': 'increases_risk',
            'weight': 0.40 * ((chol - 200) / 100),
            'note': 'Hypercholesterolemia accelerates atherosclerotic plaque development'
        })
    elif chol < 190:
        contributions.append({
            'factor': 'Serum Cholesterol',
            'value': f"{chol} mg/dL",
            'impact': 'Protective',
            'direction': 'lowers_risk',
            'weight': -0.20,
            'note': 'Optimal circulating cholesterol level'
        })

    # 6. Blood Pressure
    bp = patient_dict.get('resting_bp', 120)
    if bp >= 140:
        contributions.append({
            'factor': 'Resting Blood Pressure',
            'value': f"{bp} mmHg (Stage 2 HTN)",
            'impact': 'Moderate Risk Driver',
            'direction': 'increases_risk',
            'weight': 0.35 * ((bp - 120) / 30),
            'note': 'Hypertension increases endothelial shear stress and cardiac afterload'
        })
    elif bp <= 120:
        contributions.append({
            'factor': 'Resting Blood Pressure',
            'value': f"{bp} mmHg (Normal)",
            'impact': 'Protective',
            'direction': 'lowers_risk',
            'weight': -0.15,
            'note': 'Normotensive resting arterial pressure'
        })

    # 7. Thalassemia defect
    thal = patient_dict.get('thalassemia', 1)
    if thal == 3:
        contributions.append({
            'factor': 'Thallium Perfusion Defect',
            'value': 'Reversible Defect',
            'impact': 'High Risk Driver',
            'direction': 'increases_risk',
            'weight': 0.65,
            'note': 'Reversible perfusion defect indicates viable but critically ischemic myocardium'
        })

    contributions.sort(key=lambda x: abs(x['weight']), reverse=True)
    return contributions
