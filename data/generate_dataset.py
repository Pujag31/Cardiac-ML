"""
Synthetic Clinical Dataset Generator for Cardiovascular Disease (CVD)
Modeled closely after the UCI Heart Disease (Cleveland/Framingham) cohorts,
incorporating physiologically realistic correlations between clinical biomarkers and CAD risk.
"""

import numpy as np
import pandas as pd
import os

def generate_cardiovascular_dataset(n_samples: int = 1500, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)
    
    # 1. Age (years): Range 29 - 80, centered around 54
    age = np.clip(np.random.normal(loc=54.5, scale=9.5, size=n_samples).astype(int), 29, 82)
    
    # 2. Sex: 0 = Female (~32%), 1 = Male (~68%)
    sex = np.random.binomial(n=1, p=0.68, size=n_samples)
    
    # 3. Chest Pain Type (cp):
    # 0 = Typical Angina (48%)
    # 1 = Atypical Angina (16%)
    # 2 = Non-anginal Pain (28%)
    # 3 = Asymptomatic (8%)
    cp = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.48, 0.17, 0.27, 0.08])
    
    # 4. Resting Blood Pressure (trestbps) in mm Hg: Mean ~131, SD ~17.5
    # Correlated with age
    resting_bp = np.clip(
        (115 + 0.3 * (age - 40) + np.random.normal(0, 15, size=n_samples)).astype(int),
        94, 200
    )
    
    # 5. Serum Cholesterol (chol) in mg/dl: Mean ~246, SD ~51
    cholesterol = np.clip(
        (220 + 0.4 * (age - 40) + np.random.normal(0, 45, size=n_samples)).astype(int),
        126, 564
    )
    
    # 6. Fasting Blood Sugar > 120 mg/dl (fbs): 0 = No (~85%), 1 = Yes (~15%)
    # Slightly higher for older age
    fbs_prob = np.clip(0.10 + 0.002 * (age - 40), 0.05, 0.30)
    fasting_bs = (np.random.rand(n_samples) < fbs_prob).astype(int)
    
    # 7. Resting ECG results (restecg):
    # 0 = Normal (49%)
    # 1 = ST-T wave abnormality (48%)
    # 2 = Left ventricular hypertrophy (3%)
    resting_ecg = np.random.choice([0, 1, 2], size=n_samples, p=[0.49, 0.48, 0.03])
    
    # 8. Maximum Heart Rate Achieved (thalach):
    # Clinically, Max HR decreases with age: ~ 220 - age with variance
    expected_max_hr = 205 - 0.85 * age
    max_hr = np.clip(
        np.random.normal(expected_max_hr, 21, size=n_samples).astype(int),
        71, 202
    )
    
    # 9. Exercise-Induced Angina (exang): 0 = No, 1 = Yes
    # Higher probability if typical angina or low max HR
    exang_prob = np.clip(0.20 + 0.25 * (cp == 0) + 0.003 * (160 - max_hr), 0.05, 0.85)
    exercise_angina = (np.random.rand(n_samples) < exang_prob).astype(int)
    
    # 10. Oldpeak: ST depression induced by exercise relative to rest (0.0 - 6.2 mm)
    oldpeak = np.round(np.clip(
        np.random.exponential(scale=0.8, size=n_samples) + 0.6 * exercise_angina + 0.02 * (age - 45),
        0.0, 6.2
    ), 1)
    
    # 11. Slope of the peak exercise ST segment (slope):
    # 0 = Upsloping (mild risk), 1 = Flat (moderate risk), 2 = Downsloping (high ischemic risk)
    slope = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        if oldpeak[i] > 1.8:
            p = [0.10, 0.45, 0.45]
        elif oldpeak[i] > 0.8:
            p = [0.30, 0.50, 0.20]
        else:
            p = [0.70, 0.25, 0.05]
        slope[i] = np.random.choice([0, 1, 2], p=p)
    
    # 12. Number of major vessels (0-3) colored by fluoroscopy (ca)
    # Strongly age and disease correlated
    ca_base = 0.015 * (age - 35) + 0.3 * (oldpeak > 1.2)
    ca_probs = np.clip(ca_base, 0.0, 0.8)
    ca = np.random.binomial(n=3, p=np.clip(ca_probs / 3, 0.05, 0.7), size=n_samples)
    
    # 13. Thalassemia (thal):
    # 1 = Normal, 2 = Fixed defect, 3 = Reversible defect
    thal = np.random.choice([1, 2, 3], size=n_samples, p=[0.55, 0.10, 0.35])
    
    # 14. Target Variable: True Coronary Artery Disease Status (0 = No, 1 = Yes)
    # Modeled via a calibrated clinical logit formula
    logit = (
        -4.2
        + 0.045 * (age - 50)
        + 0.75 * sex
        - 0.55 * cp  # 0 is typical angina, higher risk
        + 0.012 * (resting_bp - 120)
        + 0.005 * (cholesterol - 200)
        + 0.45 * fasting_bs
        + 0.35 * resting_ecg
        - 0.025 * (max_hr - 140)
        + 0.95 * exercise_angina
        + 0.55 * oldpeak
        + 0.40 * slope
        + 0.85 * ca
        + 0.60 * (thal == 3)
    )
    
    prob = 1.0 / (1.0 + np.exp(-logit))
    target = (np.random.rand(n_samples) < prob).astype(int)
    
    df = pd.DataFrame({
        'age': age,
        'sex': sex,
        'chest_pain_type': cp,
        'resting_bp': resting_bp,
        'cholesterol': cholesterol,
        'fasting_bs': fasting_bs,
        'resting_ecg': resting_ecg,
        'max_hr': max_hr,
        'exercise_angina': exercise_angina,
        'oldpeak': oldpeak,
        'st_slope': slope,
        'num_major_vessels': ca,
        'thalassemia': thal,
        'target': target
    })
    
    return df

if __name__ == '__main__':
    out_dir = os.path.join(os.path.dirname(__file__))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'heart_disease_clinical.csv')
    
    df = generate_cardiovascular_dataset(n_samples=1500, random_state=42)
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} patient records saved to {out_path}")
    print(f"Target distribution:\n{df['target'].value_counts(normalize=True)}")
