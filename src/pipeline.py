"""
End-to-End Machine Learning Pipeline for Healthcare Risk Stratification
Includes:
- Clinical Preprocessing (Imputation, Scaling, One-Hot Encoding)
- Cross-Validation with Healthcare Metrics (Sensitivity, Specificity, ROC-AUC, PR-AUC, Brier score)
- Model Benchmarking (Logistic Regression, Random Forest, Gradient Boosting)
- Pipeline Export and Metric Logging
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    recall_score,
    precision_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

NUMERICAL_FEATURES = ['age', 'resting_bp', 'cholesterol', 'max_hr', 'oldpeak']
CATEGORICAL_FEATURES = [
    'sex',
    'chest_pain_type',
    'fasting_bs',
    'resting_ecg',
    'exercise_angina',
    'st_slope',
    'num_major_vessels',
    'thalassemia'
]

FEATURE_LABELS = {
    'age': 'Age (years)',
    'sex': 'Biological Sex (0=F, 1=M)',
    'chest_pain_type': 'Chest Pain Type (0-3)',
    'resting_bp': 'Resting Blood Pressure (mmHg)',
    'cholesterol': 'Serum Cholesterol (mg/dL)',
    'fasting_bs': 'Fasting Blood Sugar >120mg/dL',
    'resting_ecg': 'Resting ECG Result (0-2)',
    'max_hr': 'Max Heart Rate (bpm)',
    'exercise_angina': 'Exercise-Induced Angina',
    'oldpeak': 'ST Depression (Oldpeak mm)',
    'st_slope': 'ST Segment Slope (0-2)',
    'num_major_vessels': 'Fluoroscopy Major Vessels (0-3)',
    'thalassemia': 'Thalassemia Defect (1-3)'
}

def create_preprocessor() -> ColumnTransformer:
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, NUMERICAL_FEATURES),
            ('cat', cat_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    return preprocessor

def evaluate_clinical_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall (Catch cardiac cases)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Avoid false alarms
    ppv = precision_score(y_true, y_pred, zero_division=0)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc_roc = roc_auc_score(y_true, y_prob)
    auc_pr = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    
    return {
        'sensitivity_recall': float(np.round(sensitivity, 4)),
        'specificity': float(np.round(specificity, 4)),
        'precision_ppv': float(np.round(ppv, 4)),
        'npv': float(np.round(npv, 4)),
        'f1_score': float(np.round(f1, 4)),
        'roc_auc': float(np.round(auc_roc, 4)),
        'pr_auc': float(np.round(auc_pr, 4)),
        'brier_score': float(np.round(brier, 4)),
        'confusion_matrix': {
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn)
        }
    }

class HealthcareMLPipeline:
    def __init__(self, model_type: str = 'gradient_boosting'):
        self.model_type = model_type
        self.pipeline: Pipeline = None
        self.metrics: Dict[str, Any] = {}
        self.best_model_name: str = ""
        
    def _get_classifier(self, model_name: str):
        if model_name == 'logistic_regression':
            return LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        elif model_name == 'random_forest':
            return RandomForestClassifier(
                n_estimators=120,
                max_depth=6,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        elif model_name == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.08,
                max_depth=3,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

    def train_and_benchmark(self, df: pd.DataFrame, target_col: str = 'target') -> Dict[str, Any]:
        X = df.drop(columns=[target_col])
        y = df[target_col].values
        
        # Split train/test holdout (80/20 stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=42
        )
        
        candidates = ['logistic_regression', 'random_forest', 'gradient_boosting']
        benchmark_results = {}
        fitted_pipelines = {}
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name in candidates:
            preprocessor = create_preprocessor()
            clf = self._get_classifier(name)
            pipe = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', clf)
            ])
            
            # Cross-validation metrics
            cv_roc, cv_sens, cv_spec = [], [], []
            for train_idx, val_idx in skf.split(X_train, y_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                
                pipe.fit(X_tr, y_tr)
                y_val_prob = pipe.predict_proba(X_val)[:, 1]
                y_val_pred = (y_val_prob >= 0.5).astype(int)
                
                eval_dict = evaluate_clinical_predictions(y_val, y_val_pred, y_val_prob)
                cv_roc.append(eval_dict['roc_auc'])
                cv_sens.append(eval_dict['sensitivity_recall'])
                cv_spec.append(eval_dict['specificity'])
                
            # Fit on full training set
            pipe.fit(X_train, y_train)
            fitted_pipelines[name] = pipe
            
            # Evaluate on independent Holdout Test Set
            y_test_prob = pipe.predict_proba(X_test)[:, 1]
            y_test_pred = (y_test_prob >= 0.5).astype(int)
            test_metrics = evaluate_clinical_predictions(y_test, y_test_pred, y_test_prob)
            
            fpr, tpr, _ = roc_curve(y_test, y_test_prob)
            prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_test_prob)
            
            benchmark_results[name] = {
                'cv_mean_roc_auc': float(np.round(np.mean(cv_roc), 4)),
                'cv_mean_sensitivity': float(np.round(np.mean(cv_sens), 4)),
                'cv_mean_specificity': float(np.round(np.mean(cv_spec), 4)),
                'test_metrics': test_metrics,
                'roc_curve': {'fpr': fpr.tolist(), 'tpr': tpr.tolist()},
                'pr_curve': {'precision': prec_curve.tolist(), 'recall': rec_curve.tolist()}
            }
            
        # Select best model based on ROC-AUC & Sensitivity (prioritizing patient safety)
        best_name = max(
            candidates,
            key=lambda k: benchmark_results[k]['test_metrics']['roc_auc'] * 0.6 +
                          benchmark_results[k]['test_metrics']['sensitivity_recall'] * 0.4
        )
        
        self.best_model_name = best_name
        self.pipeline = fitted_pipelines[best_name]
        self.metrics = {
            'selected_model': best_name,
            'models_benchmark': benchmark_results,
            'features': {
                'numerical': NUMERICAL_FEATURES,
                'categorical': CATEGORICAL_FEATURES,
                'labels': FEATURE_LABELS
            }
        }
        return self.metrics
    
    def predict_patient(self, patient_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates calibrated risk probability and stratifies patient into clinical triage tiers.
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline has not been trained or loaded.")
            
        df = pd.DataFrame([patient_dict])
        prob = float(self.pipeline.predict_proba(df)[0, 1])
        prediction = int(prob >= 0.5)
        
        # Clinical Risk Stratification Tiers
        if prob < 0.20:
            tier = "Low Risk (Routine Follow-up)"
            color = "#10B981"  # Emerald
            urgency = "Standard"
            action = "Continue regular primary care check-ups and standard healthy lifestyle counselling."
        elif prob < 0.45:
            tier = "Moderate Risk (Preventative Attention)"
            color = "#F59E0B"  # Amber
            urgency = "Intermediate"
            action = "Recommend fasting lipid panel, HbA1c screening, lifestyle modifications, and repeat cardiovascular assessment in 6 months."
        elif prob < 0.75:
            tier = "High Risk (Cardiology Referral)"
            color = "#EF4444"  # Rose
            urgency = "High Priority"
            action = "Schedule formal Cardiology consultation. Consider non-invasive cardiac imaging (Stress Echo or Coronary CT Angiogram) and optimize statin/anti-hypertensive therapy."
        else:
            tier = "Critical Risk (Urgent Cardiac Workup)"
            color = "#7F1D1D"  # Deep Crimson
            urgency = "Urgent / Immediate"
            action = "Urgent evaluation by interventional cardiology. High likelihood of hemodynamically significant coronary artery stenosis. Expedite diagnostic catheterization if symptomatic."

        return {
            'risk_probability': round(prob, 4),
            'risk_percentage': round(prob * 100, 1),
            'prediction_label': prediction,
            'risk_tier': tier,
            'color': color,
            'urgency': urgency,
            'clinical_action': action
        }

    def save(self, filepath: str, metrics_filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.pipeline, filepath)
        with open(metrics_filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
            
    @classmethod
    def load(cls, filepath: str, metrics_filepath: str) -> "HealthcareMLPipeline":
        instance = cls()
        instance.pipeline = joblib.load(filepath)
        if os.path.exists(metrics_filepath):
            with open(metrics_filepath, 'r') as f:
                instance.metrics = json.load(f)
                instance.best_model_name = instance.metrics.get('selected_model', 'trained_model')
        return instance
