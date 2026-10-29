"""
Automated Pipeline Training and Clinical Validation Runner
Loads data, executes 5-fold cross validation across model architectures,
selects the top clinical performer, and exports artifacts.
"""

import os
import sys
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.pipeline import HealthcareMLPipeline

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'heart_disease_clinical.csv')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    pipeline_path = os.path.join(models_dir, 'heart_disease_pipeline.joblib')
    metrics_path = os.path.join(models_dir, 'evaluation_metrics.json')
    
    print("=" * 60)
    print("HEALTHCARE ML PIPELINE: CARDIAC RISK STRATIFICATION")
    print("=" * 60)
    
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Generating now...")
        from data.generate_dataset import generate_cardiovascular_dataset
        df = generate_cardiovascular_dataset(n_samples=1500, random_state=42)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)
        
    print(f"[1/4] Clinical Cohort Loaded: {len(df)} patients, {df.shape[1]-1} diagnostic features.")
    print(f"      Positive Disease Prevalence: {df['target'].mean() * 100:.1f}%\n")
    
    print("[2/4] Executing Stratified 5-Fold Cross-Validation on Candidate Models...")
    pipeline = HealthcareMLPipeline()
    metrics = pipeline.train_and_benchmark(df, target_col='target')
    
    print(f"\n[3/4] Model Benchmarking Results:")
    for model_name, res in metrics['models_benchmark'].items():
        tm = res['test_metrics']
        print(f"  * {model_name.upper():<22} | "
              f"ROC-AUC: {tm['roc_auc']:.4f} | "
              f"Sensitivity: {tm['sensitivity_recall']:.4f} | "
              f"Specificity: {tm['specificity']:.4f} | "
              f"F1: {tm['f1_score']:.4f}")
        
    best_model = metrics['selected_model']
    print(f"\n==> Selected Optimal Model for Clinical Deployment: {best_model.upper()}")
    
    print(f"\n[4/4] Serializing Clinical Model Artifacts to {models_dir}...")
    pipeline.save(pipeline_path, metrics_path)
    print(f"  [OK] Saved pipeline: {pipeline_path}")
    print(f"  [OK] Saved evaluation metrics: {metrics_path}")
    print("\nTraining workflow completed successfully!")

if __name__ == '__main__':
    main()
