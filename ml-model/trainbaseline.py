import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib

print("--- FortifAI: Training Baseline Isolation Forest ---")

num_samples = 5000
# MUST MATCH THE LIVE MONITOR EXACTLY
features = [
    "cpu_usage_percent", 
    "Flow_Pkts_s", 
    "encryption_ratio", 
    "files_modified_sec", 
    "failed_logon_ratio"
]

print("Generating clean baseline telemetry...")
X_train = pd.DataFrame({
    "cpu_usage_percent": np.random.normal(15, 5, num_samples).clip(0, 100),
    "Flow_Pkts_s": np.random.normal(100, 20, num_samples).clip(0, None),
    "encryption_ratio": np.random.uniform(0, 0.02, num_samples), 
    "files_modified_sec": np.random.poisson(2, num_samples),
    "failed_logon_ratio": np.random.uniform(0, 0.01, num_samples)
})

# Enforce strict column ordering
X_train = X_train[features]

print("Training model on baseline data...")
iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
iso_forest.fit(X_train)

joblib.dump(iso_forest, "fortifai_iso_model.pkl")
baseline_stats = X_train.agg(['mean', 'std']).to_dict()
joblib.dump(baseline_stats, "fortifai_baseline_stats.pkl")

print("✅ Saved Model and Baselines successfully.")