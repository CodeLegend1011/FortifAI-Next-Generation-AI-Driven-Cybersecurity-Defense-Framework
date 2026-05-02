import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib

# 1. Load your training data (Ensure this data is ONLY 'Normal' traffic)
# df = pd.read_csv("normal_baseline_telemetry.csv")

# Simulated normal baseline data for this example
features = ["cpu_usage_percent", "Flow_Pkts_s", "encryption_ratio"]
X_train = pd.DataFrame({
    "cpu_usage_percent": np.random.normal(15, 5, 1000), # Normal CPU around 15%
    "Flow_Pkts_s": np.random.normal(100, 20, 1000),     # Normal traffic around 100 pkts/s
    "encryption_ratio": np.random.uniform(0, 0.05, 1000) # Almost zero encryption
})

# 2. Initialize and Train the Isolation Forest
print("Training Isolation Forest on Baseline...")
iso_forest = IsolationForest(
    n_estimators=100, 
    contamination=0.01, # We expect 1% of our training data might accidentally be weird
    random_state=42
)

iso_forest.fit(X_train)

# 3. Save the model and the baseline statistics (Crucial for the next step)
joblib.dump(iso_forest, "fortifai_iso_model.pkl")

# Save the mean and standard deviation of your baseline to know *what* overshot later
baseline_stats = X_train.agg(['mean', 'std']).to_dict()
joblib.dump(baseline_stats, "fortifai_baseline_stats.pkl")

print("Model and baselines saved successfully.")