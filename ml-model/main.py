import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import joblib

# 1. Load your dataset (assuming you have a CSV)
# df = pd.read_csv('your_threat_data.csv')

# --- SIMULATED DATA FOR EXAMPLE PURPOSES ---
features = ["Flow_Duration", "Tot_Fwd_Pkts", "cpu_usage_percent", "encryption_ratio"] # ... all 98 features
X = pd.DataFrame(np.random.rand(1000, len(features)), columns=features)
# y is your target variable (e.g., 0 for Normal, 1 for Ransomware, 2 for DDoS, etc.)
y = np.random.randint(0, 3, 1000) 
# -------------------------------------------

# 2. Clean Infinite Values (Crucial for Network Flow Data)
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.median(), inplace=True) # Fill NaNs with the median of that column

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Scale the Data
# RobustScaler is immune to extreme outliers (like massive traffic spikes)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Initialize and Train the XGBoost Model
# XGBoost is highly effective for tabular security data
model = xgb.XGBClassifier(
    objective='multi:softprob', # Use 'binary:logistic' if only Normal vs Attack
    num_class=len(np.unique(y)), # Number of attack classes
    eval_metric='mlogloss',
    tree_method='hist',          # Faster for large datasets
    max_depth=6,                 # Prevents overfitting
    learning_rate=0.1
)

print("Training model...")
model.fit(X_train_scaled, y_train)

# 6. Predict and Evaluate
y_pred = model.predict(X_test_scaled)

print("\n--- Model Evaluation ---")
print(classification_report(y_test, y_pred))

# 7. Feature Importance (Understand how the model makes decisions)
importance = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("\n--- Top 5 Most Important Features ---")
print(feature_importance_df.head(5))



# Save the trained XGBoost model
model.save_model("fortifai_xdr_model.json")
print("Model saved to fortifai_xdr_model.json")

# Save the RobustScaler (CRITICAL: New data must be scaled exactly like the training data)
joblib.dump(scaler, "fortifai_scaler.pkl")
print("Scaler saved to fortifai_scaler.pkl")