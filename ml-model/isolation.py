
import joblib
import pandas as pd

def analyze_live_event(new_data_dict):
    # Load model and baseline stats
    model = joblib.load("fortifai_iso_model.pkl")
    baselines = joblib.load("fortifai_baseline_stats.pkl")
    
    # Convert incoming data to DataFrame
    df_new = pd.DataFrame([new_data_dict])
    
    # 1. Get the Prediction (1 = Normal, -1 = Anomaly)
    prediction = model.predict(df_new)[0]
    
    if prediction == 1:
        print("✅ Status: NORMAL - Traffic matches baseline.")
        return
        
    # 2. IF ANOMALY: Calculate which features overshot
    print("🚨 ALERT: ANOMALOUS ACTIVITY DETECTED!")
    print("Analyzing feature deviations...\n")
    
    for feature in df_new.columns:
        live_value = df_new[feature].iloc[0]
        normal_mean = baselines[feature]['mean']
        normal_std = baselines[feature]['std']
        
        # Calculate Z-Score (How many standard deviations away from normal is it?)
        z_score = abs((live_value - normal_mean) / normal_std)
        
        # If it's more than 3 standard deviations away, it's a massive spike
        if z_score > 3.0:
            print(f"  -> {feature} is severely abnormal!")
            print(f"     Normal avg: {normal_mean:.2f} | Current value: {live_value:.2f}")

# ==========================================
# TEST THE ALERT SYSTEM
# ==========================================
if __name__ == "__main__":
    # Test 1: Normal Traffic
    print("--- Test 1 ---")
    analyze_live_event({
        "cpu_usage_percent": 18.0, 
        "Flow_Pkts_s": 110.0, 
        "encryption_ratio": 0.01
    })
    
    print("\n--- Test 2 ---")
    # Test 2: Ransomware Spike
    analyze_live_event({
        "cpu_usage_percent": 99.9, # Massive Overshoot
        "Flow_Pkts_s": 105.0, 
        "encryption_ratio": 0.95   # Massive Overshoot
    })