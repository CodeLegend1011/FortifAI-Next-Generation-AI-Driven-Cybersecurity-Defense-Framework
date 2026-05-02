import time
import psutil
import joblib
import pandas as pd
import os
import datetime

print("Initializing FortifAI Live EDR Agent...")

# 1. Load the Brain and Baselines
try:
    model = joblib.load("fortifai_iso_model.pkl")
    baselines = joblib.load("fortifai_baseline_stats.pkl")
    print("✅ Intelligence Loaded Successfully.\n")
except FileNotFoundError:
    print("❌ Error: Missing .pkl files. Run train_baseline.py first!")
    exit()

def get_live_telemetry(interval=1):
    """Captures live system metrics over a specific time interval (in seconds)."""
    
    # Take initial snapshot
    net_start = psutil.net_io_counters()
    disk_start = psutil.disk_io_counters()
    
    # Wait and measure CPU over the interval
    cpu_usage = psutil.cpu_percent(interval=interval)
    
    # Take final snapshot
    net_end = psutil.net_io_counters()
    disk_end = psutil.disk_io_counters()
    
    # Calculate Per-Second Deltas
    pkts_sec = (net_end.packets_sent + net_end.packets_recv) - (net_start.packets_sent + net_start.packets_recv)
    writes_sec = disk_end.write_count - disk_start.write_count

    # Build the dictionary exactly as the ML model expects
    live_data = {
        "cpu_usage_percent": cpu_usage,
        "Flow_Pkts_s": pkts_sec,
        "encryption_ratio": 0.0,      
        "files_modified_sec": writes_sec,
        "failed_logon_ratio": 0.0     
    }
    
    return live_data

# 2. The Infinite Monitoring Loop
print("🛡️ FortifAI Active Protection Started. Press Ctrl+C to stop.\n")
print("-" * 60)

try:
    while True:
        # Capture 1 second of live data
        current_telemetry = get_live_telemetry(interval=1)
        
        # Convert to DataFrame
        df_live = pd.DataFrame([current_telemetry])
        
        # Make Prediction
        prediction = model.predict(df_live)[0]
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        if prediction == 1:
            # Output Normal Status
            print(f"[{timestamp}] ✅ NORMAL | CPU: {current_telemetry['cpu_usage_percent']:04.1f}% | Net: {current_telemetry['Flow_Pkts_s']:04} pkts/s | Disk IO: {current_telemetry['files_modified_sec']:03} ops/s", end='\r')
        else:
            # Output Alert and calculate Z-Scores
            print(f"\n\n[{timestamp}] 🚨 ANOMALY DETECTED! INITIATING TRIAGE...")
            
            # --- NEW ADDITION: Print the raw input data ---
            print("\n[Raw Telemetry Input]")
            for key, value in current_telemetry.items():
                print(f"  > {key}: {value:.2f}")
            print("\n[Triage Analysis]")
            # ----------------------------------------------
            
            for feature in df_live.columns:
                live_val = df_live[feature].iloc[0]
                mean_val = baselines[feature]['mean']
                std_val = baselines[feature]['std']
                
                # Protect against divide-by-zero if std is exactly 0
                if std_val == 0: std_val = 0.0001 
                
                z_score = abs((live_val - mean_val) / std_val)
                
                # Print only the features that are violently out of bounds
                if z_score > 3.0:
                    print(f"  ⚠️ {feature} Spike: {live_val:.1f} (Normal is ~{mean_val:.1f})")
            
            print("-" * 60)
            
except KeyboardInterrupt:
    print("\n\n🛑 FortifAI Protection Offline. Shutting down gracefully.")