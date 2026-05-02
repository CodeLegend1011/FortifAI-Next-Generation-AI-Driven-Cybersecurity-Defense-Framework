import time
import psutil
import pandas as pd
import keyboard
import datetime
import os

print("Initializing FortifAI Telemetry Collector...")

# 1. The Full 108 Feature List (from your ML Model)
all_features = [
    "Flow_Duration", "Tot_Fwd_Pkts", "Tot_Bwd_Pkts", "TotLen_Fwd_Pkts", "TotLen_Bwd_Pkts", 
    "Fwd_Pkt_Len_Max", "Fwd_Pkt_Len_Min", "Fwd_Pkt_Len_Mean", "Fwd_Pkt_Len_Std", "Bwd_Pkt_Len_Max", 
    "Bwd_Pkt_Len_Min", "Bwd_Pkt_Len_Mean", "Bwd_Pkt_Len_Std", "Flow_Byts_s", "Flow_Pkts_s", 
    "Flow_IAT_Mean", "Flow_IAT_Std", "Flow_IAT_Max", "Flow_IAT_Min", "Fwd_IAT_Mean", "Bwd_IAT_Mean", 
    "Fwd_Header_Len", "Bwd_Header_Len", "Fwd_Pkts_s", "Bwd_Pkts_s", "Pkt_Len_Min", "Pkt_Len_Max", 
    "Pkt_Len_Mean", "Pkt_Len_Std", "Pkt_Len_Var", "FIN_Flag_Cnt", "SYN_Flag_Cnt", "RST_Flag_Cnt", 
    "PSH_Flag_Cnt", "ACK_Flag_Cnt", "Down_Up_Ratio", "Pkt_Size_Avg", "Init_Fwd_Win_Byts", 
    "Init_Bwd_Win_Byts", "Active_Mean", "cpu_usage_percent", "memory_usage_bytes", "page_faults_sec", 
    "handle_count", "thread_count", "io_read_bytes_sec", "io_write_bytes_sec", "syscall_freq_create_process", 
    "syscall_freq_open_process", "syscall_freq_allocate_vm", "syscall_freq_write_vm", "syscall_freq_protect_vm", 
    "syscall_freq_create_thread", "syscall_freq_load_image", "syscall_freq_registry_write", 
    "syscall_freq_network_connect", "suspended_thread_ratio", "token_elevation_status", 
    "code_injection_indicators_score", "unpacked_code_entropy", "parent_child_anomaly_score", 
    "files_created_sec", "files_deleted_sec", "files_modified_sec", "files_renamed_sec", "bytes_written_sec", 
    "bytes_read_sec", "write_entropy_mean", "file_extension_change_rate", "access_sensitive_dir_rate", 
    "mass_deletion_indicator", "encryption_ratio", "shadow_copy_access_rate", "mft_anomaly_score", 
    "symlink_creation_rate", "alternate_data_stream_writes", "macro_execution_indicators", 
    "high_freq_read_write_ratio", "temp_folder_exec_rate", "system32_modification_rate", "logon_frequency_daily", 
    "logoff_frequency_daily", "failed_logon_ratio", "after_hours_activity_ratio", "weekend_activity_ratio", 
    "remote_logon_count", "resource_access_count", "file_copy_to_usb_bytes", "email_sent_count", 
    "email_attachment_size", "email_external_recipient_ratio", "web_uncategorized_visits", "web_download_bytes", 
    "privilege_escalation_attempts", "unusual_machine_access_pattern", "off_baseline_app_usage", 
    "sentiment_negative_score", "concurrent_logon_count", "geo_velocity_anomaly_score",
    "dns_query_entropy", "dns_nxdomain_rate", "tls_certificate_anomaly_score", "powershell_encoded_commands_rate", 
    "wmi_query_frequency", "registry_persistence_modifications", "scheduled_task_creation_rate", 
    "lsass_access_attempts", "smb_lateral_movement_score", "kerberos_ticket_anomalies"
]

# Global variables to control the recording state
is_recording = False
recorded_data = []

def get_full_telemetry():
    """Captures live system data and formats it into the 108-feature structure."""
    # Initialize all 108 features to 0.0 to ensure the Excel sheet has every column
    telemetry = {feature: 0.0 for feature in all_features}
    
    # Take initial snapshot
    net_start = psutil.net_io_counters()
    disk_start = psutil.disk_io_counters()
    
    # Measure CPU (This inherently pauses the loop for 1 second)
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # Take final snapshot
    net_end = psutil.net_io_counters()
    disk_end = psutil.disk_io_counters()
    
    # Calculate live metrics
    pkts_sec = (net_end.packets_sent + net_end.packets_recv) - (net_start.packets_sent + net_start.packets_recv)
    bytes_sec = (net_end.bytes_sent + net_end.bytes_recv) - (net_start.bytes_sent + net_start.bytes_recv)
    writes_sec = disk_end.write_count - disk_start.write_count
    
    # Populate the real data we can observe via Python
    telemetry['cpu_usage_percent'] = cpu_percent
    telemetry['memory_usage_bytes'] = psutil.virtual_memory().used
    telemetry['Flow_Pkts_s'] = float(pkts_sec)
    telemetry['Flow_Byts_s'] = float(bytes_sec)
    telemetry['files_modified_sec'] = float(writes_sec)
    
    # Approximate process and thread counts
    telemetry['syscall_freq_create_process'] = float(len(psutil.pids()))
    
    return telemetry

def start_recording(e):
    global is_recording, recorded_data
    if not is_recording:
        is_recording = True
        recorded_data = []
        print("\n🔴 [REC] Recording Started! Gathering telemetry every second...")

def stop_recording(e):
    global is_recording, recorded_data
    if is_recording:
        is_recording = False
        print("\n⏹️ [STOP] Recording Stopped. Generating Excel file, please wait...")
        
        # Convert recorded list to Pandas DataFrame
        df = pd.DataFrame(recorded_data)
        
        # Ensure columns are in the exact order
        df = df[all_features] 
        
        # Save to Excel with a timestamped filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"fortifai_baseline_{timestamp}.xlsx"
        df.to_excel(filename, index=False)
        
        print(f"✅ SUCCESS: Saved {len(recorded_data)} seconds of telemetry to {filename}")
        print("\nPress '1' to start a new recording, or 'Ctrl+C' to exit.")

# Bind the keys to the functions
keyboard.on_press_key('1', start_recording)
keyboard.on_press_key('2', stop_recording)

print("\n" + "="*50)
print("🛡️ FortifAI Baseline Data Collector Ready")
print("="*50)
print("  > Press '1' to START recording.")
print("  > Press '2' to STOP and SAVE to Excel.")
print("  > Press 'Ctrl+C' in terminal to QUIT.")
print("="*50 + "\n")

# Main infinite loop
try:
    while True:
        if is_recording:
            # Capture the data
            frame_data = get_full_telemetry()
            recorded_data.append(frame_data)
            
            # Print a live counter to the terminal so you know it's working
            print(f"  -> Recorded Frame {len(recorded_data)} | CPU: {frame_data['cpu_usage_percent']}% | Pkts/s: {frame_data['Flow_Pkts_s']}", end='\r')
        else:
            # Sleep briefly to prevent high CPU usage while waiting for keypress
            time.sleep(0.1)
            
except KeyboardInterrupt:
    print("\n\n🛑 FortifAI Collector Offline. Shutting down gracefully.")