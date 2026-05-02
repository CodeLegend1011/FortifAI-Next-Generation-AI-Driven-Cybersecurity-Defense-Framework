import pandas as pd
import numpy as np

# 1. Define Features (Original + 10 New Advanced Features)
features = [
    # Network 
    "Flow_Duration", "Tot_Fwd_Pkts", "Tot_Bwd_Pkts", "TotLen_Fwd_Pkts", "TotLen_Bwd_Pkts", 
    "Fwd_Pkt_Len_Max", "Fwd_Pkt_Len_Min", "Fwd_Pkt_Len_Mean", "Fwd_Pkt_Len_Std", "Bwd_Pkt_Len_Max", 
    "Bwd_Pkt_Len_Min", "Bwd_Pkt_Len_Mean", "Bwd_Pkt_Len_Std", "Flow_Byts_s", "Flow_Pkts_s", 
    "Flow_IAT_Mean", "Flow_IAT_Std", "Flow_IAT_Max", "Flow_IAT_Min", "Fwd_IAT_Mean", "Bwd_IAT_Mean", 
    "Fwd_Header_Len", "Bwd_Header_Len", "Fwd_Pkts_s", "Bwd_Pkts_s", "Pkt_Len_Min", "Pkt_Len_Max", 
    "Pkt_Len_Mean", "Pkt_Len_Std", "Pkt_Len_Var", "FIN_Flag_Cnt", "SYN_Flag_Cnt", "RST_Flag_Cnt", 
    "PSH_Flag_Cnt", "ACK_Flag_Cnt", "Down_Up_Ratio", "Pkt_Size_Avg", "Init_Fwd_Win_Byts", 
    "Init_Bwd_Win_Byts", "Active_Mean", 
    
    # Process/Host 
    "cpu_usage_percent", "memory_usage_bytes", "page_faults_sec", "handle_count", "thread_count", 
    "syscall_freq_create_process", "syscall_freq_open_process", "syscall_freq_allocate_vm", 
    "syscall_freq_write_vm", "syscall_freq_protect_vm", "syscall_freq_create_thread", 
    "syscall_freq_load_image", "syscall_freq_registry_write", "syscall_freq_network_connect", 
    "suspended_thread_ratio", "token_elevation_status", "code_injection_indicators_score", 
    "unpacked_code_entropy", "parent_child_anomaly_score", "macro_execution_indicators",
    "temp_folder_exec_rate", "system32_modification_rate", "privilege_escalation_attempts",
    
    # File System 
    "io_read_bytes_sec", "io_write_bytes_sec", "files_created_sec", "files_deleted_sec", 
    "files_modified_sec", "files_renamed_sec", "bytes_written_sec", "bytes_read_sec", 
    "write_entropy_mean", "file_extension_change_rate", "access_sensitive_dir_rate", 
    "mass_deletion_indicator", "encryption_ratio", "shadow_copy_access_rate", 
    "mft_anomaly_score", "symlink_creation_rate", "alternate_data_stream_writes", 
    "high_freq_read_write_ratio", 
    
    # User/UEBA 
    "logon_frequency_daily", "logoff_frequency_daily", "failed_logon_ratio", 
    "after_hours_activity_ratio", "weekend_activity_ratio", "remote_logon_count", 
    "resource_access_count", "unusual_machine_access_pattern", "off_baseline_app_usage", 
    "concurrent_logon_count", "geo_velocity_anomaly_score",
    
    # --- THE 10 NEW ADVANCED THREAT FEATURES ---
    "dns_query_entropy", "dns_nxdomain_rate", "tls_certificate_anomaly_score", 
    "powershell_encoded_commands_rate", "wmi_query_frequency", "registry_persistence_modifications", 
    "scheduled_task_creation_rate", "lsass_access_attempts", "smb_lateral_movement_score", 
    "kerberos_ticket_anomalies"
]

def generate_realistic_data(num_samples=10000):
    np.random.seed(42)
    
    # Build a dictionary first to avoid Pandas memory fragmentation warnings
    data_dict = {}
    
    # Define Target Labels: 0=Normal, 1=Network Attack, 2=Ransomware/Exploit, 3=Insider Threat
    labels = np.random.choice([0, 1, 2, 3], size=num_samples, p=[0.75, 0.10, 0.05, 0.10])
    data_dict['Attack_Class'] = labels

    # --- 1. GENERATE BASELINE (NORMAL) TRAFFIC ---
    data_dict['Flow_Duration'] = np.random.exponential(scale=50000, size=num_samples)
    data_dict['Tot_Fwd_Pkts'] = np.random.poisson(lam=10, size=num_samples).astype(float)
    data_dict['Fwd_Pkt_Len_Mean'] = np.random.normal(loc=500, scale=150, size=num_samples).clip(0, 1500)
    data_dict['Flow_Pkts_s'] = np.random.uniform(10, 500, size=num_samples)
    # Explicitly cast to float to prevent LossySetitemError later
    data_dict['SYN_Flag_Cnt'] = np.random.binomial(1, 0.05, size=num_samples).astype(float)
    
    data_dict['cpu_usage_percent'] = np.random.normal(loc=15, scale=5, size=num_samples).clip(0, 100)
    data_dict['syscall_freq_create_process'] = np.random.poisson(lam=2, size=num_samples).astype(float)
    data_dict['code_injection_indicators_score'] = np.random.uniform(0, 0.1, size=num_samples)
    
    data_dict['write_entropy_mean'] = np.random.normal(loc=4.5, scale=1.0, size=num_samples).clip(0, 8.0)
    data_dict['files_modified_sec'] = np.random.poisson(lam=1, size=num_samples).astype(float)
    data_dict['encryption_ratio'] = np.random.uniform(0, 0.05, size=num_samples)
    
    data_dict['failed_logon_ratio'] = np.random.uniform(0, 0.02, size=num_samples)
    data_dict['after_hours_activity_ratio'] = np.random.uniform(0, 0.1, size=num_samples)
    
    # Baseline for the 10 NEW features
    data_dict['dns_query_entropy'] = np.random.normal(loc=3.5, scale=0.5, size=num_samples).clip(0, 8)
    data_dict['dns_nxdomain_rate'] = np.random.uniform(0, 0.01, size=num_samples)
    data_dict['tls_certificate_anomaly_score'] = np.random.uniform(0, 0.1, size=num_samples)
    data_dict['powershell_encoded_commands_rate'] = np.random.uniform(0, 0.001, size=num_samples)
    data_dict['wmi_query_frequency'] = np.random.poisson(lam=0.5, size=num_samples).astype(float)
    data_dict['registry_persistence_modifications'] = np.random.poisson(lam=0, size=num_samples).astype(float)
    data_dict['scheduled_task_creation_rate'] = np.random.poisson(lam=0.01, size=num_samples).astype(float)
    data_dict['lsass_access_attempts'] = np.random.poisson(lam=0, size=num_samples).astype(float)
    data_dict['smb_lateral_movement_score'] = np.random.uniform(0, 0.05, size=num_samples)
    data_dict['kerberos_ticket_anomalies'] = np.random.uniform(0, 0.01, size=num_samples)

    # Fill remaining uninitialized features with low background noise
    for feature in features:
        if feature not in data_dict:
            data_dict[feature] = np.random.uniform(0, 1, size=num_samples)

    # Convert dictionary to DataFrame all at once
    df = pd.DataFrame(data_dict)

    # --- 2. INJECT ATTACK PATTERNS ---
    # Attack 1: Network Attack / C2 Communcation
    net_mask = df['Attack_Class'] == 1
    df.loc[net_mask, 'Flow_Pkts_s'] = np.random.uniform(50000, 200000, size=net_mask.sum())
    df.loc[net_mask, 'SYN_Flag_Cnt'] = np.random.uniform(0.9, 1.0, size=net_mask.sum())
    df.loc[net_mask, 'dns_query_entropy'] = np.random.uniform(6.5, 8.0, size=net_mask.sum())
    df.loc[net_mask, 'dns_nxdomain_rate'] = np.random.uniform(0.8, 1.0, size=net_mask.sum())
    df.loc[net_mask, 'tls_certificate_anomaly_score'] = np.random.uniform(0.7, 1.0, size=net_mask.sum())
    
    # Attack 2: Ransomware / Advanced Fileless Exploit (e.g. Mimikatz)
    ran_mask = df['Attack_Class'] == 2
    df.loc[ran_mask, 'cpu_usage_percent'] = np.random.uniform(80, 100, size=ran_mask.sum())
    df.loc[ran_mask, 'write_entropy_mean'] = np.random.uniform(7.8, 8.0, size=ran_mask.sum())
    df.loc[ran_mask, 'encryption_ratio'] = np.random.uniform(0.85, 1.0, size=ran_mask.sum())
    df.loc[ran_mask, 'lsass_access_attempts'] = np.random.uniform(5, 50, size=ran_mask.sum())
    df.loc[ran_mask, 'powershell_encoded_commands_rate'] = np.random.uniform(0.8, 1.0, size=ran_mask.sum())
    df.loc[ran_mask, 'registry_persistence_modifications'] = np.random.uniform(1, 5, size=ran_mask.sum())
    df.loc[ran_mask, 'smb_lateral_movement_score'] = np.random.uniform(0.8, 1.0, size=ran_mask.sum())

    # Attack 3: Insider Threat / Compromised Account
    insider_mask = df['Attack_Class'] == 3
    df.loc[insider_mask, 'after_hours_activity_ratio'] = np.random.uniform(0.7, 1.0, size=insider_mask.sum())
    df.loc[insider_mask, 'failed_logon_ratio'] = np.random.uniform(0.4, 0.9, size=insider_mask.sum())
    df.loc[insider_mask, 'kerberos_ticket_anomalies'] = np.random.uniform(0.7, 1.0, size=insider_mask.sum())
    df.loc[insider_mask, 'wmi_query_frequency'] = np.random.uniform(50, 200, size=insider_mask.sum())

    # Reorder columns to ensure Attack_Class is at the very end
    df = df[features + ['Attack_Class']]
    
    return df

if __name__ == "__main__":
    print("Generating advanced synthetic dataset with 10 new MITRE ATT&CK features...")
    dataset = generate_realistic_data(10000)
    
    # Save to CSV
    filename = "advanced_xdr_telemetry.csv"
    dataset.to_csv(filename, index=False)
    
    print(f"Dataset saved successfully as '{filename}'!")
    print(f"Total Features: {len(dataset.columns) - 1}") # Subtract 1 for the target label