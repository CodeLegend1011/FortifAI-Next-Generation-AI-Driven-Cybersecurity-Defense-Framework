"""
FortifAI — Centralized Detection & ML Configuration

ALL detection-related constants, thresholds, patterns, and magic numbers
are defined here so they can be tuned from a single location.
"""

# =============================================================================
#  1.  ENSEMBLE SCORING & SEVERITY
# =============================================================================

ENSEMBLE_WEIGHTS = {
    "zscore":           0.20,
    "isolation_forest": 0.40,
    "autoencoder":      0.35,
    "ocsvm":            0.05,
}

SEVERITY_THRESHOLDS = {
    "critical": 8.0,
    "high":     6.5,
    "medium":   5.0,
    "low":      3.5,
}

PATTERN_OVERRIDE_SCORES = {
    "ransomware_burst": 9.5,
    "c2_pattern":       8.0,
    "cred_dump_pattern": 7.5,
}

CONSENSUS_STRONG_MODEL_THRESHOLD = 5.0
CONSENSUS_BONUS_PER_EXTRA_MODEL = 0.15

# =============================================================================
#  2.  Z-SCORE DETECTION
# =============================================================================

ZSCORE_THRESHOLD = 3.5
ZSCORE_MIN_FLAGS = 2
ZSCORE_PER_FEATURE_CAP = 2.0
ZSCORE_MAX_SCORE = 10.0
ZSCORE_MIN_RUNNING_COUNT = 30

# =============================================================================
#  3.  ISOLATION FOREST
# =============================================================================

IF_CONTAMINATION = 0.05
IF_N_ESTIMATORS = 200
IF_MAX_SAMPLES = 256
IF_RANDOM_STATE = 42
IF_THRESHOLD_INIT = -0.3
IF_THRESHOLD_PERCENTILE = 10
IF_SCORE_SCALE = 12
IF_SCORE_CAP = 10.0
IF_REPORT_THRESHOLD = 4.0

# =============================================================================
#  4.  AUTOENCODER
# =============================================================================

AE_INPUT_DIM = 99
AE_LATENT_DIM = 32
AE_LAYER_SIZES = [128, 64]
AE_DROPOUT = 0.2
AE_EPOCHS = 10
AE_BATCH_SIZE = 16
AE_PATIENCE = 3
AE_VALIDATION_SPLIT = 0.2
AE_THRESHOLD_PERCENTILE = 90
AE_EXCESS_MULTIPLIER = 1.2
AE_SCORE_SCALE = 6
AE_SCORE_CAP = 10.0
AE_REPORT_THRESHOLD = 6.0

# =============================================================================
#  5.  ONE-CLASS SVM
# =============================================================================

OCSVM_NU = 0.05
OCSVM_GAMMA = "scale"
OCSVM_KERNEL = "rbf"
OCSVM_SCORE_SCALE = 8
OCSVM_SCORE_CAP = 10.0

# =============================================================================
#  6.  TRAINING & RETRAINING POLICY
# =============================================================================

MIN_TRAIN_SAMPLES = 20
IQR_OUTLIER_MULTIPLIER = 3
MIN_CLEANED_SAMPLES = 15
RETRAIN_MIN_BUFFER = 150
RETRAIN_INTERVAL_SEC = 900
RETRAIN_PERIODIC_SEC = 3600
BOOTSTRAP_MIN_SAMPLES = 10
BOOTSTRAP_PAD_NOISE_STD = 0.05

# =============================================================================
#  7.  FEATURE ENGINEERING
# =============================================================================

FEATURE_WINDOW_SEC = 60
FEATURE_BUFFER_MAX_WINDOWS = 1000
TOTAL_FEATURES = 99

EXECUTABLE_EXTENSIONS = frozenset({".exe", ".dll", ".so", ".bat", ".ps1", ".vbs", ".cmd", ".msi"})

FEATURE_CLIP_BOUNDS = {
    "conn_count":       (0, 100),
    "bytes_sent":       (0, 1e9),
    "bytes_recv":       (0, 1e9),
    "port_entropy":     (0, 8),
    "conn_rate":        (0, 10),
    "dst_churn_rate":   (0, 5),
    "proc_spawn_count": (0, 50),
    "avg_proc_cpu":     (0, 1),
    "avg_proc_memory":  (0, 1),
    "unique_proc_names": (0, 100),
    "file_create_count": (0, 100),
    "file_exec_count":  (0, 20),
    "file_hash_novelty": (0, 50),
    "delta_conn_count": (-10, 10),
    "delta_file_create": (-50, 50),
    "delta_proc_spawn": (-10, 10),
    "delta_unique_hashes": (-20, 20),
    "conn_acceleration": (-5, 5),
}

MAD_SCALE = 1.4826
EARLY_CLIP_SIGMA = 3.0

# =============================================================================
#  8.  THREAT DETECTION PATTERNS
# =============================================================================

CREDENTIAL_PROCESS_SUBSTRINGS = ("lsass", "sam", "mimikatz", "sekurlsa", "procdump")

SUSPICIOUS_PROCESS_NAMES = frozenset({
    "mimikatz", "psexec", "cobalt", "meterpreter", "nc.exe",
    "ncat", "netcat", "certutil", "bitsadmin", "regsvr32",
    "mshta", "wmic", "cscript", "wscript", "rundll32",
})

SCRIPT_PARENT_SUBSTRINGS = ("powershell", "cmd", "wscript", "cscript", "mshta", "python", "bash")

TRUSTED_PATH_SUBSTRINGS = (
    "\\windows\\system32", "\\windows\\syswow64",
    "\\program files", "\\program files (x86)",
    "/usr/bin", "/usr/sbin", "/usr/local/bin",
)

SUSPICIOUS_CMDLINE_PATTERNS = (
    "-encodedcommand", "-enc ", "invoke-expression",
    "downloadstring", "invoke-webrequest",
    "bypass", "-noprofile", "-executionpolicy",
    "certutil -urlcache", "bitsadmin /transfer",
    "reg add.*\\run", "schtasks /create",
)

SUSPICIOUS_PARENT_CHILD_PAIRS = {
    "powershell.exe": {"mshta.exe", "certutil.exe", "bitsadmin.exe"},
    "cmd.exe": {"powershell.exe", "certutil.exe", "bitsadmin.exe"},
    "explorer.exe": {"cmd.exe", "powershell.exe"},
    "winword.exe": {"cmd.exe", "powershell.exe", "mshta.exe"},
    "excel.exe": {"cmd.exe", "powershell.exe"},
    "outlook.exe": {"cmd.exe", "powershell.exe"},
}

# =============================================================================
#  9.  FILE SYSTEM DETECTION
# =============================================================================

HIGH_RISK_EXTENSIONS = frozenset({
    ".exe", ".dll", ".bat", ".cmd", ".vbs", ".js", ".wsf",
    ".ps1", ".psm1", ".psd1", ".msi", ".msp", ".scr",
    ".hta", ".cpl", ".inf", ".reg", ".lnk", ".jar",
    ".py", ".sh", ".com", ".pif", ".application",
})

RANSOMWARE_EXTENSIONS = frozenset({
    ".encrypted", ".locked", ".crypto", ".crypt",
    ".enc", ".rnsmwr", ".locky", ".zepto", ".cerber",
    ".wcry", ".wncry", ".wncryt", ".onion",
    ".aaa", ".abc", ".xyz", ".zzz",
})

DOUBLE_EXTENSION_TRICKS = (
    ".pdf.exe", ".doc.exe", ".jpg.exe", ".txt.exe",
    ".png.scr", ".xls.cmd", ".mp3.vbs",
)

SUSPICIOUS_FILENAME_KEYWORDS = (
    "ransom", "decrypt", "readme", "how_to_recover",
    "your_files", "pay", "bitcoin", "restore",
    "encrypted", "locked",
)

FS_MAX_HASH_BYTES = 50 * 1024 * 1024

# Ransomware burst heuristics
RANSOM_BURST_FILE_COUNT = 50
RANSOM_BURST_HASH_COUNT = 20
RANSOM_BURST_WINDOW_SEC = 30

# =============================================================================
# 10.  NETWORK DETECTION
# =============================================================================

MALICIOUS_PORTS = frozenset({
    135, 137, 138, 139, 445, 1433, 1434,
    3389, 4444, 5555, 5900, 8080, 8443,
    6660, 6661, 6662, 6663, 6664, 6665,
    6666, 6667, 6668, 6669, 6670, 6697,
    9050, 9051,
})

SUSPICIOUS_PORT_RANGES = (
    (49152, 65535),
)

PRIVATE_IP_PREFIXES = ("10.", "172.16.", "172.17.", "172.18.", "172.19.",
                        "172.20.", "172.21.", "172.22.", "172.23.", "172.24.",
                        "172.25.", "172.26.", "172.27.", "172.28.", "172.29.",
                        "172.30.", "172.31.", "192.168.", "127.")

LOOPBACK_ADDRESSES = frozenset({"127.0.0.1", "::1", "localhost", "0.0.0.0"})

C2_MIN_CONN_COUNT = 5
C2_MAX_UNIQUE_DST = 3
C2_MIN_PORT_COUNT = 10

# =============================================================================
# 11.  ALERT & DEDUP
# =============================================================================

ALERT_TTL_BY_SEVERITY = {
    "critical": 600,
    "high":     300,
    "medium":   180,
    "low":      60,
}

SERVER_ALERT_TTL_BY_SEVERITY = {
    "critical": 600,
    "high":     300,
    "medium":   180,
    "low":      60,
}

SERVER_ALERT_MIN_RISK = 4.0
TELEMETRY_ALERT_HISTORY_MAX = 50
ANOMALY_ALERTS_RETAIN_COUNT = 10

# =============================================================================
# 12.  PERFORMANCE GUARDS & RATE LIMITS
# =============================================================================

ML_OPS_PER_MINUTE = 30
ALERTS_PER_MINUTE = 50
MIN_TRAIN_INTERVAL_SEC = 900
RATE_LIMIT_WINDOW_SEC = 60
TRAINING_LOCK_TIMEOUT_SEC = 5
NORM_CACHE_TTL_SEC = 60
INFERENCE_CACHE_MAX = 1000
INFERENCE_CACHE_EVICT = 200
INFERENCE_HASH_PRECISION = 3
BASELINE_DEQUE_MAXLEN = 200
LOCAL_ALERT_QUEUE_MAX = 100
BASELINE_LEARNING_MIN_SAMPLES = 100

# =============================================================================
# 13.  FEDERATED LEARNING (AGGREGATION)
# =============================================================================

FL_CLIENT_CLIP_BOUND = 10.0
FL_SERVER_CLIP_BOUND = 10.0
FL_DP_NOISE_SCALE = 0.5
FL_TRIM_FRAC = 0.1
FL_LEARNING_RATE = 0.5
FL_MOMENTUM = 0.9
FL_MU = 0.01
FL_VALIDATION_AUC_DROP = 0.02
FL_MIN_VALIDATION_SAMPLES = 20
FL_MIN_CLIENT_SAMPLES = 10
FL_HIGH_ANOMALY_RATE = 0.5
FL_MAX_ABS_WEIGHT = 1000.0
FL_QUARANTINE_CLIP_FACTOR = 1.5
FL_EXTREME_NORM_FACTOR = 3.0
FL_SCALAR_EMA_OLD_WEIGHT = 0.7
FL_TENSOR_DP_SIGMA = 0.001
FL_ISO_SAMPLE_LAST_N = 50

SCALAR_CLIP_BOUNDS = {
    "network_threshold":  (1.0, 5.0),
    "process_threshold":  (1.0, 5.0),
    "file_threshold":     (1.0, 5.0),
    "iso_threshold":      (-1.0, 0.0),
    "ae_threshold":       (0.0001, 1.0),
    "network_sensitivity": (0.3, 2.0),
    "process_sensitivity": (0.3, 2.0),
    "file_sensitivity":    (0.3, 2.0),
    "anomaly_alpha":       (0.0, 1.0),
    "anomaly_beta":        (0.0, 1.0),
}

# =============================================================================
# 14.  REINFORCEMENT LEARNING
# =============================================================================

RL_ACTIONS = ["ignore", "block_ip", "kill_process", "isolate_network", "quarantine_file"]
RL_LEARNING_RATE = 0.1
RL_GAMMA = 0.9
RL_EPSILON = 0.1
RL_FEEDBACK_REWARD_MAGNITUDE = 5.0
RL_SCORE_HIGH_THRESHOLD = 8.0
RL_SCORE_MED_THRESHOLD = 5.0
RL_Q_INIT_RANGE = 0.01

# =============================================================================
# 15.  USER BEHAVIOR
# =============================================================================

AFTER_HOURS_RANGE = (22, 6)
BRUTE_FORCE_LOGIN_COUNT = 5
BRUTE_FORCE_WINDOW_MINUTES = 10
CONCURRENT_SESSION_THRESHOLD = 3

# =============================================================================
# 16.  PROCESS RISK THRESHOLDS
# =============================================================================

HIGH_CPU_PERCENT = 50.0
HIGH_MEMORY_MB = 500.0
MALICIOUS_LINEAGE_RISK_BUMP = 7.0
PROCESS_BASELINE_WINDOW = 20

# =============================================================================
# 17.  THREAT INTELLIGENCE
# =============================================================================

THREATFOX_RECENT_URL = "https://threatfox-api.abuse.ch/api/v1/"
THREAT_INTEL_USER_AGENT = "FortifAI/2.0"
THREATFOX_TIMEOUT_SEC = 15
THREATFOX_MAX_ITEMS = 200

# =============================================================================
# 18.  DATABASE / ANALYTICS
# =============================================================================

DB_STATEMENT_TIMEOUT = "30s"
ANALYTICS_LOOKBACK_HOURS = 24
SERVER_LISTEN_BACKLOG = 10
SERVER_ACCEPT_TIMEOUT_SEC = 1.0
