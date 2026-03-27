"""
FortifAI Shared Constants
Used by both client and admin/server components.
"""

# ── Federated Learning ─────────────────────────────────────────────────────────
CLIP_BOUND: float = 5.0           # Gradient L2 norm clipping bound
DP_NOISE_SCALE: float = 0.5       # Differential-privacy noise multiplier (σ)
EPSILON_CLIENT: float = 0.1       # Per-client privacy budget (ε) per round
DELTA: float = 1e-5               # Failure probability (δ)
TRIM_FRACTION: float = 0.1        # Trimmed-mean fraction (top/bottom 10 %)
CONVERGENCE_THRESHOLD: float = 0.01  # L2 distance threshold for convergence

# Server-side additional amplification
SERVER_EPSILON: float = 0.05

# ── Anomaly Detection ──────────────────────────────────────────────────────────
# Canonical values — detailed tuning lives in shared/detection_config.py
from shared.detection_config import (
    ENSEMBLE_WEIGHTS, SEVERITY_THRESHOLDS, ZSCORE_THRESHOLD,
)
WEIGHT_ISO   = ENSEMBLE_WEIGHTS["isolation_forest"]
WEIGHT_AE    = ENSEMBLE_WEIGHTS["autoencoder"]
WEIGHT_ZSCORE = ENSEMBLE_WEIGHTS["zscore"]
WEIGHT_SVM   = ENSEMBLE_WEIGHTS["ocsvm"]

SEVERITY_CRITICAL = SEVERITY_THRESHOLDS["critical"]
SEVERITY_HIGH     = SEVERITY_THRESHOLDS["high"]
SEVERITY_MEDIUM   = SEVERITY_THRESHOLDS["medium"]

# ── Feature Engineering ────────────────────────────────────────────────────────
TOTAL_FEATURES: int = 99
FEATURE_COUNTS = {
    "network":    40,
    "process":    22,
    "filesystem": 19,
    "user":       18,
}

# ── Collection / Timing ────────────────────────────────────────────────────────
COLLECTION_INTERVAL_SEC: int = 30
HEARTBEAT_INTERVAL_SEC: int = 60
FL_UPDATE_INTERVAL_SEC: int = 90
ML_TRAINING_INTERVAL_SEC: int = 600   # 10 minutes
ML_BOOTSTRAP_DELAY_SEC: int = 180     # 3 minutes after start
MIN_SAMPLES_FOR_TRAINING: int = 20