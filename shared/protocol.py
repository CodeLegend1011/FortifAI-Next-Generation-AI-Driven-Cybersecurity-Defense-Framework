"""
FortifAI Shared Protocol
Defines message schemas / TypedDicts so that client and server
always agree on field names and structure.
"""

from typing import Any, Dict, List, Optional, TypedDict


# ── Client → Server message types ─────────────────────────────────────────────

class RegistrationMessage(TypedDict):
    type: str                   # "registration"
    client_info: Dict[str, Any]
    capabilities: Dict[str, bool]


class TelemetryMessage(TypedDict):
    type: str                   # "telemetry"
    client_id: str
    timestamp: str
    network: Dict[str, Any]
    processes: Dict[str, Any]
    filesystem: Dict[str, Any]
    user_activity: List[Dict[str, Any]]
    anomaly_alerts: List[Dict[str, Any]]
    system_info: Dict[str, Any]


class FLUpdateMessage(TypedDict):
    type: str                   # "fl_update"
    client_id: str
    model_parameters: Dict[str, Any]


class HeartbeatMessage(TypedDict):
    type: str                   # "heartbeat"
    client_id: str
    timestamp: str


class CacheModelsMessage(TypedDict):
    type: str                   # "cache_models"
    client_id: str
    cached_models: Dict[str, Any]


# ── Server → Client response schemas ──────────────────────────────────────────

class RegistrationResponse(TypedDict):
    status: str                 # "registered"
    message: str
    model_weights: Optional[Dict[str, Any]]
    cached_models: Optional[Dict[str, Any]]


class TelemetryResponse(TypedDict):
    status: str                 # "received"
    message: str
    alerts: Optional[List[str]]


class FLResponse(TypedDict):
    status: str                 # "fl_received"
    message: str
    aggregated_weights: Optional[Dict[str, Any]]


# ── FL model update schema (inside FLUpdateMessage.model_parameters) ───────────

class ModelWeights(TypedDict):
    network_threshold: float
    process_threshold: float
    file_threshold: float
    network_sensitivity: float
    process_sensitivity: float
    file_sensitivity: float
    anomaly_alpha: float
    anomaly_beta: float
    network_baseline_mean: float
    network_baseline_std: float
    process_baseline_mean: float
    process_baseline_std: float
    file_baseline_mean: float
    file_baseline_std: float


class DataQuality(TypedDict):
    network_samples: int
    process_samples: int
    file_samples: int


class Statistics(TypedDict):
    network: Dict[str, float]
    process: Dict[str, float]
    file: Dict[str, float]


class ModelParameters(TypedDict):
    weights: ModelWeights
    data_quality: DataQuality
    statistics: Statistics
    anomaly_rate: float


# ── Privacy-safe telemetry (Phase 3 target format) ─────────────────────────────

class AnomalyReport(TypedDict):
    type: str                   # "network" | "process" | "filesystem" | "user"
    severity: str               # "low" | "medium" | "high" | "critical"
    feature_signature: str      # hash of features – NOT raw features
    model_votes: Dict[str, float]


class AggregatedStats(TypedDict):
    total_connections: int
    total_processes: int
    anomaly_count: int


class PrivacySafeTelemetry(TypedDict):
    client_id: str
    timestamp: str
    anomalies_detected: List[AnomalyReport]
    aggregated_stats: AggregatedStats
    model_update: Dict[str, Any]    # weight_deltas + update_norm only


# ── Shared message-type constants ──────────────────────────────────────────────
MSG_REGISTRATION = "registration"
MSG_TELEMETRY    = "telemetry"
MSG_FL_UPDATE    = "fl_update"
MSG_HEARTBEAT    = "heartbeat"
MSG_CACHE_MODELS = "cache_models"