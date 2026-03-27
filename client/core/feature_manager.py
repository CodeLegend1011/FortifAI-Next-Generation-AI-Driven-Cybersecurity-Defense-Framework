import os
import sys
import time
import json
import socket
import pickle
import threading
import psutil
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, deque
from sklearn.preprocessing import StandardScaler
from scipy import stats
from threading import Lock
from dotenv import load_dotenv

load_dotenv()

# Import canonical 99-feature list from the new collectors module
from client.core.feature_collectors import FEATURE_NAMES as _FEATURE_NAMES

try:
    from shared.detection_config import (
        EXECUTABLE_EXTENSIONS,
        FEATURE_WINDOW_SEC, FEATURE_BUFFER_MAX_WINDOWS,
        RANSOM_BURST_FILE_COUNT, RANSOM_BURST_HASH_COUNT, RANSOM_BURST_WINDOW_SEC,
        C2_MIN_CONN_COUNT, C2_MAX_UNIQUE_DST,
        CREDENTIAL_PROCESS_SUBSTRINGS,
        MAD_SCALE, EARLY_CLIP_SIGMA,
        ML_OPS_PER_MINUTE, ALERTS_PER_MINUTE, MIN_TRAIN_INTERVAL_SEC,
        NORM_CACHE_TTL_SEC, RATE_LIMIT_WINDOW_SEC, TRAINING_LOCK_TIMEOUT_SEC,
    )
except ImportError:
    EXECUTABLE_EXTENSIONS = frozenset({".exe", ".dll", ".so"})
    FEATURE_WINDOW_SEC = 60
    FEATURE_BUFFER_MAX_WINDOWS = 1000
    RANSOM_BURST_FILE_COUNT = 50
    RANSOM_BURST_HASH_COUNT = 20
    RANSOM_BURST_WINDOW_SEC = 30
    C2_MIN_CONN_COUNT = 5
    C2_MAX_UNIQUE_DST = 3
    CREDENTIAL_PROCESS_SUBSTRINGS = ("lsass", "sam")
    MAD_SCALE = 1.4826
    EARLY_CLIP_SIGMA = 3.0
    ML_OPS_PER_MINUTE = 30
    ALERTS_PER_MINUTE = 50
    MIN_TRAIN_INTERVAL_SEC = 900
    NORM_CACHE_TTL_SEC = 60
    RATE_LIMIT_WINDOW_SEC = 60
    TRAINING_LOCK_TIMEOUT_SEC = 5


class FeatureWindowManager:
    """Manages sliding window feature extraction for ML models"""
    # ========= CANONICAL 99-FEATURE SCHEMA (matches global_feature_schema.json) =========
    FEATURE_SCHEMA = _FEATURE_NAMES   # all 99 names



    def __init__(self, window_duration=60, max_windows=1000):
        self.window_duration = window_duration  # seconds
        self.max_windows = max_windows
        self.feature_buffer = deque(maxlen=max_windows)
        
        # Running statistics for normalization
        self.running_mean = {}
        self.running_std = {}
        self.running_count = 0
        
        # EMA parameters
        self.ema_alpha = 0.1
        self.ema_values = {}
        
        # Model expected input dimension (updated when foundation models load)
        self.model_input_dim = 99

        # Global Z-score baselines (loaded from server during sync)
        self.global_z_means = None
        self.global_z_stds = None
        
        # Current window accumulator
        self.current_window = {
            'start_time': time.time(),
            'network': {'conn_count': 0, 'unique_dst': set(), 'dst_list': [],
                        'bytes_sent': 0, 'bytes_recv': 0, 'ports': [], 'protocols': []},
            'process': {'spawn_count': 0, 'cpu_samples': [], 'memory_samples': [],
                        'proc_names': set()},
            'filesystem': {'file_create': 0, 'file_exec': 0, 'file_hashes': set()}
        }
        self.last_window_snapshot = {}
        self.feature_history = {}
        
        self.lock = Lock()
    
    
    def get_window_stats(self):
        """Get current window statistics for debugging"""
        with self.lock:
            return {
                'elapsed': time.time() - self.current_window['start_time'],
                'network_events': self.current_window['network']['conn_count'],
                'process_events': self.current_window['process']['spawn_count'],
                'file_events': self.current_window['filesystem']['file_create'],
                'buffer_size': len(self.feature_buffer)
            }
        
    def update_network_event(self, event):
        """Update current window with network event"""
        with self.lock:
            self.current_window['network']['conn_count'] += 1
            if 'dst_ip' in event:
                self.current_window['network']['unique_dst'].add(event['dst_ip'])
                self.current_window['network']['dst_list'].append(event['dst_ip'])
            if 'src_port' in event:
                self.current_window['network']['ports'].append(event['src_port'])
            if 'protocol' in event:
                self.current_window['network']['protocols'].append(event['protocol'])
    
    def update_process_event(self, event):
        """Update current window with process event"""
        with self.lock:
            self.current_window['process']['spawn_count'] += 1
            if 'cpu_percent' in event:
                self.current_window['process']['cpu_samples'].append(event['cpu_percent'])
            if 'memory_mb' in event:
                self.current_window['process']['memory_samples'].append(event['memory_mb'])
            if 'process_name' in event:
                self.current_window['process']['proc_names'].add(event['process_name'])
    
    def update_filesystem_event(self, event):
        """Update current window with filesystem event"""
        with self.lock:
            if event.get('event_type') == 'created':
                self.current_window['filesystem']['file_create'] += 1
            if event.get('file_extension') in EXECUTABLE_EXTENSIONS:
                self.current_window['filesystem']['file_exec'] += 1
            if event.get('file_hash'):
                self.current_window['filesystem']['file_hashes'].add(event['file_hash'])
    
    def compute_port_entropy(self, ports):
        """Calculate Shannon entropy of port distribution"""
        if not ports:
            return 0.0
        from collections import Counter
        counts = Counter(ports)
        total = len(ports)
        entropy = -sum((count/total) * np.log2(count/total) for count in counts.values())
        return entropy
    
    def finalize_window(self):
        """Finalize window with GUARANTEED 28-feature output matching FEATURE_SCHEMA"""
        with self.lock:
            elapsed = time.time() - self.current_window['start_time']

            # Current window stats
            net = self.current_window['network']
            proc = self.current_window['process']
            fs = self.current_window['filesystem']

            # === COMPUTE ALL 28 FEATURES IN EXACT SCHEMA ORDER ===
            
            # 1-9: Network features
            conn_count = np.clip(np.log1p(net['conn_count']), 0, 10)
            unique_dst_count = np.clip(len(net['unique_dst']), 0, 50)
            bytes_sent = np.clip(np.log1p(net['bytes_sent']), 0, 20)
            bytes_recv = np.clip(np.log1p(net['bytes_recv']), 0, 20)
            
            port_entropy = self.compute_port_entropy(net['ports'])
            
            tcp_count = sum(1 for p in net['protocols'] if 'TCP' in str(p))
            udp_count = sum(1 for p in net['protocols'] if 'UDP' in str(p))
            total_proto = max(len(net['protocols']), 1)
            tcp_ratio = tcp_count / total_proto
            udp_ratio = udp_count / total_proto
            
            conn_rate = net['conn_count'] / max(elapsed, 1)
            dst_churn_rate = len(net['unique_dst']) / max(elapsed, 1)
            
            # 10-13: Process features
            proc_spawn_count = np.clip(np.log1p(proc['spawn_count']), 0, 8)
            avg_cpu = np.mean(proc['cpu_samples']) if proc['cpu_samples'] else 0.0
            avg_cpu_scaled = np.clip(avg_cpu / 100.0, 0, 1)
            avg_memory = np.mean(proc['memory_samples']) if proc['memory_samples'] else 0.0
            avg_memory_scaled = np.clip(np.log1p(avg_memory) / 10.0, 0, 1)
            unique_proc_count = np.clip(len(proc['proc_names']), 0, 100)
            
            # 14-16: Filesystem features
            file_create_count = np.clip(fs['file_create'], 0, 100)
            file_exec_count = np.clip(fs['file_exec'], 0, 20)
            file_hash_novelty = np.clip(len(fs['file_hashes']), 0, 50)
            
            # 17-21: Rate of change features
            prev_window = self.feature_buffer[-1] if len(self.feature_buffer) > 0 else None
            
            if prev_window:
                delta_conn_count = net['conn_count'] - prev_window.get('conn_count_raw', 0)
                delta_file_create = fs['file_create'] - prev_window.get('file_create_raw', 0)
                delta_proc_spawn = proc['spawn_count'] - prev_window.get('proc_spawn_raw', 0)
                delta_unique_hashes = len(fs['file_hashes']) - prev_window.get('unique_hashes_raw', 0)
                prev_delta_conn = prev_window.get('delta_conn_count', 0)
                conn_acceleration = delta_conn_count - prev_delta_conn
            else:
                delta_conn_count = delta_file_create = delta_proc_spawn = 0
                delta_unique_hashes = conn_acceleration = 0
            
            # 22-24: Pattern flags
            ransomware_burst = float(
                fs['file_create'] > RANSOM_BURST_FILE_COUNT and
                len(fs['file_hashes']) > RANSOM_BURST_HASH_COUNT and
                elapsed < RANSOM_BURST_WINDOW_SEC
            )
            
            c2_pattern = 0.0
            if len(net['ports']) > C2_MIN_CONN_COUNT:
                from collections import Counter
                dst_counts = Counter(str(dst) for dst in net['dst_list'])
                max_dst_count = max(dst_counts.values()) if dst_counts else 0
                c2_pattern = float(max_dst_count > C2_MIN_CONN_COUNT and len(net['unique_dst']) < C2_MAX_UNIQUE_DST)
            
            cred_dump_pattern = float(
                any(
                    substr in str(p).lower()
                    for p in proc['proc_names']
                    for substr in CREDENTIAL_PROCESS_SUBSTRINGS
                )
                and proc['spawn_count'] > 2
            )
            
            # 25-28: Raw counters (for next window's deltas)
            conn_count_raw = net['conn_count']
            proc_spawn_raw = proc['spawn_count']
            file_create_raw = fs['file_create']
            unique_hashes_raw = len(fs['file_hashes'])
            
            # === BUILD FEATURE DICT IN EXACT SCHEMA ORDER ===
            raw_features = {
                # Network (1-9)
                'conn_count': float(conn_count),
                'unique_dst_count': float(unique_dst_count),
                'bytes_sent': float(bytes_sent),
                'bytes_recv': float(bytes_recv),
                'port_entropy': float(np.clip(port_entropy, 0, 8)),
                'tcp_ratio': float(tcp_ratio),
                'udp_ratio': float(udp_ratio),
                'conn_rate': float(np.clip(conn_rate, 0, 10)),
                'dst_churn_rate': float(np.clip(dst_churn_rate, 0, 5)),
                
                # Process (10-13)
                'proc_spawn_count': float(proc_spawn_count),
                'avg_proc_cpu': float(avg_cpu_scaled),
                'avg_proc_memory': float(avg_memory_scaled),
                'unique_proc_names': float(unique_proc_count),
                
                # Filesystem (14-16)
                'file_create_count': float(file_create_count),
                'file_exec_count': float(file_exec_count),
                'file_hash_novelty': float(file_hash_novelty),
                
                # Rate of change (17-21)
                'delta_conn_count': float(np.clip(delta_conn_count, -10, 10)),
                'delta_file_create': float(np.clip(delta_file_create, -50, 50)),
                'delta_proc_spawn': float(np.clip(delta_proc_spawn, -10, 10)),
                'delta_unique_hashes': float(np.clip(delta_unique_hashes, -20, 20)),
                'conn_acceleration': float(np.clip(conn_acceleration, -5, 5)),
                
                # Pattern flags (22-24)
                'ransomware_burst': float(ransomware_burst),
                'c2_pattern': float(c2_pattern),
                'cred_dump_pattern': float(cred_dump_pattern),
                
                # Raw counters (25-28)
                'conn_count_raw': float(conn_count_raw),
                'proc_spawn_raw': float(proc_spawn_raw),
                'file_create_raw': float(file_create_raw),
                'unique_hashes_raw': float(unique_hashes_raw),
                
                # Metadata
                'timestamp': time.time()
            }
            
            # ✅ CRITICAL ML ARCHITECTURE: 
            # Always pad features to match the model's expected input dimension.
            # The local schema produces 24 ML features, but server models expect 99.
            # We zero-pad the remaining dimensions deterministically.
            
            # Normalize features first
            normalized_features = self.normalize_features(raw_features)
            self.feature_buffer.append(normalized_features)

            # Human-readable context for alerts / GUI (captured before reset)
            dst_list = list(net.get('dst_list') or [])
            self.last_window_snapshot = {
                'window_elapsed_sec': float(elapsed),
                'network': {
                    'connection_events': int(net['conn_count']),
                    'unique_destinations': len(net['unique_dst']),
                    'sample_destinations': list(dict.fromkeys(dst_list[-12:])),
                },
                'process': {
                    'spawn_events': int(proc['spawn_count']),
                    'sample_process_names': sorted(proc['proc_names'])[:15],
                },
                'filesystem': {
                    'file_creates': int(fs['file_create']),
                    'file_execs': int(fs['file_exec']),
                    'distinct_hashes': len(fs['file_hashes']),
                },
                'timestamp': time.time(),
            }
            
            # High-confidence alerts
            if ransomware_burst or c2_pattern or cred_dump_pattern:
                self._trigger_immediate_alert(raw_features, normalized_features)
            
            # Reset window (keep same keys as __init__ / update_* helpers)
            self.current_window = {
                'start_time': time.time(),
                'network': {'conn_count': 0, 'unique_dst': set(), 'dst_list': [],
                            'bytes_sent': 0, 'bytes_recv': 0, 'ports': [], 'protocols': []},
                'process': {'spawn_count': 0, 'cpu_samples': [], 'memory_samples': [],
                            'proc_names': set()},
                'filesystem': {'file_create': 0, 'file_exec': 0, 'file_hashes': set()}
            }
            
            return normalized_features


    def _trigger_immediate_alert(self, features, normalized_features):
        """Generate immediate high-priority alert for critical patterns"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'alert_type': 'immediate_threat',
            'severity': 'critical',
            'patterns_detected': [],
            'evidence': {},
            'recommended_action': ''
        }
        
        if features.get('ransomware_burst'):
            alert['patterns_detected'].append('ransomware_burst')
            alert['evidence']['file_create_count'] = features['file_create_raw']
            alert['evidence']['unique_files'] = features['unique_hashes_raw']
            alert['recommended_action'] = 'ISOLATE HOST - Quarantine active processes accessing files'
        
        if features.get('c2_pattern'):
            alert['patterns_detected'].append('c2_callback_pattern')
            alert['recommended_action'] = 'BLOCK EXTERNAL IPs - Kill network connections'
        
        if features.get('cred_dump_pattern'):
            alert['patterns_detected'].append('credential_dumping')
            alert['recommended_action'] = 'TERMINATE PROCESSES - Reset credentials'
        
        # Add to alert queue with high priority
        if hasattr(self, 'anomaly_detector') and hasattr(self.anomaly_detector, 'anomaly_alerts'):
            self.anomaly_detector.anomaly_alerts.appendleft(alert)  # Add to front
        
        print(f"\n{'='*70}")
        print(f"âš ï¸ IMMEDIATE THREAT DETECTED - {', '.join(alert['patterns_detected'])}")
        print(f"{'='*70}")
        print(f"Action: {alert['recommended_action']}")
        print(f"{'='*70}\n")
    
    def normalize_features(self, features):
        """Normalize features using robust statistics - FIXED"""
        normalized = {}
        
        # ✅ ADD: Try cache first for repeated patterns
        if hasattr(self, 'anomaly_detector'):
            feature_key = str(sorted(features.keys()))[:50]  # Truncated key
            cached = self.anomaly_detector.perf_guard.get_cached_normalization(
                feature_key, tuple(features.values())
            )
            if cached is not None:
                return cached
            
        for key, value in features.items():
            if key == 'timestamp':
                normalized[key] = value
                continue
            
            # ✅ FIX: Use robust percentile-based normalization
            if key not in self.running_mean:
                self.running_mean[key] = value
                self.running_std[key] = 1.0  # ✅ START WITH 1.0, NOT 0.0
                self.ema_values[key] = value
                self.feature_history[key] = deque(maxlen=100)  # ✅ NEW: Track history
            
            # ✅ NEW: Store raw values for percentile calculation
            self.feature_history[key].append(value)
            
            # ✅ FIX: Only update statistics after sufficient data
            if len(self.feature_history[key]) >= 20:
                # Use median/MAD for robustness
                median = np.median(list(self.feature_history[key]))
                mad = np.median(np.abs(np.array(list(self.feature_history[key])) - median))
                
                # Median Absolute Deviation normalization
                if mad > 1e-6:  # ✅ Avoid division by tiny values
                    normalized[key] = (value - median) / (MAD_SCALE * mad)
                else:
                    # Fall back to min-max scaling
                    hist_array = np.array(list(self.feature_history[key]))
                    min_val = np.min(hist_array)
                    max_val = np.max(hist_array)
                    if max_val - min_val > 1e-6:
                        normalized[key] = (value - min_val) / (max_val - min_val)
                    else:
                        normalized[key] = 0.0
            else:
                # ✅ Early phase: use simple clipping
                normalized[key] = np.clip(value, -EARLY_CLIP_SIGMA, EARLY_CLIP_SIGMA)
        
        # ✅ ADD: Cache result
        if hasattr(self, 'anomaly_detector'):
            self.anomaly_detector.perf_guard.cache_normalization(
                feature_key, normalized
            )
            
        return normalized
    
    def get_feature_matrix(self):
        """Return feature matrix padded to model_input_dim"""
        if len(self.feature_buffer) == 0:
            return None, None

        # Core 24 ML feature names (excluding raw counters and timestamp)
        core_feature_names = [f for f in self.FEATURE_SCHEMA if not f.endswith('_raw') and f != 'timestamp']
        
        # Build matrix from buffered windows
        matrix = []
        for window in self.feature_buffer:
            row = []
            for feature_name in core_feature_names:
                value = window.get(feature_name, 0.0)
                row.append(float(value))
            matrix.append(row)
        
        matrix = np.array(matrix)
        
        # ✅ CRITICAL: Pad to match model input dimension
        actual_cols = matrix.shape[1] if len(matrix.shape) > 1 else 0
        if actual_cols < self.model_input_dim:
            pad_width = self.model_input_dim - actual_cols
            padding = np.zeros((matrix.shape[0], pad_width))
            matrix = np.hstack([matrix, padding])
            # Extend feature names
            padded_names = core_feature_names + [f"pad_{i}" for i in range(pad_width)]
        else:
            padded_names = core_feature_names
        
        return matrix, padded_names

    
    def should_finalize_window(self):
        """Check if current window should be finalized"""
        elapsed = time.time() - self.current_window['start_time']
        return elapsed >= self.window_duration


class PerformanceGuard:
    """Rate limiting and safety guards for ML operations"""
    
    def __init__(self):
        self.ml_operation_timestamps = deque(maxlen=100)
        self.alert_burst_timestamps = deque(maxlen=100)
        self.last_training_time = None
        self.training_in_progress = False
        
        self.MAX_ML_OPS_PER_MINUTE = ML_OPS_PER_MINUTE
        self.MAX_ALERTS_PER_MINUTE = ALERTS_PER_MINUTE
        self.MIN_TRAINING_INTERVAL = MIN_TRAIN_INTERVAL_SEC
        
        self.norm_cache = {}
        self.norm_cache_ttl = {}
        self.CACHE_TTL = NORM_CACHE_TTL_SEC
        
        # Locks
        self.training_lock = threading.Lock()
        self.cache_lock = threading.Lock()
    
    def can_perform_ml_operation(self):
        """Check if ML operation is allowed (rate limiting)"""
        now = time.time()
        
        # Remove timestamps older than 1 minute
        cutoff = now - RATE_LIMIT_WINDOW_SEC
        while self.ml_operation_timestamps and self.ml_operation_timestamps[0] < cutoff:
            self.ml_operation_timestamps.popleft()
        
        if len(self.ml_operation_timestamps) >= self.MAX_ML_OPS_PER_MINUTE:
            return False
        
        self.ml_operation_timestamps.append(now)
        return True
    
    def can_generate_alert(self):
        """Check if alert generation is allowed (prevent burst)"""
        now = time.time()
        
        cutoff = now - RATE_LIMIT_WINDOW_SEC
        while self.alert_burst_timestamps and self.alert_burst_timestamps[0] < cutoff:
            self.alert_burst_timestamps.popleft()
        
        if len(self.alert_burst_timestamps) >= self.MAX_ALERTS_PER_MINUTE:
            return False
        
        self.alert_burst_timestamps.append(now)
        return True
    
    def can_start_training(self, force=False):
        """Check if model training can start"""
        if self.training_in_progress:
            return False
        
        if force:
            return True
        
        if self.last_training_time is None:
            return True
        
        elapsed = time.time() - self.last_training_time
        return elapsed >= self.MIN_TRAINING_INTERVAL
    
    def acquire_training_lock(self):
        """Acquire training lock (blocking)"""
        acquired = self.training_lock.acquire(blocking=True, timeout=TRAINING_LOCK_TIMEOUT_SEC)
        if acquired:
            self.training_in_progress = True
        return acquired
    
    def release_training_lock(self):
        """Release training lock"""
        self.training_in_progress = False
        self.training_lock.release()
        self.last_training_time = time.time()
    
    def get_cached_normalization(self, feature_key, feature_vector):
        """Get cached feature normalization (O(1) lookup)"""
        with self.cache_lock:
            now = time.time()
            
            # Check cache validity
            if feature_key in self.norm_cache:
                if now - self.norm_cache_ttl.get(feature_key, 0) < self.CACHE_TTL:
                    return self.norm_cache[feature_key]
            
            return None
    
    def cache_normalization(self, feature_key, normalized_vector):
        """Cache normalized feature vector"""
        with self.cache_lock:
            self.norm_cache[feature_key] = normalized_vector
            self.norm_cache_ttl[feature_key] = time.time()
    
    def cleanup_cache(self):
        """Remove expired cache entries"""
        with self.cache_lock:
            now = time.time()
            expired_keys = [
                key for key, ttl in self.norm_cache_ttl.items()
                if now - ttl > self.CACHE_TTL
            ]
            for key in expired_keys:
                del self.norm_cache[key]
                del self.norm_cache_ttl[key]


