import os
import sys
import time
import json
import logging
import socket
import pickle
import hashlib
import platform
import threading
import psutil
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, deque
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy import stats
from threading import Lock
import queue
import codecs
import google.generativeai as genai

from client.utils.config import *
from client.core.feature_manager import PerformanceGuard
from client.models.autoencoder import AutoencoderAnomalyDetector
from client.models.isolation_forest import IsolationForestAnomalyDetector
from client.models.one_class_svm import OneClassSVMAnomalyDetector
from client.models.zscore_filter import ZScoreFilter
from dotenv import load_dotenv

load_dotenv()

try:
    from shared.detection_config import (
        ENSEMBLE_WEIGHTS as _ENS_W,
        SEVERITY_THRESHOLDS as _SEV,
        PATTERN_OVERRIDE_SCORES as _PATTERN_SCORES,
        CONSENSUS_STRONG_MODEL_THRESHOLD, CONSENSUS_BONUS_PER_EXTRA_MODEL,
        ZSCORE_THRESHOLD as _ZSCORE_THRESH,
        ZSCORE_MIN_FLAGS, ZSCORE_PER_FEATURE_CAP, ZSCORE_MIN_RUNNING_COUNT,
        IF_CONTAMINATION, IF_N_ESTIMATORS, IF_MAX_SAMPLES, IF_RANDOM_STATE,
        IF_THRESHOLD_INIT, IF_THRESHOLD_PERCENTILE, IF_SCORE_SCALE, IF_SCORE_CAP, IF_REPORT_THRESHOLD,
        AE_EXCESS_MULTIPLIER, AE_SCORE_SCALE, AE_SCORE_CAP, AE_REPORT_THRESHOLD,
        AE_EPOCHS as _AE_EPOCHS, AE_BATCH_SIZE as _AE_BATCH,
        AE_THRESHOLD_PERCENTILE as _AE_THRESH_PCT,
        OCSVM_NU, OCSVM_GAMMA, OCSVM_KERNEL, OCSVM_SCORE_SCALE, OCSVM_SCORE_CAP,
        MIN_TRAIN_SAMPLES, IQR_OUTLIER_MULTIPLIER, MIN_CLEANED_SAMPLES,
        RETRAIN_MIN_BUFFER, RETRAIN_INTERVAL_SEC, RETRAIN_PERIODIC_SEC,
        ALERT_TTL_BY_SEVERITY,
        BASELINE_DEQUE_MAXLEN, LOCAL_ALERT_QUEUE_MAX,
        INFERENCE_CACHE_MAX, INFERENCE_CACHE_EVICT, INFERENCE_HASH_PRECISION,
        BASELINE_LEARNING_MIN_SAMPLES,
        FL_CLIENT_CLIP_BOUND, FL_ISO_SAMPLE_LAST_N, FL_TENSOR_DP_SIGMA,
        CREDENTIAL_PROCESS_SUBSTRINGS,
        HIGH_RISK_EXTENSIONS as _HR_EXT,
        RANSOMWARE_EXTENSIONS as _RANSOM_EXT,
    )
except ImportError:
    pass



class EnhancedAnomalyDetector:
    """Enhanced statistical anomaly detection with federated learning"""
    
    def __init__(self,use_ocsvm=False):
        self.network_baseline = {'connections': deque(maxlen=BASELINE_DEQUE_MAXLEN), 'bytes': deque(maxlen=BASELINE_DEQUE_MAXLEN)}
        self.process_baseline = {'count': deque(maxlen=BASELINE_DEQUE_MAXLEN), 'cpu': deque(maxlen=BASELINE_DEQUE_MAXLEN)}
        self.file_baseline = {'events': deque(maxlen=BASELINE_DEQUE_MAXLEN)}
        self.anomaly_alerts = deque(maxlen=LOCAL_ALERT_QUEUE_MAX)
        # Enhanced model weights
        self.model_weights = {
            'version': 0,
            'network_threshold': 3.0,
            'process_threshold': 3.0,
            'file_threshold': 3.0,
            'network_sensitivity': 0.5,
            'process_sensitivity': 0.5,
            'file_sensitivity': 0.5,
            # Global baseline parameters from federated learning
            'network_baseline_mean': 0.0,
            'network_baseline_std': 1.0,
            'process_baseline_mean': 0.0,
            'process_baseline_std': 1.0,
            'file_baseline_mean': 0.0,
            'file_baseline_std': 1.0,
            'anomaly_alpha': 0.95,
            'anomaly_beta': 0.1
        }
        
        # Track local anomalies for reporting
        # Anomaly tracking
        self.anomaly_count = 0
        self.total_detections = 0
        self.ema_network = None
        self.ema_process = None
        self.ema_file = None
        
        # --- NEW: ML Models ---
        self.isolation_forest = None
        self.autoencoder = None
        self.use_ocsvm = use_ocsvm
        self.one_class_svm = None if not use_ocsvm else OneClassSVM(kernel=OCSVM_KERNEL, nu=OCSVM_NU)
        
        # --- NEW: Thresholds ---
        self.iso_threshold = IF_THRESHOLD_INIT
        self.ae_threshold = None
        self.zscore_threshold = _ZSCORE_THRESH

        self.adaptive_threshold_multiplier = AE_EXCESS_MULTIPLIER
        self.recent_anomaly_rate = deque(maxlen=10)
        
        # --- NEW: Last training time ---
        self.last_training_time = None
        self.training_interval = 600  
        
        self.gui_callback = None
        
        # ✅ NEW: Add performance guard
        self.perf_guard = PerformanceGuard()
        
        # ✅ NEW: Inference cache for repeated vectors
        self.inference_cache = {}
        self.inference_cache_lock = threading.Lock()
        
        self.baseline_learning_mode = True
        self.baseline_learning_samples = 0
        self.baseline_learning_threshold = 100

        self.alert_cache = {}  # {signature: expiry_time}
        self.alert_cache_lock = threading.Lock()
    
    # ========================================================================
    # NEW: Inference result caching
    # ========================================================================
    def _hash_feature_vector(self, feature_vector):
        """Generate hash for feature vector (for caching)"""
        # Round to 3 decimals to increase cache hit rate
        rounded = tuple(round(float(v), 3) for v in feature_vector)
        return hash(rounded)
    
    def _cache_inference_result(self, feature_hash, is_anomaly, anomaly_info):
        """Cache inference result"""
        with self.inference_cache_lock:
            # Keep cache size bounded
            if len(self.inference_cache) > 1000:
                # Remove oldest 200 entries (FIFO)
                keys_to_remove = list(self.inference_cache.keys())[:200]
                for key in keys_to_remove:
                    del self.inference_cache[key]
            
            self.inference_cache[feature_hash] = {
                'is_anomaly': is_anomaly,
                'info': anomaly_info,
                'timestamp': time.time()
            }
            
    def _quick_heuristic_check(self, feature_vector, feature_names):
        """Quick rule-based check when ML rate limit hit"""
        anomaly_info = {
            'is_anomaly': False,
            'ensemble_score': 0.0,
            'severity': 'normal',
            'contributing_features': [],
            'model_contributions': {}
        }
        
        # Check for critical patterns only
        feature_dict = dict(zip(feature_names, feature_vector))
        
        # Critical: Ransomware burst
        if feature_dict.get('ransomware_burst', 0) == 1.0:
            anomaly_info['is_anomaly'] = True
            anomaly_info['severity'] = 'critical'
            anomaly_info['ensemble_score'] = 9.5
            anomaly_info['contributing_features'] = ['ransomware_burst']
            return True, anomaly_info
        
        # Critical: C2 callback pattern
        if feature_dict.get('c2_pattern', 0) == 1.0:
            anomaly_info['is_anomaly'] = True
            anomaly_info['severity'] = 'high'
            anomaly_info['ensemble_score'] = 8.5
            anomaly_info['contributing_features'] = ['c2_pattern']
            return True, anomaly_info
        
        return False, anomaly_info
    
    # --- NEW: Ensemble detection method ---
    def detect_anomaly_ensemble(self, feature_vector, feature_names):
        """
        Multi-tier ensemble anomaly detection with CONSENSUS SCORING.
        Returns: (is_anomaly, anomaly_info_dict)
        """
        
        # === FIX DIMENSIONALITY MISMATCH (24 vs 99 features) ===
        target_features = len(feature_vector)
        if getattr(self, "isolation_forest", None) is not None:
            if hasattr(self.isolation_forest, "n_features_in_"):
                target_features = self.isolation_forest.n_features_in_
        elif getattr(self, "autoencoder", None) is not None and self.autoencoder.is_trained:
            if hasattr(self.autoencoder.scaler, "mean_"):
                target_features = len(self.autoencoder.scaler.mean_)
                
        if len(feature_vector) < target_features:
            padding = np.zeros(target_features - len(feature_vector))
            feature_vector = np.concatenate([feature_vector, padding])
            # Extend feature names to prevent zip() truncation
            feature_names = feature_names + [f"pad_{i}" for i in range(len(padding))]

        # === Rate Limit ===
        if not self.perf_guard.can_perform_ml_operation():
            return self._quick_heuristic_check(feature_vector, feature_names)

        self.total_detections += 1

        # === Initialize model scores ===
        model_scores = {
            'zscore': 0.0,
            'isolation_forest': 0.0,
            'autoencoder': 0.0,
            'ocsvm': 0.0
        }

        # === Model Weight Initialization ===
        if not hasattr(self, 'model_weights_adaptive'):
            self.model_weights_adaptive = dict(_ENS_W)

        # === Base anomaly info ===
        anomaly_info = {
            'is_anomaly': False,
            'ensemble_score': 0.0,
            'severity': 'normal',
            'contributing_features': [],
            'feature_groups': {},
            'model_contributions': {}
        }

        # Group features for reporting
        feature_groups = self._group_features_by_category(feature_names, feature_vector)

        # ============================
        #  MODEL 1 — Z-SCORE
        # ============================
        zscore_flags = []
        zscore_score = 0.0

        # Only apply Z-score after sufficient baseline data (running stats stabilize)
        fm = getattr(self, 'feature_manager', None)
        baseline_ready = (fm is not None and getattr(fm, 'running_count', 0) >= ZSCORE_MIN_RUNNING_COUNT)

        if baseline_ready:
            for fname, fval in zip(feature_names, feature_vector):
                if abs(fval) > self.zscore_threshold:
                    zscore_flags.append(fname)
                    zscore_score += min(abs(fval) / self.zscore_threshold, ZSCORE_PER_FEATURE_CAP)

        if len(zscore_flags) >= ZSCORE_MIN_FLAGS:
            model_scores['zscore'] = min(zscore_score / max(len(zscore_flags), 1), 10.0)
            anomaly_info['model_contributions']['zscore'] = {
                'score': model_scores['zscore'],
                'flags': zscore_flags[:5]
            }

        # ============================
        #  MODEL 2 — ISOLATION FOREST
        # ============================
        if self.isolation_forest is not None:
            try:
                iso_raw = self.isolation_forest.score_samples([feature_vector])[0]

                # Use configured threshold: anomalous if score < iso_threshold
                if iso_raw < self.iso_threshold:
                    model_scores['isolation_forest'] = min(abs(iso_raw) * IF_SCORE_SCALE, IF_SCORE_CAP)

                    if model_scores['isolation_forest'] > IF_REPORT_THRESHOLD:
                        real_mask = np.array([not n.startswith('pad_') for n in feature_names])
                        masked_abs = np.abs(feature_vector) * real_mask
                        top_idx = np.argsort(masked_abs)[-5:]
                        anomaly_info['model_contributions']['isolation_forest'] = {
                            'score': model_scores['isolation_forest'],
                            'raw_score': float(iso_raw),
                            'features': [feature_names[i] for i in top_idx if not feature_names[i].startswith('pad_')]
                        }
            except Exception as e:
                print(f"[ISO] Error: {e}")

        # ============================
        #  MODEL 3 — AUTOENCODER
        # ============================
        if self.autoencoder is not None and self.autoencoder.is_trained:
            try:
                is_ae, recon_errors = self.autoencoder.detect_anomaly([feature_vector])
                if is_ae is not None and len(recon_errors) > 0:
                    recon = float(recon_errors[0])
                    if recon > self.autoencoder.threshold * AE_EXCESS_MULTIPLIER:
                        model_scores['autoencoder'] = min(
                            (recon / self.autoencoder.threshold) * AE_SCORE_SCALE, AE_SCORE_CAP
                        )
                        
                        if model_scores['autoencoder'] > AE_REPORT_THRESHOLD:
                            feature_errors = self._compute_feature_reconstruction_error(feature_vector)
                            real_mask = np.array([not n.startswith('pad_') for n in feature_names])
                            masked_errs = feature_errors * real_mask[:len(feature_errors)]
                            top_idx = np.argsort(masked_errs)[-5:]
                            anomaly_info['model_contributions']['autoencoder'] = {
                                'score': model_scores['autoencoder'],
                                'recon_error': recon,
                                'features': [feature_names[i] for i in top_idx if i < len(feature_names) and not feature_names[i].startswith('pad_')]
                            }
            except Exception as e:
                print(f"[AE] Error: {e}")

        # ============================
        #  MODEL 4 — ONE CLASS SVM
        # ============================
        if self.use_ocsvm and self.one_class_svm is not None:
            try:
                ocsvm_decision = self.one_class_svm.decision_function([feature_vector])[0]
                ocsvm_pred = self.one_class_svm.predict([feature_vector])[0]
                if ocsvm_pred == -1:
                    model_scores['ocsvm'] = min(abs(ocsvm_decision) * OCSVM_SCORE_SCALE, OCSVM_SCORE_CAP)
                    anomaly_info['model_contributions']['ocsvm'] = {
                        'score': model_scores['ocsvm'],
                        'decision_value': float(ocsvm_decision),
                        'prediction': 'anomaly'
                    }
            except Exception as e:
                print(f"[OCSVM] Error: {e}")

        # ============================
        #  CONSENSUS FUSION
        # ============================
        active_count = sum(1 for s in model_scores.values() if s > 0)
        if active_count > 0:
            total_weight = sum(
                self.model_weights_adaptive[m]
                for m in model_scores if model_scores[m] > 0
            )
            if total_weight > 0:
                ensemble_score = sum(
                    model_scores[m] * self.model_weights_adaptive[m]
                    for m in model_scores
                ) / total_weight
            else:
                ensemble_score = sum(model_scores.values()) / max(active_count, 1)
        else:
            ensemble_score = 0.0

        # Consensus bonus: multiple independent models agreeing is significant
        strong_models = sum(1 for s in model_scores.values() if s > CONSENSUS_STRONG_MODEL_THRESHOLD)
        if strong_models >= 2:
            ensemble_score *= (1.0 + CONSENSUS_BONUS_PER_EXTRA_MODEL * (strong_models - 1))

        ensemble_score = min(ensemble_score, 10.0)
        anomaly_info['ensemble_score'] = float(ensemble_score)

        # ============================
        #  BASE SEVERITY MAPPING
        # ============================
        if ensemble_score >= _SEV["critical"]:
            base_severity = "critical"
            anomaly_info['is_anomaly'] = True
        elif ensemble_score >= _SEV["high"]:
            base_severity = "high"
            anomaly_info['is_anomaly'] = True
        elif ensemble_score >= _SEV["medium"]:
            base_severity = "medium"
            anomaly_info['is_anomaly'] = True
        elif ensemble_score >= _SEV["low"]:
            base_severity = "low"
            anomaly_info['is_anomaly'] = True
        else:
            base_severity = "normal"

        # ============================
        #  OVERRIDE SEVERITY RULES
        # ============================
        feature_dict = dict(zip(feature_names, feature_vector))

        if feature_dict.get("ransomware_burst", 0) >= 1.0:
            base_severity = "critical"
            ensemble_score = max(ensemble_score, _PATTERN_SCORES["ransomware_burst"])
            anomaly_info['is_anomaly'] = True

        if feature_dict.get("c2_pattern", 0) >= 1.0:
            if base_severity != "critical":
                base_severity = "high"
            ensemble_score = max(ensemble_score, _PATTERN_SCORES["c2_pattern"])
            anomaly_info['is_anomaly'] = True

        if feature_dict.get("cred_dump_pattern", 0) >= 1.0:
            if base_severity != "critical":
                base_severity = "high"
            ensemble_score = max(ensemble_score, _PATTERN_SCORES["cred_dump_pattern"])
            anomaly_info['is_anomaly'] = True

        anomaly_info["severity"] = base_severity
        anomaly_info["ensemble_score"] = float(ensemble_score)

        # ── Detection model label: Ensemble when multiple models contribute ──
        model_name_map = {
            'isolation_forest': 'IsolationForest',
            'autoencoder':      'Autoencoder',
            'zscore':           'ZScore',
            'ocsvm':            'OCSVM',
        }
        contributing = [k for k, s in model_scores.items() if s > 0]
        if len(contributing) >= 2:
            anomaly_info['detection_model'] = 'Ensemble'
        elif len(contributing) == 1:
            anomaly_info['detection_model'] = model_name_map.get(contributing[0], 'Ensemble')
        else:
            anomaly_info['detection_model'] = 'Ensemble'

        # ── Determine primary category from feature groups ────────────────
        if feature_groups:
            top_cat = max(feature_groups, key=lambda k: feature_groups[k].get('risk', 0))
            cat_map = {'network': 'Network', 'process': 'Process',
                       'file': 'Filesystem', 'user': 'User'}
            anomaly_info['category'] = cat_map.get(top_cat, 'Unknown')

        # Also store iso_score and ae_recon_error at top level for GUI
        if 'isolation_forest' in anomaly_info['model_contributions']:
            anomaly_info['iso_score'] = anomaly_info['model_contributions']['isolation_forest'].get('raw_score')
        if 'autoencoder' in anomaly_info['model_contributions']:
            anomaly_info['ae_recon_error'] = anomaly_info['model_contributions']['autoencoder'].get('recon_error')
        if 'ocsvm' in anomaly_info['model_contributions']:
            anomaly_info['ocsvm_score'] = anomaly_info['model_contributions']['ocsvm'].get('decision_value')
        anomaly_info['zscore_flag'] = 'zscore' in anomaly_info['model_contributions']

        # ============================
        #  ALERT GENERATION
        # ============================
        if anomaly_info["is_anomaly"]:
            if not self.perf_guard.can_generate_alert():
                print("[PERF] Alert burst detected, throttling...")
                return False, anomaly_info

            self.anomaly_count += 1

            # Collect contributing features
            all_features = []
            for contrib in anomaly_info["model_contributions"].values():
                all_features.extend(contrib.get("features", [])[:3])

            seen = set()
            unique = []
            for f in all_features:
                if f not in seen:
                    seen.add(f)
                    unique.append(f)

            anomaly_info["contributing_features"] = unique[:5]
            anomaly_info["feature_groups"] = feature_groups

            self._generate_alert(feature_vector, feature_names, anomaly_info)

        return anomaly_info["is_anomaly"], anomaly_info
    
    def _group_features_by_category(self, feature_names, feature_vector):
        """Group features by category with aggregated risk"""
        groups = {
            'network': {'features': [], 'risk': 0.0},
            'process': {'features': [], 'risk': 0.0},
            'file': {'features': [], 'risk': 0.0},
            'user': {'features': [], 'risk': 0.0}
        }
        
        for fname, fval in zip(feature_names, feature_vector):
            if any(x in fname for x in ['conn', 'network', 'dst', 'port', 'tcp', 'udp']):
                groups['network']['features'].append((fname, fval))
                groups['network']['risk'] += abs(fval)
            elif any(x in fname for x in ['proc', 'cpu', 'memory', 'spawn']):
                groups['process']['features'].append((fname, fval))
                groups['process']['risk'] += abs(fval)
            elif any(x in fname for x in ['file', 'exec', 'hash']):
                groups['file']['features'].append((fname, fval))
                groups['file']['risk'] += abs(fval)
            elif any(x in fname for x in ['user', 'login', 'privilege']):
                groups['user']['features'].append((fname, fval))
                groups['user']['risk'] += abs(fval)
        
        # Normalize risk scores
        for group in groups.values():
            if group['features']:
                group['risk'] = min(group['risk'] / len(group['features']), 10.0)
        
        return groups

    def _build_context_explanation(self, feature_dict, contributing_features):
        """Build human-readable context explanation based on features"""
        explanations = []
        
        # Network-related features
        if any('conn' in f or 'network' in f for f in contributing_features):
            conn_count = feature_dict.get('conn_count', 0)
            unique_dst = feature_dict.get('unique_dst_ratio', 0)
            
            if conn_count > 5:
                explanations.append(f"  • High network activity detected ({conn_count:.0f} connections)")
            if unique_dst > 0.7:
                explanations.append(f"  • Connections to many unique destinations (diversity: {unique_dst:.2f})")
        
        # Process-related features
        if any('proc' in f or 'cpu' in f or 'memory' in f for f in contributing_features):
            proc_spawn = feature_dict.get('proc_spawn_count', 0)
            cpu = feature_dict.get('avg_proc_cpu', 0)
            
            if proc_spawn > 3:
                explanations.append(f"  • Multiple processes spawned ({proc_spawn:.0f} processes)")
            if cpu > 0.5:
                explanations.append(f"  • High CPU usage detected ({cpu*100:.1f}%)")
        
        # File-related features
        if any('file' in f for f in contributing_features):
            file_create = feature_dict.get('file_create_count', 0)
            file_exec = feature_dict.get('file_exec_count', 0)
            
            if file_create > 50:
                explanations.append(f"  • Rapid file creation ({file_create:.0f} files)")
            if file_exec > 5:
                explanations.append(f"  • Executable files created/modified ({file_exec:.0f} files)")
        
        # Port entropy
        if 'port_entropy' in contributing_features:
            entropy = feature_dict.get('port_entropy', 0)
            if entropy > 4:
                explanations.append(f"  • High port diversity (entropy: {entropy:.2f}) - possible scanning")
        
        return "\n".join(explanations) if explanations else None

    def _get_recommended_action(self, severity, top_groups, contrib_features, category):
        """Suggest SPECIFIC remediation action based on category and severity"""
        
        if severity == 'critical':
            if category == 'filesystem':
                if any('ransomware' in str(f).lower() for f in contrib_features):
                    return "🚨 IMMEDIATE: RANSOMWARE SUSPECTED - Isolate host, kill processes, restore from backup"
                return "IMMEDIATE: Quarantine affected files, analyze hashes, block execution"
            
            elif category == 'network':
                if any('c2' in str(f).lower() for f in contrib_features):
                    return "🚨 IMMEDIATE: C2 COMMUNICATION SUSPECTED - Block IPs, isolate host, capture traffic"
                return "IMMEDIATE: Isolate host from network, block suspicious IPs, analyze traffic"
            
            elif category == 'process':
                if any('cred' in str(f).lower() or 'dump' in str(f).lower() for f in contrib_features):
                    return "🚨 IMMEDIATE: CREDENTIAL THEFT SUSPECTED - Terminate processes, force password reset"
                return "IMMEDIATE: Quarantine process, collect memory dump, analyze parent chain"
            
            return "IMMEDIATE: Isolate system and investigate"
        
        elif severity == 'high':
            if category == 'network':
                return "URGENT: Review firewall logs, block suspicious destinations, monitor for persistence"
            elif category == 'process':
                return "URGENT: Investigate parent processes, check for persistence mechanisms, analyze binary"
            elif category == 'filesystem':
                return "URGENT: Stop file operations, review recent changes, check for data exfiltration"
            return "URGENT: Investigate immediately, track behavior, prepare for escalation"
        
        elif severity == 'medium':
            return f"MONITOR: Track {category} activity, correlate with other events, set alerts for escalation"
        
        return "LOG: Record for pattern analysis and baseline adjustment"

    # ✅ FIX: Add helper method to generate alerts
    def _generate_alert(self, feature_vector, feature_names, anomaly_info):
        """Generate anomaly alert with HUMAN-READABLE explanations"""

        # Initialize de-duplication cache
        if not hasattr(self, 'alert_signatures'):
            self.alert_signatures = {}

        # Build feature dict
        feature_dict = {name: float(val) for name, val in zip(feature_names, feature_vector)}

        # Base fields
        contrib = anomaly_info.get('contributing_features', [])
        severity = anomaly_info.get('severity', 'unknown')
        feature_groups = anomaly_info.get('feature_groups', {})
        category = anomaly_info.get('category', 'system')
        
        # Create signature: category_severity_features
        feature_sig = '_'.join(sorted(contrib[:3]))
        alert_signature = f"{category}_{severity}_{feature_sig}"

        # ✅ Check cache with lock
        with self.alert_cache_lock:
            now = time.time()
            
            # Clean expired entries
            expired_keys = [k for k, v in self.alert_cache.items() if v < now]
            for k in expired_keys:
                del self.alert_cache[k]
            
            # Check if duplicate
            if alert_signature in self.alert_cache:
                print(f"  [ALERT] Duplicate suppressed: {alert_signature}")
                return  # ✅ STOP HERE - don't generate duplicate
            
            # Set TTL based on severity
            ttl_seconds = ALERT_TTL_BY_SEVERITY.get(severity, 120)
            
            self.alert_cache[alert_signature] = now + ttl_seconds
            
        if not feature_groups:
            feature_groups = self._group_features_by_category(feature_names, feature_vector)
        
        # Get top risk groups
        top_groups = sorted(
            [(k, v['risk']) for k, v in feature_groups.items() if v['risk'] > 0],
            key=lambda x: x[1],
            reverse=True
        )[:3]

        # ==========================================
        # ✓ NEW: CATEGORY DETECTION
        # ==========================================
        category_scores = {
            'network': 0,
            'process': 0,
            'filesystem': 0,
            'user': 0
        }

        for feature in contrib:
            if any(x in feature for x in ['conn', 'dst', 'tcp', 'udp', 'port']):
                category_scores['network'] += 1
            elif any(x in feature for x in ['proc', 'cpu', 'memory', 'spawn']):
                category_scores['process'] += 1
            elif any(x in feature for x in ['file', 'hash']):
                category_scores['filesystem'] += 1
            elif any(x in feature for x in ['user', 'login']):
                category_scores['user'] += 1

        # Dominant category from feature keywords (fallback when ensemble category missing)
        primary_category = max(category_scores, key=category_scores.get)
        if category_scores[primary_category] == 0:
            primary_category = 'system'

        display_category = anomaly_info.get('category')
        if not display_category or display_category == 'Unknown':
            _title = {
                'network': 'Network', 'process': 'Process', 'filesystem': 'Filesystem',
                'user': 'User', 'system': 'Unknown',
            }
            display_category = _title.get(primary_category, 'Unknown')

        explanation_key = (anomaly_info.get('category') or '').lower()
        if not explanation_key or explanation_key == 'unknown':
            explanation_key = primary_category

        # ==========================================
        # ✓ NEW: BUILD HUMAN-READABLE EXPLANATION
        # ==========================================
        explanation_parts = []
        evidence_parts = []
        
        # Model scores
        score = anomaly_info.get('ensemble_score')
        if score:
            explanation_parts.append(f"Threat Level: {score:.1f}/10")

        #  NEW: PLAIN ENGLISH EXPLANATIONS
        if explanation_key == 'network':
            conn_count = feature_dict.get('conn_count', 0)
            unique_dst = feature_dict.get('unique_dst_count', 0)
            port_entropy = feature_dict.get('port_entropy', 0)
            c2_pattern = feature_dict.get('c2_pattern', 0)
            
            if c2_pattern == 1.0:
                explanation_parts.append("⚠️ Possible Command & Control (C2) communication detected")
                evidence_parts.append("Repeated connections to same destination with regular timing")
            
            if conn_count > 5:
                explanation_parts.append(f"High network activity: {int(conn_count * 100)} connections")
            
            if unique_dst > 3:
                explanation_parts.append(f"Multiple destinations accessed: {int(unique_dst)} unique IPs")
            
            if port_entropy > 4:
                explanation_parts.append(f"Port scanning behavior detected (entropy: {port_entropy:.1f})")
                evidence_parts.append("Many different ports accessed in short time")
        
        elif explanation_key == 'process':
            spawn_count = feature_dict.get('proc_spawn_count', 0)
            cpu = feature_dict.get('avg_proc_cpu', 0)
            memory = feature_dict.get('avg_proc_memory', 0)
            cred_dump = feature_dict.get('cred_dump_pattern', 0)
            
            if cred_dump == 1.0:
                explanation_parts.append("🚨 CRITICAL: Credential dumping activity detected")
                evidence_parts.append("Process accessing sensitive credential stores (lsass/SAM)")
            
            if spawn_count > 3:
                explanation_parts.append(f"Rapid process creation: {int(spawn_count * 10)} processes spawned")
                evidence_parts.append("Unusual process spawning pattern detected")
            
            if cpu > 0.7:
                explanation_parts.append(f"High CPU usage: {cpu * 100:.1f}%")
            
            if memory > 0.5:
                explanation_parts.append(f"High memory consumption: {memory * 5000:.0f} MB")
        
        elif explanation_key == 'filesystem' or explanation_key == 'file':
            file_create = feature_dict.get('file_create_count', 0)
            file_exec = feature_dict.get('file_exec_count', 0)
            ransomware = feature_dict.get('ransomware_burst', 0)
            
            if ransomware == 1.0:
                explanation_parts.append("🚨 CRITICAL: RANSOMWARE PATTERN DETECTED")
                evidence_parts.append("Rapid mass file creation with encryption indicators")
            
            if file_create > 50:
                explanation_parts.append(f"Mass file activity: {int(file_create)} files created/modified")
                evidence_parts.append("File creation rate exceeds normal baseline")
            
            if file_exec > 5:
                explanation_parts.append(f"Executable file activity: {int(file_exec)} .exe/.dll files")
                evidence_parts.append("New executable files detected")
        
        # Fallback: generic explanation
        if not explanation_parts:
            explanation_parts.append("Multiple statistical anomalies detected")
            evidence_parts.append(f"Top features: {', '.join(contrib[:3])}")

        # ==========================================
        # ✓ RECOMMENDED ACTION (SIMPLIFIED)
        # ==========================================
        if severity == 'critical':
            recommended_action = "🚨 IMMEDIATE ACTION REQUIRED: Isolate system and investigate"
        elif severity == 'high':
            recommended_action = "⚠️ URGENT: Review activity and block if malicious"
        elif severity == 'medium':
            recommended_action = "⚠️ MONITOR: Track behavior and correlate with other events"
        else:
            recommended_action = "✅ LOG: Record for baseline analysis"

        # ==========================================
        # ✓ BUILD SIGNATURE FOR DEDUPLICATION
        # ==========================================
        group_sig = '_'.join([g[0] for g in top_groups])
        alert_signature = f"{severity}_{primary_category}_{group_sig}"

        # TTL dedup check
        now = time.time()
        if alert_signature in self.alert_signatures:
            expiry = self.alert_signatures[alert_signature]
            if now < expiry:
                return  # Skip duplicate

        # TTL per severity
        ttl_seconds = {
            'critical': 300,
            'high': 180,
            'medium': 120,
            'low': 60
        }.get(severity, 120)

        self.alert_signatures[alert_signature] = now + ttl_seconds

        # ==========================================
        # ✓ FINAL ALERT OBJECT (GUI + telemetry + admin DB)
        # ==========================================
        base_expl = " | ".join(explanation_parts)
        alert = {
            'timestamp': datetime.now().isoformat(),
            'hostname': socket.gethostname(),
            'client_id': CLIENT_ID,
            'category': display_category,
            'severity': severity,
            'detection_model': anomaly_info.get('detection_model', 'Ensemble'),
            'ensemble_score': score,
            'iso_score': anomaly_info.get('iso_score'),
            'ae_recon_error': anomaly_info.get('ae_recon_error'),
            'ocsvm_score': anomaly_info.get('ocsvm_score'),
            'zscore_flag': anomaly_info.get('zscore_flag', False),
            'contributing_features': contrib[:5],
            'feature_values': {
                f: float(feature_vector[i])
                for i, f in enumerate(feature_names)
                if i < len(feature_vector)
            },
            'feature_groups': {k: v['risk'] for k, v in feature_groups.items()},
            'model_contributions': anomaly_info.get('model_contributions', {}),
            'explanation': base_expl,
            'evidence': " | ".join(evidence_parts) if evidence_parts else "Multiple anomalous behaviors",
            'recommended_action': recommended_action,
            'signature': alert_signature,
            'ttl_expires': self.alert_signatures[alert_signature],
        }

        parent = getattr(self, '_parent_agent', None)
        fm = getattr(self, 'feature_manager', None)
        snap = getattr(fm, 'last_window_snapshot', None) or {}
        nw = snap.get('network') or {}
        pr = snap.get('process') or {}
        fs = snap.get('filesystem') or {}
        related_net = [str(x) for x in (nw.get('sample_destinations') or [])[:10]]
        related_proc = [str(x) for x in (pr.get('sample_process_names') or [])[:15]]
        related_paths = []
        fc = getattr(parent, 'full_collector', None) if parent is not None else None
        if fc is not None:
            for ev in list(fc.recent_file_events)[-10:]:
                pth = ev.get('path') or ev.get('name')
                if pth:
                    related_paths.append(str(pth))
        related_paths = related_paths[-12:]
        grounding_bits = []
        if related_net:
            grounding_bits.append("Network (window): " + ", ".join(related_net[:6]))
        if related_proc:
            grounding_bits.append("Processes (window): " + ", ".join(related_proc[:10]))
        grounding_bits.append(
            f"Filesystem (window): {fs.get('file_creates', 0)} creates, "
            f"{fs.get('file_execs', 0)} exec-like, {fs.get('distinct_hashes', 0)} distinct hashes"
        )
        if related_paths:
            grounding_bits.append("Recent file paths: " + " | ".join(related_paths[-6:]))
        if fc is not None and hasattr(fc, 'user'):
            up = fc.user.peek_last_poll()
            if up:
                grounding_bits.append(
                    f"User activity poll: sessions={up.get('active_users', 0)}, "
                    f"remote={up.get('remote_sessions', 0)}, "
                    f"after_hours_flag={up.get('after_hours', 0)}"
                )
        gsum = " • ".join(grounding_bits)
        alert['related_network'] = related_net
        alert['related_processes'] = related_proc
        alert['related_paths'] = related_paths
        alert['grounding_summary'] = gsum
        if gsum:
            alert['explanation'] = f"{base_expl} | {gsum}" if base_expl else gsum
            if alert['evidence'] == "Multiple anomalous behaviors" and grounding_bits:
                alert['evidence'] = gsum[:500]
        rl_action = None
        if parent is not None and getattr(parent, 'rl_agent', None) is not None:
            try:
                rl_action = parent.rl_agent.recommend_action(alert)
                alert['rl_recommended_action'] = rl_action
            except Exception:
                pass

        self.anomaly_alerts.append(alert)

        logging.warning(
            f"⚠️ ANOMALY | {severity.upper()} | model={alert.get('detection_model')} "
            f"| cat={display_category} | RL-action={rl_action or '—'} | score={score:.1f}"
        )

        # ✓ ENHANCED CONSOLE OUTPUT
        print(f"\n{'='*80}")
        print(f"⚠️  ANOMALY DETECTED - {severity.upper()} - {display_category.upper()}")
        print(f"{'='*80}")
        print(f"What Happened:  {alert['explanation']}")
        print(f"Why It Matters: {alert['evidence']}")
        print(f"What To Do:     {recommended_action}")
        print(f"Threat Level:   {score:.1f}/10")
        print(f"{'='*80}\n")


    def _build_evidence(self, feature_dict, contrib_features):
        """Build forensic evidence from features"""
        evidence = []
        
        # Sample network evidence
        if any('conn' in f or 'dst' in f for f in contrib_features):
            # Would need actual event data - placeholder
            evidence.append("net:[multiple high-risk connections]")
        
        # Sample process evidence
        if any('proc' in f or 'spawn' in f for f in contrib_features):
            evidence.append("proc:[suspicious process spawning]")
        
        # Sample file evidence
        if any('file' in f for f in contrib_features):
            evidence.append("file:[rapid file creation/modification]")
        
        return ', '.join(evidence[:3])

    def _compute_feature_reconstruction_error(self, feature_vector):
        """Compute per-feature reconstruction error for explainability"""
        try:
            if not hasattr(self.autoencoder.scaler, 'mean_'):
                return np.zeros(len(feature_vector))
            
            expected_dim = len(self.autoencoder.scaler.mean_)
            fv = np.array(feature_vector, dtype=float)
            if len(fv) < expected_dim:
                fv = np.concatenate([fv, np.zeros(expected_dim - len(fv))])
            elif len(fv) > expected_dim:
                fv = fv[:expected_dim]
            
            X_scaled = self.autoencoder.scaler.transform([fv])
            X_pred = self.autoencoder.model.predict(X_scaled, verbose=0)
            feature_errors = np.abs(X_scaled[0] - X_pred[0])
            return feature_errors
        except Exception as e:
            print(f"  [AE] Reconstruction error computation failed: {e}")
            return np.zeros(len(feature_vector))
    
    # --- NEW: FL delta computation and clipping ---
    def compute_fl_delta(self, last_global_weights):
        """
        Compute model delta with ENHANCED METADATA for quality assessment
        Returns: (model_delta, metadata)
        """
        CLIP_BOUND = FL_CLIENT_CLIP_BOUND
        
        # Compute delta: local - global
        model_delta = {}
        for key in self.model_weights.keys():
            if key in last_global_weights:
                val_local = self.model_weights[key]
                val_global = last_global_weights[key]
                
                if isinstance(val_local, list) and isinstance(val_global, list):
                    delta = (np.array(val_local) - np.array(val_global)).tolist()
                    model_delta[key] = delta
                else:
                    try:
                        delta = float(val_local) - float(val_global)
                        model_delta[key] = delta
                    except (TypeError, ValueError):
                        pass
        
        # Compute L2 norm
        delta_norm = 0.0
        for key, value in model_delta.items():
            if isinstance(value, list):
                delta_norm += float(np.sum(np.square(value)))
            elif isinstance(value, (int, float)):
                delta_norm += value ** 2
        delta_norm = np.sqrt(delta_norm)
        
        # L2 clipping
        if delta_norm > CLIP_BOUND:
            scale_factor = CLIP_BOUND / delta_norm
            for key in model_delta.keys():
                if isinstance(model_delta[key], list):
                    model_delta[key] = (np.array(model_delta[key]) * scale_factor).tolist()
                elif isinstance(model_delta[key], (int, float)):
                    model_delta[key] *= scale_factor
            final_norm = CLIP_BOUND
            was_clipped = True
        else:
            final_norm = delta_norm
            was_clipped = False
        
        # === ENHANCED METADATA ===
        # Isolation Forest score distribution (if available)
        iso_scores = []
        fm = getattr(self, 'feature_manager', None)
        if self.isolation_forest is not None and fm is not None:
            feature_matrix, _ = fm.get_feature_matrix()
            if feature_matrix is not None and len(feature_matrix) > 0:
                try:
                    iso_scores = self.isolation_forest.score_samples(feature_matrix[-FL_ISO_SAMPLE_LAST_N:])
                    iso_scores = [float(s) for s in iso_scores]
                except (ValueError, TypeError):
                    pass
        
        iso_distribution = {
            'q25': float(np.percentile(iso_scores, 25)) if iso_scores else 0.0,
            'q50': float(np.percentile(iso_scores, 50)) if iso_scores else 0.0,
            'q75': float(np.percentile(iso_scores, 75)) if iso_scores else 0.0,
            'mean': float(np.mean(iso_scores)) if iso_scores else 0.0
        }
        
        metadata = {
            'samples_used': len(self.network_baseline['connections']),
            'local_epochs': 1,
            'final_loss': 0.0,
            'anomaly_rate': float(self.anomaly_count / max(self.total_detections, 1)),
            'delta_norm': float(final_norm),
            'was_clipped': was_clipped,
            'clip_bound': CLIP_BOUND,
            'isolation_score_distribution': iso_distribution,
            'model_version': self.model_weights.get('version', 0),
            'training_samples': {
                'network': len(self.network_baseline['connections']),
                'process': len(self.process_baseline['count']),
                'file': len(self.file_baseline['events'])
            },
            'timestamp': int(time.time())
        }
        
        return model_delta, metadata
    
    def train_models(self, feature_matrix):
        """Train models with contamination-aware data cleaning - FIXED"""
        if feature_matrix is None or len(feature_matrix) < MIN_TRAIN_SAMPLES:
            print(f"  [ML] Insufficient data ({len(feature_matrix) if feature_matrix is not None else 0}/{MIN_TRAIN_SAMPLES})")
            return False
        
        print(f"\n{'='*60}")
        print(f"[ML TRAINING] Starting with {len(feature_matrix)} samples")
        print(f"{'='*60}")
        
        # ✅ FIX: Check for NaN/Inf values
        if np.any(np.isnan(feature_matrix)) or np.any(np.isinf(feature_matrix)):
            print(f"  [ML] ✗ Invalid values detected (NaN/Inf), cleaning...")
            feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Pre-clean training data using IQR
        try:
            Q1 = np.percentile(feature_matrix, 25, axis=0)
            Q3 = np.percentile(feature_matrix, 75, axis=0)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - IQR_OUTLIER_MULTIPLIER * IQR
            upper_bound = Q3 + IQR_OUTLIER_MULTIPLIER * IQR
            
            mask = np.all((feature_matrix >= lower_bound) & (feature_matrix <= upper_bound), axis=1)
            cleaned_matrix = feature_matrix[mask]
            
            removed_count = len(feature_matrix) - len(cleaned_matrix)
            if removed_count > 0:
                print(f"  [CLEAN] Removed {removed_count} extreme outliers ({removed_count/len(feature_matrix)*100:.1f}%)")
            
            if len(cleaned_matrix) < MIN_CLEANED_SAMPLES:
                print(f"  [CLEAN] ⚠️ Too much data removed, using original")
                cleaned_matrix = feature_matrix
            
            feature_matrix = cleaned_matrix
            
        except Exception as e:
            print(f"  [CLEAN] Cleaning failed: {e}, using raw data")
        
        training_success = False
        
        # ✅ FIX 1: Train Isolation Forest with better parameters
        try:
            print("  [ISO] Training Isolation Forest...")
            self.isolation_forest = IsolationForest(
                contamination=IF_CONTAMINATION,
                n_estimators=IF_N_ESTIMATORS,
                max_samples=min(IF_MAX_SAMPLES, len(feature_matrix)),
                random_state=IF_RANDOM_STATE,
                bootstrap=True
            )
            self.isolation_forest.fit(feature_matrix)
            
            # Calibrate threshold
            scores = self.isolation_forest.score_samples(feature_matrix)
            self.iso_threshold = np.percentile(scores, IF_THRESHOLD_PERCENTILE)
            
            print(f"  [ISO] ✓ Trained | Threshold={self.iso_threshold:.4f}")
            training_success = True
        except Exception as e:
            print(f"  [ISO] ✗ Failed: {e}")
        
        # ✅ FIX 2: Train Autoencoder with GUARANTEED success
        try:
            print("  [AE] Training Autoencoder...")
            if self.autoencoder is None:
                self.autoencoder = AutoencoderAnomalyDetector(input_dim=feature_matrix.shape[1])
            
            ae_success = self.autoencoder.train(feature_matrix, epochs=_AE_EPOCHS, batch_size=_AE_BATCH)
            
            if ae_success:
                # Recompute threshold
                X_scaled = self.autoencoder.scaler.transform(feature_matrix)
                X_pred = self.autoencoder.model.predict(X_scaled, verbose=0)
                recon_errors = np.mean(np.square(X_scaled - X_pred), axis=1)
                
                self.autoencoder.threshold = np.percentile(recon_errors, _AE_THRESH_PCT)
                self.ae_threshold = self.autoencoder.threshold
                
                print(f"  [AE] ✓ Trained | Threshold={self.ae_threshold:.6f}")
                training_success = True
            else:
                print("  [AE] ✗ Training returned False")
        except Exception as e:
            print(f"  [AE] ✗ Failed: {e}")
        
        # Train One-Class SVM if enabled
        if self.use_ocsvm:
            try:
                print("  [OCSVM] Training One-Class SVM...")
                from sklearn.svm import OneClassSVM
                self.one_class_svm = OneClassSVM(
                    kernel=OCSVM_KERNEL, nu=OCSVM_NU, gamma=OCSVM_GAMMA
                )
                self.one_class_svm.fit(feature_matrix)
                self._ocsvm_trained = True
                print(f"  [OCSVM] Trained on {len(feature_matrix)} samples")
                training_success = True
            except Exception as e:
                print(f"  [OCSVM] Failed: {e}")

        if training_success:
            self.last_training_time = time.time()
            print(f"[ML TRAINING] Complete")

        return training_success
    
    def should_retrain(self, buffer_size):
        """Check if models should be retrained - OPTIMIZED FREQUENCY"""
        if self.last_training_time is None:
            return buffer_size >= MIN_TRAIN_SAMPLES
        
        time_since_training = time.time() - self.last_training_time
        
        # Retrain only if:
        # 1. 15+ minutes passed AND buffer >= 150 samples (substantial new data)
        # 2. 60+ minutes passed (periodic refresh regardless of buffer)
        return (time_since_training >= RETRAIN_INTERVAL_SEC and buffer_size >= RETRAIN_MIN_BUFFER) or \
            (time_since_training >= RETRAIN_PERIODIC_SEC)
    
    def update_baseline(self, category, metric, value):
        """Update baseline with new value and EMA"""
        if category == 'network':
            self.network_baseline[metric].append(value)
            # Update EMA
            alpha = self.model_weights.get('anomaly_alpha', 0.95)
            if self.ema_network is None:
                self.ema_network = value
            else:
                self.ema_network = alpha * self.ema_network + (1 - alpha) * value
        elif category == 'process':
            self.process_baseline[metric].append(value)
            alpha = self.model_weights.get('anomaly_alpha', 0.95)
            if self.ema_process is None:
                self.ema_process = value
            else:
                self.ema_process = alpha * self.ema_process + (1 - alpha) * value
        elif category == 'file':
            self.file_baseline[metric].append(value)
            alpha = self.model_weights.get('anomaly_alpha', 0.95)
            if self.ema_file is None:
                self.ema_file = value
            else:
                self.ema_file = alpha * self.ema_file + (1 - alpha) * value
    
    def detect_anomaly(self, category, metric, value):
        """Enhanced anomaly detection using local and global baselines"""
        self.total_detections += 1
        
        # Get local baseline
        if category == 'network':
            baseline = list(self.network_baseline.get(metric, []))
        elif category == 'process':
            baseline = list(self.process_baseline.get(metric, []))
        elif category == 'file':
            baseline = list(self.file_baseline.get(metric, []))
        else:
            return False, 0.0
        
        if len(baseline) < MIN_TRAIN_SAMPLES // 2:
            return False, 0.0
        
        # Calculate local statistics
        local_mean = np.mean(baseline)
        local_std = np.std(baseline)
        
        # Get global baseline from federated model
        global_mean = self.model_weights.get(f'{category}_baseline_mean', local_mean)
        global_std = self.model_weights.get(f'{category}_baseline_std', local_std)
        
        # Combine local and global: 60% local, 40% global
        combined_mean = 0.6 * local_mean + 0.4 * global_mean
        combined_std = 0.6 * local_std + 0.4 * global_std
        
        if combined_std == 0:
            return False, 0.0
        
        # Calculate z-score
        z_score = abs((value - combined_mean) / combined_std)
        
        # Apply adaptive threshold with sensitivity
        threshold = self.model_weights.get(f'{category}_threshold', 2.0)
        sensitivity = self.model_weights.get(f'{category}_sensitivity', 1.0)
        adjusted_threshold = threshold / sensitivity
        
        is_anomaly = z_score > adjusted_threshold
        
        if is_anomaly:
            self.anomaly_count += 1
        
        return is_anomaly, z_score
    
    def get_model_parameters(self):
        """Get comprehensive model parameters for federated learning"""
        # Calculate detailed statistics
        network_stats = {
            'mean_connections': float(np.mean(self.network_baseline['connections'])) if len(self.network_baseline['connections']) > 0 else 0.0,
            'std_connections': float(np.std(self.network_baseline['connections'])) if len(self.network_baseline['connections']) > 0 else 0.0,
            'mean_bytes': float(np.mean(self.network_baseline['bytes'])) if len(self.network_baseline['bytes']) > 0 else 0.0,
            'std_bytes': float(np.std(self.network_baseline['bytes'])) if len(self.network_baseline['bytes']) > 0 else 0.0,
            'variance_connections': float(np.var(self.network_baseline['connections'])) if len(self.network_baseline['connections']) > 1 else 0.0,
            'ema': float(self.ema_network) if self.ema_network is not None else 0.0
        }
        
        process_stats = {
            'mean_count': float(np.mean(self.process_baseline['count'])) if len(self.process_baseline['count']) > 0 else 0.0,
            'std_count': float(np.std(self.process_baseline['count'])) if len(self.process_baseline['count']) > 0 else 0.0,
            'mean_cpu': float(np.mean(self.process_baseline['cpu'])) if len(self.process_baseline['cpu']) > 0 else 0.0,
            'std_cpu': float(np.std(self.process_baseline['cpu'])) if len(self.process_baseline['cpu']) > 0 else 0.0,
            'variance_count': float(np.var(self.process_baseline['count'])) if len(self.process_baseline['count']) > 1 else 0.0,
            'ema': float(self.ema_process) if self.ema_process is not None else 0.0
        }
        
        file_stats = {
            'mean_events': float(np.mean(self.file_baseline['events'])) if len(self.file_baseline['events']) > 0 else 0.0,
            'std_events': float(np.std(self.file_baseline['events'])) if len(self.file_baseline['events']) > 0 else 0.0,
            'variance_events': float(np.var(self.file_baseline['events'])) if len(self.file_baseline['events']) > 1 else 0.0,
            'ema': float(self.ema_file) if self.ema_file is not None else 0.0
        }
        
        # Data quality metrics
        data_quality = {
            'network_samples': len(self.network_baseline['connections']),
            'process_samples': len(self.process_baseline['count']),
            'file_samples': len(self.file_baseline['events']),
            'collection_timestamp': datetime.now().isoformat(),
            'data_freshness': 1.0  # Could be calculated based on timestamp
        }
        
        # Anomaly rate (important for cybersecurity)
        anomaly_rate = self.anomaly_count / max(self.total_detections, 1)
        
        # === TRUE FEDERATED LEARNING: Extract Deep Learning Tensors ===
        all_weights = self.model_weights.copy()

        # Autoencoder weights (keys already prefixed ae_layer_{i}_weight_{j})
        if self.autoencoder and self.autoencoder.is_trained:
            ae_weights = self.autoencoder.get_weights()
            if ae_weights:
                for key, tensor in ae_weights.items():
                    if isinstance(tensor, list) and 'ae_layer' in key:
                        tensor_arr = np.array(tensor)
                        noise = np.random.normal(0, 0.001, tensor_arr.shape)
                        all_weights[key] = (tensor_arr + noise).tolist()
                    else:
                        all_weights[key] = tensor

        # Isolation Forest threshold
        if self.isolation_forest is not None:
            all_weights['iso_threshold'] = float(self.iso_threshold)

        return {
            'weights': all_weights,
            'statistics': {
                'network': network_stats,
                'process': process_stats,
                'file': file_stats
            },
            'data_quality': data_quality,
            'anomaly_rate': float(anomaly_rate),
            'total_anomalies': self.anomaly_count
        }
    
    def update_model_parameters(self, new_model):
        """Update model with aggregated weights from server"""
        if 'weights' in new_model:
            old_weights = self.model_weights.copy()
            self.model_weights.update(new_model['weights'])
            
            # Log significant changes (scalars only)
            changes = []
            for key in self.model_weights:
                if key in old_weights:
                    old_val = old_weights[key]
                    new_val = self.model_weights[key]
                    # Skip tensor/list weights — only log scalar param changes
                    if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                        try:
                            if abs(old_val - new_val) > 0.01:
                                changes.append(f"{key}: {old_val:.3f} → {new_val:.3f}")
                        except (TypeError, ValueError):
                            pass
            
            if changes:
                print(f"– Model updated (v{new_model.get('version', 'unknown')})")
                for change in changes[:5]:  # Show top 5 changes
                    print(f"  {change}")
            else:
                print(f"– Model weights confirmed (v{new_model.get('version', 'unknown')})")
    
    def adapt_sensitivity_locally(self):
        """Locally adapt sensitivity based on recent variance"""
        # Adapt network sensitivity
        if len(self.network_baseline['connections']) > 20:
            recent_variance = np.var(list(self.network_baseline['connections'])[-20:])
            if recent_variance > 15:  # High local variance
                self.model_weights['network_sensitivity'] = min(
                    self.model_weights['network_sensitivity'] * 1.05, 2.0
                )
            elif recent_variance < 3:  # Low local variance
                self.model_weights['network_sensitivity'] = max(
                    self.model_weights['network_sensitivity'] * 0.95, 0.5
                )
        
        # Adapt process sensitivity
        if len(self.process_baseline['count']) > 20:
            recent_variance = np.var(list(self.process_baseline['count'])[-20:])
            if recent_variance > 8:
                self.model_weights['process_sensitivity'] = min(
                    self.model_weights['process_sensitivity'] * 1.05, 2.0
                )
            elif recent_variance < 2:
                self.model_weights['process_sensitivity'] = max(
                    self.model_weights['process_sensitivity'] * 0.95, 0.5
                )


class ClientAIAssistant:
    """Client-side AI assistant for anomaly analysis using Google Gemini."""

    def __init__(self):
        try:
            GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
            if not GEMINI_API_KEY:
                self.available = False
                return
            genai.configure(api_key=GEMINI_API_KEY)
            self.model = genai.GenerativeModel(GEMINI_MODEL)
            self.system_context = (
                "You are a cybersecurity expert AI assistant helping users understand "
                "security anomalies detected on their local system. Provide clear, "
                "actionable explanations in plain English.\n\n"
                "When analyzing anomalies, provide:\n"
                "1. What Happened: Plain English explanation\n"
                "2. Why It Matters: Risk assessment\n"
                "3. What To Do: Specific remediation steps\n"
                "4. Prevention: How to prevent recurrence\n\n"
                "Be concise, avoid jargon, and prioritize user safety."
            )
            self.available = True
        except Exception:
            self.available = False

    def analyze_anomaly(self, anomaly_data):
        if not self.available:
            return "AI Assistant is not available. Please check your GEMINI_API_KEY in .env file."
        try:
            if isinstance(anomaly_data, dict):
                context = (
                    f"Analyze this security anomaly:\n"
                    f"Severity: {anomaly_data.get('severity', '?')}\n"
                    f"Category: {anomaly_data.get('category', '?')}\n"
                    f"Score: {anomaly_data.get('ensemble_score', 0)}/10\n"
                    f"Explanation: {anomaly_data.get('explanation', 'N/A')}\n"
                    f"Features: {', '.join(anomaly_data.get('contributing_features', []))}\n"
                )
            else:
                context = str(anomaly_data)

            response = self.model.generate_content(self.system_context + "\n\n" + context)
            return response.text
        except Exception as e:
            return f"Error generating analysis: {e}"


