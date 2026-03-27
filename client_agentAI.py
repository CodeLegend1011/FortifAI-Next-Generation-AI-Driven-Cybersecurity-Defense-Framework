"""
FortifAI Client Edge Agent - Complete Federated Learning Implementation
Collects cybersecurity telemetry with intelligent filtering and local ML
Cross-platform support for Windows and Linux

"""

import os
import sys
import time
import json
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
import tkinter as tk
from tkinter import ttk, scrolledtext
from threading import Lock
import queue
import sys
import codecs
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Handle AF_LINK constant for different platforms
if hasattr(psutil, 'AF_LINK'):
    AF_LINK = psutil.AF_LINK
elif hasattr(socket, 'AF_PACKET'):
    AF_LINK = socket.AF_PACKET
else:
    AF_LINK = -1

if sys.platform == 'win32':
    # Fix Windows console encoding for Unicode characters
    try:
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')
    except:
        pass  # Ignore if already configured
    
# Configuration
SERVER_HOST = '192.168.56.1'  # Change to server IP
SERVER_PORT = 9999
CLIENT_ID = None
COLLECTION_INTERVAL = 30
HEARTBEAT_INTERVAL = 60

# Federated Learning Configuration
ENABLE_FL = True
FL_UPDATE_INTERVAL = 90  # Send model updates every 5 minutes
COLLECTION_INTERVAL = 30

class EnhancedAnomalyDetector:
    """Enhanced statistical anomaly detection with federated learning"""
    
    def __init__(self,use_ocsvm=False):
        self.network_baseline = {'connections': deque(maxlen=200), 'bytes': deque(maxlen=200)}
        self.process_baseline = {'count': deque(maxlen=200), 'cpu': deque(maxlen=200)}
        self.file_baseline = {'events': deque(maxlen=200)}
        self.anomaly_alerts = deque(maxlen=100)
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
        self.one_class_svm = None if not use_ocsvm else OneClassSVM(kernel='rbf', nu=0.05)
        
        # --- NEW: Thresholds ---
        self.iso_threshold = -0.5  # Lower score = anomaly
        self.ae_threshold = None  # Set after training
        self.zscore_threshold = 8.0
        
        self.adaptive_threshold_multiplier = 1.2
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
            self.model_weights_adaptive = {
                'zscore': 0.2,
                'isolation_forest': 0.4,
                'autoencoder': 0.35,
                'ocsvm': 0.05
            }

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

        for fname, fval in zip(feature_names, feature_vector):
            if abs(fval) > self.zscore_threshold:
                zscore_flags.append(fname)
                zscore_score += min(abs(fval) / self.zscore_threshold, 2.0)

        if len(zscore_flags) > 7:
            model_scores['zscore'] = min(zscore_score / len(zscore_flags), 10.0)
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
                
                if iso_raw < -0.7:  # CALIBRATED threshold
                    model_scores['isolation_forest'] = min(abs(iso_raw) * 12, 10.0)
                    
                    if model_scores['isolation_forest'] > 6.0:
                        top_idx = np.argsort(np.abs(feature_vector))[-5:]
                        anomaly_info['model_contributions']['isolation_forest'] = {
                            'score': model_scores['isolation_forest'],
                            'raw_score': float(iso_raw),
                            'features': [feature_names[i] for i in top_idx]
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
                    if recon > self.autoencoder.threshold * 1.5:  # 1.5x threshold
                        model_scores['autoencoder'] = min(
                            (recon / self.autoencoder.threshold) * 6, 10.0
                        )
                        
                        if model_scores['autoencoder'] > 6.0:
                            feature_errors = self._compute_feature_reconstruction_error(feature_vector)
                            top_idx = np.argsort(feature_errors)[-5:]
                            anomaly_info['model_contributions']['autoencoder'] = {
                                'score': model_scores['autoencoder'],
                                'recon_error': recon,
                                'features': [feature_names[i] for i in top_idx]
                            }
            except Exception as e:
                print(f"[AE] Error: {e}")

        # ============================
        #  MODEL 4 — ONE CLASS SVM
        # ============================
        if self.use_ocsvm and self.one_class_svm is not None:
            try:
                ocsvm_pred = self.one_class_svm.predict([feature_vector])[0]
                if ocsvm_pred == -1:
                    model_scores['ocsvm'] = 7.0
                    anomaly_info['model_contributions']['ocsvm'] = {
                        'score': 7.0,
                        'prediction': 'anomaly'
                    }
            except Exception as e:
                print(f"[OCSVM] Error: {e}")

        # ============================
        #  CONSENSUS FUSION
        # ============================
        ensemble_score = sum(
            model_scores[m] * self.model_weights_adaptive[m]
            for m in model_scores.keys()
        )
        ensemble_score = ensemble_score * 0.8  # Scale down

        # Consensus bonus
        active_models = sum(1 for s in model_scores.values() if s > 6.0)
        if active_models >= 2:
            ensemble_score *= (1 + 0.1 * (active_models - 1))

        ensemble_score = min(ensemble_score, 10.0)
        anomaly_info['ensemble_score'] = float(ensemble_score)

        # ============================
        #  BASE SEVERITY MAPPING
        # ============================
        if ensemble_score >= 9.5:  # INCREASED from 9.0
            base_severity = "critical"
            anomaly_info['is_anomaly'] = True
        elif ensemble_score >= 8.5:  # INCREASED from 7.5
            base_severity = "high"
            anomaly_info['is_anomaly'] = True
        elif ensemble_score >= 7.0:  # INCREASED from 6.0
            base_severity = "medium"
            anomaly_info['is_anomaly'] = True
        elif ensemble_score >= 5.5:  # INCREASED from 4.5
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
            ensemble_score = max(ensemble_score, 9.5)
            anomaly_info['is_anomaly'] = True

        if feature_dict.get("c2_pattern", 0) >= 1.0:
            if base_severity != "critical":
                base_severity = "high"
            ensemble_score = max(ensemble_score, 8.0)
            anomaly_info['is_anomaly'] = True

        if feature_dict.get("cred_dump_pattern", 0) >= 1.0:
            if base_severity != "critical":
                base_severity = "high"
            ensemble_score = max(ensemble_score, 7.5)
            anomaly_info['is_anomaly'] = True

        anomaly_info["severity"] = base_severity
        anomaly_info["ensemble_score"] = float(ensemble_score)

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
            ttl_seconds = {
                'critical': 300,  # 5 min
                'high': 180,      # 3 min
                'medium': 120,    # 2 min
                'low': 60         # 1 min
            }.get(severity, 120)
            
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

        # Determine dominant category
        primary_category = max(category_scores, key=category_scores.get)
        if category_scores[primary_category] == 0:
            primary_category = 'system'

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
        if primary_category == 'network':
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
        
        elif primary_category == 'process':
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
        
        elif primary_category == 'filesystem':
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
        # ✓ FINAL ALERT OBJECT (HUMAN-READABLE)
        # ==========================================
        alert = {
            'timestamp': datetime.now().isoformat(),
            'hostname': socket.gethostname(),
            'client_id': CLIENT_ID,
            'category': primary_category,
            'severity': severity,
            'ensemble_score': score,
            'contributing_features': contrib[:5],
            'feature_groups': {k: v['risk'] for k, v in feature_groups.items()},
            'model_contributions': anomaly_info.get('model_contributions', {}),
            'explanation': " | ".join(explanation_parts),
            'evidence': " | ".join(evidence_parts) if evidence_parts else "Multiple anomalous behaviors",
            'recommended_action': recommended_action,
            'signature': alert_signature,
            'ttl_expires': self.alert_signatures[alert_signature]
        }

        self.anomaly_alerts.append(alert)
        
        # ✓ ENHANCED CONSOLE OUTPUT
        print(f"\n{'='*80}")
        print(f"⚠️  ANOMALY DETECTED - {severity.upper()} - {primary_category.upper()}")
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

    def _get_recommended_action(self, severity, feature_groups, contrib_features, category):
        """Suggest SPECIFIC remediation action based on category and severity"""
        
        # Extract top risk groups
        top_groups = sorted(
            [(k, v['risk']) for k, v in feature_groups.items() if v['risk'] > 0],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
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
    
    def _compute_feature_reconstruction_error(self, feature_vector):
        """Compute per-feature reconstruction error for explainability"""
        try:
            # ✅ FIX: Check if scaler is fitted before using
            if not hasattr(self.autoencoder.scaler, 'mean_'):
                print("  [AE] Warning: Scaler not fitted in reconstruction error computation")
                return np.zeros(len(feature_vector))
            
            X_scaled = self.autoencoder.scaler.transform([feature_vector])
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
        CLIP_BOUND = 10.0
        
        # Compute delta: local - global
        model_delta = {}
        for key in self.model_weights.keys():
            if key in last_global_weights:
                delta = float(self.model_weights[key]) - float(last_global_weights[key])
                model_delta[key] = delta
        
        # Compute L2 norm
        delta_norm = 0.0
        for key, value in model_delta.items():
            if isinstance(value, (int, float)):
                delta_norm += value ** 2
        delta_norm = np.sqrt(delta_norm)
        
        # L2 clipping
        if delta_norm > CLIP_BOUND:
            scale_factor = CLIP_BOUND / delta_norm
            for key in model_delta.keys():
                if isinstance(model_delta[key], (int, float)):
                    model_delta[key] *= scale_factor
            final_norm = CLIP_BOUND
            was_clipped = True
        else:
            final_norm = delta_norm
            was_clipped = False
        
        # === ENHANCED METADATA ===
        # Isolation Forest score distribution (if available)
        iso_scores = []
        if self.isolation_forest is not None:
            feature_matrix, _ = self.feature_manager.get_feature_matrix()
            if feature_matrix is not None and len(feature_matrix) > 0:
                try:
                    iso_scores = self.isolation_forest.score_samples(feature_matrix[-50:])
                    iso_scores = [float(s) for s in iso_scores]
                except:
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
        if feature_matrix is None or len(feature_matrix) < 20:
            print(f"  [ML] Insufficient data ({len(feature_matrix) if feature_matrix is not None else 0}/20)")
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
            
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            mask = np.all((feature_matrix >= lower_bound) & (feature_matrix <= upper_bound), axis=1)
            cleaned_matrix = feature_matrix[mask]
            
            removed_count = len(feature_matrix) - len(cleaned_matrix)
            if removed_count > 0:
                print(f"  [CLEAN] Removed {removed_count} extreme outliers ({removed_count/len(feature_matrix)*100:.1f}%)")
            
            if len(cleaned_matrix) < 15:
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
                contamination=0.05,  
                n_estimators=200,   
                max_samples=min(256, len(feature_matrix)),
                random_state=42,
                bootstrap=True
            )
            self.isolation_forest.fit(feature_matrix)
            
            # Calibrate threshold
            scores = self.isolation_forest.score_samples(feature_matrix)
            self.iso_threshold = np.percentile(scores, 10)  
            
            print(f"  [ISO] ✓ Trained | Threshold={self.iso_threshold:.4f}")
            training_success = True
        except Exception as e:
            print(f"  [ISO] ✗ Failed: {e}")
        
        # ✅ FIX 2: Train Autoencoder with GUARANTEED success
        try:
            print("  [AE] Training Autoencoder...")
            if self.autoencoder is None:
                self.autoencoder = AutoencoderAnomalyDetector(input_dim=feature_matrix.shape[1])
            
            ae_success = self.autoencoder.train(feature_matrix, epochs=10, batch_size=16)
            
            if ae_success:
                # Recompute threshold
                X_scaled = self.autoencoder.scaler.transform(feature_matrix)
                X_pred = self.autoencoder.model.predict(X_scaled, verbose=0)
                recon_errors = np.mean(np.square(X_scaled - X_pred), axis=1)
                
                self.autoencoder.threshold = np.percentile(recon_errors, 90)
                self.ae_threshold = self.autoencoder.threshold
                
                print(f"  [AE] ✓ Trained | Threshold={self.ae_threshold:.6f}")
                training_success = True
            else:
                print("  [AE] ✗ Training returned False")
        except Exception as e:
            print(f"  [AE] ✗ Failed: {e}")
        
        if training_success:
            self.last_training_time = time.time()
            print(f"[ML TRAINING] ✓ Complete")
        
        return training_success
    
    def should_retrain(self, buffer_size):
        """Check if models should be retrained - OPTIMIZED FREQUENCY"""
        if self.last_training_time is None:
            return buffer_size >= 20  # Initial training threshold
        
        time_since_training = time.time() - self.last_training_time
        
        # Retrain only if:
        # 1. 15+ minutes passed AND buffer >= 150 samples (substantial new data)
        # 2. 60+ minutes passed (periodic refresh regardless of buffer)
        return (time_since_training >= 900 and buffer_size >= 150) or \
            (time_since_training >= 3600)
    
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
        
        if len(baseline) < 10:
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
        
        return {
            'weights': self.model_weights.copy(),
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
            
            # Log significant changes
            changes = []
            for key in self.model_weights:
                if key in old_weights:
                    old_val = old_weights[key]
                    new_val = self.model_weights[key]
                    if abs(old_val - new_val) > 0.01:
                        changes.append(f"{key}: {old_val:.3f} → {new_val:.3f}")
            
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

# --- NEW: Feature extraction and sliding window management ---
class FeatureWindowManager:
    """Manages sliding window feature extraction for ML models"""
        # ========= FIXED FEATURE SCHEMA (MUST MATCH MODEL TRAINING) =========
    FEATURE_SCHEMA = [
        # Network
        'conn_count',
        'unique_dst_count',
        'bytes_sent',
        'bytes_recv',
        'port_entropy',
        'tcp_ratio',
        'udp_ratio',
        'conn_rate',
        'dst_churn_rate',

        # Process
        'proc_spawn_count',
        'avg_proc_cpu',
        'avg_proc_memory',
        'unique_proc_names',

        # Filesystem
        'file_create_count',
        'file_exec_count',
        'file_hash_novelty',

        # Rate-of-change
        'delta_conn_count',
        'delta_file_create',
        'delta_proc_spawn',
        'delta_unique_hashes',
        'conn_acceleration',

        # Burst / Pattern flags
        'ransomware_burst',
        'c2_pattern',
        'cred_dump_pattern',

        # Raw numeric counters (needed for next-window deltas)
        'conn_count_raw',
        'proc_spawn_raw',
        'file_create_raw',
        'unique_hashes_raw'
    ]

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
        
        # Current window accumulator
        self.current_window = {
            'start_time': time.time(),
            'network': {'conn_count': 0, 'unique_dst': set(), 'bytes_sent': 0, 
                       'bytes_recv': 0, 'ports': [], 'protocols': []},
            'process': {'spawn_count': 0, 'cpu_samples': [], 'memory_samples': [], 
                       'proc_names': set()},
            'filesystem': {'file_create': 0, 'file_exec': 0, 'file_hashes': set()}
        }
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
            if event.get('file_extension') in ['.exe', '.dll', '.so']:
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
                fs['file_create'] > 50 and
                len(fs['file_hashes']) > 20 and
                elapsed < 30
            )
            
            c2_pattern = 0.0
            if len(net['ports']) > 10:
                from collections import Counter
                dst_counts = Counter(str(dst) for dst in net['unique_dst'])
                max_dst_count = max(dst_counts.values()) if dst_counts else 0
                c2_pattern = float(max_dst_count > 5 and len(net['unique_dst']) < 3)
            
            cred_dump_pattern = float(
                (any('lsass' in str(p).lower() for p in proc['proc_names']) or
                any('sam' in str(p).lower() for p in proc['proc_names']))
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
            
            # ✅ CRITICAL: Validate 28 features present
            feature_count = len([k for k in raw_features.keys() if k != 'timestamp'])
            assert feature_count == 28, f"Feature count mismatch: got {feature_count}, expected 28"
            
            # Normalize features
            normalized_features = self.normalize_features(raw_features)
            self.feature_buffer.append(normalized_features)
            
            # High-confidence alerts
            if ransomware_burst or c2_pattern or cred_dump_pattern:
                self._trigger_immediate_alert(raw_features, normalized_features)
            
            # Reset window
            self.current_window = {
                'start_time': time.time(),
                'network': {'conn_count': 0, 'unique_dst': set(), 'bytes_sent': 0,
                            'bytes_recv': 0, 'ports': [], 'protocols': []},
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
                    normalized[key] = (value - median) / (1.4826 * mad)
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
                normalized[key] = np.clip(value, -3, 3)
        
        # ✅ ADD: Cache result
        if hasattr(self, 'anomaly_detector'):
            self.anomaly_detector.perf_guard.cache_normalization(
                feature_key, normalized
            )
            
        return normalized
    
    def get_feature_matrix(self):
        """Return feature matrix with GUARANTEED 28-feature consistency"""
        if len(self.feature_buffer) == 0:
            return None, None

        # ✅ EXACT 28-feature ordering
        feature_names = self.FEATURE_SCHEMA[:]  # Network(9) + Process(4) + File(3) + Delta(5) + Flags(3) + Raw(4)
        
        # ✅ CRITICAL: Exclude raw counters and timestamp from ML input
        ml_feature_names = [f for f in feature_names if not f.endswith('_raw') and f != 'timestamp']
        
        # Build matrix
        matrix = []
        for window in self.feature_buffer:
            row = []
            for feature_name in ml_feature_names:
                value = window.get(feature_name, 0.0)
                row.append(float(value))
            matrix.append(row)
        
        matrix = np.array(matrix)
        
        # ✅ VALIDATION: Ensure exactly 24 ML features (28 - 4 raw counters)
        expected_ml_features = 24
        actual_features = matrix.shape[1] if len(matrix.shape) > 1 else 0
        
        if actual_features != expected_ml_features:
            print(f"  [FEATURE MATRIX] ❌ Dimension mismatch: got {actual_features}, expected {expected_ml_features}")
            print(f"  [FEATURE MATRIX] Missing features: {set(ml_feature_names) - set(feature_names[:actual_features])}")
            return None, None
        
        print(f"  [FEATURE MATRIX] ✓ Shape: {matrix.shape}, Features: {len(ml_feature_names)}")
        print(f"  [FEATURE MATRIX] Sample row: {matrix[0][:5]}...")
        
        return matrix, ml_feature_names

    
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
        
        # Rate limits
        self.MAX_ML_OPS_PER_MINUTE = 30
        self.MAX_ALERTS_PER_MINUTE = 50
        self.MIN_TRAINING_INTERVAL = 900  # 15 minutes
        
        # Feature normalization cache
        self.norm_cache = {}
        self.norm_cache_ttl = {}
        self.CACHE_TTL = 60  # 60 seconds
        
        # Locks
        self.training_lock = threading.Lock()
        self.cache_lock = threading.Lock()
    
    def can_perform_ml_operation(self):
        """Check if ML operation is allowed (rate limiting)"""
        now = time.time()
        
        # Remove timestamps older than 1 minute
        cutoff = now - 60
        while self.ml_operation_timestamps and self.ml_operation_timestamps[0] < cutoff:
            self.ml_operation_timestamps.popleft()
        
        if len(self.ml_operation_timestamps) >= self.MAX_ML_OPS_PER_MINUTE:
            return False
        
        self.ml_operation_timestamps.append(now)
        return True
    
    def can_generate_alert(self):
        """Check if alert generation is allowed (prevent burst)"""
        now = time.time()
        
        cutoff = now - 60
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
        acquired = self.training_lock.acquire(blocking=True, timeout=5)
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

# --- NEW: Autoencoder for anomaly detection ---
class AutoencoderAnomalyDetector:
    """Deep autoencoder for detecting anomalies via reconstruction error"""
    
    def __init__(self, input_dim=14, latent_dim=8):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.model = None
        self.threshold = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        self.build_model()
    
    def build_model(self):
        """Build autoencoder architecture"""
        # Encoder
        input_layer = keras.Input(shape=(self.input_dim,))
        encoded = layers.Dense(64, activation='relu')(input_layer)
        encoded = layers.Dense(32, activation='relu')(encoded)
        latent = layers.Dense(self.latent_dim, activation='relu', name='latent')(encoded)
        
        # Decoder
        decoded = layers.Dense(32, activation='relu')(latent)
        decoded = layers.Dense(64, activation='relu')(decoded)
        output = layers.Dense(self.input_dim, activation='linear')(decoded)
        
        # Compile model
        self.model = keras.Model(inputs=input_layer, outputs=output)
        self.model.compile(optimizer='adam', loss='mse')
    
    def train(self, X, epochs=10, batch_size=16):
        """Train autoencoder on normal data - FIXED"""
        if len(X) < 20:
            print("  [AE] Insufficient data for training (need >= 20 samples)")
            return False
        
        try:
            # ✅ FIX: Clean input data
            X_clean = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
            
            # ✅ FIX: Fit scaler with validation
            print(f"  [AE] Fitting scaler on {len(X_clean)} samples...")
            self.scaler.fit(X_clean)
            
            # Verify scaler was fitted
            if not hasattr(self.scaler, 'mean_') or not hasattr(self.scaler, 'scale_'):
                print("  [AE] ✗ Scaler fitting failed")
                return False
            
            X_scaled = self.scaler.transform(X_clean)
            
            print(f"  [AE] Scaler fitted: mean={self.scaler.mean_[:3]}, scale={self.scaler.scale_[:3]}")
            
            # ✅ FIX: Train with early stopping
            from tensorflow.keras.callbacks import EarlyStopping
            early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            
            history = self.model.fit(
                X_scaled, X_scaled,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                validation_split=0.2,
                callbacks=[early_stop]
            )
            
            # Compute reconstruction errors
            X_pred = self.model.predict(X_scaled, verbose=0)
            recon_errors = np.mean(np.square(X_scaled - X_pred), axis=1)
            
            # ✅ FIX: Set threshold at 95th percentile
            self.threshold = np.percentile(recon_errors, 98)
            self.is_trained = True
            
            final_loss = history.history['loss'][-1]
            val_loss = history.history['val_loss'][-1] if 'val_loss' in history.history else final_loss
            print(f"  [AE] Training complete: loss={final_loss:.6f}, val_loss={val_loss:.6f}, threshold={self.threshold:.6f}")
            
            # ✅ VERIFY: Double-check scaler parameters
            assert hasattr(self.scaler, 'mean_'), "Scaler mean_ missing after training"
            assert hasattr(self.scaler, 'scale_'), "Scaler scale_ missing after training"
            
            return True
        
        except Exception as e:
            print(f"  [AE] Training error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict(self, X):
        """Predict reconstruction error for samples"""
        if not self.is_trained:
            return None
        
        try:
            # ✅ FIX: Check if scaler is fitted
            if not hasattr(self.scaler, 'mean_') or not hasattr(self.scaler, 'scale_'):
                print("  [AE] Warning: Scaler not fitted, fitting now...")
                self.scaler.fit(X)
            
            X_scaled = self.scaler.transform(X)
            X_pred = self.model.predict(X_scaled, verbose=0)
            recon_errors = np.mean(np.square(X_scaled - X_pred), axis=1)
            return recon_errors
        except Exception as e:
            print(f"  [AE] Prediction error: {e}")
            return None
    
    def detect_anomaly(self, X):
        """Detect if samples are anomalies"""
        recon_errors = self.predict(X)
        if recon_errors is None:
            return None, None
        
        # ✅ FIX: Handle case where threshold is not set
        if self.threshold is None:
            print("  [AE] Warning: Threshold not set, using 95th percentile of current errors")
            self.threshold = np.percentile(recon_errors, 95)
        
        is_anomaly = recon_errors > self.threshold
        return is_anomaly, recon_errors
    
    def get_weights(self):
        """Extract model weights for federated learning"""
        if self.model is None:
            return None
        
        weights_dict = {}
        
        # Extract layer weights
        for i, layer in enumerate(self.model.layers):
            layer_weights = layer.get_weights()
            if len(layer_weights) > 0:
                weights_dict[f'layer_{i}_kernel'] = layer_weights[0].tolist()
                if len(layer_weights) > 1:
                    weights_dict[f'layer_{i}_bias'] = layer_weights[1].tolist()
        
        # ✅ FIX: Include scaler parameters
        if hasattr(self.scaler, 'mean_') and hasattr(self.scaler, 'scale_'):
            weights_dict['scaler_mean'] = self.scaler.mean_.tolist()
            weights_dict['scaler_scale'] = self.scaler.scale_.tolist()
        
        # ✅ FIX: Include threshold
        if self.threshold is not None:
            weights_dict['threshold'] = float(self.threshold)
        
        # ✅ FIX: Include input dimension
        weights_dict['input_dim'] = self.input_dim
        
        return weights_dict
    
    def set_weights(self, weights_dict):
        """Set model weights from federated learning"""
        try:
            # Restore model weights
            for i, layer in enumerate(self.model.layers):
                kernel_key = f'layer_{i}_kernel'
                bias_key = f'layer_{i}_bias'
                
                if kernel_key in weights_dict:
                    kernel = np.array(weights_dict[kernel_key])
                    if bias_key in weights_dict:
                        bias = np.array(weights_dict[bias_key])
                        layer.set_weights([kernel, bias])
                    else:
                        layer.set_weights([kernel])
            
            # ✅ FIX: Restore scaler parameters if available
            if 'scaler_mean' in weights_dict and 'scaler_scale' in weights_dict:
                self.scaler.mean_ = np.array(weights_dict['scaler_mean'])
                self.scaler.scale_ = np.array(weights_dict['scaler_scale'])
                self.scaler.n_features_in_ = len(self.scaler.mean_)
                self.scaler.n_samples_seen_ = 100  # Dummy value
                print("  [AE] ✓ Scaler parameters restored")
            else:
                print("  [AE] ⚠️ Scaler parameters not in weights_dict")
            
            # ✅ FIX: Restore threshold
            if 'threshold' in weights_dict:
                self.threshold = float(weights_dict['threshold'])
                print(f"  [AE] ✓ Threshold restored: {self.threshold:.6f}")
            
            # Mark as trained
            self.is_trained = True
            print("  [AE] ✓ Weights updated from global model")
            
        except Exception as e:
            print(f"  [AE] ✗ Error setting weights: {e}")
            import traceback
            traceback.print_exc()
            
class SystemInfo:
    """Collect system information"""
    
    @staticmethod
    def get_client_id():
        """Generate unique client ID based on hardware"""
        hostname = socket.gethostname()
        
        mac_address = "000000000000"
        try:
            net_if_addrs = psutil.net_if_addrs()
            for iface_name, addr_list in net_if_addrs.items():
                for addr in addr_list:
                    if (hasattr(addr, 'family') and 
                        (addr.family == AF_LINK or 
                         (hasattr(socket, 'AF_PACKET') and addr.family == socket.AF_PACKET))):
                        mac_raw = addr.address.replace(':', '').replace('-', '').upper()
                        if mac_raw and mac_raw != '000000000000' and len(mac_raw) == 12:
                            mac_address = mac_raw
                            break
                    elif hasattr(addr, 'address') and addr.address and (':' in addr.address or '-' in addr.address):
                        mac_raw = addr.address.replace(':', '').replace('-', '').upper()
                        if len(mac_raw) == 12 and mac_raw != '000000000000':
                            mac_address = mac_raw
                            break
                if mac_address != "000000000000":
                    break
        except Exception as e:
            print(f"Warning: Could not get MAC address: {e}")
            import random
            mac_address = hashlib.md5(f"{hostname}{random.random()}".encode()).hexdigest()[:12]
        
        unique_str = f"{hostname}_{mac_address}_{platform.system()}"
        return hashlib.sha256(unique_str.encode()).hexdigest()[:16]
    
    @staticmethod
    def get_system_info():
        """Get basic system information"""
        try:
            ip_address = socket.gethostbyname(socket.gethostname())
        except:
            ip_address = '127.0.0.1'
        
        return {
            'client_id': CLIENT_ID,
            'hostname': socket.gethostname(),
            'ip_address': ip_address,
            'os_type': platform.system(),
            'os_version': platform.version(),
            'device_role': 'workstation',
            'department': 'IT',
            'criticality_level': 'medium'
        }

class EnhancedNetworkCollector:
    """Enhanced network telemetry with comprehensive threat detection"""
    
    def __init__(self, anomaly_detector):
        self.last_connections = {}
        self.dns_cache = {}
        self.connection_history = defaultdict(lambda: {
            'count': 0, 'last_seen': None, 'first_seen': None,
            'bytes_sent': 0, 'bytes_recv': 0, 'ports_used': set()
        })
        self.anomaly_detector = anomaly_detector
        self.collection_count = 0
        
        # Connection rate tracking
        self.connection_timestamps = deque(maxlen=500)
        self.failed_connections = defaultdict(int)
        
        # ========== THREAT INTELLIGENCE ==========
        # Known malicious ports (C2, backdoors, etc.)
        self.malicious_ports = {
            4444, 5555, 6666, 7777, 8888,  # Common RAT/C2 ports
            31337, 12345, 1337, 1234,  # Classic backdoor ports
            6667, 6668, 6669, 6697,  # IRC (often used by botnets)
            9001, 9030, 9050, 9150, 9051,  # Tor
            1080, 1081,  # SOCKS proxy
            3128, 8080, 8118,  # Common proxy ports (suspicious if unexpected)
            5900, 5901, 5902,  # VNC (suspicious if unexpected)
            2323,  # Alternative Telnet
            4443, 8443,  # Alternative HTTPS (C2)
            9999, 10000,  # Common malware ports
            20, 21,  # FTP (unencrypted)
            23,  # Telnet (unencrypted)
            25, 465, 587,  # SMTP (data exfiltration)
            110, 995, 143, 993,  # Email (data exfiltration)
            137, 138, 139, 445,  # SMB (lateral movement)
            1433, 1434,  # MSSQL
            3306,  # MySQL
            5432,  # PostgreSQL
            27017, 27018,  # MongoDB
            6379,  # Redis
            11211,  # Memcached
        }
        
        # Suspicious port ranges
        self.suspicious_port_ranges = [
            (1024, 1100),  # Often used by malware
            (4440, 4450),  # Metasploit range
            (5550, 5560),  # RAT range
            (6660, 6670),  # IRC range
            (31330, 31340),  # Backdoor range
        ]
        
        # Known legitimate high-traffic destinations
        self.safe_destinations = {
                # Microsoft
                '*.microsoft.com', '*.windows.com', '*.windowsupdate.com', '*.live.com',
                '*.azure.com', '*.msedge.net', '*.office.com', '*.office365.com',
                '*.msftconnecttest.com', '*.msftncsi.com',  # Windows network connectivity tests
                
                # Google
                '*.google.com', '*.googleapis.com', '*.gstatic.com', '*.googlevideo.com',
                '*.youtube.com', '*.ytimg.com', '*.ggpht.com', '*.googleusercontent.com',
                
                # Amazon/AWS
                '*.amazon.com', '*.amazonaws.com', '*.cloudfront.net', '*.awsstatic.com',
                
                # Cloudflare
                '*.cloudflare.com', '*.cloudflare-dns.com', '*.cloudflareinsights.com',
                
                # Apple
                '*.apple.com', '*.icloud.com', '*.cdn-apple.com',
                
                # Facebook/Meta
                '*.facebook.com', '*.fbcdn.net', '*.instagram.com', '*.whatsapp.com',
                
                # CDNs
                '*.akamai.net', '*.akamaized.net', '*.akadns.net', '*.fastly.net',
                
                # Developer tools
                '*.github.com', '*.githubusercontent.com', '*.npmjs.org', '*.pypi.org',
                
                # Communication
                '*.slack.com', '*.zoom.us', '*.teams.microsoft.com', '*.discord.com',
                
                # ✅ NEW: Add your frequently visited domains
                '*.stackoverflow.com', '*.reddit.com', '*.twitter.com',
                '*.linkedin.com', '*.medium.com', '*.wikipedia.org',
            }
        
        self.safe_ip_ranges = [
            '192.168.',  # Private network
            '10.',       # Private network
            '172.16.', '172.17.', '172.18.', '172.19.',  # Private network
            '127.',      # Loopback
            '169.254.',  # Link-local
        ]
        
        # Protocol mapping
        self.port_protocols = {
            20: 'FTP-Data', 21: 'FTP', 22: 'SSH', 23: 'Telnet',
            25: 'SMTP', 53: 'DNS', 67: 'DHCP', 68: 'DHCP',
            80: 'HTTP', 110: 'POP3', 119: 'NNTP', 123: 'NTP',
            143: 'IMAP', 161: 'SNMP', 162: 'SNMP-Trap',
            389: 'LDAP', 443: 'HTTPS', 445: 'SMB',
            465: 'SMTPS', 587: 'SMTP-Sub', 636: 'LDAPS',
            993: 'IMAPS', 995: 'POP3S',
            1433: 'MSSQL', 1434: 'MSSQL-Browser',
            3306: 'MySQL', 3389: 'RDP', 5432: 'PostgreSQL',
            5900: 'VNC', 6379: 'Redis', 8080: 'HTTP-Proxy',
            8443: 'HTTPS-Alt', 9050: 'Tor-SOCKS', 27017: 'MongoDB'
        }
    
    def compute_port_entropy(self, ports):
        """Calculate Shannon entropy of port distribution"""
        if not ports:
            return 0.0
        from collections import Counter
        counts = Counter(ports)
        total = len(ports)
        entropy = -sum((count/total) * np.log2(count/total) for count in counts.values())
        return entropy

    def collect(self):
        """Collect network data with ML-DRIVEN threat detection"""
        network_data = []
        significant_events = []
        current_time = datetime.now()
        
        try:
            connections = psutil.net_connections(kind='inet')
            active_connections = 0
            unique_destinations = set()
            protocol_counts = defaultdict(int)
            ports_used = []
            
            net_io = psutil.net_io_counters()
            current_bytes = net_io.bytes_sent + net_io.bytes_recv if net_io else 0
            
            # Metrics for ML feature extraction
            high_risk_port_count = 0
            external_ip_count = 0
            new_connection_count = 0
            
            for conn in connections:
                if conn.status == 'ESTABLISHED':
                    active_connections += 1
                    remote_addr = conn.raddr if conn.raddr else None
                    
                    if not remote_addr:
                        continue
                    
                    self.feature_manager.update_network_event({
                        'dst_ip': remote_addr.ip,
                        'src_port': conn.laddr.port,
                        'protocol': 'TCP' if conn.type == socket.SOCK_STREAM else 'UDP',
                        'bytes_sent': 0,  # Placeholder
                        'bytes_recv': 0   # Placeholder
                    })
                    
                    remote_ip = remote_addr.ip
                    remote_port = remote_addr.port
                    local_port = conn.laddr.port
                    
                    # ✅ SKIP SERVER AND LOCALHOST EARLY
                    if remote_ip == SERVER_HOST and remote_port == SERVER_PORT:
                        continue
                    if remote_ip in ['127.0.0.1', '::1', 'localhost']:
                        continue
                    
                    unique_destinations.add(remote_ip)
                    ports_used.append(remote_port)
                    
                    # Track connection
                    conn_key = f"{remote_ip}:{remote_port}"
                    history = self.connection_history[conn_key]
                    history['count'] += 1
                    if history['first_seen'] is None:
                        history['first_seen'] = current_time
                        new_connection_count += 1
                    history['last_seen'] = current_time
                    history['ports_used'].add(local_port)
                    
                    protocol = self._identify_protocol(conn, remote_port)
                    protocol_counts[protocol] += 1
                    
                    if remote_port in self.malicious_ports:
                        high_risk_port_count += 1
                    if not self._is_local_ip(remote_ip):
                        external_ip_count += 1
                    
                    dns_name = self.resolve_dns(remote_ip)
                    
                    # ═══════════════════════════════════════════════════════════════
                    # ✅ BUILD COMPLETE 24-FEATURE VECTOR (MATCHING FEATURE_SCHEMA)
                    # ═══════════════════════════════════════════════════════════════

                    # Get window stats for context
                    window_stats = self.feature_manager.get_window_stats()

                    # Network features (9 features)
                    conn_count_scaled = float(active_connections) / 100.0
                    unique_dst_ratio = float(len(unique_destinations)) / max(active_connections, 1)
                    bytes_sent_scaled = 0.0  # Not tracked per-connection in this collector
                    bytes_recv_scaled = 0.0  # Not tracked per-connection in this collector
                    port_entropy_val = self.compute_port_entropy(ports_used) if len(ports_used) > 1 else 0.0
                    tcp_ratio = float(protocol_counts.get('TCP', 0)) / max(active_connections, 1)
                    udp_ratio = float(protocol_counts.get('UDP', 0)) / max(active_connections, 1)
                    conn_rate = float(active_connections) / 60.0  # Per minute
                    dst_churn_rate = float(len(unique_destinations)) / 60.0

                    # Process features (4 features) - use aggregated stats
                    proc_spawn_count = float(window_stats.get('process_events', 0)) / 10.0
                    avg_proc_cpu = 0.0  # Not available in network collector
                    avg_proc_memory = 0.0  # Not available in network collector
                    unique_proc_names = 0.0  # Not available in network collector

                    # Filesystem features (3 features) - use aggregated stats
                    file_create_count = float(window_stats.get('file_events', 0)) / 50.0
                    file_exec_count = 0.0  # Not available in network collector
                    file_hash_novelty = 0.0  # Not available in network collector

                    # Rate of change features (5 features) - approximate
                    delta_conn_count = float(new_connection_count) / 10.0
                    delta_file_create = 0.0
                    delta_proc_spawn = 0.0
                    delta_unique_hashes = 0.0
                    conn_acceleration = 0.0

                    # Pattern flags (3 features)
                    ransomware_burst = 0.0
                    c2_pattern = 1.0 if (history['count'] > 10 and len(history['ports_used']) < 3) else 0.0
                    cred_dump_pattern = 0.0

                    # ✅ ASSEMBLE 24-FEATURE VECTOR IN EXACT SCHEMA ORDER
                    feature_vector = np.array([
                        # Network (9)
                        conn_count_scaled,
                        unique_dst_ratio,
                        bytes_sent_scaled,
                        bytes_recv_scaled,
                        port_entropy_val,
                        tcp_ratio,
                        udp_ratio,
                        conn_rate,
                        dst_churn_rate,
                        
                        # Process (4)
                        proc_spawn_count,
                        avg_proc_cpu,
                        avg_proc_memory,
                        unique_proc_names,
                        
                        # Filesystem (3)
                        file_create_count,
                        file_exec_count,
                        file_hash_novelty,
                        
                        # Rate of change (5)
                        delta_conn_count,
                        delta_file_create,
                        delta_proc_spawn,
                        delta_unique_hashes,
                        conn_acceleration,
                        
                        # Pattern flags (3)
                        ransomware_burst,
                        c2_pattern,
                        cred_dump_pattern
                    ])

                    # ✅ FEATURE NAMES (24 total, matching FeatureWindowManager.FEATURE_SCHEMA)
                    feature_names = [
                        # Network
                        'conn_count', 'unique_dst_count', 'bytes_sent', 'bytes_recv',
                        'port_entropy', 'tcp_ratio', 'udp_ratio', 'conn_rate', 'dst_churn_rate',
                        # Process
                        'proc_spawn_count', 'avg_proc_cpu', 'avg_proc_memory', 'unique_proc_names',
                        # Filesystem
                        'file_create_count', 'file_exec_count', 'file_hash_novelty',
                        # Rate of change
                        'delta_conn_count', 'delta_file_create', 'delta_proc_spawn',
                        'delta_unique_hashes', 'conn_acceleration',
                        # Pattern flags
                        'ransomware_burst', 'c2_pattern', 'cred_dump_pattern'
                    ]

                    # ✅ VALIDATION: Ensure 24 features
                    assert len(feature_vector) == 24, f"Network collector feature mismatch: {len(feature_vector)} != 24"
                    assert len(feature_names) == 24, f"Network collector name mismatch: {len(feature_names)} != 24"
                    
                    # ╔═══════════════════════════════════════════════════════════╗
                    # ║ ✅ STEP 1: ML ENSEMBLE DETECTION (PRIMARY)
                    # ╚═══════════════════════════════════════════════════════════╝
                    
                    ml_is_anomaly = False
                    ml_risk_score = 0.0
                    ml_indicators = []
                    
                    if (self.anomaly_detector.isolation_forest is not None or 
                        (self.anomaly_detector.autoencoder and self.anomaly_detector.autoencoder.is_trained)):
                        
                        try:
                            # ✅ CRITICAL: Actually call the ensemble detector
                            is_anomaly, anomaly_info = self.anomaly_detector.detect_anomaly_ensemble(
                                feature_vector,
                                feature_names
                            )
                            
                            if is_anomaly:
                                ml_is_anomaly = True
                                severity_map = {'high': 9.0, 'medium': 6.5, 'low': 4.5}
                                ml_risk_score = severity_map.get(anomaly_info['severity'], 5.0)
                                ml_indicators = [f"ml_{ind}" for ind in anomaly_info.get('contributing_features', [])[:3]]
                                
                                # ✅ FIX: Log detection for debugging
                                print(f"  [ML NETWORK] Anomaly detected: {remote_ip}:{remote_port} "
                                      f"(severity={anomaly_info['severity']}, score={ml_risk_score:.1f})")
                        
                        except Exception as e:
                            print(f"  [ML] Network detection error: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    # ═══════════════════════════════════════════════════════
                    # ✅ STEP 2: RULE-BASED (ONLY IF ML DIDN'T DETECT)
                    # ═══════════════════════════════════════════════════════
                    rule_risk_score = 0.0
                    rule_indicators = []
                    
                    if not ml_is_anomaly:  # ✅ Only evaluate rules if ML didn't flag
                        # Rule 1: Known malicious ports to external IPs
                        if remote_port in self.malicious_ports and not self._is_local_ip(remote_ip):
                            rule_risk_score += 7.0
                            rule_indicators.append(f'malicious_port:{remote_port}')
                        
                        # Rule 2: No DNS + Low port (privileged service)
                        if not dns_name and not self._is_local_ip(remote_ip) and remote_port < 1024:
                            rule_risk_score += 4.0
                            rule_indicators.append('suspicious_no_dns_privileged_port')
                        
                        # Rule 3: Very high connection frequency
                        if history['count'] > 1000:
                            rule_risk_score += 3.0
                            rule_indicators.append(f'extreme_frequency:{history["count"]}')
                        
                        # Rule 4: Multiple local ports to same destination (port scanning)
                        if len(history['ports_used']) > 50:
                            rule_risk_score += 5.0
                            rule_indicators.append(f'port_scanning:{len(history["ports_used"])}')
                    
                    # ═══════════════════════════════════════════════════════
                    # ✅ STEP 3: COMBINE DETECTIONS (ML PRIORITY)
                    # ═══════════════════════════════════════════════════════
                    should_report = False
                    final_risk_score = 0.0
                    detection_method = 'baseline'
                    threat_indicators = []
                    
                    if ml_is_anomaly:
                        # ML detected anomaly - HIGHEST PRIORITY
                        final_risk_score = ml_risk_score
                        threat_indicators = ml_indicators
                        detection_method = 'ml'
                        should_report = True
                        
                    elif rule_risk_score > 6.0:
                        # Rule-based detection - MEDIUM PRIORITY
                        final_risk_score = rule_risk_score
                        threat_indicators = rule_indicators
                        detection_method = 'rule'
                        should_report = True
                        
                    else:
                        # Baseline monitoring - LOW PRIORITY
                        if history['count'] == 1 and not self._is_local_ip(remote_ip):
                            # New external connection
                            final_risk_score = 2.0
                            threat_indicators = ['new_external_connection']
                            detection_method = 'baseline'
                            should_report = True
                        elif remote_port in self.malicious_ports and self._is_local_ip(remote_ip):
                            # Malicious port but internal (lower risk)
                            final_risk_score = 3.0
                            threat_indicators = ['internal_suspicious_port']
                            detection_method = 'baseline'
                            should_report = True
                    
                    # ═══════════════════════════════════════════════════════
                    # ✅ REPORT EVENT
                    # ═══════════════════════════════════════════════════════
                    if should_report:
                        record = {
                            'src_ip': conn.laddr.ip,
                            'src_port': local_port,
                            'dst_ip': remote_ip,
                            'dst_port': remote_port,
                            'protocol': protocol,
                            'connection_count': history['count'],
                            'first_seen': history['first_seen'].isoformat() if history['first_seen'] else None,
                            'dns_query': dns_name,
                            'geolocation': self._get_geolocation(remote_ip),
                            'risk_score': final_risk_score,
                            'is_anomaly': bool(ml_is_anomaly),
                            'anomaly_score': float(ml_risk_score) if ml_is_anomaly else 0.0,
                            'threat_indicators': threat_indicators,
                            'detection_method': detection_method
                        }
                        network_data.append(record)
            
            # Update baselines
            self.anomaly_detector.update_baseline('network', 'connections', active_connections)
            self.anomaly_detector.update_baseline('network', 'bytes', current_bytes)
            self.connection_timestamps.append(current_time)
            
            # Summary
            ml_detections = len([r for r in network_data if r.get('detection_method') == 'ml'])
            if active_connections > 0:
                significant_events.append({
                    'event_type': 'network_summary',
                    'active_connections': active_connections,
                    'unique_destinations': len(unique_destinations),
                    'ml_anomalies': ml_detections,
                    'rule_detections': len([r for r in network_data if r.get('detection_method') == 'rule']),
                    'timestamp': current_time.isoformat()
                })
        
        except Exception as e:
            print(f"Network collection error: {e}")
            import traceback
            traceback.print_exc()
        
        return {
            'details': network_data[:100],
            'all_events': network_data,
            'summary': significant_events
        }
    
    def _identify_protocol(self, conn, port):
        """Identify application protocol"""
        base_protocol = 'TCP' if conn.type == socket.SOCK_STREAM else 'UDP'
        app_protocol = self.port_protocols.get(port, 'Unknown')
        
        if app_protocol != 'Unknown':
            return f"{base_protocol}/{app_protocol}"
        return f"{base_protocol}:{port}"
    
    def _is_local_ip(self, ip):
        """Check if IP is local/private"""
        return (ip.startswith('10.') or ip.startswith('192.168.') or 
                ip.startswith('172.') or ip.startswith('127.') or
                ip.startswith('169.254.'))  # Link-local
    
    def resolve_dns(self, ip):
        """Attempt reverse DNS lookup with caching"""
        if ip in self.dns_cache:
            return self.dns_cache[ip]
        
        try:
            hostname = socket.gethostbyaddr(ip)[0]
            self.dns_cache[ip] = hostname
            return hostname
        except:
            return None
    
    def _get_geolocation(self, ip):
        """Get geolocation - simplified"""
        if self._is_local_ip(ip):
            return 'Local Network'
        return 'External'
    
    def _is_safe_domain(self, dns_name):
        """Check if domain matches safe destinations"""
        if not dns_name:
            return False
        
        dns_lower = dns_name.lower()
        for pattern in self.safe_destinations:
            if pattern.startswith('*.'):
                suffix = pattern[2:]
                if dns_lower.endswith(suffix):
                    return True
            elif pattern in dns_lower:
                return True
        return False
    
    def _calculate_comprehensive_network_risk(self, remote_ip, remote_port, local_port, 
                                          protocol, dns_name, history, conn):
        """Calculate comprehensive risk score - BALANCED ( Patch + Your Enhancements)"""

        risk = 0.0
        indicators = []

        # ============================================================
        # SAFE / WHITELISTED DESTINATIONS
        # ============================================================

        # Skip server
        if remote_ip == SERVER_HOST and remote_port == SERVER_PORT:
            return 0.0, ['fortifai_server_connection']

        # Localhost
        if remote_ip in ['127.0.0.1', '::1', 'localhost']:
            return 0.0, ['localhost_connection']

        # Local LAN ranges ( lowered risk dramatically to avoid FP)
        if any(remote_ip.startswith(prefix) for prefix in self.safe_ip_ranges):
            return 0.5, ['local_network_connection']   # ↓ from your 1.0

        # Safe known domains
        if dns_name and self._is_safe_domain(dns_name):
            return 0.5, ['known_safe_domain']  # unchanged

        # Cloud/CDN auto-safe ( added)
        if remote_port == 443 and dns_name:
            cloud_indicators = [
                'amazonaws', 'cloudfront', 'azure', 'googleusercontent',
                'cloudflare', 'akamai', 'fastly', 'cdn', 'facebook', 'fbcdn'
            ]
            if any(c in dns_name.lower() for c in cloud_indicators):
                return 1.0, ['https_cloud_service']  # new ( soft whitelist)

        # ============================================================
        # PORT-BASED RISK
        # ============================================================

        # Malicious ports
        if remote_port in self.malicious_ports:
            risk += 7.0  # ↓ from your 8.0 (: reduce false-positives)
            indicators.append(f'known_malicious_port:{remote_port}')

        # Suspicious port ranges
        for start, end in self.suspicious_port_ranges:
            if start <= remote_port <= end:
                risk += 4.0  # ↓ from your 5.0
                indicators.append(f'suspicious_port_range:{remote_port}')
                break

        # ============================================================
        # PROTOCOL-BASED RISK
        # ============================================================

        # Unencrypted protocols to external hosts
        if any(x in protocol for x in ['FTP', 'Telnet', 'HTTP:']):
            if not self._is_local_ip(remote_ip):
                risk += 2.5  # ↓ from your 3.0
                indicators.append('unencrypted_protocol_external')

        # Unknown protocol + suspicious port
        if 'Unknown' in protocol:
            if remote_port < 1024 or remote_port in self.malicious_ports:
                risk += 1.5  # ↓ from your 2.0
                indicators.append('unknown_protocol_suspicious_port')

        # ============================================================
        # DESTINATION RISK
        # ============================================================

        if not self._is_local_ip(remote_ip):

            risk += 0.5  # ↓ from your 1.5 ( says external connections normal)

            # No DNS = suspicious
            if not dns_name:
                risk += 1.5  # ↓ from your 2.0
                indicators.append('no_reverse_dns_external')

            # Unknown external domain (soft risk)
            elif dns_name and not self._is_safe_domain(dns_name):
                risk += 0.5  # ↓ from your 1.0
                indicators.append('unknown_external_domain')

        # ============================================================
        # BEHAVIOR / FREQUENCY ANALYSIS
        # ============================================================

        # Very frequent connections (C2 avoidance)
        if history['count'] > 500:
            risk += 2.0  # ↓ from your 3.0
            indicators.append(f'very_high_frequency:{history["count"]}')
        elif history['count'] > 200:
            risk += 1.0  # ↓ from your 2.0
            indicators.append(f'high_frequency_connection:{history["count"]}')

        # Multiple local ports = scanning
        if len(history['ports_used']) > 30:
            risk += 3.0  # ↓ from your 4.0
            indicators.append(f'port_scanning:{len(history["ports_used"])}')
        elif len(history['ports_used']) > 15:
            risk += 1.5  # ↓ from your 2.0
            indicators.append(f'multiple_ports_probing:{len(history["ports_used"])}')

        # ============================================================
        # TIMING-BASED RISK (you had this; using softened numbers)
        # ============================================================

        if history['count'] == 1:

            if remote_port in self.malicious_ports:
                risk += 2.0  # ↓ from your 3.0
                indicators.append('first_connection_malicious_port')

            elif not self._is_local_ip(remote_ip) and not dns_name:
                risk += 1.0  # ↓ from your 1.5
                indicators.append('new_external_no_dns')

        # ============================================================
        # DATABASE/SERVICE EXPOSURE
        # ============================================================

        if remote_port in [1433, 1434, 3306, 5432, 27017, 6379, 11211]:

            if not self._is_local_ip(remote_ip):
                risk += 5.0  # ↓ from your 6.0
                indicators.append(f'database_port_external:{remote_port}')

            elif history['count'] > 50:
                risk += 1.5  # ↓ from your 2.0
                indicators.append(f'high_frequency_database_access:{remote_port}')

        # ============================================================
        # ANONYMIZATION / TOR / PROXY
        # ============================================================

        if remote_port in [9050, 9150, 1080, 1081, 3128, 8080]:
            risk += 3.0  # ↓ from your 4.0
            indicators.append('anonymization_detected')

        # ============================================================
        # RDP / REMOTE ACCESS
        # ============================================================

        if remote_port == 3389:
            if not self._is_local_ip(remote_ip):
                risk += 4.0  # ↓ from your 5.0
                indicators.append('external_rdp_connection')

        # ============================================================
        # IRC/BOTNET
        # ============================================================

        if remote_port in [6667, 6668, 6669, 6697]:
            risk += 4.0  # ↓ from your 5.0
            indicators.append('irc_potential_botnet')

        return min(risk, 10.0), indicators


class EnhancedProcessCollector:
    """Enhanced process collector with threat detection"""
    
    def __init__(self, anomaly_detector):
        self.process_cache = {}
        self.suspicious_processes = [
            'mimikatz', 'psexec', 'procdump', 'netcat', 'nc.exe',
            'pwdump', 'wce.exe', 'gsecdump', 'fgdump', 'crackmapexec',
            'metasploit', 'meterpreter', 'beacon', 'cobalt', 'empire',
            'lazagne', 'dumpert', 'nanodump', 'sqlmap', 'hydra'
        ]
        
        # ========== SAFE PATHS WHITELIST ==========
        self.safe_paths = [
            'program files', 'program files (x86)', 
            'windows\\system32', 'windows\\syswow64',
            'google', 'microsoft', 'adobe', 'mozilla', 'apple',
            'steam', 'epic games', 'nvidia', 'amd', 'intel',
            'python', 'node', 'npm', 'java', 'git',
            'visual studio', 'vscode', 'pycharm', 'intellij',
            'slack', 'discord', 'zoom', 'teams', 'chrome', 'firefox'
        ]
        
        # ========== SUSPICIOUS COMMAND LINE PATTERNS ==========
        self.suspicious_cmdline_patterns = [
            # PowerShell obfuscation
            'powershell -enc', '-encodedcommand', '-e ', '-nop', '-w hidden',
            'invoke-expression', 'invoke-webrequest', 'downloadstring', 'iex',
            'bypass', '-noni', 'hidden', '-windowstyle hidden',
            # Credential dumping
            'sekurlsa', 'lsadump', 'sam', 'credentials', 'passwords',
            # Remote execution
            'psexec', 'wmic process call create', 'schtasks /create',
            # Reverse shells
            'ncat', 'nc.exe', 'powercat', 'tcp', 'shell',
            # Obfuscation
            'base64', 'frombase64string', 'gzip', 'compress',
            # Network recon
            'net user', 'net group', 'net localgroup', 'nltest', 'dsquery'
        ]
        
        self.SUSPICIOUS_PARENT_CHILD_COMBOS = [
            ('powershell.exe', 'mshta.exe'),
            ('powershell.exe', 'regsvr32.exe'),
            ('cmd.exe', 'bitsadmin.exe'),
            ('cmd.exe', 'certutil.exe'),
            ('winword.exe', 'powershell.exe'),
            ('winword.exe', 'cmd.exe'),
            ('excel.exe', 'powershell.exe'),
            ('excel.exe', 'cmd.exe'),
            ('wscript.exe', 'powershell.exe'),
            ('cscript.exe', 'cmd.exe'),
            ('explorer.exe', 'regsvr32.exe'),
            ('svchost.exe', 'cmd.exe'),  # Unusual
        ]
        
        self.anomaly_detector = anomaly_detector
        self.baseline_process_count = deque(maxlen=20)
    
    def _check_suspicious_parent_child(self, parent_name, process_name):
        """Check for known malicious parent-child relationships"""
        if not parent_name or not process_name:
            return False, None
        
        parent_lower = parent_name.lower()
        proc_lower = process_name.lower()
        
        for susp_parent, susp_child in self.SUSPICIOUS_PARENT_CHILD_COMBOS:
            if susp_parent in parent_lower and susp_child in proc_lower:
                return True, f"{susp_parent}→{susp_child}"
        
        return False, None

    def collect(self):
        """Collect process data with ML-DRIVEN threat detection"""
        process_data = []
        high_risk_processes = []
        
        # Irrelevant system processes to skip
        IGNORE_PROCESSES = {
            'system idle process', 'system', 'memcompression', 'registry', 'idle',
            'dwm.exe', 'csrss.exe', 'wininit.exe', 'services.exe', 'lsass.exe',
            'svchost.exe', 'winlogon.exe', 'smss.exe', 'audiodg.exe'
        }
        
        # Trusted applications
        TRUSTED_PROCESSES = {
            # Browsers
            'explorer.exe', 'chrome.exe', 'firefox.exe', 'msedge.exe', 'brave.exe',
            'opera.exe', 'vivaldi.exe', 'iexplore.exe',
            
            # Development
            'code.exe', 'pycharm64.exe', 'notepad.exe', 'notepad++.exe',
            'python.exe', 'pythonw.exe', 'node.exe', 'java.exe', 'javaw.exe',
            
            # System
            'conhost.exe', 'cmd.exe', 'powershell.exe', 'svchost.exe',
            'dwm.exe', 'csrss.exe', 'wininit.exe', 'services.exe',
            'taskmgr.exe', 'taskhostw.exe', 'sihost.exe',
            
            # Communication
            'discord.exe', 'slack.exe', 'teams.exe', 'zoom.exe', 
            'outlook.exe', 'thunderbird.exe',
            
            # ✅ NEW: Add multimedia and common apps
            'spotify.exe', 'vlc.exe', 'winamp.exe', 'foobar2000.exe',
            'steam.exe', 'epicgameslauncher.exe', 'origin.exe',
            'dropbox.exe', 'onedrive.exe', 'googledrivesync.exe',
        }
        
        try:
            total_processes = 0
            high_cpu_count = 0
            high_memory_count = 0
            total_cpu = 0.0
            
            script_spawned_count = 0
            temp_execution_count = 0
            
            for proc in psutil.process_iter(['pid', 'name', 'username', 'cpu_percent', 
                                            'memory_info', 'create_time', 'exe', 'cmdline']):
                try:
                    pinfo = proc.info
                    process_name = pinfo.get('name', '').lower()
                    
                    if process_name in IGNORE_PROCESSES:
                        continue
                    
                    mem_info = pinfo.get('memory_info')
                    memory_mb = mem_info.rss / (1024 * 1024) if mem_info else 0
                    cpu_percent = pinfo.get('cpu_percent', 0)
                    
                    self.feature_manager.update_process_event({
                        'process_name': pinfo['name'],
                        'cpu_percent': cpu_percent,
                        'memory_mb': memory_mb
                    })
                    
                    total_processes += 1
                    
                    mem_info = pinfo.get('memory_info')
                    memory_mb = mem_info.rss / (1024 * 1024) if mem_info else 0
                    cpu_percent = pinfo.get('cpu_percent', 0)
                    total_cpu += cpu_percent
                    
                    if cpu_percent > 50:
                        high_cpu_count += 1
                    if memory_mb > 500:
                        high_memory_count += 1
                    
                    exe_path = pinfo.get('exe')
                    cmdline = pinfo.get('cmdline')
                    cmdline_str = ' '.join(cmdline) if cmdline else ''
                    
                    if exe_path and 'temp' in exe_path.lower():
                        temp_execution_count += 1
                    
                    # Get parent
                    parent = None
                    parent_name = None
                    try:
                        parent_proc = psutil.Process(proc.ppid())
                        parent = proc.ppid()
                        parent_name = parent_proc.name().lower()
                        
                        if any(p in parent_name for p in ['powershell', 'cmd', 'wscript', 'cscript']):
                            script_spawned_count += 1
                    except:
                        pass
                    
                    is_suspicious_lineage, lineage_pattern = self._check_suspicious_parent_child(
                        parent_name, process_name
                    )

                    if is_suspicious_lineage:
                        rule_risk_score += 7.0
                        rule_indicators.append(f'malicious_lineage:{lineage_pattern}')
                        
                    # ═══════════════════════════════════════════════════════════════
                    # ✅ BUILD COMPLETE 24-FEATURE VECTOR
                    # ═══════════════════════════════════════════════════════════════

                    window_stats = self.feature_manager.get_window_stats()

                    # Network features (9) - use aggregated stats
                    conn_count_scaled = float(window_stats.get('network_events', 0)) / 100.0
                    unique_dst_ratio = 0.0
                    bytes_sent_scaled = 0.0
                    bytes_recv_scaled = 0.0
                    port_entropy_val = 0.0
                    tcp_ratio = 0.0
                    udp_ratio = 0.0
                    conn_rate = 0.0
                    dst_churn_rate = 0.0

                    # Process features (4) - PRIMARY DATA
                    proc_spawn_count_scaled = float(script_spawned_count) / 10.0
                    avg_proc_cpu_scaled = float(cpu_percent) / 100.0
                    avg_proc_memory_scaled = float(memory_mb) / 5000.0
                    unique_proc_names_scaled = 1.0  # Current process

                    # Filesystem features (3)
                    file_create_count = float(window_stats.get('file_events', 0)) / 50.0
                    file_exec_count = 1.0 if exe_path and 'temp' in exe_path.lower() else 0.0
                    file_hash_novelty = 0.0

                    # Rate of change (5)
                    delta_conn_count = 0.0
                    delta_file_create = 0.0
                    delta_proc_spawn = float(script_spawned_count) / 10.0
                    delta_unique_hashes = 0.0
                    conn_acceleration = 0.0

                    # Pattern flags (3)
                    ransomware_burst = 0.0
                    c2_pattern = 0.0
                    cred_dump_pattern = 1.0 if (
                        parent_name and any(x in parent_name for x in ['powershell', 'cmd']) and
                        any(susp in process_name for susp in self.suspicious_processes[:5])
                    ) else 0.0

                    # Lineage flag override
                    if is_suspicious_lineage:
                        cred_dump_pattern = 1.0

                    # ✅ ASSEMBLE 24-FEATURE VECTOR
                    feature_vector = np.array([
                        # Network (9)
                        conn_count_scaled, unique_dst_ratio, bytes_sent_scaled, bytes_recv_scaled,
                        port_entropy_val, tcp_ratio, udp_ratio, conn_rate, dst_churn_rate,
                        # Process (4)
                        proc_spawn_count_scaled, avg_proc_cpu_scaled, avg_proc_memory_scaled, unique_proc_names_scaled,
                        # Filesystem (3)
                        file_create_count, file_exec_count, file_hash_novelty,
                        # Rate of change (5)
                        delta_conn_count, delta_file_create, delta_proc_spawn, delta_unique_hashes, conn_acceleration,
                        # Pattern flags (3)
                        ransomware_burst, c2_pattern, cred_dump_pattern
                    ])

                    feature_names = [
                        'conn_count', 'unique_dst_count', 'bytes_sent', 'bytes_recv',
                        'port_entropy', 'tcp_ratio', 'udp_ratio', 'conn_rate', 'dst_churn_rate',
                        'proc_spawn_count', 'avg_proc_cpu', 'avg_proc_memory', 'unique_proc_names',
                        'file_create_count', 'file_exec_count', 'file_hash_novelty',
                        'delta_conn_count', 'delta_file_create', 'delta_proc_spawn',
                        'delta_unique_hashes', 'conn_acceleration',
                        'ransomware_burst', 'c2_pattern', 'cred_dump_pattern'
                    ]

                    assert len(feature_vector) == 24, f"Process collector feature mismatch: {len(feature_vector)}"
                    assert len(feature_names) == 24, f"Process collector name mismatch: {len(feature_names)}"

                    # ✅ FIX: INVOKE ML DETECTION ON EVERY PROCESS EVENT
                    ml_is_anomaly = False
                    ml_risk_score = 0.0
                    ml_indicators = []

                    if (self.anomaly_detector.isolation_forest is not None or 
                        (self.anomaly_detector.autoencoder and self.anomaly_detector.autoencoder.is_trained)):
                        
                        try:
                            is_anomaly, anomaly_info = self.anomaly_detector.detect_anomaly_ensemble(
                                feature_vector,
                                feature_names
                            )
                            
                            if is_anomaly:
                                ml_is_anomaly = True
                                severity_map = {'critical': 9.5, 'high': 9.0, 'medium': 6.5, 'low': 4.5}
                                ml_risk_score = severity_map.get(anomaly_info['severity'], 5.0)
                                ml_indicators = [f"ml_{ind}" for ind in anomaly_info.get('contributing_features', [])[:3]]
                                
                                # ✅ If lineage is in top features, add specific indicator
                                if 'malicious_lineage' in anomaly_info.get('contributing_features', []):
                                    ml_indicators.append(f"lineage:{lineage_pattern}")
         
                                print(f"  [ML PROCESS] Anomaly detected: {process_name} (PID={pinfo['pid']}) "
                                      f"(severity={anomaly_info['severity']}, score={ml_risk_score:.1f})")
                        
                        except Exception as e:
                            print(f"  [ML] Process detection error: {e}")

                    # ═══════════════════════════════════════════════════════
                    # ✅ STEP 2: RULE-BASED (ONLY IF ML DIDN'T DETECT)
                    # ═══════════════════════════════════════════════════════
                    rule_risk_score = 0.0
                    rule_indicators = []

                    if not ml_is_anomaly:  # ✅ Only evaluate rules if ML didn't flag
                        # Rule 1: Known malicious process names
                        critical_malware = ['mimikatz', 'psexec', 'procdump', 'pwdump', 'gsecdump']
                        if any(mal in process_name for mal in critical_malware):
                            rule_risk_score += 9.5  # ✅ INCREASED from 8.0
                            rule_indicators.append('critical_malware_process')
                        elif any(susp in process_name for susp in self.suspicious_processes):
                            rule_risk_score += 8.0
                            rule_indicators.append('suspicious_process')
                        
                        if is_suspicious_lineage:
                            rule_risk_score += 7.5
                            rule_indicators.append(f'malicious_lineage:{lineage_pattern}')
                            
                        # Rule 2: Temp execution with suspicious name
                        if exe_path and 'temp' in exe_path.lower() and process_name not in TRUSTED_PROCESSES:
                            # Check if suspicious patterns in name
                            if any(p in process_name for p in ['crack', 'keygen', 'hack', 'exploit', 'payload']):
                                rule_risk_score += 7.0
                                rule_indicators.append('temp_execution_suspicious_name')
                        
                        # Rule 3: PowerShell with encoded command
                        if 'powershell' in process_name and cmdline_str:
                            obfuscation_flags = ['-enc', '-encodedcommand', 'bypass', '-w hidden', 
                                               'invoke-expression', 'downloadstring', 'iex']
                            if any(flag in cmdline_str.lower() for flag in obfuscation_flags):
                                rule_risk_score += 8.0  # ✅ INCREASED from 7.0
                                rule_indicators.append('powershell_obfuscation')
                        
                        # Rule 4: Office spawning executable
                        if parent_name and any(office in parent_name for office in ['winword', 'excel', 'powerpnt']):
                            if exe_path and not self._is_safe_path(exe_path):
                                rule_risk_score += 7.0
                                rule_indicators.append('office_macro_execution')
                        
                        # Rule 5: Downloads/Desktop execution
                        if exe_path and any(x in exe_path.lower() for x in ['downloads\\', 'desktop\\']):
                            if process_name not in TRUSTED_PROCESSES:
                                rule_risk_score += 3.0
                                rule_indicators.append('user_directory_execution')

                    # ═══════════════════════════════════════════════════════
                    # ✅ STEP 3: COMBINE DETECTIONS (ML PRIORITY)
                    # ═══════════════════════════════════════════════════════
                    final_risk_score = 0.0
                    detection_method = 'baseline'
                    threat_indicators = []
                    should_report = False

                    if ml_is_anomaly:
                        # ML detected anomaly - HIGHEST PRIORITY
                        final_risk_score = ml_risk_score
                        threat_indicators = ml_indicators
                        detection_method = 'ml'
                        should_report = True
                        
                    elif rule_risk_score > 6.0:
                        # Rule-based detection - MEDIUM PRIORITY
                        final_risk_score = rule_risk_score
                        threat_indicators = rule_indicators
                        detection_method = 'rule'
                        should_report = True
                        
                    else:
                        # Baseline monitoring - LOW PRIORITY
                        if cpu_percent > 80 or memory_mb > 2000:
                            final_risk_score = 3.0
                            threat_indicators = ['high_resource_usage']
                            detection_method = 'resource'
                            should_report = True
                        elif total_processes % 30 == 0 and process_name not in TRUSTED_PROCESSES:
                            # Sample every 20th process for monitoring
                            final_risk_score = 1.0
                            threat_indicators = ['analyzed_for_baseline']
                            detection_method = 'baseline'
                            should_report = True

                    # ✅ Skip trusted processes with low risk
                    if process_name in TRUSTED_PROCESSES and final_risk_score < 5.0:
                        should_report = False

                    # ═══════════════════════════════════════════════════════
                    # ✅ REPORT EVENT
                    # ═══════════════════════════════════════════════════════
                    if should_report:
                        exe_hash = None
                        if exe_path and os.path.exists(exe_path) and final_risk_score > 6.0:
                            exe_hash = self.calculate_file_hash(exe_path)
                        
                        record = {
                            'process_name': pinfo['name'],
                            'pid': pinfo['pid'],
                            'ppid': parent,
                            'parent_name': parent_name,
                            'executable_path': exe_path,
                            'executable_hash': exe_hash,
                            'command_line_preview': cmdline_str[:100] if final_risk_score > 6.0 else None,
                            'start_time': datetime.fromtimestamp(pinfo['create_time']),
                            'cpu_percent': cpu_percent,
                            'memory_mb': memory_mb,
                            'privilege_level': pinfo.get('username', 'unknown'),
                            'risk_score': final_risk_score,
                            'threat_indicators': threat_indicators,
                            'detection_method': detection_method
                        }
                        
                        if final_risk_score > 6.0:
                            high_risk_processes.append(record)
                        else:
                            process_data.append(record)
                        
                        if final_risk_score > 7.0:
                            alert_dict = {
                                'timestamp': datetime.now().isoformat(),
                                'category': 'process',
                                'severity': 'high' if final_risk_score > 8 else 'medium',
                                'ensemble_score': final_risk_score,
                                'contributing_features': threat_indicators[:3],
                                'explanation': f"Suspicious process: {record['process_name']} (PID {record['pid']})",
                                'model_contributions': {'rule_based': {'score': final_risk_score}}
                            }
                            self.anomaly_detector.anomaly_alerts.append(alert_dict)
                
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            
            self.anomaly_detector.update_baseline('process', 'count', total_processes)
            self.anomaly_detector.update_baseline('process', 'cpu', total_cpu)
            
            is_anomaly, z_score = self.anomaly_detector.detect_anomaly('process', 'count', total_processes)
            
            ml_detections = len([p for p in (process_data + high_risk_processes) if p.get('detection_method') == 'ml'])
            
            summary = {
                'event_type': 'process_summary',
                'total_processes': total_processes,
                'high_cpu_processes': high_cpu_count,
                'high_memory_processes': high_memory_count,
                'high_risk_count': len(high_risk_processes),
                'ml_detections': ml_detections,
                'rule_detections': len(process_data + high_risk_processes) - ml_detections,
                'is_anomalous_count': bool(is_anomaly),
                'anomaly_score': float(z_score) if is_anomaly else 0.0,
                'timestamp': datetime.now().isoformat()
            }
            
            if len(process_data) == 0 and len(high_risk_processes) == 0:
                try:
                    top_procs = sorted(
                        psutil.process_iter(['pid','name','cpu_percent','memory_info']),
                        key=lambda p: (p.info.get('cpu_percent',0) + (p.info.get('memory_info').rss/1024/1024 if p.info.get('memory_info') else 0)),
                        reverse=True
                    )[:5]

                    summary['top_processes'] = [
                        {
                            'process_name': p.info.get('name'),
                            'pid': p.info.get('pid'),
                            'cpu_percent': p.info.get('cpu_percent', 0),
                            'memory_mb': (p.info.get('memory_info').rss/1024/1024 if p.info.get('memory_info') else 0)
                        }
                        for p in top_procs
                    ]
                except:
                    pass
        
        except Exception as e:
            print(f"Process collection error: {e}")
            return {'details': [], 'high_risk': [], 'summary': {}}
        
        return {
            'details': process_data[:30],
            'high_risk': high_risk_processes[:20],
            'summary': summary
        }
    
    # ADD THIS METHOD (if not already present):
    def calculate_file_hash(self, filepath):
        """Calculate SHA256 hash of executable - SHARED WITH FILESYSTEM"""
        try:
            if not os.path.exists(filepath):
                return None
            
            if not os.access(filepath, os.R_OK):
                return None
            
            file_size = os.path.getsize(filepath)
            
            # Skip files > 100MB
            if file_size > 100 * 1024 * 1024:
                return None
            
            sha256 = hashlib.sha256()
            chunk_size = 8192
            
            with open(filepath, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    sha256.update(chunk)
            
            return sha256.hexdigest()
        
        except (PermissionError, FileNotFoundError, OSError):
            return None
        except Exception as e:
            print(f"  [HASH] Error: {e}")
            return None
    
    def _is_safe_path(self, path):
        """Check if path is in safe locations"""
        if not path:
            return False
        
        path_lower = path.lower()
        return any(safe in path_lower for safe in self.safe_paths)
    
    def _calculate_comprehensive_process_risk(self, name, path, cmdline, memory, cpu, pid=None):
        """Calculate comprehensive process risk - ENHANCED DETECTION"""
        import psutil

        risk = 0.0
        indicators = []

        name_lower = name.lower() if name else ''
        path_lower = path.lower() if path else ''
        cmdline_lower = cmdline.lower() if cmdline else ''

        # ✅ EARLY EXIT for safe software
        if self._is_safe_path(path):
            # Still check for suspicious command lines
            if any(p in cmdline_lower for p in ['bypass', '-enc', 'invoke-expression', 'downloadstring']):
                risk = 6.0
                indicators.append('suspicious_cmdline_safe_path')
                return 6.0, indicators
            return 0.5, ['safe_location']

        # ========== CRITICAL MALWARE NAMES (INSTANT HIGH RISK) ==========
        critical_malware = [
            'mimikatz', 'keylog', 'pwdump', 'gsecdump', 'wce',
            'lazagne', 'dumpert', 'nanodump', 'procdump',
            'metasploit', 'meterpreter', 'beacon', 'cobalt',
            'empire', 'crackmapexec', 'psexec'
        ]
        
        for malware in critical_malware:
            if malware in name_lower:
                risk += 9.0
                indicators.append(f'critical_malware:{malware}')
                return 9.0, indicators  # Immediate return

        # ========== SUSPICIOUS PROCESS NAMES ==========
        for suspicious in self.suspicious_processes:
            if suspicious in name_lower:
                risk += 8.0  # Increased from 7.0
                indicators.append(f'known_suspicious_process:{suspicious}')
                break

        # ========== PATH-BASED RISK ==========
        if path_lower:
            # User-space execution
            if any(x in path_lower for x in ['\\downloads\\', '\\desktop\\', '\\public\\']):
                risk += 4.0  # Increased from 3.0
                indicators.append('user_directory_execution')

            # Temp execution (VERY SUSPICIOUS)
            elif 'temp' in path_lower or 'tmp' in path_lower:
                if not any(safe in path_lower for safe in ['microsoft', 'google', 'windows']):
                    risk += 3.0  # Increased from 2.0
                    indicators.append('temp_execution')

            # AppData\Roaming
            elif 'appdata\\roaming' in path_lower:
                if any(susp in name_lower for susp in self.suspicious_processes[:5]):
                    risk += 3.0  # Increased from 2.0
                    indicators.append('appdata_roaming_suspicious')

            # ProgramData
            elif 'programdata' in path_lower:
                risk += 2.0  # Increased from 1.5
                indicators.append('programdata_execution')

        # ========== COMMAND LINE ANALYSIS (ENHANCED) ==========
        cmdline_flags = 0
        matched_patterns = []

        # Check ALL suspicious patterns
        for pattern in self.suspicious_cmdline_patterns:
            if pattern in cmdline_lower:
                cmdline_flags += 1
                matched_patterns.append(pattern)

        if cmdline_flags >= 3:
            risk += 8.0  # Increased from 7.0
            indicators.append(f'highly_suspicious_cmdline:{",".join(matched_patterns[:3])}')
        elif cmdline_flags == 2:
            risk += 6.0  # Increased from 5.0
            indicators.append(f'suspicious_cmdline:{",".join(matched_patterns)}')
        elif cmdline_flags == 1:
            risk += 3.0  # Increased from 2.0
            indicators.append(f'suspicious_cmdline_pattern:{matched_patterns[0]}')

        # ========== SPECIFIC DANGEROUS PATTERNS ==========
        # PowerShell encoded commands (CRITICAL)
        if 'powershell' in name_lower:
            if any(x in cmdline_lower for x in ['-enc', '-encodedcommand', 'frombase64string']):
                risk += 7.0
                indicators.append('powershell_encoded_command')
            elif 'bypass' in cmdline_lower:
                risk += 6.0
                indicators.append('powershell_execution_policy_bypass')
            elif '-w hidden' in cmdline_lower or 'windowstyle hidden' in cmdline_lower:
                risk += 5.0
                indicators.append('powershell_hidden_window')

        # ========== RESOURCE USAGE ==========
        if memory and memory > 5000:
            risk += 2.5  # Increased from 2.0
            indicators.append('very_high_memory')
        elif memory and memory > 3000:
            risk += 1.5  # Increased from 1.0
            indicators.append('high_memory')

        if cpu and cpu > 95:
            risk += 2.0  # Increased from 1.5
            indicators.append('very_high_cpu')
        elif cpu and cpu > 85:
            risk += 1.0  # Increased from 0.5
            indicators.append('high_cpu')

        # ========== PROCESS LINEAGE ANALYSIS ==========
        if pid:
            try:
                proc = psutil.Process(pid)
                parent = proc.parent()

                if parent:
                    parent_name = parent.name().lower()

                    # Script spawning executables (VERY SUSPICIOUS)
                    suspicious_parents = ['powershell', 'cmd.exe', 'wscript.exe', 'cscript.exe', 'mshta.exe']
                    if any(p in parent_name for p in suspicious_parents):
                        risk += 4.0  # Increased from 3.0
                        indicators.append(f'script_spawned_process:{parent_name}')

                    # Download execution via script
                    if ('\\downloads\\' in path_lower and 
                        any(p in parent_name for p in suspicious_parents)):
                        risk += 5.0  # Increased from 4.0
                        indicators.append('download_script_execution')

                    # Office macro execution (CRITICAL)
                    office_sources = ['winword.exe', 'excel.exe', 'powerpnt.exe']
                    if parent_name in office_sources:
                        risk += 6.0  # Increased from 4.0
                        indicators.append('office_macro_spawned_process')

            except Exception:
                pass

        return min(risk, 10.0), indicators


class EnhancedFilesystemCollector:
    """Enhanced filesystem monitoring with comprehensive threat detection"""
    
    def __init__(self, anomaly_detector):
        self.watched_directories = self.get_critical_directories()
        self.anomaly_detector = anomaly_detector
        self.last_scan = {}
        self.file_event_count = 0
        self.known_hashes = set()  # Track known file hashes
        self.file_creation_rate = deque(maxlen=100)  # Track creation timestamps
        self.baseline_extensions = defaultdict(int)  # Normal extension distribution
        self.anomaly_alerts = deque(maxlen=100)
                   
        # ========== THREAT INTELLIGENCE ==========
        # Truly malicious extensions (not just executables)
        self.high_risk_extensions = {
            '.exe', '.scr', '.pif', '.com',  # Windows executables
            '.hta', '.vbs', '.vbe', '.ws', '.wsf', '.wsc', '.wsh',  # Scripts
            '.ps1', '.psm1', '.psd1',  # PowerShell
            '.bat', '.cmd',  # Batch files
            '.msp', '.mst',  # Installers
            '.dll', '.ocx', '.cpl', '.drv',  # Libraries/drivers
            '.sys', '.scf', '.inf',  # System files
            '.reg', '.hiv',  # Registry files
            '.docm', '.xlsm', '.pptm', '.dotm',  # Macro-enabled Office
            '.jar', '.jnlp',  # Java
            '.appx', '.appxbundle', '.msix',  # Modern Windows packages
        }
        
        # Double extension patterns (common malware trick)
        self.double_extension_tricks = {
            '.pdf.exe', '.doc.exe', '.jpg.exe', '.png.exe', '.mp3.exe',
            '.txt.scr', '.pdf.scr', '.doc.scr', '.jpg.scr',
            '.pdf.js', '.doc.js', '.txt.js',
            '.pdf.vbs', '.doc.vbs', '.txt.vbs',
            '.doc.bat', '.pdf.bat', '.txt.bat',
        }
        
        # Suspicious filename patterns (regex-like matching)
        self.suspicious_patterns = [
            'crack', 'keygen', 'patch', 'loader', 'activator', 'serial',
            'hack', 'cheat', 'exploit', 'payload', 'inject', 'dump',
            'mimikatz', 'pwdump', 'gsecdump', 'wce', 'lazagne',
            'shell', 'backdoor', 'trojan', 'virus', 'malware', 'ransom',
            'cryptolocker', 'wannacry', 'petya', 'locky',
            'keylog', 'stealer', 'rat', 'botnet', 'rootkit',
            'bypass', 'disable', 'kill_av', 'killav', 'stop_av',
        ]
        
        self.malware_signatures = [
            'mimikatz', 'keylog', 'ransom', 'cryptolocker', 'wannacry', 'petya',
            'trojan', 'backdoor', 'rootkit', 'virus', 'malware', 'spyware',
            'crack', 'keygen', 'hack', 'exploit', 'payload', 'inject', 'dump',
            'stealer', 'rat', 'botnet', 'worm', 'adware', 'dropper', 'loader'
        ]
        
        # Legitimate software paths to whitelist
        self.safe_paths = [
            'microsoft', 'google', 'mozilla', 'adobe', 'oracle', 'java',
            'python', 'nodejs', 'npm', 'git', 'vscode', 'visual studio',
            'intellij', 'jetbrains', 'slack', 'discord', 'zoom', 'teams',
            'chrome', 'firefox', 'edge', 'opera', 'brave',
            'steam', 'epic games', 'nvidia', 'amd', 'intel',
            'windows defender', 'kaspersky', 'norton', 'avast', 'malwarebytes',
            'program files', 'program files (x86)', 'windows\\system32',
        ]
        
        # Ransomware extension patterns
        self.ransomware_extensions = {
            '.encrypted', '.locked', '.crypto', '.crypt', '.enc', '.encoded',
            '.locky', '.cerber', '.zepto', '.odin', '.thor', '.zzzzz',
            '.micro', '.mp3', '.vvv', '.ccc', '.xyz', '.aaa', '.abc',
            '.exx', '.ezz', '.ecc', '.xxx', '.ttt', '.qqq', '.crinf',
            '.r5a', '.XRNT', '.XTBL', '.LOL!', '.fun', '.gws', '.btc',
        }
    
    def calculate_file_hash(self, filepath):
        """Calculate SHA256 hash of file"""
        try:
            sha256 = hashlib.sha256()
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except Exception as e:
            return None
            
    def get_critical_directories(self):
        """Get directories to monitor - EXPANDED"""
        dirs = []
        
        if platform.system() == 'Windows':
            user_profile = os.environ.get('USERPROFILE', 'C:\\Users\\Default')
            dirs = [
                os.path.join(user_profile, 'Downloads'),
                os.path.join(user_profile, 'Documents'),
                os.path.join(user_profile, 'Desktop'),
                os.path.join(user_profile, 'AppData', 'Local', 'Temp'),
                os.path.join(user_profile, 'AppData', 'Roaming'),
                os.path.join(user_profile, 'AppData', 'Local'),
                'C:\\Windows\\Temp',
                'C:\\ProgramData',
                'C:\\Users\\Public',
            ]
        else:
            home = os.path.expanduser('~')
            dirs = [
                os.path.join(home, 'Downloads'),
                os.path.join(home, 'Documents'),
                os.path.join(home, 'Desktop'),
                '/tmp', '/var/tmp',
                '/dev/shm',  # RAM disk often used by malware
                os.path.join(home, '.local/share'),
                os.path.join(home, '.config'),
            ]
        
        return [d for d in dirs if os.path.exists(d)]
    
    def collect(self):
        """Collect filesystem changes with ML-DRIVEN threat detection"""
        fs_data = []
        current_time = time.time()
        new_file_count = 0
        
        recent_exec_count = 0
        recent_script_count = 0
        reported_count = 0  # –… ADD: Track reported files
        
        try:
            for directory in self.watched_directories:
                try:
                    for root, dirs, files in os.walk(directory):
                        depth = root[len(directory):].count(os.sep)
                        if depth > 4:
                            dirs[:] = []
                            continue
                        
                        root_lower = root.lower()
                        if any(skip in root_lower for skip in ['windows\\winsxs', 'windows\\assembly', 
                                                            '.git', 'node_modules', '__pycache__',
                                                            'windows\\temp\\chocolatey', 'programdata\\microsoft']):
                            dirs[:] = []
                            continue
                        
                        for filename in files:
                            filepath = os.path.join(root, filename)
                            
                            try:
                                stat = os.stat(filepath)
                                mtime = stat.st_mtime
                                ctime = stat.st_ctime
                                file_ext = os.path.splitext(filename)[1].lower()
                                filename_lower = filename.lower()
                                file_hash = None
                                is_new = filepath not in self.last_scan
                                is_modified = not is_new and self.last_scan.get(filepath, 0) != mtime
                                time_since_modify = current_time - mtime
                                time_since_create = current_time - ctime
                                
                                if file_ext in ['.exe', '.dll', '.sys']:
                                    recent_exec_count += 1
                                if file_ext in ['.ps1', '.bat', '.cmd', '.vbs', '.js']:
                                    recent_script_count += 1
                                
                                # –… FIX: More liberal analysis criteria
                                should_analyze = (
                                    time_since_modify < COLLECTION_INTERVAL + 600 or  
                                    time_since_create < COLLECTION_INTERVAL + 600 or
                                    is_new or is_modified or
                                    (file_ext in self.high_risk_extensions and time_since_create < 86400) or
                                    (file_ext in ['.exe', '.dll', '.ps1', '.bat', '.scr']) or 
                                    reported_count < 50  
                                )
                                
                                if not should_analyze:
                                    self.last_scan[filepath] = mtime
                                    continue
                                
                                if is_new:
                                    new_file_count += 1
                                    self.file_creation_rate.append(current_time)
                                
                                # ═══════════════════════════════════════════════════════════════
                                # ✅ BUILD COMPLETE 24-FEATURE VECTOR
                                # ═══════════════════════════════════════════════════════════════

                                window_stats = self.feature_manager.get_window_stats()

                                # Network features (9)
                                conn_count_scaled = float(window_stats.get('network_events', 0)) / 100.0
                                unique_dst_ratio = 0.0
                                bytes_sent_scaled = 0.0
                                bytes_recv_scaled = 0.0
                                port_entropy_val = 0.0
                                tcp_ratio = 0.0
                                udp_ratio = 0.0
                                conn_rate = 0.0
                                dst_churn_rate = 0.0

                                # Process features (4)
                                proc_spawn_count = float(window_stats.get('process_events', 0)) / 10.0
                                avg_proc_cpu = 0.0
                                avg_proc_memory = 0.0
                                unique_proc_names = 0.0

                                # Filesystem features (3) - PRIMARY DATA
                                file_create_count_scaled = float(new_file_count) / 50.0
                                file_exec_count_scaled = 1.0 if file_ext in ['.exe', '.dll', '.so'] else 0.0
                                file_hash_novelty_scaled = 1.0 if file_hash and file_hash not in self.known_hashes else 0.0

                                # Rate of change (5)
                                delta_conn_count = 0.0
                                delta_file_create = float(new_file_count) / 50.0
                                delta_proc_spawn = 0.0
                                delta_unique_hashes = file_hash_novelty_scaled
                                conn_acceleration = 0.0

                                # Pattern flags (3)
                                ransomware_burst_flag = 1.0 if file_ext in self.ransomware_extensions else 0.0
                                c2_pattern = 0.0
                                cred_dump_pattern = 0.0

                                # ✅ ASSEMBLE 24-FEATURE VECTOR
                                feature_vector = np.array([
                                    # Network (9)
                                    conn_count_scaled, unique_dst_ratio, bytes_sent_scaled, bytes_recv_scaled,
                                    port_entropy_val, tcp_ratio, udp_ratio, conn_rate, dst_churn_rate,
                                    # Process (4)
                                    proc_spawn_count, avg_proc_cpu, avg_proc_memory, unique_proc_names,
                                    # Filesystem (3)
                                    file_create_count_scaled, file_exec_count_scaled, file_hash_novelty_scaled,
                                    # Rate of change (5)
                                    delta_conn_count, delta_file_create, delta_proc_spawn, delta_unique_hashes, conn_acceleration,
                                    # Pattern flags (3)
                                    ransomware_burst_flag, c2_pattern, cred_dump_pattern
                                ])

                                feature_names = [
                                    'conn_count', 'unique_dst_count', 'bytes_sent', 'bytes_recv',
                                    'port_entropy', 'tcp_ratio', 'udp_ratio', 'conn_rate', 'dst_churn_rate',
                                    'proc_spawn_count', 'avg_proc_cpu', 'avg_proc_memory', 'unique_proc_names',
                                    'file_create_count', 'file_exec_count', 'file_hash_novelty',
                                    'delta_conn_count', 'delta_file_create', 'delta_proc_spawn',
                                    'delta_unique_hashes', 'conn_acceleration',
                                    'ransomware_burst', 'c2_pattern', 'cred_dump_pattern'
                                ]

                                assert len(feature_vector) == 24, f"Filesystem collector feature mismatch: {len(feature_vector)}"
                                assert len(feature_names) == 24, f"Filesystem collector name mismatch: {len(feature_names)}"
                                                                
                                # –… Run ML ensemble
                                ml_is_anomaly = False
                                ml_risk_score = 0.0
                                ml_indicators = []
                                
                                if (self.anomaly_detector.isolation_forest is not None or 
                                    (self.anomaly_detector.autoencoder and self.anomaly_detector.autoencoder.is_trained)):
                                    
                                    try:
                                        is_anomaly, anomaly_info = self.anomaly_detector.detect_anomaly_ensemble(
                                            feature_vector,
                                            feature_names
                                        )
                                        
                                        if is_anomaly:
                                            ml_is_anomaly = True
                                            severity_map = {'critical': 9.5, 'high': 8.0, 'medium': 6.5, 'low': 5.0}
                                            ml_risk_score = severity_map.get(anomaly_info['severity'], 5.0)
                                            ml_indicators = [f"ml_{ind}" for ind in anomaly_info.get('contributing_features', [])[:3]]
                                    
                                    except Exception as e:
                                        print(f"  [ML] File detection error: {e}")
                                
                                # –… RULE-BASED (MINIMAL - only truly dangerous patterns)
                                rule_risk_score = 0.0
                                rule_indicators = []
                                
                                # Only flag ransomware extensions (not regular .exe)
                                if file_ext in self.ransomware_extensions and file_ext != '.enc':
                                    rule_risk_score += 8.0
                                    rule_indicators.append(f'ransomware_ext:{file_ext}')
                                
                                # Double extension tricks
                                for trick in self.double_extension_tricks:
                                    if filename_lower.endswith(trick):
                                        rule_risk_score += 8.0
                                        rule_indicators.append(f'double_ext:{trick}')
                                        break
                                
                                # Known malware patterns
                                malware_keywords = ['mimikatz', 'keylog', 'ransom', 'cryptolocker']
                                if any(kw in filename_lower for kw in malware_keywords):
                                    rule_risk_score += 8.0
                                    rule_indicators.append('malware_keyword')
                                
                                # –… FIX: MORE LIBERAL REPORTING
                                should_report = False
                                
                                if ml_is_anomaly:
                                    final_risk_score = max(ml_risk_score, rule_risk_score)
                                    threat_indicators = ml_indicators + rule_indicators[:1]
                                    detection_method = 'ml'
                                    should_report = True
                                elif rule_risk_score > 7.0:
                                    final_risk_score = rule_risk_score
                                    threat_indicators = rule_indicators
                                    detection_method = 'rule'
                                    should_report = True
                                elif is_new or time_since_create < 600:  # New files in last 10 min
                                    final_risk_score = 2.0
                                    threat_indicators = ['recent_file_activity']
                                    detection_method = 'baseline'
                                    should_report = True
                                elif reported_count < 20:  # –… FIX: Report first 20 files
                                    final_risk_score = 1.0
                                    threat_indicators = ['sampled_file']
                                    detection_method = 'baseline'
                                    should_report = True
                                    reported_count += 1
                                
                                # –… FIX: Only skip if in safe path AND no ML detection AND low risk
                                if (any(safe in filepath.lower() for safe in self.safe_paths) and 
                                    not ml_is_anomaly and 
                                    rule_risk_score < 6.0):
                                    should_report = False
                                
                                if should_report:
                                    file_hash = None
                                    if stat.st_size < 50*1024*1024 and final_risk_score > 5.0:
                                        file_hash = self.calculate_file_hash(filepath)
                                        if file_hash:
                                            if file_hash in self.known_hashes:
                                                final_risk_score = max(0, final_risk_score - 1.0)
                                            else:
                                                self.known_hashes.add(file_hash)
                                    
                                    record = {
                                        'event_type': 'created' if is_new else ('modified' if is_modified else 'existing'),
                                        'file_path': filepath,
                                        'file_name': filename,
                                        'file_extension': file_ext,
                                        'file_size': stat.st_size,
                                        'file_hash': file_hash,
                                        'modification_time': datetime.fromtimestamp(mtime),
                                        'creation_time': datetime.fromtimestamp(ctime),
                                        'directory': directory,
                                        'is_suspicious': True,
                                        'risk_score': final_risk_score,
                                        'threat_indicators': threat_indicators,
                                        'detection_method': detection_method
                                    }
                                    fs_data.append(record)
                                    self.file_event_count += 1

                                    if final_risk_score > 6.0:
                                        alert_dict = {
                                            'timestamp': datetime.now().isoformat(),
                                            'category': 'filesystem',
                                            'severity': 'critical' if final_risk_score > 8 else 'high',
                                            'ensemble_score': final_risk_score,
                                            'contributing_features': threat_indicators[:3],
                                            'explanation': f"Suspicious file: {filename} ({file_ext}) | Risk: {final_risk_score:.1f}",
                                            'model_contributions': {
                                                'rule_based': {
                                                    'score': final_risk_score,
                                                    'indicators': threat_indicators[:3]
                                                }
                                            }
                                        }
                                        
                                        # CRITICAL: Add to agent's anomaly queue for GUI sync
                                        if hasattr(self.anomaly_detector, 'anomaly_alerts'):
                                            self.anomaly_detector.anomaly_alerts.append(alert_dict)
                                        
                                        print(f"  [FILESYSTEM] Generated alert for file: {filename} (risk={final_risk_score:.1f})")
                                        
                                self.last_scan[filepath] = mtime
                            
                            except (OSError, PermissionError):
                                continue
                
                except (OSError, PermissionError):
                    continue
            
            # Check for rapid file creation
            recent_creations = sum(1 for t in self.file_creation_rate if current_time - t < 60)
            if recent_creations > 20:
                for record in fs_data:
                    if record['event_type'] == 'created':
                        record['risk_score'] = min(10.0, record['risk_score'] + 2.0)
                        record['threat_indicators'].append('rapid_file_creation')
            
            self.anomaly_detector.update_baseline('file', 'events', self.file_event_count)
            fs_data.sort(key=lambda x: x['risk_score'], reverse=True)
        
        except Exception as e:
            print(f"Filesystem collection error: {e}")
        
        print(f"  [FILESYSTEM] Collected: {len(fs_data)} files, {reported_count} sampled")
        
        return fs_data[:50]
    
    def _is_potentially_suspicious(self, filename_lower, file_ext, filepath):
        """Quick check if file warrants deeper analysis"""
        # High-risk extensions always analyzed
        if file_ext in self.high_risk_extensions:
            return True
        
        # Ransomware extensions
        if file_ext in self.ransomware_extensions:
            return True
        
        # Suspicious patterns in name
        if any(pattern in filename_lower for pattern in self.suspicious_patterns[:10]):
            return True
        
        # Hidden files in user directories
        if filename_lower.startswith('.') and 'appdata' not in filepath.lower():
            return True
        
        return False

    def _calculate_comprehensive_file_risk(self, filename, filename_lower, file_ext, 
                                       filepath, file_size, is_new, time_since_create):
        """Calculate comprehensive file risk - BALANCED """
        
        risk = 0.0
        indicators = []
        filepath_lower = filepath.lower()

        # ============================================================
        # SAFE PATHS (WHITELIST)
        # ============================================================
        if any(safe in filepath_lower for safe in self.safe_paths):

            # → Your version allowed .enc lower risk in safe locations
            if file_ext in self.ransomware_extensions and file_ext != '.enc':
                indicators.append(f'ransomware_extension_safe_location:{file_ext}')
                return 9.0, indicators   # ↑ : 8 → 9

            if file_ext == '.enc':
                return 0.5, ['encrypted_file_safe_location']

            return 0.0, ['safe_location']

        # ============================================================
        # CRITICAL MALWARE KEYWORDS
        # ============================================================
        malware_keywords_critical = [
            'mimikatz','keylog','ransom','cryptolocker','wannacry','petya','locky','cerber',
            'trojan','backdoor','rootkit','virus','worm','spyware','stealer','rat','botnet'
        ]

        for keyword in malware_keywords_critical:
            if keyword in filename_lower:
                # →  bump (your: 9.0 → : 9.5)
                indicators.append(f'critical_malware_keyword:{keyword}')
                return 9.5, indicators

        # ============================================================
        # HIGH-RISK MALWARE KEYWORDS
        # ============================================================
        malware_keywords_high = [
            'crack','keygen','patch','loader','activator','hack','exploit','payload','inject',
            'dump','pwdump','gsecdump','wce','lazagne'
        ]

        for keyword in malware_keywords_high:
            if keyword in filename_lower:
                # →  lowered severity (your: 7.0 → : 6.0)
                risk += 6.0
                indicators.append(f'high_risk_keyword:{keyword}')
                break

        # ============================================================
        # EXTENSION-BASED RISK
        # ============================================================
        if file_ext in self.high_risk_extensions:
            risk += 2.0  # ↓ : 4 → 2
            indicators.append(f'high_risk_extension:{file_ext}')

        # ============================================================
        # RANSOMWARE EXTENSIONS 
        # ============================================================
        if file_ext in self.ransomware_extensions:

            # Encoded ransomware indicator
            if file_ext == '.enc':
                if any(x in filepath_lower for x in ['downloads','desktop','documents']):
                    risk += 8.5  # consistent with  scoring
                    indicators.append(f'enc_file_user_directory:{file_ext}')
                elif is_new and time_since_create < 60:
                    risk += 7.0  # ↑ normalized
                    indicators.append(f'recent_enc_file:{file_ext}')

            # True ransomware extension
            else:
                risk += 9.5  # ↑ unified high score
                indicators.append(f'ransomware_extension:{file_ext}')

        # ============================================================
        # DOUBLE EXTENSION TRICKS
        # ============================================================
        for trick in self.double_extension_tricks:
            if filename_lower.endswith(trick):
                risk += 8.0
                indicators.append(f'double_extension_trick:{trick}')
                break

        # Hidden file with fake extension
        if filename.count('.') > 1:
            parts = filename.rsplit('.', 2)
            if len(parts) == 3:
                if parts[1].lower() in ['pdf','doc','docx','xls','xlsx','jpg','png','txt'] and \
                parts[2].lower() in ['exe','scr','bat','cmd','vbs','js']:
                    risk += 7.0
                    indicators.append('hidden_executable_extension')

        # ============================================================
        # SUSPICIOUS FILENAME PATTERNS
        # ============================================================
        pattern_matches = [p for p in self.suspicious_patterns if p in filename_lower]

        if len(pattern_matches) >= 2:
            risk += 7.0   # ↓ your 8.0 → balance
            indicators.append(f'multiple_suspicious_keywords:{",".join(pattern_matches[:3])}')

        elif len(pattern_matches) == 1:
            risk += 4.0   # ↓ your 5.0
            indicators.append(f'suspicious_keyword:{pattern_matches[0]}')

        # ============================================================
        # LOCATION-BASED RISK ( reduced)
        # ============================================================

        # TEMP folder
        if 'temp' in filepath_lower or 'tmp' in filepath_lower:

            if file_ext in self.high_risk_extensions:
                risk += 2.0  # ↓ your 4.0
                indicators.append('executable_in_temp')

            elif file_ext in ['.ps1','.bat','.cmd','.vbs']:
                risk += 3.0  # ↓ your 5.0
                indicators.append('script_in_temp')

        # Downloads folder (new + high risk executable)
        if 'download' in filepath_lower and file_ext in self.high_risk_extensions and is_new:
            risk += 2.0  # ↓ your 3.0
            indicators.append('new_executable_in_downloads')

        # Startup = persistence
        if any(x in filepath_lower for x in ['startup','autostart']):
            risk += 5.0  # ↓ your 6.0
            indicators.append('persistence:startup_folder')

        # Recycle Bin abuse
        if '$recycle.bin' in filepath_lower and file_ext in self.high_risk_extensions:
            risk += 6.0  # ↓ your 7.0
            indicators.append('executable_in_recycle_bin')

        # System32 drop
        if ('windows\\system32' in filepath_lower or 'windows\\syswow64' in filepath_lower) and \
            file_ext in ['.exe','.dll'] and is_new:
            risk += 4.0  # ↓ your 5.0
            indicators.append('new_file_in_system32')

        # ============================================================
        # SIZE-BASED ANOMALIES
        # ============================================================
        if file_ext in ['.exe','.dll']:
            if file_size < 10 * 1024:
                risk += 2.0  # ↓ your 3.0
                indicators.append('tiny_executable')

            elif file_size > 500 * 1024 * 1024:
                risk += 1.5  # ↓ your 2.0
                indicators.append('unusually_large_executable')

        # ============================================================
        # TIMING-BASED (NEW + RECENT EXECUTABLE)
        # ============================================================
        if is_new and time_since_create < 10 and file_ext in self.high_risk_extensions:
            risk += 1.5  # ↓ your 2.0
            indicators.append('very_recent_executable')

        # ============================================================
        # HIDDEN FILES (Linux/macOS)
        # ============================================================
        if filename_lower.startswith('.') and platform.system() != 'Windows':
            if file_ext in self.high_risk_extensions:
                risk += 2.0  # ↓ your 3.0
                indicators.append('hidden_executable')

        # ============================================================
        # Windows Hidden/System Attributes
        # ============================================================
        try:
            if platform.system() == 'Windows':
                import ctypes
                attrs = ctypes.windll.kernel32.GetFileAttributesW(filepath)
                if attrs != -1:

                    if attrs & 0x2 and file_ext in self.high_risk_extensions:
                        risk += 2.0  # ↓ your 3.0
                        indicators.append('hidden_attribute_executable')

                    if attrs & 0x4 and is_new:
                        risk += 3.0  # ↓ your 4.0
                        indicators.append('new_system_attribute_file')

        except:
            pass

        return min(risk, 10.0), indicators


class UserActivityCollector:
    """Collect user activity events with anomaly detection - ENHANCED"""
    
    def __init__(self, anomaly_detector):
        self.known_sessions = {}  # Track {session_key: metadata}
        self.failed_login_count = defaultdict(int)
        self.last_collection = None
        self.anomaly_detector = anomaly_detector
        
        # User behavior baseline
        self.user_login_times = defaultdict(list)  # {username: [login_times]}
        self.user_source_ips = defaultdict(set)    # {username: {known_ips}}
        
        # Suspicious patterns
        self.brute_force_threshold = 5  # Failed logins in short time
        self.concurrent_session_threshold = 3
        
    def collect(self):
        """Collect user activity with ML-DRIVEN anomaly detection"""
        user_data = []
        current_time = datetime.now()
        
        remote_login_count = 0
        privilege_count = 0
        
        try:
            current_users = psutil.users()
            current_session_keys = set()
            concurrent_sessions = defaultdict(int)
            
            for user in current_users:
                concurrent_sessions[user.name] += 1
            
            for user in current_users:
                session_key = f"{user.name}:{user.terminal or 'console'}:{user.host or 'local'}"
                current_session_keys.add(session_key)
                
                # Only report NEW sessions
                if session_key not in self.known_sessions:
                    login_time = datetime.fromtimestamp(user.started) if user.started else current_time
                    
                    is_privileged = self._is_privileged_user(user.name)
                    if is_privileged:
                        privilege_count += 1
                    
                    if user.host and user.host not in ['local', 'localhost', '127.0.0.1']:
                        remote_login_count += 1
                    
                    hour = login_time.hour
                    
                    # ═══════════════════════════════════════════════════════
                    # ✅ BUILD ML FEATURE VECTOR
                    # ═══════════════════════════════════════════════════════
                    feature_vector = np.array([
                        1.0 if is_privileged else 0.0,
                        1.0 if user.host and user.host not in ['local', 'localhost', '127.0.0.1'] else 0.0,
                        float(concurrent_sessions[user.name]) / 5.0,
                        1.0 if hour < 6 or hour > 22 else 0.0,
                        float(hour) / 24.0,
                        1.0 if user.host and user.host in self.user_source_ips.get(user.name, set()) else 0.0,
                        float(len(self.user_login_times.get(user.name, []))) / 20.0,
                        float(remote_login_count) / max(len(current_users), 1),
                        float(privilege_count) / max(len(current_users), 1),
                        float(len(current_users)) / 10.0,
                        1.0 if len(self.user_login_times.get(user.name, [])) > 0 else 0.0,
                        float(datetime.now().weekday()) / 7.0
                    ])
                    
                    feature_names = [
                        'privileged_account', 'remote_login', 'concurrent_sessions',
                        'off_hours', 'hour_of_day', 'known_ip', 'login_frequency',
                        'remote_login_ratio', 'privilege_ratio', 'total_users',
                        'has_history', 'day_of_week'
                    ]
                    
                    # ═══════════════════════════════════════════════════════
                    # ✅ STEP 1: ML ENSEMBLE DETECTION (PRIMARY)
                    # ═══════════════════════════════════════════════════════
                    ml_is_anomaly = False
                    ml_risk_score = 0.0
                    ml_indicators = []
                    
                    if (self.anomaly_detector.isolation_forest is not None or 
                        (self.anomaly_detector.autoencoder and self.anomaly_detector.autoencoder.is_trained)):
                        
                        try:
                            is_anomaly, anomaly_info = self.anomaly_detector.detect_anomaly_ensemble(
                                feature_vector,
                                feature_names
                            )
                            
                            if is_anomaly:
                                ml_is_anomaly = True
                                severity_map = {'high': 9.0, 'medium': 6.5, 'low': 4.5}
                                ml_risk_score = severity_map.get(anomaly_info['severity'], 5.0)
                                ml_indicators = [f"ml_{ind}" for ind in anomaly_info.get('contributing_features', [])[:3]]
                        
                        except Exception as e:
                            print(f"  [ML] User activity detection error: {e}")
                    
                    # ═══════════════════════════════════════════════════════
                    # ✅ STEP 2: RULE-BASED (ONLY IF ML DIDN'T DETECT)
                    # ═══════════════════════════════════════════════════════
                    rule_risk_score = 0.0
                    rule_indicators = []
                    
                    if not ml_is_anomaly:  # ✅ Only evaluate rules if ML didn't flag
                        # Rule 1: Impossible travel (new IP within 1 hour)
                        if user.host and user.host not in ['local', 'localhost', '127.0.0.1']:
                            if user.name in self.user_source_ips:
                                if user.host not in self.user_source_ips[user.name]:
                                    recent_logins = [t for t in self.user_login_times.get(user.name, []) 
                                                if (login_time - t).total_seconds() < 3600]
                                    if len(recent_logins) > 0:
                                        rule_risk_score += 7.0
                                        rule_indicators.append('impossible_travel')
                            
                            self.user_source_ips.setdefault(user.name, set()).add(user.host)
                        
                        # Rule 2: Privileged + Remote + Off-hours
                        if is_privileged and user.host not in ['local', 'localhost', '127.0.0.1']:
                            if hour < 6 or hour > 22:
                                rule_risk_score += 6.0
                                rule_indicators.append('privileged_remote_offhours')
                        
                        # Rule 3: Very high concurrent sessions
                        if concurrent_sessions[user.name] >= 5:
                            rule_risk_score += 5.0
                            rule_indicators.append(f'excessive_concurrent_sessions:{concurrent_sessions[user.name]}')
                        
                        # Rule 4: Brute force detection (check failed login history)
                        failed_count = self.failed_login_count.get(user.host or 'local', 0)
                        if failed_count >= self.brute_force_threshold:
                            rule_risk_score += 7.0
                            rule_indicators.append(f'brute_force_detected:{failed_count}')
                    
                    # ═══════════════════════════════════════════════════════
                    # ✅ STEP 3: COMBINE DETECTIONS (ML PRIORITY)
                    # ═══════════════════════════════════════════════════════
                    final_risk_score = 0.0
                    detection_method = 'baseline'
                    threat_indicators = []
                    
                    if ml_is_anomaly:
                        # ML detected anomaly - HIGHEST PRIORITY
                        final_risk_score = ml_risk_score
                        threat_indicators = ml_indicators
                        detection_method = 'ml'
                        
                    elif rule_risk_score > 5.0:
                        # Rule-based detection - MEDIUM PRIORITY
                        final_risk_score = rule_risk_score
                        threat_indicators = rule_indicators
                        detection_method = 'rule'
                        
                    else:
                        # Baseline monitoring - LOW PRIORITY
                        if is_privileged or remote_login_count > 0:
                            final_risk_score = 2.0
                            threat_indicators = ['monitored_session']
                            detection_method = 'baseline'
                        else:
                            final_risk_score = 1.0
                            threat_indicators = []
                            detection_method = 'baseline'
                    
                    # Track login times
                    self.user_login_times.setdefault(user.name, []).append(login_time)
                    if len(self.user_login_times[user.name]) > 20:
                        self.user_login_times[user.name] = self.user_login_times[user.name][-20:]
                    
                    # Store session
                    self.known_sessions[session_key] = {
                        'first_seen': current_time,
                        'username': user.name,
                        'source_ip': user.host or 'local'
                    }
                    
                    # ═══════════════════════════════════════════════════════
                    # ✅ REPORT EVENT
                    # ═══════════════════════════════════════════════════════
                    record = {
                        'event_type': 'login',
                        'username': user.name,
                        'session_id': str(user.terminal) if user.terminal else 'console',
                        'source_ip': user.host if user.host else 'local',
                        'login_success': True,
                        'privilege_escalation': is_privileged,
                        'risk_score': final_risk_score,
                        'threat_indicators': threat_indicators,
                        'login_time': login_time.isoformat(),
                        'concurrent_sessions': concurrent_sessions[user.name],
                        'detection_method': detection_method
                    }
                    
                    user_data.append(record)
                    
                    if ml_is_anomaly or rule_risk_score > 5.0:
                        print(f"  [USER] Anomalous session: {user.name} from {user.host or 'local'} "
                            f"(risk={final_risk_score:.1f}, method={detection_method})")
            
            # Detect LOGOUTS
            for session_key in list(self.known_sessions.keys()):
                if session_key not in current_session_keys:
                    session_meta = self.known_sessions[session_key]
                    username = session_meta['username']
                    
                    user_data.append({
                        'event_type': 'logout',
                        'username': username,
                        'session_id': session_key.split(':')[1],
                        'source_ip': session_meta['source_ip'],
                        'login_success': True,
                        'privilege_escalation': False,
                        'risk_score': 0.0,
                        'threat_indicators': [],
                        'session_duration': (current_time - session_meta['first_seen']).total_seconds(),
                        'detection_method': 'none'
                    })
                    del self.known_sessions[session_key]
            
            self.last_collection = current_time
        
        except Exception as e:
            print(f"User activity collection error: {e}")
        
        return user_data
    
    def _is_privileged_user(self, username):
        """Check if user has elevated privileges"""
        username_lower = username.lower()
        return username_lower in ['root', 'administrator', 'admin', 'system', 'sudo']
    
    def _calculate_user_risk(self, username, source_ip, login_time, concurrent_count):
        """Calculate risk score for user activity"""
        risk = 0.0
        indicators = []
        
        # Privileged users
        if self._is_privileged_user(username):
            risk += 2.0
            indicators.append('privileged_account')
        
        # Remote access (non-local)
        if source_ip and source_ip not in ['local', 'localhost', '127.0.0.1', '::1', '']:
            risk += 2.0
            indicators.append('remote_login')
            
            # Check if new source IP for this user
            if username in self.user_source_ips:
                if source_ip not in self.user_source_ips[username]:
                    risk += 2.0
                    indicators.append('new_source_ip')
                    self.user_source_ips[username].add(source_ip)
            else:
                self.user_source_ips[username] = {source_ip}
        
        # Multiple concurrent sessions
        if concurrent_count >= self.concurrent_session_threshold:
            risk += 3.0
            indicators.append(f'multiple_sessions:{concurrent_count}')
        
        # Off-hours login (outside 6 AM - 10 PM)
        hour = login_time.hour
        if hour < 6 or hour > 22:
            risk += 1.5
            indicators.append(f'off_hours_login:{hour}h')
        
        return min(risk, 10.0), indicators
    
    def _detect_login_anomaly(self, username, source_ip, login_time):
        """Detect anomalous login patterns"""
        # Track login times for this user
        if username not in self.user_login_times:
            self.user_login_times[username] = []
        
        self.user_login_times[username].append(login_time)
        
        # Keep only last 20 logins
        if len(self.user_login_times[username]) > 20:
            self.user_login_times[username] = self.user_login_times[username][-20:]
        
        # Check for rapid successive logins (< 5 minutes apart)
        if len(self.user_login_times[username]) > 1:
            last_login = self.user_login_times[username][-2]
            time_diff = (login_time - last_login).total_seconds()
            
            if time_diff < 300:  # < 5 minutes
                return True
        
        # Check for unusual hour (statistical anomaly)
        hour = login_time.hour
        if len(self.user_login_times[username]) >= 5:
            # Calculate typical login hours
            typical_hours = [lt.hour for lt in self.user_login_times[username][:-1]]
            avg_hour = sum(typical_hours) / len(typical_hours)
            
            # If this login is > 6 hours different from typical
            if abs(hour - avg_hour) > 6:
                return True
        
        return False

class ClientAgent:
    """Main client agent with complete federated learning implementation"""
        
    def __init__(self):
        global CLIENT_ID
        CLIENT_ID = SystemInfo.get_client_id()
        
        self.running = False
        
        # ✅ CRITICAL FIX: Create feature manager FIRST
        self.feature_manager = FeatureWindowManager(window_duration=60, max_windows=1000)
        
        # ✅ CRITICAL FIX: Create anomaly detector SECOND
        self.anomaly_detector = EnhancedAnomalyDetector(use_ocsvm=False)
        
        # ✅ CRITICAL FIX: Link feature manager to anomaly detector
        self.anomaly_detector.feature_manager = self.feature_manager
        
        # ✅ CRITICAL FIX: Create collectors THIRD with both dependencies
        self.collectors = {
            'network': EnhancedNetworkCollector(self.anomaly_detector),
            'process': EnhancedProcessCollector(self.anomaly_detector),
            'filesystem': EnhancedFilesystemCollector(self.anomaly_detector),
            'user': UserActivityCollector(self.anomaly_detector)
        }
        
        # ✅ CRITICAL FIX: Link feature_manager to ALL collectors
        for collector_name, collector in self.collectors.items():
            collector.feature_manager = self.feature_manager
            print(f"  ✓ Linked feature_manager to {collector_name} collector")
        
        # Rest of __init__ remains the same...
        self.anomaly_alerts = deque(maxlen=100)
        self.models_fetched_from_server = False
        self.last_global_model = None
        
        self.fl_upload_callback = None  # Will be set by GUI
        
        print(f"FortifAI Client Agent (Enhanced with Full ML Pipeline)")
        print(f"Client ID: {CLIENT_ID}")
        print(f"Hostname: {socket.gethostname()}")
        print(f"OS: {platform.system()} {platform.version()}")
        print(f"Server: {SERVER_HOST}:{SERVER_PORT}")
        print(f"Federated Learning: {'Enabled' if ENABLE_FL else 'Disabled'}")
        print(f"ML Models: IsolationForest + Autoencoder + Z-Score")
        print("-" * 50)
    
    def load_models_from_server(self, cached_models):
        """Load previously trained models from server cache"""
        try:
            print("\n[MODEL CACHE] Loading models from server...")
            
            models_loaded = False
            
            # Load Isolation Forest
            if 'isolation_forest' in cached_models and cached_models['isolation_forest']:
                try:
                    # Server sends the pickled model as bytes
                    iso_forest_bytes = cached_models['isolation_forest']
                    self.anomaly_detector.isolation_forest = pickle.loads(iso_forest_bytes)
                    print("  ✓ Isolation Forest restored from cache")
                    models_loaded = True
                except Exception as e:
                    print(f"  ✗ Failed to load Isolation Forest: {e}")
                    if self.anomaly_detector.autoencoder:
                        self.anomaly_detector.autoencoder.is_trained = False
                        
            # Load Autoencoder weights
            if 'autoencoder_config' in cached_models and cached_models['autoencoder_config']:
                try:
                    ae_config = cached_models['autoencoder_config']
                    
                    # Build autoencoder if needed
                    if self.anomaly_detector.autoencoder is None:
                        input_dim = ae_config.get('input_dim', 14)
                        self.anomaly_detector.autoencoder = AutoencoderAnomalyDetector(input_dim=input_dim)
                    
                    # Restore weights
                    if 'weights' in ae_config:
                        self.anomaly_detector.autoencoder.set_weights(ae_config['weights'])
                    
                    # Restore scaler
                    if 'scaler_mean' in ae_config and 'scaler_scale' in ae_config:
                        self.anomaly_detector.autoencoder.scaler.mean_ = np.array(ae_config['scaler_mean'])
                        self.anomaly_detector.autoencoder.scaler.scale_ = np.array(ae_config['scaler_scale'])
                        self.anomaly_detector.autoencoder.is_trained = True
                    
                    # Restore threshold
                    if 'threshold' in ae_config:
                        self.anomaly_detector.ae_threshold = ae_config['threshold']
                    
                    print("  – Autoencoder restored from cache")
                    models_loaded = True
                except Exception as e:
                    print(f"  ✗ Failed to load Autoencoder: {e}")
            
            # Restore last training time
            if 'last_training_time' in cached_models:
                self.anomaly_detector.last_training_time = cached_models['last_training_time']
            
            # Restore feature buffer (optional, for continuity)
            if 'feature_buffer' in cached_models and cached_models['feature_buffer']:
                try:
                    # Restore last 50 windows
                    for feature in cached_models['feature_buffer'][-50:]:
                        self.feature_manager.feature_buffer.append(feature)
                    print(f"  – Restored {len(cached_models['feature_buffer'][-50:])} feature windows")
                except Exception as e:
                    print(f"  ⚠️  Could not restore feature buffer: {e}")
            
            if models_loaded:
                iso_ok = self.anomaly_detector.isolation_forest is not None
                ae_ok = (self.anomaly_detector.autoencoder and 
                        self.anomaly_detector.autoencoder.is_trained)
                
                if not (iso_ok or ae_ok):
                    print("[MODEL CACHE] ⚠️  Models loaded but verification failed")
                    return False
                
                print(f"[MODEL CACHE] – Verified: ISO={iso_ok}, AE={ae_ok}")
                print("[MODEL CACHE] – Models successfully loaded from server")
                if hasattr(self, 'gui') and self.gui:
                    self.gui.add_log("– Pre-trained models loaded from server cache")
                return True
            else:
                print("[MODEL CACHE] ⓘ No cached models available")
                return False
                
        except Exception as e:
            print(f"[MODEL CACHE] ✗ Error loading models: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def register_with_server(self):
        """Register client with server and fetch any cached models - ENHANCED"""
        try:
            client_info = SystemInfo.get_system_info()
            
            data = {
                'type': 'registration',
                'client_info': client_info,
                'capabilities': {
                    'federated_learning': ENABLE_FL,
                    'anomaly_detection': True
                },
                'request_cached_models': True
            }
            
            response = self.send_to_server(data)
            if response and response.get('status') == 'registered':
                print("✓ Successfully registered with server")
                
                # ✅ FIX: Apply global FL model weights IMMEDIATELY
                if 'model_weights' in response:
                    print("[FL] Received global model weights from server")
                    self.anomaly_detector.update_model_parameters(response['model_weights'])
                    self._apply_fl_parameters_to_detector()
                    print(f"[FL] ✓ Applied global model v{response['model_weights'].get('version', 0)}")
                
                # Check for cached models
                if 'cached_models' in response and response['cached_models']:
                    models_loaded = self.load_models_from_server(response['cached_models'])
                    
                    if models_loaded:
                        iso_ok = self.anomaly_detector.isolation_forest is not None
                        ae_ok = (self.anomaly_detector.autoencoder and 
                                self.anomaly_detector.autoencoder.is_trained)
                        
                        if iso_ok or ae_ok:
                            self.models_fetched_from_server = True
                            print(f"[CACHE] ✓ Models verified: ISO={iso_ok}, AE={ae_ok}")
                        else:
                            self.models_fetched_from_server = False
                            print(f"[CACHE] ✗ Models loaded but verification failed")
                    else:
                        self.models_fetched_from_server = False
                        print("[CACHE] ✗ No usable cached models")
                else:
                    print("[CACHE] ℹ No cached models available on server")
                    self.models_fetched_from_server = False
                
                return True
            else:
                print("✗ Registration failed")
                return False
        
        except Exception as e:
            print(f"✗ Registration error: {e}")
            return False

    def _apply_fl_parameters_to_detector(self):
        """Apply federated learning parameters to local anomaly detector - NEW METHOD"""
        try:
            weights = self.anomaly_detector.model_weights
            
            # ✅ Apply thresholds to detection logic
            print(f"\n[FL APPLICATION] Applying global model parameters:")
            print(f"  - Network Threshold: {weights.get('network_threshold', 2.0):.3f}")
            print(f"  - Process Threshold: {weights.get('process_threshold', 2.0):.3f}")
            print(f"  - File Threshold: {weights.get('file_threshold', 2.0):.3f}")
            print(f"  - Network Sensitivity: {weights.get('network_sensitivity', 1.0):.3f}")
            print(f"  - Process Sensitivity: {weights.get('process_sensitivity', 1.0):.3f}")
            print(f"  - File Sensitivity: {weights.get('file_sensitivity', 1.0):.3f}")
            
            # ✅ Update z-score threshold
            avg_threshold = (
                weights.get('network_threshold', 2.0) +
                weights.get('process_threshold', 2.0) +
                weights.get('file_threshold', 2.0)
            ) / 3.0
            
            self.anomaly_detector.zscore_threshold = avg_threshold * 2.5
            print(f"  - Z-Score Threshold: {self.anomaly_detector.zscore_threshold:.3f}")
            
            # ✅ Update baseline means/stds
            self.anomaly_detector.model_weights['network_baseline_mean'] = weights.get('network_baseline_mean', 0.0)
            self.anomaly_detector.model_weights['network_baseline_std'] = weights.get('network_baseline_std', 1.0)
            self.anomaly_detector.model_weights['process_baseline_mean'] = weights.get('process_baseline_mean', 0.0)
            self.anomaly_detector.model_weights['process_baseline_std'] = weights.get('process_baseline_std', 1.0)
            
            print(f"[FL APPLICATION] ✓ Parameters applied successfully")
            
            if hasattr(self, 'gui') and self.gui:
                self.gui.add_log(f"✓ FL parameters applied (thresholds, sensitivities updated)")
            
        except Exception as e:
            print(f"[FL APPLICATION] ✗ Error applying parameters: {e}")

    
    def send_to_server(self, data):
        """Send data to server - FIXED FOR LARGE RESPONSES"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            sock.connect((SERVER_HOST, SERVER_PORT))
            
            # Serialize data
            serialized = pickle.dumps(data)
            data_size = len(serialized)
            
            # Send size first
            sock.send(data_size.to_bytes(8, 'big'))
            
            # Send data
            sock.sendall(serialized)
            
            # –… FIX: Receive response size first
            response_size_data = sock.recv(8)
            if len(response_size_data) < 8:
                raise ConnectionError("Failed to receive response size")
            
            response_size = int.from_bytes(response_size_data, 'big')
            
            # –… FIX: Receive complete response in chunks
            response_data = b''
            while len(response_data) < response_size:
                chunk = sock.recv(min(4096, response_size - len(response_data)))
                if not chunk:
                    break
                response_data += chunk
            
            if len(response_data) < response_size:
                raise ConnectionError(f"Incomplete response: {len(response_data)}/{response_size} bytes")
            
            response = pickle.loads(response_data)
            
            sock.close()
            return response
        
        except Exception as e:
            print(f"✗ Communication error: {e}")
            return None
    
    # --- NEW: Background ML training loop ---
    def ml_training_loop(self):
        """Background thread for periodic ML model training - ENHANCED"""
        while self.running:
            try:
                # ✅ Finalize window every 60 seconds
                if self.feature_manager.should_finalize_window():
                    feature_vector = self.feature_manager.finalize_window()
                    
                    buffer_size = len(self.feature_manager.feature_buffer)
                    print(f"[ML] Window finalized → {buffer_size} total windows in buffer")
                    
                    # ✅ ADD: Use performance guard for training
                    if buffer_size >= 20 and not self.anomaly_detector.isolation_forest:
                        if self.anomaly_detector.perf_guard.can_start_training():
                            if self.anomaly_detector.perf_guard.acquire_training_lock():
                                try:
                                    print(f"\n[ML TRAINING] ⚡ IMMEDIATE TRAINING with {buffer_size} samples...")
                                    feature_matrix, feature_names = self.feature_manager.get_feature_matrix()
                                    
                                    if feature_matrix is not None and len(feature_matrix) >= 20:
                                        success = self.anomaly_detector.train_models(feature_matrix)
                                        
                                        if success:
                                            print(f"✓ ML models trained successfully!")
                                            if hasattr(self, 'gui') and self.gui:
                                                self.gui.add_log(f"✓ Models trained ({len(feature_matrix)} samples)")
                                            
                                            self.send_models_to_server_cache()
                                        else:
                                            print(f"✗ ML training failed")
                                finally:
                                    self.anomaly_detector.perf_guard.release_training_lock()
                    
                    # ✅ Existing retraining logic also uses guard
                    elif self.anomaly_detector.should_retrain(buffer_size):
                        if self.anomaly_detector.perf_guard.can_start_training(force=False):
                            if self.anomaly_detector.perf_guard.acquire_training_lock():
                                try:
                                    feature_matrix, feature_names = self.feature_manager.get_feature_matrix()
                                    
                                    if feature_matrix is not None and len(feature_matrix) >= 20:
                                        print(f"\n[ML TRAINING] Starting with {len(feature_matrix)} samples...")
                                        success = self.anomaly_detector.train_models(feature_matrix)
                                        
                                        if success:
                                            print(f"✓ ML models retrained successfully")
                                            if hasattr(self, 'gui') and self.gui:
                                                self.gui.add_log(f"✓ Models retrained ({len(feature_matrix)} samples)")
                                            
                                            self.send_models_to_server_cache()
                                finally:
                                    self.anomaly_detector.perf_guard.release_training_lock()
                
                # ✅ ADD: Periodic cache cleanup
                if int(time.time()) % 300 == 0:  # Every 5 minutes
                    self.anomaly_detector.perf_guard.cleanup_cache()
                
                # Sleep for 10 seconds
                time.sleep(10)
                
            except Exception as e:
                print(f"✗ ML training loop error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(30)

    
    # --- NEW: Generate anomaly alert with explainability ---
    def generate_anomaly_alert(self, feature_vector, anomaly_info, feature_names):
        """Generate detailed anomaly alert - deduplicated + explainability"""

        # –… Build alert signature for duplicate suppression
        contrib = anomaly_info.get('contributing_features', [])
        severity = anomaly_info.get('severity', 'unknown')

        alert_signature = f"{severity}_{','.join(contrib[:3])}"

        # –… Check last 10 alerts for duplicates
        recent_signatures = [
            f"{a.get('severity','')}_{','.join(a.get('contributing_features',[])[:3])}"
            for a in list(self.anomaly_alerts)[-10:]
        ]

        if alert_signature in recent_signatures:
            print(f"  [ALERT] Duplicate anomaly suppressed: {alert_signature}")
            return  # –… STOP HERE

        # =====================================================
        # –… Build explainability text
        # =====================================================
        explanation_parts = []

        if anomaly_info.get('zscore_flag'):
            explanation_parts.append("Extreme Z-score deviation")

        if anomaly_info.get('iso_score') is not None:
            explanation_parts.append(f"Isolation Forest score: {anomaly_info['iso_score']:.4f}")

        if anomaly_info.get('ae_recon_error') is not None:
            explanation_parts.append(
                f"Autoencoder reconstruction error: {anomaly_info['ae_recon_error']:.4f}"
            )

        if contrib:
            explanation_parts.append(
                f"Top contributing features: {', '.join(contrib[:5])}"
            )

        alert = {
            'timestamp': datetime.now().isoformat(),
            'hostname': socket.gethostname(),
            'client_id': CLIENT_ID,
            'severity': severity,
            'is_anomaly': anomaly_info.get('is_anomaly'),
            'zscore_flag': anomaly_info.get('zscore_flag'),
            'iso_score': anomaly_info.get('iso_score'),
            'ae_recon_error': anomaly_info.get('ae_recon_error'),
            'contributing_features': contrib,
            'feature_vector': {k: v for k, v in feature_vector.items() if k != 'timestamp'},
            'explanation': " | ".join(explanation_parts),
            'signature': alert_signature   # –… stored for future pattern detection
        }

        # –… Add alert AFTER duplicate suppression
        self.anomaly_alerts.append(alert)

        # =====================================================
        # –… Output
        # =====================================================
        print(f"\n{'='*60}")
        print(f"⚠️ ANOMALY DETECTED - Severity: {severity.upper()}")
        print(f"{'='*60}")
        print(f"Time: {alert['timestamp']}")
        print(f"Explanation: {alert['explanation']}")
        print(f"{'='*60}\n")

        # –… GUI logging
        if hasattr(self, 'gui') and self.gui:
            self.gui.add_log(f"⚠️ ANOMALY DETECTED - {severity.upper()}")
            self.gui.add_log(f"   {alert['explanation']}")

            
    def collect_and_send(self):
        """Collect telemetry and send to server with REAL ML anomaly detection"""
        try:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Collecting telemetry...")
            
            if hasattr(self, 'gui') and self.gui:
                self.gui.add_log("📊 Collecting telemetry data...")
            
            # ✅ CRITICAL FIX: Collect data FIRST before any processing
            network_data = self.collectors['network'].collect()
            process_data = self.collectors['process'].collect()
            filesystem_data = self.collectors['filesystem'].collect()
            user_data = self.collectors['user'].collect()
            
            # ✅ CRITICAL FIX: Verify data collection
            print(f"  [DEBUG] Collected: Network={len(network_data.get('all_events', []))}, "
                f"Process={len(process_data.get('details', []))}, "
                f"Filesystem={len(filesystem_data)}, User={len(user_data)}")
            
            # ✅ Update feature window AFTER collection
            if 'all_events' in network_data:
                for event in network_data['all_events']:
                    self.feature_manager.update_network_event(event)
            
            if (self.anomaly_detector.isolation_forest is not None or
                (self.anomaly_detector.autoencoder and self.anomaly_detector.autoencoder.is_trained)):
                
                # Get current feature matrix
                feature_matrix, feature_names = self.feature_manager.get_feature_matrix()
                
                if feature_matrix is not None and len(feature_matrix) > 0:
                    # Test detection on latest window
                    latest_features = feature_matrix[-1]
                    
                    try:
                        is_anomaly, anomaly_info = self.anomaly_detector.detect_anomaly_ensemble(
                            latest_features,
                            feature_names
                        )
                        
                        if is_anomaly:
                            print(f"  [DETECTION] ⚠️ Anomaly detected in latest window!")
                            print(f"    Severity: {anomaly_info.get('severity')}")
                            print(f"    Features: {anomaly_info.get('contributing_features', [])[:3]}")
                    except Exception as e:
                        print(f"  [DETECTION] Error: {e}")
                        
            if 'details' in process_data:
                for event in process_data['details']:
                    self.feature_manager.update_process_event(event)
            
            if 'high_risk' in process_data:
                for event in process_data['high_risk']:
                    self.feature_manager.update_process_event(event)
            
            for event in filesystem_data:
                self.feature_manager.update_filesystem_event(event)
            
            # ✅ ML anomaly detection (happens after data collection)
            ml_anomaly_alerts = []
            
            # Check for ML-detected anomalies in collected data
            if 'all_events' in network_data:
                for event in network_data['all_events']:
                    if event.get('detection_method') == 'ml' and event.get('is_anomaly'):
                        ml_anomaly_alerts.append({
                            'timestamp': datetime.now().isoformat(),
                            'category': 'network',
                            'severity': 'high' if event['risk_score'] > 8 else 'medium',
                            'iso_score': event.get('anomaly_score'),
                            'ae_recon_error': None,
                            'contributing_features': [ind.replace('ml_', '') for ind in event.get('threat_indicators', []) if ind.startswith('ml_')],
                            'explanation': f"ML-detected anomalous network connection to {event.get('dst_ip')}:{event.get('dst_port')}"
                        })
            
            # ✅ CRITICAL FIX: Build telemetry payload with ALL collected data
            recent_anomaly_alerts = []
            if hasattr(self.anomaly_detector, 'anomaly_alerts'):
                # Get last 50 alerts
                recent_anomaly_alerts = list(self.anomaly_detector.anomaly_alerts)[-50:]
            
            telemetry = {
                'type': 'telemetry',
                'client_id': CLIENT_ID,
                'timestamp': datetime.now().isoformat(),
                'network': network_data,
                'processes': process_data,
                'filesystem': filesystem_data,
                'user_activity': user_data,
                'ml_anomaly_alerts': ml_anomaly_alerts,  # ✅ Keep existing
                'anomaly_alerts': recent_anomaly_alerts  # ✅ NEW: Send all alerts
            }
            
            # ✅ FIX: Clear sent alerts to prevent duplicates
            if hasattr(self.anomaly_detector, 'anomaly_alerts'):
                # Keep only last 10 for GUI display
                alerts_to_keep = list(self.anomaly_detector.anomaly_alerts)[-10:]
                self.anomaly_detector.anomaly_alerts.clear()
                for alert in alerts_to_keep:
                    self.anomaly_detector.anomaly_alerts.append(alert)
            
            # ✅ Enhanced logging
            print(f"  Network: {len(network_data.get('details', []))} events")
            print(f"  Process: {len(process_data.get('details', []))} monitored, "
                f"{len(process_data.get('high_risk', []))} high-risk")
            print(f"  Filesystem: {len(filesystem_data)} events")
            print(f"  User: {len(user_data)} events")
            print(f"  Feature Buffer: {len(self.feature_manager.feature_buffer)} windows")
            print(f"  ML Anomalies: {len(ml_anomaly_alerts)}")
            
            iso_trained = "✓" if self.anomaly_detector.isolation_forest else "✗"
            ae_trained = "✓" if (self.anomaly_detector.autoencoder and 
                                 self.anomaly_detector.autoencoder.is_trained) else "✗"
            print(f"  ML Models: ISO={iso_trained}, AE={ae_trained}")
            
            # ✅ Send to server
            response = self.send_to_server(telemetry)
            if response and response.get('status') == 'received':
                print("✓ Telemetry sent successfully")
                if hasattr(self, 'gui') and self.gui:
                    self.gui.add_log("✓ Telemetry sent successfully")
                
                if 'alerts' in response:
                    for alert in response['alerts']:
                        print(f"⚠️ ALERT: {alert}")
                        if hasattr(self, 'gui') and self.gui:
                            self.gui.add_log(f"⚠️ ALERT: {alert}")
            else:
                print("✗ Failed to send telemetry")
        
        except Exception as e:
            print(f"✗ Collection error: {e}")
            import traceback
            traceback.print_exc()

    def _build_anomaly_explanation(self, anomaly_info, feature_names, feature_values):
        """Build human-readable explanation of anomaly"""
        parts = []
        
        if anomaly_info.get('zscore_flag'):
            parts.append("Extreme statistical deviation detected")
        
        if anomaly_info.get('iso_score') is not None:
            parts.append(f"Isolation Forest score: {anomaly_info['iso_score']:.4f}")
        
        if anomaly_info.get('ae_recon_error') is not None:
            parts.append(f"Autoencoder error: {anomaly_info['ae_recon_error']:.4f}")
        
        # Add top contributing features with VALUES
        contrib = anomaly_info.get('contributing_features', [])
        if contrib:
            feature_details = []
            for feat in contrib[:3]:
                try:
                    idx = feature_names.index(feat)
                    val = feature_values[idx]
                    feature_details.append(f"{feat}={val:.2f}")
                except:
                    feature_details.append(feat)
            
            parts.append(f"Key features: {', '.join(feature_details)}")
        
        return " | ".join(parts)

    def send_models_to_server_cache(self):
        """Send trained models to server for caching (called after FL updates)"""
        try:
            cached_models = {}
            
            # Serialize Isolation Forest
            if self.anomaly_detector.isolation_forest is not None:
                try:
                    iso_bytes = pickle.dumps(self.anomaly_detector.isolation_forest)
                    cached_models['isolation_forest'] = iso_bytes
                except Exception as e:
                    print(f"  âš  Could not serialize Isolation Forest: {e}")
            
            # Serialize Autoencoder
            if self.anomaly_detector.autoencoder and self.anomaly_detector.autoencoder.is_trained:
                try:
                    ae_config = {
                        'input_dim': self.anomaly_detector.autoencoder.input_dim,
                        'weights': self.anomaly_detector.autoencoder.get_weights(),
                        'threshold': self.anomaly_detector.ae_threshold,
                        'scaler_mean': self.anomaly_detector.autoencoder.scaler.mean_.tolist() if hasattr(self.anomaly_detector.autoencoder.scaler, 'mean_') else None,
                        'scaler_scale': self.anomaly_detector.autoencoder.scaler.scale_.tolist() if hasattr(self.anomaly_detector.autoencoder.scaler, 'scale_') else None
                    }
                    cached_models['autoencoder_config'] = ae_config
                except Exception as e:
                    print(f"  âš  Could not serialize Autoencoder: {e}")
            
            # Add metadata
            cached_models['last_training_time'] = self.anomaly_detector.last_training_time
            cached_models['feature_buffer'] = list(self.feature_manager.feature_buffer)[-50:]  # Last 50 windows
            
            # Send to server
            if cached_models:
                data = {
                    'type': 'cache_models',
                    'client_id': CLIENT_ID,
                    'timestamp': datetime.now().isoformat(),
                    'cached_models': cached_models
                }
                
                response = self.send_to_server(data)
                
                if response and response.get('status') == 'models_cached':
                    print("  – Models cached on server for future sessions")
                else:
                    print("  âš  Model caching may have failed")
                    
        except Exception as e:
            print(f"  ✗ Error caching models: {e}")

    
    def send_fl_update(self):
        """Send enhanced federated learning model update with full metadata"""
        if not ENABLE_FL:
            return
        
        try:
            if self.fl_upload_callback:
                self.fl_upload_callback("📤 Preparing FL model update...\n")
    
            if hasattr(self, 'gui') and self.gui:
                self.gui.add_log("- Preparing FL model update...")
            # First, adapt sensitivity locally
            self.anomaly_detector.adapt_sensitivity_locally()
            
            # Get feature matrix for metadata
            feature_matrix, feature_names = self.feature_manager.get_feature_matrix()
            
            # Get comprehensive model parameters
            model_params = self.anomaly_detector.get_model_parameters()
            
            # --- NEW: Compute model delta if we have a global model ---
            model_delta = None
            delta_metadata = None
            
            if self.last_global_model is not None:
                model_delta, delta_metadata = self.anomaly_detector.compute_fl_delta(
                    self.last_global_model['weights']
                )
            
            # Build FL update payload
            data = {
                'type': 'fl_update',
                'client_id': CLIENT_ID,
                'timestamp': datetime.now().isoformat(),
                'model_parameters': model_params,
                'model_delta': model_delta,  # NEW: Send delta instead of full weights
                'delta_metadata': delta_metadata if model_delta else None,
                'ml_model_status': {
                    'isolation_forest_trained': self.anomaly_detector.isolation_forest is not None,
                    'autoencoder_trained': (self.anomaly_detector.autoencoder is not None and 
                                        self.anomaly_detector.autoencoder.is_trained),
                    'feature_buffer_size': len(self.feature_manager.feature_buffer),
                    'last_training_time': self.anomaly_detector.last_training_time
                }
            }
            
            print(f"\n{'='*60}")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔥 Sending Enhanced FL Update")
            print(f"{'='*60}")
            
            response = self.send_to_server(data)
            
            # Validate server response
            if response and response.get('status') == 'fl_received':
                print("– Server acknowledged FL update")
                
                status_text = "✅ Server acknowledged FL update\n"
            
                # ✅ NEW: Send to FL Upload Status box
                if self.fl_upload_callback:
                    self.fl_upload_callback(status_text)
                
                print(status_text)

                # Receive aggregated model
                if 'aggregated_weights' in response:
                    old_version = self.anomaly_detector.model_weights.get('version', 0)
                    self.anomaly_detector.update_model_parameters(response['aggregated_weights'])
                    
                    # ✅ FIX: Apply parameters immediately
                    self._apply_fl_parameters_to_detector()
                    
                    new_version = response['aggregated_weights'].get('version', 0)

                    update_msg = f"✅ Model updated: v{old_version} → v{new_version}\n"
                
                    if self.fl_upload_callback:
                        self.fl_upload_callback(update_msg)
                        
                    if hasattr(self, 'gui') and self.gui:
                        self.gui.add_log(f"✓ FL update sent | Version: {new_version}")

                    self.last_global_model = response['aggregated_weights']

                
                # NEW: Send trained models to server for caching
                if (self.anomaly_detector.isolation_forest is not None or (self.anomaly_detector.autoencoder and self.anomaly_detector.autoencoder.is_trained)):
                    self.send_models_to_server_cache()
                
                # Display data quality                    
                quality = model_params.get('data_quality', {})
                anomaly_rate = model_params.get('anomaly_rate', 0)
                
                print(f"✓ FL update sent successfully")
                print(f"  Data Quality:")
                print(f"    Network samples: {quality.get('network_samples', 0)}")
                print(f"    Process samples: {quality.get('process_samples', 0)}")
                print(f"    File samples: {quality.get('file_samples', 0)}")
                print(f"    Feature windows: {len(self.feature_manager.feature_buffer)}")
                print(f"  Anomaly Rate: {anomaly_rate:.2%}")
                
                if model_delta and delta_metadata:
                    print(f"  Model Delta:")
                    print(f"    Delta norm: {delta_metadata['delta_norm']:.4f}")
                    print(f"    Samples used: {delta_metadata['samples_used']}")
                
                print(f"  ML Models:")
                print(f"    Isolation Forest: {'–' if data['ml_model_status']['isolation_forest_trained'] else '✗'}")
                print(f"    Autoencoder: {'– ' if data['ml_model_status']['autoencoder_trained'] else '✗ '}")
                
                # Receive aggregated model
                if 'aggregated_weights' in response:
                    old_version = self.anomaly_detector.model_weights.get('version', 0)
                    self.anomaly_detector.update_model_parameters(response['aggregated_weights'])
                    new_version = response['aggregated_weights'].get('version', 0)
                    
                    # Store global model for next delta computation
                    self.last_global_model = response['aggregated_weights']
                    
                    # Update autoencoder weights if available
                    if 'autoencoder' in response['aggregated_weights'].get('weights', {}):
                        ae_weights = response['aggregated_weights']['weights']['autoencoder']
                        if self.anomaly_detector.autoencoder is not None:
                            self.anomaly_detector.autoencoder.set_weights(ae_weights)
                            print(f"  ✓ Autoencoder weights updated")
                    
                    if new_version > old_version:
                        print(f"  ✓ New global model received: v{new_version}")
                    else:
                        print(f"  ✓ Model confirmed: v{new_version}")
            else:
                print(f"✗  FL update failed")
            
            print(f"{'='*60}\n")
        
        except Exception as e:
            print(f"✗ FL update error: {e}")
            import traceback
            traceback.print_exc() 
                     
    def send_heartbeat(self):
        """Send heartbeat to server"""
        try:
            data = {
                'type': 'heartbeat',
                'client_id': CLIENT_ID,
                'timestamp': datetime.now().isoformat()
            }
            
            response = self.send_to_server(data)
            if response and response.get('status') == 'alive':
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ❤️  Heartbeat sent")
        
        except Exception as e:
            print(f"✗ Heartbeat error: {e}")
    
    def heartbeat_loop(self):
        """Background heartbeat thread"""
        while self.running:
            time.sleep(HEARTBEAT_INTERVAL)
            if self.running:
                self.send_heartbeat()
    
    def collection_loop(self):
        """Main collection loop"""
        while self.running:
            time.sleep(COLLECTION_INTERVAL)
            if self.running:
                self.collect_and_send()
    
    def fl_loop(self):
        """Federated learning update loop with immediate first update"""
        # Send first update immediately after 30 seconds
        time.sleep(30)
        if self.running and ENABLE_FL:
            print("\n[FL] Sending initial model update...")
            self.send_fl_update()
        
        # Then continue with regular interval
        while self.running:
            time.sleep(FL_UPDATE_INTERVAL)
            if self.running and ENABLE_FL:
                self.send_fl_update()
    
    def start(self):
        """Start the agent with all threads"""
        print("\nStarting FortifAI Client Agent with Enhanced ML Pipeline...")
        
        # Register with server
        if not self.register_with_server():
            print("✗ Failed to register with server. Retrying in 10 seconds...")
            time.sleep(10)
            if not self.register_with_server():
                print("Cannot connect to server. Exiting.")
                return
        
        self.running = True
        
        # ✅ FIX: Link anomaly_detector alerts to agent-level queue
        # This ensures GUI can access them
        self.anomaly_detector.anomaly_alerts = self.anomaly_alerts
        
        def delayed_initial_training():
            """Bootstrap training - skip if models already loaded from server"""
            time.sleep(180)
            
            if self.models_fetched_from_server:
                print(f"\n[BOOTSTRAP] ✓ Server provided cached models during registration")
                time.sleep(2)
                
                iso_loaded = self.anomaly_detector.isolation_forest is not None
                ae_loaded = (self.anomaly_detector.autoencoder and 
                            self.anomaly_detector.autoencoder.is_trained)
                
                models_applied = iso_loaded or ae_loaded
                
                if models_applied:
                    print(f"[BOOTSTRAP] ✓ Models verified:")
                    print(f"  - Isolation Forest: {'✓' if iso_loaded else '✗'}")
                    print(f"  - Autoencoder: {'✓' if ae_loaded else '✗'}")
                    print(f"[BOOTSTRAP] ✓ Skipping initial training")
                    if hasattr(self, 'gui') and self.gui:
                        self.gui.add_log("✓ Using cached models from server")
                    return
                else:
                    print(f"[BOOTSTRAP] ⚠ Models flag set but not loaded - will train fresh")
                    print(f"  Debug: ISO={iso_loaded}, AE={ae_loaded}")
                    self.models_fetched_from_server = False
            
            feature_matrix, feature_names = self.feature_manager.get_feature_matrix()
            
            print(f"\n[BOOTSTRAP CHECK] Buffer has {len(self.feature_manager.feature_buffer)} windows")
            
            if feature_matrix is not None and len(feature_matrix) >= 20:
                print(f"\n[BOOTSTRAP] Forcing initial training with {len(feature_matrix)} samples...")
                success = self.anomaly_detector.train_models(feature_matrix)
                
                if success:
                    print("[BOOTSTRAP] ✓ Models trained successfully!")
                    if hasattr(self, 'gui') and self.gui:
                        self.gui.add_log(f"✓ ML Models trained: {len(feature_matrix)} samples")
                    
                    self.send_models_to_server_cache()
                else:
                    print("[BOOTSTRAP] ✗ Training failed")
            else:
                actual_size = len(feature_matrix) if feature_matrix is not None else 0
                print(f"[BOOTSTRAP] Insufficient data: {actual_size}/20 samples needed")
        
        # Start threads
        bootstrap_thread = threading.Thread(target=delayed_initial_training, daemon=True)
        heartbeat_thread = threading.Thread(target=self.heartbeat_loop, daemon=True)
        collection_thread = threading.Thread(target=self.collection_loop, daemon=True)
        fl_thread = threading.Thread(target=self.fl_loop, daemon=True)
        training_thread = threading.Thread(target=self.ml_training_loop, daemon=True)
        
        # ✅ FIX: Start GUI refresh thread
        if hasattr(self, 'gui') and self.gui:
            gui_refresh_thread = threading.Thread(target=self.gui_refresh_loop, daemon=True)
            gui_refresh_thread.start()
        
        heartbeat_thread.start()
        collection_thread.start()
        fl_thread.start()
        training_thread.start()
        bootstrap_thread.start()
        
        print("\n✓ Agent is running")
        print("✓ ML training loop active")
        print("Press Ctrl+C to stop\n")
        
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\nStopping agent...")
            self.running = False
            time.sleep(2)
            print("✓ Agent stopped")

    def gui_refresh_loop(self):
        """Periodically refresh GUI anomaly monitor - ENHANCED"""
        while self.running:
            time.sleep(2)  # Refresh every 2 seconds (faster response)
            if hasattr(self, 'gui') and self.gui:
                try:
                    # ✅ FIX: Force refresh on GUI thread
                    if hasattr(self.gui, 'root') and self.gui.root.winfo_exists():
                        self.gui.root.after(0, self.gui.refresh_anomalies)
                except Exception as e:
                    print(f"GUI refresh error: {e}")
    
    def stop(self):
        """Stop the agent"""
        self.running = False


class ClientAIAssistant:
    """Client-side AI assistant for anomaly analysis"""
    
    def __init__(self):
        try:
            GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
            if not GEMINI_API_KEY:
                print("⚠️  GEMINI_API_KEY not found in .env file")
                self.available = False
                return
            
            genai.configure(api_key=GEMINI_API_KEY)
            from client.utils.config import GEMINI_MODEL
            self.model = genai.GenerativeModel(GEMINI_MODEL)
            self.conversation_history = []
            self.system_context = """You are a cybersecurity expert AI assistant helping users understand 
            security anomalies detected on their local system. Provide clear, actionable explanations in plain English.
            
            When analyzing anomalies, provide:
            1. **What Happened**: Plain English explanation
            2. **Why It Matters**: Risk assessment
            3. **What To Do**: Specific remediation steps
            4. **Prevention**: How to prevent recurrence
            
            Be concise, avoid jargon, and prioritize user safety."""
            self.available = True
            print("✓ Client AI Assistant initialized")
        except Exception as e:
            print(f"⚠️  AI Assistant not available: {e}")
            self.available = False
    
    def analyze_anomaly(self, anomaly_data):
        """Analyze a specific anomaly and provide detailed explanation - FIXED"""
        if not self.available:
            return "AI Assistant is not available. Please check your GEMINI_API_KEY in .env file."
        
        try:
            # Build context from anomaly
            context = f"""
    Analyze this security anomaly detected on the local system:

    **Anomaly Details:**
    - Timestamp: {anomaly_data.get('timestamp', 'Unknown')}
    - Severity: {anomaly_data.get('severity', 'Unknown')}
    - Category: {anomaly_data.get('category', 'Unknown')}
    - Threat Level: {anomaly_data.get('ensemble_score', 0):.1f}/10
    - Explanation: {anomaly_data.get('explanation', 'N/A')}
    - Evidence: {anomaly_data.get('evidence', 'N/A')}

    **Contributing Features:**
    {', '.join(anomaly_data.get('contributing_features', []))}

    **Recommended Action:**
    {anomaly_data.get('recommended_action', 'Monitor system')}

    Provide a comprehensive analysis following the structure:
    1. What Happened (plain English)
    2. Why It Matters (risk level and impact)
    3. What To Do (specific steps)
    4. Prevention (how to avoid future occurrences)
    """
            
            response = self.model.generate_content(context)
            return self._format_markdown_to_plain(response.text)
        
        except Exception as e:
            return f"Error generating analysis: {str(e)}"
    
    def chat(self, user_message, context_data=None):
        """General chat interface"""
        if not self.available:
            return "AI Assistant is not available. Please check your GEMINI_API_KEY."
        
        try:
            full_message = self.system_context + "\n\n"
            
            if context_data:
                full_message += f"**Current Context:**\n{context_data}\n\n"
            
            full_message += f"**User Question:** {user_message}"
            
            response = self.model.generate_content(full_message)
            
            # Store in history
            self.conversation_history.append({
                'user': user_message,
                'assistant': response.text,
                'timestamp': datetime.now().isoformat()
            })
            
            return self._format_markdown_to_plain(response.text)
        
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _format_markdown_to_plain(self, text):
        """
        Convert markdown formatting to plain text with better readability
        Removes ** ## *** and replaces with readable formatting
        """
        import re
        
        # Replace headers (##, ###)
        text = re.sub(r'^###\s+(.+)$', r'   \1', text, flags=re.MULTILINE)
        text = re.sub(r'^##\s+(.+)$', r'\n━━━ \1 ━━━', text, flags=re.MULTILINE)
        text = re.sub(r'^#\s+(.+)$', r'\n═══ \1 ═══', text, flags=re.MULTILINE)
        
        # Replace bold (**text**)
        text = re.sub(r'\*\*(.+?)\*\*', r'[\1]', text)
        
        # Replace italic (*text*)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        
        # Replace bullet points
        text = re.sub(r'^\*\s+', r'  • ', text, flags=re.MULTILINE)
        text = re.sub(r'^\-\s+', r'  • ', text, flags=re.MULTILINE)
        
        # Replace numbered lists
        text = re.sub(r'^(\d+)\.\s+', r'\n\1. ', text, flags=re.MULTILINE)
        
        # Clean up code blocks
        text = re.sub(r'```[\w]*\n', r'\n', text)
        text = re.sub(r'```', r'', text)
        
        # Clean up excessive newlines
        text = re.sub(r'\n{3,}', r'\n\n', text)
        
        return text.strip()
    
# --- NEW: Desktop GUI for client agent ---
class ClientGUI:
    """Tkinter-based GUI for client agent monitoring and control"""
    
    def __init__(self, agent):
        self.agent = agent
        self.root = tk.Tk()
        self.root.title(f"FortifAI Client Agent - {socket.gethostname()}")
        self.root.geometry("1200x800")
        self.ai_assistant = ClientAIAssistant()

        # Update queue for thread-safe GUI updates
        self.update_queue = queue.Queue()
        
        # Style configuration
        self.colors = {
            'bg': '#2c3e50',
            'fg': '#ecf0f1',
            'accent': '#3498db',
            'success': '#2ecc71',
            'warning': '#f39c12',
            'danger': '#e74c3c',
            'panel': '#34495e'
        }
        
        self.root.configure(bg=self.colors['bg'])
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create tabs
        self.create_dashboard_tab()
        self.create_anomaly_monitor_tab()
        self.create_fl_status_tab()
        self.create_logs_tab()
        self.create_settings_tab()
        
        self.agent.anomaly_detector.gui_callback = self.on_backend_event
        self.agent.fl_upload_callback = self.on_fl_upload_status
        
        # Status bar
        self.status_bar = tk.Label(
            self.root, 
            text="Status: Initializing...", 
            bd=1, 
            relief=tk.SUNKEN, 
            anchor=tk.W,
            bg=self.colors['panel'],
            fg=self.colors['fg']
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Start update loop
        self.update_gui()
    
    def on_backend_event(self, event_type, data):
        """Thread-safe callback from backend (called from worker threads)"""
        # Queue the event for processing in GUI thread
        self.update_queue.put(('backend_event', event_type, data))

    def on_fl_upload_status(self, status_text):
        """Thread-safe callback for FL upload status"""
        # Queue the status update for processing in GUI thread
        self.update_queue.put(('fl_status', status_text))

    def process_update_queue(self):
        """Process queued updates in GUI thread"""
        try:
            while True:
                try:
                    item = self.update_queue.get_nowait()
                    
                    if item[0] == 'backend_event':
                        event_type, data = item[1], item[2]
                        
                        if event_type == 'anomaly':
                            self.handle_anomaly_event(data)
                    
                    elif item[0] == 'fl_status':
                        status_text = item[1]
                        self.handle_fl_status_update(status_text)
                    
                except queue.Empty:
                    break
        except Exception as e:
            print(f"Queue processing error: {e}")

    def handle_anomaly_event(self, alert):
        """Handle anomaly alert in GUI thread with ENHANCED DETAILS"""
        try:
            # Refresh anomaly list
            self.refresh_anomalies()

            # Extract Isolation Forest score
            iso_score_display = "N/A"
            if "isolation_forest" in model_contrib:
                iso_raw = model_contrib["isolation_forest"].get("raw_score")
                if iso_raw is not None:
                    iso_score_display = f"{iso_raw:.4f}"

            # Extract Autoencoder error
            ae_error_display = "N/A"
            if "autoencoder" in model_contrib:
                ae_recon = model_contrib["autoencoder"].get("recon_error")
                if ae_recon is not None:
                    ae_error_display = f"{ae_recon:.6f}"

            # Only update detail panel if no item selected
            if not self.anomaly_tree.selection():
                # âœ… BUILD HUMAN-READABLE DETAIL TEXT
                detail_text = f"""
    {'='*80}
        ANOMALY DETECTED - {alert.get('severity', 'unknown').upper()}
    {'='*80}

    📅 Time:     {alert.get('timestamp', 'Unknown')}
    🏷️  Category: {alert.get('category', 'Unknown').upper()}
     Severity: {alert.get('severity', 'unknown').upper()}
     Threat Level: {alert.get('ensemble_score', 0):.1f}/10

    ──────────────────────────────────────────────────────────────────────────────
     WHAT HAPPENED
    ──────────────────────────────────────────────────────────────────────────────
    {alert.get('explanation', 'No explanation available')}

    ──────────────────────────────────────────────────────────────────────────────
    ❓ WHY IT MATTERS
    ──────────────────────────────────────────────────────────────────────────────
    {alert.get('evidence', 'No evidence available')}

    ──────────────────────────────────────────────────────────────────────────────
     WHAT TO DO
    ──────────────────────────────────────────────────────────────────────────────
    {alert.get('recommended_action', 'Monitor system')}

    ──────────────────────────────────────────────────────────────────────────────
     AI MODEL ANALYSIS
    ──────────────────────────────────────────────────────────────────────────────
    """
                
                # Add model contributions
                model_contrib = alert.get('model_contributions', {})
                if model_contrib:
                    for model_name, data in model_contrib.items():
                        score = data.get('score', 0)
                        detail_text += f"  • {model_name.replace('_', ' ').title()}: {score:.1f}/10\n"
                else:
                    detail_text += "  (No AI model data available)\n"
                
                # Add key features
                detail_text += f"\n──────────────────────────────────────────────────────────────────────────────\n"
                detail_text += f"🔑 KEY INDICATORS\n"
                detail_text += f"──────────────────────────────────────────────────────────────────────────────\n"
                
                contrib = alert.get('contributing_features', [])
                if contrib:
                    for feat in contrib[:5]:
                        # Translate technical features to human-readable
                        readable_feat = self._translate_feature_name(feat)
                        detail_text += f"  • {readable_feat}\n"
                else:
                    detail_text += "  (No specific indicators identified)\n"
                
                self.anomaly_detail_text.delete("1.0", tk.END)
                self.anomaly_detail_text.insert("1.0", detail_text)

            # Add log with human-readable summary
            self.add_log(
                f"{alert.get('category', 'SYSTEM').upper()}: "
                f"{alert.get('explanation', 'Anomaly detected')[:80]}"
            )

        except Exception as e:
            print(f"Error handling anomaly event: {e}")
            import traceback
            traceback.print_exc()
            
    def _translate_feature_name(self, feature_name):
        """Translate technical feature names to human-readable descriptions"""
        translations = {
            'conn_count': 'High number of network connections',
            'unique_dst_count': 'Multiple destination IPs accessed',
            'port_entropy': 'Port scanning behavior detected',
            'c2_pattern': 'Command & Control communication pattern',
            'proc_spawn_count': 'Rapid process creation',
            'avg_proc_cpu': 'High CPU usage',
            'avg_proc_memory': 'High memory consumption',
            'file_create_count': 'Mass file creation/modification',
            'ransomware_burst': 'Ransomware encryption pattern',
            'cred_dump_pattern': 'Credential theft attempt',
            'delta_conn_count': 'Sudden change in connection rate',
            'bytes_sent': 'High data upload detected',
            'bytes_recv': 'High data download detected',
            'tcp_ratio': 'Unusual network protocol distribution',
            'dst_churn_rate': 'Rapid switching between destinations',
        }
        
        return translations.get(feature_name, feature_name.replace('_', ' ').title())

    def handle_fl_status_update(self, status_text):
        """Handle FL status update in GUI thread"""
        try:
            # Append to FL Upload Status text box
            self.fl_upload_text.insert(tk.END, status_text)
            self.fl_upload_text.see(tk.END)
            
            # Also add to logs
            self.add_log(status_text.strip())
            
        except Exception as e:
            print(f"Error handling FL status: {e}")
            
    def create_dashboard_tab(self):
        """Create dashboard overview tab"""
        dashboard = ttk.Frame(self.notebook)
        self.notebook.add(dashboard, text="🏛️ Dashboard")

        # Top frame - Agent status
        status_frame = tk.LabelFrame(
            dashboard, 
            text="Agent Status", 
            bg=self.colors['panel'],
            fg=self.colors['fg'],
            font=('Arial', 12, 'bold')
        )
        status_frame.pack(fill='x', padx=10, pady=5)
        
        # Status indicators
        status_grid = tk.Frame(status_frame, bg=self.colors['panel'])
        status_grid.pack(fill='x', padx=10, pady=10)
        
        self.status_labels = {}
        
        # Agent running status
        self.status_labels['running'] = self.create_status_indicator(
            status_grid, "Agent Status:", "Running", self.colors['success'], 0, 0
        )
        
        # Server connection
        self.status_labels['server'] = self.create_status_indicator(
            status_grid, "Server:", "Connected", self.colors['success'], 0, 2
        )
        
        # FL status
        self.status_labels['fl'] = self.create_status_indicator(
            status_grid, "FL Status:", "Active", self.colors['accent'], 1, 0
        )
        
        # Model version
        self.status_labels['model_version'] = self.create_status_indicator(
            status_grid, "Model Version:", "0", self.colors['fg'], 1, 2
        )
        
        # System metrics frame
        metrics_frame = tk.LabelFrame(
            dashboard,
            text="System Metrics",
            bg=self.colors['panel'],
            fg=self.colors['fg'],
            font=('Arial', 12, 'bold')
        )
        metrics_frame.pack(fill='x', padx=10, pady=5)
        
        metrics_grid = tk.Frame(metrics_frame, bg=self.colors['panel'])
        metrics_grid.pack(fill='x', padx=10, pady=10)
        
        # CPU and RAM
        self.status_labels['cpu'] = self.create_metric_display(
            metrics_grid, "CPU Usage:", "0%", 0, 0
        )
        
        self.status_labels['ram'] = self.create_metric_display(
            metrics_grid, "RAM Usage:", "0%", 0, 2
        )
        
        # Network connections
        self.status_labels['connections'] = self.create_metric_display(
            metrics_grid, "Active Connections:", "0", 1, 0
        )
        
        # Running processes
        self.status_labels['processes'] = self.create_metric_display(
            metrics_grid, "Monitored Processes:", "0", 1, 2
        )
        
        # ML Model status frame
        ml_frame = tk.LabelFrame(
            dashboard,
            text="ML Model Status",
            bg=self.colors['panel'],
            fg=self.colors['fg'],
            font=('Arial', 12, 'bold')
        )
        ml_frame.pack(fill='x', padx=10, pady=5)
        
        ml_grid = tk.Frame(ml_frame, bg=self.colors['panel'])
        ml_grid.pack(fill='x', padx=10, pady=10)
        
        self.status_labels['iso_forest'] = self.create_status_indicator(
            ml_grid, "Isolation Forest:", "Not Trained", self.colors['warning'], 0, 0
        )
        
        self.status_labels['autoencoder'] = self.create_status_indicator(
            ml_grid, "Autoencoder:", "Not Trained", self.colors['warning'], 0, 2
        )
        
        self.status_labels['feature_buffer'] = self.create_metric_display(
            ml_grid, "Feature Windows:", "0", 1, 0
        )
        
        self.status_labels['anomaly_rate'] = self.create_metric_display(
            ml_grid, "Anomaly Rate:", "0.0%", 1, 2
        )
    
    def create_status_indicator(self, parent, label_text, value_text, color, row, col):
        """Create a status indicator widget"""
        frame = tk.Frame(parent, bg=self.colors['panel'])
        frame.grid(row=row, column=col, padx=20, pady=5, sticky='w')
        
        label = tk.Label(
            frame,
            text=label_text,
            bg=self.colors['panel'],
            fg=self.colors['fg'],
            font=('Arial', 10)
        )
        label.pack(side='left')
        
        value = tk.Label(
            frame,
            text=value_text,
            bg=self.colors['panel'],
            fg=color,
            font=('Arial', 10, 'bold')
        )
        value.pack(side='left', padx=5)
        
        return value
    
    def create_metric_display(self, parent, label_text, value_text, row, col):
        """Create a metric display widget"""
        frame = tk.Frame(parent, bg=self.colors['panel'])
        frame.grid(row=row, column=col, padx=20, pady=5, sticky='w')
        
        label = tk.Label(
            frame,
            text=label_text,
            bg=self.colors['panel'],
            fg=self.colors['fg'],
            font=('Arial', 10)
        )
        label.pack(side='left')
        
        value = tk.Label(
            frame,
            text=value_text,
            bg=self.colors['panel'],
            fg=self.colors['accent'],
            font=('Arial', 10, 'bold')
        )
        value.pack(side='left', padx=5)
        
        return value
    
    def analyze_selected_anomaly(self):
        """Analyze selected anomaly with AI (CLIENT VERSION) - FIXED"""
        selection = self.anomaly_tree.selection()
        if not selection:
            self.ai_analysis_text.delete('1.0', tk.END)
            self.ai_analysis_text.insert('1.0', "⚠️ No anomaly selected. Please select an anomaly to analyze.")
            return
        
        # Show loading
        self.ai_analysis_text.delete('1.0', tk.END)
        self.ai_analysis_text.insert('1.0', "🔄 Analyzing anomaly with AI...\n\nPlease wait...")
        self.root.update()
        
        try:
            item = selection[0]
            item_id = int(self.anomaly_tree.item(item, 'text')) - 1
            
            # ✅ FIX: Properly access anomaly data
            anomalies = []
            if hasattr(self.agent, 'anomaly_detector') and hasattr(self.agent.anomaly_detector, 'anomaly_alerts'):
                anomalies = list(self.agent.anomaly_detector.anomaly_alerts)
            elif hasattr(self.agent, 'anomaly_alerts'):
                anomalies = list(self.agent.anomaly_alerts)
            
            if not anomalies or item_id < 0 or item_id >= len(anomalies):
                self.ai_analysis_text.delete('1.0', tk.END)
                self.ai_analysis_text.insert('1.0', 
                    "❌ Error: Could not retrieve anomaly data\n\n"
                    f"Debug Info:\n"
                    f"  - Selected ID: {item_id}\n"
                    f"  - Total Anomalies: {len(anomalies)}\n"
                    f"  - Has detector: {hasattr(self.agent, 'anomaly_detector')}\n"
                    f"  - Has alerts: {bool(anomalies)}"
                )
                return
            
            alert = anomalies[item_id]
            
            # ✅ FIX: Build proper alert data structure for AI
            alert_data = {
                'timestamp': alert.get('timestamp', 'Unknown'),
                'severity': alert.get('severity', 'unknown'),
                'category': alert.get('category', 'system'),
                'ensemble_score': alert.get('ensemble_score', 0),
                'explanation': alert.get('explanation', 'No explanation available'),
                'evidence': alert.get('evidence', 'No evidence available'),
                'contributing_features': alert.get('contributing_features', []),
                'recommended_action': alert.get('recommended_action', 'Monitor system')
            }
            
            # Get AI analysis
            analysis = self.ai_assistant.analyze_anomaly(alert_data)
            
            # Display result
            result = f"""
    ════════════════════════════════════════════════════════════
                AI ANALYSIS REPORT
    ════════════════════════════════════════════════════════════

    [Alert] {alert_data['category'].upper()}: {alert_data['severity'].upper()}
    [Time] {alert_data['timestamp']}
    [Threat Level] {alert_data['ensemble_score']:.1f}/10

    ════════════════════════════════════════════════════════════

    {analysis}

    ════════════════════════════════════════════════════════════
    Generated by FortifAI AI Assistant
    ════════════════════════════════════════════════════════════
    """
            
            self.ai_analysis_text.delete('1.0', tk.END)
            self.ai_analysis_text.insert('1.0', result)
        
        except Exception as e:
            self.ai_analysis_text.delete('1.0', tk.END)
            self.ai_analysis_text.insert('1.0', f"❌ Error analyzing anomaly: {str(e)}\n\n{traceback.format_exc()}")
            import traceback
            print(f"AI analysis error: {e}")
            traceback.print_exc()


    def send_ai_chat(self):
        """Send message to AI assistant (CLIENT VERSION)"""
        message = self.ai_chat_input.get().strip()
        if not message or message == "Ask the AI assistant anything about security...":
            return
        
        # Get context from selected anomaly if any
        context = None
        selection = self.anomaly_tree.selection()
        if selection:
            item = selection[0]
            item_id = int(self.anomaly_tree.item(item, 'text')) - 1
            
            if 0 <= item_id < len(self.agent.anomaly_alerts):
                alert = list(self.agent.anomaly_alerts)[item_id]
                context = f"Current Anomaly Context:\nCategory: {alert.get('category')}\nSeverity: {alert.get('severity')}\nExplanation: {alert.get('explanation')}"
        
        # Show user message
        current_text = self.ai_analysis_text.get('1.0', tk.END)
        self.ai_analysis_text.delete('1.0', tk.END)
        self.ai_analysis_text.insert('1.0', current_text + f"\n\n[You] {message}\n\n🤖 AI Assistant: Thinking...")
        self.root.update()
        
        # Get AI response
        response = self.ai_assistant.chat(message, context)
        
        # Update display
        current_text = self.ai_analysis_text.get('1.0', tk.END)
        current_text = current_text.replace("🤖 AI Assistant: Thinking...", response)
        self.ai_analysis_text.delete('1.0', tk.END)
        self.ai_analysis_text.insert('1.0', current_text)
        
        # Scroll to bottom
        self.ai_analysis_text.see(tk.END)
        
        # Clear input
        self.ai_chat_input.delete(0, tk.END)
        self.ai_chat_input.insert(0, "Ask the AI assistant anything about security...")

    def create_anomaly_monitor_tab(self):
        """Create real-time anomaly monitoring tab with AI analysis"""
        anomaly_tab = ttk.Frame(self.notebook)
        self.notebook.add(anomaly_tab, text="⚠️ Anomaly Monitor")

        # Control buttons
        control_frame = tk.Frame(anomaly_tab, bg=self.colors['panel'])
        control_frame.pack(fill='x', padx=5, pady=5)
        
        refresh_btn = tk.Button(
            control_frame,
            text="🔄 Refresh",
            command=self.refresh_anomalies,
            bg=self.colors['accent'],
            fg='white',
            font=('Arial', 10, 'bold')
        )
        refresh_btn.pack(side='left', padx=5, pady=5)
        
        clear_btn = tk.Button(
            control_frame,
            text="🗑️ Clear",
            command=self.clear_anomalies,
            bg=self.colors['danger'],
            fg='white',
            font=('Arial', 10, 'bold')
        )
        clear_btn.pack(side='left', padx=5, pady=5)
        
        # Anomaly Table
        table_frame = tk.LabelFrame(
            anomaly_tab,
            text="Detected Anomalies",
            bg=self.colors['panel'],
            fg=self.colors['fg'],
            font=('Arial', 11, 'bold')
        )
        table_frame.pack(fill='both', expand=True, padx=5, pady=(0, 5))
        
        # Scrollbar
        scrollbar = tk.Scrollbar(table_frame)
        scrollbar.pack(side='right', fill='y')
        
        # ✅ MERGED COLUMNS: Timestamp, Severity, Category, ISO, AE, Score, Features
        columns = ('Timestamp', 'Severity', 'Category', 'ISO Score', 'AE Error', 'Threat Score', 'Features')
        self.anomaly_tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show='tree headings',
            yscrollcommand=scrollbar.set,
            height=15
        )
        
        # Column headers
        self.anomaly_tree.heading('#0', text='ID')
        self.anomaly_tree.heading('Timestamp', text='Timestamp')
        self.anomaly_tree.heading('Severity', text='Severity')
        self.anomaly_tree.heading('Category', text='Category')
        self.anomaly_tree.heading('ISO Score', text='ISO Score')
        self.anomaly_tree.heading('AE Error', text='AE Error')
        self.anomaly_tree.heading('Threat Score', text='Threat Score')
        self.anomaly_tree.heading('Features', text='Top Contributing Features')
        
        # Column widths
        self.anomaly_tree.column('#0', width=40)
        self.anomaly_tree.column('Timestamp', width=160)
        self.anomaly_tree.column('Severity', width=120)
        self.anomaly_tree.column('Category', width=100)
        self.anomaly_tree.column('ISO Score', width=90)
        self.anomaly_tree.column('AE Error', width=90)
        self.anomaly_tree.column('Threat Score', width=100)
        self.anomaly_tree.column('Features', width=350)
        
        self.anomaly_tree.pack(fill='both', expand=True, padx=5, pady=5)
        scrollbar.config(command=self.anomaly_tree.yview)
        
        # Bind selection
        self.anomaly_tree.bind('<<TreeviewSelect>>', self.on_anomaly_select)
        
        # AI Analysis Panel (unchanged)
        ai_panel = tk.LabelFrame(
            anomaly_tab,
            text="🤖 AI Security Assistant",
            bg=self.colors['panel'],
            fg=self.colors['fg'],
            font=('Arial', 11, 'bold')
        )
        ai_panel.pack(fill='both', expand=True, padx=5, pady=(0, 5))
        
        self.ai_analysis_text = scrolledtext.ScrolledText(
            ai_panel,
            bg=self.colors['bg'],
            fg=self.colors['fg'],
            font=('Courier', 9),
            wrap=tk.WORD,
            height=12
        )
        self.ai_analysis_text.pack(fill='both', expand=True, padx=5, pady=(5, 5))
        self.ai_analysis_text.insert('1.0', 
            "Select an anomaly from the table above to see detailed AI analysis...\n\n"
            "Or use the chat below to ask security questions."
        )
        
        # Chat interface
        chat_frame = tk.Frame(ai_panel, bg=self.colors['panel'])
        chat_frame.pack(fill='x', padx=5, pady=(0, 5))
        
        self.ai_chat_input = tk.Entry(
            chat_frame,
            bg=self.colors['bg'],
            fg=self.colors['fg'],
            font=('Arial', 10)
        )
        self.ai_chat_input.pack(side='left', fill='x', expand=True, padx=(0, 5))
        self.ai_chat_input.insert(0, "Ask the AI assistant anything about security...")
        self.ai_chat_input.bind('<FocusIn>', lambda e: self.ai_chat_input.delete(0, tk.END) 
                            if self.ai_chat_input.get() == "Ask the AI assistant anything about security..." else None)
        self.ai_chat_input.bind('<Return>', lambda e: self.send_ai_chat())
        
        send_btn = tk.Button(
            chat_frame,
            text="Send",
            command=self.send_ai_chat,
            bg=self.colors['accent'],
            fg='white',
            font=('Arial', 9, 'bold'),
            width=8
        )
        send_btn.pack(side='left', padx=(0, 5))
        
        analyze_btn = tk.Button(
            chat_frame,
            text="🔍 Analyze Selected",
            command=self.analyze_selected_anomaly,
            bg=self.colors['success'],
            fg='white',
            font=('Arial', 9, 'bold'),
            width=15
        )
        analyze_btn.pack(side='left')
    
    def create_fl_status_tab(self):
        """Create federated learning status tab"""
        fl_tab = ttk.Frame(self.notebook)
        self.notebook.add(fl_tab, text="🤝 FL Status")
        
        # Training status
        training_frame = tk.LabelFrame(
            fl_tab,
            text="Local Training Status",
            bg=self.colors['panel'],
            fg=self.colors['fg'],
            font=('Arial', 12, 'bold')
        )
        training_frame.pack(fill='x', padx=10, pady=5)
        
        training_grid = tk.Frame(training_frame, bg=self.colors['panel'])
        training_grid.pack(fill='x', padx=10, pady=10)
        
        self.fl_labels = {}
        
        self.fl_labels['local_version'] = self.create_metric_display(
            training_grid, "Local Model Version:", "0", 0, 0
        )
        
        self.fl_labels['global_version'] = self.create_metric_display(
            training_grid, "Global Model Version:", "0", 0, 2
        )
        
        self.fl_labels['samples_used'] = self.create_metric_display(
            training_grid, "Samples Used:", "0", 1, 0
        )
        
        self.fl_labels['delta_norm'] = self.create_metric_display(
            training_grid, "Delta Norm:", "0.0", 1, 2
        )
        
        self.fl_labels['last_training'] = self.create_metric_display(
            training_grid, "Last Training:", "Never", 2, 0
        )
        
        self.fl_labels['last_update'] = self.create_metric_display(
            training_grid, "Last FL Update:", "Never", 2, 2
        )
        
        # Upload status
        upload_frame = tk.LabelFrame(
            fl_tab,
            text="FL Upload Status",
            bg=self.colors['panel'],
            fg=self.colors['fg'],
            font=('Arial', 12, 'bold')
        )
        upload_frame.pack(fill='x', padx=10, pady=5)
        
        self.fl_upload_text = scrolledtext.ScrolledText(
            upload_frame,
            height=10,
            bg=self.colors['bg'],
            fg=self.colors['fg'],
            font=('Courier', 9)
        )
        self.fl_upload_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Model weights display
        weights_frame = tk.LabelFrame(
            fl_tab,
            text="Current Model Weights",
            bg=self.colors['panel'],
            fg=self.colors['fg'],
            font=('Arial', 12, 'bold')
        )
        weights_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.fl_weights_text = scrolledtext.ScrolledText(
            weights_frame,
            bg=self.colors['bg'],
            fg=self.colors['fg'],
            font=('Courier', 9)
        )
        self.fl_weights_text.pack(fill='both', expand=True, padx=5, pady=5)
    
    def create_logs_tab(self):
        """Create logs/history tab"""
        logs_tab = ttk.Frame(self.notebook)
        self.notebook.add(logs_tab, text="📞 Logs")
        
        # Control buttons
        control_frame = tk.Frame(logs_tab, bg=self.colors['panel'])
        control_frame.pack(fill='x', padx=5, pady=5)
        
        clear_btn = tk.Button(
            control_frame,
            text="Clear Logs",
            command=self.clear_logs,
            bg=self.colors['danger'],
            fg='white',
            font=('Arial', 10, 'bold')
        )
        clear_btn.pack(side='left', padx=5, pady=5)
        
        export_btn = tk.Button(
            control_frame,
            text="Export Logs",
            command=self.export_logs,
            bg=self.colors['accent'],
            fg='white',
            font=('Arial', 10, 'bold')
        )
        export_btn.pack(side='left', padx=5, pady=5)
        
        # Log display
        self.log_text = scrolledtext.ScrolledText(
            logs_tab,
            bg=self.colors['bg'],
            fg=self.colors['fg'],
            font=('Courier', 9)
        )
        self.log_text.pack(fill='both', expand=True, padx=5, pady=5)
    
    # --- PATCH 3: Complete create_settings_tab method ---
    def create_settings_tab(self):
        """Create settings view tab"""
        settings_tab = ttk.Frame(self.notebook)
        self.notebook.add(settings_tab, text="⚙️ Settings")
        
        # Settings display (read-only)
        settings_frame = tk.LabelFrame(
            settings_tab,
            text="Current Configuration (Read-Only)",
            bg=self.colors['panel'],
            fg=self.colors['fg'],
            font=('Arial', 12, 'bold')
        )
        settings_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.settings_text = scrolledtext.ScrolledText(
            settings_frame,
            bg=self.colors['bg'],
            fg=self.colors['fg'],
            font=('Courier', 9)
        )
        self.settings_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # --- FIXED: Properly closed triple-quoted f-string ---
        settings_info = f"""
    ┌──────────────────────────────────────────────────────────────┐
    │           FortifAI Client Agent Configuration                │
    └──────────────────────────────────────────────────────────────┘

    ─── Server Configuration ───
    FL Server URL:        {SERVER_HOST}:{SERVER_PORT}
    Connection Timeout:   10 seconds

    ─── Client Information ───
    Client ID:            {CLIENT_ID}
    Hostname:             {socket.gethostname()}
    OS:                   {platform.system()} {platform.version()}

    ─── Collection Settings ───
    Collection Interval:  {COLLECTION_INTERVAL} seconds
    Heartbeat Interval:   {HEARTBEAT_INTERVAL} seconds
    FL Update Interval:   {FL_UPDATE_INTERVAL} seconds

    ─── Federated Learning ───
    FL Enabled:           {ENABLE_FL}
    Model Type:           IsolationForest + Autoencoder + Z-Score

    ─── Feature Window Settings ───
    Window Duration:      60 seconds
    Max Windows:          1000
    Buffer Size:          Dynamic

    ─── Anomaly Detection ───
    Z-Score Threshold:    3.0
    Isolation Forest:     contamination=0.005, n_estimators=100
    Autoencoder:          64 → 32 → 8 → 32 → 64

    ─── Security & Privacy ───
    Raw Data Export:      DISABLED
    PII Collection:       DISABLED
    Local Hashing Only:   ENABLED
    L2 Clip Bound:        10.0
    """
        
        self.settings_text.insert('1.0', settings_info)
        self.settings_text.config(state='disabled')  # Make read-only

    # --- PATCH 4: Add missing GUI methods after create_settings_tab ---
    def refresh_anomalies(self):
        """Refresh anomaly list - MERGED: Extract ISO/AE and populate all columns"""
        try:
            # Clear existing items
            for item in self.anomaly_tree.get_children():
                self.anomaly_tree.delete(item)
            
            # Get anomalies
            anomalies = []
            if hasattr(self.agent, 'anomaly_detector') and hasattr(self.agent.anomaly_detector, 'anomaly_alerts'):
                anomalies = list(self.agent.anomaly_detector.anomaly_alerts)
            elif hasattr(self.agent, 'anomaly_alerts'):
                anomalies = list(self.agent.anomaly_alerts)
            
            if not anomalies:
                self.anomaly_tree.insert('', 'end', text='0', 
                                    values=('No anomalies detected', '', '', '', '', '', ''), 
                                    tags=('normal',))
                return
            
            print(f"[GUI] Refreshing {len(anomalies)} anomalies with ISO/AE columns...")
            
            for i, alert in enumerate(anomalies):
                # Extract basic info
                timestamp = alert.get('timestamp', 'Unknown')
                severity = alert.get('severity', 'normal').upper()
                category = alert.get('category', 'system').upper()
                ensemble_score = alert.get('ensemble_score', 0)
                contrib = alert.get('contributing_features', [])
                features_str = ', '.join(contrib[:3]) if contrib else 'N/A'
                
                # ✅ EXTRACT ISO SCORE (multi-level fallback)
                iso_score_display = "N/A"
                model_contrib = alert.get('model_contributions', {})
                
                if "isolation_forest" in model_contrib:
                    iso_raw = model_contrib["isolation_forest"].get("raw_score")
                    if iso_raw is not None:
                        iso_score_display = f"{iso_raw:.4f}"
                    else:
                        iso_contrib = model_contrib["isolation_forest"].get("score")
                        if iso_contrib is not None and iso_contrib > 0:
                            iso_score_display = f"{iso_contrib:.2f}"
                
                if iso_score_display == "N/A":
                    alert_iso = alert.get('iso_score')
                    if alert_iso is not None:
                        iso_score_display = f"{alert_iso:.4f}"
                
                # ✅ EXTRACT AE ERROR (multi-level fallback)
                ae_error_display = "N/A"
                if "autoencoder" in model_contrib:
                    ae_recon = model_contrib["autoencoder"].get("recon_error")
                    if ae_recon is not None:
                        ae_error_display = f"{ae_recon:.6f}"
                    else:
                        ae_contrib = model_contrib["autoencoder"].get("score")
                        if ae_contrib is not None and ae_contrib > 0:
                            ae_error_display = f"{ae_contrib:.2f}"
                
                if ae_error_display == "N/A":
                    alert_ae = alert.get('ae_recon_error')
                    if alert_ae is not None:
                        ae_error_display = f"{alert_ae:.6f}"
                
                # Build severity display
                severity_display = f"{severity}"
                threat_score_display = f"{ensemble_score:.1f}/10"
                
                # ✅ INSERT ROW WITH ALL COLUMNS
                self.anomaly_tree.insert(
                    '', 'end', 
                    text=str(i + 1),
                    values=(
                        timestamp,
                        severity_display,
                        category,
                        iso_score_display,      # ✅ NEW
                        ae_error_display,       # ✅ NEW
                        threat_score_display,
                        features_str
                    ),
                    tags=(severity.lower(),)
                )
            
            # Color coding
            self.anomaly_tree.tag_configure('critical', background='#c0392b', foreground='white')
            self.anomaly_tree.tag_configure('high', background='#e74c3c', foreground='white')
            self.anomaly_tree.tag_configure('medium', background='#f39c12', foreground='black')
            self.anomaly_tree.tag_configure('low', background='#3498db', foreground='white')
            self.anomaly_tree.tag_configure('normal', background='#ecf0f1', foreground='black')
        
            print(f"[GUI] ✓ Table populated with {len(anomalies)} items (ISO + AE columns)")
        
        except Exception as e:
            print(f"[GUI] ✗ Error refreshing anomalies: {e}")
            import traceback
            traceback.print_exc()
    
    def clear_anomalies(self):
        """Clear anomaly list"""
        for item in self.anomaly_tree.get_children():
            self.anomaly_tree.delete(item)
        
        if hasattr(self.agent, 'anomaly_alerts'):
            self.agent.anomaly_alerts.clear()
        
        self.anomaly_detail_text.delete('1.0', tk.END)
    
    def on_anomaly_select(self, event):
        """Handle anomaly selection - Display ISO/AE in AI panel"""
        selection = self.anomaly_tree.selection()
        if not selection:
            return
        
        try:
            item = selection[0]
            item_id = int(self.anomaly_tree.item(item, 'text')) - 1
            
            # Get anomalies
            anomalies = []
            if hasattr(self.agent, 'anomaly_detector') and hasattr(self.agent.anomaly_detector, 'anomaly_alerts'):
                anomalies = list(self.agent.anomaly_detector.anomaly_alerts)
            elif hasattr(self.agent, 'anomaly_alerts'):
                anomalies = list(self.agent.anomaly_alerts)
            
            if not anomalies or item_id < 0 or item_id >= len(anomalies):
                return
            
            alert = anomalies[item_id]
            
            # Extract ISO/AE with fallback
            model_contrib = alert.get('model_contributions', {})
            
            # ISO Score
            iso_score = "N/A"
            if "isolation_forest" in model_contrib:
                iso_raw = model_contrib["isolation_forest"].get("raw_score")
                if iso_raw is not None:
                    iso_score = f"{iso_raw:.4f}"
                else:
                    iso_contrib = model_contrib["isolation_forest"].get("score")
                    if iso_contrib:
                        iso_score = f"{iso_contrib:.2f}"
            
            if iso_score == "N/A":
                alert_iso = alert.get('iso_score')
                if alert_iso is not None:
                    iso_score = f"{alert_iso:.4f}"
            
            # AE Error
            ae_error = "N/A"
            if "autoencoder" in model_contrib:
                ae_recon = model_contrib["autoencoder"].get("recon_error")
                if ae_recon is not None:
                    ae_error = f"{ae_recon:.6f}"
                else:
                    ae_contrib = model_contrib["autoencoder"].get("score")
                    if ae_contrib:
                        ae_error = f"{ae_contrib:.2f}"
            
            if ae_error == "N/A":
                alert_ae = alert.get('ae_recon_error')
                if alert_ae is not None:
                    ae_error = f"{alert_ae:.6f}"
            
            # ✅ BUILD DETAILED VIEW WITH ISO/AE
            detail_text = f"""
    {'='*80}
        ANOMALY DETAILS #{item_id + 1} - {alert.get('severity', 'unknown').upper()}
    {'='*80}

    📅 Timestamp:     {alert.get('timestamp', 'Unknown')}
    🏷️  Category:      {alert.get('category', 'Unknown').upper()}
    ⚠️  Severity:      {alert.get('severity', 'unknown').upper()}
    🔴 Threat Level:  {alert.get('ensemble_score', 0):.1f}/10

    {'─'*80}
    🤖 AI MODEL ANALYSIS
    {'─'*80}
    - Isolation Forest Score:  {iso_score}
    - Autoencoder Recon Error: {ae_error}
    - Z-Score Triggered:       {alert.get('zscore_flag', False)}

    {'─'*80}
    🔑 KEY INDICATORS
    {'─'*80}
    """
            
            contrib = alert.get('contributing_features', [])
            if contrib:
                for feat in contrib[:5]:
                    readable = self._translate_feature_name(feat)
                    detail_text += f"  • {readable}\n"
            else:
                detail_text += "  (No specific indicators)\n"
            
            detail_text += f"""
    {'─'*80}
    📋 WHAT HAPPENED
    {'─'*80}
    {alert.get('explanation', 'No explanation available')}

    {'─'*80}
    ❓ WHY IT MATTERS
    {'─'*80}
    {alert.get('evidence', 'No evidence available')}

    {'─'*80}
    ✅ RECOMMENDED ACTION
    {'─'*80}
    {alert.get('recommended_action', 'Monitor system')}
    """
            
            # Update AI panel
            self.ai_analysis_text.delete('1.0', tk.END)
            self.ai_analysis_text.insert('1.0', detail_text)
        
        except Exception as e:
            print(f"[GUI] Error in anomaly selection: {e}")
            import traceback
            traceback.print_exc()
    
    def clear_logs(self):
        """Clear log display"""
        self.log_text.delete('1.0', tk.END)
    
    def export_logs(self):
        """Export logs to file"""
        from tkinter import filedialog
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Export Logs"
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(self.log_text.get('1.0', tk.END))
                self.add_log(f"– Logs exported to {filename}")
            except Exception as e:
                self.add_log(f"✗ Export failed: {e}")
    
    def add_log(self, message):
        """Add message to log display"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
    
    def update_gui(self):
        """Periodic GUI update"""
        try:
            self.process_update_queue()
            # Update system metrics
            cpu_percent = psutil.cpu_percent()
            ram_percent = psutil.virtual_memory().percent
            
            self.status_labels['cpu'].config(text=f"{cpu_percent:.1f}%")
            self.status_labels['ram'].config(text=f"{ram_percent:.1f}%")
            
            # Color code CPU/RAM
            if cpu_percent > 80:
                self.status_labels['cpu'].config(fg=self.colors['danger'])
            elif cpu_percent > 50:
                self.status_labels['cpu'].config(fg=self.colors['warning'])
            else:
                self.status_labels['cpu'].config(fg=self.colors['success'])
            
            if ram_percent > 80:
                self.status_labels['ram'].config(fg=self.colors['danger'])
            elif ram_percent > 50:
                self.status_labels['ram'].config(fg=self.colors['warning'])
            else:
                self.status_labels['ram'].config(fg=self.colors['success'])
            
            # Update agent status
            if self.agent.running:
                self.status_labels['running'].config(text="Running", fg=self.colors['success'])
            else:
                self.status_labels['running'].config(text="Stopped", fg=self.colors['danger'])
            
            # Update ML model status
            if self.agent.anomaly_detector.isolation_forest is not None:
                self.status_labels['iso_forest'].config(text="Trained ", fg=self.colors['success'])
            else:
                self.status_labels['iso_forest'].config(text="Not Trained", fg=self.colors['warning'])
            
            if (self.agent.anomaly_detector.autoencoder is not None and 
                self.agent.anomaly_detector.autoencoder.is_trained):
                self.status_labels['autoencoder'].config(text="Trained ", fg=self.colors['success'])
            else:
                self.status_labels['autoencoder'].config(text="Not Trained", fg=self.colors['warning'])
            
            # Update feature buffer count
            buffer_size = len(self.agent.feature_manager.feature_buffer)
            self.status_labels['feature_buffer'].config(text=str(buffer_size))
            
            # Update anomaly rate
            detector = self.agent.anomaly_detector
            anomaly_rate = detector.anomaly_count / max(detector.total_detections, 1)
            self.status_labels['anomaly_rate'].config(text=f"{anomaly_rate:.2%}")
            
            # Update connection count (approximate)
            try:
                connections = len([c for c in psutil.net_connections(kind='inet') 
                                  if c.status == 'ESTABLISHED'])
                self.status_labels['connections'].config(text=str(connections))
            except:
                pass
            
            # Update process count
            try:
                proc_count = len(list(psutil.process_iter()))
                self.status_labels['processes'].config(text=str(proc_count))
            except:
                pass
            
            # Update model version
            model_version = self.agent.anomaly_detector.model_weights.get('version', 0)
            self.status_labels['model_version'].config(text=str(model_version))
            
            # Update FL tab
            self.update_fl_tab()
            
            # Update status bar
            self.status_bar.config(
                text=f"Status: Running | CPU: {cpu_percent:.1f}% | RAM: {ram_percent:.1f}% | "
                     f"Anomalies: {detector.anomaly_count} | Buffer: {buffer_size} windows"
            )
            
        except Exception as e:
            print(f"GUI update error: {e}")
        
        # Schedule next update (every 5 seconds)
        self.root.after(5000, self.update_gui)
    
    def update_fl_tab(self):
        """Update FL status tab - FIXED: Show correct delta norms and metadata"""
        try:
            detector = self.agent.anomaly_detector
            
            # Local / Global versions
            local_version = detector.model_weights.get('version', 0)
            self.fl_labels['local_version'].config(text=str(local_version))

            if self.agent.last_global_model:
                global_version = self.agent.last_global_model.get('version', 0)
                self.fl_labels['global_version'].config(text=str(global_version))
            
            # –… Correct sample counts
            network_samples = len(detector.network_baseline.get('connections', []))
            process_samples = len(detector.process_baseline.get('count', []))
            file_samples = len(detector.file_baseline.get('events', []))
            total_samples = network_samples + process_samples + file_samples
            self.fl_labels['samples_used'].config(text=str(total_samples))
            
            # –… Delta norm from pending FL updates
            if hasattr(self.agent, 'fl_manager') and self.agent.fl_manager.pending_updates:
                recent_update = self.agent.fl_manager.pending_updates[-1]
                delta_norm = recent_update.get('post_norm', 0.0)
                self.fl_labels['delta_norm'].config(text=f"{delta_norm:.4f}")
            else:
                self.fl_labels['delta_norm'].config(text="No updates yet")
            
            # –… Last local training time
            if detector.last_training_time:
                last_train = datetime.fromtimestamp(detector.last_training_time)
                self.fl_labels['last_training'].config(
                    text=last_train.strftime('%Y-%m-%d %H:%M:%S')
                )
            else:
                self.fl_labels['last_training'].config(text="Never")
            
            # –… Last global update timestamp
            if self.agent.last_global_model:
                last_update_str = self.agent.last_global_model.get('last_update', 'Unknown')
                if isinstance(last_update_str, datetime):
                    last_update_str = last_update_str.strftime('%Y-%m-%d %H:%M:%S')
                self.fl_labels['last_update'].config(text=str(last_update_str))

            # –… Keep existing weight display
            self.fl_weights_text.delete('1.0', tk.END)
            weights_text = "─── Current Model Weights ───\n\n"
            
            for key, value in sorted(detector.model_weights.items()):
                if isinstance(value, float):
                    weights_text += f"  {key}: {value:.6f}\n"
                else:
                    weights_text += f"  {key}: {value}\n"

            self.fl_weights_text.insert('1.0', weights_text)

        except Exception as e:
            print(f"FL tab update error: {e}")

    
    def run(self):
        """Start the GUI main loop"""
        self.root.mainloop()
    
    def stop(self):
        """Stop the GUI"""
        self.root.quit()
        self.root.destroy()
        
def main():
    """Main entry point with GUI support"""
    import argparse
    global SERVER_HOST, SERVER_PORT   # <-- MUST come before any usage

    parser = argparse.ArgumentParser(description='FortifAI Client Agent')
    parser.add_argument('--no-gui', action='store_true', help='Run without GUI')
    parser.add_argument('--server', type=str, default=SERVER_HOST, help='Server IP address')
    parser.add_argument('--port', type=int, default=SERVER_PORT, help='Server port')
    args = parser.parse_args()
    
    # Update global config
    SERVER_HOST = args.server
    SERVER_PORT = args.port
    
    agent = ClientAgent()
    
    if args.no_gui:
        agent.start()
    else:
        try:
            # Start agent in background
            agent_thread = threading.Thread(
                target=run_agent_background,
                args=(agent,),
                daemon=True
            )
            agent_thread.start()
            
            # Start GUI
            gui = ClientGUI(agent)
            agent.gui = gui   # <-- PATCH 7B: Link agent with GUI
            
            # Handle window close
            def on_closing():
                agent.stop()
                gui.stop()
            
            gui.root.protocol("WM_DELETE_WINDOW", on_closing)
            gui.run()
        
        except tk.TclError as e:
            print(f"GUI not available: {e}")
            print("Falling back to console mode...")
            agent.start()


# PATCH 7 â€“ FIXED run_agent_background()
def run_agent_background(agent):
    """Run agent in background for GUI mode"""
    
    max_retries = 3
    for attempt in range(max_retries):
        if agent.register_with_server():
            break
        print(f"Registration attempt {attempt+1}/{max_retries} failed, retrying...")
        time.sleep(5)
    else:
        print("Failed to register with server after multiple attempts")
        return
    
    agent.running = True
    
    # ✅ FIX: Define delayed_initial_training with proper closure
    def delayed_initial_training():
        """Bootstrap training - skip if models already loaded from server"""
        time.sleep(180)  # Wait 3 minutes
        
        # ✅ ENHANCED: Re-verify models after waiting period
        if agent.models_fetched_from_server:
            # Double-check models are still loaded
            iso_loaded = agent.anomaly_detector.isolation_forest is not None
            ae_loaded = (agent.anomaly_detector.autoencoder and 
                        agent.anomaly_detector.autoencoder.is_trained)
            
            models_applied = iso_loaded or ae_loaded
            
            if models_applied:
                print(f"\n[BOOTSTRAP] ✓ Models verified after restart:")
                print(f"  - Isolation Forest: {'✓' if iso_loaded else '✗'}")
                print(f"  - Autoencoder: {'✓' if ae_loaded else '✗'}")
                print(f"[BOOTSTRAP] ✓ Skipping initial training")
                if hasattr(agent, 'gui') and agent.gui:
                    agent.gui.add_log("✓ Using cached models from server")
                return  # ✅ EXIT - models loaded successfully
            else:
                print(f"[BOOTSTRAP] ⚠ Flag set but models missing - will train fresh")
                print(f"  Debug: ISO={iso_loaded}, AE={ae_loaded}")
                agent.models_fetched_from_server = False  # Reset flag
        
        # Only train if no cached models
        feature_matrix, feature_names = agent.feature_manager.get_feature_matrix()
        
        buffer_size = len(agent.feature_manager.feature_buffer)
        print(f"\n[BOOTSTRAP CHECK] Buffer has {buffer_size} windows")
        
        # ✅ CRITICAL FIX: Train with as few as 10 samples for bootstrap
        if feature_matrix is not None and len(feature_matrix) >= 10:  # ✅ Lowered from 20
            print(f"\n[BOOTSTRAP] ⚡ FORCING initial training with {len(feature_matrix)} samples...")
            
            # ✅ Pad feature matrix if needed to reach minimum
            if len(feature_matrix) < 20:
                print(f"  [BOOTSTRAP] Padding {len(feature_matrix)} samples to 20 with synthetic data...")
                # Duplicate existing samples with slight noise
                padded_matrix = []
                for i in range(20):
                    idx = i % len(feature_matrix)
                    sample = feature_matrix[idx].copy()
                    # Add small random noise (5% variance)
                    noise = np.random.normal(0, 0.05, size=sample.shape)
                    sample += noise
                    padded_matrix.append(sample)
                feature_matrix = np.array(padded_matrix)
            
            success = agent.anomaly_detector.train_models(feature_matrix)
            
            if success:
                print("[BOOTSTRAP] ✓ Models trained successfully!")
                if hasattr(agent, 'gui') and agent.gui:
                    agent.gui.add_log(f"✓ ML Models trained: {len(feature_matrix)} samples")
                
                agent.send_models_to_server_cache()
            else:
                print("[BOOTSTRAP] ✗ Training failed")
        else:
            actual_size = len(feature_matrix) if feature_matrix is not None else 0
            print(f"[BOOTSTRAP] Insufficient data: {actual_size}/10 samples needed")
    
    # ✅ FIX: Start threads with proper targets
    threads = [
        threading.Thread(target=delayed_initial_training, daemon=True),
        threading.Thread(target=agent.heartbeat_loop, daemon=True),
        threading.Thread(target=agent.collection_loop, daemon=True),
        threading.Thread(target=agent.fl_loop, daemon=True),
        threading.Thread(target=agent.ml_training_loop, daemon=True)
    ]
    
    for t in threads:
        t.start()
    
    # Keep background alive
    while agent.running:
        time.sleep(1)


if __name__ == '__main__':
    main()