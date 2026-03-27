import os
import sys

# Project root for ``shared.*`` before Google imports (avoids FutureWarning noise)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import warnings

try:
    from shared.warning_filters import silence_google_sdk_future_warnings
    silence_google_sdk_future_warnings()
except ImportError:
    pass

import time
import json
import socket
import pickle
import hashlib
import platform
import threading
import logging
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
from dotenv import load_dotenv

load_dotenv()




from client.utils.config import *
from client.gui.client_gui import ClientGUI
from client.core.data_collection import *
from client.core.feature_manager import *
from client.models.anomaly_detector import *
from client.core.feature_collectors import FullFeatureCollector, FEATURE_NAMES as SCHEMA_99

# ── Client-side RL agent ─────────────────────────────────────────────────
_RL_ACTIONS = ["ignore", "block_ip", "kill_process", "isolate_network", "quarantine_file"]

class _ClientRLAgent:
    """
    Lightweight tabular Q-learning agent (client-side mirror of admin/rl/agent.py).
    State = (severity_bucket, top_model, category)
    Action = one of _RL_ACTIONS
    """
    SEV_MAP = {'normal': 0, 'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
    MODEL_MAP = {'ZScore': 0, 'IsolationForest': 1, 'Autoencoder': 2, 'OCSVM': 3, 'Ensemble': 4}
    CAT_MAP = {'Network': 0, 'Process': 1, 'Filesystem': 2, 'User': 3, 'Unknown': 4}

    def __init__(self):
        self.q_table = {}       # state_key -> action_idx -> Q-value
        self.alpha   = 0.1
        self.gamma   = 0.9
        self.epsilon = 0.15

    def _state(self, anomaly_info):
        s = self.SEV_MAP.get(anomaly_info.get('severity', 'normal'), 0)
        m = self.MODEL_MAP.get(anomaly_info.get('detection_model', 'Ensemble'), 4)
        c = self.CAT_MAP.get(anomaly_info.get('category', 'Unknown'), 4)
        return (s, m, c)

    def recommend_action(self, anomaly_info) -> str:
        state = self._state(anomaly_info)
        import random
        if random.random() < self.epsilon or state not in self.q_table:
            # Heuristic defaults instead of pure random (for early training)
            sev = anomaly_info.get('severity', 'normal')
            if sev == 'critical': return 'isolate_network'
            if sev == 'high':     return 'block_ip'
            if sev == 'medium':   return 'kill_process'
            return 'ignore'
        best_action_idx = max(self.q_table[state], key=lambda a: self.q_table[state][a])
        return _RL_ACTIONS[best_action_idx]

    def update(self, anomaly_info, action: str, reward: float):
        state  = self._state(anomaly_info)
        a_idx  = _RL_ACTIONS.index(action) if action in _RL_ACTIONS else 0
        if state not in self.q_table:
            self.q_table[state] = {i: 0.0 for i in range(len(_RL_ACTIONS))}
        old = self.q_table[state][a_idx]
        # Bellman update (no next state — episodic)
        self.q_table[state][a_idx] = old + self.alpha * (reward + self.gamma * max(self.q_table[state].values()) - old)



class ClientAgent:
    """Main client agent with complete federated learning implementation"""
        
    def __init__(self):
        self.running = False

        # ── Core subsystems ────────────────────────────────────────────────────────
        self.feature_manager  = FeatureWindowManager(window_duration=60, max_windows=1000)
        self.anomaly_detector = EnhancedAnomalyDetector(use_ocsvm=False)
        self.anomaly_detector.feature_manager = self.feature_manager
        # Single alert queue for GUI, telemetry, and server (see run_agent_background / start)
        self.anomaly_detector._parent_agent = self

        # ── NEW: 99-feature collector (all 4 domains) ───────────────────────────
        self.full_collector = FullFeatureCollector()

        # Legacy collectors kept for server telemetry payload compat
        self.collectors = {
            'network':    EnhancedNetworkCollector(self.anomaly_detector),
            'process':    EnhancedProcessCollector(self.anomaly_detector),
            'filesystem': EnhancedFilesystemCollector(self.anomaly_detector),
            'user':       UserActivityCollector(self.anomaly_detector)
        }
        for name, col in self.collectors.items():
            col.feature_manager = self.feature_manager

        # ── Alert queue (GUI reads this) ─────────────────────────────────────────
        self.anomaly_alerts           = deque(maxlen=500)
        self.models_fetched_from_server = False
        self.last_global_model        = None
        self.last_fl_delta_meta       = None  # last upload: norm / clip (for FL Status tab)
        self.fl_upload_callback       = None

        # ── Client-side RL agent ───────────────────────────────────────────────
        self.rl_agent = _ClientRLAgent()

        logging.info(f"FortifAI Client Agent | ID:{CLIENT_ID} | {socket.gethostname()}")
        logging.info(f"Server: {SERVER_HOST}:{SERVER_PORT} | FL: {'ON' if ENABLE_FL else 'OFF'}")
        logging.info("ML: IsoForest + Autoencoder + ZScore + OCSVM + RL (ensemble)")

    
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
                    self.anomaly_detector.isolation_forest = None
                        
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
                
                # ✅ ADD: Fetch massive foundation models
                self._sync_and_load_foundation_models()
                
                return True
            else:
                print("✗ Registration failed")
                return False
        
        except Exception as e:
            print(f"✗ Registration error: {e}")
            return False

    def _sync_and_load_foundation_models(self):
        """Fetch binary models from admin/models and initialize"""
        try:
            print("\n[SYNC] Requesting Deep Learning Foundation Models...")
            sync_req = {
                'type': 'sync_foundation_models',
                'client_id': CLIENT_ID
            }
            response = self.send_to_server(sync_req)
            if response and response.get('status') == 'success':
                models = response.get('models', {})
                import tempfile
                import os
                
                # 1. Load Autoencoder H5
                if models.get('global_autoencoder.h5'):
                    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
                        f.write(models['global_autoencoder.h5'])
                        tmp_ae = f.name
                    try:
                        from tensorflow.keras.models import load_model
                        self.anomaly_detector.autoencoder = AutoencoderAnomalyDetector(input_dim=99, latent_dim=32)
                        self.anomaly_detector.autoencoder.model = load_model(tmp_ae)
                        self.anomaly_detector.autoencoder.is_trained = True
                        print("[SYNC] ✓ Loaded global_autoencoder.h5")
                        os.unlink(tmp_ae)
                    except Exception as e:
                        print(f"[SYNC] ✗ Failed to load AE: {e}")
                
                # 2. Load Isolation Forest PKL
                if models.get('global_iso_forest.pkl'):
                    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
                        f.write(models['global_iso_forest.pkl'])
                        tmp_iso = f.name
                    try:
                        import pickle
                        with open(tmp_iso, 'rb') as f:
                            self.anomaly_detector.isolation_forest = pickle.load(f)
                        print("[SYNC] ✓ Loaded global_iso_forest.pkl")
                        os.unlink(tmp_iso)
                    except Exception as e:
                        print(f"[SYNC] ✗ Failed to load ISO: {e}")
                
                # 3. Load Scaler PKL (only if AE exists)
                if models.get('global_scaler.pkl') and self.anomaly_detector.autoencoder is not None:
                    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
                        f.write(models['global_scaler.pkl'])
                        tmp_scaler = f.name
                    try:
                        import pickle
                        with open(tmp_scaler, 'rb') as f:
                            self.anomaly_detector.autoencoder.scaler = pickle.load(f)
                        print("[SYNC] ✓ Loaded global_scaler.pkl")
                        os.unlink(tmp_scaler)
                    except Exception as e:
                        print(f"[SYNC] ✗ Failed to load Scaler: {e}")
                
                # 4. Load Baseline Z-Scores for stable normalization
                if models.get('global_z_baselines.json'):
                    try:
                        import json
                        baselines = json.loads(models['global_z_baselines.json'].decode('utf-8'))
                        self.feature_manager.global_z_means = np.array(baselines.get('means', []))
                        self.feature_manager.global_z_stds = np.array(baselines.get('stds', []))
                        print(f"[SYNC] Loaded global_z_baselines.json "
                              f"({len(self.feature_manager.global_z_means)} features)")
                    except Exception as e:
                        print(f"[SYNC] Failed to load Z-baselines: {e}")
                        
                # 5. Load the precise 99-Feature Schema
                if models.get('global_feature_schema.json'):
                    try:
                        import json
                        schema = json.loads(models['global_feature_schema.json'].decode('utf-8'))
                        # Don't overwrite FEATURE_SCHEMA (it's local 28-feature, not the model's 99)
                        # Instead, update the dimension the feature matrix pads to
                        self.feature_manager.model_input_dim = len(schema)
                        print(f"[SYNC] ✓ Loaded global_feature_schema.json ({len(schema)} features)")
                        print(f"[SYNC] Feature matrix will pad to {len(schema)} dimensions")
                    except Exception as e:
                        print(f"[SYNC] ✗ Failed to load Feature Schema: {e}")
                        
                print("[SYNC] Foundation models fully loaded into prediction engine!")
                self.models_fetched_from_server = True
                
        except Exception as e:
            print(f"[SYNC] ✗ Connection error fetching models: {e}")

    def _apply_fl_parameters_to_detector(self):
        """Apply federated learning parameters to ALL local ML models."""
        try:
            weights = self.anomaly_detector.model_weights

            logging.info("[FL APPLICATION] Applying global model parameters to all 5 models")

            # ── Z-Score threshold (average of 3 domain thresholds) ──────────
            nt = weights.get('network_threshold', 2.0)
            pt = weights.get('process_threshold', 2.0)
            ft = weights.get('file_threshold', 2.0)
            avg_thresh = (abs(nt) + abs(pt) + abs(ft)) / 3.0
            self.anomaly_detector.zscore_threshold = max(avg_thresh * 2.5, 2.0)
            logging.info(f"  Z-Score Threshold   → {self.anomaly_detector.zscore_threshold:.3f}")

            # ── Isolation Forest threshold ──────────────────────────────────
            iso_thresh = weights.get('iso_threshold', -0.5)
            if isinstance(iso_thresh, (int, float)):
                self.anomaly_detector.iso_threshold = float(iso_thresh)
            logging.info(f"  ISO Threshold       → {self.anomaly_detector.iso_threshold:.3f}")

            # ── Autoencoder threshold ───────────────────────────────────────
            ae_thresh = weights.get('ae_threshold', None)
            if ae_thresh and isinstance(ae_thresh, (int, float)):
                self.anomaly_detector.ae_threshold = float(ae_thresh)
                if self.anomaly_detector.autoencoder:
                    self.anomaly_detector.autoencoder.threshold = float(ae_thresh)
            logging.info(f"  AE Threshold        → {self.anomaly_detector.ae_threshold}")

            # ── OCSVM: re-enable if server sends flag ───────────────────────
            if weights.get('ocsvm_enabled', False):
                self.anomaly_detector.use_ocsvm = True
                if self.anomaly_detector.one_class_svm is None:
                    self.anomaly_detector.one_class_svm = OneClassSVM(kernel='rbf', nu=0.05)
                logging.info("  OCSVM               → Enabled by server")

            # ── Baseline means/stds for ensemble calibration ────────────────
            for k in ['network_baseline_mean', 'network_baseline_std',
                      'process_baseline_mean', 'process_baseline_std',
                      'file_baseline_mean', 'file_baseline_std']:
                self.anomaly_detector.model_weights[k] = weights.get(k, 0.0)

            # ── Sensitivities: scale ensemble weights ───────────────────────
            self.anomaly_detector.model_weights['network_sensitivity'] = weights.get('network_sensitivity', 1.0)
            self.anomaly_detector.model_weights['process_sensitivity'] = weights.get('process_sensitivity', 1.0)
            self.anomaly_detector.model_weights['file_sensitivity']    = weights.get('file_sensitivity', 1.0)

            # ── Load NN autoencoder weights from FL update ──────────────────
            ae_layer_keys = [k for k in weights if k.startswith('ae_layer_')]
            if ae_layer_keys and self.anomaly_detector.autoencoder:
                try:
                    weight_dict = {k: weights[k] for k in ae_layer_keys if isinstance(weights[k], list)}
                    if weight_dict:
                        self.anomaly_detector.autoencoder.set_weights(weight_dict)
                        self.anomaly_detector.autoencoder.is_trained = True
                        logging.info(f"  AE Layer weights    → {len(weight_dict)} tensors applied")
                except Exception as e:
                    logging.warning(f"  AE weight load error: {e}")

            logging.info("[FL APPLICATION] ✓ Parameters applied successfully to all models")

            if hasattr(self, 'gui') and self.gui:
                self.gui.add_log("✅ FL params applied: ZScore + ISO + AE + OCSVM thresholds updated")
                if hasattr(self.gui, 'fl_status_updated'):
                    self.gui.fl_status_updated.emit("✅ FL parameters applied to all 5 models")

        except Exception as e:
            logging.error(f"[FL APPLICATION] ✗ Error applying parameters: {e}")


    
    def send_to_server(self, data):
        """Pass-through to dedicated NetworkUtils"""
        from client.utils.network_utils import NetworkClient
        return NetworkClient.send_to_server(data)
    
    # --- NEW: Background ML training loop ---
    def ml_training_loop(self):
        """Background thread for periodic ML model training - ENHANCED"""
        while self.running:
            try:
                if self.feature_manager.should_finalize_window():
                    # Signal legacy collectors that the window is done (resets accumulators)
                    self.feature_manager.finalize_window()
                    
                    # ── Collect 99-feature window from FullFeatureCollector ────────────
                    raw_features = self.full_collector.get_feature_vector(
                        window_sec=self.feature_manager.window_duration)

                    # Normalise and append to feature_buffer
                    from client.core.feature_collectors import FEATURE_NAMES as SCHEMA_99
                    # Build ordered array
                    raw_arr = np.array([raw_features.get(n, 0.0) for n in SCHEMA_99], dtype=float)

                    # Running z-score normalisation using feature_manager stats
                    fm = self.feature_manager
                    fm.running_count += 1
                    for i, n in enumerate(SCHEMA_99):
                        v = float(raw_arr[i])
                        m = fm.running_mean.get(n, v)
                        s = fm.running_std.get(n, 1.0)
                        # Welford online update
                        alpha = 0.1
                        fm.running_mean[n] = (1 - alpha) * m + alpha * v
                        diff = abs(v - fm.running_mean[n])
                        fm.running_std[n]  = max((1 - alpha) * s + alpha * diff, 1e-6)
                        raw_arr[i] = (v - fm.running_mean[n]) / fm.running_std[n]

                    window_dict = {SCHEMA_99[i]: raw_arr[i] for i in range(len(SCHEMA_99))}
                    window_dict['timestamp'] = datetime.now().timestamp()
                    fm.feature_buffer.append(window_dict)

                    feature_vector = raw_arr
                    feature_names  = SCHEMA_99
                    buffer_size    = len(fm.feature_buffer)
                    logging.info(f"[ML] 99-feature window buffered → {buffer_size} total")

                    # Expose raw dict on feature_manager so telemetry.py can read it
                    fm.last_raw_features = raw_features

                    # ── Always run ensemble detection on this window ───────────
                    if (self.anomaly_detector.isolation_forest is not None or
                        (self.anomaly_detector.autoencoder and self.anomaly_detector.autoencoder.is_trained)):
                        try:
                            is_anomaly, anomaly_info = self.anomaly_detector.detect_anomaly_ensemble(
                                feature_vector, feature_names
                            )
                            if is_anomaly:
                                logging.warning(f"[DETECTION] ⚠️ Score={anomaly_info.get('ensemble_score',0):.1f} "
                                                f"Model={anomaly_info.get('detection_model','?')} "
                                                f"Cat={anomaly_info.get('category','?')}")
                                # Alert is appended inside detect_anomaly_ensemble → _generate_alert
                        except Exception as e:
                            logging.debug(f"[DETECTION] Error: {e}")

                    
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
            'severity':  severity,
            'category':  anomaly_info.get('category', 'unknown'),
            'detection_model': anomaly_info.get('detection_model', 'Ensemble'),
            'is_anomaly': anomaly_info.get('is_anomaly'),
            'ensemble_score': anomaly_info.get('ensemble_score', 0.0),
            'zscore_flag': anomaly_info.get('zscore_flag'),
            'iso_score': anomaly_info.get('iso_score'),
            'ae_recon_error': anomaly_info.get('ae_recon_error'),
            'ocsvm_score': anomaly_info.get('ocsvm_score'),
            'contributing_features': contrib,
            'feature_vector': {k: v for k, v in feature_vector.items() if k != 'timestamp'}
                               if isinstance(feature_vector, dict)
                               else {f'feat_{i}': float(v) for i, v in enumerate(feature_vector)},
            'explanation': " | ".join(explanation_parts),
            'signature': alert_signature,
        }

        # ── RL classification (recommend only — no self-update) ───────────
        # Rewards should come from analyst feedback, NOT from severity alone.
        # Self-reinforcing on severity creates a degenerate policy.
        rl_action = self.rl_agent.recommend_action(alert)
        alert['rl_recommended_action'] = rl_action
        alert['rl_awaiting_feedback'] = True

        # ── Add alert AFTER duplicate suppression ────────────────────────────
        self.anomaly_alerts.append(alert)

        logging.warning(f"⚠️ ANOMALY | {severity.upper()} | model={alert['detection_model']} "
                        f"| cat={alert.get('category','?')} | RL-action={rl_action} "
                        f"| score={alert['ensemble_score']:.1f}")

        if hasattr(self, 'gui') and self.gui:
            self.gui.add_log(f"⚠️ {severity.upper()} | {alert['detection_model']} | "
                             f"{alert.get('category','?')} | RL→{rl_action}")


            
    def collect_and_send(self):
        """Delegate telemetry collection to core telemetry module"""
        from client.core.telemetry import TelemetrySender
        TelemetrySender.collect_and_send(self)

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
        """Delegate FL Update sending to dedicated FLClient module"""
        from client.fl.fl_client import FLClient
        FLClient.send_fl_update(self)
                     
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

        # ── START NEW 99-FEATURE COLLECTOR ────────────────────────────────────────
        self.full_collector.start()
        logging.info("[COLLECTOR] FullFeatureCollector started (Net+Proc+FS+User polling)")

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
        """Periodically refresh GUI anomaly monitor via thread-safe signal."""
        from PyQt5.QtCore import QMetaObject, Qt as QtConst, Q_ARG
        while self.running:
            time.sleep(2)
            if hasattr(self, 'gui') and self.gui:
                try:
                    QMetaObject.invokeMethod(
                        self.gui, "refresh_anomalies",
                        QtConst.QueuedConnection
                    )
                except Exception:
                    pass

    
    def stop(self):
        """Stop the agent"""
        self.running = False




def main():
    """Main entry point with PyQt5 GUI support"""
    import argparse
    from PyQt5.QtWidgets import QApplication
    from client.gui.client_gui import ClientGUI, apply_dark_palette
    global SERVER_HOST, SERVER_PORT

    parser = argparse.ArgumentParser(description='FortifAI Client Agent')
    _mode = parser.add_mutually_exclusive_group()
    _mode.add_argument(
        '--gui',
        action='store_true',
        help='Run with PyQt dashboard (default if neither mode flag is set)',
    )
    _mode.add_argument(
        '--no-gui',
        action='store_true',
        help='Run headless in the terminal only',
    )
    parser.add_argument('--server', type=str, default=SERVER_HOST, help='Server IP address')
    parser.add_argument('--port',   type=int, default=SERVER_PORT, help='Server port')
    args = parser.parse_args()

    SERVER_HOST = args.server
    SERVER_PORT = args.port
    use_gui = not args.no_gui

    # Wire logging so all print() via logging.info go to GUI
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )

    agent = ClientAgent()

    if not use_gui:
        agent.start()
    else:
        app = QApplication(sys.argv)
        app.setApplicationName("FortifAI Client Agent")
        apply_dark_palette(app)

        gui = ClientGUI(agent)
        agent.gui = gui

        # Redirect FL upload callback to GUI log
        agent.fl_upload_callback = gui.on_fl_upload_status

        # Start backend in background thread
        agent_thread = threading.Thread(
            target=run_agent_background,
            args=(agent,),
            daemon=True
        )
        agent_thread.start()

        gui.show()
        sys.exit(app.exec_())



# PATCH 7 â€“ FIXED run_agent_background()
def run_agent_background(agent):
    """Run agent in background for GUI mode"""
    # Same deque as GUI: telemetry must not read detector-only queue
    agent.anomaly_detector.anomaly_alerts = agent.anomaly_alerts

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
    
    agent.full_collector.start()

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


if __name__ == "__main__":
    main()
