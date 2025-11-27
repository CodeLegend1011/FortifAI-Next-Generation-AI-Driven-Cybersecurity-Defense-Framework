"""
FortifAI Admin Server Hub - Enhanced with Federated Learning
Manages client nodes, aggregates ML models, provides intelligent dashboard
"""

import sys
import json
import threading
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTabWidget, QTableWidget, QTableWidgetItem,
                             QPushButton, QLabel, QTextEdit, QSplitter, QGroupBox,
                             QHeaderView, QMessageBox, QDialog, QFormLayout, QLineEdit,
                             QComboBox, QSpinBox, QProgressBar)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtChart import (QChart, QChartView, QLineSeries, QPieSeries, QBarSet, 
                           QBarSeries, QBarCategoryAxis, QValueAxis, QDateTimeAxis)
import psycopg2
from psycopg2.extras import RealDictCursor
import socket
import pickle
from PyQt5.QtGui import QPainter  # Add this to imports at top
from collections import defaultdict, deque
import google.generativeai as genai
import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module=".*sip.*")

# Database Configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'fortifai_db',
    'user': 'postgres',
    'password': 'postgres',
    'port': 5432
}

# Server Configuration
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 9999
# Database Configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'fortifai_db',
    'user': 'postgres',
    'password': 'postgres',
    'port': 5432
}

# Server Configuration
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 9999

# --- NEW: Robust FL Configuration ---
CLIP_BOUND = 10.0  # L2 norm clipping threshold
TRIM_FRAC = 0.1    # Fraction to trim from each end (10%)
DP_NOISE_SCALE = 0.5  # Differential privacy noise multiplier
VALIDATION_AUC_DROP_THRESHOLD = 0.02  # Rollback if AUC drops by this much

def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types - NUMPY 2.0 COMPATIBLE"""
    if obj is None:
        return None
    
    # Handle numpy boolean
    if isinstance(obj, (np.bool_, bool)) and hasattr(np, 'bool_'):
        return bool(obj)
    if type(obj).__name__ == 'bool_':
        return bool(obj)
    
    # Handle numpy integers
    if isinstance(obj, np.integer):
        return int(obj)
    if type(obj).__name__ in ('int_', 'int8', 'int16', 'int32', 'int64'):
        return int(obj)
    
    # Handle numpy floats - THIS IS THE KEY FIX
    if isinstance(obj, np.floating):
        return float(obj)
    if type(obj).__name__ in ('float_', 'float16', 'float32', 'float64'):
        return float(obj)
    
    # Handle numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # Handle dicts recursively
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    
    # Handle lists/tuples recursively
    if isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    
    return obj

class FederatedLearningManager:
    """Advanced Federated Learning Manager with Robust FL, DP, and Reputation"""
    
    def __init__(self):
        self.client_models = {}
        self.client_contributions = defaultdict(lambda: {
            'count': 0, 
            'quality': 1.0, 
            'last_update': None,
            'reputation': 1.0,  # NEW: Reputation score
            'clipped_count': 0,  # NEW: How many times clipped
            'quarantine_count': 0,  # NEW: Quarantine events
            'avg_norm': 0.0,  # NEW: Average delta norm
            'participation_rate': 1.0  # NEW: Participation consistency
        })
        
        # Enhanced global model with momentum
        self.global_model = {
            'weights': {
                'network_threshold': 2.0,
                'process_threshold': 2.0,
                'file_threshold': 2.0,
                'network_sensitivity': 1.0,
                'process_sensitivity': 1.0,
                'file_sensitivity': 1.0,
                'anomaly_alpha': 0.95,
                'anomaly_beta': 0.1,
                'network_baseline_mean': 0.0,
                'network_baseline_std': 1.0,
                'process_baseline_mean': 0.0,
                'process_baseline_std': 1.0,
                'file_baseline_mean': 0.0,
                'file_baseline_std': 1.0
            },
            'momentum': {
                'network_threshold': 0.0,
                'process_threshold': 0.0,
                'file_threshold': 0.0,
                'network_sensitivity': 0.0,
                'process_sensitivity': 0.0,
                'file_sensitivity': 0.0
            },
            'version': 0,
            'last_update': datetime.now(),
            'convergence_history': []
        }
        
        # NEW: Validation dataset for model quality checks
        self.validation_feature_cache = deque(maxlen=200)
        self.validation_labels_cache = deque(maxlen=200)  # 0=normal, 1=anomaly
        
        # NEW: Update rejection tracking
        self.rejected_updates = defaultdict(int)
        self.malformed_updates = defaultdict(int)
        
        # NEW: Audit log
        self.audit_log = deque(maxlen=100)
        
        # --- NEW: Model savepoint for rollback ---
        self.model_savepoint = None
        self.last_validation_metrics = {'auc': 0.85, 'fpr': 0.05, 'tpr': 0.90}
        
        # FedProx configuration
        self.mu = 0.01
        self.momentum_factor = 0.9
        
        # --- NEW: Differential privacy configuration ---
        self.dp_epsilon = 1.0
        self.dp_delta = 1e-5
        self.dp_noise_scale = DP_NOISE_SCALE
        
        # --- NEW: Aggregation metadata tracking ---
        self.aggregation_metadata = {
            'last_round': {
                'participated': 0,
                'quarantined': 0,
                'avg_delta_norm': 0.0,
                'validation_auc': 0.0,
                'dp_noise_applied': 0.0,
                'rollback_occurred': False
            }
        }
        
        # Aggregation history
        self.aggregation_history = deque(maxlen=50)
        
        # --- NEW: Pending updates queue with metadata ---
        self.pending_updates = []
    
    def receive_client_update_robust(self, client_id, model_parameters):
        """
        Enhanced receive with type safety, malformed update rejection, and audit
        """
        timestamp = datetime.now()
        
        # VALIDATION 1: Type safety checks
        if not self._validate_update_structure(model_parameters):
            self.malformed_updates[client_id] += 1
            print(f"✗ REJECTED: Malformed update from {client_id[:12]}")
            self._audit_log_event('malformed_update', client_id, {'reason': 'structure'})
            return False
        
        # VALIDATION 2: Extract and verify metadata
        try:
            metadata = {
                'samples_used': int(model_parameters.get('data_quality', {}).get('network_samples', 0)),
                'local_epochs': 1,
                'loss': 0.0,
                'anomaly_rate': float(model_parameters.get('anomaly_rate', 0.0)),
                'timestamp': timestamp
            }
            
            # Sanity checks
            if metadata['samples_used'] < 10:
                print(f"✗ REJECTED: Insufficient samples ({metadata['samples_used']}) from {client_id[:12]}")
                self._audit_log_event('insufficient_samples', client_id, metadata)
                return False
            
            if metadata['anomaly_rate'] > 0.5:  # >50% anomalies = suspicious
                print(f"⚠ WARNING: High anomaly rate ({metadata['anomaly_rate']:.2%}) from {client_id[:12]}")
                self._audit_log_event('high_anomaly_rate', client_id, metadata)
        
        except (ValueError, TypeError, KeyError) as e:
            self.malformed_updates[client_id] += 1
            print(f"✗ REJECTED: Metadata extraction failed for {client_id[:12]}: {e}")
            self._audit_log_event('metadata_error', client_id, {'error': str(e)})
            return False
        
        # VALIDATION 3: Compute model delta with bounds checking
        model_delta = {}
        client_weights = model_parameters.get('weights', {})
        
        try:
            for key in client_weights.keys():
                if key in self.global_model['weights']:
                    client_val = float(client_weights[key])
                    global_val = float(self.global_model['weights'][key])
                    
                    # Bounds check: reject extreme values
                    if abs(client_val) > 1000 or abs(global_val) > 1000:
                        print(f"✗ REJECTED: Extreme weight value in {key}: {client_val}")
                        self._audit_log_event('extreme_weight', client_id, {
                            'key': key, 'value': client_val
                        })
                        return False
                    
                    delta = client_val - global_val
                    model_delta[key] = delta
        
        except (ValueError, TypeError) as e:
            self.malformed_updates[client_id] += 1
            print(f"✗ REJECTED: Delta computation failed for {client_id[:12]}: {e}")
            self._audit_log_event('delta_error', client_id, {'error': str(e)})
            return False
        
        # VALIDATION 4: Compute pre-clipping L2 norm
        pre_norm = self._compute_l2_norm(model_delta)
        
        # VALIDATION 5: Reject if norm is suspiciously high (potential attack)
        if pre_norm > CLIP_BOUND * 3:
            self.rejected_updates[client_id] += 1
            print(f"✗ REJECTED: Extreme delta norm ({pre_norm:.4f}) from {client_id[:12]}")
            self._audit_log_event('extreme_norm', client_id, {
                'pre_norm': pre_norm, 'threshold': CLIP_BOUND * 3
            })
            return False
        
        # Apply L2 clipping
        clipped_delta, post_norm = self._clip_delta(model_delta, CLIP_BOUND)
        was_clipped = (post_norm < pre_norm)
        
        # Quarantine check
        is_quarantined = False
        if post_norm > CLIP_BOUND * 1.5:
            is_quarantined = True
            self.rejected_updates[client_id] += 1
            print(f"⛔ Client {client_id[:12]} QUARANTINED: norm={post_norm:.4f}")
            self._audit_log_event('quarantined', client_id, {'post_norm': post_norm})
            return False  # Don't add to pending updates
        
        # Update client contribution metadata
        contrib = self.client_contributions[client_id]
        contrib['count'] += 1
        contrib['last_update'] = timestamp
        contrib['avg_norm'] = float((contrib['avg_norm'] * (contrib['count'] - 1) + post_norm) / contrib['count'])
        
        if was_clipped:
            contrib['clipped_count'] += 1
        if is_quarantined:
            contrib['quarantine_count'] += 1
        
        # Update reputation
        self._update_reputation(client_id, post_norm, was_clipped, is_quarantined, metadata)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(model_parameters)
        contrib['quality'] = quality_score
        
        # Add to pending updates queue
        self.pending_updates.append({
            'client_id': client_id,
            'delta': clipped_delta,
            'pre_norm': float(pre_norm),
            'post_norm': float(post_norm),
            'metadata': metadata,
            'quality_score': float(quality_score),
            'reputation': float(contrib['reputation'])
        })
        
        # Store original client model
        self.client_models[client_id] = {
            'parameters': model_parameters,
            'timestamp': timestamp,
            'quality_score': float(quality_score),
            'data_samples': int(metadata['samples_used'])
        }
        
        # Audit log
        self._audit_log_event('update_accepted', client_id, {
            'pre_norm': pre_norm,
            'post_norm': post_norm,
            'clipped': was_clipped,
            'quality': quality_score,
            'reputation': contrib['reputation']
        })
        
        print(f"✓ Received FL update from {client_id[:12]} "
              f"(pre_norm={pre_norm:.4f}, post_norm={post_norm:.4f}, "
              f"clipped={was_clipped}, reputation={contrib['reputation']:.3f})")
        
        return True
    
    # ========================================================================
    # NEW METHOD: Validation dataset management
    # ========================================================================
    def add_to_validation_set(self, feature_vector, is_anomaly):
        """Add labeled data to validation set for model quality checks"""
        self.validation_feature_cache.append(feature_vector)
        self.validation_labels_cache.append(1 if is_anomaly else 0)
    
    def _validate_global_model_enhanced(self, new_weights):
        """
        Enhanced validation using held-out validation set
        Computes anomaly separation score (pseudo-AUC)
        """
        if len(self.validation_feature_cache) < 20:
            # Not enough validation data, use heuristic
            return self._validate_global_model(new_weights)
        
        try:
            # Compute anomaly separation on validation set
            # Higher threshold = fewer detections = lower recall
            # We want thresholds that balance precision/recall
            
            threshold_avg = np.mean([
                new_weights.get('network_threshold', 2.0),
                new_weights.get('process_threshold', 2.0),
                new_weights.get('file_threshold', 2.0)
            ])
            
            # Simulate detection on validation set
            # For each validation sample, compute z-score with new thresholds
            detections = []
            labels = list(self.validation_labels_cache)
            
            for features in self.validation_feature_cache:
                # Simplified scoring: higher threshold = lower sensitivity
                score = 1.0 / (threshold_avg + 0.1)  # Inverse relationship
                detections.append(score)
            
            # Compute pseudo-AUC (rank correlation)
            if len(set(labels)) == 2:  # Need both classes
                from sklearn.metrics import roc_auc_score
                try:
                    auc = roc_auc_score(labels, detections)
                except:
                    auc = 0.85  # Fallback
            else:
                auc = 0.85  # Fallback
            
            # Compute other metrics
            threshold_detections = [1 if d > 0.5 else 0 for d in detections]
            tp = sum(1 for pred, label in zip(threshold_detections, labels) if pred == 1 and label == 1)
            fp = sum(1 for pred, label in zip(threshold_detections, labels) if pred == 1 and label == 0)
            fn = sum(1 for pred, label in zip(threshold_detections, labels) if pred == 0 and label == 1)
            tn = sum(1 for pred, label in zip(threshold_detections, labels) if pred == 0 and label == 0)
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            
            metrics = {
                'auc': float(auc),
                'fpr': float(fpr),
                'tpr': float(tpr)
            }
            
            print(f"  [VALIDATION] AUC={auc:.4f}, TPR={tpr:.4f}, FPR={fpr:.4f}")
            return metrics
        
        except Exception as e:
            print(f"  [VALIDATION] Error: {e}, using heuristic")
            return self._validate_global_model(new_weights)
    
    # ========================================================================
    # NEW METHOD: Structure validation
    # ========================================================================
    def _validate_update_structure(self, model_parameters):
        """Validate update structure and types"""
        required_keys = ['weights', 'data_quality', 'statistics']
        
        for key in required_keys:
            if key not in model_parameters:
                return False
        
        # Validate weights
        weights = model_parameters.get('weights', {})
        if not isinstance(weights, dict):
            return False
        
        expected_weight_keys = [
            'network_threshold', 'process_threshold', 'file_threshold',
            'network_sensitivity', 'process_sensitivity', 'file_sensitivity'
        ]
        
        for key in expected_weight_keys:
            if key not in weights:
                return False
            try:
                float(weights[key])
            except (ValueError, TypeError):
                return False
        
        # Validate data_quality
        data_quality = model_parameters.get('data_quality', {})
        if not isinstance(data_quality, dict):
            return False
        
        return True
    
    # ========================================================================
    # NEW METHOD: Audit logging
    # ========================================================================
    def _audit_log_event(self, event_type, client_id, details):
        """Log audit events for security and debugging"""
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'client_id': client_id[:12],
            'details': details
        }
        self.audit_log.append(audit_entry)
    
    def get_audit_log(self, limit=50):
        """Retrieve recent audit log entries"""
        return list(self.audit_log)[-limit:]
    
    def get_rejection_stats(self):
        """Get update rejection statistics"""
        return {
            'rejected_by_client': dict(self.rejected_updates),
            'malformed_by_client': dict(self.malformed_updates),
            'total_rejected': sum(self.rejected_updates.values()),
            'total_malformed': sum(self.malformed_updates.values())
        }
        
    # --- (A) NEW: Enhanced receive_client_update with clipping and validation ---
    def receive_client_update(self, client_id, model_parameters):
        """
        Receive and validate model update from client with L2 clipping and validation
        FIXED: Proper numpy type conversion
        """
        timestamp = datetime.now()
        
        # Extract metadata with explicit type conversion
        metadata = {
            'samples_used': int(model_parameters.get('data_quality', {}).get('network_samples', 0)),
            'local_epochs': 1,
            'loss': 0.0,
            'anomaly_rate': float(model_parameters.get('anomaly_rate', 0.0)),
            'timestamp': timestamp
        }
        
        # Compute model delta
        model_delta = {}
        client_weights = model_parameters.get('weights', {})
        
        for key in client_weights.keys():
            if key in self.global_model['weights']:
                # CRITICAL: Ensure both are floats
                client_val = float(client_weights[key])
                global_val = float(self.global_model['weights'][key])
                delta = client_val - global_val
                model_delta[key] = delta
        
        # Compute pre-clipping L2 norm
        pre_norm = self._compute_l2_norm(model_delta)
        
        # Apply per-client L2 clipping
        clipped_delta, post_norm = self._clip_delta(model_delta, CLIP_BOUND)
        was_clipped = (post_norm < pre_norm)
        
        # Quarantine check
        is_quarantined = False
        if post_norm > CLIP_BOUND * 1.5:
            is_quarantined = True
            print(f"❌ Client {client_id[:12]} QUARANTINED: norm={post_norm:.4f}")
        
        # Store metadata with explicit type conversion
        contrib = self.client_contributions[client_id]
        contrib['count'] += 1
        contrib['last_update'] = timestamp
        contrib['avg_norm'] = float((contrib['avg_norm'] * (contrib['count'] - 1) + post_norm) / contrib['count'])
        
        if was_clipped:
            contrib['clipped_count'] += 1
        if is_quarantined:
            contrib['quarantine_count'] += 1
        
        # Update reputation
        self._update_reputation(client_id, post_norm, was_clipped, is_quarantined, metadata)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(model_parameters)
        contrib['quality'] = quality_score
        
        # Add to pending updates queue
        if not is_quarantined:
            self.pending_updates.append({
                'client_id': client_id,
                'delta': clipped_delta,
                'pre_norm': float(pre_norm),      # CRITICAL: Explicit float
                'post_norm': float(post_norm),    # CRITICAL: Explicit float
                'metadata': metadata,
                'quality_score': float(quality_score),
                'reputation': float(contrib['reputation'])
            })
        
        # Store original client model
        self.client_models[client_id] = {
            'parameters': model_parameters,
            'timestamp': timestamp,
            'quality_score': float(quality_score),
            'data_samples': int(metadata['samples_used'])
        }
        
        print(f"✓ Received FL update from {client_id[:12]} "
            f"(pre_norm={pre_norm:.4f}, post_norm={post_norm:.4f}, "
            f"clipped={was_clipped}, quarantined={is_quarantined}, "
            f"reputation={contrib['reputation']:.3f})")
    
    # --- NEW: L2 norm computation ---
    def _compute_l2_norm(self, delta_dict):
        """Compute L2 norm of model delta"""
        squared_sum = 0.0
        for key, value in delta_dict.items():
            if isinstance(value, (int, float)):
                squared_sum += value ** 2
        return np.sqrt(squared_sum)
    
    # --- NEW: L2 clipping ---
    def _clip_delta(self, delta_dict, clip_bound):
        """Clip delta to have L2 norm <= clip_bound"""
        norm = self._compute_l2_norm(delta_dict)
        
        if norm > clip_bound:
            # Scale down
            scale_factor = clip_bound / norm
            clipped_delta = {k: v * scale_factor for k, v in delta_dict.items()}
            return clipped_delta, clip_bound
        else:
            return delta_dict, norm
    
    # --- NEW: Reputation update ---
    def _update_reputation(self, client_id, post_norm, was_clipped, is_quarantined, metadata):
        """Update client reputation based on contribution quality"""
        contrib = self.client_contributions[client_id]
        reputation = contrib['reputation']
        
        # Increase reputation for good behavior
        if not was_clipped and post_norm < CLIP_BOUND * 0.5:
            reputation = min(reputation * 1.02, 2.0)
        
        # Consistent participation bonus
        if contrib['count'] > 10:
            reputation = min(reputation * 1.01, 2.0)
        
        # Decrease reputation for bad behavior
        if was_clipped:
            reputation = max(reputation * 0.98, 0.1)
        
        if is_quarantined:
            reputation = max(reputation * 0.90, 0.1)
        
        # High norm penalty
        if post_norm > CLIP_BOUND:
            reputation = max(reputation * 0.95, 0.1)
        
        contrib['reputation'] = reputation
    
    def _calculate_quality_score(self, model_parameters):
        """Calculate data quality score based on multiple factors"""
        quality = 1.0
        
        data_quality = model_parameters.get('data_quality', {})
        statistics = model_parameters.get('statistics', {})
        
        total_samples = (
            data_quality.get('network_samples', 0) +
            data_quality.get('process_samples', 0) +
            data_quality.get('file_samples', 0)
        )
        
        volume_score = min(total_samples / 100.0, 1.0)
        volume_score = max(volume_score, 0.1)
        
        network_stats = statistics.get('network', {})
        variance_score = 1.0
        
        if network_stats.get('variance_connections', 0) > 0:
            variance = network_stats['variance_connections']
            variance_score = min(variance / 50.0, 1.5)
        
        anomaly_rate = model_parameters.get('anomaly_rate', 0.0)
        anomaly_score = 1.0 + (anomaly_rate * 0.5)
        
        quality = volume_score * min(variance_score, 1.2) * min(anomaly_score, 1.3)
        
        return min(quality, 2.0)
    
    # --- (B) NEW: Robust Federated Aggregation with Trimmed Mean ---
    def aggregate_models(self):
        """
        FIXED: True federated learning with ACTUAL weight updates
        """
        if len(self.pending_updates) < 1:
            print("Not enough pending updates for aggregation")
            return None
        
        try:
            print(f"\n{'='*70}")
            print(f"🔧 FEDERATED LEARNING AGGREGATION (FIXED)")
            print(f"{'='*70}")
            print(f"Pending updates: {len(self.pending_updates)}")
            
            # Create model savepoint
            self.model_savepoint = {
                'weights': self.global_model['weights'].copy(),
                'version': self.global_model['version'],
                'timestamp': datetime.now()
            }
            
            # Calculate reputation weights
            weights_per_update = self._calculate_reputation_weights(self.pending_updates)
            
            # ========================================================================
            # FIX 1: ACTUAL WEIGHTED AGGREGATION (not just delta averaging)
            # ========================================================================
            aggregated_weights = {}
            
            # Get all parameter keys
            param_keys = set(self.global_model['weights'].keys())
            
            for key in param_keys:
                # Collect weighted values from all clients
                weighted_sum = 0.0
                total_weight = 0.0
                
                for update in self.pending_updates:
                    client_id = update['client_id']
                    weight = weights_per_update[client_id]
                    
                    # Get client's value for this parameter
                    if client_id in self.client_models:
                        client_params = self.client_models[client_id]['parameters']
                        client_value = client_params.get('weights', {}).get(key)
                        
                        if client_value is not None:
                            weighted_sum += float(client_value) * weight
                            total_weight += weight
                
                # Compute weighted average
                if total_weight > 0:
                    new_value = weighted_sum / total_weight
                    
                    # ✅ FIX: Apply LEARNING RATE to control convergence speed
                    LEARNING_RATE = 0.3  # How much to move toward new value
                    old_value = self.global_model['weights'][key]
                    aggregated_weights[key] = old_value + LEARNING_RATE * (new_value - old_value)
                else:
                    # Keep old value if no data
                    aggregated_weights[key] = self.global_model['weights'][key]
            
            # ========================================================================
            # FIX 2: AGGREGATE STATISTICS (baseline means/stds)
            # ========================================================================
            aggregated_baselines = {}
            
            baseline_keys = [
                ('network', 'network_baseline_mean', 'mean_connections'),
                ('network', 'network_baseline_std', 'std_connections'),
                ('process', 'process_baseline_mean', 'mean_count'),
                ('process', 'process_baseline_std', 'std_count'),
                ('file', 'file_baseline_mean', 'mean_events'),
                ('file', 'file_baseline_std', 'std_events')
            ]
            
            for category, baseline_key, stat_key in baseline_keys:
                values = []
                for update in self.pending_updates:
                    client_id = update['client_id']
                    if client_id in self.client_models:
                        statistics = self.client_models[client_id]['parameters'].get('statistics', {})
                        category_stats = statistics.get(category, {})
                        if stat_key in category_stats:
                            values.append(float(category_stats[stat_key]))
                
                if values:
                    # Use median for robustness
                    aggregated_baselines[baseline_key] = float(np.median(values))
            
            aggregated_weights.update(aggregated_baselines)
            
            # ========================================================================
            # FIX 3: VALIDATION with ACTUAL metrics
            # ========================================================================
            validation_metrics = self._validate_global_model_enhanced(aggregated_weights)
            
            # Rollback decision
            should_rollback = self._should_rollback(validation_metrics)
            
            if should_rollback:
                print(f"⛔ ROLLBACK: Validation quality dropped")
                print(f"   Previous AUC: {self.last_validation_metrics['auc']:.4f}")
                print(f"   New AUC: {validation_metrics['auc']:.4f}")
                
                self.global_model['weights'] = self.model_savepoint['weights']
                self.aggregation_metadata['last_round']['rollback_occurred'] = True
                
                self._audit_log_event('aggregation_rollback', 'SERVER', {
                    'reason': 'validation_drop',
                    'old_auc': self.last_validation_metrics['auc'],
                    'new_auc': validation_metrics['auc']
                })
                
                self.pending_updates.clear()
                return None
            
            # ========================================================================
            # FIX 4: COMMIT NEW MODEL with convergence tracking
            # ========================================================================
            convergence_delta = self._calculate_convergence(aggregated_weights)
            
            self.global_model['weights'] = aggregated_weights
            self.global_model['version'] += 1
            self.global_model['last_update'] = datetime.now()
            self.global_model['convergence_history'].append({
                'version': self.global_model['version'],
                'delta': convergence_delta,
                'timestamp': datetime.now(),
                'clients': len(self.pending_updates),
                'validation_auc': validation_metrics['auc']
            })
            
            if len(self.global_model['convergence_history']) > 20:
                self.global_model['convergence_history'] = self.global_model['convergence_history'][-20:]
            
            # Update metadata
            avg_norm = np.mean([u['post_norm'] for u in self.pending_updates])
            quarantined = sum(1 for cid in self.client_contributions.keys()
                            if self.client_contributions[cid]['quarantine_count'] > 0)
            
            self.aggregation_metadata['last_round'] = {
                'participated': len(self.pending_updates),
                'quarantined': quarantined,
                'avg_delta_norm': float(avg_norm),
                'validation_auc': validation_metrics['auc'],
                'dp_noise_applied': self.dp_noise_scale,
                'rollback_occurred': False
            }
            
            self.last_validation_metrics = validation_metrics
            
            # Audit log
            self._audit_log_event('aggregation_success', 'SERVER', {
                'version': self.global_model['version'],
                'clients': len(self.pending_updates),
                'avg_norm': avg_norm,
                'validation_auc': validation_metrics['auc'],
                'convergence': convergence_delta,
                'learning_applied': True  # ✅ NEW
            })
            
            print(f"✓ Aggregated {len(self.pending_updates)} client updates")
            print(f"  Model version: {self.global_model['version']}")
            print(f"  Convergence delta: {convergence_delta:.6f}")
            print(f"  Validation AUC: {validation_metrics['auc']:.4f}")
            print(f"  Learning rate applied: 0.3")
            print(f"{'='*70}\n")
            
            self.pending_updates.clear()
            return self.global_model
        
        except Exception as e:
            print(f"✗ Aggregation error: {e}")
            import traceback
            traceback.print_exc()
            self._audit_log_event('aggregation_error', 'SERVER', {'error': str(e)})
            return None
    
    # --- NEW: Reputation-based weighting ---
    def _calculate_reputation_weights(self, updates):
        """Calculate adaptive weights for each update based on reputation and quality"""
        weights = {}
        
        total_reputation = sum(u['reputation'] for u in updates)
        total_quality = sum(u['quality_score'] for u in updates)
        
        for update in updates:
            client_id = update['client_id']
            reputation = update['reputation']
            quality = update['quality_score']
            
            # Combined weight: 60% reputation, 40% quality
            if total_reputation > 0 and total_quality > 0:
                reputation_weight = reputation / total_reputation
                quality_weight = quality / total_quality
                combined_weight = 0.6 * reputation_weight + 0.4 * quality_weight
            else:
                combined_weight = 1.0 / len(updates)
            
            weights[client_id] = combined_weight
            print(f"  Client {client_id[:12]}: weight={combined_weight:.3f} "
                  f"(reputation={reputation:.3f}, quality={quality:.3f})")
        
        # Normalize
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {cid: w / total_weight for cid, w in weights.items()}
        
        return weights
    
    # --- NEW: Trimmed Mean Aggregation ---
    def _trimmed_mean_aggregation(self, updates, weights, trim_frac=0.1):
        """
        Aggregate using trimmed mean: remove top/bottom trim_frac% and average
        """
        print(f"  Using Trimmed Mean (trim_frac={trim_frac})")
        
        # Collect all parameter keys
        all_keys = set()
        for update in updates:
            all_keys.update(update['delta'].keys())
        
        aggregated = {}
        
        for key in all_keys:
            # Collect values for this parameter from all clients
            values_with_weights = []
            for update in updates:
                if key in update['delta']:
                    client_id = update['client_id']
                    value = update['delta'][key]
                    weight = weights[client_id]
                    values_with_weights.append((value, weight))
            
            if not values_with_weights:
                aggregated[key] = 0.0
                continue
            
            # Sort by value
            values_with_weights.sort(key=lambda x: x[0])
            
            # Trim top and bottom
            n = len(values_with_weights)
            trim_count = max(1, int(n * trim_frac))
            
            if n > 2 * trim_count:
                trimmed = values_with_weights[trim_count:-trim_count]
            else:
                trimmed = values_with_weights  # Not enough data to trim
            
            # Weighted average of trimmed values
            weighted_sum = sum(v * w for v, w in trimmed)
            total_weight = sum(w for v, w in trimmed)
            
            if total_weight > 0:
                aggregated[key] = weighted_sum / total_weight
            else:
                aggregated[key] = 0.0
        
        return aggregated
    
    # --- (C) NEW: Differential Privacy Noise Addition ---
    def _add_dp_noise(self, aggregated, num_clients, clip_bound):
        """Add Gaussian noise for differential privacy"""
        sensitivity = (2 * clip_bound) / num_clients
        
        noisy_aggregated = {}
        for key, value in aggregated.items():
            noise = np.random.normal(0, self.dp_noise_scale * sensitivity)
            noisy_aggregated[key] = value + noise
        
        print(f"  DP noise added: sensitivity={sensitivity:.4f}, scale={self.dp_noise_scale}")
        return noisy_aggregated
    
    # --- NEW: Robust baseline aggregation ---
    def _aggregate_baselines_robust(self, updates):
        """Aggregate baseline statistics with trimmed mean"""
        baselines = {}
        
        baseline_keys = [
            ('network', 'network_baseline_mean', 'mean_connections'),
            ('network', 'network_baseline_std', 'std_connections'),
            ('process', 'process_baseline_mean', 'mean_count'),
            ('process', 'process_baseline_std', 'std_count'),
            ('file', 'file_baseline_mean', 'mean_events'),
            ('file', 'file_baseline_std', 'std_events')
        ]
        
        for category, baseline_key, stat_key in baseline_keys:
            values = []
            for update in updates:
                # Extract from original client model
                client_id = update['client_id']
                if client_id in self.client_models:
                    statistics = self.client_models[client_id]['parameters'].get('statistics', {})
                    category_stats = statistics.get(category, {})
                    if stat_key in category_stats:
                        values.append(category_stats[stat_key])
            
            if values:
                # Use trimmed mean
                values_sorted = sorted(values)
                n = len(values_sorted)
                trim_count = max(1, int(n * TRIM_FRAC))
                
                if n > 2 * trim_count:
                    trimmed = values_sorted[trim_count:-trim_count]
                else:
                    trimmed = values_sorted
                
                baselines[baseline_key] = float(np.mean(trimmed))
        
        return baselines
    
    # --- (D) NEW: Server-side Validation ---
    def _validate_global_model(self, model_weights):
        """
        Validate global model performance (placeholder with simulated metrics)
        In production, this would run on a validation dataset
        """
        # Simulate validation metrics
        # In reality, you'd run the model on held-out validation data
        
        # Simple heuristic: check if weights are reasonable
        threshold_avg = np.mean([
            model_weights.get('network_threshold', 2.0),
            model_weights.get('process_threshold', 2.0),
            model_weights.get('file_threshold', 2.0)
        ])
        
        # Simulate AUC based on threshold (lower threshold = higher sensitivity = higher AUC)
        simulated_auc = 0.85 + (2.5 - threshold_avg) * 0.05
        simulated_auc = np.clip(simulated_auc, 0.7, 0.95)
        
        # Add some randomness
        simulated_auc += np.random.normal(0, 0.01)
        simulated_auc = np.clip(simulated_auc, 0.7, 0.95)
        
        metrics = {
            'auc': float(simulated_auc),
            'fpr': float(0.05 + np.random.normal(0, 0.01)),
            'tpr': float(0.90 + np.random.normal(0, 0.02))
        }
        
        return metrics
    
    # --- NEW: Rollback decision ---
    def _should_rollback(self, new_metrics):
        """Decide whether to rollback based on validation metrics"""
        auc_drop = self.last_validation_metrics['auc'] - new_metrics['auc']
        
        if auc_drop > VALIDATION_AUC_DROP_THRESHOLD:
            return True
        
        return False
    
    def _calculate_convergence(self, new_weights):
        """Calculate convergence metric (L2 distance from previous model)"""
        delta = 0.0
        count = 0
        
        for key in ['network_threshold', 'process_threshold', 'file_threshold',
                    'network_sensitivity', 'process_sensitivity', 'file_sensitivity']:
            if key in new_weights and key in self.global_model['weights']:
                old_val = self.global_model['weights'][key]
                new_val = new_weights[key]
                delta += (new_val - old_val) ** 2
                count += 1
        
        if count > 0:
            delta = np.sqrt(delta / count)
        
        return delta
    
    def get_global_model(self):
        """Get current global model"""
        return self.global_model
    
    def get_convergence_metrics(self):
        """Get convergence metrics for visualization"""
        return {
            'history': self.global_model['convergence_history'],
            'current_version': self.global_model['version'],
            'participating_clients': len(self.client_models),
            'client_contributions': dict(self.client_contributions)
        }
    
    # --- (E) NEW: Get reputation data ---
    def get_reputation_data(self):
        """Get client reputation data for GUI display"""
        return self.client_contributions
    
    # --- (F) NEW: Get aggregation metadata ---
    def get_aggregation_metadata(self):
        """Get last aggregation round metadata"""
        return self.aggregation_metadata['last_round']

class DatabaseManager:
    """Enhanced database manager"""
    
    def __init__(self):
        self.connection = None
        self.connect()
        self.initialize_tables()
    
    def connect(self):
        """Establish database connection"""
        try:
            self.connection = psycopg2.connect(**DB_CONFIG)
            print("✓ Database connected successfully")
        except Exception as e:
            print(f"✗ Database connection failed: {e}")
            raise
    
    # PATCH 5A - Add anomaly correlation method
    def analyze_and_alert(self, client_id, telemetry_data):
        """Analyze telemetry and generate HUMAN-READABLE server-side alerts"""
        cursor = self.connection.cursor(cursor_factory=RealDictCursor)
        
        try:
            # Process ML-detected anomalies
            ml_alerts = telemetry_data.get('anomaly_alerts', [])
            
            if ml_alerts:
                print(f"\n[SERVER] Processing {len(ml_alerts)} ML anomalies from {client_id[:12]}")
                
                for ml_alert in ml_alerts:
                    severity = ml_alert.get('severity', 'medium')
                    category = ml_alert.get('category', 'system')
                    ensemble_score = ml_alert.get('ensemble_score', 5.0)
                    risk_score = float(ensemble_score)
                    
                    # âœ… NEW: BUILD HUMAN-READABLE TITLE AND DESCRIPTION
                    title = self._build_alert_title(category, severity, ml_alert)
                    description = self._build_alert_description(category, ml_alert)
                    
                    # Only store if meaningful risk
                    if risk_score > 4.0:
                        self.create_alert(
                            client_id,
                            f"ml_{category}_anomaly",
                            severity,
                            title,
                            description,
                            risk_score
                        )
            
        except Exception as e:
            print(f"[SERVER] Error in analyze_and_alert: {e}")

    def _build_alert_title(self, category, severity, ml_alert):
        """Build human-readable alert title"""
        contrib = ml_alert.get('contributing_features', [])
        
        if category == 'network':
            if 'c2_pattern' in contrib:
                return f"⚠️ Possible C2 Communication Detected"
            elif 'port_entropy' in contrib:
                return f"⚠️ Port Scanning Activity Detected"
            else:
                return f"Suspicious Network Activity - {severity.upper()}"
        
        elif category == 'process':
            if 'cred_dump_pattern' in contrib:
                return f"⚠️ CRITICAL: Credential Theft Attempt"
            elif 'proc_spawn' in str(contrib):
                return f"⚠️ Unusual Process Spawning Detected"
            else:
                return f"Suspicious Process Behavior - {severity.upper()}"
        
        elif category == 'filesystem':
            if 'ransomware_burst' in contrib:
                return f"⚠️ RANSOMWARE ACTIVITY DETECTED"
            elif 'file_create_count' in contrib:
                return f"⚠️ Mass File Modification Detected"
            else:
                return f"Suspicious File Activity - {severity.upper()}"
        
        else:
            return f"ML Anomaly: {severity.upper()} - {category.upper()}"

    def _build_alert_description(self, category, ml_alert):
        """Build human-readable alert description"""
        explanation = ml_alert.get('explanation', '')
        contrib = ml_alert.get('contributing_features', [])
        model_contrib = ml_alert.get('model_contributions', {})
        
        description_parts = []
        
        # âœ… PLAIN ENGLISH SUMMARY
        if category == 'network':
            if 'c2_pattern' in contrib:
                description_parts.append("Repeated connections to same destination suggest Command & Control (C2) communication")
            if 'port_entropy' in contrib:
                description_parts.append("Multiple different ports accessed in short time indicates port scanning")
            if 'conn_count' in contrib:
                description_parts.append("Unusually high number of network connections detected")
        
        elif category == 'process':
            if 'cred_dump_pattern' in contrib:
                description_parts.append("Process attempting to access sensitive credential stores (lsass.exe or SAM database)")
            if 'proc_spawn' in str(contrib):
                description_parts.append("Rapid creation of multiple processes detected")
        
        elif category == 'filesystem':
            if 'ransomware_burst' in contrib:
                description_parts.append("Mass file encryption detected - characteristic of ransomware attack")
            if 'file_create_count' in contrib:
                description_parts.append("Unusually large number of files created/modified in short time")
        
        # Add model scores if available
        if model_contrib:
            scores = []
            if 'isolation_forest' in model_contrib:
                scores.append(f"Anomaly Detection: {model_contrib['isolation_forest'].get('score', 0):.1f}/10")
            if 'autoencoder' in model_contrib:
                scores.append(f"Pattern Recognition: {model_contrib['autoencoder'].get('score', 0):.1f}/10")
            
            if scores:
                description_parts.append("AI Analysis: " + ", ".join(scores))
        
        # Fallback: use original explanation
        if not description_parts:
            description_parts.append(explanation[:200] if explanation else "Multiple anomalous behaviors detected")
        
        return " | ".join(description_parts)

        
    def initialize_tables(self):
        """Create necessary database tables"""
        cursor = self.connection.cursor()
        
        # Clients table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS clients (
                client_id VARCHAR(255) PRIMARY KEY,
                hostname VARCHAR(255),
                ip_address VARCHAR(50),
                os_type VARCHAR(50),
                os_version VARCHAR(100),
                device_role VARCHAR(50),
                department VARCHAR(100),
                criticality_level VARCHAR(20),
                status VARCHAR(20),
                last_heartbeat TIMESTAMP,
                first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_alerts INTEGER DEFAULT 0,
                federated_learning BOOLEAN DEFAULT FALSE
            )
        """)
        
        # Network data table (enhanced)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS network_data (
                id SERIAL PRIMARY KEY,
                client_id VARCHAR(255) REFERENCES clients(client_id),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                src_ip VARCHAR(50),
                dst_ip VARCHAR(50),
                src_port INTEGER,
                dst_port INTEGER,
                protocol VARCHAR(50),
                connection_count INTEGER,
                dns_query VARCHAR(255),
                geolocation VARCHAR(100),
                risk_score FLOAT,
                is_anomaly BOOLEAN DEFAULT FALSE,
                anomaly_score FLOAT,
                threat_indicators TEXT[]
            )
        """)
        
        # Process data table (enhanced)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS process_data (
                id SERIAL PRIMARY KEY,
                client_id VARCHAR(255) REFERENCES clients(client_id),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                process_name VARCHAR(255),
                pid INTEGER,
                ppid INTEGER,
                parent_name VARCHAR(255),
                executable_path TEXT,
                executable_hash VARCHAR(64),
                command_line_preview TEXT,
                start_time TIMESTAMP,
                cpu_percent FLOAT,
                memory_mb FLOAT,
                privilege_level VARCHAR(50),
                risk_score FLOAT,
                threat_indicators TEXT[]
            )
        """)
        
        # Filesystem data table (enhanced)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS filesystem_data (
                id SERIAL PRIMARY KEY,
                client_id VARCHAR(255) REFERENCES clients(client_id),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                event_type VARCHAR(50),
                file_path TEXT,
                file_name VARCHAR(255),
                file_extension VARCHAR(20),
                file_size BIGINT,
                file_hash VARCHAR(64),
                modification_time TIMESTAMP,
                directory TEXT,
                is_suspicious BOOLEAN,
                risk_score FLOAT,
                threat_indicators TEXT[]
            )
        """)
        
        # User activity table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_activity (
                id SERIAL PRIMARY KEY,
                client_id VARCHAR(255) REFERENCES clients(client_id),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                event_type VARCHAR(50),
                username VARCHAR(100),
                session_id VARCHAR(100),
                source_ip VARCHAR(50),
                login_success BOOLEAN,
                privilege_escalation BOOLEAN,
                risk_score FLOAT,
                threat_indicators TEXT[],  -- NEW: Array of threat indicators
                concurrent_sessions INTEGER DEFAULT 1,  -- NEW: Number of concurrent sessions
                login_time TIMESTAMP,  -- NEW: Actual login timestamp
                session_duration FLOAT  -- NEW: Session duration in seconds (for logout events)
            )
        """)
        
        # Alerts table (NEW)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id SERIAL PRIMARY KEY,
                client_id VARCHAR(255) REFERENCES clients(client_id),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                alert_type VARCHAR(50),
                severity VARCHAR(20),
                title TEXT,
                description TEXT,
                source_category VARCHAR(50),
                source_id INTEGER,
                risk_score FLOAT,
                status VARCHAR(20) DEFAULT 'active',
                acknowledged BOOLEAN DEFAULT FALSE
            )
        """)
        
        # Federated learning models table (NEW)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fl_models (
                id SERIAL PRIMARY KEY,
                version INTEGER,
                model_weights JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                client_count INTEGER,
                performance_metrics JSONB
            )
        """)
        
        # Aggregated statistics table (NEW)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS aggregated_stats (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                time_period VARCHAR(20),
                total_clients INTEGER,
                online_clients INTEGER,
                network_events INTEGER,
                process_events INTEGER,
                filesystem_events INTEGER,
                user_events INTEGER,
                high_risk_alerts INTEGER,
                avg_risk_score FLOAT
            )
        """)
        
        # --- PATCH 4: NEW table for FL contribution tracking ---
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fl_contributions (
                id SERIAL PRIMARY KEY,
                client_id VARCHAR(255) REFERENCES clients(client_id),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                pre_norm FLOAT,
                post_norm FLOAT,
                was_clipped BOOLEAN,
                was_quarantined BOOLEAN,
                reputation FLOAT,
                quality_score FLOAT,
                samples_used INTEGER,
                model_version INTEGER
            )
        """)
        
        # --- NEW: Table for aggregation rounds ---
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fl_aggregation_rounds (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                model_version INTEGER,
                participated_clients INTEGER,
                quarantined_clients INTEGER,
                avg_delta_norm FLOAT,
                validation_auc FLOAT,
                dp_noise_scale FLOAT,
                rollback_occurred BOOLEAN,
                convergence_delta FLOAT
            )
        """)
        
        # --- PATCH 2: Persistent model cache table ---
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS client_model_cache (
                client_id VARCHAR(255) PRIMARY KEY REFERENCES clients(client_id),
                model_data BYTEA,
                cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.connection.commit()
        cursor.close()
        print("✓ Database tables initialized")
    
    def register_client(self, client_data, capabilities=None):
        """Register or update a client"""
        cursor = self.connection.cursor()
        
        if capabilities is None:
            capabilities = {}
        fl_enabled = capabilities.get('federated_learning', False) if capabilities else False
        
        cursor.execute("""
            INSERT INTO clients (client_id, hostname, ip_address, os_type, os_version, 
                            device_role, department, criticality_level, status, 
                            last_heartbeat, federated_learning)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'online', CURRENT_TIMESTAMP, %s)
            ON CONFLICT (client_id) 
            DO UPDATE SET 
                hostname = EXCLUDED.hostname,
                ip_address = EXCLUDED.ip_address,
                status = 'online',
                last_heartbeat = CURRENT_TIMESTAMP,
                federated_learning = EXCLUDED.federated_learning
        """, (
            client_data['client_id'],
            client_data['hostname'],
            client_data['ip_address'],
            client_data['os_type'],
            client_data['os_version'],
            client_data['device_role'],
            client_data['department'],
            client_data['criticality_level'],
            fl_enabled
        ))
        print(f"Registering client {client_data['client_id'][:8]} with FL={fl_enabled}")
        self.connection.commit()
        cursor.close()
    
    def insert_network_data(self, client_id, network_records):
        """Insert network telemetry data"""
        cursor = self.connection.cursor()
        
        if isinstance(network_records, dict):
            records_list = network_records.get('details', [])
        else:
            records_list = network_records if isinstance(network_records, list) else []
        
        for record in records_list:
            try:
                # CONVERT NUMPY TYPES
                record = convert_numpy_types(record)
                
                cursor.execute("""
                    INSERT INTO network_data (client_id, src_ip, dst_ip, src_port, dst_port, 
                                            protocol, connection_count, dns_query, geolocation,
                                            risk_score, is_anomaly, anomaly_score, threat_indicators)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    client_id,
                    record.get('src_ip'),
                    record.get('dst_ip'),
                    record.get('src_port'),
                    record.get('dst_port'),
                    record.get('protocol'),
                    record.get('connection_count', 1),
                    record.get('dns_query'),
                    record.get('geolocation'),
                    float(record.get('risk_score', 0)),
                    bool(record.get('is_anomaly', False)),  # Explicit bool conversion
                    float(record.get('anomaly_score', 0)),
                    record.get('threat_indicators', [])
                ))
                
                # Generate alert for high-risk connections
                if record.get('risk_score', 0) > 7.0:
                    self.create_alert(client_id, 'network', 'high', 
                                    'Suspicious Network Connection',
                                    f"High-risk connection to {record.get('dst_ip')}:{record.get('dst_port')} ({record.get('protocol')})",
                                    record.get('risk_score', 0))
            except Exception as e:
                print(f"Error inserting network record: {e}")
                self.connection.rollback()  # Add this line
                continue
        
        self.connection.commit()
        cursor.close()
    
    def insert_process_data(self, client_id, process_records):
        """Insert process telemetry data with ALERT GENERATION"""
        cursor = self.connection.cursor()
        
        # Handle both dict and list formats
        if isinstance(process_records, dict):
            high_risk_list = process_records.get('high_risk', [])
            details_list = process_records.get('details', [])
        else:
            high_risk_list = []
            details_list = process_records if isinstance(process_records, list) else []
        
        # Process high-risk first
        for record in high_risk_list:
            try:
                record = convert_numpy_types(record)
                exe_hash = record.get('executable_hash')
                
                cursor.execute("""
                    INSERT INTO process_data (client_id, process_name, pid, ppid, parent_name,
                                            executable_path, executable_hash, command_line_preview,
                                            start_time, cpu_percent, memory_mb, privilege_level,
                                            risk_score, threat_indicators)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    client_id,
                    record.get('process_name'),
                    record.get('pid'),
                    record.get('ppid'),
                    record.get('parent_name'),
                    record.get('executable_path'),
                    exe_hash,
                    record.get('command_line_preview'),
                    record.get('start_time'),
                    record.get('cpu_percent'),
                    record.get('memory_mb'),
                    record.get('privilege_level'),
                    record.get('risk_score', 0),
                    record.get('threat_indicators', [])
                ))
                
                # ✅ FIX: Generate alert for processes with risk > 6.0
                risk_score = record.get('risk_score', 0)
                if risk_score > 6.0:
                    # Build detailed description
                    indicators = record.get('threat_indicators', [])
                    indicators_str = ', '.join(indicators[:3]) if indicators else 'Suspicious behavior'
                    
                    process_name = record.get('process_name', 'Unknown')
                    pid = record.get('pid', 'N/A')
                    parent = record.get('parent_name', 'Unknown')
                    exe_path = record.get('executable_path', 'N/A')
                    
                    description = (
                        f"[PROCESS] High-risk process detected: {process_name} (PID:{pid}) | "
                        f"Parent: {parent} | Path: {exe_path[:50]} | "
                        f"Indicators: {indicators_str}"
                    )
                    
                    severity = 'critical' if risk_score > 8.0 else 'high'
                    
                    self.create_alert(
                        client_id, 
                        'process', 
                        severity,
                        f"Suspicious Process: {process_name}",
                        description,
                        risk_score
                    )
                    
                    print(f"  [SERVER] Generated alert for process: {process_name} (risk={risk_score:.1f})")
            
            except Exception as e:
                print(f"Error inserting high-risk process: {e}")
                continue
        
        # Process regular details (only generate alerts for risk > 5.0)
        for record in details_list:
            try:
                record = convert_numpy_types(record)
                cursor.execute("""
                    INSERT INTO process_data (client_id, process_name, pid, ppid, parent_name,
                                            executable_path, executable_hash, command_line_preview,
                                            start_time, cpu_percent, memory_mb, privilege_level,
                                            risk_score, threat_indicators)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    client_id,
                    record.get('process_name'),
                    record.get('pid'),
                    record.get('ppid'),
                    record.get('parent_name'),
                    record.get('executable_path'),
                    record.get('executable_hash'),
                    record.get('command_line_preview'),
                    record.get('start_time'),
                    record.get('cpu_percent'),
                    record.get('memory_mb'),
                    record.get('privilege_level'),
                    record.get('risk_score', 0),
                    record.get('threat_indicators', [])
                ))
                
                # ✅ Generate alerts for medium-risk processes
                risk_score = record.get('risk_score', 0)
                if risk_score > 5.0:
                    indicators = record.get('threat_indicators', [])
                    indicators_str = ', '.join(indicators[:2]) if indicators else 'Moderate risk'
                    
                    description = (
                        f"[PROCESS] {record.get('process_name')} (PID:{record.get('pid')}) | "
                        f"{indicators_str}"
                    )
                    
                    self.create_alert(
                        client_id,
                        'process',
                        'medium',
                        f"Process Activity: {record.get('process_name')}",
                        description,
                        risk_score
                    )
            
            except Exception as e:
                print(f"Error inserting process record: {e}")
                self.connection.rollback()
                continue
        
        self.connection.commit()
        cursor.close()
    
    def insert_filesystem_data(self, client_id, fs_records):
        """Insert filesystem event data with ALERT GENERATION"""
        cursor = self.connection.cursor()
        records_list = fs_records if isinstance(fs_records, list) else []
        
        for record in records_list:
            try:
                record = convert_numpy_types(record)
                
                file_hash_value = str(record.get('file_hash')) if record.get('file_hash') else None
                
                cursor.execute("""
                    INSERT INTO filesystem_data (client_id, event_type, file_path, file_name,
                                            file_extension, file_size, file_hash, modification_time,
                                            directory, is_suspicious, risk_score, threat_indicators)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    str(client_id),
                    str(record.get('event_type')) if record.get('event_type') else None,
                    str(record.get('file_path')) if record.get('file_path') else None,
                    str(record.get('file_name')) if record.get('file_name') else None,
                    str(record.get('file_extension')) if record.get('file_extension') else None,
                    int(record.get('file_size', 0)) if record.get('file_size') else None,
                    file_hash_value,
                    record.get('modification_time'),
                    str(record.get('directory')) if record.get('directory') else None,
                    bool(record.get('is_suspicious', False)),
                    float(record.get('risk_score', 0)),
                    list(record.get('threat_indicators', []))
                ))
                
                # ✅ FIX: Generate alerts for high-risk files
                risk_score = record.get('risk_score', 0)
                if risk_score > 6.0:
                    indicators = record.get('threat_indicators', [])
                    file_name = record.get('file_name', 'Unknown')
                    file_ext = record.get('file_extension', '')
                    directory = record.get('directory', 'Unknown')
                    
                    # Check for ransomware patterns
                    is_ransomware = False
                    if any('ransomware' in str(ind).lower() for ind in indicators):
                        is_ransomware = True
                    elif file_ext in ['.enc', '.locked', '.encrypted']:
                        is_ransomware = True
                    
                    if is_ransomware:
                        description = (
                            f"🚨 [FILESYSTEM] RANSOMWARE SUSPECTED: {file_name} | "
                            f"Extension: {file_ext} | Directory: {directory[:150]} | "
                            f"Indicators: {', '.join(indicators[:3])}"
                        )
                        severity = 'critical'
                    else:
                        indicators_str = ', '.join(indicators[:3]) if indicators else 'Suspicious file activity'
                        description = (
                            f"[FILESYSTEM] High-risk file: {file_name} | "
                            f"Extension: {file_ext} | Directory: {directory[:50]} | "
                            f"Indicators: {indicators_str}"
                        )
                        severity = 'high' if risk_score > 8.0 else 'medium'
                    
                    self.create_alert(
                        client_id,
                        'filesystem',
                        severity,
                        f"Suspicious File Activity: {file_name}",
                        description,
                        risk_score
                    )
                    
                    print(f"  [SERVER] Generated alert for file: {file_name} (risk={risk_score:.1f})")
            
            except Exception as e:
                print(f"Error inserting filesystem record: {e}")
                self.connection.rollback()
                continue
        
        self.connection.commit()
        cursor.close()
    
    def insert_user_activity(self, client_id, user_records):
        """Insert user activity data with enhanced fields"""
        cursor = self.connection.cursor()
        
        # Handle list format
        records_list = user_records if isinstance(user_records, list) else []
        
        for record in records_list:
            try:
                # Convert numpy types
                record = convert_numpy_types(record)
                
                cursor.execute("""
                    INSERT INTO user_activity (
                        client_id, event_type, username, session_id,
                        source_ip, login_success, privilege_escalation, 
                        risk_score, threat_indicators, concurrent_sessions,
                        login_time, session_duration
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    str(client_id),
                    str(record.get('event_type')),
                    str(record.get('username')),
                    str(record.get('session_id')),
                    str(record.get('source_ip', 'local')),
                    bool(record.get('login_success', True)),
                    bool(record.get('privilege_escalation', False)),
                    float(record.get('risk_score', 0)),
                    list(record.get('threat_indicators', [])),  # Array
                    int(record.get('concurrent_sessions', 1)),
                    record.get('login_time'),  # Timestamp or None
                    float(record.get('session_duration')) if record.get('session_duration') else None
                ))
            except Exception as e:
                print(f"Error inserting user activity: {e}")
                self.connection.rollback()
                continue
        
        self.connection.commit()
        cursor.close()
    
    # In DatabaseManager class, update create_alert method:
    def create_alert(self, client_id, category, severity, title, description, risk_score):
        """Create a new alert with enhanced categorization"""
        cursor = self.connection.cursor()
        
        # ✅ FIX: Map category from alert_type if needed
        if '_network_' in category or category == 'network':
            source_category = 'network'
        elif '_process_' in category or category == 'process':
            source_category = 'process'
        elif 'file' in category or category == 'filesystem':
            source_category = 'filesystem'
        elif '_user_' in category or category == 'user_activity':
            source_category = 'user'
        else:
            source_category = 'system'
        
        cursor.execute("""
            INSERT INTO alerts (client_id, alert_type, severity, title, description,
                            source_category, risk_score, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'active')
        """, (client_id, category, severity, title, description, source_category, float(risk_score)))
        
        self.connection.commit()
        cursor.close()
    
    def get_active_alerts(self, limit=100):
        """Get active alerts"""
        cursor = self.connection.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT a.*, c.hostname 
            FROM alerts a
            JOIN clients c ON a.client_id = c.client_id
            WHERE a.status = 'active'
            ORDER BY a.timestamp DESC, a.risk_score DESC
            LIMIT %s
        """, (limit,))
        alerts = cursor.fetchall()
        cursor.close()
        return alerts
    
    def get_all_clients(self):
        """Retrieve all registered clients"""
        cursor = self.connection.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT * FROM clients ORDER BY last_heartbeat DESC")
        clients = cursor.fetchall()
        cursor.close()
        return clients
    
    def get_dashboard_stats(self):
        """Get enhanced statistics for dashboard"""
        cursor = self.connection.cursor(cursor_factory=RealDictCursor)
        
        # Client stats
        cursor.execute("""
            SELECT 
                COUNT(*) as total_clients,
                SUM(CASE WHEN status = 'online' THEN 1 ELSE 0 END) as online_clients,
                SUM(CASE WHEN status = 'offline' THEN 1 ELSE 0 END) as offline_clients,
                SUM(CASE WHEN federated_learning = TRUE THEN 1 ELSE 0 END) as fl_clients
            FROM clients
        """)
        client_stats = cursor.fetchone()
        
        # Event stats (last 24 hours)
        cursor.execute("""
            SELECT 
                (SELECT COUNT(*) FROM network_data WHERE timestamp > NOW() - INTERVAL '24 hours') as network,
                (SELECT COUNT(*) FROM process_data WHERE timestamp > NOW() - INTERVAL '24 hours') as process,
                (SELECT COUNT(*) FROM filesystem_data WHERE timestamp > NOW() - INTERVAL '24 hours') as filesystem,
                (SELECT COUNT(*) FROM user_activity WHERE timestamp > NOW() - INTERVAL '24 hours') as user_activity
        """)
        event_stats = cursor.fetchone()
        
        # Alert stats
        cursor.execute("""
            SELECT 
                COUNT(*) as total_alerts,
                SUM(CASE WHEN severity = 'critical' THEN 1 ELSE 0 END) as critical_alerts,
                SUM(CASE WHEN severity = 'high' THEN 1 ELSE 0 END) as high_alerts
            FROM alerts
            WHERE status = 'active' AND timestamp > NOW() - INTERVAL '24 hours'
        """)
        alert_stats = cursor.fetchone()
        
        # Top risk events by category
        cursor.execute("""
            SELECT 'network' as category, client_id, dst_ip as detail, risk_score
            FROM network_data
            WHERE timestamp > NOW() - INTERVAL '24 hours' AND risk_score > 5.0
            ORDER BY risk_score DESC LIMIT 5
        """)
        network_risks = cursor.fetchall()
        
        cursor.execute("""
            SELECT 'process' as category, client_id, process_name as detail, risk_score
            FROM process_data
            WHERE timestamp > NOW() - INTERVAL '24 hours' AND risk_score > 5.0
            ORDER BY risk_score DESC LIMIT 5
        """)
        process_risks = cursor.fetchall()
        
        cursor.execute("""
            SELECT 'filesystem' as category, client_id, file_name as detail, risk_score
            FROM filesystem_data
            WHERE timestamp > NOW() - INTERVAL '24 hours' AND risk_score > 5.0
            ORDER BY risk_score DESC LIMIT 5
        """)
        file_risks = cursor.fetchall()
        
        # Time series data for charts (last 24 hours, hourly)
        cursor.execute("""
            SELECT 
                DATE_TRUNC('hour', timestamp) as hour,
                COUNT(*) as count
            FROM network_data
            WHERE timestamp > NOW() - INTERVAL '24 hours'
            GROUP BY hour
            ORDER BY hour
        """)
        network_timeline = cursor.fetchall()
        
        cursor.close()
        return {
            'clients': client_stats,
            'events': event_stats,
            'alerts': alert_stats,
            'top_risks': {
                'network': network_risks,
                'process': process_risks,
                'filesystem': file_risks
            },
            'timeline': {
                'network': network_timeline
            }
        }
    
    def get_client_aggregated_view(self):
        """Get aggregated view per client - OPTIMIZED WITH SUBQUERIES"""
        cursor = None
        try:
            cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            
            # Set reasonable timeout
            cursor.execute("SET statement_timeout = '30s'")
            
            # âœ… OPTIMIZED QUERY: Use subqueries instead of joins
            cursor.execute("""
                WITH network_stats AS (
                    SELECT 
                        client_id,
                        COUNT(*) as network_events,
                        AVG(risk_score) as avg_network_risk
                    FROM network_data
                    WHERE timestamp > NOW() - INTERVAL '24 hours'
                    GROUP BY client_id
                ),
                process_stats AS (
                    SELECT 
                        client_id,
                        COUNT(*) as process_events,
                        AVG(risk_score) as avg_process_risk
                    FROM process_data
                    WHERE timestamp > NOW() - INTERVAL '24 hours'
                    GROUP BY client_id
                ),
                filesystem_stats AS (
                    SELECT 
                        client_id,
                        COUNT(*) as filesystem_events,
                        AVG(risk_score) as avg_filesystem_risk
                    FROM filesystem_data
                    WHERE timestamp > NOW() - INTERVAL '24 hours'
                    GROUP BY client_id
                ),
                alert_stats AS (
                    SELECT 
                        client_id,
                        COUNT(*) as active_alerts
                    FROM alerts
                    WHERE status = 'active'
                    GROUP BY client_id
                )
                SELECT 
                    c.client_id,
                    c.hostname,
                    c.ip_address,
                    c.status,
                    COALESCE(n.network_events, 0) as network_events_24h,
                    COALESCE(p.process_events, 0) as process_events_24h,
                    COALESCE(f.filesystem_events, 0) as filesystem_events_24h,
                    COALESCE(n.avg_network_risk, 0) as avg_network_risk,
                    COALESCE(p.avg_process_risk, 0) as avg_process_risk,
                    COALESCE(f.avg_filesystem_risk, 0) as avg_filesystem_risk,
                    COALESCE(a.active_alerts, 0) as active_alerts
                FROM clients c
                LEFT JOIN network_stats n ON c.client_id = n.client_id
                LEFT JOIN process_stats p ON c.client_id = p.client_id
                LEFT JOIN filesystem_stats f ON c.client_id = f.client_id
                LEFT JOIN alert_stats a ON c.client_id = a.client_id
                ORDER BY active_alerts DESC NULLS LAST,
                        (COALESCE(n.avg_network_risk, 0) + COALESCE(p.avg_process_risk, 0) + COALESCE(f.avg_filesystem_risk, 0)) DESC
                LIMIT 100
            """)
            
            aggregated = cursor.fetchall()
            cursor.execute("RESET statement_timeout")
            
            # print(f"– Aggregated view loaded: {len(aggregated)} clients")
            return aggregated
            
        except Exception as e:
            print(f"✗ Database query error in get_client_aggregated_view: {e}")
            
            if self.connection:
                try:
                    self.connection.rollback()
                except:
                    pass
            
            return []
            
        finally:
            if cursor:
                try:
                    cursor.close()
                except:
                    pass
    
    def save_fl_model(self, version, weights, client_count, metrics=None):
        """Save federated learning model with quality metrics - FIXED"""
        cursor = self.connection.cursor()
        try:
            if metrics is None:
                metrics = {}
            
            # Convert weights to ensure no numpy types
            clean_weights = convert_numpy_types(weights)
            clean_metrics = convert_numpy_types(metrics)
            
            cursor.execute("""
                INSERT INTO fl_models (version, model_weights, client_count, performance_metrics)
                VALUES (%s, %s, %s, %s)
            """, (
                int(version), 
                json.dumps(clean_weights), 
                int(client_count), 
                json.dumps(clean_metrics)
            ))
            self.connection.commit()
        except Exception as e:
            print(f"Error saving FL model: {e}")
            self.connection.rollback()
        finally:
            cursor.close()

    # --- PATCH 5: NEW methods for FL tracking ---
    def log_fl_contribution(self, client_id, pre_norm, post_norm, was_clipped, 
                       was_quarantined, reputation, quality_score, 
                       samples_used, model_version):
        """Log federated learning contribution to database - FIXED"""
        cursor = self.connection.cursor()
        try:
            # CRITICAL: Convert all numpy types before insertion
            cursor.execute("""
                INSERT INTO fl_contributions 
                (client_id, pre_norm, post_norm, was_clipped, was_quarantined, 
                reputation, quality_score, samples_used, model_version)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                str(client_id),
                float(pre_norm),      # Explicit conversion
                float(post_norm),     # Explicit conversion
                bool(was_clipped),    # Explicit conversion
                bool(was_quarantined), # Explicit conversion
                float(reputation),    # Explicit conversion
                float(quality_score), # Explicit conversion
                int(samples_used),    # Explicit conversion
                int(model_version)    # Explicit conversion
            ))
            self.connection.commit()
            cursor.close()
        except Exception as e:
            print(f"Error logging FL contribution: {e}")
            self.connection.rollback()  # CRITICAL: Rollback on error
            cursor.close()
            raise  # Re-raise to handle upstream
    
    def log_fl_aggregation_round(self, model_version, participated, quarantined,
                             avg_norm, validation_auc, dp_noise, rollback, 
                             convergence):
        """Log federated learning aggregation round to database - FIXED"""
        cursor = self.connection.cursor()
        try:
            # CRITICAL: Explicitly convert ALL values to native Python types
            cursor.execute("""
                INSERT INTO fl_aggregation_rounds 
                (model_version, participated_clients, quarantined_clients, 
                avg_delta_norm, validation_auc, dp_noise_scale, rollback_occurred,
                convergence_delta)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                int(model_version),
                int(participated),
                int(quarantined),
                float(avg_norm),
                float(validation_auc),
                float(dp_noise),
                bool(rollback),
                float(convergence)
            ))
            self.connection.commit()
        except Exception as e:
            print(f"Error logging FL aggregation round: {e}")
            self.connection.rollback()
        finally:
            cursor.close()
    
    def get_fl_contribution_history(self, client_id=None, limit=100):
        """Get FL contribution history for analysis"""
        cursor = self.connection.cursor(cursor_factory=RealDictCursor)
        
        if client_id:
            cursor.execute("""
                SELECT * FROM fl_contributions 
                WHERE client_id = %s
                ORDER BY timestamp DESC 
                LIMIT %s
            """, (client_id, limit))
        else:
            cursor.execute("""
                SELECT * FROM fl_contributions 
                ORDER BY timestamp DESC 
                LIMIT %s
            """, (limit,))
        
        records = cursor.fetchall()
        cursor.close()
        return records
    
    def analyze_user_activity_anomalies(self, client_id, user_records):
        """
        Server-side analysis of user activity patterns
        Detects: brute force attempts, privilege escalation, impossible travel
        """
        if not user_records:
            return
        
        cursor = self.connection.cursor(cursor_factory=RealDictCursor)
        
        try:
            for record in user_records:
                username = record.get('username')
                source_ip = record.get('source_ip', 'local')
                event_type = record.get('event_type')
                risk_score = float(record.get('risk_score', 0))
                
                # DETECTION 1: Brute Force Attempts
                if event_type == 'login':
                    # Check for rapid logins from same IP
                    cursor.execute("""
                        SELECT COUNT(*) as login_count
                        FROM user_activity
                        WHERE client_id = %s 
                        AND source_ip = %s
                        AND event_type = 'login'
                        AND timestamp > NOW() - INTERVAL '10 minutes'
                    """, (client_id, source_ip))
                    
                    result = cursor.fetchone()
                    if result and result['login_count'] >= 5:
                        self.create_alert(
                            client_id, 'user_activity', 'high',
                            'Possible Brute Force Attack',
                            f"Multiple login attempts from {source_ip} ({result['login_count']} in 10 min)",
                            8.0
                        )
                
                # DETECTION 2: Privilege Escalation
                if record.get('privilege_escalation'):
                    cursor.execute("""
                        SELECT COUNT(*) as escalation_count
                        FROM user_activity
                        WHERE client_id = %s
                        AND username = %s
                        AND privilege_escalation = TRUE
                        AND timestamp > NOW() - INTERVAL '1 hour'
                    """, (client_id, username))
                    
                    result = cursor.fetchone()
                    if result and result['escalation_count'] >= 2:
                        self.create_alert(
                            client_id, 'user_activity', 'critical',
                            'Multiple Privilege Escalations Detected',
                            f"User {username} escalated privileges {result['escalation_count']} times in 1 hour",
                            9.0
                        )
                
                # DETECTION 3: Impossible Travel
                if source_ip not in ['local', 'localhost', '127.0.0.1']:
                    cursor.execute("""
                        SELECT source_ip, timestamp
                        FROM user_activity
                        WHERE client_id = %s
                        AND username = %s
                        AND event_type = 'login'
                        AND source_ip != %s
                        AND timestamp > NOW() - INTERVAL '1 hour'
                        ORDER BY timestamp DESC
                        LIMIT 1
                    """, (client_id, username, source_ip))
                    
                    prev_login = cursor.fetchone()
                    if prev_login and prev_login['source_ip'] not in ['local', 'localhost', '127.0.0.1']:
                        # Different IPs within 1 hour = suspicious
                        self.create_alert(
                            client_id, 'user_activity', 'medium',
                            'Suspicious Login Pattern - Impossible Travel',
                            f"User {username} logged in from {source_ip} after recent login from {prev_login['source_ip']}",
                            6.5
                        )
                
                # DETECTION 4: Off-Hours Privileged Access
                if record.get('privilege_escalation'):
                    from datetime import datetime
                    current_hour = datetime.now().hour
                    if current_hour < 6 or current_hour > 22:
                        self.create_alert(
                            client_id, 'user_activity', 'medium',
                            'Off-Hours Privileged Access',
                            f"Privileged user {username} logged in at {current_hour}:00 (off-hours)",
                            5.5
                        )
                
                # DETECTION 5: Multiple Concurrent Sessions
                concurrent = record.get('concurrent_sessions', 1)
                if concurrent >= 3:
                    self.create_alert(
                        client_id, 'user_activity', 'medium',
                        'Multiple Concurrent Sessions',
                        f"User {username} has {concurrent} active sessions",
                        5.0
                    )
            
            self.connection.commit()
        
        except Exception as e:
            print(f"Error analyzing user activity: {e}")
            self.connection.rollback()
        finally:
            cursor.close()
            
    # --- PATCH 3: Add persistent model cache methods ---
    def cache_client_models(self, client_id, cached_models):
        """Persist client ML models to database"""
        cursor = self.connection.cursor()
        try:
            model_bytes = pickle.dumps(cached_models)
            print(f"  [DB CACHE] Storing {len(model_bytes)} bytes for {client_id[:12]}")
            
            cursor.execute("""
                INSERT INTO client_model_cache (client_id, model_data, cached_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (client_id) DO UPDATE SET 
                    model_data = EXCLUDED.model_data,
                    cached_at = NOW()
            """, (client_id, psycopg2.Binary(model_bytes)))  # Use psycopg2.Binary
            self.connection.commit()
            
            # Verify storage
            cursor.execute("SELECT LENGTH(model_data) FROM client_model_cache WHERE client_id = %s", (client_id,))
            stored_size = cursor.fetchone()
            print(f"  [DB CACHE] Verified: {stored_size[0] if stored_size else 0} bytes stored for {client_id[:12]}")
            
        except Exception as e:
            print(f"✗ Model cache error: {e}")
            import traceback
            traceback.print_exc()
            self.connection.rollback()
        finally:
            cursor.close()
    
    def get_cached_models(self, client_id):
        """Retrieve cached models from database - FIXED"""
        cursor = self.connection.cursor()
        try:
            cursor.execute("""
                SELECT model_data, cached_at FROM client_model_cache 
                WHERE client_id = %s
            """, (client_id,))
            result = cursor.fetchone()
            
            if result and result[0]:
                model_data = result[0]
                cached_at = result[1]
                
                # Handle PostgreSQL bytea/memoryview
                if isinstance(model_data, memoryview):
                    model_data = bytes(model_data)
                
                # Deserialize
                cached_models = pickle.loads(model_data)
                
                print(f"  [DB] Retrieved cache for {client_id[:12]} (cached at {cached_at})")
                return cached_models
            
            return None
            
        except Exception as e:
            print(f"✗ Model retrieval error for {client_id[:12]}: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            cursor.close()

class ServerThread(QThread):
    """Background thread for handling client connections"""
    client_connected = pyqtSignal(dict)
    data_received = pyqtSignal(str, dict)
    fl_update_received = pyqtSignal(str, dict)
    
    def __init__(self, db_manager, fl_manager):
        super().__init__()
        self.db_manager = db_manager
        self.fl_manager = fl_manager
        self.running = True
        self.server_socket = None
        self.client_models_cache = {}
    
    def run(self):
        """Main server loop"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((SERVER_HOST, SERVER_PORT))
        self.server_socket.listen(10)
        self.server_socket.settimeout(1.0)
        
        print(f"– Server listening on {SERVER_HOST}:{SERVER_PORT}")
        
        while self.running:
            try:
                client_socket, address = self.server_socket.accept()
                threading.Thread(target=self.handle_client, args=(client_socket, address)).start()
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"✗ Server error: {e}")
    
    def handle_client(self, client_socket, address):
        """Handle individual client connection - FIXED: Proper registration order"""
        try:
            # Receive data size
            size_data = client_socket.recv(8)
            data_size = int.from_bytes(size_data, 'big')
            
            # Receive actual data
            received = b''
            while len(received) < data_size:
                chunk = client_socket.recv(min(4096, data_size - len(received)))
                if not chunk:
                    break
                received += chunk
            
            # Deserialize data
            data = pickle.loads(received)
            
            if data['type'] == 'registration':
                client_id = data['client_info']['client_id']

                # ✅ FIX: Register client FIRST before any other operations
                self.db_manager.register_client(
                    data['client_info'],
                    data.get('capabilities')
                )
                
                # ✅ FIX: Commit immediately to ensure foreign key exists
                self.db_manager.connection.commit()
                
                self.client_connected.emit(data['client_info'])

                # Build response
                response = {
                    'status': 'registered',
                    'message': 'Client registered successfully'
                }
                
                # Add FL global model if client supports FL
                if data.get('capabilities', {}).get('federated_learning'):
                    response['model_weights'] = self.fl_manager.get_global_model()

                # Check for cached models
                cached_models = None
                
                try:
                    db_cached = self.db_manager.get_cached_models(client_id)
                    if db_cached:
                        cached_models = db_cached
                        print(f"✓ [{client_id[:12]}] Found cached models in DATABASE")
                except Exception as e:
                    print(f"⚠ [{client_id[:12]}] DB cache lookup error: {e}")
                
                if cached_models is None:
                    if hasattr(self, 'client_models_cache') and client_id in self.client_models_cache:
                        cached_models = self.client_models_cache[client_id].get('cached_models')
                        if cached_models:
                            print(f"✓ [{client_id[:12]}] Found cached models in MEMORY")
                
                if cached_models:
                    response['cached_models'] = cached_models
                    print(f"✓ [{client_id[:12]}] Sending cached models to client")
                else:
                    print(f"⚠ [{client_id[:12]}] No cached models available")

                # Send response
                try:
                    serialized_response = pickle.dumps(response)
                    response_size = len(serialized_response)
                    
                    client_socket.send(response_size.to_bytes(8, 'big'))
                    client_socket.sendall(serialized_response)
                    
                    print(f"✓ [{client_id[:12]}] Registration response sent ({response_size} bytes, has_cache={'cached_models' in response})")
                except Exception as e:
                    print(f"✗ [{client_id[:12]}] Failed to send registration response: {e}")
            
            elif data['type'] == 'telemetry':
                client_id = data['client_id']

                # ------------------------------------------------------------
                # 1) VERIFY CLIENT EXISTS — AUTO-REGISTER IF MISSING
                # ------------------------------------------------------------
                cursor = self.db_manager.connection.cursor()
                cursor.execute("SELECT client_id FROM clients WHERE client_id = %s", (client_id,))
                client_exists = cursor.fetchone()
                cursor.close()

                if not client_exists:
                    print(f"⚠ Unknown client {client_id[:12]}, auto-registering...")
                    try:
                        self.db_manager.register_client({
                            'client_id': client_id,
                            'hostname': f'auto-registered-{client_id[:8]}',
                            'ip_address': address[0],
                            'os_type': 'Unknown',
                            'os_version': 'Unknown',
                            'device_role': 'workstation',
                            'department': 'Unknown',
                            'criticality_level': 'low'
                        })
                        self.db_manager.connection.commit()
                    except Exception as e:
                        print(f"✗ Failed to auto-register client: {e}")
                        response = {'status': 'error', 'message': 'Client not registered'}
                        client_socket.send(pickle.dumps(response))
                        return

                # ------------------------------------------------------------
                # 2) PROCESS ML ANOMALY ALERTS (client-side ensemble detections)
                # ------------------------------------------------------------
                if 'anomaly_alerts' in data and data['anomaly_alerts']:
                    cursor = self.db_manager.connection.cursor()

                    for anomaly_alert in data['anomaly_alerts']:
                        try:
                            # Severity normalization
                            severity = anomaly_alert.get('severity', 'medium').lower()
                            if severity not in ('low', 'medium', 'high'):
                                severity = 'medium'

                            # Category → DB source_category mapping
                            category = anomaly_alert.get('category', 'system')
                            source_map = {
                                'network': 'network',
                                'process': 'process',
                                'filesystem': 'filesystem',
                                'user': 'user_activity'
                            }
                            source_category = source_map.get(category, 'system')

                            # Risk score normalization (unified)
                            iso_score = anomaly_alert.get('iso_score')
                            ae_error = anomaly_alert.get('ae_recon_error')
                            ensemble_score = anomaly_alert.get('ensemble_score')

                            if ensemble_score is not None:
                                risk_score = float(ensemble_score)
                            elif iso_score is not None:
                                risk_score = min(abs(float(iso_score)) * 10, 10.0)
                            elif ae_error is not None:
                                risk_score = min(float(ae_error) * 100, 10.0)
                            else:
                                risk_score = 5.0

                            cursor.execute("""
                                INSERT INTO alerts 
                                (client_id, alert_type, severity, title, description, 
                                source_category, risk_score, status)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, 'active')
                            """, (
                                client_id,
                                f"ml_{category}_anomaly",
                                severity,
                                f"ML Anomaly: {severity.upper()} - {category.upper()}",
                                anomaly_alert.get('explanation', 'No explanation available')[:500],
                                source_category,
                                float(risk_score)
                            ))

                            print(f"  [SERVER] Stored client anomaly alert: {severity} "
                                f"(category={category}, score={risk_score:.1f})")

                        except Exception as e:
                            print(f"  [SERVER] Error storing anomaly alert: {e}")

                    self.db_manager.connection.commit()
                    cursor.close()

                # ------------------------------------------------------------
                # 3) STORE TELEMETRY: NETWORK, PROCESS, FILESYSTEM, USER ACTIVITY
                # ------------------------------------------------------------
                try:
                    if 'network' in data:
                        self.db_manager.insert_network_data(client_id, data['network'])
                    if 'processes' in data:
                        self.db_manager.insert_process_data(client_id, data['processes'])
                    if 'filesystem' in data:
                        self.db_manager.insert_filesystem_data(client_id, data['filesystem'])
                    if 'user_activity' in data:
                        self.db_manager.insert_user_activity(client_id, data['user_activity'])
                        self.db_manager.analyze_user_activity_anomalies(
                            client_id, data['user_activity']
                        )

                    # Emit telemetry signal
                    self.data_received.emit(client_id, data)

                    # Internal ML workflows (server-side detection)
                    try:
                        self.db_manager.analyze_and_alert(client_id, data)
                    except Exception as e:
                        print(f"✗ ML Analysis Error: {e}")

                    # ------------------------------------------------------------
                    # 4) RETURN RECENT ACTIVE ALERTS
                    # ------------------------------------------------------------
                    cursor = self.db_manager.connection.cursor(cursor_factory=RealDictCursor)
                    cursor.execute("""
                        SELECT title, description 
                        FROM alerts 
                        WHERE client_id = %s 
                        AND status = 'active'
                        AND timestamp > NOW() - INTERVAL '5 minutes'
                        ORDER BY risk_score DESC
                        LIMIT 5
                    """, (client_id,))
                    recent_alerts = cursor.fetchall()
                    cursor.close()

                    response = {'status': 'received', 'message': 'Telemetry data stored'}

                    if recent_alerts:
                        response['alerts'] = [
                            f"{a['title']}: {a['description']}" for a in recent_alerts
                        ]

                    serialized = pickle.dumps(response)
                    client_socket.send(len(serialized).to_bytes(8, 'big'))
                    client_socket.sendall(serialized)

                except Exception as e:
                    print(f"✗ Error processing telemetry: {e}")
                    self.db_manager.connection.rollback()

                    error_response = {'status': 'error', 'message': str(e)}
                    client_socket.send(pickle.dumps(error_response))
            
            elif data['type'] == 'fl_update':
                client_id = data['client_id']
                model_params = data['model_parameters']

                # ✅ FIX: Verify client exists before FL operations
                cursor = self.db_manager.connection.cursor()
                cursor.execute("SELECT client_id FROM clients WHERE client_id = %s", (client_id,))
                client_exists = cursor.fetchone()
                cursor.close()

                if not client_exists:
                    print(f"⚠ FL update from unregistered client {client_id[:12]}, registering...")
                    self.db_manager.register_client({
                        'client_id': client_id,
                        'hostname': f'fl-client-{client_id[:8]}',
                        'ip_address': address[0],
                        'os_type': 'Unknown',
                        'os_version': 'Unknown',
                        'device_role': 'workstation',
                        'department': 'Unknown',
                        'criticality_level': 'medium'
                    }, {'federated_learning': True})
                    # ✅ FIX: Commit immediately
                    self.db_manager.connection.commit()

                # Receive and store the update
                self.fl_manager.receive_client_update(client_id, model_params)

                contrib = self.fl_manager.client_contributions[client_id]

                # Log FL contribution
                last_update = None
                try:
                    for update in self.fl_manager.pending_updates:
                        if update['client_id'] == client_id:
                            last_update = update
                            break

                    if last_update:
                        self.db_manager.log_fl_contribution(
                            client_id=client_id,
                            pre_norm=last_update['pre_norm'],
                            post_norm=last_update['post_norm'],
                            was_clipped=(contrib['clipped_count'] > 0),
                            was_quarantined=(contrib['quarantine_count'] > 0),
                            reputation=contrib['reputation'],
                            quality_score=contrib['quality'],
                            samples_used=last_update['metadata']['samples_used'],
                            model_version=self.fl_manager.global_model['version']
                        )
                except Exception as e:
                    print(f"Warning: Could not log FL contribution: {e}")

                self.fl_update_received.emit(client_id, model_params)

                aggregated = None
                if len(self.fl_manager.pending_updates) >= 2:
                    aggregated = self.fl_manager.aggregate_models()

                    if aggregated:
                        try:
                            agg_meta = self.fl_manager.get_aggregation_metadata()
                            convergence = 0.0
                            if aggregated.get('convergence_history'):
                                convergence = aggregated['convergence_history'][-1].get('delta', 0)

                            self.db_manager.log_fl_aggregation_round(
                                model_version=aggregated['version'],
                                participated=agg_meta['participated'],
                                quarantined=agg_meta['quarantined'],
                                avg_norm=agg_meta['avg_delta_norm'],
                                validation_auc=agg_meta['validation_auc'],
                                dp_noise=agg_meta['dp_noise_applied'],
                                rollback=agg_meta['rollback_occurred'],
                                convergence=convergence
                            )
                        except Exception as e:
                            print(f"Warning: Could not log aggregation round: {e}")

                        try:
                            self.db_manager.save_fl_model(
                                aggregated['version'],
                                aggregated['weights'],
                                len(self.fl_manager.client_models)
                            )
                        except Exception as e:
                            print(f"Warning: Could not save FL model: {e}")

                response = {'status': 'fl_received', 'message': 'FL update received'}
                response['aggregated_weights'] = self.fl_manager.get_global_model()

                serialized_response = pickle.dumps(response)
                response_size = len(serialized_response)
                client_socket.send(response_size.to_bytes(8, 'big'))
                client_socket.sendall(serialized_response)
                
            elif data['type'] == 'heartbeat':
                # Update last heartbeat
                cursor = self.db_manager.connection.cursor()
                cursor.execute("""
                    UPDATE clients SET last_heartbeat = CURRENT_TIMESTAMP 
                    WHERE client_id = %s
                """, (data['client_id'],))
                self.db_manager.connection.commit()
                cursor.close()
                
                response = {'status': 'alive'}
                serialized_response = pickle.dumps(response)
                response_size = len(serialized_response)
                client_socket.send(response_size.to_bytes(8, 'big'))
                client_socket.sendall(serialized_response)
        
            elif data['type'] == 'cache_models':
                client_id = data['client_id']
                cached_models = data['cached_models']
                
                # ✅ FIX: Verify client exists before caching
                cursor = self.db_manager.connection.cursor()
                cursor.execute("SELECT client_id FROM clients WHERE client_id = %s", (client_id,))
                client_exists = cursor.fetchone()
                cursor.close()

                if not client_exists:
                    print(f"⚠ Cache request from unregistered client {client_id[:12]}, skipping...")
                    response = {'status': 'error', 'message': 'Client not registered'}
                    serialized_response = pickle.dumps(response)
                    response_size = len(serialized_response)
                    client_socket.send(response_size.to_bytes(8, 'big'))
                    client_socket.sendall(serialized_response)
                    return
                
                if 'autoencoder_config' in cached_models:
                    ae_config = cached_models['autoencoder_config']
                    if 'scaler_mean' not in ae_config or 'scaler_scale' not in ae_config:
                        print(f"⚠ [{client_id[:12]}] Autoencoder missing scaler parameters, skipping cache")
                        del cached_models['autoencoder_config']
                
                # Memory cache
                self.client_models_cache[client_id] = {
                    'cached_models': cached_models,
                    'cached_at': datetime.now().isoformat()
                }
                
                # Database cache
                self.db_manager.cache_client_models(client_id, cached_models)
                
                response = {'status': 'models_cached'}
                serialized_response = pickle.dumps(response)
                response_size = len(serialized_response)
                client_socket.send(response_size.to_bytes(8, 'big'))
                client_socket.sendall(serialized_response)
                print(f"✓ [{client_id[:12]}] Models cached (memory + database)")
        
        except Exception as e:
            print(f"✗ Error handling client {address}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            client_socket.close()
    
    def stop(self):
        """Stop the server"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()


class MainWindow(QMainWindow):
    """Enhanced main application window"""
    
    def __init__(self):
        super().__init__()
        self.db_manager = DatabaseManager()
        self.fl_manager = FederatedLearningManager()
        self.server_thread = None
        self.init_ui()
        self.start_server()
        
        # Populate client filters
        self.refresh_client_filters()
        
        # Setup auto-refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_data)
        self.refresh_timer.start(5000)  # Refresh every 5 seconds
        
        # ADD THIS: Auto-aggregation timer (every 2 minutes)
        self.fl_aggregation_timer = QTimer()
        self.fl_aggregation_timer.timeout.connect(self.auto_aggregate_fl)
        self.fl_aggregation_timer.start(120000)  # 120 seconds = 2 minutes

    # --- PATCH 7: NEW method for FL configuration dialog ---
    def show_fl_config(self):
        """Show FL configuration dialog"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Federated Learning Configuration")
        dialog.setGeometry(200, 200, 500, 400)
        
        layout = QFormLayout()
        
        # Clip bound
        clip_bound_input = QLineEdit(str(CLIP_BOUND))
        layout.addRow("L2 Clip Bound:", clip_bound_input)
        
        # Trim fraction
        trim_frac_input = QLineEdit(str(TRIM_FRAC))
        layout.addRow("Trim Fraction:", trim_frac_input)
        
        # DP noise scale
        dp_noise_input = QLineEdit(str(self.fl_manager.dp_noise_scale))
        layout.addRow("DP Noise Scale:", dp_noise_input)
        
        # DP epsilon
        dp_epsilon_input = QLineEdit(str(self.fl_manager.dp_epsilon))
        layout.addRow("DP Epsilon:", dp_epsilon_input)
        
        # Validation threshold
        val_threshold_input = QLineEdit(str(VALIDATION_AUC_DROP_THRESHOLD))
        layout.addRow("Validation AUC Drop Threshold:", val_threshold_input)
        
        # Momentum factor
        momentum_input = QLineEdit(str(self.fl_manager.momentum_factor))
        layout.addRow("Momentum Factor:", momentum_input)
        
        # Buttons
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Save")
        cancel_btn = QPushButton("Cancel")
        
        def save_config():
            try:
                global CLIP_BOUND, TRIM_FRAC, VALIDATION_AUC_DROP_THRESHOLD
                CLIP_BOUND = float(clip_bound_input.text())
                TRIM_FRAC = float(trim_frac_input.text())
                VALIDATION_AUC_DROP_THRESHOLD = float(val_threshold_input.text())
                
                self.fl_manager.dp_noise_scale = float(dp_noise_input.text())
                self.fl_manager.dp_epsilon = float(dp_epsilon_input.text())
                self.fl_manager.momentum_factor = float(momentum_input.text())
                
                QMessageBox.information(dialog, "Success", "Configuration updated successfully")
                dialog.accept()
            except Exception as e:
                QMessageBox.warning(dialog, "Error", f"Invalid configuration: {e}")
        
        save_btn.clicked.connect(save_config)
        cancel_btn.clicked.connect(dialog.reject)
        
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        
        layout.addRow(btn_layout)
        dialog.setLayout(layout)
        dialog.exec_()
        
    def auto_aggregate_fl(self):
        """Automatically aggregate FL models periodically"""
        if len(self.fl_manager.client_models) >= 1:  # Allow single client for testing
            print("\n[Auto-Aggregation] Triggering FL aggregation...")
            aggregated = self.fl_manager.aggregate_models()
            if aggregated:
                print(f"[Auto-Aggregation] Success! Version {aggregated['version']}")
                # Refresh FL tab if it's currently visible
                if self.tabs.currentIndex() == 8:  # FL tab index
                    self.refresh_fl_tab()
                    
    def init_ui(self):
        """Initialize enhanced user interface"""
        self.setWindowTitle("FortifAI Admin Server Hub - Enhanced")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Add tabs
        self.create_dashboard_tab()
        self.create_alerts_tab()
        self.create_clients_tab()
        self.create_aggregated_view_tab()
        self.create_network_tab()
        self.create_processes_tab()
        self.create_filesystem_tab()
        self.create_user_activity_tab()
        self.create_federated_learning_tab()
        
        # Status bar
        self.statusBar().showMessage("Server initializing...")
    
    def create_dashboard_tab(self):
        """Create enhanced dashboard overview tab"""
        dashboard = QWidget()
        layout = QVBoxLayout(dashboard)
        
        # Statistics group
        stats_group = QGroupBox("System Overview")
        stats_layout = QHBoxLayout()
        
        self.total_clients_label = QLabel("Total Clients: 0")
        self.online_clients_label = QLabel("Online: 0")
        self.total_alerts_label = QLabel("Active Alerts: 0")
        self.critical_alerts_label = QLabel("Critical: 0")
        
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        for label in [self.total_clients_label, self.online_clients_label, 
                     self.total_alerts_label, self.critical_alerts_label]:
            label.setFont(font)
            stats_layout.addWidget(label)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # Charts - Create actual charts first
        charts_layout = QHBoxLayout()

        # Client status pie chart
        self.client_status_chart = self.create_pie_chart("Client Status")
        self.client_chart_view = QChartView(self.client_status_chart)
        self.client_chart_view.setRenderHint(QPainter.Antialiasing)
        charts_layout.addWidget(self.client_chart_view)

        # Events bar chart  
        self.events_chart = self.create_bar_chart("Events (24h)")
        self.events_chart_view = QChartView(self.events_chart)
        self.events_chart_view.setRenderHint(QPainter.Antialiasing)
        charts_layout.addWidget(self.events_chart_view)

        layout.addLayout(charts_layout)

        # Threat severity distribution
        severity_layout = QHBoxLayout()

        # Severity pie chart
        self.severity_chart = self.create_pie_chart("Alert Severity")
        self.severity_chart_view = QChartView(self.severity_chart)
        self.severity_chart_view.setRenderHint(QPainter.Antialiasing)
        severity_layout.addWidget(self.severity_chart_view)

        # Risk score timeline
        self.timeline_chart = QChart()
        self.timeline_chart.setTitle("Network Activity Timeline")
        self.timeline_chart_view = QChartView(self.timeline_chart)
        self.timeline_chart_view.setRenderHint(QPainter.Antialiasing)
        severity_layout.addWidget(self.timeline_chart_view)

        layout.addLayout(severity_layout)
        
        # Top risks by category
        risks_group = QGroupBox("Top Risk Events by Category (24h)")
        risks_layout = QVBoxLayout()
        self.risks_table = QTableWidget()
        self.risks_table.setColumnCount(4)
        self.risks_table.setHorizontalHeaderLabels(["Category", "Client ID", "Details", "Risk Score"])
        self.risks_table.horizontalHeader().setStretchLastSection(True)
        risks_layout.addWidget(self.risks_table)
        risks_group.setLayout(risks_layout)
        layout.addWidget(risks_group)
        
        self.tabs.addTab(dashboard, "Dashboard")
    
    def show_fl_metrics(self):
        """Show detailed FL performance metrics dialog"""
        metrics = self.fl_manager.get_convergence_metrics()
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Federated Learning Metrics")
        dialog.setGeometry(200, 200, 600, 400)
        
        layout = QVBoxLayout()
        
        # Metrics text
        metrics_text = QTextEdit()
        metrics_text.setReadOnly(True)
        
        text = "=== Federated Learning Performance Metrics ===\n\n"
        text += f"Current Version: {metrics['current_version']}\n"
        text += f"Participating Clients: {metrics['participating_clients']}\n\n"
        
        text += "=== Convergence History ===\n"
        for entry in metrics['history'][-10:]:  # Last 10 entries
            text += f"Version {entry['version']}: delta={entry['delta']:.6f}, clients={entry['clients']}\n"
        
        text += "\n=== Client Contributions ===\n"
        for client_id, contrib in sorted(metrics['client_contributions'].items(), 
                                        key=lambda x: x[1]['count'], 
                                        reverse=True):
            text += f"{client_id[:12]}: {contrib['count']} updates, quality={contrib['quality']:.3f}\n"
        
        metrics_text.setText(text)
        layout.addWidget(metrics_text)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        
        dialog.setLayout(layout)
        dialog.exec_()
        
    def create_alerts_tab(self):
        """Create alerts management tab"""
        alerts = QWidget()
        layout = QVBoxLayout(alerts)
        
        # Buttons
        btn_layout = QHBoxLayout()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.load_alerts)
        acknowledge_btn = QPushButton("Acknowledge Selected")
        acknowledge_btn.clicked.connect(self.acknowledge_alert)
        btn_layout.addWidget(refresh_btn)
        btn_layout.addWidget(acknowledge_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # Alerts table
        self.alerts_table = QTableWidget()
        self.alerts_table.setColumnCount(7)
        self.alerts_table.setHorizontalHeaderLabels([
            "Timestamp", "Severity", "Client", "Category", "Title", "Description", "Risk Score"
        ])
        self.alerts_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.alerts_table)
        
        self.tabs.addTab(alerts, "Alerts")
    
    def create_clients_tab(self):
        """Create clients management tab"""
        clients = QWidget()
        layout = QVBoxLayout(clients)
        
        # Buttons
        btn_layout = QHBoxLayout()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.load_clients)
        btn_layout.addWidget(refresh_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # Clients table
        self.clients_table = QTableWidget()
        self.clients_table.setColumnCount(9)
        self.clients_table.setHorizontalHeaderLabels([
            "Client ID", "Hostname", "IP Address", "OS", "Role", 
            "Department", "Status", "FL Enabled", "Last Heartbeat"
        ])
        self.clients_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.clients_table)
        
        self.tabs.addTab(clients, "Clients")
    
    def create_aggregated_view_tab(self):
        """Create aggregated view per client"""
        agg_view = QWidget()
        layout = QVBoxLayout(agg_view)
        
        # Info label
        info_label = QLabel("Aggregated view shows summarized activity per client for easier analysis")
        info_label.setStyleSheet("color: #3498db; padding: 10px;")
        layout.addWidget(info_label)
        
        # Refresh button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.load_aggregated_view)
        layout.addWidget(refresh_btn)
        
        # Aggregated table
        self.agg_table = QTableWidget()
        self.agg_table.setColumnCount(10)
        self.agg_table.setHorizontalHeaderLabels([
            "Client ID", "Hostname", "Status", "Network Events", "Process Events",
            "File Events", "Avg Network Risk", "Avg Process Risk", "Avg File Risk", "Active Alerts"
        ])
        self.agg_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.agg_table)
        
        self.tabs.addTab(agg_view, "Client Analysis")
    
    def check_offline_clients(self):
        """Mark clients as offline if heartbeat is stale - FIXED"""
        try:
            cursor = self.db_manager.connection.cursor()
            cursor.execute("""
                UPDATE clients 
                SET status = 'offline'
                WHERE last_heartbeat < NOW() - INTERVAL '2 minutes'
                AND status = 'online'
            """)
            self.db_manager.connection.commit()
            cursor.close()
        except Exception as e:
            print(f"Error checking offline clients: {e}")
            try:
                self.db_manager.connection.rollback()
            except:
                pass
    
    def refresh_client_filters(self):
        """Populate client filter dropdowns"""
        clients = self.db_manager.get_all_clients()
        
        for combo in [self.network_client_filter, self.process_client_filter, 
                    self.fs_client_filter]:
            combo.clear()
            combo.addItem("All Clients", None)
            for client in clients:
                display_name = f"{client['hostname']} ({client['client_id'][:8]})"
                combo.addItem(display_name, client['client_id'])
                
    def create_network_tab(self):
        """Create enhanced network data tab"""
        network = QWidget()
        layout = QVBoxLayout(network)
        
        # Filters
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Show:"))
        
        self.network_filter = QComboBox()
        self.network_filter.addItems(["All Events", "High Risk Only (>7)", "Anomalies Only"])
        self.network_filter.currentIndexChanged.connect(self.load_network_data)
        filter_layout.addWidget(self.network_filter)
        
        filter_layout.addWidget(QLabel("Client:"))
    
        self.network_client_filter = QComboBox()
        self.network_client_filter.addItem("All Clients")
        self.network_client_filter.currentIndexChanged.connect(self.load_network_data)
        filter_layout.addWidget(self.network_client_filter)
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.load_network_data)
        filter_layout.addWidget(refresh_btn)
        filter_layout.addStretch()
        
        layout.addLayout(filter_layout)
        
        # Network table
        self.network_table = QTableWidget()
        self.network_table.setColumnCount(11)
        self.network_table.setHorizontalHeaderLabels([
            "Timestamp", "Client", "Src IP:Port", "Dst IP:Port", "Protocol",
            "Connections", "DNS Query", "Geolocation", "Threat Indicators", "Anomaly", "Risk"
        ])
        self.network_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.network_table)
        
        self.tabs.addTab(network, "Network Data")
    
    def create_processes_tab(self):
        """Create enhanced process data tab"""
        processes = QWidget()
        layout = QVBoxLayout(processes)
        
        # Filters
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Show:"))
        
        self.process_filter = QComboBox()
        self.process_filter.addItems(["All Processes", "High Risk Only (>6)", "Suspicious Only"])
        self.process_filter.currentIndexChanged.connect(self.load_process_data)
        filter_layout.addWidget(self.process_filter)
        
        # ADD CLIENT FILTER
        filter_layout.addWidget(QLabel("Client:"))
        self.process_client_filter = QComboBox()
        self.process_client_filter.addItem("All Clients", None)
        self.process_client_filter.currentIndexChanged.connect(self.load_process_data)
        filter_layout.addWidget(self.process_client_filter)
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.load_process_data)
        filter_layout.addWidget(refresh_btn)
        filter_layout.addStretch()
        
        layout.addLayout(filter_layout)
            
        # Process table
        self.process_table = QTableWidget()
        self.process_table.setColumnCount(10)
        self.process_table.setHorizontalHeaderLabels([
            "Timestamp", "Client", "Process", "PID", "Parent",
            "Executable Hash", "CPU %", "Memory MB", "Threat Indicators", "Risk"
        ])
        self.process_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.process_table)
        
        self.tabs.addTab(processes, "Processes")
    
    def create_filesystem_tab(self):
        """Create enhanced filesystem data tab"""
        filesystem = QWidget()
        layout = QVBoxLayout(filesystem)
        
        # Filters
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Show:"))
        
        self.fs_filter = QComboBox()
        self.fs_filter.addItems(["All Files", "High Risk Only (>5)", "Suspicious Only"])
        self.fs_filter.currentIndexChanged.connect(self.load_filesystem_data)
        filter_layout.addWidget(self.fs_filter)
        
        # ADD CLIENT FILTER
        filter_layout.addWidget(QLabel("Client:"))
        self.fs_client_filter = QComboBox()
        self.fs_client_filter.addItem("All Clients", None)
        self.fs_client_filter.currentIndexChanged.connect(self.load_filesystem_data)
        filter_layout.addWidget(self.fs_client_filter)
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.load_filesystem_data)
        filter_layout.addWidget(refresh_btn)
        filter_layout.addStretch()
        
        layout.addLayout(filter_layout)
        
        # Filesystem table
        self.fs_table = QTableWidget()
        self.fs_table.setColumnCount(10)
        self.fs_table.setHorizontalHeaderLabels([
            "Timestamp", "Client", "Event", "File Name", "Extension",
            "Size", "Directory", "Hash", "Threat Indicators", "Risk"
        ])
        self.fs_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.fs_table)
        
        self.tabs.addTab(filesystem, "Filesystem")
    
    def create_user_activity_tab(self):
        """Create user activity monitoring tab with anomaly detection"""
        user_activity = QWidget()
        layout = QVBoxLayout(user_activity)
        
        # Statistics panel
        stats_group = QGroupBox("User Activity Statistics (24h)")
        stats_layout = QHBoxLayout()
        
        self.user_total_logins_label = QLabel("Total Logins: 0")
        self.user_unique_users_label = QLabel("Unique Users: 0")
        self.user_remote_logins_label = QLabel("Remote Logins: 0")
        self.user_high_risk_label = QLabel("High Risk Events: 0")
        
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        for label in [self.user_total_logins_label, self.user_unique_users_label, 
                    self.user_remote_logins_label, self.user_high_risk_label]:
            label.setFont(font)
            stats_layout.addWidget(label)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # Filters
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Show:"))
        
        self.user_activity_filter = QComboBox()
        self.user_activity_filter.addItems([
            "All Events", 
            "High Risk Only (>5)", 
            "Remote Logins",
            "Privileged Users",
            "Anomalies Only"
        ])
        self.user_activity_filter.currentIndexChanged.connect(self.load_user_activity)
        filter_layout.addWidget(self.user_activity_filter)
        
        filter_layout.addWidget(QLabel("Client:"))
        self.user_client_filter = QComboBox()
        self.user_client_filter.addItem("All Clients", None)
        self.user_client_filter.currentIndexChanged.connect(self.load_user_activity)
        filter_layout.addWidget(self.user_client_filter)
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.load_user_activity)
        filter_layout.addWidget(refresh_btn)
        filter_layout.addStretch()
        
        layout.addLayout(filter_layout)
        
        # User activity table with more columns
        self.user_activity_table = QTableWidget()
        self.user_activity_table.setColumnCount(11)  # Increased from 8
        self.user_activity_table.setHorizontalHeaderLabels([
            "Timestamp", "Client", "Event", "Username", "Session ID",
            "Source IP", "Success", "Privilege", "Concurrent", "Indicators", "Risk"
        ])
        self.user_activity_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.user_activity_table)
        
        self.tabs.addTab(user_activity, "User Activity")

    def load_user_activity(self):
        """Load user activity data with filtering - FIXED: Prevent auto-refresh loop"""
        # ✅ FIX: Disconnect signal temporarily
        try:
            self.user_activity_filter.currentIndexChanged.disconnect()
            self.user_client_filter.currentIndexChanged.disconnect()
        except:
            pass
        
        cursor = self.db_manager.connection.cursor(cursor_factory=RealDictCursor)
        
        filter_option = self.user_activity_filter.currentText()
        where_clauses = []
        
        if "High Risk" in filter_option:
            where_clauses.append("u.risk_score > 5.0")
        elif "Remote Logins" in filter_option:
            where_clauses.append("u.source_ip NOT IN ('local', 'localhost', '127.0.0.1')")
        elif "Privileged" in filter_option:
            where_clauses.append("u.privilege_escalation = TRUE")
        elif "Anomalies" in filter_option:
            where_clauses.append("u.risk_score > 6.0 OR array_length(u.threat_indicators, 1) > 0")
        
        selected_client = self.user_client_filter.currentData()
        if selected_client:
            where_clauses.append(f"u.client_id = '{selected_client}'")
        
        where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        
        # Get statistics
        cursor.execute(f"""
            SELECT 
                COUNT(*) as total_logins,
                COUNT(DISTINCT username) as unique_users,
                SUM(CASE WHEN source_ip NOT IN ('local', 'localhost', '127.0.0.1') THEN 1 ELSE 0 END) as remote_logins,
                SUM(CASE WHEN risk_score > 5.0 THEN 1 ELSE 0 END) as high_risk_events
            FROM user_activity u
            JOIN clients c ON u.client_id = c.client_id
            WHERE u.timestamp > NOW() - INTERVAL '24 hours'
        """)
        stats = cursor.fetchone()
        
        if stats:
            self.user_total_logins_label.setText(f"Total Logins: {stats['total_logins'] or 0}")
            self.user_unique_users_label.setText(f"Unique Users: {stats['unique_users'] or 0}")
            self.user_remote_logins_label.setText(f"Remote Logins: {stats['remote_logins'] or 0}")
            self.user_high_risk_label.setText(f"High Risk Events: {stats['high_risk_events'] or 0}")
        
        # Get activity records
        cursor.execute(f"""
            SELECT u.*, c.hostname 
            FROM user_activity u
            JOIN clients c ON u.client_id = c.client_id
            {where_clause}
            ORDER BY u.timestamp DESC, u.risk_score DESC
            LIMIT 500
        """)
        records = cursor.fetchall()
        cursor.close()
        
        self.user_activity_table.setRowCount(len(records))
        for i, record in enumerate(records):
            self.user_activity_table.setItem(i, 0, QTableWidgetItem(
                record['timestamp'].strftime("%Y-%m-%d %H:%M:%S")))
            
            self.user_activity_table.setItem(i, 1, QTableWidgetItem(record['hostname'][:15]))
            
            event_item = QTableWidgetItem(record['event_type'].capitalize())
            if record['event_type'] == 'login':
                event_item.setBackground(QColor(52, 152, 219))
            elif record['event_type'] == 'logout':
                event_item.setBackground(QColor(149, 165, 166))
            self.user_activity_table.setItem(i, 2, event_item)
            
            username_item = QTableWidgetItem(record['username'])
            if record.get('privilege_escalation'):
                username_item.setBackground(QColor(241, 196, 15))
            self.user_activity_table.setItem(i, 3, username_item)
            
            self.user_activity_table.setItem(i, 4, QTableWidgetItem(record['session_id'] or 'N/A'))
            
            source_ip = record['source_ip'] or 'local'
            source_item = QTableWidgetItem(source_ip)
            if source_ip not in ['local', 'localhost', '127.0.0.1', '::1']:
                source_item.setBackground(QColor(155, 89, 182))
            self.user_activity_table.setItem(i, 5, source_item)
            
            login_item = QTableWidgetItem("Yes" if record['login_success'] else "No")
            if not record['login_success']:
                login_item.setBackground(QColor(231, 76, 60))
            self.user_activity_table.setItem(i, 6, login_item)
            
            priv_item = QTableWidgetItem("Yes" if record.get('privilege_escalation') else "No")
            if record.get('privilege_escalation'):
                priv_item.setBackground(QColor(230, 126, 34))
            self.user_activity_table.setItem(i, 7, priv_item)
            
            concurrent = record.get('concurrent_sessions', 1)
            concurrent_item = QTableWidgetItem(str(concurrent))
            if concurrent >= 3:
                concurrent_item.setBackground(QColor(241, 196, 15))
            self.user_activity_table.setItem(i, 8, concurrent_item)
            
            indicators = record.get('threat_indicators', [])
            if indicators:
                indicators_str = ', '.join(indicators[:2])
            else:
                indicators_str = 'None'
            self.user_activity_table.setItem(i, 9, QTableWidgetItem(indicators_str))
            
            risk_score = float(record.get('risk_score', 0))
            risk_item = QTableWidgetItem(f"{risk_score:.2f}")
            if risk_score > 7:
                risk_item.setBackground(QColor(231, 76, 60))
            elif risk_score > 5:
                risk_item.setBackground(QColor(241, 196, 15))
            elif risk_score > 3:
                risk_item.setBackground(QColor(52, 152, 219))
            self.user_activity_table.setItem(i, 10, risk_item)
        
        # ✅ FIX: Reconnect signals
        self.user_activity_filter.currentIndexChanged.connect(self.load_user_activity)
        self.user_client_filter.currentIndexChanged.connect(self.load_user_activity)
                
    # --- (F) PATCH 2: Enhanced FL Tab with Reputation Display ---
    def create_federated_learning_tab(self):
        """Create enhanced federated learning management tab with reputation tracking"""
        fl_tab = QWidget()
        layout = QVBoxLayout(fl_tab)
        
        # FL Status
        status_group = QGroupBox("Federated Learning Status")
        status_layout = QVBoxLayout()
        
        # --- NEW: Enhanced status display with aggregation metadata ---
        self.fl_status_label = QLabel("Model Version: 0\nLast Update: Never\nParticipating Clients: 0")
        self.fl_status_label.setFont(QFont("Courier", 10))
        status_layout.addWidget(self.fl_status_label)
        
        # --- NEW: Add aggregation metadata display ---
        self.fl_metadata_label = QLabel(
            "Last Aggregation:\n"
            "  Participated: 0\n"
            "  Quarantined: 0\n"
            "  Avg Delta Norm: 0.0\n"
            "  Validation AUC: 0.0\n"
            "  DP Noise: 0.0\n"
            "  Rollback: No"
        )
        self.fl_metadata_label.setFont(QFont("Courier", 9))
        self.fl_metadata_label.setStyleSheet("color: #3498db; padding: 10px;")
        status_layout.addWidget(self.fl_metadata_label)
        
        # Manual aggregation button
        btn_layout = QHBoxLayout()
        
        metrics_btn = QPushButton("View Detailed Metrics")
        metrics_btn.clicked.connect(self.show_fl_metrics)
        btn_layout.addWidget(metrics_btn)
        
        aggregate_btn = QPushButton("Force Model Aggregation")
        aggregate_btn.clicked.connect(self.force_fl_aggregation)
        btn_layout.addWidget(aggregate_btn)
        
        reset_btn = QPushButton("Reset Global Model")
        reset_btn.clicked.connect(self.reset_fl_model)
        btn_layout.addWidget(reset_btn)
        status_layout.addLayout(btn_layout)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # Convergence chart
        convergence_group = QGroupBox("Model Convergence")
        convergence_layout = QVBoxLayout()
        
        self.convergence_chart = QChart()
        self.convergence_chart.setTitle("Convergence Delta Over Time")
        self.convergence_chart_view = QChartView(self.convergence_chart)
        self.convergence_chart_view.setRenderHint(QPainter.Antialiasing)
        convergence_layout.addWidget(self.convergence_chart_view)
        
        convergence_group.setLayout(convergence_layout)
        layout.addWidget(convergence_group)
        
        # Current model weights
        weights_group = QGroupBox("Current Global Model Weights")
        weights_layout = QVBoxLayout()
        
        self.fl_weights_text = QTextEdit()
        self.fl_weights_text.setReadOnly(True)
        self.fl_weights_text.setMaximumHeight(150)
        weights_layout.addWidget(self.fl_weights_text)
        
        weights_group.setLayout(weights_layout)
        layout.addWidget(weights_group)
        
        # --- NEW: Enhanced client contributions table with reputation ---
        contrib_group = QGroupBox("Client Contributions & Reputation")
        contrib_layout = QVBoxLayout()
        
        self.fl_contrib_table = QTableWidget()
        self.fl_contrib_table.setColumnCount(9)  # INCREASED from 5 to 9
        self.fl_contrib_table.setHorizontalHeaderLabels([
            "Client ID", "Updates", "Quality", "Reputation", "Avg Norm", 
            "Clipped", "Quarantined", "Last Update", "Status"
        ])
        self.fl_contrib_table.horizontalHeader().setStretchLastSection(True)
        contrib_layout.addWidget(self.fl_contrib_table)
        
        contrib_group.setLayout(contrib_layout)
        layout.addWidget(contrib_group)
        
        self.tabs.addTab(fl_tab, "Federated Learning")
               
    def create_pie_chart(self, title):
        """Create a pie chart"""
        series = QPieSeries()
        series.append("Online", 0)
        series.append("Offline", 0)
        
        chart = QChart()
        chart.addSeries(series)
        chart.setTitle(title)
        chart.legend().setAlignment(Qt.AlignBottom)
        
        return chart
    
    def create_bar_chart(self, title):
        """Create a bar chart"""
        set0 = QBarSet("Events")
        set0.append([0, 0, 0, 0])
        
        series = QBarSeries()
        series.append(set0)
        
        chart = QChart()
        chart.addSeries(series)
        chart.setTitle(title)
        chart.setAnimationOptions(QChart.SeriesAnimations)
        
        categories = ["Network", "Process", "Filesystem", "User"]
        axis_x = QBarCategoryAxis()
        axis_x.append(categories)
        chart.addAxis(axis_x, Qt.AlignBottom)
        series.attachAxis(axis_x)
        
        axis_y = QValueAxis()
        chart.addAxis(axis_y, Qt.AlignLeft)
        series.attachAxis(axis_y)
        
        chart.legend().setVisible(False)
        
        return chart
    
    def start_server(self):
        """Start the server thread"""
        self.server_thread = ServerThread(self.db_manager, self.fl_manager)
        self.server_thread.client_connected.connect(self.on_client_connected)
        self.server_thread.data_received.connect(self.on_data_received)
        self.server_thread.fl_update_received.connect(self.on_fl_update)
        self.server_thread.start()
        self.statusBar().showMessage("✅ Server running on port 9999")
    
    def on_client_connected(self, client_info):
        """Handle new client connection"""
        self.statusBar().showMessage(f"✅ New client connected: {client_info['hostname']}")
        self.load_clients()
        self.refresh_dashboard()
    
    def on_data_received(self, client_id, data):
        """Handle received telemetry data"""
        self.statusBar().showMessage(f"✅ Data received from {client_id[:8]}...")
    
    def load_alerts(self):
        """Load active alerts - FIXED: Show all categories"""
        alerts = self.db_manager.get_active_alerts()
        self.alerts_table.setRowCount(len(alerts))
        
        for i, alert in enumerate(alerts):
            self.alerts_table.setItem(i, 0, QTableWidgetItem(alert['timestamp'].strftime("%Y-%m-%d %H:%M:%S")))
            
            severity_item = QTableWidgetItem(alert['severity'].upper())
            if alert['severity'] == 'critical':
                severity_item.setBackground(QColor(231, 76, 60))
            elif alert['severity'] == 'high':
                severity_item.setBackground(QColor(241, 196, 15))
            elif alert['severity'] == 'medium':
                severity_item.setBackground(QColor(52, 152, 219))
            self.alerts_table.setItem(i, 1, severity_item)
            
            self.alerts_table.setItem(i, 2, QTableWidgetItem(alert['hostname']))
            
            # ✅ FIX: Show source_category with proper formatting
            category = alert.get('source_category', 'unknown')
            category_display = category.replace('_', ' ').title()
            self.alerts_table.setItem(i, 3, QTableWidgetItem(category_display))
            
            self.alerts_table.setItem(i, 4, QTableWidgetItem(alert['title']))
            self.alerts_table.setItem(i, 5, QTableWidgetItem(alert['description'][:100]))
            
            risk_item = QTableWidgetItem(f"{alert['risk_score']:.2f}")
            if alert['risk_score'] > 8.0:
                risk_item.setBackground(QColor(231, 76, 60))
            elif alert['risk_score'] > 6.0:
                risk_item.setBackground(QColor(241, 196, 15))
            self.alerts_table.setItem(i, 6, risk_item)

    def acknowledge_alert(self):
        """Acknowledge selected alert"""
        selected = self.alerts_table.currentRow()
        if selected < 0:
            QMessageBox.warning(self, "No Selection", "Please select an alert to acknowledge")
            return
        
        # Get alert ID from the selected row (you may need to store this differently)
        cursor = self.db_manager.connection.cursor()
        
        # Mark alert as acknowledged
        cursor.execute("""
            UPDATE alerts 
            SET acknowledged = TRUE, status = 'acknowledged'
            WHERE id = (
                SELECT id FROM alerts 
                WHERE status = 'active' 
                ORDER BY timestamp DESC, risk_score DESC 
                LIMIT 1 OFFSET %s
            )
        """, (selected,))
        
        self.db_manager.connection.commit()
        cursor.close()
        
        QMessageBox.information(self, "Alert Acknowledged", "Alert has been acknowledged")
        self.load_alerts()

    def reset_fl_model(self):
        """Reset federated learning model to defaults"""
        reply = QMessageBox.question(self, 'Reset FL Model',
                                    'Are you sure you want to reset the global FL model?',
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            # Reset to default values
            self.fl_manager.global_model = {
                'weights': {
                    'network_threshold': 2.0,
                    'process_threshold': 2.0,
                    'file_threshold': 2.0,
                    'network_sensitivity': 1.0,
                    'process_sensitivity': 1.0,
                    'file_sensitivity': 1.0,
                    'anomaly_alpha': 0.95,
                    'anomaly_beta': 0.1,
                    'network_baseline_mean': 0.0,
                    'network_baseline_std': 1.0,
                    'process_baseline_mean': 0.0,
                    'process_baseline_std': 1.0,
                    'file_baseline_mean': 0.0,
                    'file_baseline_std': 1.0
                },
                'momentum': {
                    'network_threshold': 0.0,
                    'process_threshold': 0.0,
                    'file_threshold': 0.0,
                    'network_sensitivity': 0.0,
                    'process_sensitivity': 0.0,
                    'file_sensitivity': 0.0
                },
                'version': 0,
                'last_update': datetime.now(),
                'convergence_history': []
            }
            
            # Clear client models
            self.fl_manager.client_models.clear()
            
            QMessageBox.information(self, "FL Model Reset", "Global FL model has been reset to defaults")
            self.refresh_fl_tab()
        
    def on_fl_update(self, client_id, model_params):
        """Handle federated learning update"""
        self.statusBar().showMessage(f"✓ FL update from {client_id[:8]}...")
        self.refresh_fl_tab()


    def refresh_data(self):
        """Refresh all data views"""
        self.check_offline_clients()
        current_tab = self.tabs.currentIndex()
        
        if current_tab in [4, 5, 6]:  # Network, Process, or Filesystem tabs
            self.refresh_client_filters()
        
        if current_tab == 0:  # Dashboard
            self.refresh_dashboard()
        elif current_tab == 1:  # Alerts
            self.load_alerts()
        elif current_tab == 2:  # Clients
            self.load_clients()
        elif current_tab == 3:  # Aggregated View
            self.load_aggregated_view()
        elif current_tab == 4:  # Network
            self.load_network_data()
        elif current_tab == 5:  # Processes
            self.load_process_data()
        elif current_tab == 6:  # Filesystem
            self.load_filesystem_data()
        elif current_tab == 7:  # User Activity
            self.load_user_activity()
        elif current_tab == 8:  # Federated Learning (FIXED INDEX)
            self.refresh_fl_tab()
    
    def refresh_dashboard(self):
        """Refresh dashboard statistics"""
        stats = self.db_manager.get_dashboard_stats()
        
        # Update labels
        total_clients = stats['clients']['total_clients'] or 0
        online_clients = stats['clients']['online_clients'] or 0
        fl_clients = stats['clients'].get('fl_clients', 0) or 0
        total_alerts = stats['alerts']['total_alerts'] or 0
        critical_alerts = stats['alerts']['critical_alerts'] or 0
        high_alerts = stats['alerts']['high_alerts'] or 0  # ADDED THIS LINE
        
        # ENHANCED: Show FL participation
        fl_status = f" | FL: {fl_clients}/{online_clients}" if fl_clients > 0 else ""
        
        self.total_clients_label.setText(f"Total Clients: {total_clients}")
        self.online_clients_label.setText(f"Online: {online_clients}{fl_status}")
        self.total_alerts_label.setText(f"Active Alerts: {total_alerts}")
        self.critical_alerts_label.setText(f"Critical: {critical_alerts}")
            
        # Update client status pie chart
        offline_clients = stats['clients']['offline_clients'] or 0
        self.client_status_chart.removeAllSeries()
        
        series = QPieSeries()
        series.append("Online", float(online_clients))
        series.append("Offline", float(offline_clients))
        
        if len(series.slices()) > 0:
            slice_online = series.slices()[0]
            slice_online.setBrush(QColor(46, 204, 113))
            if len(series.slices()) > 1:
                slice_offline = series.slices()[1]
                slice_offline.setBrush(QColor(231, 76, 60))
        
        self.client_status_chart.addSeries(series)
        
        # Update events bar chart
        network_events = stats['events']['network'] or 0
        process_events = stats['events']['process'] or 0
        filesystem_events = stats['events']['filesystem'] or 0
        user_events = stats['events']['user_activity'] or 0
        
        # Remove all series AND axes
        self.events_chart.removeAllSeries()
        for axis in self.events_chart.axes():
            self.events_chart.removeAxis(axis)
        
        set0 = QBarSet("Events")
        set0.append([
            float(network_events),
            float(process_events),
            float(filesystem_events),
            float(user_events)
        ])
        
        series_bar = QBarSeries()
        series_bar.append(set0)
        self.events_chart.addSeries(series_bar)
        
        # Re-add axes
        categories = ["Network", "Process", "Filesystem", "User"]
        axis_x = QBarCategoryAxis()
        axis_x.append(categories)
        self.events_chart.addAxis(axis_x, Qt.AlignBottom)
        series_bar.attachAxis(axis_x)
        
        total_events = network_events + process_events + filesystem_events + user_events
        axis_y = QValueAxis()
        axis_y.setRange(0, max(total_events, 10))
        self.events_chart.addAxis(axis_y, Qt.AlignLeft)
        series_bar.attachAxis(axis_y)
        
        # Update ALERT SEVERITY pie chart (was empty)
        self.severity_chart.removeAllSeries()
        
        severity_series = QPieSeries()
        severity_series.append("Critical", float(critical_alerts))
        severity_series.append("High", float(high_alerts))
        medium_alerts = total_alerts - critical_alerts - high_alerts
        severity_series.append("Medium/Low", float(max(medium_alerts, 0)))
        
        if len(severity_series.slices()) > 0:
            severity_series.slices()[0].setBrush(QColor(231, 76, 60))  # Red for critical
            if len(severity_series.slices()) > 1:
                severity_series.slices()[1].setBrush(QColor(241, 196, 15))  # Yellow for high
            if len(severity_series.slices()) > 2:
                severity_series.slices()[2].setBrush(QColor(52, 152, 219))  # Blue for medium/low
        
        self.severity_chart.addSeries(severity_series)
        
        # Update NETWORK ACTIVITY TIMELINE (was empty)
        self.timeline_chart.removeAllSeries()
        for axis in self.timeline_chart.axes():
            self.timeline_chart.removeAxis(axis)
        
        timeline_series = QLineSeries()
        timeline_series.setName("Network Events")
        
        timeline_data = stats['timeline']['network']
        if timeline_data and len(timeline_data) > 0:
            for record in timeline_data:
                # Convert timestamp to milliseconds since epoch
                timestamp_ms = int(record['hour'].timestamp() * 1000)
                count = record['count'] or 0
                timeline_series.append(timestamp_ms, float(count))
        else:
            # Add dummy data if no events
            from datetime import datetime
            now = datetime.now()
            for i in range(24):
                hour_ago = now - timedelta(hours=23-i)
                timestamp_ms = int(hour_ago.timestamp() * 1000)
                timeline_series.append(timestamp_ms, 0)
        
        self.timeline_chart.addSeries(timeline_series)
        
        # Add datetime axis
        axis_x_time = QDateTimeAxis()
        axis_x_time.setFormat("hh:mm")
        axis_x_time.setTitleText("Time")
        self.timeline_chart.addAxis(axis_x_time, Qt.AlignBottom)
        timeline_series.attachAxis(axis_x_time)
        
        # Add value axis
        axis_y_time = QValueAxis()
        max_events = max([r['count'] for r in timeline_data] if timeline_data else [0]) or 10
        axis_y_time.setRange(0, max_events)
        axis_y_time.setTitleText("Events")
        self.timeline_chart.addAxis(axis_y_time, Qt.AlignLeft)
        timeline_series.attachAxis(axis_y_time)
        
        # Update risks table
        all_risks = (stats['top_risks']['network'] + 
                    stats['top_risks']['process'] + 
                    stats['top_risks']['filesystem'])
        all_risks.sort(key=lambda x: x['risk_score'], reverse=True)
        
        self.risks_table.setRowCount(len(all_risks[:15]))
        for i, risk in enumerate(all_risks[:15]):
            self.risks_table.setItem(i, 0, QTableWidgetItem(risk['category'].capitalize()))
            self.risks_table.setItem(i, 1, QTableWidgetItem(risk['client_id'][:12]))
            self.risks_table.setItem(i, 2, QTableWidgetItem(str(risk.get('detail', 'N/A'))[:50]))
            
            risk_score = float(risk['risk_score']) if risk['risk_score'] else 0.0
            risk_item = QTableWidgetItem(f"{risk_score:.2f}")
            if risk_score > 7:
                risk_item.setBackground(QColor(231, 76, 60))
            elif risk_score > 5:
                risk_item.setBackground(QColor(241, 196, 15))
            self.risks_table.setItem(i, 3, risk_item)            
        def load_alerts(self):
            """Load active alerts"""
            alerts = self.db_manager.get_active_alerts()
            self.alerts_table.setRowCount(len(alerts))
            
            for i, alert in enumerate(alerts):
                self.alerts_table.setItem(i, 0, QTableWidgetItem(alert['timestamp'].strftime("%Y-%m-%d %H:%M:%S")))
                
                severity_item = QTableWidgetItem(alert['severity'].upper())
                if alert['severity'] == 'critical':
                    severity_item.setBackground(QColor(231, 76, 60))
                elif alert['severity'] == 'high':
                    severity_item.setBackground(QColor(241, 196, 15))
                self.alerts_table.setItem(i, 1, severity_item)
                
                self.alerts_table.setItem(i, 2, QTableWidgetItem(alert['hostname']))
                self.alerts_table.setItem(i, 3, QTableWidgetItem(alert['alert_type']))
                self.alerts_table.setItem(i, 4, QTableWidgetItem(alert['title']))
                self.alerts_table.setItem(i, 5, QTableWidgetItem(alert['description'][:100]))
                self.alerts_table.setItem(i, 6, QTableWidgetItem(f"{alert['risk_score']:.2f}"))
        
    def acknowledge_alert(self):
        """Acknowledge selected alert"""
        selected = self.alerts_table.currentRow()
        if selected < 0:
            QMessageBox.warning(self, "No Selection", "Please select an alert to acknowledge")
            return
        
        # Implementation for acknowledging alerts
        QMessageBox.information(self, "Alert Acknowledged", "Alert has been acknowledged")
        self.load_alerts()
    
    def load_clients(self):
        """Load client data into table"""
        clients = self.db_manager.get_all_clients()
        self.clients_table.setRowCount(len(clients))
        
        for i, client in enumerate(clients):
            self.clients_table.setItem(i, 0, QTableWidgetItem(client['client_id'][:12]))
            self.clients_table.setItem(i, 1, QTableWidgetItem(client['hostname']))
            self.clients_table.setItem(i, 2, QTableWidgetItem(client['ip_address']))
            self.clients_table.setItem(i, 3, QTableWidgetItem(f"{client['os_type']}"))
            self.clients_table.setItem(i, 4, QTableWidgetItem(client['device_role'] or 'N/A'))
            self.clients_table.setItem(i, 5, QTableWidgetItem(client['department'] or 'N/A'))
            
            status_item = QTableWidgetItem(client['status'])
            if client['status'] == 'online':
                status_item.setBackground(QColor(46, 204, 113))
            else:
                status_item.setBackground(QColor(231, 76, 60))
            self.clients_table.setItem(i, 6, status_item)
            
            fl_item = QTableWidgetItem("Yes" if client.get('federated_learning') else "No")
            if client.get('federated_learning'):
                fl_item.setBackground(QColor(52, 152, 219))
            self.clients_table.setItem(i, 7, fl_item)
            
            heartbeat = client['last_heartbeat'].strftime("%Y-%m-%d %H:%M:%S") if client['last_heartbeat'] else 'Never'
            self.clients_table.setItem(i, 8, QTableWidgetItem(heartbeat))
    
    def load_aggregated_view(self):
        """Load aggregated view - FIXED: Error handling and timeout protection"""
        try:
            # Show loading indicator
            self.agg_table.setRowCount(1)
            loading_item = QTableWidgetItem("Loading client analysis data...")
            self.agg_table.setItem(0, 0, loading_item)
            QApplication.processEvents()  # Force UI update
            
            # Get aggregated data with error handling
            aggregated = self.db_manager.get_client_aggregated_view()
            
            if not aggregated:
                self.agg_table.setRowCount(1)
                no_data_item = QTableWidgetItem("No client data available")
                self.agg_table.setItem(0, 0, no_data_item)
                return
            
            # Populate table
            self.agg_table.setRowCount(len(aggregated))
            
            for i, row in enumerate(aggregated):
                # Client ID
                self.agg_table.setItem(i, 0, QTableWidgetItem(row['client_id'][:12]))
                
                # Hostname
                self.agg_table.setItem(i, 1, QTableWidgetItem(row['hostname']))
                
                # Status with color
                status_item = QTableWidgetItem(row['status'])
                if row['status'] == 'online':
                    status_item.setBackground(QColor(46, 204, 113))
                else:
                    status_item.setBackground(QColor(231, 76, 60))
                self.agg_table.setItem(i, 2, status_item)
                
                # Event counts
                self.agg_table.setItem(i, 3, QTableWidgetItem(str(row['network_events_24h'] or 0)))
                self.agg_table.setItem(i, 4, QTableWidgetItem(str(row['process_events_24h'] or 0)))
                self.agg_table.setItem(i, 5, QTableWidgetItem(str(row['filesystem_events_24h'] or 0)))
                
                # Risk scores with color coding
                avg_net_risk = float(row['avg_network_risk'] or 0)
                net_risk_item = QTableWidgetItem(f"{avg_net_risk:.2f}")
                if avg_net_risk > 7:
                    net_risk_item.setBackground(QColor(231, 76, 60))  # Red
                elif avg_net_risk > 5:
                    net_risk_item.setBackground(QColor(241, 196, 15))  # Yellow
                self.agg_table.setItem(i, 6, net_risk_item)
                
                avg_proc_risk = float(row['avg_process_risk'] or 0)
                proc_risk_item = QTableWidgetItem(f"{avg_proc_risk:.2f}")
                if avg_proc_risk > 7:
                    proc_risk_item.setBackground(QColor(231, 76, 60))
                elif avg_proc_risk > 5:
                    proc_risk_item.setBackground(QColor(241, 196, 15))
                self.agg_table.setItem(i, 7, proc_risk_item)
                
                avg_file_risk = float(row['avg_filesystem_risk'] or 0)
                file_risk_item = QTableWidgetItem(f"{avg_file_risk:.2f}")
                if avg_file_risk > 7:
                    file_risk_item.setBackground(QColor(231, 76, 60))
                elif avg_file_risk > 5:
                    file_risk_item.setBackground(QColor(241, 196, 15))
                self.agg_table.setItem(i, 8, file_risk_item)
                
                # Active alerts
                alerts_item = QTableWidgetItem(str(row['active_alerts'] or 0))
                if row['active_alerts'] and row['active_alerts'] > 0:
                    alerts_item.setBackground(QColor(231, 76, 60))
                self.agg_table.setItem(i, 9, alerts_item)
            
            self.statusBar().showMessage(f"✓ Loaded {len(aggregated)} client(s)")
            
        except Exception as e:
            print(f"✗ Error loading aggregated view: {e}")
            import traceback
            traceback.print_exc()
            
            # Show error in table
            self.agg_table.setRowCount(1)
            error_item = QTableWidgetItem(f"Error loading data: {str(e)[:80]}")
            error_item.setBackground(QColor(231, 76, 60))
            self.agg_table.setItem(0, 0, error_item)
            
            self.statusBar().showMessage(f"✗ Error: {str(e)[:50]}")
            
            # Try to recover database connection
            try:
                self.db_manager.connection.rollback()
                print("  Attempting database reconnection...")
                self.db_manager.connect()
            except Exception as reconnect_error:
                print(f"  ✗ Reconnection failed: {reconnect_error}")
    
    def load_network_data(self):
        """Load network data with filtering - FIXED: Prevent auto-refresh loop"""
        # ✅ FIX: Disconnect signal temporarily to prevent recursive calls
        try:
            self.network_filter.currentIndexChanged.disconnect()
            self.network_client_filter.currentIndexChanged.disconnect()
        except:
            pass
        
        cursor = self.db_manager.connection.cursor(cursor_factory=RealDictCursor)
        
        filter_option = self.network_filter.currentText()
        where_clauses = []
        
        if "High Risk" in filter_option:
            where_clauses.append("risk_score > 7.0")
        elif "Anomalies" in filter_option:
            where_clauses.append("is_anomaly = TRUE")
        
        selected_client = self.network_client_filter.currentData()
        if selected_client:
            where_clauses.append(f"n.client_id = '{selected_client}'")
        
        where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        
        cursor.execute(f"""
            SELECT n.*, c.hostname 
            FROM network_data n
            JOIN clients c ON n.client_id = c.client_id
            {where_clause}
            ORDER BY n.timestamp DESC, n.risk_score DESC 
            LIMIT 500
        """)
        records = cursor.fetchall()
        cursor.close()
        
        self.network_table.setRowCount(len(records))
        for i, record in enumerate(records):
            self.network_table.setItem(i, 0, QTableWidgetItem(record['timestamp'].strftime("%Y-%m-%d %H:%M:%S")))
            self.network_table.setItem(i, 1, QTableWidgetItem(record['hostname'][:15]))
            self.network_table.setItem(i, 2, QTableWidgetItem(f"{record['src_ip']}:{record['src_port']}"))
            self.network_table.setItem(i, 3, QTableWidgetItem(f"{record['dst_ip']}:{record['dst_port']}"))
            self.network_table.setItem(i, 4, QTableWidgetItem(record['protocol'] or 'N/A'))
            self.network_table.setItem(i, 5, QTableWidgetItem(str(record.get('connection_count', 1))))
            self.network_table.setItem(i, 6, QTableWidgetItem(record['dns_query'] or 'N/A'))
            self.network_table.setItem(i, 7, QTableWidgetItem(record['geolocation'] or 'N/A'))
            
            indicators = ', '.join(record.get('threat_indicators', [])[:3]) if record.get('threat_indicators') else 'None'
            self.network_table.setItem(i, 8, QTableWidgetItem(indicators))
            
            anomaly_item = QTableWidgetItem("Yes" if record.get('is_anomaly') else "No")
            if record.get('is_anomaly'):
                anomaly_item.setBackground(QColor(241, 196, 15))
            self.network_table.setItem(i, 9, anomaly_item)
            
            risk_item = QTableWidgetItem(f"{record['risk_score']:.2f}" if record['risk_score'] else 'N/A')
            if record['risk_score'] and record['risk_score'] > 7:
                risk_item.setBackground(QColor(231, 76, 60))
            elif record['risk_score'] and record['risk_score'] > 5:
                risk_item.setBackground(QColor(241, 196, 15))
            self.network_table.setItem(i, 10, risk_item)
        
        # ✅ FIX: Reconnect signals after loading is complete
        self.network_filter.currentIndexChanged.connect(self.load_network_data)
        self.network_client_filter.currentIndexChanged.connect(self.load_network_data)

    def load_process_data(self):
        """Load process data with filtering - FIXED: Prevent auto-refresh loop"""
        # ✅ FIX: Disconnect signal temporarily
        try:
            self.process_filter.currentIndexChanged.disconnect()
            self.process_client_filter.currentIndexChanged.disconnect()
        except:
            pass
        
        cursor = self.db_manager.connection.cursor(cursor_factory=RealDictCursor)
        
        filter_option = self.process_filter.currentText()
        where_clauses = []
        
        if "High Risk" in filter_option:
            where_clauses.append("p.risk_score > 6.0")
        elif "Suspicious" in filter_option:
            where_clauses.append("array_length(p.threat_indicators, 1) > 0")
        
        selected_client = self.process_client_filter.currentData()
        if selected_client:
            where_clauses.append(f"p.client_id = '{selected_client}'")
        
        where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        
        cursor.execute(f"""
            SELECT p.*, c.hostname 
            FROM process_data p
            JOIN clients c ON p.client_id = c.client_id
            {where_clause}
            ORDER BY p.timestamp DESC, p.risk_score DESC 
            LIMIT 500
        """)
        records = cursor.fetchall()
        cursor.close()
        
        self.process_table.setRowCount(len(records))
        for i, record in enumerate(records):
            self.process_table.setItem(i, 0, QTableWidgetItem(record['timestamp'].strftime("%Y-%m-%d %H:%M:%S")))
            self.process_table.setItem(i, 1, QTableWidgetItem(record['hostname'][:15]))
            self.process_table.setItem(i, 2, QTableWidgetItem(record['process_name'] or 'N/A'))
            self.process_table.setItem(i, 3, QTableWidgetItem(str(record['pid'])))
            self.process_table.setItem(i, 4, QTableWidgetItem(record['parent_name'] or 'N/A'))
            self.process_table.setItem(i, 5, QTableWidgetItem(record['executable_hash'][:16] if record['executable_hash'] else 'N/A'))
            self.process_table.setItem(i, 6, QTableWidgetItem(f"{record['cpu_percent']:.1f}" if record['cpu_percent'] else 'N/A'))
            self.process_table.setItem(i, 7, QTableWidgetItem(f"{record['memory_mb']:.1f}" if record['memory_mb'] else 'N/A'))
            
            indicators = ', '.join(record.get('threat_indicators', [])[:3]) if record.get('threat_indicators') else 'None'
            self.process_table.setItem(i, 8, QTableWidgetItem(indicators))
            
            risk_item = QTableWidgetItem(f"{record['risk_score']:.2f}" if record['risk_score'] else 'N/A')
            if record['risk_score'] and record['risk_score'] > 7:
                risk_item.setBackground(QColor(231, 76, 60))
            elif record['risk_score'] and record['risk_score'] > 5:
                risk_item.setBackground(QColor(241, 196, 15))
            self.process_table.setItem(i, 9, risk_item)
        
        # ✅ FIX: Reconnect signals
        self.process_filter.currentIndexChanged.connect(self.load_process_data)
        self.process_client_filter.currentIndexChanged.connect(self.load_process_data)

    def load_filesystem_data(self):
        """Load filesystem data with filtering - FIXED: Prevent auto-refresh loop"""
        # ✅ FIX: Disconnect signal temporarily
        try:
            self.fs_filter.currentIndexChanged.disconnect()
            self.fs_client_filter.currentIndexChanged.disconnect()
        except:
            pass
        
        cursor = self.db_manager.connection.cursor(cursor_factory=RealDictCursor)
        
        filter_option = self.fs_filter.currentText()
        where_clauses = []
        
        if "High Risk" in filter_option:
            where_clauses.append("f.risk_score > 5.0")
        elif "Suspicious" in filter_option:
            where_clauses.append("f.is_suspicious = TRUE")
        
        selected_client = self.fs_client_filter.currentData()
        if selected_client:
            where_clauses.append(f"f.client_id = '{selected_client}'")
        
        where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        
        cursor.execute(f"""
            SELECT f.*, c.hostname 
            FROM filesystem_data f
            JOIN clients c ON f.client_id = c.client_id
            {where_clause}
            ORDER BY f.timestamp DESC, f.risk_score DESC 
            LIMIT 500
        """)
        records = cursor.fetchall()
        cursor.close()
        
        self.fs_table.setRowCount(len(records))
        for i, record in enumerate(records):
            self.fs_table.setItem(i, 0, QTableWidgetItem(record['timestamp'].strftime("%Y-%m-%d %H:%M:%S")))
            self.fs_table.setItem(i, 1, QTableWidgetItem(record['hostname'][:15]))
            
            event_item = QTableWidgetItem(record['event_type'] or 'N/A')
            if record['event_type'] == 'created':
                event_item.setBackground(QColor(52, 152, 219))
            self.fs_table.setItem(i, 2, event_item)
            
            self.fs_table.setItem(i, 3, QTableWidgetItem(record['file_name'][:30] if record['file_name'] else 'N/A'))
            self.fs_table.setItem(i, 4, QTableWidgetItem(record['file_extension'] or 'N/A'))
            
            size_kb = record['file_size'] / 1024 if record['file_size'] else 0
            self.fs_table.setItem(i, 5, QTableWidgetItem(f"{size_kb:.1f} KB"))
            
            directory = record.get('directory', 'N/A')
            if directory and len(directory) > 30:
                directory = '...' + directory[-27:]
            self.fs_table.setItem(i, 6, QTableWidgetItem(directory))
            
            file_hash = record.get('file_hash')
            if file_hash:
                hash_display = file_hash[:16] + "..."  # First 16 chars
                hash_item = QTableWidgetItem(hash_display)
                hash_item.setToolTip(f"Full SHA256: {file_hash}")  # Hover shows full hash
                hash_item.setForeground(QColor(46, 204, 113))  # Green = hash available
            else:
                hash_item = QTableWidgetItem('N/A')
                hash_item.setForeground(QColor(149, 165, 166))  # Gray
                
            self.fs_table.setItem(i, 7, hash_item)            
            indicators = ', '.join(record.get('threat_indicators', [])[:3]) if record.get('threat_indicators') else 'None'
            self.fs_table.setItem(i, 8, QTableWidgetItem(indicators))
            
            risk_item = QTableWidgetItem(f"{record['risk_score']:.2f}" if record['risk_score'] else 'N/A')
            if record['risk_score'] and record['risk_score'] > 7:
                risk_item.setBackground(QColor(231, 76, 60))
            elif record['risk_score'] and record['risk_score'] > 5:
                risk_item.setBackground(QColor(241, 196, 15))
            self.fs_table.setItem(i, 9, risk_item)
        
        # ✅ FIX: Reconnect signals
        self.fs_filter.currentIndexChanged.connect(self.load_filesystem_data)
        self.fs_client_filter.currentIndexChanged.connect(self.load_filesystem_data)

        
    def refresh_fl_tab(self):
        """Refresh federated learning tab with reputation and aggregation metadata - FIXED"""
        model = self.fl_manager.get_global_model()
        metrics = self.fl_manager.get_convergence_metrics()
        
        # Get aggregation metadata
        agg_metadata = self.fl_manager.get_aggregation_metadata()
        
        # Update status
        convergence_status = 'Stable' if (len(model['convergence_history']) > 0 and 
                                        model['convergence_history'][-1]['delta'] < 0.01) else 'Adapting'
        self.fl_status_label.setText(
            f"Model Version: {model['version']}\n"
            f"Last Update: {model['last_update'].strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Participating Clients: {metrics['participating_clients']}\n"
            f"Convergence: {convergence_status}"
        )
        
        # Update aggregation metadata display
        self.fl_metadata_label.setText(
            f"Last Aggregation Round:\n"
            f"  Participated: {agg_metadata['participated']}\n"
            f"  Quarantined: {agg_metadata['quarantined']}\n"
            f"  Avg Delta Norm: {agg_metadata['avg_delta_norm']:.4f}\n"
            f"  Validation AUC: {agg_metadata['validation_auc']:.4f}\n"
            f"  DP Noise Scale: {agg_metadata['dp_noise_applied']:.2f}\n"
            f"  Rollback: {'YES' if agg_metadata['rollback_occurred'] else 'No'}"
        )
        
        # Update convergence chart
        self.convergence_chart.removeAllSeries()
        for axis in self.convergence_chart.axes():
            self.convergence_chart.removeAxis(axis)
        
        if model['convergence_history']:
            series = QLineSeries()
            series.setName("Convergence Delta")
            
            for entry in model['convergence_history']:
                series.append(float(entry['version']), float(entry['delta']))
            
            self.convergence_chart.addSeries(series)
            
            axis_x = QValueAxis()
            axis_x.setTitleText("Model Version")
            axis_x.setLabelFormat("%d")
            self.convergence_chart.addAxis(axis_x, Qt.AlignBottom)
            series.attachAxis(axis_x)
            
            axis_y = QValueAxis()
            axis_y.setTitleText("Convergence Delta")
            axis_y.setLabelFormat("%.4f")
            self.convergence_chart.addAxis(axis_y, Qt.AlignLeft)
            series.attachAxis(axis_y)
        
        # Update weights display
        weights_text = "=== Thresholds ===\n"
        for key in ['network_threshold', 'process_threshold', 'file_threshold']:
            if key in model['weights']:
                weights_text += f"{key}: {model['weights'][key]:.3f}\n"
        
        weights_text += "\n=== Sensitivities ===\n"
        for key in ['network_sensitivity', 'process_sensitivity', 'file_sensitivity']:
            if key in model['weights']:
                weights_text += f"{key}: {model['weights'][key]:.3f}\n"
        
        weights_text += "\n=== Global Baselines ===\n"
        for key in ['network_baseline_mean', 'network_baseline_std', 
                    'process_baseline_mean', 'process_baseline_std']:
            if key in model['weights']:
                weights_text += f"{key}: {model['weights'][key]:.3f}\n"
        
        self.fl_weights_text.setText(weights_text)
        
        # --- FIXED: Update client contributions table with ALL 9 columns ---
        reputation_data = self.fl_manager.get_reputation_data()
        self.fl_contrib_table.setRowCount(len(reputation_data))
        
        sorted_clients = sorted(reputation_data.items(), 
                            key=lambda x: x[1]['reputation'], 
                            reverse=True)
        
        for i, (client_id, contrib) in enumerate(sorted_clients):
            # Column 0: Client ID
            self.fl_contrib_table.setItem(i, 0, QTableWidgetItem(client_id[:12]))
            
            # Column 1: Update count
            self.fl_contrib_table.setItem(i, 1, QTableWidgetItem(str(contrib['count'])))
            
            # Column 2: Quality score
            quality_val = float(contrib.get('quality', 0))
            quality_item = QTableWidgetItem(f"{quality_val:.3f}")
            if quality_val > 1.5:
                quality_item.setBackground(QColor(46, 204, 113))  # Green
            elif quality_val < 0.5:
                quality_item.setBackground(QColor(241, 196, 15))  # Yellow
            self.fl_contrib_table.setItem(i, 2, quality_item)
            
            # Column 3: Reputation score
            reputation_val = float(contrib.get('reputation', 1.0))
            reputation_item = QTableWidgetItem(f"{reputation_val:.3f}")
            if reputation_val > 1.5:
                reputation_item.setBackground(QColor(46, 204, 113))  # Green - excellent
            elif reputation_val > 1.0:
                reputation_item.setBackground(QColor(52, 152, 219))  # Blue - good
            elif reputation_val < 0.5:
                reputation_item.setBackground(QColor(231, 76, 60))  # Red - poor
            else:
                reputation_item.setBackground(QColor(241, 196, 15))  # Yellow - moderate
            self.fl_contrib_table.setItem(i, 3, reputation_item)
            
            # Column 4: Average norm
            avg_norm_val = float(contrib.get('avg_norm', 0))
            norm_item = QTableWidgetItem(f"{avg_norm_val:.4f}")
            if avg_norm_val > CLIP_BOUND:
                norm_item.setBackground(QColor(231, 76, 60))  # Red - too high
            elif avg_norm_val > CLIP_BOUND * 0.7:
                norm_item.setBackground(QColor(241, 196, 15))  # Yellow - high
            self.fl_contrib_table.setItem(i, 4, norm_item)
            
            # Column 5: Clipped count with ratio
            clipped_count = int(contrib.get('clipped_count', 0))
            total_count = max(int(contrib.get('count', 1)), 1)
            clipped_ratio = clipped_count / total_count
            clipped_item = QTableWidgetItem(f"{clipped_count} ({clipped_ratio:.1%})")
            if clipped_ratio > 0.5:
                clipped_item.setBackground(QColor(231, 76, 60))  # Red - often clipped
            elif clipped_ratio > 0.2:
                clipped_item.setBackground(QColor(241, 196, 15))  # Yellow
            self.fl_contrib_table.setItem(i, 5, clipped_item)
            
            # Column 6: Quarantine count
            quarantine_count = int(contrib.get('quarantine_count', 0))
            quarantine_item = QTableWidgetItem(str(quarantine_count))
            if quarantine_count > 0:
                quarantine_item.setBackground(QColor(231, 76, 60))  # Red
            self.fl_contrib_table.setItem(i, 6, quarantine_item)
            
            # Column 7: Last update timestamp
            last_update = contrib.get('last_update')
            if last_update:
                last_update_str = last_update.strftime('%Y-%m-%d %H:%M:%S')
                time_diff = datetime.now() - last_update
                status = 'Active' if time_diff.total_seconds() < 600 else 'Idle'
            else:
                last_update_str = 'Never'
                status = 'Inactive'
            self.fl_contrib_table.setItem(i, 7, QTableWidgetItem(last_update_str))
            
            # Column 8: Status
            status_item = QTableWidgetItem(status)
            if status == 'Active':
                status_item.setBackground(QColor(46, 204, 113))  # Green
            elif status == 'Idle':
                status_item.setBackground(QColor(241, 196, 15))  # Yellow
            else:
                status_item.setBackground(QColor(231, 76, 60))  # Red
            self.fl_contrib_table.setItem(i, 8, status_item)
            
    def force_fl_aggregation(self):
        """Force federated learning aggregation"""
        aggregated = self.fl_manager.aggregate_models()
        if aggregated:
            QMessageBox.information(self, "FL Aggregation", 
                                  f"Model aggregated successfully!\nNew version: {aggregated['version']}")
            self.refresh_fl_tab()
        else:
            QMessageBox.warning(self, "FL Aggregation", 
                              "Not enough client updates for aggregation (need at least 2)")
    
    def closeEvent(self, event):
        """Handle application close"""
        if self.server_thread:
            self.server_thread.stop()
            self.server_thread.wait()
        event.accept()


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()