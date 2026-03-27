"""
admin/fl/fl_server.py

FederatedLearningManager — client update reception, validation, clipping,
reputation tracking, and quality scoring.

aggregate_models() and its private helpers live in aggregator.py.
"""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime
from typing import Optional
import os

import numpy as np

from admin.utils.config import (
    CLIP_BOUND,
    DP_NOISE_SCALE,
    FL_MU,
    TRIM_FRAC,
    VALIDATION_AUC_DROP_THRESHOLD,
)

try:
    from shared.detection_config import (
        FL_QUARANTINE_CLIP_FACTOR,
        FL_MIN_CLIENT_SAMPLES,
        FL_HIGH_ANOMALY_RATE,
        FL_MAX_ABS_WEIGHT,
    )
except ImportError:
    FL_QUARANTINE_CLIP_FACTOR = 1.5
    FL_MIN_CLIENT_SAMPLES = 10
    FL_HIGH_ANOMALY_RATE = 0.5
    FL_MAX_ABS_WEIGHT = 1000.0


class FederatedLearningManager:
    """
    Advanced Federated Learning Manager.
    Handles:
      • Per-client L2 gradient clipping
      • Quarantine of misbehaving clients
      • Client reputation & quality scoring
      • Robust update validation (structure, metadata, bounds, norm)
      • Audit logging
      • Validation dataset management
    """

    def __init__(self):
        self.client_models: dict = {}
        self.client_contributions: defaultdict = defaultdict(lambda: {
            'count': 0,
            'quality': 1.0,
            'last_update': None,
            'reputation': 1.0,
            'clipped_count': 0,
            'quarantine_count': 0,
            'avg_norm': 0.0,
            'participation_rate': 1.0
        })

        # Enhanced global model with momentum
        self.global_model: dict = {
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
                'file_baseline_std': 1.0,
            },
            'momentum': {
                'network_threshold': 0.0,
                'process_threshold': 0.0,
                'file_threshold': 0.0,
                'network_sensitivity': 0.0,
                'process_sensitivity': 0.0,
                'file_sensitivity': 0.0,
            },
            'version': 0,
            'last_update': datetime.now(),
            'convergence_history': [],
        }

        # --- LOADS TRUE PRE-TRAINED MODELS ---
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        autoencoder_path = os.path.join(models_dir, 'global_autoencoder.h5')
        if os.path.exists(autoencoder_path):
            try:
                from tensorflow import keras

                print(f"Loading Global Autoencoder from {autoencoder_path}")
                autoencoder = keras.models.load_model(autoencoder_path)
                for i, layer in enumerate(autoencoder.layers):
                    if layer.weights:
                        weights = layer.get_weights()
                        for j, w in enumerate(weights):
                            key = f"ae_layer_{i}_weight_{j}"
                            self.global_model['weights'][key] = w.tolist()
                            self.global_model['momentum'][key] = np.zeros_like(w).tolist()
                print(f"Loaded {len([k for k in self.global_model['weights'].keys() if 'ae_layer' in k])} tensor layers.")
            except Exception as e:
                print(f"Error loading global autoencoder: {e}")

        # Validation dataset for model quality checks
        self.validation_feature_cache: deque = deque(maxlen=200)
        self.validation_labels_cache: deque = deque(maxlen=200)  # 0=normal, 1=anomaly

        # Update rejection tracking
        self.rejected_updates: defaultdict = defaultdict(int)
        self.malformed_updates: defaultdict = defaultdict(int)

        # Audit log
        self.audit_log: deque = deque(maxlen=100)

        # Model savepoint for rollback
        self.model_savepoint: Optional[dict] = None
        self.last_validation_metrics: dict = {'auc': 0.85, 'fpr': 0.05, 'tpr': 0.90}

        # FedProx configuration
        self.mu: float = FL_MU
        self.momentum_factor: float = 0.9

        # Differential privacy configuration
        self.dp_epsilon: float = 1.0
        self.dp_delta: float = 1e-5
        self.dp_noise_scale: float = DP_NOISE_SCALE

        # Aggregation metadata tracking
        self.aggregation_metadata: dict = {
            'last_round': {
                'participated': 0,
                'quarantined': 0,
                'avg_delta_norm': 0.0,
                'avg_pre_clip_norm': 0.0,
                'convergence_delta': 0.0,
                'scalar_mean_abs_delta': 0.0,
                'scalar_max_abs_delta': 0.0,
                'validation_auc': 0.0,
                'dp_noise_applied': 0.0,
                'rollback_occurred': False,
            }
        }

        self.aggregation_history: deque = deque(maxlen=50)

        # Pending updates queue
        self.pending_updates: list = []

    # =========================================================================
    # Public: receive_client_update (used by ServerThread / fl_server handler)
    # =========================================================================

    def receive_client_update(self, client_id: str, model_parameters: dict) -> None:
        """
        Receive and validate a model update from a client with L2 clipping.
        FIXED: Proper numpy type conversion.
        """
        timestamp = datetime.now()

        # Extract metadata with explicit type conversion
        metadata = {
            'samples_used': int(
                model_parameters.get('data_quality', {}).get('network_samples', 0)
            ),
            'local_epochs': 1,
            'loss': 0.0,
            'anomaly_rate': float(model_parameters.get('anomaly_rate', 0.0)),
            'timestamp': timestamp,
        }

        # Compute model delta
        model_delta: dict = {}
        client_weights = model_parameters.get('weights', {})

        for key in client_weights.keys():
            if key in self.global_model['weights']:
                client_val = client_weights[key]
                global_val = self.global_model['weights'][key]
                
                if isinstance(client_val, (list, np.ndarray)):
                    model_delta[key] = np.array(client_val) - np.array(global_val)
                else:
                    model_delta[key] = float(client_val) - float(global_val)

        # Compute pre-clipping L2 norm
        pre_norm = self._compute_l2_norm(model_delta)

        # Apply per-client L2 clipping
        clipped_delta, post_norm = self._clip_delta(model_delta, CLIP_BOUND)
        was_clipped = post_norm < pre_norm

        # Quarantine check
        is_quarantined = False
        if post_norm > CLIP_BOUND * FL_QUARANTINE_CLIP_FACTOR:
            is_quarantined = True
            print(f"❌ Client {client_id[:12]} QUARANTINED: norm={post_norm:.4f}")

        # Update contribution metadata with explicit type conversion
        contrib = self.client_contributions[client_id]
        contrib['count'] += 1
        contrib['last_update'] = timestamp
        contrib['avg_norm'] = float(
            (contrib['avg_norm'] * (contrib['count'] - 1) + post_norm) / contrib['count']
        )

        if was_clipped:
            contrib['clipped_count'] += 1
        if is_quarantined:
            contrib['quarantine_count'] += 1

        self._update_reputation(client_id, post_norm, was_clipped, is_quarantined, metadata)

        quality_score = self._calculate_quality_score(model_parameters)
        contrib['quality'] = quality_score

        # Add to pending updates queue (only if not quarantined)
        if not is_quarantined:
            self.pending_updates.append({
                'client_id': client_id,
                'delta': clipped_delta,
                'pre_norm': float(pre_norm),
                'post_norm': float(post_norm),
                'metadata': metadata,
                'quality_score': float(quality_score),
                'reputation': float(contrib['reputation']),
            })

        # Store original client model
        self.client_models[client_id] = {
            'parameters': model_parameters,
            'timestamp': timestamp,
            'quality_score': float(quality_score),
            'data_samples': int(metadata['samples_used']),
        }

        print(
            f"✓ Received FL update from {client_id[:12]} "
            f"(pre_norm={pre_norm:.4f}, post_norm={post_norm:.4f}, "
            f"clipped={was_clipped}, quarantined={is_quarantined}, "
            f"reputation={contrib['reputation']:.3f})"
        )

    def receive_client_update_robust(self, client_id: str, model_parameters: dict) -> bool:
        """
        Enhanced receive with type safety, malformed update rejection, and audit.
        Returns True if update was accepted, False otherwise.
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
                'samples_used': int(
                    model_parameters.get('data_quality', {}).get('network_samples', 0)
                ),
                'local_epochs': 1,
                'loss': 0.0,
                'anomaly_rate': float(model_parameters.get('anomaly_rate', 0.0)),
                'timestamp': timestamp,
            }

            if metadata['samples_used'] < FL_MIN_CLIENT_SAMPLES:
                print(f"✗ REJECTED: Insufficient samples ({metadata['samples_used']}) from {client_id[:12]}")
                self._audit_log_event('insufficient_samples', client_id, metadata)
                return False

            if metadata['anomaly_rate'] > FL_HIGH_ANOMALY_RATE:
                print(f"⚠ WARNING: High anomaly rate ({metadata['anomaly_rate']:.2%}) from {client_id[:12]}")
                self._audit_log_event('high_anomaly_rate', client_id, metadata)

        except (ValueError, TypeError, KeyError) as e:
            self.malformed_updates[client_id] += 1
            print(f"✗ REJECTED: Metadata extraction failed for {client_id[:12]}: {e}")
            self._audit_log_event('metadata_error', client_id, {'error': str(e)})
            return False

        # VALIDATION 3: Compute model delta with bounds checking
        model_delta: dict = {}
        client_weights = model_parameters.get('weights', {})

        try:
            for key in client_weights.keys():
                if key in self.global_model['weights']:
                    # Support both old scalar variables and new NN tensors
                    client_val = client_weights[key]
                    global_val = self.global_model['weights'][key]

                    if isinstance(client_val, (list, np.ndarray)):
                        c_arr = np.array(client_val)
                        g_arr = np.array(global_val)
                        
                        if np.max(np.abs(c_arr)) > FL_MAX_ABS_WEIGHT:
                            print(f"✗ REJECTED: Extreme tensor weight value in {key}")
                            self._audit_log_event('extreme_tensor_weight', client_id, {'key': key})
                            return False
                            
                        model_delta[key] = c_arr - g_arr
                    else:
                        c_scalar = float(client_val)
                        g_scalar = float(global_val)

                        if abs(c_scalar) > FL_MAX_ABS_WEIGHT or abs(g_scalar) > FL_MAX_ABS_WEIGHT:
                            print(f"✗ REJECTED: Extreme weight value in {key}: {c_scalar}")
                            self._audit_log_event('extreme_weight', client_id, {'key': key, 'value': c_scalar})
                            return False

                        model_delta[key] = c_scalar - g_scalar

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
            self._audit_log_event('extreme_norm', client_id, {'pre_norm': pre_norm, 'threshold': CLIP_BOUND * 3})
            return False

        # Apply L2 clipping
        clipped_delta, post_norm = self._clip_delta(model_delta, CLIP_BOUND)
        was_clipped = post_norm < pre_norm

        # Quarantine check
        is_quarantined = False
        if post_norm > CLIP_BOUND * 1.5:
            is_quarantined = True
            self.rejected_updates[client_id] += 1
            print(f"⛔ Client {client_id[:12]} QUARANTINED: norm={post_norm:.4f}")
            self._audit_log_event('quarantined', client_id, {'post_norm': post_norm})
            return False  # Don't add to pending updates

        # Update contribution metadata
        contrib = self.client_contributions[client_id]
        contrib['count'] += 1
        contrib['last_update'] = timestamp
        contrib['avg_norm'] = float(
            (contrib['avg_norm'] * (contrib['count'] - 1) + post_norm) / contrib['count']
        )

        if was_clipped:
            contrib['clipped_count'] += 1
        if is_quarantined:
            contrib['quarantine_count'] += 1

        self._update_reputation(client_id, post_norm, was_clipped, is_quarantined, metadata)

        quality_score = self._calculate_quality_score(model_parameters)
        contrib['quality'] = quality_score

        self.pending_updates.append({
            'client_id': client_id,
            'delta': clipped_delta,
            'pre_norm': float(pre_norm),
            'post_norm': float(post_norm),
            'metadata': metadata,
            'quality_score': float(quality_score),
            'reputation': float(contrib['reputation']),
        })

        self.client_models[client_id] = {
            'parameters': model_parameters,
            'timestamp': timestamp,
            'quality_score': float(quality_score),
            'data_samples': int(metadata['samples_used']),
        }

        self._audit_log_event('update_accepted', client_id, {
            'pre_norm': pre_norm,
            'post_norm': post_norm,
            'clipped': was_clipped,
            'quality': quality_score,
            'reputation': contrib['reputation'],
        })

        print(
            f"✓ Received FL update from {client_id[:12]} "
            f"(pre_norm={pre_norm:.4f}, post_norm={post_norm:.4f}, "
            f"clipped={was_clipped}, reputation={contrib['reputation']:.3f})"
        )
        return True

    # =========================================================================
    # Validation dataset management
    # =========================================================================

    def add_to_validation_set(self, feature_vector, is_anomaly: bool) -> None:
        """Add labelled data to validation set for model quality checks."""
        self.validation_feature_cache.append(feature_vector)
        self.validation_labels_cache.append(1 if is_anomaly else 0)

    # =========================================================================
    # Public getters (used by GUI and aggregator)
    # =========================================================================

    def get_global_model(self) -> dict:
        """Return a shallow copy of the current global model to prevent accidental mutation."""
        import copy
        return copy.deepcopy(self.global_model)

    def get_convergence_metrics(self) -> dict:
        """Return convergence metrics for visualisation."""
        return {
            'history': self.global_model['convergence_history'],
            'current_version': self.global_model['version'],
            'participating_clients': len(self.client_models),
            'client_contributions': dict(self.client_contributions),
        }

    def get_reputation_data(self) -> defaultdict:
        """Return client reputation data for GUI display."""
        return self.client_contributions

    def get_aggregation_metadata(self) -> dict:
        """Return last aggregation round metadata."""
        return self.aggregation_metadata['last_round']

    def get_audit_log(self, limit: int = 50) -> list:
        """Retrieve recent audit log entries."""
        return list(self.audit_log)[-limit:]

    def get_rejection_stats(self) -> dict:
        """Return update rejection statistics."""
        return {
            'rejected_by_client': dict(self.rejected_updates),
            'malformed_by_client': dict(self.malformed_updates),
            'total_rejected': sum(self.rejected_updates.values()),
            'total_malformed': sum(self.malformed_updates.values()),
        }

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _compute_l2_norm(self, delta_dict: dict) -> float:
        """Compute L2 norm of model delta."""
        squared_sum = 0.0
        for value in delta_dict.values():
            if isinstance(value, (int, float)):
                squared_sum += value ** 2
            elif isinstance(value, (list, np.ndarray)):
                squared_sum += np.sum(np.array(value) ** 2)
        return float(np.sqrt(squared_sum))

    def _clip_delta(self, delta_dict: dict, clip_bound: float):
        """Clip delta so its L2 norm <= clip_bound. Returns (clipped_dict, post_norm)."""
        norm = self._compute_l2_norm(delta_dict)
        if norm > clip_bound:
            scale_factor = clip_bound / norm
            return {k: v * scale_factor for k, v in delta_dict.items()}, clip_bound
        return delta_dict, norm

    def _update_reputation(
        self,
        client_id: str,
        post_norm: float,
        was_clipped: bool,
        is_quarantined: bool,
        metadata: dict,
    ) -> None:
        """Update client reputation based on contribution quality."""
        contrib = self.client_contributions[client_id]
        reputation = contrib['reputation']

        # Good behaviour
        if not was_clipped and post_norm < CLIP_BOUND * 0.5:
            reputation = min(reputation * 1.02, 2.0)
        if contrib['count'] > 10:
            reputation = min(reputation * 1.01, 2.0)

        # Bad behaviour
        if was_clipped:
            reputation = max(reputation * 0.98, 0.1)
        if is_quarantined:
            reputation = max(reputation * 0.90, 0.1)
        if post_norm > CLIP_BOUND:
            reputation = max(reputation * 0.95, 0.1)

        contrib['reputation'] = reputation

    def _calculate_quality_score(self, model_parameters: dict) -> float:
        """Calculate data quality score based on multiple factors."""
        data_quality = model_parameters.get('data_quality', {})
        statistics = model_parameters.get('statistics', {})

        total_samples = (
            data_quality.get('network_samples', 0)
            + data_quality.get('process_samples', 0)
            + data_quality.get('file_samples', 0)
        )

        volume_score = max(min(total_samples / 100.0, 1.0), 0.1)

        network_stats = statistics.get('network', {})
        variance_score = 1.0
        if network_stats.get('variance_connections', 0) > 0:
            variance = network_stats['variance_connections']
            variance_score = min(variance / 50.0, 1.5)

        anomaly_rate = model_parameters.get('anomaly_rate', 0.0)
        anomaly_score = 1.0 + (anomaly_rate * 0.5)

        quality = volume_score * min(variance_score, 1.2) * min(anomaly_score, 1.3)
        return float(min(quality, 2.0))

    def _validate_update_structure(self, model_parameters: dict) -> bool:
        """Validate update structure and types."""
        required_keys = ['weights', 'data_quality', 'statistics']
        for key in required_keys:
            if key not in model_parameters:
                return False

        weights = model_parameters.get('weights', {})
        if not isinstance(weights, dict):
            return False

        expected_weight_keys = [
            'network_threshold', 'process_threshold', 'file_threshold',
            'network_sensitivity', 'process_sensitivity', 'file_sensitivity',
        ]
        for key in expected_weight_keys:
            if key not in weights:
                return False
            try:
                float(weights[key])
            except (ValueError, TypeError):
                return False

        data_quality = model_parameters.get('data_quality', {})
        if not isinstance(data_quality, dict):
            return False

        return True

    def _audit_log_event(self, event_type: str, client_id: str, details: dict) -> None:
        """Log an audit event."""
        self.audit_log.append({
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'client_id': client_id[:12],
            'details': details,
        })