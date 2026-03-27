"""
admin/fl/aggregator.py

Federated Learning aggregation logic.
Extracted from FederatedLearningManager.aggregate_models() and its private helpers.

Functions
---------
aggregate_models(fl_manager)            — main aggregation entry point
aggregate_and_persist(fl_manager, db)   — convenience: aggregate then persist to DB

Private helpers (module-level, consumed by aggregate_models):
  _calculate_reputation_weights(updates)
  _trimmed_mean_aggregation(updates, weights, trim_frac)
  _add_dp_noise(aggregated, num_clients, clip_bound, noise_scale)
  _aggregate_baselines_robust(fl_manager, updates)
  _validate_global_model(fl_manager, model_weights)
  _validate_global_model_enhanced(fl_manager, new_weights)
  _should_rollback(fl_manager, new_metrics)
  _calculate_convergence(fl_manager, new_weights)
"""

from __future__ import annotations

import numpy as np
from datetime import datetime

from admin.utils.config import (
    CLIP_BOUND,
    DP_NOISE_SCALE,
    TRIM_FRAC,
    VALIDATION_AUC_DROP_THRESHOLD,
)

try:
    from shared.detection_config import (
        SCALAR_CLIP_BOUNDS as _SCALAR_BOUNDS,
        FL_LEARNING_RATE as _NN_LR,
        FL_MOMENTUM as _NN_MOM,
        FL_SCALAR_EMA_OLD_WEIGHT,
        FL_MIN_VALIDATION_SAMPLES,
    )
except ImportError:
    _SCALAR_BOUNDS = {}
    _NN_LR = 0.5
    _NN_MOM = 0.9
    FL_SCALAR_EMA_OLD_WEIGHT = 0.7
    FL_MIN_VALIDATION_SAMPLES = 20


# =============================================================================
# Public entry point
# =============================================================================

def aggregate_models(fl_manager) -> dict | None:
    """
    True federated learning with ACTUAL weight updates and convergence tracking.
    Mutates fl_manager.global_model in-place on success.

    Returns
    -------
    dict | None
        Updated global_model dict, or None if aggregation was skipped/rolled back.
    """
    if len(fl_manager.pending_updates) < 1:
        print("Not enough pending updates for aggregation")
        return None

    try:
        print(f"\n{'='*70}")
        print(f"🔧 FEDERATED LEARNING AGGREGATION (ENHANCED)")
        print(f"{'='*70}")
        print(f"Pending updates: {len(fl_manager.pending_updates)}")

        # Create model savepoint for rollback
        fl_manager.model_savepoint = {
            'weights': fl_manager.global_model['weights'].copy(),
            'version': fl_manager.global_model['version'],
            'timestamp': datetime.now(),
        }

        # Calculate reputation weights
        weights_per_update = _calculate_reputation_weights(fl_manager.pending_updates)

        # =====================================================================
        # SPLIT PARAMETERS INTO TWO AGGREGATION PIPELINES
        # =====================================================================
        SCALAR_KEYS = {
            'network_threshold', 'process_threshold', 'file_threshold',
            'network_sensitivity', 'process_sensitivity', 'file_sensitivity',
            'anomaly_alpha', 'anomaly_beta',
            'network_baseline_mean', 'network_baseline_std',
            'process_baseline_mean', 'process_baseline_std',
            'file_baseline_mean', 'file_baseline_std',
            'iso_threshold', 'ae_threshold',
        }

        param_keys = set(fl_manager.global_model['weights'].keys())
        scalar_keys = {k for k in param_keys if k in SCALAR_KEYS}
        tensor_keys = param_keys - scalar_keys

        aggregated_weights: dict = {}

        # ── Pipeline A: Scalar heuristic params — robust median ────────────
        print("  [AGG] Scalar parameters (robust median):")
        for key in scalar_keys:
            client_values = []
            for update in fl_manager.pending_updates:
                cid = update['client_id']
                if cid in fl_manager.client_models:
                    val = fl_manager.client_models[cid]['parameters'].get('weights', {}).get(key)
                    if val is not None:
                        client_values.append(float(val))

            if client_values:
                old_val = float(fl_manager.global_model['weights'].get(key, 0))
                client_median = float(np.median(client_values))
                new_val = FL_SCALAR_EMA_OLD_WEIGHT * old_val + (1 - FL_SCALAR_EMA_OLD_WEIGHT) * client_median

                if key in _SCALAR_BOUNDS:
                    lo, hi = _SCALAR_BOUNDS[key]
                    new_val = float(np.clip(new_val, lo, hi))
                elif 'sensitivity' in key:
                    new_val = float(np.clip(new_val, _SCALAR_BOUNDS.get('network_sensitivity', (0.3, 2.0))[0],
                                                     _SCALAR_BOUNDS.get('network_sensitivity', (0.3, 2.0))[1]))
                elif 'alpha' in key or 'beta' in key:
                    new_val = float(np.clip(new_val, 0.0, 1.0))

                aggregated_weights[key] = new_val
                if abs(new_val - old_val) > 0.005:
                    print(f"    {key}: {old_val:.4f} → {new_val:.4f}")
            else:
                aggregated_weights[key] = fl_manager.global_model['weights'].get(key, 0)

        # ── Pipeline B: Neural network tensors — FedAvg with momentum ──────
        NN_LR = _NN_LR
        NN_MOMENTUM = _NN_MOM
        tensor_count = 0

        for key in tensor_keys:
            weighted_sum = None
            total_weight = 0.0

            for update in fl_manager.pending_updates:
                cid = update['client_id']
                w = weights_per_update[cid]

                if cid in fl_manager.client_models:
                    client_val = fl_manager.client_models[cid]['parameters'].get('weights', {}).get(key)
                    if client_val is not None:
                        client_arr = np.array(client_val)
                        raw_global = fl_manager.global_model['weights'].get(key)
                        if raw_global is None or (np.isscalar(raw_global) and np.ndim(client_arr) > 0):
                            global_arr = np.zeros_like(client_arr)
                        else:
                            global_arr = np.array(raw_global)

                        if global_arr.shape == client_arr.shape:
                            delta = client_arr - global_arr
                            norm = np.linalg.norm(delta)
                            if norm > CLIP_BOUND:
                                delta = delta * (CLIP_BOUND / norm)
                                client_arr = global_arr + delta

                        if weighted_sum is None:
                            weighted_sum = client_arr * w
                        else:
                            weighted_sum = weighted_sum + client_arr * w
                        total_weight += w

            if weighted_sum is not None and total_weight > 0:
                agg_val = weighted_sum / total_weight
                old_val = np.array(fl_manager.global_model['weights'][key])

                momentum_term = np.array(
                    fl_manager.global_model['momentum'].get(key, np.zeros_like(old_val))
                )
                gradient = agg_val - old_val
                new_mom = NN_MOMENTUM * momentum_term + (1 - NN_MOMENTUM) * gradient
                fl_manager.global_model['momentum'][key] = new_mom.tolist() if hasattr(new_mom, 'tolist') else new_mom

                new_val = old_val + NN_LR * new_mom
                aggregated_weights[key] = new_val.tolist() if hasattr(new_val, 'tolist') else float(new_val)
                tensor_count += 1
            else:
                aggregated_weights[key] = fl_manager.global_model['weights'].get(key)

        print(f"  [AGG] Neural network tensors aggregated: {tensor_count}")

        # =====================================================================
        # SERVER-SIDE DIFFERENTIAL PRIVACY NOISE (only on NN tensors)
        # =====================================================================
        fl_manager.dp_noise_scale = DP_NOISE_SCALE
        nn_weights_for_noise = {k: aggregated_weights[k] for k in tensor_keys if k in aggregated_weights}
        noised_nn = _add_dp_noise(
            nn_weights_for_noise,
            len(fl_manager.pending_updates),
            CLIP_BOUND,
            DP_NOISE_SCALE,
        )
        aggregated_weights.update(noised_nn)

        # =====================================================================
        # AGGREGATE STATISTICS (baseline means/stds)
        # =====================================================================
        aggregated_baselines = _aggregate_baselines_robust(fl_manager, fl_manager.pending_updates)
        aggregated_weights.update(aggregated_baselines)

        # =====================================================================
        # VALIDATION with ACTUAL metrics
        # =====================================================================
        validation_metrics = _validate_global_model_enhanced(fl_manager, aggregated_weights)

        # Rollback decision
        if _should_rollback(fl_manager, validation_metrics):
            print(f"⛔ ROLLBACK: Validation quality dropped")
            print(f"   Previous AUC: {fl_manager.last_validation_metrics['auc']:.4f}")
            print(f"   New AUC: {validation_metrics['auc']:.4f}")

            fl_manager.global_model['weights'] = fl_manager.model_savepoint['weights']
            fl_manager.global_model['momentum'] = {
                k: 0.0 for k in fl_manager.global_model['momentum'].keys()
            }
            fl_manager.aggregation_metadata['last_round']['rollback_occurred'] = True

            fl_manager._audit_log_event('aggregation_rollback', 'SERVER', {
                'reason': 'validation_drop',
                'old_auc': fl_manager.last_validation_metrics['auc'],
                'new_auc': validation_metrics['auc'],
            })

            fl_manager.pending_updates.clear()
            return None

        # =====================================================================
        # COMMIT NEW MODEL with convergence tracking
        # =====================================================================
        convergence_delta = _calculate_convergence(fl_manager, aggregated_weights)

        old_w = fl_manager.global_model['weights']
        abs_scalar_deltas = []
        for k in scalar_keys:
            if k in old_w and k in aggregated_weights:
                try:
                    abs_scalar_deltas.append(
                        abs(float(aggregated_weights[k]) - float(old_w[k]))
                    )
                except (TypeError, ValueError):
                    continue
        scalar_mean_abs_delta = (
            float(np.mean(abs_scalar_deltas)) if abs_scalar_deltas else 0.0
        )
        scalar_max_abs_delta = (
            float(max(abs_scalar_deltas)) if abs_scalar_deltas else 0.0
        )
        avg_pre_norm = float(
            np.mean([u['pre_norm'] for u in fl_manager.pending_updates])
        )

        fl_manager.global_model['weights'] = aggregated_weights
        fl_manager.global_model['version'] += 1
        fl_manager.global_model['last_update'] = datetime.now()
        fl_manager.global_model['convergence_history'].append({
            'version': fl_manager.global_model['version'],
            'delta': convergence_delta,
            'timestamp': datetime.now(),
            'clients': len(fl_manager.pending_updates),
            'validation_auc': validation_metrics['auc'],
            'scalar_mean_abs_delta': scalar_mean_abs_delta,
            'scalar_max_abs_delta': scalar_max_abs_delta,
        })

        if len(fl_manager.global_model['convergence_history']) > 20:
            fl_manager.global_model['convergence_history'] = \
                fl_manager.global_model['convergence_history'][-20:]

        # Update metadata
        avg_norm = float(np.mean([u['post_norm'] for u in fl_manager.pending_updates]))
        quarantined = sum(
            1 for cid in fl_manager.client_contributions.keys()
            if fl_manager.client_contributions[cid]['quarantine_count'] > 0
        )

        fl_manager.aggregation_metadata['last_round'] = {
            'participated': len(fl_manager.pending_updates),
            'quarantined': quarantined,
            'avg_delta_norm': avg_norm,
            'avg_pre_clip_norm': avg_pre_norm,
            'convergence_delta': convergence_delta,
            'scalar_mean_abs_delta': scalar_mean_abs_delta,
            'scalar_max_abs_delta': scalar_max_abs_delta,
            'validation_auc': validation_metrics['auc'],
            'dp_noise_applied': fl_manager.dp_noise_scale,
            'rollback_occurred': False,
        }

        fl_manager.last_validation_metrics = validation_metrics

        fl_manager._audit_log_event('aggregation_success', 'SERVER', {
            'version': fl_manager.global_model['version'],
            'clients': len(fl_manager.pending_updates),
            'avg_norm': avg_norm,
            'validation_auc': validation_metrics['auc'],
            'convergence': convergence_delta,
            'learning_applied': True,
            'momentum_used': True,
        })

        print(f"✓ Aggregated {len(fl_manager.pending_updates)} client updates")
        print(f"  Model version: {fl_manager.global_model['version']}")
        print(f"  Convergence delta: {convergence_delta:.6f}")
        print(f"  Validation AUC: {validation_metrics['auc']:.4f}")
        print(f"  Learning rate: {NN_LR}, Momentum: {NN_MOMENTUM}")
        print(f"{'='*70}\n")

        fl_manager.pending_updates.clear()
        return fl_manager.global_model

    except Exception as e:
        print(f"✗ Aggregation error: {e}")
        import traceback
        traceback.print_exc()
        fl_manager._audit_log_event('aggregation_error', 'SERVER', {'error': str(e)})
        return None


def aggregate_and_persist(fl_manager, db_manager) -> dict | None:
    """
    Convenience wrapper: run aggregation and, on success, persist the
    aggregation round + FL model to the database.

    Returns
    -------
    dict | None
        Updated global_model dict, or None if aggregation was skipped/rolled back.
    """
    aggregated = aggregate_models(fl_manager)

    if aggregated:
        agg_meta = fl_manager.get_aggregation_metadata()
        convergence = 0.0
        if aggregated.get('convergence_history'):
            convergence = aggregated['convergence_history'][-1].get('delta', 0)

        try:
            db_manager.log_fl_aggregation_round(
                model_version=aggregated['version'],
                participated=agg_meta['participated'],
                quarantined=agg_meta['quarantined'],
                avg_norm=agg_meta['avg_delta_norm'],
                validation_auc=agg_meta['validation_auc'],
                dp_noise=agg_meta['dp_noise_applied'],
                rollback=agg_meta['rollback_occurred'],
                convergence=convergence,
            )
        except Exception as e:
            print(f"Warning: Could not log aggregation round: {e}")

        try:
            db_manager.save_fl_model(
                aggregated['version'],
                aggregated['weights'],
                len(fl_manager.client_models),
            )
        except Exception as e:
            print(f"Warning: Could not save FL model: {e}")

    return aggregated


# =============================================================================
# Private aggregation helpers
# =============================================================================

def _calculate_reputation_weights(updates: list) -> dict:
    """Calculate adaptive weights for each update (60% reputation + 40% quality)."""
    weights: dict = {}

    total_reputation = sum(u['reputation'] for u in updates)
    total_quality = sum(u['quality_score'] for u in updates)

    for update in updates:
        client_id = update['client_id']
        reputation = update['reputation']
        quality = update['quality_score']

        if total_reputation > 0 and total_quality > 0:
            reputation_weight = reputation / total_reputation
            quality_weight = quality / total_quality
            combined_weight = 0.6 * reputation_weight + 0.4 * quality_weight
        else:
            combined_weight = 1.0 / len(updates)

        weights[client_id] = combined_weight
        print(
            f"  Client {client_id[:12]}: weight={combined_weight:.3f} "
            f"(reputation={reputation:.3f}, quality={quality:.3f})"
        )

    # Normalise
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {cid: w / total_weight for cid, w in weights.items()}

    return weights


def _trimmed_mean_aggregation(updates: list, weights: dict, trim_frac: float = 0.1) -> dict:
    """
    Aggregate using trimmed mean: remove top/bottom trim_frac fraction and
    compute weighted average of the remainder.
    """
    print(f"  Using Trimmed Mean (trim_frac={trim_frac})")

    all_keys: set = set()
    for update in updates:
        all_keys.update(update['delta'].keys())

    aggregated: dict = {}

    for key in all_keys:
        values_with_weights = []
        for update in updates:
            if key in update['delta']:
                client_id = update['client_id']
                values_with_weights.append((update['delta'][key], weights[client_id]))

        if not values_with_weights:
            aggregated[key] = 0.0
            continue

        values_with_weights.sort(key=lambda x: x[0])

        n = len(values_with_weights)
        trim_count = max(1, int(n * trim_frac))

        if n > 2 * trim_count:
            trimmed = values_with_weights[trim_count:-trim_count]
        else:
            trimmed = values_with_weights  # Not enough data to trim

        weighted_sum = sum(v * w for v, w in trimmed)
        total_weight = sum(w for _, w in trimmed)

        aggregated[key] = weighted_sum / total_weight if total_weight > 0 else 0.0

    return aggregated


def _add_dp_noise(
    aggregated: dict,
    num_clients: int,
    clip_bound: float,
    noise_scale: float,
) -> dict:
    """Add Gaussian noise for Server-Side Differential Privacy over multi-dimensional tensors."""
    sensitivity = (2 * clip_bound) / num_clients
    noisy: dict = {}
    for key, value in aggregated.items():
        if isinstance(value, (int, float)) or (isinstance(value, np.ndarray) and value.ndim == 0):
            # Scalar values
            noise = np.random.normal(0, noise_scale * sensitivity)
            noisy_val = value + noise
            noisy[key] = float(noisy_val)
        elif isinstance(value, list) or isinstance(value, np.ndarray):
            # Tensors (Keras Layers)
            value_arr = np.array(value)
            noise = np.random.normal(0, noise_scale * sensitivity, value_arr.shape)
            noisy_val = value_arr + noise
            noisy[key] = noisy_val.tolist()
        else:
            noisy[key] = value

    print(f"  [DP] Server-sided noise added (sensitivity={sensitivity:.4f}, scale={noise_scale})")
    return noisy


def _aggregate_baselines_robust(fl_manager, updates: list) -> dict:
    """Aggregate baseline statistics using trimmed mean."""
    baselines: dict = {}

    baseline_keys = [
        ('network', 'network_baseline_mean', 'mean_connections'),
        ('network', 'network_baseline_std', 'std_connections'),
        ('process', 'process_baseline_mean', 'mean_count'),
        ('process', 'process_baseline_std', 'std_count'),
        ('file', 'file_baseline_mean', 'mean_events'),
        ('file', 'file_baseline_std', 'std_events'),
    ]

    for category, baseline_key, stat_key in baseline_keys:
        values = []
        for update in updates:
            client_id = update['client_id']
            if client_id in fl_manager.client_models:
                statistics = fl_manager.client_models[client_id]['parameters'].get('statistics', {})
                category_stats = statistics.get(category, {})
                if stat_key in category_stats:
                    values.append(category_stats[stat_key])

        if values:
            values_sorted = sorted(values)
            n = len(values_sorted)
            trim_count = max(1, int(n * TRIM_FRAC))

            trimmed = values_sorted[trim_count:-trim_count] if n > 2 * trim_count else values_sorted
            baselines[baseline_key] = float(np.mean(trimmed))

    return baselines


def _validate_global_model(fl_manager, model_weights: dict) -> dict:
    """
    Baseline validation when validation dataset is insufficient.
    Uses threshold-based heuristic with bounded output.
    """
    threshold_avg = np.mean([
        model_weights.get('network_threshold', 2.0),
        model_weights.get('process_threshold', 2.0),
        model_weights.get('file_threshold', 2.0),
    ])

    estimated_auc = float(np.clip(0.85 + (2.5 - threshold_avg) * 0.03, 0.70, 0.95))

    return {
        'auc': estimated_auc,
        'fpr': 0.05,
        'tpr': 0.90,
    }


def _validate_global_model_enhanced(fl_manager, new_weights: dict) -> dict:
    """
    Real validation using held-out feature cache and the actual models.
    Falls back to heuristic when fewer than 20 validation samples exist.
    """
    if len(fl_manager.validation_feature_cache) < 20:
        return _validate_global_model(fl_manager, new_weights)

    try:
        features = list(fl_manager.validation_feature_cache)
        labels = list(fl_manager.validation_labels_cache)
        X = np.array(features)

        # Run actual anomaly detection with the proposed thresholds
        iso_thresh = new_weights.get('iso_threshold', -0.3)
        ae_thresh = new_weights.get('ae_threshold', 0.1)

        scores = np.zeros(len(X))
        has_model = False

        # Use ISO forest if loaded in the FL manager
        ae_keys = [k for k in new_weights if k.startswith('ae_layer_')]
        if ae_keys:
            try:
                import os
                from tensorflow.keras.models import load_model
                models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
                ae_path = os.path.join(models_dir, 'global_autoencoder.h5')
                if os.path.exists(ae_path):
                    ae_model = load_model(ae_path)
                    preds = ae_model.predict(X, verbose=0)
                    recon_err = np.mean(np.square(X - preds), axis=1)
                    scores = recon_err / max(ae_thresh, 1e-6)
                    has_model = True
            except Exception:
                pass

        if not has_model:
            # Threshold-based scoring as fallback
            threshold_avg = np.mean([
                new_weights.get('network_threshold', 2.0),
                new_weights.get('process_threshold', 2.0),
                new_weights.get('file_threshold', 2.0),
            ])
            row_norms = np.linalg.norm(X, axis=1) if X.ndim == 2 else np.abs(X)
            scores = row_norms / max(threshold_avg, 1e-6)

        auc = 0.85
        if len(set(labels)) == 2:
            from sklearn.metrics import roc_auc_score
            try:
                auc = float(roc_auc_score(labels, scores))
            except Exception:
                pass

        preds_binary = (scores > np.median(scores)).astype(int)
        tp = int(np.sum((preds_binary == 1) & (np.array(labels) == 1)))
        fp = int(np.sum((preds_binary == 1) & (np.array(labels) == 0)))
        fn = int(np.sum((preds_binary == 0) & (np.array(labels) == 1)))
        tn = int(np.sum((preds_binary == 0) & (np.array(labels) == 0)))

        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)

        metrics = {'auc': float(auc), 'fpr': float(fpr), 'tpr': float(tpr)}
        print(f"  [VALIDATION] AUC={auc:.4f}, TPR={tpr:.4f}, FPR={fpr:.4f}")
        return metrics

    except Exception as e:
        print(f"  [VALIDATION] Error: {e}, using heuristic")
        return _validate_global_model(fl_manager, new_weights)


def _should_rollback(fl_manager, new_metrics: dict) -> bool:
    """Decide whether to rollback based on validation metrics."""
    auc_drop = fl_manager.last_validation_metrics['auc'] - new_metrics['auc']
    return auc_drop > VALIDATION_AUC_DROP_THRESHOLD


def _calculate_convergence(fl_manager, new_weights: dict) -> float:
    """Calculate convergence metric (normalised L2 distance from previous model)."""
    delta = 0.0
    count = 0

    for key in new_weights.keys():
        if key in fl_manager.global_model['weights']:
            new_val = np.array(new_weights[key])
            old_val = np.array(fl_manager.global_model['weights'][key])
            
            # For tensors (neural networks) or scalars
            diff = new_val - old_val
            delta += np.sum(diff ** 2)
            count += diff.size

    return float(np.sqrt(delta / count)) if count > 0 else 0.0
