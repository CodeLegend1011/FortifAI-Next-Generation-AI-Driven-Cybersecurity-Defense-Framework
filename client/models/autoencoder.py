"""
FortifAI Client — AutoencoderAnomalyDetector

Deep autoencoder for detecting anomalies via reconstruction error.
Architecture MUST match the global model built in admin/fl/train_global_initial.py:
  input → 128(relu) → Dropout(0.2) → 64(relu) → Dropout(0.2) → 32(bottleneck)
        → 64(relu) → Dropout(0.2) → 128(relu) → Dropout(0.2) → input(linear)

Weight keys use the server convention: ae_layer_{i}_weight_{j}
so that FL weight exchange works bidirectionally.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

try:
    from shared.detection_config import (
        AE_INPUT_DIM, AE_LATENT_DIM, AE_LAYER_SIZES, AE_DROPOUT,
        AE_EPOCHS, AE_BATCH_SIZE, AE_PATIENCE, AE_VALIDATION_SPLIT,
        AE_THRESHOLD_PERCENTILE,
    )
except ImportError:
    AE_INPUT_DIM = 99
    AE_LATENT_DIM = 32
    AE_LAYER_SIZES = [128, 64]
    AE_DROPOUT = 0.2
    AE_EPOCHS = 10
    AE_BATCH_SIZE = 16
    AE_PATIENCE = 3
    AE_VALIDATION_SPLIT = 0.2
    AE_THRESHOLD_PERCENTILE = 90


class AutoencoderAnomalyDetector:
    """Deep autoencoder for detecting anomalies via reconstruction error."""

    DEFAULT_INPUT_DIM = AE_INPUT_DIM
    DEFAULT_LATENT_DIM = AE_LATENT_DIM

    def __init__(self, input_dim=None, latent_dim=None):
        self.input_dim = input_dim or self.DEFAULT_INPUT_DIM
        self.latent_dim = latent_dim or self.DEFAULT_LATENT_DIM
        self.model = None
        self.threshold = None
        self.scaler = StandardScaler()
        self.is_trained = False

        self.build_model()

    def build_model(self):
        """
        Build autoencoder matching the global architecture from
        train_global_initial.py — 128→64→32→64→128 with Dropout(0.2).
        """
        inp = keras.Input(shape=(self.input_dim,))
        x = layers.Dense(AE_LAYER_SIZES[0], activation='relu')(inp)
        x = layers.Dropout(AE_DROPOUT)(x)
        x = layers.Dense(AE_LAYER_SIZES[1], activation='relu')(x)
        x = layers.Dropout(AE_DROPOUT)(x)
        encoded = layers.Dense(self.latent_dim, activation='relu', name='bottleneck')(x)
        x = layers.Dense(AE_LAYER_SIZES[1], activation='relu')(encoded)
        x = layers.Dropout(AE_DROPOUT)(x)
        x = layers.Dense(AE_LAYER_SIZES[0], activation='relu')(x)
        x = layers.Dropout(AE_DROPOUT)(x)
        out = layers.Dense(self.input_dim, activation='linear')(x)

        self.model = keras.Model(inp, out, name='fortifai_autoencoder')
        self.model.compile(optimizer='adam', loss='mse')

    def train(self, X, epochs=AE_EPOCHS, batch_size=AE_BATCH_SIZE):
        """Train autoencoder on normal data."""
        if len(X) < 20:
            print("  [AE] Insufficient data for training (need >= 20 samples)")
            return False

        try:
            X_clean = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)

            print(f"  [AE] Fitting scaler on {len(X_clean)} samples...")
            self.scaler.fit(X_clean)

            if not hasattr(self.scaler, 'mean_') or not hasattr(self.scaler, 'scale_'):
                print("  [AE] Scaler fitting failed")
                return False

            X_scaled = self.scaler.transform(X_clean)

            from tensorflow.keras.callbacks import EarlyStopping
            early_stop = EarlyStopping(monitor='val_loss', patience=AE_PATIENCE,
                                       restore_best_weights=True)

            history = self.model.fit(
                X_scaled, X_scaled,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                validation_split=AE_VALIDATION_SPLIT,
                callbacks=[early_stop],
            )

            X_pred = self.model.predict(X_scaled, verbose=0)
            recon_errors = np.mean(np.square(X_scaled - X_pred), axis=1)

            self.threshold = float(np.percentile(recon_errors, AE_THRESHOLD_PERCENTILE))
            self.is_trained = True

            final_loss = history.history['loss'][-1]
            val_loss = history.history.get('val_loss', [final_loss])[-1]
            print(f"  [AE] Training complete: loss={final_loss:.6f}, "
                  f"val_loss={val_loss:.6f}, threshold={self.threshold:.6f}")
            return True

        except Exception as e:
            print(f"  [AE] Training error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def predict(self, X):
        """Return per-sample reconstruction errors."""
        if not self.is_trained:
            return None

        try:
            if not hasattr(self.scaler, 'mean_') or not hasattr(self.scaler, 'scale_'):
                print("  [AE] Warning: Scaler not fitted, fitting now...")
                self.scaler.fit(X)

            X_scaled = self.scaler.transform(X)
            X_pred = self.model.predict(X_scaled, verbose=0)
            return np.mean(np.square(X_scaled - X_pred), axis=1)
        except Exception as e:
            print(f"  [AE] Prediction error: {e}")
            return None

    def detect_anomaly(self, X):
        """Detect if samples are anomalies based on reconstruction error."""
        recon_errors = self.predict(X)
        if recon_errors is None:
            return None, None

        if self.threshold is None:
            self.threshold = float(np.percentile(recon_errors, 95))

        is_anomaly = recon_errors > self.threshold
        return is_anomaly, recon_errors

    # ── FL weight exchange (server-compatible keys) ────────────────────────

    def get_weights(self):
        """
        Extract model weights for federated learning.
        Uses ae_layer_{i}_weight_{j} key convention matching fl_server.py.
        """
        if self.model is None:
            return None

        weights_dict = {}

        for i, layer in enumerate(self.model.layers):
            layer_weights = layer.get_weights()
            for j, w in enumerate(layer_weights):
                weights_dict[f'ae_layer_{i}_weight_{j}'] = w.tolist()

        if hasattr(self.scaler, 'mean_') and hasattr(self.scaler, 'scale_'):
            weights_dict['scaler_mean'] = self.scaler.mean_.tolist()
            weights_dict['scaler_scale'] = self.scaler.scale_.tolist()

        if self.threshold is not None:
            weights_dict['ae_threshold'] = float(self.threshold)

        weights_dict['input_dim'] = self.input_dim

        return weights_dict

    def set_weights(self, weights_dict):
        """
        Set model weights from federated learning.
        Accepts both ae_layer_{i}_weight_{j} (server) and
        layer_{i}_kernel (legacy) key formats.
        """
        try:
            for i, layer in enumerate(self.model.layers):
                current_weights = layer.get_weights()
                if not current_weights:
                    continue

                new_weights = []
                for j in range(len(current_weights)):
                    key = f'ae_layer_{i}_weight_{j}'
                    legacy_key = f'layer_{i}_kernel' if j == 0 else f'layer_{i}_bias'

                    if key in weights_dict:
                        new_weights.append(np.array(weights_dict[key]))
                    elif legacy_key in weights_dict:
                        new_weights.append(np.array(weights_dict[legacy_key]))
                    else:
                        new_weights.append(current_weights[j])

                try:
                    layer.set_weights(new_weights)
                except ValueError as ve:
                    print(f"  [AE] Shape mismatch layer {i}: {ve}")

            if 'scaler_mean' in weights_dict and 'scaler_scale' in weights_dict:
                self.scaler.mean_ = np.array(weights_dict['scaler_mean'])
                self.scaler.scale_ = np.array(weights_dict['scaler_scale'])
                self.scaler.n_features_in_ = len(self.scaler.mean_)
                self.scaler.var_ = self.scaler.scale_ ** 2
                self.scaler.n_samples_seen_ = np.full(len(self.scaler.mean_), 100, dtype=np.int64)
                print("  [AE] Scaler parameters restored")

            if 'ae_threshold' in weights_dict:
                self.threshold = float(weights_dict['ae_threshold'])
                print(f"  [AE] Threshold restored: {self.threshold:.6f}")
            elif 'threshold' in weights_dict:
                self.threshold = float(weights_dict['threshold'])

            self.is_trained = True
            print("  [AE] Weights updated from global model")

        except Exception as e:
            print(f"  [AE] Error setting weights: {e}")
            import traceback
            traceback.print_exc()
