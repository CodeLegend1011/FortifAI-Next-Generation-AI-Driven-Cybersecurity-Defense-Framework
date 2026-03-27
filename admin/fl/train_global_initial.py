import os
import numpy as np
import pandas as pd
import pickle

# Windows: "DLL load failed" for TensorFlow usually means missing MSVC runtime.
# Install "Microsoft Visual C++ Redistributable" 2015–2022 (x64), then: pip install -U tensorflow
# https://www.tensorflow.org/install/pip
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

DATASETS_DIR = r"e:\Migration\Determination2.0\FortifAI\datasets"
MODELS_DIR = r"e:\Migration\Determination2.0\FortifAI\admin\models"

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

def build_combined_dataset():
    print("Loading component datasets...")
    df_net = pd.read_csv(os.path.join(DATASETS_DIR, "network_traffic_cicids2017_derived.csv"))
    df_proc = pd.read_csv(os.path.join(DATASETS_DIR, "processes_beth_adfawd_derived.csv"))
    df_file = pd.read_csv(os.path.join(DATASETS_DIR, "filesystem_threats_derived.csv"))
    df_user = pd.read_csv(os.path.join(DATASETS_DIR, "user_activity_cmucert_derived.csv"))

    # Unified Labels: If ANY subsystem has an anomaly, the row is anomalous
    # (Since 20% of each are anomalous and they are randomly generated independent rows, 
    # the unified anomaly rate might be high, around 1 - (0.8**4) = 59%. That's okay for testing.)
    unified_labels = (df_net['Label'] | df_proc['Label'] | df_file['Label'] | df_user['Label']).values

    # Drop label columns to form pure feature matrix
    df_net = df_net.drop(columns=['Label', 'Attack_Type'])
    df_proc = df_proc.drop(columns=['Label', 'Attack_Type'])
    df_file = df_file.drop(columns=['Label', 'Attack_Type'])
    df_user = df_user.drop(columns=['Label', 'Attack_Type'])

    # Horizontally concatenate features
    combined_df = pd.concat([df_net, df_proc, df_file, df_user], axis=1)
    
    # Save the ordering of feature names so the client can conform to it!
    feature_names = list(combined_df.columns)
    with open(os.path.join(MODELS_DIR, 'global_feature_schema.json'), 'w') as f:
        import json
        json.dump(feature_names, f)

    print(f"Combined Dataset Shape: {combined_df.shape} (Features: {len(feature_names)})")
    
    X = combined_df.values
    y = unified_labels
    
    return X, y, feature_names

def train_global_models():
    X, y, feature_names = build_combined_dataset()
    input_dim = X.shape[1]

    # Pre-train ONLY on Benign data (y == 0) for Autoencoder and Forest
    # This prevents the models from thinking existing exploits are "normal"
    X_benign = X[y == 0]
    print(f"Training on {len(X_benign)} pure benign samples...")

    # 1. Scale data
    scaler = StandardScaler()
    X_benign_scaled = scaler.fit_transform(X_benign)
    
    # Save scaler
    with open(os.path.join(MODELS_DIR, 'global_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    # 2. Train Z-Score Baseline Filters
    z_baselines = {
        'means': np.mean(X_benign_scaled, axis=0).tolist(),
        'stds': np.std(X_benign_scaled, axis=0).tolist()
    }
    with open(os.path.join(MODELS_DIR, 'global_z_baselines.json'), 'w') as f:
        import json
        json.dump(z_baselines, f)

    # 3. Train Isolation Forest
    print("Training Global Isolation Forest...")
    iso_forest = IsolationForest(
        n_estimators=100, 
        max_samples='auto', 
        contamination=0.01, # Expecting very little contamination in benign set
        random_state=42
    )
    iso_forest.fit(X_benign_scaled)
    
    with open(os.path.join(MODELS_DIR, 'global_iso_forest.pkl'), 'wb') as f:
        pickle.dump(iso_forest, f)

    # 4. Train Keras Autoencoder
    print("Building Global Autoencoder Architecture...")
    
    inputs = layers.Input(shape=(input_dim,))
    # Encoder
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    # Bottleneck
    encoded = layers.Dense(32, activation="relu", name="bottleneck")(x)
    # Decoder
    x = layers.Dense(64, activation="relu")(encoded)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(input_dim, activation="linear")(x) # Linear for scaled numeric reconstruction
    
    autoencoder = keras.Model(inputs, outputs, name="global_autoencoder")
    autoencoder.compile(optimizer='adam', loss='mse')
    
    print("Training Global Autoencoder...")
    # Early stopping to prevent overfitting
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True
    )
    
    autoencoder.fit(
        X_benign_scaled, X_benign_scaled,
        epochs=50,
        batch_size=256,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Save full Keras H5 model
    autoencoder.save(os.path.join(MODELS_DIR, 'global_autoencoder.h5'))
    
    print("\n" + "="*50)
    print("✅ GLOBAL MODELS SUCCESSFULLY TRAINED AND PACKAGED")
    print(f"Saved to: {MODELS_DIR}")
    print("These models serve as the foundational FedAvg weights.")
    print("="*50)

if __name__ == '__main__':
    train_global_models()
