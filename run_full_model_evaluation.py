#!/usr/bin/env python3
"""
run_full_model_evaluation_fixed.py

Corrected evaluation pipeline that uses:
 - ClientAgent (client_agent.ClientAgent)
 - FederatedLearningManager (admin_server.FederatedLearningManager) if available

Saves plots and metrics to /data/model_reports/
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys

from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix

# Update imports to use the correct files
try:
    from client_agentAI import ClientAgent
except Exception as e:
    raise ImportError(f"Unable to import ClientAgent from client_agentAI.py: {e}")

# admin_serverAI optional
try:
    from admin_serverAI import FederatedLearningManager
    HAS_SERVER = True
except Exception:
    FederatedLearningManager = None
    HAS_SERVER = False

SAVE_DIR = os.path.join(os.path.dirname(__file__), 'data', 'model_reports')
os.makedirs(SAVE_DIR, exist_ok=True)

def save_plot(fig, name):
    path = os.path.join(SAVE_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"[SAVED] {path}")

def safe_auc(labels, scores):
    try:
        fpr, tpr, _ = roc_curve(labels, scores)
        return auc(fpr, tpr), (fpr, tpr)
    except Exception:
        return None, (None, None)

def evaluate_client(client_id, client):
    print(f"\n=== Evaluating Client: {client_id} ===")
    detector = client.anomaly_detector
    feature_manager = client.feature_manager

    # Define client-specific save directory early
    client_save_dir = os.path.join(SAVE_DIR, f"client_{client_id}")
    os.makedirs(client_save_dir, exist_ok=True)

    # Get the feature matrix and feature names
    fm, feature_names = feature_manager.get_feature_matrix()
    if fm is None or feature_names is None:
        print(f"[WARN] Client {client_id} feature buffer is empty. Generating synthetic telemetry data.")
        num_samples = 100  # Number of synthetic samples
        num_features = 10  # Number of synthetic features
        fm = np.random.rand(num_samples, num_features)  # Random feature matrix
        feature_names = [f"feature_{i}" for i in range(num_features)]
        print(f"[INFO] Generated synthetic feature matrix for Client {client_id}: shape={fm.shape}, features={len(feature_names)}")
    X = np.array(fm)
    print(f"[OK] Loaded feature matrix for Client {client_id}: shape={X.shape}, features={len(feature_names)}")

    # Labels: try to load /data/labels.json (optional). Otherwise synthesize.
    labels_path = "/data/labels.json"
    if os.path.exists(labels_path):
        with open(labels_path, "r") as f:
            labels = np.array(json.load(f))
        print(f"[OK] Loaded {len(labels)} ground truth labels.")
    else:
        # heuristic (optional) - mark last 10% as anomalies just to produce metrics
        n = len(X)
        split = int(n * 0.9)
        labels = np.zeros(n, dtype=int)
        if n - split > 0:
            labels[split:] = 1
        print(f"[INFO] No labels file found, generated synthetic labels (last 10% anomalies).")

    # Evaluate IsolationForest if present
    iso_scores = None
    if hasattr(detector, "isolation_forest") and detector.isolation_forest is not None:
        try:
            raw_iso_scores = detector.isolation_forest.score_samples(X)
            # Higher anomaly -> larger positive score (normalize so larger = more anomalous)
            iso_scores = -raw_iso_scores
            iso_scores = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-9)
            print("[OK] IsolationForest scoring complete.")
            fpr, tpr, _ = roc_curve(labels, iso_scores)
            fig = plt.figure()
            plt.plot(fpr, tpr, label=f"AUC={auc(fpr, tpr):.4f}")
            plt.title("IsolationForest ROC")
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.legend()
            save_plot(fig, "iso_roc.png")
            plt.close(fig)
        except Exception as e:
            print(f"[WARN] IsolationForest evaluation failed: {e}")
            iso_scores = None
    else:
        print("[INFO] No IsolationForest model available on client.")

    # Evaluate Autoencoder if present and trained
    ae_scores = None
    if hasattr(detector, "autoencoder") and detector.autoencoder is not None and detector.autoencoder.is_trained:
        try:
            is_anom, recon_errors = detector.autoencoder.detect_anomaly(X)
            ae_scores = np.array(recon_errors)
            print("[OK] Autoencoder reconstruction errors computed.")
            fig = plt.figure()
            plt.plot(recon_errors, label="Reconstruction error")
            thresh = getattr(detector.autoencoder, "threshold", None)
            if thresh is not None:
                plt.axhline(thresh, color="r", linestyle="--", label=f"Threshold={thresh:.6f}")
            plt.title("Autoencoder reconstruction errors")
            plt.legend()
            save_plot(fig, "ae_recon_curve.png")
            plt.close(fig)

            fpr, tpr, _ = roc_curve(labels, ae_scores)
            fig = plt.figure()
            plt.plot(fpr, tpr, label=f"AUC={auc(fpr, tpr):.4f}")
            plt.title("Autoencoder ROC")
            plt.legend()
            save_plot(fig, "ae_roc.png")
            plt.close(fig)
        except Exception as e:
            print(f"[WARN] Autoencoder evaluation failed: {e}")
            ae_scores = None
    else:
        print("[INFO] Autoencoder missing or not trained on client.")

    # Evaluate Ensemble: call detector.detect_anomaly_ensemble() for each row
    print("\n=== Ensemble scoring ===")
    ensemble_scores = []
    ensemble_infos = []
    for row in X:
        try:
            is_anom, info = detector.detect_anomaly_ensemble(row, feature_names)
            ensemble_scores.append(info.get("ensemble_score", 0.0))
            ensemble_infos.append(info)
        except Exception as e:
            # If detection errors for some rows, append 0
            ensemble_scores.append(0.0)
            ensemble_infos.append({})
            print(f"[WARN] Ensemble scoring error for one sample: {e}")
    ensemble_scores = np.array(ensemble_scores)

    # Save ensemble ROC/PR with detailed metrics
    try:
        fpr, tpr, _ = roc_curve(labels, ensemble_scores)
        ens_auc = auc(fpr, tpr)
        fig = plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={ens_auc:.4f}")
        plt.title(f"Client {client_id} - Ensemble ROC")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
        save_plot(fig, os.path.join(client_save_dir, "ensemble_roc.png"))
        plt.close(fig)

        precision, recall, _ = precision_recall_curve(labels, ensemble_scores)
        fig = plt.figure()
        plt.plot(recall, precision)
        plt.title(f"Client {client_id} - Ensemble Precision-Recall")
        save_plot(fig, os.path.join(client_save_dir, "ensemble_pr.png"))
        plt.close(fig)

        print(f"[OK] Client {client_id} - Ensemble AUC={ens_auc:.4f}")
    except Exception as e:
        print(f"[WARN] Client {client_id} - Ensemble ROC generation failed: {e}")

    # Confusion Matrix with detailed visualization
    try:
        thresh = 5.0
        preds = (ensemble_scores > thresh).astype(int)
        cm = confusion_matrix(labels, preds)
        fig = plt.figure(figsize=(4,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Client {client_id} - Confusion Matrix (threshold {thresh})")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        save_plot(fig, os.path.join(client_save_dir, "confusion_matrix.png"))
        plt.close(fig)
        print(f"[OK] Client {client_id} - Confusion matrix saved.")
    except Exception as e:
        print(f"[WARN] Client {client_id} - Confusion matrix generation failed: {e}")

    # Save summary metrics to JSON
    metrics = {
        "samples": int(len(X)),
        "features": int(len(feature_names)),
        "ensemble_auc": None,
        "iso_auc": None,
        "ae_auc": None,
        "generated_at": datetime.utcnow().isoformat() + "Z"
    }

    try:
        if 'fpr' in locals() and 'tpr' in locals():
            metrics["ensemble_auc"] = float(auc(fpr, tpr))
    except Exception:
        pass

    try:
        if iso_scores is not None:
            iso_auc, _ = safe_auc(labels, iso_scores)
            metrics["iso_auc"] = iso_auc
    except Exception:
        pass

    try:
        if ae_scores is not None:
            ae_auc, _ = safe_auc(labels, ae_scores)
            metrics["ae_auc"] = ae_auc
    except Exception:
        pass

    # Load server global model if available
    global_model = None
    if HAS_SERVER and FederatedLearningManager is not None:
        try:
            flm = FederatedLearningManager()
            global_model = flm.get_global_model()
            print(f"[OK] Loaded server global model for Client {client_id} (metadata present).")
        except Exception as e:
            print(f"[WARN] Could not load server model for Client {client_id}: {e}")
            global_model = None
    else:
        print(f"[INFO] admin_server not available or failed to import for Client {client_id}. Skipping server model fetch.")

    # Server global model metadata (if loaded)
    if global_model is not None:
        metrics["server_model_version"] = global_model.get("version", None) if isinstance(global_model, dict) else None

    with open(os.path.join(client_save_dir, "metrics_summary.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVED] metrics_summary.json -> {metrics}")

    # Save raw arrays (optionally)
    np.save(os.path.join(client_save_dir, "ensemble_scores.npy"), ensemble_scores)
    if iso_scores is not None:
        np.save(os.path.join(client_save_dir, "iso_scores.npy"), iso_scores)
    if ae_scores is not None:
        np.save(os.path.join(client_save_dir, "ae_scores.npy"), ae_scores)

    print(f"[INFO] Evaluation results for Client {client_id} saved in: {client_save_dir}")

def main():
    print("\n=== LOAD CLIENT AGENT (no background threads) ===")
    client_ids = ["client_1", "client_2", "client_3"]  # Example client IDs

    for client_id in client_ids:
        client = ClientAgent()  # Create a new ClientAgent for each client
        evaluate_client(client_id, client)

    print("\n=== EVALUATION COMPLETE FOR ALL CLIENTS ===")

if __name__ == "__main__":
    main()
