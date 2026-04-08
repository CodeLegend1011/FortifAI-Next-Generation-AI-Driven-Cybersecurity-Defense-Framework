"""FortifAI — One-Class SVM anomaly detector wrapper."""

from sklearn.svm import OneClassSVM
import numpy as np

try:
    from shared.detection_config import OCSVM_KERNEL, OCSVM_NU, OCSVM_GAMMA
except ImportError:
    OCSVM_KERNEL = 'rbf'
    OCSVM_NU = 0.05
    OCSVM_GAMMA = 'scale'


class OneClassSVMAnomalyDetector:
    """Wrapper for SKLearn One-Class SVM with training and inference."""

    def __init__(self, kernel=OCSVM_KERNEL, nu=OCSVM_NU, gamma=OCSVM_GAMMA):
        self.model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
        self.is_trained = False

    def train(self, data):
        try:
            clean = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)
            self.model.fit(clean)
            self.is_trained = True
            return True
        except Exception as e:
            print(f"[OCSVM] Training error: {e}")
            return False

    def predict(self, data):
        if not self.is_trained:
            raise RuntimeError("OCSVM not trained")
        return self.model.predict(data)

    def decision_function(self, data):
        if not self.is_trained:
            raise RuntimeError("OCSVM not trained")
        return self.model.decision_function(data)

    def score_samples(self, data):
        return self.decision_function(data)
