from sklearn.ensemble import IsolationForest

try:
    from shared.detection_config import IF_CONTAMINATION, IF_N_ESTIMATORS, IF_RANDOM_STATE
except ImportError:
    IF_CONTAMINATION = 0.05
    IF_N_ESTIMATORS = 200
    IF_RANDOM_STATE = 42

class IsolationForestAnomalyDetector:
    """Wrapper for SKLearn Isolation Forest"""
    def __init__(self, contamination=IF_CONTAMINATION, n_estimators=IF_N_ESTIMATORS):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=IF_RANDOM_STATE,
            n_jobs=-1
        )
        self.is_trained = False

    def train(self, data):
        try:
            self.model.fit(data)
            self.is_trained = True
            return True
        except Exception as e:
            print(f"[ISO] Training error: {e}")
            return False

    def predict(self, data):
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        return self.model.predict(data)

    def score_samples(self, data):
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        return self.model.score_samples(data)
