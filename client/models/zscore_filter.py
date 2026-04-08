"""FortifAI — Z-Score statistical anomaly filter."""

import numpy as np

try:
    from shared.detection_config import (
        ZSCORE_THRESHOLD, ZSCORE_MIN_FLAGS,
        ZSCORE_PER_FEATURE_CAP, ZSCORE_MAX_SCORE,
    )
except ImportError:
    ZSCORE_THRESHOLD = 3.5
    ZSCORE_MIN_FLAGS = 2
    ZSCORE_PER_FEATURE_CAP = 2.0
    ZSCORE_MAX_SCORE = 10.0


class ZScoreFilter:
    """Statistical Z-Score filter for pre-normalized feature vectors."""

    def __init__(self, threshold=ZSCORE_THRESHOLD, min_flags=ZSCORE_MIN_FLAGS):
        self.threshold = threshold
        self.min_flags = min_flags

    def evaluate(self, feature_vector, feature_names):
        """
        Evaluate if a feature vector contains extreme Z-score outliers.
        feature_vector values are expected to already be Z-normalized.
        Returns: (normalized_score, list_of_flagged_feature_names)
        """
        flags = []
        score = 0.0

        for fname, fval in zip(feature_names, feature_vector):
            if abs(fval) > self.threshold:
                flags.append(fname)
                score += min(abs(fval) / self.threshold, ZSCORE_PER_FEATURE_CAP)

        normalized_score = 0.0
        if len(flags) >= self.min_flags:
            normalized_score = min(score / max(len(flags), 1), ZSCORE_MAX_SCORE)

        return normalized_score, flags
