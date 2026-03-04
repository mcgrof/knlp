"""Future-Q prediction: linear autoregressive predictor on recent Q states."""

import numpy as np


class LinearQPredictor:
    """Simple linear autoregressive Q predictor.

    Predicts Q_{t+h} from recent Q history using least-squares fit.
    """

    def __init__(self, window_size: int = 4, horizon: int = 1):
        self.window_size = window_size
        self.horizon = horizon
        self.history = []

    def update(self, q_vector: np.ndarray):
        """Add a Q observation."""
        self.history.append(q_vector.copy())

    def predict(self) -> np.ndarray:
        """Predict Q at t+horizon using linear extrapolation.

        Falls back to last Q if insufficient history.
        """
        if len(self.history) < 2:
            return self.history[-1] if self.history else None

        # Use last window_size observations
        recent = self.history[-self.window_size :]
        n = len(recent)

        if n < 2:
            return recent[-1]

        # Simple linear extrapolation: fit line through recent Qs
        # Q_predicted = Q_last + horizon * (Q_last - Q_first) / (n - 1)
        delta = (recent[-1] - recent[0]) / (n - 1)
        predicted = recent[-1] + self.horizon * delta

        # Normalize per-layer
        for l in range(predicted.shape[0]):
            norm = np.linalg.norm(predicted[l])
            if norm > 1e-8:
                predicted[l] = predicted[l] / norm

        return predicted

    def reset(self):
        """Clear history."""
        self.history = []
