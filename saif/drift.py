"""Clean-data drift reference (pure numpy; no torch dependency so
analysis scripts run on CPU-only machines)."""
from __future__ import annotations

import numpy as np


class DriftReference:
    """Clean-data reference manifold: per-dimension (mu, sigma) of the final
    hidden state, fitted ONLY on the clean calibration split (label-free).
    d(x) = sqrt(mean_j ((h_j - mu_j)/sigma_j)^2)."""

    def __init__(self):
        self.mu = None
        self.sigma = None

    def fit(self, H: np.ndarray):                 # H: [n, d] clean hidden states
        self.mu = H.mean(axis=0)
        self.sigma = H.std(axis=0) + 1e-6

    def distance(self, H: np.ndarray) -> np.ndarray:
        z = (H - self.mu) / self.sigma
        return np.sqrt((z ** 2).mean(axis=1))
