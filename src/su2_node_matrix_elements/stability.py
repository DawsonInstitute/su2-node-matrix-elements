from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .model import build_K, validate_spins


@dataclass(frozen=True)
class StabilityMetrics:
    det: float
    cond: float
    epsilon: float


def stability_metrics(*, spins: Sequence, epsilon: float = 1e-10) -> StabilityMetrics:
    """Compute basic stability metrics for the simplified node model."""

    js = validate_spins(spins)
    n = len(js)
    K = build_K(js, backend="numpy")
    I = np.eye(n)
    A = I - K + (epsilon * I)

    det_val = float(np.linalg.det(A))
    cond_val = float(np.linalg.cond(A))

    return StabilityMetrics(det=det_val, cond=cond_val, epsilon=float(epsilon))
