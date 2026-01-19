"""Source-derivative prototype for node matrix elements.

This module implements a finite-difference approximation to the source derivative
construction described in the paper:

    M_v = ∂^k G(x_e)/∂s_1...∂s_k |_{s=0}

where G(x_e) ≈ 1/det(I - K(x_e)) is the generating functional.

This is a research-stage prototype for k=4 valence nodes. It validates against
the determinant placeholder model for small test cases and provides stability
comparison data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .model import SpinDomainError, build_K, validate_spins


@dataclass(frozen=True)
class DerivativeConfig:
    """Configuration for finite-difference derivative computation."""

    h: float = 1e-5  # Step size for finite difference
    method: str = "central"  # 'central', 'forward', or 'backward'
    epsilon: float = 1e-10  # Regularization for determinant


def _generating_functional(
    spins: Sequence, source: Sequence[float], *, epsilon: float = 1e-10
) -> float:
    """Evaluate G(x_e, s) = 1/det(I - K(x_e) - diag(s))."""

    js = validate_spins(spins)
    n = len(js)

    if len(source) != n:
        raise ValueError(f"Source vector length {len(source)} != valence {n}")

    K = build_K(js, backend="numpy")
    I = np.eye(n)
    S = np.diag(source)

    A = I - K - S + (epsilon * I)
    det_val = np.linalg.det(A)

    if abs(det_val) < 1e-300:
        raise ZeroDivisionError("det(I-K-S) is numerically zero")

    return 1.0 / det_val


def _finite_diff_central(
    spins: Sequence, config: DerivativeConfig, indices: tuple[int, ...]
) -> float:
    """Central difference approximation for mixed partial derivative."""

    n = len(spins)
    h = config.h
    k = len(indices)

    if k == 0:
        s_zero = [0.0] * n
        return _generating_functional(spins, s_zero, epsilon=config.epsilon)

    if k == 1:
        idx = indices[0]
        s_plus = [0.0] * n
        s_minus = [0.0] * n
        s_plus[idx] = h
        s_minus[idx] = -h
        g_plus = _generating_functional(spins, s_plus, epsilon=config.epsilon)
        g_minus = _generating_functional(spins, s_minus, epsilon=config.epsilon)
        return (g_plus - g_minus) / (2 * h)

    # For higher-order (k > 1), use recursive Richardson extrapolation
    # This is a simplified placeholder; a production version would use
    # a more sophisticated scheme (e.g., automatic differentiation).
    if k == 2:
        i, j = indices
        s_pp = [0.0] * n
        s_pm = [0.0] * n
        s_mp = [0.0] * n
        s_mm = [0.0] * n

        s_pp[i] = h
        s_pp[j] += h
        s_pm[i] = h
        s_pm[j] -= h
        s_mp[i] = -h
        s_mp[j] += h
        s_mm[i] = -h
        s_mm[j] -= h

        g_pp = _generating_functional(spins, s_pp, epsilon=config.epsilon)
        g_pm = _generating_functional(spins, s_pm, epsilon=config.epsilon)
        g_mp = _generating_functional(spins, s_mp, epsilon=config.epsilon)
        g_mm = _generating_functional(spins, s_mm, epsilon=config.epsilon)

        return (g_pp - g_pm - g_mp + g_mm) / (4 * h * h)

    if k == 3:
        # Third-order mixed partial (simplified)
        i, j, k_idx = indices
        # Use a symmetric 2^3 stencil
        configs = [
            (1, 1, 1),
            (1, 1, -1),
            (1, -1, 1),
            (1, -1, -1),
            (-1, 1, 1),
            (-1, 1, -1),
            (-1, -1, 1),
            (-1, -1, -1),
        ]

        signs = [1, -1, -1, 1, -1, 1, 1, -1]
        total = 0.0

        for (si, sj, sk), sign in zip(configs, signs):
            s_vec = [0.0] * n
            s_vec[i] = si * h
            s_vec[j] = sj * h
            s_vec[k_idx] = sk * h
            total += sign * _generating_functional(
                spins, s_vec, epsilon=config.epsilon
            )

        return total / (8 * h * h * h)

    if k == 4:
        # Fourth-order mixed partial for tetravalent node (k=4)
        i, j, k_idx, m = indices

        # Use a symmetric 2^4 stencil
        configs = [
            (1, 1, 1, 1),
            (1, 1, 1, -1),
            (1, 1, -1, 1),
            (1, 1, -1, -1),
            (1, -1, 1, 1),
            (1, -1, 1, -1),
            (1, -1, -1, 1),
            (1, -1, -1, -1),
            (-1, 1, 1, 1),
            (-1, 1, 1, -1),
            (-1, 1, -1, 1),
            (-1, 1, -1, -1),
            (-1, -1, 1, 1),
            (-1, -1, 1, -1),
            (-1, -1, -1, 1),
            (-1, -1, -1, -1),
        ]

        signs = [1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1]
        total = 0.0

        for (si, sj, sk, sm), sign in zip(configs, signs):
            s_vec = [0.0] * n
            s_vec[i] = si * h
            s_vec[j] = sj * h
            s_vec[k_idx] = sk * h
            s_vec[m] = sm * h
            total += sign * _generating_functional(
                spins, s_vec, epsilon=config.epsilon
            )

        return total / (16 * h**4)

    raise NotImplementedError(f"Derivative order k={k} not yet implemented")


def node_matrix_element_derivative(
    *,
    spins: Sequence,
    config: DerivativeConfig | None = None,
) -> float:
    """Compute node matrix element via finite-difference source derivative.

    This is a prototype implementation for validation and stability comparison.
    It approximates:

        M_v = ∂^k G(x_e, s)/∂s_1...∂s_k |_{s=0}

    using finite differences.

    Args:
        spins: Spin labels for the node legs (length k = valence)
        config: Finite-difference configuration (step size, method, epsilon)

    Returns:
        Approximation of the k-th order mixed partial derivative at s=0

    Raises:
        SpinDomainError: if spins are invalid
        NotImplementedError: if k > 4 (higher valence not yet supported)
    """

    if config is None:
        config = DerivativeConfig()

    js = validate_spins(spins)
    k = len(js)

    if k > 4:
        raise NotImplementedError(
            f"Derivative-based computation for valence k={k} not yet implemented; "
            "current prototype supports k ≤ 4"
        )

    indices = tuple(range(k))
    return _finite_diff_central(spins, config, indices)
