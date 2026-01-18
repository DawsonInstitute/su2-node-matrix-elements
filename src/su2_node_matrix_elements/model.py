from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np
import sympy as sp


Backend = Literal["numpy", "sympy"]


class SpinDomainError(ValueError):
    pass


def _as_rational_spin(j) -> sp.Rational:
    if isinstance(j, sp.Rational):
        return j
    if isinstance(j, (sp.Integer, int)):
        return sp.Rational(j)
    if isinstance(j, float):
        # Only accept exact halves in float form.
        twoj = 2 * j
        if abs(twoj - round(twoj)) > 1e-12:
            raise SpinDomainError(f"Spin {j} is not integer/half-integer")
        return sp.Rational(int(round(twoj)), 2)
    return sp.Rational(j)


def validate_spins(spins: Sequence, *, valence: int | None = None) -> list[sp.Rational]:
    if valence is None:
        valence = len(spins)
    if len(spins) != valence:
        raise SpinDomainError(f"Expected {valence} spins, got {len(spins)}")

    rational_spins: list[sp.Rational] = []
    for j in spins:
        jr = _as_rational_spin(j)
        if jr < 0:
            raise SpinDomainError("Spins must be non-negative")
        if (2 * jr) % 1 != 0:
            raise SpinDomainError(f"Spin {j} is not integer/half-integer")
        rational_spins.append(jr)

    return rational_spins


def build_K(spins: Sequence, *, backend: Backend = "numpy"):
    """Build a simplified antisymmetric K matrix for an n-valent node.

    This is a deterministic placeholder coupling model:
    K[i,j] = +s_i s_j for i<j and antisymmetric for i>j.

    The intended long-term target is K(x_e) from the source-extended
    generating functional construction.
    """

    js = validate_spins(spins)
    n = len(js)

    if backend == "numpy":
        K = np.zeros((n, n), dtype=float)
        j_float = [float(j) for j in js]
        for i in range(n):
            for j in range(i + 1, n):
                val = j_float[i] * j_float[j]
                K[i, j] = val
                K[j, i] = -val
        return K

    if backend == "sympy":
        K = sp.zeros(n, n)
        for i in range(n):
            for j in range(i + 1, n):
                val = js[i] * js[j]
                K[i, j] = val
                K[j, i] = -val
        return K

    raise ValueError(f"Unknown backend: {backend}")


def det_I_minus_K(
    spins: Sequence,
    *,
    epsilon: float = 1e-10,
    backend: Backend = "numpy",
) -> float | sp.Expr:
    """Compute det(I - K + eps I)."""

    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")

    K = build_K(spins, backend=backend)
    n = len(spins)

    if backend == "numpy":
        I = np.eye(n)
        A = I - K + (epsilon * I)
        return float(np.linalg.det(A))

    A = sp.eye(n) - K + (sp.Rational(int(epsilon * 10**12), 10**12) * sp.eye(n))
    return sp.simplify(A.det())


def node_matrix_element(
    *,
    spins: Sequence,
    epsilon: float = 1e-10,
    backend: Backend = "numpy",
    det_power: int = 1,
) -> float:
    """Compute a simplified determinant-based node matrix element.

    Returns:
        1 / det(I - K + eps I)^det_power

    Notes:
        - This is a deterministic placeholder for the full source-derivative
          construction described in the paper.
        - det_power=1 matches the example snippet used for testing scaffolds.
        - det_power=2 can be used to mimic a 1/sqrt(det) style dependence.
    """

    if det_power <= 0:
        raise ValueError("det_power must be positive")

    det_val = det_I_minus_K(spins, epsilon=epsilon, backend=backend)
    if backend == "sympy":
        det_val = float(sp.N(det_val, 50))

    if abs(det_val) < 1e-300:
        raise ZeroDivisionError("det(I-K) is numerically zero")

    return float(det_val) ** (-det_power)
