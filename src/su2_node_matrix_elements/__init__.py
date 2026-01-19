"""SU(2) node matrix elements (research-stage).

This package provides a minimal, deterministic computational entrypoint to
support validation, stability analysis, and reproducible paper tables.

Important: the initial implementation is a simplified determinant-based model.
It is intended as a scaffold for the full source-derivative construction.
"""

from .model import (
    SpinDomainError,
    validate_spins,
    build_K,
    det_I_minus_K,
    node_matrix_element,
)

from .stability import (
    stability_metrics,
)

from .derivative import (
    DerivativeConfig,
    node_matrix_element_derivative,
)

__all__ = [
    "SpinDomainError",
    "validate_spins",
    "build_K",
    "det_I_minus_K",
    "node_matrix_element",
    "stability_metrics",
    "DerivativeConfig",
    "node_matrix_element_derivative",
]

