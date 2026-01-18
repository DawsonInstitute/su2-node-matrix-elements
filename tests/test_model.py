import itertools

import numpy as np
import pytest
import sympy as sp

from su2_node_matrix_elements.model import (
    SpinDomainError,
    validate_spins,
    build_K,
    det_I_minus_K,
    node_matrix_element,
)


def test_validate_spins_accepts_integers_and_halves():
    js = validate_spins([0, 1, 2, 0.5, 1.5])
    assert js == [sp.Rational(0), sp.Rational(1), sp.Rational(2), sp.Rational(1, 2), sp.Rational(3, 2)]


def test_validate_spins_rejects_non_half_integer_float():
    with pytest.raises(SpinDomainError):
        validate_spins([1 / 3])


def test_validate_spins_rejects_negative():
    with pytest.raises(SpinDomainError):
        validate_spins([-1])


def test_validate_spins_length_mismatch():
    with pytest.raises(SpinDomainError, match="Expected 4 spins"):
        validate_spins([1, 1, 1], valence=4)


def test_build_K_is_antisymmetric_numpy():
    K = build_K([1, 2, 3], backend="numpy")
    assert np.allclose(K, -K.T)
    assert np.allclose(np.diag(K), 0.0)


def test_build_K_is_antisymmetric_sympy():
    K = build_K([1, 2, 3], backend="sympy")
    assert (K + K.T) == sp.zeros(3, 3)


def test_det_numpy_matches_sympy_small_cases():
    cases = [
        (3, [1, 1, 0]),
        (3, [sp.Rational(1, 2), sp.Rational(1, 2), 1]),
        (4, [1, 1, 1, 1]),
        (4, [0, 1, 2, 3]),
    ]

    for _, spins in cases:
        det_np = det_I_minus_K(spins, epsilon=1e-10, backend="numpy")
        det_sp = det_I_minus_K(spins, epsilon=1e-10, backend="sympy")
        det_sp_f = float(sp.N(det_sp, 50))
        assert abs(det_np - det_sp_f) < 1e-7


def test_node_matrix_element_permutation_invariant():
    spins = [1, 2, 0.5, 1.5]
    base = node_matrix_element(spins=spins, epsilon=1e-10, backend="numpy")

    for perm in itertools.permutations(spins):
        val = node_matrix_element(spins=list(perm), epsilon=1e-10, backend="numpy")
        assert abs(val - base) < 1e-9


def test_node_matrix_element_matches_sympy_backend():
    spins = [sp.Rational(1, 2), 1, sp.Rational(3, 2), 2]
    val_np = node_matrix_element(spins=spins, epsilon=1e-10, backend="numpy")
    val_sp = node_matrix_element(spins=spins, epsilon=1e-10, backend="sympy")
    assert abs(val_np - val_sp) < 1e-7


def test_epsilon_regularization_changes_smoothly():
    spins = [1, 1, 1, 1]
    v1 = node_matrix_element(spins=spins, epsilon=1e-8)
    v2 = node_matrix_element(spins=spins, epsilon=1e-9)
    assert np.isfinite(v1)
    assert np.isfinite(v2)
    # Not identical but should be same order of magnitude.
    assert 0.01 < abs(v1 / v2) < 100


def test_det_power_validation():
    with pytest.raises(ValueError):
        node_matrix_element(spins=[1, 1, 1], det_power=0)
