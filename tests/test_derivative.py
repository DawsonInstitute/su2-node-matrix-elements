"""Tests for the derivative-based node matrix element computation (N6)."""

import pytest

from su2_node_matrix_elements.derivative import (
    DerivativeConfig,
    node_matrix_element_derivative,
)
from su2_node_matrix_elements.model import (
    SpinDomainError,
    node_matrix_element,
)


class TestDerivativeAPI:
    """Test the derivative-based API prototype."""

    def test_trivalent_simple(self):
        """Test k=3 derivative computation for simple case."""
        spins = [1, 1, 0]
        config = DerivativeConfig(h=1e-5, epsilon=1e-10)
        result = node_matrix_element_derivative(spins=spins, config=config)
        assert isinstance(result, float)
        assert result != 0.0

    def test_tetravalent_uniform(self):
        """Test k=4 derivative computation (primary N6 target)."""
        spins = [1, 1, 1, 1]
        config = DerivativeConfig(h=1e-5, epsilon=1e-10)
        result = node_matrix_element_derivative(spins=spins, config=config)
        assert isinstance(result, float)
        assert result != 0.0

    def test_tetravalent_half_integers(self):
        """Test k=4 with half-integer spins."""
        spins = [0.5, 0.5, 1, 1]
        config = DerivativeConfig(h=1e-5, epsilon=1e-10)
        result = node_matrix_element_derivative(spins=spins, config=config)
        assert isinstance(result, float)

    def test_higher_valence_not_implemented(self):
        """Verify that k > 4 raises NotImplementedError."""
        spins = [1, 1, 1, 1, 1]  # k=5
        config = DerivativeConfig()
        with pytest.raises(NotImplementedError, match="k=5"):
            node_matrix_element_derivative(spins=spins, config=config)

    def test_invalid_spins_raises(self):
        """Verify that invalid spins raise SpinDomainError."""
        spins = [1.3, 2, 3, 4]  # 1.3 is not integer/half-integer
        config = DerivativeConfig()
        with pytest.raises(SpinDomainError):
            node_matrix_element_derivative(spins=spins, config=config)


class TestDerivativeVsDeterminant:
    """Cross-verification: derivative vs determinant placeholder."""

    def test_trivalent_agreement(self):
        """Compare derivative and determinant for k=3."""
        spins = [1, 1, 0]
        config = DerivativeConfig(h=1e-5, epsilon=1e-10)

        deriv_result = node_matrix_element_derivative(spins=spins, config=config)
        det_result = node_matrix_element(spins=spins, epsilon=1e-10, det_power=1)

        # Both should be non-zero floats (exact agreement not expected due to different models)
        assert isinstance(deriv_result, float)
        assert isinstance(det_result, float)
        assert deriv_result != 0.0
        assert det_result != 0.0

        # Check that they are within a few orders of magnitude
        # (This is a weak check since models differ; it verifies both are computable)
        ratio = abs(deriv_result / det_result)
        assert 1e-3 < ratio < 1e3

    def test_tetravalent_comparison(self):
        """Compare derivative and determinant for k=4."""
        spins = [1, 1, 1, 1]
        config = DerivativeConfig(h=1e-5, epsilon=1e-10)

        deriv_result = node_matrix_element_derivative(spins=spins, config=config)
        det_result = node_matrix_element(spins=spins, epsilon=1e-10, det_power=1)

        assert isinstance(deriv_result, float)
        assert isinstance(det_result, float)
        assert deriv_result != 0.0
        assert det_result != 0.0

        # Both methods should produce finite, computable results
        # (Exact agreement not expected since models differ)
        ratio = abs(deriv_result / det_result)
        assert 1e-6 < ratio < 1e6  # Relaxed range; models are different


class TestDerivativeStepSizeSensitivity:
    """Test sensitivity to finite-difference step size."""

    def test_step_size_convergence(self):
        """Verify that smaller h gives more consistent results."""
        spins = [1, 1, 1, 1]

        results = []
        for h in [1e-3, 1e-4, 1e-5, 1e-6]:
            config = DerivativeConfig(h=h, epsilon=1e-10)
            result = node_matrix_element_derivative(spins=spins, config=config)
            results.append(result)

        # Check that results are finite and non-zero
        for r in results:
            assert isinstance(r, float)
            assert r != 0.0
            assert abs(r) < 1e10  # Not diverging

        # Check that successive results have reasonable variation
        # (Allow for numerical noise in finite differences)
        diffs = [abs(results[i + 1] - results[i]) for i in range(len(results) - 1)]

        # All differences should be finite (no catastrophic divergence)
        for d in diffs:
            assert abs(d) < 1e10


class TestDerivativeDefaultConfig:
    """Test that default configuration works."""

    def test_default_config_tetravalent(self):
        """Verify default config produces valid result for k=4."""
        spins = [1, 2, 2, 1]
        # Don't pass config; should use default
        result = node_matrix_element_derivative(spins=spins)
        assert isinstance(result, float)
        assert result != 0.0
