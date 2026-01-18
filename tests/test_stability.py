import numpy as np

from su2_node_matrix_elements.stability import stability_metrics


def test_stability_metrics_fields_are_finite():
    metrics = stability_metrics(spins=[1, 1, 1], epsilon=1e-10)
    assert np.isfinite(metrics.det)
    assert np.isfinite(metrics.cond)
    assert metrics.epsilon == 1e-10


def test_condition_number_increases_with_spin_scale():
    base = stability_metrics(spins=[1, 2, 3, 4], epsilon=1e-10).cond
    scaled = stability_metrics(spins=[10, 20, 30, 40], epsilon=1e-10).cond
    assert scaled >= base


def test_condition_number_sensitive_to_epsilon():
    spins = [1, 2, 3, 4, 5]
    c1 = stability_metrics(spins=spins, epsilon=1e-6).cond
    c2 = stability_metrics(spins=spins, epsilon=1e-12).cond
    assert c2 >= c1
