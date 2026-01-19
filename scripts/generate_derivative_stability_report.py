#!/usr/bin/env python3
"""Generate derivative vs determinant stability comparison report (N6)."""

import json
import os
from datetime import datetime, timezone

from su2_node_matrix_elements.derivative import (
    DerivativeConfig,
    node_matrix_element_derivative,
)
from su2_node_matrix_elements.model import node_matrix_element


def main() -> int:
    os.makedirs("data/derivative", exist_ok=True)

    # Test cases for valence k ≤ 4
    cases = [
        {"label": "trivalent_uniform", "spins": [1, 1, 1]},
        {"label": "trivalent_half_int", "spins": [0.5, 0.5, 1]},
        {"label": "tetravalent_uniform", "spins": [1, 1, 1, 1]},
        {"label": "tetravalent_mixed", "spins": [1, 2, 2, 1]},
        {"label": "tetravalent_half_int", "spins": [0.5, 0.5, 1, 1]},
    ]

    step_sizes = [1e-4, 1e-5, 1e-6]
    epsilon = 1e-10

    results = []

    for case in cases:
        spins = case["spins"]
        label = case["label"]

        # Determinant baseline
        det_value = node_matrix_element(spins=spins, epsilon=epsilon, det_power=1)

        # Derivative for multiple step sizes
        for h in step_sizes:
            config = DerivativeConfig(h=h, epsilon=epsilon)
            deriv_value = node_matrix_element_derivative(spins=spins, config=config)

            ratio = abs(deriv_value / det_value) if det_value != 0 else float("inf")
            rel_diff = (
                abs(deriv_value - det_value) / abs(det_value)
                if det_value != 0
                else float("inf")
            )

            results.append(
                {
                    "label": label,
                    "valence": len(spins),
                    "spins": [float(s) for s in spins],
                    "step_size": float(h),
                    "epsilon": float(epsilon),
                    "determinant_value": float(det_value),
                    "derivative_value": float(deriv_value),
                    "ratio": float(ratio),
                    "relative_difference": float(rel_diff),
                }
            )

    out = {
        "metadata": {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "note": "Stability comparison: derivative-based vs determinant placeholder.",
        },
        "results": results,
        "summary": {
            "total_cases": len(results),
            "step_sizes_tested": step_sizes,
            "max_ratio": max(r["ratio"] for r in results if r["ratio"] != float("inf")),
            "min_ratio": min(r["ratio"] for r in results if r["ratio"] != float("inf")),
        },
    }

    path = "data/derivative/derivative_stability_comparison.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("=" * 70)
    print("Derivative vs Determinant Stability Comparison (N6)")
    print("=" * 70)
    print()

    for case in cases:
        label = case["label"]
        case_results = [r for r in results if r["label"] == label]

        print(f"{label} (valence={len(case['spins'])})")
        print("-" * 70)
        for r in case_results:
            print(f"  h={r['step_size']:.1e}: ratio={r['ratio']:.2e}, rel_diff={r['relative_difference']:.2e}")
        print()

    print("=" * 70)
    print(f"Report saved to: {path}")
    print(f"Total cases: {len(results)}")
    print(f"Ratio range: {out['summary']['min_ratio']:.2e} to {out['summary']['max_ratio']:.2e}")
    print()
    print("Note: Derivative and determinant are different mathematical models;")
    print("      order-of-magnitude differences are expected. This report validates")
    print("      that both produce finite, computable results.")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
