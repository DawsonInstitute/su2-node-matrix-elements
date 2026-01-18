#!/usr/bin/env python3

import json
import os
from datetime import datetime, timezone

import numpy as np

from su2_node_matrix_elements.stability import stability_metrics


def main() -> int:
    os.makedirs("data/stability", exist_ok=True)

    epsilons = [1e-6, 1e-8, 1e-10, 1e-12]
    cases = [
        {"label": "trivalent_small", "spins": [1, 1, 1]},
        {"label": "tetravalent_mixed", "spins": [1, 2, 3, 4]},
        {"label": "pentavalent_mixed", "spins": [0.5, 1, 1.5, 2, 2.5]},
        {"label": "higher_valence", "spins": [1, 2, 3, 4, 5, 6]},
    ]

    results = []
    for case in cases:
        for eps in epsilons:
            m = stability_metrics(spins=case["spins"], epsilon=eps)
            results.append(
                {
                    "label": case["label"],
                    "valence": len(case["spins"]),
                    "spins": [float(s) for s in case["spins"]],
                    "epsilon": float(eps),
                    "det": float(m.det),
                    "cond": float(m.cond),
                }
            )

    out = {
        "metadata": {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "note": "Stability metrics for determinant-based placeholder model.",
        },
        "results": results,
        "summary": {
            "max_cond": float(max(r["cond"] for r in results)),
            "min_cond": float(min(r["cond"] for r in results)),
        },
    }

    path = "data/stability/node_matrix_stability_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print(f"Wrote {path} with {len(results)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
