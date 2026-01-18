#!/usr/bin/env python3

import json
import os
from datetime import datetime, timezone

from su2_node_matrix_elements.model import node_matrix_element


def main() -> int:
    os.makedirs("data/reference", exist_ok=True)

    cases = [
        {"label": "trivalent_normalized_example", "valence": 3, "spins": [1, 1, 0]},
        {"label": "trivalent_half_integers", "valence": 3, "spins": [0.5, 0.5, 1]},
        {"label": "tetravalent_uniform", "valence": 4, "spins": [1, 1, 1, 1]},
        {"label": "tetravalent_mixed", "valence": 4, "spins": [0, 1, 2, 3]},
        {"label": "pentavalent_mixed", "valence": 5, "spins": [0.5, 1, 1.5, 2, 2.5]},
    ]

    epsilon = 1e-10

    out = {
        "metadata": {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "epsilon": epsilon,
            "det_power": 1,
            "note": "Determinant-based placeholder model for validation scaffolding.",
        },
        "cases": [],
    }

    for case in cases:
        value = node_matrix_element(spins=case["spins"], epsilon=epsilon, det_power=1)
        out["cases"].append({**case, "value": value})

    path = "data/reference/node_matrix_reference.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print(f"Wrote {path} with {len(out['cases'])} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
