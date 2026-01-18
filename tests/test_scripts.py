import json

from su2_node_matrix_elements.model import node_matrix_element


def test_reference_table_schema(tmp_path):
    # Simulate reference table generation deterministically.
    cases = [
        ("a", [1, 1, 0]),
        ("b", [0.5, 0.5, 1]),
        ("c", [1, 2, 3, 4]),
    ]

    payload = {"cases": []}
    for label, spins in cases:
        payload["cases"].append({"label": label, "spins": spins, "value": node_matrix_element(spins=spins)})

    path = tmp_path / "ref.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert len(loaded["cases"]) == 3
    assert all("value" in c for c in loaded["cases"])