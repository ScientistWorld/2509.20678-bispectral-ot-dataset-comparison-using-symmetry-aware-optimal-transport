"""Evaluate OT coupling artifacts without importing the method implementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def class_preservation(path: str | Path) -> dict[str, float]:
    data = np.load(path)
    gamma = data["gamma"].astype(np.float64)
    source_labels = data["source_labels"].astype(int)
    target_labels = data["target_labels"].astype(int)
    labels = np.array(sorted(set(source_labels.tolist()) | set(target_labels.tolist())), dtype=int)
    label_to_pos = {int(label): i for i, label in enumerate(labels)}
    one_hot = np.zeros((len(target_labels), len(labels)), dtype=np.float64)
    for j, label in enumerate(target_labels):
        one_hot[j, label_to_pos[int(label)]] = 1.0
    row_sums = np.maximum(gamma.sum(axis=1, keepdims=True), 1e-40)
    class_mass = (gamma / row_sums) @ one_hot
    pred_pos = class_mass.argmax(axis=1)
    true_pos = np.array([label_to_pos[int(label)] for label in source_labels], dtype=int)
    confusion = np.zeros((len(labels), len(labels)), dtype=np.float64)
    for truth, pred in zip(true_pos, pred_pos, strict=True):
        confusion[truth, pred] += 1.0
    class_counts = np.maximum(confusion.sum(axis=1, keepdims=True), 1.0)
    per_class = confusion / class_counts
    return {
        "class_preservation_accuracy": float((pred_pos == true_pos).mean()),
        "mean_diagonal_class_accuracy": float(np.diag(per_class).mean()),
        "label_cost_used": 0,
        "num_source": int(len(source_labels)),
        "num_target": int(len(target_labels)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        nargs="+",
        default=["outputs/bot/manifest.json"],
        help="One or more method output manifests to evaluate and merge.",
    )
    parser.add_argument("--out", default="scoring/scores.json")
    args = parser.parse_args()

    scores = {"experiments": {}}
    for manifest_path in args.manifest:
        manifest = json.loads(Path(manifest_path).read_text())
        for exp_name, methods in manifest["experiments"].items():
            scores["experiments"].setdefault(exp_name, {"results": {}})
            for method_name, artifact in methods.items():
                if method_name in scores["experiments"][exp_name]["results"]:
                    raise ValueError(f"Duplicate method for {exp_name}: {method_name}")
                scores["experiments"][exp_name]["results"][method_name] = class_preservation(artifact)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2) + "\n")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
