"""Evaluate OT coupling artifacts without importing the method implementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def resolve_artifact(manifest_path: Path, artifact: str) -> Path:
    artifact_path = Path(artifact)
    if artifact_path.exists():
        return artifact_path
    candidate = manifest_path.parent / artifact_path
    if candidate.exists():
        return candidate
    return artifact_path


def class_preservation(path: str | Path) -> tuple[dict[str, float], dict[str, object]]:
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
    same_class_mass = class_mass[np.arange(len(source_labels)), true_pos]
    wrong_class_mass = class_mass.copy()
    wrong_class_mass[np.arange(len(source_labels)), true_pos] = -np.inf
    class_mass_margin = same_class_mass - wrong_class_mass.max(axis=1)
    target_mass = gamma.sum(axis=0)
    target_mass_sum = target_mass.sum()
    if target_mass_sum > 0:
        target_prob = target_mass / target_mass_sum
        entropy = -np.sum(target_prob[target_prob > 0] * np.log(target_prob[target_prob > 0]))
        effective_support_ratio = float(np.exp(entropy) / len(target_labels))
    else:
        effective_support_ratio = 0.0
    confusion = np.zeros((len(labels), len(labels)), dtype=np.float64)
    for truth, pred in zip(true_pos, pred_pos, strict=True):
        confusion[truth, pred] += 1.0
    class_counts = np.maximum(confusion.sum(axis=1, keepdims=True), 1.0)
    per_class = confusion / class_counts
    metrics = {
        "class_preservation_accuracy": float((pred_pos == true_pos).mean()),
        "mean_diagonal_class_accuracy": float(np.diag(per_class).mean()),
        "same_class_mass_mean": float(same_class_mass.mean()),
        "class_mass_margin_mean": float(class_mass_margin.mean()),
        "target_effective_support_ratio": effective_support_ratio,
        "label_cost_used": 0,
        "num_source": int(len(source_labels)),
        "num_target": int(len(target_labels)),
    }
    matrix = {
        "labels": labels.astype(int).tolist(),
        "counts": confusion.astype(int).tolist(),
        "row_normalized": per_class.tolist(),
    }
    return metrics, matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        nargs="+",
        default=["outputs/bot/manifest.json"],
        help="One or more method output manifests to evaluate and merge.",
    )
    parser.add_argument("--out", default="scoring/scores.json")
    parser.add_argument(
        "--confusion-out",
        default=None,
        help="Optional path for class-confusion matrices derived from saved couplings.",
    )
    parser.add_argument(
        "--duplicate-policy",
        choices=["error", "keep-last"],
        default="error",
        help="How to handle duplicate experiment/method entries across manifests.",
    )
    args = parser.parse_args()

    scores = {"experiments": {}}
    confusions = {"experiments": {}}
    for manifest_path in args.manifest:
        manifest_file = Path(manifest_path)
        manifest = json.loads(manifest_file.read_text())
        for exp_name, methods in manifest["experiments"].items():
            scores["experiments"].setdefault(exp_name, {"results": {}})
            confusions["experiments"].setdefault(exp_name, {"results": {}})
            for method_name, artifact in methods.items():
                if method_name in scores["experiments"][exp_name]["results"]:
                    if args.duplicate_policy == "error":
                        raise ValueError(f"Duplicate method for {exp_name}: {method_name}")
                metrics, matrix = class_preservation(resolve_artifact(manifest_file, artifact))
                scores["experiments"][exp_name]["results"][method_name] = metrics
                confusions["experiments"][exp_name]["results"][method_name] = matrix

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2) + "\n")
    if args.confusion_out is not None:
        confusion_out = Path(args.confusion_out)
        confusion_out.parent.mkdir(parents=True, exist_ok=True)
        confusion_out.write_text(json.dumps(confusions, indent=2) + "\n")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
