"""Visible train-slice evaluator for saved OT coupling artifacts.

This module intentionally does not import from method/ or eval/test/.
It reads the same manifest/artifact interface as scripts/evaluate.sh and
scores a deterministic stratified half of the source rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SLICE = "train"
SPLIT_SEED = 1729


def stable_sort_key(experiment: str, index: int, seed: int) -> int:
    payload = f"{seed}:{experiment}:{index}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def split_source_indices(labels: np.ndarray, experiment: str, seed: int) -> np.ndarray:
    selected: list[int] = []
    for label in sorted(set(labels.astype(int).tolist())):
        label_indices = np.flatnonzero(labels == label).astype(int).tolist()
        label_indices.sort(key=lambda idx: stable_sort_key(experiment, idx, seed))
        selected.extend(label_indices[: len(label_indices) // 2])
    return np.array(sorted(selected), dtype=int)


def resolve_artifact(manifest_path: Path, artifact: str) -> Path:
    artifact_path = Path(artifact)
    if artifact_path.exists():
        return artifact_path
    candidate = manifest_path.parent / artifact_path
    if candidate.exists():
        return candidate
    return artifact_path


def class_preservation(
    path: str | Path, experiment: str, seed: int
) -> tuple[dict[str, float], dict[str, object], list[int]]:
    data = np.load(path)
    gamma_all = data["gamma"].astype(np.float64)
    source_labels_all = data["source_labels"].astype(int)
    target_labels = data["target_labels"].astype(int)
    source_indices = split_source_indices(source_labels_all, experiment, seed)
    if len(source_indices) == 0:
        raise ValueError(f"No source rows selected for {experiment} in {path}")

    gamma = gamma_all[source_indices]
    source_labels = source_labels_all[source_indices]
    labels = np.array(
        sorted(set(source_labels.tolist()) | set(target_labels.tolist())), dtype=int
    )
    label_to_pos = {int(label): i for i, label in enumerate(labels)}
    one_hot = np.zeros((len(target_labels), len(labels)), dtype=np.float64)
    for j, label in enumerate(target_labels):
        one_hot[j, label_to_pos[int(label)]] = 1.0
    row_sums = np.maximum(gamma.sum(axis=1, keepdims=True), 1e-40)
    normalized_gamma = gamma / row_sums
    class_mass = normalized_gamma @ one_hot
    pred_pos = class_mass.argmax(axis=1)
    true_pos = np.array([label_to_pos[int(label)] for label in source_labels], dtype=int)
    correct_mass = class_mass[np.arange(len(source_labels)), true_pos]
    off_class_mass = class_mass.copy()
    off_class_mass[np.arange(len(source_labels)), true_pos] = -np.inf
    class_margin = correct_mass - off_class_mass.max(axis=1)
    confusion = np.zeros((len(labels), len(labels)), dtype=np.float64)
    for truth, pred in zip(true_pos, pred_pos, strict=True):
        confusion[truth, pred] += 1.0
    class_counts = np.maximum(confusion.sum(axis=1, keepdims=True), 1.0)
    per_class = confusion / class_counts
    total_mass = gamma.sum()
    if total_mass <= 0:
        target_effective_support = 0.0
    else:
        target_mass = gamma.sum(axis=0) / total_mass
        positive_target_mass = target_mass[target_mass > 0]
        target_entropy = float(
            -(positive_target_mass * np.log(positive_target_mass)).sum()
        )
        target_effective_support = float(np.exp(target_entropy) / len(target_labels))
    metrics = {
        "class_preservation_accuracy": float((pred_pos == true_pos).mean()),
        "mean_diagonal_class_accuracy": float(np.diag(per_class).mean()),
        "same_class_mass_mean": float(correct_mass.mean()),
        "class_mass_margin_mean": float(class_margin.mean()),
        "target_effective_support_ratio": target_effective_support,
        "label_cost_used": 0,
        "num_source": int(len(source_labels)),
        "num_target": int(len(target_labels)),
    }
    matrix = {
        "slice": SLICE,
        "split_style": "stratified_source_row_half",
        "split_seed": seed,
        "labels": labels.astype(int).tolist(),
        "counts": confusion.astype(int).tolist(),
        "row_normalized": per_class.tolist(),
    }
    return metrics, matrix, source_indices.astype(int).tolist()


def load_manifest_entries(
    manifest_paths: list[str], duplicate_policy: str
) -> list[dict[str, str]]:
    entries_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for raw_manifest_path in manifest_paths:
        manifest_path = Path(raw_manifest_path)
        manifest = json.loads(manifest_path.read_text())
        for exp_name, methods in manifest["experiments"].items():
            for method_name, artifact in methods.items():
                key = (exp_name, method_name)
                if key in entries_by_key and duplicate_policy == "error":
                    raise ValueError(f"Duplicate method for {exp_name}: {method_name}")
                entries_by_key[key] = {
                    "experiment": exp_name,
                    "method": method_name,
                    "manifest": str(manifest_path),
                    "artifact": str(resolve_artifact(manifest_path, artifact)),
                }
    return [entries_by_key[key] for key in sorted(entries_by_key)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        nargs="+",
        required=True,
        help="One or more method output manifests to evaluate and merge.",
    )
    parser.add_argument("--out", default="scoring/scores_train.json")
    parser.add_argument(
        "--confusion-out",
        default=None,
        help="Optional path for train-slice confusion matrices.",
    )
    parser.add_argument(
        "--artifact-index",
        default=None,
        help="Optional path recording the exact artifacts scored for the test slice.",
    )
    parser.add_argument(
        "--duplicate-policy",
        choices=["error", "keep-last"],
        default="error",
        help="How to handle duplicate experiment/method entries across manifests.",
    )
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    args = parser.parse_args()

    scores = {"experiments": {}}
    confusions = {"slice": SLICE, "experiments": {}}
    artifact_entries = []
    for entry in load_manifest_entries(args.manifest, args.duplicate_policy):
        exp_name = entry["experiment"]
        method_name = entry["method"]
        scores["experiments"].setdefault(exp_name, {"results": {}})
        confusions["experiments"].setdefault(exp_name, {"results": {}})
        metrics, matrix, source_indices = class_preservation(
            entry["artifact"], exp_name, args.split_seed
        )
        scores["experiments"][exp_name]["results"][method_name] = metrics
        confusions["experiments"][exp_name]["results"][method_name] = matrix
        artifact_entries.append(
            {
                **entry,
                "slice": SLICE,
                "split_style": "stratified_source_row_half",
                "split_seed": args.split_seed,
                "num_selected_source": len(source_indices),
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2) + "\n")
    if args.confusion_out is not None:
        confusion_out = Path(args.confusion_out)
        confusion_out.parent.mkdir(parents=True, exist_ok=True)
        confusion_out.write_text(json.dumps(confusions, indent=2) + "\n")
    if args.artifact_index is not None:
        artifact_index = Path(args.artifact_index)
        artifact_index.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "slice": SLICE,
            "split_style": "stratified_source_row_half",
            "split_seed": args.split_seed,
            "artifacts": artifact_entries,
        }
        artifact_index.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
