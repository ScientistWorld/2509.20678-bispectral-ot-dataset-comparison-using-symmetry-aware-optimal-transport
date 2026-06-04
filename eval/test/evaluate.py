"""Held-out test-slice evaluator for saved OT coupling artifacts.

This module intentionally duplicates the train-slice mechanics instead of
importing from eval/train/. It reads scoring/scores_train.json to require the
same experiment/method result keys, then scores the complementary source rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SLICE = "test"
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
        selected.extend(label_indices[len(label_indices) // 2 :])
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
    class_mass = (gamma / row_sums) @ one_hot
    pred_pos = class_mass.argmax(axis=1)
    true_pos = np.array([label_to_pos[int(label)] for label in source_labels], dtype=int)
    confusion = np.zeros((len(labels), len(labels)), dtype=np.float64)
    for truth, pred in zip(true_pos, pred_pos, strict=True):
        confusion[truth, pred] += 1.0
    class_counts = np.maximum(confusion.sum(axis=1, keepdims=True), 1.0)
    per_class = confusion / class_counts
    metrics = {
        "class_preservation_accuracy": float((pred_pos == true_pos).mean()),
        "mean_diagonal_class_accuracy": float(np.diag(per_class).mean()),
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


def required_results(train_scores_path: str | None) -> dict[str, set[str]]:
    if train_scores_path is None:
        return {}
    path = Path(train_scores_path)
    if not path.exists():
        return {}
    train_scores = json.loads(path.read_text())
    required: dict[str, set[str]] = {}
    for exp_name, exp in train_scores.get("experiments", {}).items():
        required[exp_name] = set(exp.get("results", {}).keys())
    return required


def entries_from_artifact_index(artifact_index_path: str) -> list[dict[str, str]]:
    payload = json.loads(Path(artifact_index_path).read_text())
    entries: list[dict[str, str]] = []
    for entry in payload.get("artifacts", []):
        entries.append(
            {
                "experiment": entry["experiment"],
                "method": entry["method"],
                "manifest": entry.get("manifest", ""),
                "artifact": entry["artifact"],
            }
        )
    return entries


def entries_from_manifests(manifest_paths: list[str]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw_manifest_path in manifest_paths:
        manifest_path = Path(raw_manifest_path)
        manifest = json.loads(manifest_path.read_text())
        for exp_name, methods in manifest["experiments"].items():
            for method_name, artifact in methods.items():
                key = (exp_name, method_name)
                if key in seen:
                    raise ValueError(f"Duplicate method for {exp_name}: {method_name}")
                seen.add(key)
                entries.append(
                    {
                        "experiment": exp_name,
                        "method": method_name,
                        "manifest": str(manifest_path),
                        "artifact": str(resolve_artifact(manifest_path, artifact)),
                    }
                )
    return entries


def filter_required(
    entries: list[dict[str, str]], required: dict[str, set[str]]
) -> list[dict[str, str]]:
    if not required:
        return entries
    by_key = {(entry["experiment"], entry["method"]): entry for entry in entries}
    missing = []
    selected = []
    for exp_name in sorted(required):
        for method_name in sorted(required[exp_name]):
            key = (exp_name, method_name)
            if key not in by_key:
                missing.append(f"{exp_name}/{method_name}")
            else:
                selected.append(by_key[key])
    if missing:
        formatted = ", ".join(missing)
        raise ValueError(f"Missing artifacts for train-scored entries: {formatted}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        nargs="+",
        default=None,
        help="Optional manifests to evaluate; defaults to the train artifact index.",
    )
    parser.add_argument("--out", default="scoring/scores_test.json")
    parser.add_argument(
        "--confusion-out",
        default=None,
        help="Optional path for test-slice confusion matrices.",
    )
    parser.add_argument(
        "--train-scores",
        default="scoring/scores_train.json",
        help="Train score file whose experiment/method result keys must be matched.",
    )
    parser.add_argument(
        "--artifact-index",
        default="scoring/eval_train_artifacts.json",
        help="Train-produced artifact index used when manifests are not supplied.",
    )
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    args = parser.parse_args()

    required = required_results(args.train_scores)
    if args.manifest is not None:
        entries = entries_from_manifests(args.manifest)
    elif Path(args.artifact_index).exists():
        entries = entries_from_artifact_index(args.artifact_index)
    else:
        entries = entries_from_manifests(["outputs/bot/manifest.json"])
    entries = filter_required(entries, required)

    scores = {"experiments": {}}
    confusions = {"slice": SLICE, "experiments": {}}
    for entry in entries:
        exp_name = entry["experiment"]
        method_name = entry["method"]
        scores["experiments"].setdefault(exp_name, {"results": {}})
        confusions["experiments"].setdefault(exp_name, {"results": {}})
        metrics, matrix, _ = class_preservation(
            entry["artifact"], exp_name, args.split_seed
        )
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
