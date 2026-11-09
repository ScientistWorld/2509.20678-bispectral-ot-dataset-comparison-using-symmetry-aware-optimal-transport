"""Visible (train) evaluator for symmetry-robust dataset matching.

Builds matching instances from the public *training* splits with a public seed,
runs the solver on them through the same interface as the hidden test, and
recomputes every score from the raw couplings the solver wrote.

Instances are presented exactly as in the hidden test: shuffled rows, no
labels, no condition flag, opaque instance ids. The answer key for these
visible instances can be written with ``--key-out`` for debugging.

Couplings must be finite, non-negative, of shape (n, m), carry positive mass,
and have uniform marginals within ``MARGINAL_TOL`` (L1, after normalising
total mass to 1). Any invalid coupling fails the run: every metric is set to
its floor and the process exits with status 1.

This file intentionally shares no code with eval/test/ or data/.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("mnist", "kmnist", "fmnist", "emnist")
CONDITIONS = {"rotated": "rotated_matching", "unrotated": "unrotated_baseline"}
DATASET_STATS = {
    "mnist": (0.1306604787, 0.3081078053),
    "kmnist": (0.1917621507, 0.3483428300),
    "fmnist": (0.2860405969, 0.3530242445),
    "emnist": (0.1736, 0.3317),
}
PER_CLASS = 100
DEFAULT_SEED = 20250925
DEFAULT_REPLICATES = 1
MARGINAL_TOL = 1e-2
NEG_TOL = 1e-12
FLOOR = {"class_preservation_accuracy": 0.0, "same_class_mass_mean": 0.0, "class_mass_margin_mean": -1.0}
BASELINE_REG = 0.01


class InvalidCoupling(Exception):
    pass


def experiment_names() -> list[str]:
    return [f"{d}_{suffix}" for d in DATASETS for suffix in CONDITIONS.values()]


def floor_scores() -> dict:
    return {"experiments": {e: {"results": {"method": dict(FLOOR)}} for e in experiment_names()}}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


def read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        ndim = magic & 0xFF
        dims = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(dims)


def load_training_split(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    raw = ROOT / "data" / "downloads" / dataset / "raw"
    images = read_idx(raw / "train-images-idx3-ubyte.gz")
    labels = read_idx(raw / "train-labels-idx1-ubyte.gz").astype(np.int64)
    if dataset == "emnist":
        labels = labels - 1  # letters are encoded 1..26
    return images, labels


def normalize(images: np.ndarray, dataset: str) -> np.ndarray:
    mean, std = DATASET_STATS[dataset]
    return (images.astype(np.float32) / 255.0 - mean) / std


def rotate(images: np.ndarray, angles: np.ndarray) -> np.ndarray:
    out = np.empty_like(images, dtype=np.float32)
    for i, (img, angle) in enumerate(zip(images, angles, strict=True)):
        out[i] = ndimage.rotate(img, float(angle), reshape=False, order=1, mode="constant", cval=0.0, prefilter=False)
    return out


def build_instances(seed: int, replicates: int) -> list[dict]:
    """Class-balanced disjoint source/target sets; rotated and unrotated use disjoint images."""
    instances = []
    for d_index, dataset in enumerate(DATASETS):
        images, labels = load_training_split(dataset)
        rng = np.random.default_rng([seed, d_index])
        classes = np.unique(labels)
        blocks = replicates * len(CONDITIONS) * 2
        chosen = {int(c): rng.permutation(np.flatnonzero(labels == c))[: blocks * PER_CLASS] for c in classes}
        block = 0
        for r in range(replicates):
            for cond, suffix in CONDITIONS.items():
                sides = []
                for _ in range(2):
                    idx = np.concatenate([chosen[int(c)][block * PER_CLASS : (block + 1) * PER_CLASS] for c in classes])
                    block += 1
                    sides.append(idx[rng.permutation(len(idx))])
                src_idx, tgt_idx = sides
                src = normalize(images[src_idx], dataset)
                tgt = normalize(images[tgt_idx], dataset)
                angles = np.zeros(len(src_idx), dtype=np.float32)
                if cond == "rotated":
                    angles = rng.uniform(0.0, 360.0, size=len(src_idx)).astype(np.float32)
                    src = rotate(src, angles)
                iid = hashlib.sha256(f"train|{seed}|{dataset}|{r}|{cond}".encode()).hexdigest()[:16]
                instances.append(
                    {
                        "id": iid,
                        "dataset": dataset,
                        "condition": cond,
                        "experiment": f"{dataset}_{suffix}",
                        "replicate": r,
                        "source": src,
                        "target": tgt,
                        "source_labels": labels[src_idx],
                        "target_labels": labels[tgt_idx],
                        "source_index": src_idx,
                        "target_index": tgt_idx,
                        "angles": angles,
                    }
                )
    return instances


def load_coupling(path: Path, n: int, m: int) -> np.ndarray:
    if not path.is_file():
        raise InvalidCoupling("missing output file")
    try:
        with np.load(path, allow_pickle=False) as z:
            if "coupling" not in z.files:
                raise InvalidCoupling("output has no 'coupling' array")
            gamma = np.asarray(z["coupling"])
    except InvalidCoupling:
        raise
    except Exception as exc:
        raise InvalidCoupling(f"unreadable output: {type(exc).__name__}") from None
    if gamma.dtype.kind not in "fiu":
        raise InvalidCoupling(f"coupling dtype {gamma.dtype} is not real-valued")
    if gamma.shape != (n, m):
        raise InvalidCoupling(f"coupling shape {gamma.shape} != {(n, m)}")
    gamma = gamma.astype(np.float64)
    if not np.all(np.isfinite(gamma)):
        raise InvalidCoupling("coupling has non-finite entries")
    total = float(gamma.sum())
    if not total > 0:
        raise InvalidCoupling("coupling has no positive mass")
    gamma = gamma / total
    if float(gamma.min()) < -NEG_TOL:
        raise InvalidCoupling("coupling has negative entries")
    gamma = np.clip(gamma, 0.0, None)
    row_err = float(np.abs(gamma.sum(axis=1) - 1.0 / n).sum())
    col_err = float(np.abs(gamma.sum(axis=0) - 1.0 / m).sum())
    if row_err > MARGINAL_TOL or col_err > MARGINAL_TOL:
        raise InvalidCoupling(f"marginals not uniform (row L1 {row_err:.3g}, col L1 {col_err:.3g}, tol {MARGINAL_TOL})")
    return gamma


def score_rows(gamma: np.ndarray, ys: np.ndarray, yt: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    classes = np.unique(np.concatenate([ys, yt]))
    lookup = {int(c): i for i, c in enumerate(classes)}
    onehot = np.zeros((len(yt), len(classes)))
    onehot[np.arange(len(yt)), [lookup[int(c)] for c in yt]] = 1.0
    row_mass = gamma.sum(axis=1, keepdims=True)
    empty = row_mass[:, 0] <= 0
    class_mass = (gamma / np.where(row_mass > 0, row_mass, 1.0)) @ onehot
    truth = np.array([lookup[int(c)] for c in ys])
    pred = class_mass.argmax(axis=1)
    rows = np.arange(len(ys))
    same = class_mass[rows, truth]
    rivals = class_mass.copy()
    rivals[rows, truth] = -np.inf
    margin = same - rivals.max(axis=1)
    correct = (pred == truth) & ~empty
    margin[empty] = -1.0
    pred = np.where(empty, -1, pred)
    return {"correct": correct.astype(np.float64), "same": same, "margin": margin}, truth, pred


def summarize(per_instance: list[dict]) -> tuple[dict, dict]:
    scores, confusion = {"experiments": {}}, {"slice": "train", "experiments": {}}
    for exp in experiment_names():
        parts = [p for p in per_instance if p["experiment"] == exp]
        correct = np.concatenate([p["rows"]["correct"] for p in parts])
        same = np.concatenate([p["rows"]["same"] for p in parts])
        margin = np.concatenate([p["rows"]["margin"] for p in parts])
        scores["experiments"][exp] = {
            "results": {
                "method": {
                    "class_preservation_accuracy": float(correct.mean()),
                    "same_class_mass_mean": float(same.mean()),
                    "class_mass_margin_mean": float(margin.mean()),
                }
            }
        }
        k = int(max(p["truth"].max() for p in parts)) + 1
        counts = np.zeros((k, k), dtype=int)
        for p in parts:
            for t, q in zip(p["truth"], p["pred"]):
                if q >= 0:
                    counts[t, q] += 1
        confusion["experiments"][exp] = {"counts": counts.tolist()}
    return scores, confusion


def raw_ot_baseline(inst: dict) -> np.ndarray:
    """Evaluator-owned comparator: raw-pixel OT with an L1 ground cost."""
    import ot
    from scipy.spatial.distance import cdist

    xs = inst["source"].reshape(len(inst["source"]), -1)
    xt = inst["target"].reshape(len(inst["target"]), -1)
    cost = cdist(xs, xt, metric="cityblock").astype(np.float64)
    cost /= max(float(cost.max()), 1e-300)
    a = np.full(len(xs), 1.0 / len(xs))
    b = np.full(len(xt), 1.0 / len(xt))
    return ot.sinkhorn(a, b, cost, reg=BASELINE_REG, method="greenkhorn", numItermax=1_000_000, stopThr=1e-9, warn=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--solver", default="scripts/solve.sh", help="Solver entry point: bash SOLVER INSTANCES_DIR OUTPUT_DIR")
    parser.add_argument("--out", default="scoring/scores_train.json")
    parser.add_argument("--status-out", default="scoring/train_status.json")
    parser.add_argument("--confusion-out", default="scoring/confusion_matrices_train.json")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Public instance seed; change it to draw fresh development instances.")
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--key-out", default=None, help="Write the answer key (labels, conditions, source indices) of these visible instances.")
    parser.add_argument("--baselines-out", default=None, help="Also score the evaluator-owned raw-pixel OT baseline here.")
    parser.add_argument("--details-out", default=None, help="Optional per-row results for paired comparisons.")
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--keep-workdir", action="store_true", help="Keep the instances and couplings for inspection.")
    args = parser.parse_args()

    out = ROOT / args.out
    status_path = ROOT / args.status_out
    write_json(out, floor_scores())
    write_json(status_path, {"status": "running"})

    instances = build_instances(args.seed, args.replicates)
    scratch_root = ROOT / "outputs"
    scratch_root.mkdir(parents=True, exist_ok=True)
    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix=".train_eval_", dir=scratch_root))
    inst_dir, sol_dir = workdir / "instances", workdir / "couplings"
    for d in (inst_dir, sol_dir):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    try:
        for inst in instances:
            np.savez(inst_dir / f"{inst['id']}.npz", source=inst["source"], target=inst["target"], dataset=np.str_(inst["dataset"]))
        if args.key_out:
            write_json(
                ROOT / args.key_out,
                {
                    i["id"]: {
                        "dataset": i["dataset"],
                        "condition": i["condition"],
                        "experiment": i["experiment"],
                        "replicate": i["replicate"],
                        "source_labels": i["source_labels"].tolist(),
                        "target_labels": i["target_labels"].tolist(),
                        "source_train_index": i["source_index"].tolist(),
                        "target_train_index": i["target_index"].tolist(),
                        "rotation_degrees": i["angles"].tolist(),
                    }
                    for i in instances
                },
            )
        started = time.time()
        proc = subprocess.run(["bash", str(ROOT / args.solver), str(inst_dir), str(sol_dir)], cwd=ROOT)
        print(f"[eval/train] solver exited with {proc.returncode} after {time.time() - started:.0f}s", flush=True)

        failures, per_instance = [], []
        for inst in instances:
            try:
                gamma = load_coupling(sol_dir / f"{inst['id']}.npz", len(inst["source"]), len(inst["target"]))
            except Exception as exc:  # any unreadable/invalid output fails the run
                failures.append({"instance": inst["id"], "experiment": inst["experiment"], "reason": str(exc)})
                continue
            rows, truth, pred = score_rows(gamma, inst["source_labels"], inst["target_labels"])
            per_instance.append({**inst, "rows": rows, "truth": truth, "pred": pred})

        status = {"solver": args.solver, "solver_returncode": proc.returncode, "seed": args.seed, "instances": len(instances), "failures": failures}
        if failures:
            status["status"] = "failed"
            write_json(out, floor_scores())  # the solver may have written to this path while it ran
            write_json(status_path, status)
            print(f"[eval/train] RUN FAILED: {len(failures)} invalid coupling(s); every metric set to its floor.", flush=True)
            for f in failures:
                print(f"  {f['instance']} ({f['experiment']}): {f['reason']}", flush=True)
            return 1

        scores, confusion = summarize(per_instance)
        write_json(out, scores)
        if args.confusion_out:
            write_json(ROOT / args.confusion_out, confusion)
        status["status"] = "ok"
        write_json(status_path, status)
        if args.details_out:
            write_json(
                ROOT / args.details_out,
                {p["id"]: {"experiment": p["experiment"], "replicate": p["replicate"], **{k: v.tolist() for k, v in p["rows"].items()}} for p in per_instance},
            )
        if args.baselines_out:
            base = []
            for inst in instances:
                rows, truth, pred = score_rows(raw_ot_baseline(inst), inst["source_labels"], inst["target_labels"])
                base.append({**inst, "rows": rows, "truth": truth, "pred": pred})
            payload, _ = summarize(base)
            for exp in payload["experiments"].values():
                exp["results"] = {"raw_ot_l1": exp["results"]["method"]}
            write_json(ROOT / args.baselines_out, payload)
        print(json.dumps(scores, indent=2))
        return 0
    finally:
        if not args.keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
