"""Hidden test evaluator for symmetry-robust dataset matching.

The evaluator owns everything on the scoring side of the comparison:
  * the test images, their labels, the rotation angles and the row order all
    come from ``eval/test/pool/`` (integrity-checked by SHA-256);
  * matching instances are written WITHOUT labels and WITHOUT the condition
    (rotated or not) under opaque instance ids;
  * the submitted solver (``scripts/solve.sh`` by default) is run on them;
  * every scored number is recomputed here from the raw coupling the solver
    wrote. Nothing else the solver writes is read.

A coupling that is missing, has the wrong shape, holds non-finite or negative
entries, has no mass, or whose marginals deviate from uniform by more than
``MARGINAL_TOL`` (L1, after normalising total mass to 1) makes the whole run
FAILED: every experiment receives the floor value of every metric and the
process exits with status 1.

This file intentionally shares no code with eval/train/ or data/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
POOL_DIR = Path(__file__).resolve().parent / "pool"
POOL_SHA256 = {
    "mnist": "5e3b39007dad62c4f8abc93f9d396eb8e14fc4fe49c10f0c593ece848ca3d8ae",
    "kmnist": "4f087a7905f4eba2ca2ae30cba33eff8a9acf63e58ea1208e2b8305516441831",
    "fmnist": "a9f1d9fb39c08100ad2c8e891b744b37033f48d0e8a27293b487b8a032c33f25",
    "emnist": "d3bc340b9397ad4b98f824836982877fcda4b8262853b427844f2376cc9ff236",
}
ID_SALT = "bot-test-instance::c0a8e47d"
DATASETS = ("mnist", "kmnist", "fmnist", "emnist")
CONDITIONS = {"rotated": "rotated_matching", "unrotated": "unrotated_baseline"}
REPLICATES = 2
# Paper preprocessing: per-dataset mean/std normalisation, then rotation.
DATASET_STATS = {
    "mnist": (0.1306604787, 0.3081078053),
    "kmnist": (0.1917621507, 0.3483428300),
    "fmnist": (0.2860405969, 0.3530242445),
    "emnist": (0.1736, 0.3317),
}
MARGINAL_TOL = 1e-2
NEG_TOL = 1e-12
METRICS = ("class_preservation_accuracy", "same_class_mass_mean", "class_mass_margin_mean")
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


def normalize(images: np.ndarray, dataset: str) -> np.ndarray:
    mean, std = DATASET_STATS[dataset]
    return (images.astype(np.float32) / 255.0 - mean) / std


def rotate(images: np.ndarray, angles: np.ndarray) -> np.ndarray:
    out = np.empty_like(images, dtype=np.float32)
    for i, (img, angle) in enumerate(zip(images, angles, strict=True)):
        out[i] = ndimage.rotate(img, float(angle), reshape=False, order=1, mode="constant", cval=0.0, prefilter=False)
    return out


def load_instances() -> list[dict]:
    """Load the hidden pool into memory and build every test instance."""
    instances = []
    for dataset in DATASETS:
        path = POOL_DIR / f"{dataset}.npz"
        blob = path.read_bytes()
        if hashlib.sha256(blob).hexdigest() != POOL_SHA256[dataset]:
            raise RuntimeError(f"hidden test pool {path} failed its integrity check")
        with np.load(path, allow_pickle=False) as pool:
            arrays = {k: pool[k] for k in pool.files}
        for r in range(REPLICATES):
            for cond, suffix in CONDITIONS.items():
                src = normalize(arrays[f"r{r}_{cond}_source_images"], dataset)
                tgt = normalize(arrays[f"r{r}_{cond}_target_images"], dataset)
                if cond == "rotated":
                    src = rotate(src, arrays[f"r{r}_rotated_source_angles"])
                iid = hashlib.sha256(f"{ID_SALT}|{dataset}|{r}|{cond}".encode()).hexdigest()[:16]
                instances.append(
                    {
                        "id": iid,
                        "dataset": dataset,
                        "experiment": f"{dataset}_{suffix}",
                        "replicate": r,
                        "source": src,
                        "target": tgt,
                        "source_labels": arrays[f"r{r}_{cond}_source_labels"].astype(np.int64),
                        "target_labels": arrays[f"r{r}_{cond}_target_labels"].astype(np.int64),
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
    except Exception as exc:  # unreadable / pickled / corrupt
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


def row_scores(gamma: np.ndarray, ys: np.ndarray, yt: np.ndarray) -> dict[str, np.ndarray]:
    classes = np.unique(np.concatenate([ys, yt]))
    pos = {int(c): i for i, c in enumerate(classes)}
    onehot = np.zeros((len(yt), len(classes)))
    onehot[np.arange(len(yt)), [pos[int(c)] for c in yt]] = 1.0
    row_mass = gamma.sum(axis=1, keepdims=True)
    empty = row_mass[:, 0] <= 0
    class_mass = (gamma / np.where(row_mass > 0, row_mass, 1.0)) @ onehot
    truth = np.array([pos[int(c)] for c in ys])
    rows = np.arange(len(ys))
    same = class_mass[rows, truth]
    other = class_mass.copy()
    other[rows, truth] = -np.inf
    margin = same - other.max(axis=1)
    correct = (class_mass.argmax(axis=1) == truth) & ~empty
    margin[empty] = -1.0
    return {"correct": correct.astype(np.float64), "same": same, "margin": margin}


def aggregate(per_instance: list[dict]) -> dict:
    scores = {"experiments": {}}
    for exp in experiment_names():
        rows = [p for p in per_instance if p["experiment"] == exp]
        correct = np.concatenate([p["rows"]["correct"] for p in rows])
        same = np.concatenate([p["rows"]["same"] for p in rows])
        margin = np.concatenate([p["rows"]["margin"] for p in rows])
        scores["experiments"][exp] = {
            "results": {
                "method": {
                    "class_preservation_accuracy": float(correct.mean()),
                    "same_class_mass_mean": float(same.mean()),
                    "class_mass_margin_mean": float(margin.mean()),
                }
            }
        }
    return scores


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
    parser.add_argument("--out", default="scoring/scores_test.json")
    parser.add_argument("--status-out", default="scoring/test_status.json")
    parser.add_argument("--baselines-out", default=None, help="Also score the evaluator-owned raw-pixel OT baseline here.")
    parser.add_argument("--details-out", default=None, help="Optional per-row results for paired comparisons.")
    parser.add_argument("--workdir", default=None, help="Scratch directory (default: a fresh one under outputs/).")
    parser.add_argument("--keep-workdir", action="store_true")
    args = parser.parse_args()

    out = ROOT / args.out
    status_path = ROOT / args.status_out
    # A stale score file must never survive a failed run.
    write_json(out, floor_scores())
    write_json(status_path, {"status": "running"})

    instances = load_instances()
    scratch_root = ROOT / "outputs"
    scratch_root.mkdir(parents=True, exist_ok=True)
    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix=".test_eval_", dir=scratch_root))
    inst_dir, sol_dir = workdir / "instances", workdir / "couplings"
    for d in (inst_dir, sol_dir):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    try:
        for inst in instances:
            np.savez(inst_dir / f"{inst['id']}.npz", source=inst["source"], target=inst["target"], dataset=np.str_(inst["dataset"]))
        started = time.time()
        proc = subprocess.run(["bash", str(ROOT / args.solver), str(inst_dir), str(sol_dir)], cwd=ROOT)
        print(f"[eval/test] solver exited with {proc.returncode} after {time.time() - started:.0f}s", flush=True)

        failures, per_instance = [], []
        for inst in instances:
            n, m = len(inst["source"]), len(inst["target"])
            try:
                gamma = load_coupling(sol_dir / f"{inst['id']}.npz", n, m)
            except Exception as exc:  # any unreadable/invalid output fails the run
                failures.append({"instance": inst["id"], "reason": str(exc)})
                continue
            per_instance.append({**inst, "rows": row_scores(gamma, inst["source_labels"], inst["target_labels"])})

        status = {"solver": args.solver, "solver_returncode": proc.returncode, "instances": len(instances), "failures": failures}
        if failures:
            status["status"] = "failed"
            write_json(out, floor_scores())  # the solver may have written to this path while it ran
            write_json(status_path, status)
            print(f"[eval/test] RUN FAILED: {len(failures)} invalid coupling(s); every metric set to its floor.", flush=True)
            for f in failures:
                print(f"  {f['instance']}: {f['reason']}", flush=True)
            return 1

        scores = aggregate(per_instance)
        write_json(out, scores)
        status["status"] = "ok"
        write_json(status_path, status)
        if args.details_out:
            write_json(
                ROOT / args.details_out,
                {
                    p["id"]: {
                        "experiment": p["experiment"],
                        "replicate": p["replicate"],
                        **{k: v.tolist() for k, v in p["rows"].items()},
                    }
                    for p in per_instance
                },
            )
        if args.baselines_out:
            base = []
            for inst in instances:
                base.append({**inst, "rows": row_scores(raw_ot_baseline(inst), inst["source_labels"], inst["target_labels"])})
            payload = aggregate(base)
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
