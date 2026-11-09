"""Rewrite scoring/reference.json for the hardened gym.

Keeps every paper-reported number unchanged under `_paper_results` (the paper
runs at 3,000 images per class per side; the gym at 100, so those numbers are
context, not targets). `results` holds the gym-scale reference: the paper's
method (BOT, L1) and the evaluator-owned raw-pixel OT baseline, both scored on
the hidden test. Pass the job outputs to fill them; without them the existing
values are kept.

    python3 redteam/hardening/build_reference.py [--scores S --baselines B]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "scoring" / "reference.json"

DATASET_TEXT = {
    "mnist": "MNIST digits (10 classes)",
    "kmnist": "Kuzushiji-MNIST characters (10 classes)",
    "fmnist": "Fashion-MNIST clothing (10 classes)",
    "emnist": "EMNIST letters (26 classes)",
}
METRICS = {
    "class_preservation_accuracy": {
        "higher_is_better": True,
        "coefficient": 1.0,
        "description": "Paper Table 1 quantity: fraction of source images whose transported mass, summed per target class, is largest on their own class. Recomputed by the evaluator from the submitted coupling.",
    },
    "same_class_mass_mean": {
        "higher_is_better": True,
        "coefficient": 0.4,
        "description": "Mean fraction of each source image's transported mass that lands on targets of its own class (soft class preservation, in [0, 1]).",
    },
    "class_mass_margin_mean": {
        "higher_is_better": True,
        "coefficient": 0.2,
        "description": "Mean of (own-class mass - strongest other-class mass) per source image, in [-1, 1]; separates confident from barely-correct correspondences.",
    },
}
SCALE_NOTE = (
    "Gym scale: each hidden-test instance has 100 images per class on each side (MNIST/KMNIST/FMNIST 1,000 x 1,000; "
    "EMNIST 2,600 x 2,600), two independent instances per experiment, drawn from the held-out t10k split. The paper "
    "matches two halves of the full training split (about 3,000 images per class per side), so its numbers, kept "
    "unchanged under _paper_results, are not on this scale and are not targets; `results` holds the gym-scale reference."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=None)
    parser.add_argument("--baselines", type=Path, default=None)
    args = parser.parse_args()
    old = json.loads(PATH.read_text())["experiments"]
    scores = json.loads(args.scores.read_text())["experiments"] if args.scores else None
    base = json.loads(args.baselines.read_text())["experiments"] if args.baselines else None
    new = {}
    for exp, spec in old.items():
        dataset, cond = exp.split("_", 1)
        rotated = cond == "rotated_matching"
        paper = spec.get("_paper_results", spec["results"])
        if rotated:
            desc = (
                f"Paper Table 1, {DATASET_TEXT[dataset]}: one image set is randomly rotated (uniform 0-360 degrees), "
                "the other is a disjoint unrotated set; is the transport plan class-preserving despite the rotations?"
            )
        else:
            desc = (
                f"Paper Table 1 control, {DATASET_TEXT[dataset]}: both disjoint image sets are unrotated; checks that "
                "rotation robustness is not bought by destroying ordinary matching. The solver is not told which condition it faces."
            )
        results = {}
        prev = spec.get("results", {})
        method_vals = scores[exp]["results"]["method"] if scores else prev.get("method")
        base_vals = base[exp]["results"]["raw_ot_l1"] if base else prev.get("raw_ot_l1")
        if method_vals is None:  # first conversion: placeholder from the paper until the job fills it
            method_vals = {"class_preservation_accuracy": paper["bispectral_ot_l1"]["class_preservation_accuracy"]}
            base_vals = {"class_preservation_accuracy": paper["raw_ot_l1"]["class_preservation_accuracy"]}
        results["method"] = {"type": "proposed", **{k: round(float(v), 6) for k, v in method_vals.items() if k in METRICS}}
        results["raw_ot_l1"] = {"type": "baseline", **{k: round(float(v), 6) for k, v in base_vals.items() if k in METRICS}}
        new[exp] = {
            "description": desc,
            "weight": spec["weight"],
            "primary_metric": "class_preservation_accuracy",
            "metrics": METRICS,
            "results": results,
            "_results_note": (
                "method = the paper's Bispectral OT (L1 ground cost, Greenkhorn, reg 0.01) scored by the hidden test; "
                "raw_ot_l1 = raw-pixel OT with L1 cost computed by the evaluator itself on the same instances."
            ),
            "_scale_note": SCALE_NOTE,
            "_paper_results": paper,
        }
    PATH.write_text(json.dumps({"experiments": new}, indent=2) + "\n")
    print(f"wrote {PATH}")


if __name__ == "__main__":
    main()
