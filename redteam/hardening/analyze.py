"""Rank every hardening attempt against the reference and measure paired test noise.

    python -m redteam.hardening.analyze

rank delta = sum_{experiment, metric} weight_e * |coef_m| * dir_m * (attempt - reference) / std_pop(e, m)
std_pop is taken over the valid (non-failed) attempts including the reference.
Paired noise: rows of every hidden-test instance are bootstrap-resampled with the
same indices for attempt and reference; the std of the rank delta over
resamples is the paired standard error.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = Path(__file__).resolve().parent / "results"
REF_JSON = json.loads((ROOT / "scoring" / "reference.json").read_text())["experiments"]
METRIC_KEYS = {"class_preservation_accuracy": "correct", "same_class_mass_mean": "same", "class_mass_margin_mean": "margin"}
GENUINE = ["reference", "bilinear", "adaptive", "combined", "raw_geometry", "lower_reg", "bot_l22"]


def load(name: str, kind: str):
    p = RES / name / f"{kind}.json"
    return json.loads(p.read_text()) if p.exists() else None


def metric_table(scores: dict) -> dict:
    return {e: v["results"]["method"] for e, v in scores["experiments"].items()}


def coords():
    out = []
    for e, spec in REF_JSON.items():
        for m, cfg in spec["metrics"].items():
            if cfg["coefficient"] == 0:
                continue
            out.append((e, m, spec["weight"] * abs(cfg["coefficient"]), 1.0 if cfg["higher_is_better"] else -1.0))
    return out


def rank(att: dict, ref: dict, sigma: dict) -> float:
    total = 0.0
    for e, m, w, d in coords():
        s = sigma[(e, m)]
        if s > 1e-12:
            total += w * d * (att[e][m] - ref[e][m]) / s
    return total


def mean_acc(tab: dict, cond: str) -> float:
    vals = [(REF_JSON[e]["weight"], v["class_preservation_accuracy"]) for e, v in tab.items() if cond in e]
    return sum(w * x for w, x in vals) / sum(w for w, _ in vals)


def bootstrap(att_details: dict, ref_details: dict, sigma: dict, reps: int = 300, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    ids = sorted(ref_details)
    draws, acc_rot, acc_unrot = [], [], []
    for _ in range(reps):
        a_tab, r_tab = {}, {}
        acc = {}
        for e in REF_JSON:
            a_rows = {k: [] for k in METRIC_KEYS.values()}
            r_rows = {k: [] for k in METRIC_KEYS.values()}
            for iid in ids:
                if ref_details[iid]["experiment"] != e:
                    continue
                n = len(ref_details[iid]["correct"])
                pick = rng.integers(0, n, n)
                for k in METRIC_KEYS.values():
                    a_rows[k].append(np.asarray(att_details[iid][k])[pick])
                    r_rows[k].append(np.asarray(ref_details[iid][k])[pick])
            a_tab[e] = {m: float(np.concatenate(a_rows[k]).mean()) for m, k in METRIC_KEYS.items()}
            r_tab[e] = {m: float(np.concatenate(r_rows[k]).mean()) for m, k in METRIC_KEYS.items()}
        draws.append(rank(a_tab, r_tab, sigma))
        acc_rot.append(mean_acc({e: {"class_preservation_accuracy": a_tab[e]["class_preservation_accuracy"] - r_tab[e]["class_preservation_accuracy"]} for e in a_tab}, "_rotated_matching"))
        acc_unrot.append(mean_acc({e: {"class_preservation_accuracy": a_tab[e]["class_preservation_accuracy"] - r_tab[e]["class_preservation_accuracy"]} for e in a_tab}, "_unrotated_baseline"))
    return {
        "rank_delta_paired_se": float(np.std(draws, ddof=1)),
        "rotated_acc_delta_paired_se": float(np.std(acc_rot, ddof=1)),
        "unrotated_acc_delta_paired_se": float(np.std(acc_unrot, ddof=1)),
    }


def main() -> None:
    names = sorted(p.name for p in RES.iterdir() if (p / "scores_test.json").exists())
    status, tables = {}, {}
    for name in names:
        st = load(name, "status_test") or {}
        tables[name] = metric_table(load(name, "scores_test"))
        status[name] = st.get("status", "unknown")
    ref = tables["reference"]
    valid = [n for n in names if status[n] == "ok"]

    def sigma_over(pop):
        return {(e, m): statistics.pstdev([tables[n][e][m] for n in pop]) for e, m, _, _ in coords()}

    sigma_valid = sigma_over(valid)
    sigma_all = sigma_over(names)
    sigma_genuine = sigma_over([n for n in GENUINE if n in valid])
    ref_details = load("reference", "details_test")
    summary = {"population_valid": valid, "failed": [n for n in names if status[n] != "ok"], "attempts": {}}
    for name in names:
        tab = tables[name]
        entry = {
            "status": status[name],
            "rank_delta_valid_population": rank(tab, ref, sigma_valid),
            "rank_delta_all_population": rank(tab, ref, sigma_all),
            "rank_delta_genuine_population": rank(tab, ref, sigma_genuine),
            "rotated_accuracy": mean_acc(tab, "_rotated_matching"),
            "unrotated_accuracy": mean_acc(tab, "_unrotated_baseline"),
            "rotated_same_class_mass": sum(REF_JSON[e]["weight"] * v["same_class_mass_mean"] for e, v in tab.items() if e.endswith("_rotated_matching"))
            / sum(REF_JSON[e]["weight"] for e in tab if e.endswith("_rotated_matching")),
        }
        det = load(name, "details_test")
        if det and ref_details and name != "reference" and status[name] == "ok":
            entry.update(bootstrap(det, ref_details, sigma_valid))
        train = load(name, "scores_train")
        if train:
            ttab = metric_table(train)
            entry["train_rotated_accuracy"] = mean_acc(ttab, "_rotated_matching")
            entry["train_unrotated_accuracy"] = mean_acc(ttab, "_unrotated_baseline")
        summary["attempts"][name] = entry

    base = load("reference", "baselines_test")
    if base:
        summary["baseline_raw_ot_l1_test"] = {e: v["results"]["raw_ot_l1"] for e, v in base["experiments"].items()}
    summary["reference_test"] = ref
    summary["reference_at_bound"] = [
        f"{e}/{m}" for e, row in ref.items() for m, v in row.items() if math.isclose(v, 1.0, abs_tol=1e-9)
    ]
    (RES.parent / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{'attempt':22} {'status':7} {'rank(valid)':>12} {'paired SE':>10} {'rank(all)':>10} {'rot acc':>8} {'unrot acc':>9} {'train rot':>9} {'train unrot':>11}")
    for name, e in sorted(summary["attempts"].items(), key=lambda kv: -kv[1]["rank_delta_valid_population"]):
        print(
            f"{name:22} {e['status']:7} {e['rank_delta_valid_population']:12.3f} {e.get('rank_delta_paired_se', float('nan')):10.3f} "
            f"{e['rank_delta_all_population']:10.3f} {e['rotated_accuracy']:8.4f} {e['unrotated_accuracy']:9.4f} "
            f"{e.get('train_rotated_accuracy', float('nan')):9.4f} {e.get('train_unrotated_accuracy', float('nan')):11.4f}"
        )


if __name__ == "__main__":
    main()
