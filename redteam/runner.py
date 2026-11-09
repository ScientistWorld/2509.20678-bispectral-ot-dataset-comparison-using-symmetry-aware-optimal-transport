"""Reproducible audit variants and explicitly named scoring shortcuts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

ROOT = Path('/home/user')
RT = ROOT / 'redteam'
DATASETS = ('mnist', 'kmnist', 'fmnist', 'emnist')
METHODS = ('raw_ot_l1', 'raw_ot_l2', 'raw_ot_l22', 'raw_ot_cos',
           'bispectral_ot_l1', 'bispectral_ot_l2', 'bispectral_ot_l22', 'bispectral_ot_cos')
METRICS = {'l1': 'cityblock', 'l2': 'euclidean', 'l22': 'sqeuclidean', 'cos': 'cosine'}
EXPERIMENTS = tuple(f'{d}_{condition}' for d in DATASETS
                    for condition in ('rotated_matching', 'unrotated_baseline'))
SCORING_FILES = ('scores_train.json', 'scores_test.json', 'eval_train_artifacts.json',
                 'confusion_matrices_train.json', 'confusion_matrices_test.json')


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def snapshot():
    state = RT / '_state'
    state.mkdir(exist_ok=True)
    for name in SCORING_FILES:
        dest = state / name
        if not dest.exists() and (ROOT / 'scoring' / name).exists():
            shutil.copyfile(ROOT / 'scoring' / name, dest)
    ref = RT / 'reference'
    ref.mkdir(exist_ok=True)
    for src, dst in [('scores_test.json', 'initial_scores_test.json'),
                     ('eval_train_artifacts.json', 'initial_artifact_index.json')]:
        if not (ref / dst).exists():
            shutil.copyfile(state / src, ref / dst)
    if not (ref / 'rank_specification.json').exists():
        shutil.copyfile(ROOT / 'scoring/reference.json', ref / 'rank_specification.json')


def restore():
    for name in SCORING_FILES:
        path = RT / '_state' / name
        if path.exists():
            shutil.copyfile(path, ROOT / 'scoring' / name)


def reference_entries():
    # Reference artifacts may be reused by the audit. Their public manifests
    # are the same interface an improving agent can produce; no test index or
    # hidden scores are used to select answers for shortcut builders.
    entries = {}
    for manifest_path in sorted((ROOT / 'outputs').glob('*/manifest.json')):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('samples_per_class') != 100 or manifest.get('seed') != 0:
            continue
        for exp, methods in manifest['experiments'].items():
            for key, artifact in methods.items():
                path = Path(artifact)
                if not path.exists():
                    path = manifest_path.parent / artifact
                if path.exists():
                    entries[(exp, key)] = str(path)
    if len(entries) != 64:
        # Standalone rerun in improve mode: produce the reusable outputs from
        # the supplied method when no reference outputs have been retained.
        for dataset in DATASETS:
            dest = ROOT / 'outputs' / f'redteam_reference_{dataset}'
            subprocess.run(['python', '-m', 'method.run_bot', '--dataset', dataset,
                            '--samples-per-class', '100', '--metrics',
                            'cityblock,euclidean,sqeuclidean,cosine',
                            '--out-dir', str(dest)], cwd=ROOT, check=True)
        return reference_entries()
    return entries


def save_artifact(folder, exp, key, gamma, source_labels, target_labels, **extra):
    dest = folder / 'artifacts' / f'{exp}__{key}.npz'
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, gamma=gamma, source_labels=source_labels,
                        target_labels=target_labels, **extra)
    return str(dest.relative_to(ROOT))


def visible_split(dataset):
    from data.idx import read_idx_labels
    from data.preprocessing import class_balanced_halves
    path = ROOT / 'data/downloads' / dataset / 'raw/train-labels-idx1-ubyte.gz'
    labels = read_idx_labels(path).astype(int)
    if dataset == 'emnist':
        labels -= 1
    src, tgt = class_balanced_halves(labels, 100, 0)
    return labels[src], labels[tgt]


def synthetic_shortcut(folder, name):
    """Shortcut inputs are exclusively the visible data/interface."""
    manifest = {'experiments': {}}
    for dataset in DATASETS:
        source_labels, target_labels = visible_split(dataset)
        n, m = len(source_labels), len(target_labels)
        k = len(np.unique(source_labels))
        if name == 'shortcut_visible_labels':
            # Shortcut: actual target-label leakage, while declaring its use.
            gamma = (source_labels[:, None] == target_labels[None, :]).astype(float)
            gamma /= gamma.sum(axis=1, keepdims=True) * n
        elif name == 'shortcut_order_answers':
            # Shortcut: labels are contiguous class blocks in both halves.
            # No semantic features or target labels are used to build gamma.
            gamma = (np.arange(n)[:, None] // 100 == np.arange(m)[None, :] // 100).astype(float)
            gamma /= 100 * n
        elif name == 'shortcut_forged_truth':
            # Shortcut: a fixed, content-independent, balanced binary coupling.
            # Replace truth with its own invented grouping; keep full array size.
            source_labels = np.arange(n) % 2
            target_labels = np.arange(m) % 2
            gamma = (source_labels[:, None] == target_labels[None, :]).astype(float)
            gamma /= gamma.sum(axis=1, keepdims=True) * n
        elif name == 'shortcut_tiny_test':
            # Shortcut: retain just two examples per class, one in each test slice.
            source_labels = np.repeat(np.arange(k), 2)
            target_labels = np.repeat(np.arange(k), 2)
            n = len(source_labels)
            gamma = (source_labels[:, None] == target_labels[None, :]).astype(float) / (2 * n)
        elif name == 'shortcut_uniform':
            gamma = np.full((n, m), 1.0 / (n * m))
        elif name == 'shortcut_zero':
            gamma = np.zeros((n, m))
        elif name == 'shortcut_signed_failed_matching':
            # Shortcut: only 10% of each test class is matched correctly, but
            # inflate those rows with signed mass. Use the visible train split
            # to balance that 10% in its complement, without reading eval/test.
            from eval.train.evaluate import split_source_indices
            # Each condition has a different visible partition, so defer below.
            gamma = None
        elif name == 'shortcut_nan':
            gamma = np.full((n, m), np.nan)
        elif name == 'shortcut_empty':
            source_labels = source_labels[:0]
            gamma = np.empty((0, m))
        else:
            raise ValueError(name)
        for condition in ('rotated_matching', 'unrotated_baseline'):
            exp = f'{dataset}_{condition}'
            if name == 'shortcut_signed_failed_matching':
                train = split_source_indices(source_labels, exp, 1729)
                in_train = np.zeros(n, dtype=bool)
                in_train[train] = True
                chosen = np.zeros(n, dtype=bool)
                for cls in np.unique(source_labels):
                    for split in (in_train, ~in_train):
                        indices = np.flatnonzero((source_labels == cls) & split)
                        chosen[indices[:len(indices) // 10]] = True
                same = source_labels[:, None] == target_labels[None, :]
                wrong = ((source_labels[:, None] + 1) % k) == target_labels[None, :]
                gamma = wrong.astype(float) / 100
                gamma[chosen] = (1_000_000 * same[chosen] - 999_999 * wrong[chosen]) / 100
            path = save_artifact(folder, exp, name, gamma, source_labels, target_labels,
                                 label_cost_used=np.array(1 if name == 'shortcut_visible_labels' else 0))
            manifest['experiments'][exp] = {key: path for key in METHODS}
    return manifest


def reference_shortcut(folder, name):
    entries = reference_entries()
    manifest = {'experiments': {}}
    for exp in EXPERIMENTS:
        methods = {}
        for key in METHODS:
            if name == 'weaker_raw_geometry':
                src_key = key.replace('bispectral_ot_', 'raw_ot_')
                path = entries[(exp, src_key)]
            elif name == 'better_control_geometry':
                src_key = key.replace('bispectral_ot_', 'raw_ot_') if exp.endswith('unrotated_baseline') else key
                path = entries[(exp, src_key)]
            elif name == 'shortcut_replace_baseline':
                src_key = key.replace('raw_ot_', 'bispectral_ot_')
                path = entries[(exp, src_key)]
            elif name == 'shortcut_remove_rotation':
                # Shortcut: relabel the unrotated comparison as rotated too.
                dataset = exp.split('_')[0]
                src_key = key.replace('bispectral_ot_', 'raw_ot_')
                path = entries[(f'{dataset}_unrotated_baseline', src_key)]
            elif name in ('shortcut_score_fields', 'reference'):
                path = entries[(exp, key)]
            else:
                raise ValueError(name)
            methods[key] = path
        manifest['experiments'][exp] = methods
    return manifest


def polar_bilinear(images, num_angles):
    # A genuine representation fix: match scipy rotation's center, interpolate
    # polar coordinates instead of asymmetric integer truncation.
    from scipy.ndimage import map_coordinates
    n, h, w = images.shape
    radii = np.arange(min(h, w) // 2, dtype=float)
    angles = 2 * np.pi * np.arange(num_angles) / num_angles
    ys = (h - 1) / 2 + radii[:, None] * np.sin(angles)[None, :]
    xs = (w - 1) / 2 + radii[:, None] * np.cos(angles)[None, :]
    idx = (np.arange(num_angles)[:, None] + np.arange(num_angles)[None, :]) % num_angles
    out = []
    for image in images:
        polar = map_coordinates(image, [ys, xs], order=1, mode='constant', cval=0, prefilter=False)
        polar *= radii[:, None]
        fft = np.fft.fft(polar, axis=1)
        bispec = fft[:, :, None] * fft[:, None, :] * np.conj(fft[:, idx])
        norm = np.linalg.norm(bispec.ravel())
        if norm > 0:
            bispec /= norm
        out.append(np.concatenate((bispec.real.ravel(), bispec.imag.ravel())).astype(np.float32))
    return np.stack(out)


def compute_costs(seed, dataset, feature_type, condition):
    """Actual feature/cost computation; no labels in costs, no surrogate."""
    from data.idx import load_training_split
    from data.preprocessing import class_balanced_halves, normalize_images, raw_features, rotate_images
    from method.bot_features import polar_bispectrum_features
    from scipy.spatial.distance import cdist
    cache = RT / 'cache' / f'seed{seed}_{dataset}_{feature_type}_{condition}'
    cache.mkdir(parents=True, exist_ok=True)
    if all((cache / f'{suffix}.npy').exists() for suffix in METRICS):
        return cache
    images, labels = load_training_split(ROOT / 'data/downloads', dataset)
    src, tgt = class_balanced_halves(labels, 100, seed)
    # Normalize only the selected samples; identical per-image preprocessing.
    source_images = normalize_images(images[src], dataset)
    target_images = normalize_images(images[tgt], dataset)
    if condition == 'rotated_matching':
        source_images, _ = rotate_images(source_images, seed + 1)
    del images
    if feature_type == 'raw':
        feature_fn = raw_features
    elif feature_type == 'reference':
        feature_fn = lambda x: polar_bispectrum_features(x, 40)
    elif feature_type == 'bilinear':
        feature_fn = lambda x: polar_bilinear(x, 40)
    else:
        raise ValueError(feature_type)
    a, b = feature_fn(source_images), feature_fn(target_images)
    for suffix, metric in METRICS.items():
        if (cache / f'{suffix}.npy').exists():
            continue
        # Reuse squared L2 for L2 when available; same ground-cost operation.
        if suffix == 'l22' and (cache / 'l2.npy').exists():
            # l2.npy is normalized: squaring commutes with max normalization.
            cost = np.load(cache / 'l2.npy') ** 2
        else:
            cost = cdist(a, b, metric=metric).astype(np.float64)
            maximum = float(cost.max())
            if maximum > 0:
                cost /= maximum
        np.save(cache / f'{suffix}.npy', cost)
    return cache


def solve_variant(folder, name):
    import ot
    from data.idx import read_idx_labels
    from data.preprocessing import class_balanced_halves
    if name.startswith('reference_seed_'):
        seed = int(name.rsplit('_', 1)[1]); reg = 0.01; proposed = 'reference'
    elif name == 'better_bilinear':
        seed = 0; reg = 0.01; proposed = 'bilinear'
    elif name == 'better_lower_reg':
        seed = 0; reg = 0.003; proposed = 'reference'
    else:
        raise ValueError(name)
    manifest_path = folder / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'experiments': {}}
    original = reference_entries()
    for dataset in DATASETS:
        labels = read_idx_labels(ROOT / 'data/downloads' / dataset / 'raw/train-labels-idx1-ubyte.gz').astype(int)
        if dataset == 'emnist': labels -= 1
        src, tgt = class_balanced_halves(labels, 100, seed)
        for condition in ('rotated_matching', 'unrotated_baseline'):
            exp = f'{dataset}_{condition}'
            manifest['experiments'].setdefault(exp, {})
            for feature in ('raw', 'proposed'):
                feature_type = 'raw' if feature == 'raw' else proposed
                # Genuine stronger representation/hyperparameter variants hold
                # the raw comparison fixed; seeds rerun every result entry.
                if feature == 'raw' and not name.startswith('reference_seed_'):
                    for suffix in METRICS:
                        key = f'raw_ot_{suffix}'
                        manifest['experiments'][exp][key] = original[(exp, key)]
                    continue
                keys = [f'{"raw_ot" if feature == "raw" else "bispectral_ot"}_{s}' for s in METRICS]
                if all(k in manifest['experiments'][exp] and (ROOT / manifest['experiments'][exp][k]).exists() for k in keys):
                    continue
                print(f'{name}: {exp} {feature_type}', flush=True)
                cache = compute_costs(seed, dataset, feature_type, condition)
                n, m = len(src), len(tgt)
                aa, bb = np.full(n, 1 / n), np.full(m, 1 / m)
                for suffix in METRICS:
                    key = f'{"raw_ot" if feature == "raw" else "bispectral_ot"}_{suffix}'
                    if key in manifest['experiments'][exp] and (ROOT / manifest['experiments'][exp][key]).exists():
                        continue
                    cost = np.load(cache / f'{suffix}.npy')
                    gamma = ot.sinkhorn(aa, bb, cost, reg=reg, method='greenkhorn',
                                        numItermax=1_000_000, stopThr=1e-9, warn=True)
                    manifest['experiments'][exp][key] = save_artifact(
                        folder, exp, key, gamma, labels[src], labels[tgt],
                        seed=np.array(seed), rotation_seed=np.array(seed + 1),
                        regularization=np.array(reg))
                    dump(manifest_path, manifest)
    return manifest


def evaluate(folder, manifest, name):
    dump(folder / 'manifest.json', manifest)
    started = time.time()
    outcome = {'name': name, 'hidden_test_arguments': [], 'train_returncode': None,
               'test_returncode': None}
    try:
        with (folder / 'train.log').open('w') as log:
            result = subprocess.run(['bash', 'scripts/evaluate_train.sh', str(folder / 'manifest.json')],
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        outcome['train_returncode'] = result.returncode
        # Explicitly execute even the empty-output shortcut through the hidden
        # test; no synthetic claim based solely on the visible evaluator.
        if (ROOT / 'scoring/scores_train.json').exists():
            shutil.copyfile(ROOT / 'scoring/scores_train.json', folder / 'scores_train.json')
        if name == 'shortcut_score_fields':
            # Shortcut attempt: forge score values/flags directly. The evaluator
            # must recalculate them from unchanged reference artifacts.
            forged = json.loads((ROOT / 'scoring/scores_train.json').read_text())
            for exp in forged['experiments'].values():
                for metrics in exp['results'].values():
                    for key in ('class_preservation_accuracy', 'mean_diagonal_class_accuracy',
                                'same_class_mass_mean', 'class_mass_margin_mean', 'target_effective_support_ratio'):
                        metrics[key] = 1.0
                    metrics['label_cost_used'] = -1000000
                    metrics['elapsed_seconds'] = -1000000
                    metrics['success'] = True
            dump(ROOT / 'scoring/scores_train.json', forged)
            dump(ROOT / 'scoring/scores_test.json', forged)
            dump(folder / 'forged_scores.json', forged)
        with (folder / 'test.log').open('w') as log:
            result = subprocess.run(['bash', 'scripts/evaluate_test.sh'], cwd=ROOT,
                                    stdout=log, stderr=subprocess.STDOUT)
        outcome['test_returncode'] = result.returncode
        if result.returncode == 0:
            shutil.copyfile(ROOT / 'scoring/scores_test.json', folder / 'scores_test.json')
            shutil.copyfile(ROOT / 'scoring/confusion_matrices_test.json', folder / 'confusion_test.json')
        else:
            outcome['test_error_tail'] = (folder / 'test.log').read_text()[-2000:]
        if outcome['train_returncode']:
            outcome['train_error_tail'] = (folder / 'train.log').read_text()[-2000:]
    finally:
        outcome['elapsed_evaluation_seconds'] = time.time() - started
        dump(folder / 'outcome.json', outcome)
        restore()
    print(json.dumps(outcome), flush=True)


def run(name):
    os.chdir(ROOT)
    snapshot()
    folder = RT / name
    folder.mkdir(exist_ok=True)
    outcome_path = folder / 'outcome.json'
    if outcome_path.exists() and (folder / 'scores_test.json').exists():
        print(f'Already saved {name}', flush=True)
        return
    if name.startswith('reference_seed_') or name in ('better_bilinear', 'better_lower_reg'):
        manifest = solve_variant(folder, name)
    elif name in ('reference', 'weaker_raw_geometry', 'better_control_geometry',
                  'shortcut_replace_baseline', 'shortcut_remove_rotation', 'shortcut_score_fields'):
        manifest = reference_shortcut(folder, name)
    else:
        manifest = synthetic_shortcut(folder, name)
    evaluate(folder, manifest, name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    args = parser.parse_args()
    run(args.name)
