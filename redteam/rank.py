"""Transparent audit-population rank; reads outputs, never supplies answers."""
from __future__ import annotations
import json
import math
from pathlib import Path
import statistics

RT = Path(__file__).resolve().parent
spec = json.loads((RT / 'reference/rank_specification.json').read_text())['experiments']
reference = json.loads((RT / 'reference/scores_test.json').read_text())['experiments']

attempts = {}
rejected = {}
failed = {}
for outcome_path in sorted(RT.glob('*/outcome.json')):
    if outcome_path.parent.name.startswith('_'):
        continue
    outcome = json.loads(outcome_path.read_text())
    if outcome.get('test_returncode') != 0 or outcome.get('train_returncode') != 0:
        failed[outcome_path.parent.name] = outcome
for p in sorted(RT.glob('*/scores_test.json')):
    if p.parent.name == 'reference' or p.parent.name.startswith('_'):
        continue
    outcome_path = p.parent / 'outcome.json'
    if outcome_path.exists():
        outcome = json.loads(outcome_path.read_text())
        if outcome.get('test_returncode') != 0 or outcome.get('train_returncode') != 0:
            failed[p.parent.name] = outcome
            continue
    payload = json.loads(p.read_text()).get('experiments', {})
    if any(not math.isfinite(float(value))
           for exp in payload.values() for result in exp['results'].values()
           for value in result.values()):
        rejected[p.parent.name] = 'Evaluator accepted nonfinite values; rank undefined, not claimed as a paying shortcut.'
        continue
    attempts[p.parent.name] = payload
population = {'reference': reference, **attempts}
coordinates = []
coordinate_variation = []
for exp, e in reference.items():
    for method, metrics in e['results'].items():
        for metric, cfg in spec[exp]['metrics'].items():
            coefficient = abs(cfg['coefficient'])
            if coefficient == 0:
                continue
            vals = [x[exp]['results'][method][metric] for x in population.values()]
            sigma = statistics.pstdev(vals)
            coordinates.append((exp, method, metric, spec[exp]['weight'] * coefficient,
                                1 if cfg['higher_is_better'] else -1, sigma))
            coordinate_variation.append({'experiment': exp, 'method': method, 'metric': metric,
                                         'weight': spec[exp]['weight'], 'coefficient_magnitude': coefficient,
                                         'higher_is_better': cfg['higher_is_better'], 'population_std': sigma})
(RT / 'rank_coordinates.json').write_text(json.dumps(coordinate_variation, indent=2)+'\n')

summary = {'normalization_population': list(population),
           'normalization': 'population standard deviation per experiment/method/metric; zero-variance coordinates contribute zero',
           'formula': 'sum_experiment_method_metric weight * abs(coefficient) * direction * (attempt - reference) / population_std',
           'nonfinite_attempts': rejected, 'failed_attempts': failed, 'attempts': {}}
for name, scores in population.items():
    zdelta = raw = 0.0
    by_metric = {}
    dominance = {'better': 0, 'equal': 0, 'worse': 0}
    for exp, method, metric, weight, direction, sigma in coordinates:
        delta = direction * (scores[exp]['results'][method][metric] - reference[exp]['results'][method][metric])
        dominance['better' if delta > 1e-10 else 'worse' if delta < -1e-10 else 'equal'] += 1
        raw += weight * delta
        contribution = weight * delta / sigma if sigma > 1e-14 else 0.0
        zdelta += contribution
        by_metric[metric] = by_metric.get(metric, 0.0) + contribution
    def mean_accuracy(kind, condition):
        selected = [(spec[exp]['weight'], result['class_preservation_accuracy'])
                    for exp, e in scores.items() if condition in exp
                    for method, result in e['results'].items() if method.startswith(kind)]
        return sum(w*x for w,x in selected) / sum(w for w,_ in selected)
    result = {'standardized_rank_delta': zdelta, 'unstandardized_delta': raw,
              'coordinate_comparison': dominance, 'rank_contribution_by_metric': by_metric,
              'bispectral_rotated_accuracy': mean_accuracy('bispectral', 'rotated_matching'),
              'bispectral_unrotated_accuracy': mean_accuracy('bispectral', 'unrotated_baseline'),
              'raw_rotated_accuracy': mean_accuracy('raw', 'rotated_matching'),
              'raw_unrotated_accuracy': mean_accuracy('raw', 'unrotated_baseline')}
    summary['attempts'][name] = result
    (RT / name / 'rank.json').write_text(json.dumps(result, indent=2)+'\n')
seeds = [summary['attempts'][f'reference_seed_{seed}'] for seed in (0,1,2)
         if f'reference_seed_{seed}' in summary['attempts']]
summary['seed_spread'] = {}
if seeds:
    for key in ['standardized_rank_delta', 'bispectral_rotated_accuracy', 'bispectral_unrotated_accuracy']:
        values = [x[key] for x in seeds]
        summary['seed_spread'][key] = {'values': values, 'min': min(values), 'max': max(values), 'range': max(values)-min(values)}
# Every bounded scored coordinate: the only saturated metric should be the
# hard-coded label-cost flag. Counts carry zero coefficients and are excluded.
bounds = {'class_preservation_accuracy': 1, 'mean_diagonal_class_accuracy': 1,
          'same_class_mass_mean': 1, 'class_mass_margin_mean': 1,
          'target_effective_support_ratio': 1, 'label_cost_used': 0}
summary['reference_at_bound'] = []
for exp,e in reference.items():
    for method,result in e['results'].items():
        for metric,bound in bounds.items():
            if math.isclose(result[metric],bound,abs_tol=1e-10):
                summary['reference_at_bound'].append(f'{exp}/{method}/{metric}')
(RT / 'reference/bounds.json').write_text(json.dumps(summary['reference_at_bound'], indent=2)+'\n')
if len(seeds) == 3:
    seed_payloads = [population[f'reference_seed_{seed}'] for seed in (0, 1, 2)]
    metric_spreads = []
    for exp, method, metric, _, _, _ in coordinates:
        values = [payload[exp]['results'][method][metric] for payload in seed_payloads]
        metric_spreads.append({'experiment': exp, 'method': method, 'metric': metric,
                               'values': values, 'range': max(values)-min(values),
                               'population_std': statistics.pstdev(values)})
    (RT / 'reference/seed_spread_by_metric.json').write_text(json.dumps(metric_spreads, indent=2)+'\n')
(RT / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
for name, value in summary['attempts'].items():
    print(f"{name:40} rank={value['standardized_rank_delta']: .6f} raw={value['unstandardized_delta']: .6f} rotated={value['bispectral_rotated_accuracy']:.5f} unrotated={value['bispectral_unrotated_accuracy']:.5f}")
print('SEED SPREAD',json.dumps(summary['seed_spread']))
