"""Aggregate E1 by independent source model seeds, never by minibatch blocks."""
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tqg_gda.utils import sha256, write_json


def aggregate(summaries, plan, plan_hash):
    if not summaries:
        raise ValueError('No seed summaries')
    if any(s['plan_sha256'] != plan_hash or s['experiment'] != plan['experiment'] for s in summaries):
        raise ValueError('Every seed must use the exact same frozen experiment plan')
    if any(s['blocks'] != plan['blocks'] for s in summaries):
        raise ValueError('Seed did not complete the registered block budget')
    if len({s['task_fingerprint'] for s in summaries}) != 1:
        raise ValueError('Seeds used different dataset/source/target roles')
    if len({s['comparison_fingerprint'] for s in summaries}) != 1:
        raise ValueError('Seeds used different source settings, code or numerical libraries')
    seeds = [s['seed'] for s in summaries]
    if len(set(seeds)) != len(seeds) or not set(seeds) <= set(plan['source_seeds']):
        raise ValueError('Duplicate or unregistered source model seeds')
    complete = set(seeds) == set(plan['source_seeds']) and len(seeds) == plan['decision']['required_seeds']
    per_seed = [dict(seed=s['seed'], **s['target_minus_random']) for s in summaries]
    gains = [s['mean_cosine'] for s in per_seed]
    valid = all(g is not None for g in gains)
    mean_gain = statistics.mean(gains) if valid else None
    gain_std = statistics.stdev(gains) if valid and len(gains) > 1 else None
    steps = []
    for i, fraction in enumerate(plan['step_fractions']):
        values = [s['steps'][i]['mean_ce_decrease_gain'] for s in per_seed]
        target_values = [s['methods']['target']['steps'][i]['mean_ce_decrease'] for s in summaries]
        steps.append(dict(fraction=fraction, mean_ce_decrease_gain=statistics.mean(values),
            seed_std_ce_decrease_gain=statistics.stdev(values) if len(values) > 1 else None,
            positive_gain_seeds=sum(x > 0 for x in values),
            target_mean_ce_decrease=statistics.mean(target_values)))
    d = plan['decision']
    informative = all(s['uninformative_fraction'] <= d['maximum_uninformative_fraction'] for s in summaries)
    target_cosines = [s['methods']['target']['mean_cosine'] for s in summaries]
    positive = sum(g > 0 for g in gains) if valid else 0
    support = (valid and informative and mean_gain >= d['minimum_seed_mean_cosine_gain']
               and positive >= d['minimum_positive_seeds'])
    if d['require_positive_target_alignment']:
        support = support and all(x is not None for x in target_cosines) and statistics.mean(target_cosines) > 0
    if d['require_positive_loss_gain_at_all_steps']:
        support = support and all(s['mean_ce_decrease_gain'] > 0 and s['target_mean_ce_decrease'] > 0
            and s['positive_gain_seeds'] >= d['minimum_positive_seeds'] for s in steps)
    if not complete:
        decision = 'incomplete_registered_seeds'
    elif not valid or not informative:
        decision = 'inconclusive_uninformative_blocks'
    elif support:
        decision = 'initial_mechanism_support'
    elif mean_gain <= 0 and sum(g <= 0 for g in gains) >= 3:
        decision = 'target_direction_claim_unsupported_in_this_setting'
    else:
        decision = 'inconclusive_mixed_or_small_effect'
    return dict(experiment=plan['experiment'], seeds=seeds, complete_registered_seeds=complete,
        independent_unit='source_warmup_seed', per_seed=per_seed,
        mean_cosine_gain=mean_gain, seed_std_cosine_gain=gain_std,
        positive_cosine_gain_seeds=positive, steps=steps, decision=decision,
        decision_is_heuristic_not_significance_test=True, formal_paper_result=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summaries', nargs='+', required=True)
    p.add_argument('--plan', default=str(ROOT / 'configs/diagnostics/e1_target_vs_random.json'))
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if Path(args.output).exists():
        raise FileExistsError('Aggregation output already exists')
    plan = json.loads(Path(args.plan).read_text())
    summaries = [json.loads(Path(s).read_text()) for s in args.summaries]
    result = aggregate(summaries, plan, sha256(args.plan))
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
