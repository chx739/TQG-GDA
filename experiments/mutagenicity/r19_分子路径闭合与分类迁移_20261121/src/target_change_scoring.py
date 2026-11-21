"""Score the locked R19 U complement using its original direction identities."""
from fractions import Fraction

from experiments.common.io import write_json
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.pilot import write_table
from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import residual_inputs as data, true_rate_replay
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import prior_run, roc_score
from . import target_change as change


PAIRS = [('Delta', m) for m in change.ARMS if m != 'Delta'] + [(m, 'WL') for m in ('SourceBg', 'TargetBg', 'Raw')]


def gate(overall, sources, leave):
    baseline = max(('WL', 'Intercept'), key=overall.__getitem__)
    gain = overall['Delta'] - overall[baseline]
    positive_sources = sum(v > 0 for v in sources.values())
    checks = dict(half_pp=gain >= Fraction(1, 200), beats_count=overall['Delta'] > overall['Count'],
                  beats_mixture=overall['Delta'] > overall['MixtureDelta'], beats_reverse=overall['Delta'] > overall['Reverse'],
                  at_least_three_sources=positive_sources >= 3, leave_all_positive=all(v > 0 for v in leave.values()))
    return dict(advance=all(checks.values()), checks=checks, strongest_fixed=baseline,
                gain_over_strongest_pp=float(100*gain), gain_over_strongest_exact=str(gain),
                positive_sources=positive_sources)


def evaluate(out, stage, frozen, obs_sha, blind, execution):
    predictions = change.observed(out, frozen, obs_sha)
    verification = change.verify_blind(predictions, blind, frozen.cfg, obs_sha)
    print(f'Blind verification passed {verification["graphs"]} graph IDs and {verification["exact_actions"]} actions', flush=True)
    write_json(stage / 'independent-verification.json', verification)

    roles = change.evaluation_roles(frozen.roles)
    endpoints, paired = [], []
    metrics, comparisons = {}, {}
    with data.ScopedLabels(frozen.archive.by_setting, roles, 'replay_E') as port:
        for key, row in frozen.roles.items():
            x = predictions[key]
            with port.direction(key):
                y = port.load(key, 'E', row['supplement'])
            identity = {k: row[k] for k in data.IDENTITY} | dict(key=key, n=len(y))
            for name in change.ARMS:
                tn, fp, fn, tp = roc_score.confusion(x[name], y)
                a, f = Fraction(tn+tp, len(y)), roc_score.f1((tn, fp, fn, tp))
                metrics[key, name] = a, f
                endpoints.append(identity | dict(method=name, predicted_positive=int(x[name].sum()), tn=tn,
                                     fp=fp, fn=fn, tp=tp, accuracy=float(a), macro_f1=float(f)))
            for l, r in PAIRS:
                stats = true_rate_replay.paired_counts(x[l], x[r], y)
                a, f = (u-v for u, v in zip(metrics[key, l], metrics[key, r], strict=True))
                if a != Fraction(stats['net'], len(y)):
                    raise ArithmeticError('R19 paired counts and Accuracy disagree')
                comparisons[key, l, r] = (a, f, stats)
                paired.append(identity | dict(left=l, right=r, **stats, accuracy_pp=float(100*a), macro_f1_pp=float(100*f)))
    change.validate_score_receipt(frozen.roles, port.actual)

    summary, contrasts = [], []

    def aggregate(keys, scope, source):
        info = dict(scope=scope, source=source, directions=len(keys),
                    n=sum(len(frozen.roles[k]['supplement']) for k in keys))
        scores = {}
        for method in change.ARMS:
            a, f = (sum(metrics[k, method][i] for k in keys)/len(keys) for i in (0, 1))
            scores[method] = a
            summary.append(info | dict(method=method, accuracy=float(a), macro_f1=float(f)))
        for l, r in PAIRS:
            rows = [comparisons[k, l, r] for k in keys]
            a, f = (sum(row[i] for row in rows)/len(rows) for i in (0, 1))
            counts = {name: sum(stats[name] for _, _, stats in rows) for name in ('corrected', 'harmed', 'net', 'changed')}
            contrasts.append(info | dict(left=l, right=r, accuracy_pp=float(100*a), macro_f1_pp=float(100*f),
                                         accuracy_exact=str(a), wins=sum(v>0 for v, _, _ in rows),
                                         ties=sum(v==0 for v, _, _ in rows), losses=sum(v<0 for v, _, _ in rows), **counts))
        return scores

    overall = aggregate(list(frozen.roles), 'all', None)
    source_gains, leave_gains = {}, {}
    for source in frozen.archive.cfg['domains']:
        for keep, name, store in ((True, 'source', source_gains), (False, 'leave_source', leave_gains)):
            keys = [k for k, row in frozen.roles.items() if (row['source'] == source) == keep]
            result = aggregate(keys, name, source)
            store[source] = result['Delta'] - result['WL']
    decision = gate(overall, source_gains, leave_gains)
    tables = dict(endpoints=endpoints, pairs=paired, summary=summary, contrasts=contrasts)
    records = {'decision': decision, 'score-execution': execution,
               'label-access': {'supplement_by_direction': {k: r['E'] for k, r in port.actual.items()},
                                 'original_e_read': False, 'real_target_read': False}}
    for name, rows in tables.items():
        write_table(stage / f'{name}.csv', rows)
    for name, entry in records.items():
        write_json(stage / f'{name}.json', entry)
    print(decision, flush=True)
    return prior_run.seal(stage, 'FINAL',
                          [*(f'{k}.csv' for k in tables), *(f'{k}.json' for k in records), 'independent-verification.json'],
                          {'obs_sha256': obs_sha})
