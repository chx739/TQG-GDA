"""Evaluate the three WL-radius scales with exact direction-weighted gates."""
from fractions import Fraction

from experiments.common.io import write_json
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.pilot import write_table
from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import residual_inputs as data, true_rate_replay
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import prior_run, roc_score
from . import wl_channel_scale as scale


CONTROLS = ('Source', 'Common', 'Count')


def gate(setting, overall, by_source, leave):
    fixed = ['Static', *scale.R17_ARMS]
    if setting == 'proteins_edge':
        fixed.extend(scale.CURVE_ARMS)
    best = max(fixed, key=overall.__getitem__)
    gap = overall[scale.MAIN] - overall[best]
    source_wins = sum(v > 0 for v in by_source.values())
    checks = dict(half_pp=gap >= Fraction(1, 200), positive_source_groups=source_wins >= 3,
                  positive_after_any_omission=all(v > 0 for v in leave.values()))
    checks.update({f'beats_{m}': overall[scale.MAIN] > overall[m] for m in CONTROLS})
    return dict(advance=all(checks.values()), strongest_fixed=best, checks=checks, positive_sources=source_wins,
                gain_over_strongest_pp=float(100*gap), gain_over_strongest_exact=str(gap),
                by_source_vs_static_pp={str(k): float(100*v) for k, v in by_source.items()},
                leave_source_vs_static_pp={str(k): float(100*v) for k, v in leave.items()})


def evaluate(out, output, frozen, sha, blind, execution):
    pred, certs = scale.observed(out, frozen, sha)
    roles = frozen.roles
    receipt = scale.verify_blind(pred, certs, roles, blind, frozen.cfg, sha)
    print(f'Blind verification passed {receipt["graphs"]} graph trials and {receipt["exact_actions"]} actions', flush=True)
    write_json(output/'independent-verification.json', receipt)
    endpoints, pairs = [], []
    scores, deltas = {}, {}
    archive = frozen.screen.archive
    with data.ScopedLabels(archive.by_setting, roles, 'replay_E') as port:
        for key, row in roles.items():
            with port.direction(key):
                y = port.load(key, 'E', row['roles']['E'])
            x = pred[key]
            n = len(y)
            ident = {k: row[k] for k in data.IDENTITY} | dict(key=key, n=n)
            arms = scale.methods(row['setting'])
            for arm in arms:
                tn, fp, fn, tp = roc_score.confusion(x[arm], y)
                a, f = Fraction(tn+tp, n), roc_score.f1((tn, fp, fn, tp))
                scores[key, arm] = a, f
                endpoints.append(ident | dict(method=arm, tn=tn, fp=fp, fn=fn, tp=tp,
                                              predicted_positive=fp+tp, accuracy=float(a), macro_f1=float(f)))
            for arm in arms[1:]:
                counts = true_rate_replay.paired_counts(x[scale.MAIN], x[arm], y)
                a, f = (v-w for v, w in zip(scores[key, scale.MAIN], scores[key, arm], strict=True))
                if a != Fraction(counts['net'], n):
                    raise ArithmeticError('R13 correction counts differ from paired Accuracy')
                deltas[key, arm] = a, f, counts
                pairs.append(ident | dict(left=scale.MAIN, right=arm, **counts, accuracy_pp=float(100*a),
                                          macro_f1_pp=float(100*f), accuracy_exact=str(a)))
    data.validate_receipt(roles, port.actual, 'replay_E')

    summary, contrasts, decision = [], [], {}

    def aggregate(setting, keys, scope, source):
        arms = scale.methods(setting)
        stamp = dict(setting=setting, scope=scope, source=source, directions=len(keys),
                     n=sum(len(roles[k]['roles']['E']) for k in keys))
        ranked = {}
        for arm in arms:
            a, f = (sum(scores[k, arm][i] for k in keys)/len(keys) for i in (0, 1))
            ranked[arm] = a
            summary.append(stamp | dict(method=arm, accuracy=float(a), macro_f1=float(f)))
        for arm in arms[1:]:
            changes = [deltas[k, arm] for k in keys]
            a, f = (sum(d[i] for d in changes)/len(keys) for i in (0, 1))
            counts = {name: sum(p[name] for _, _, p in changes) for name in ('corrected', 'harmed', 'net', 'changed')}
            contrasts.append(stamp | dict(left=scale.MAIN, right=arm, **counts, accuracy_pp=float(100*a),
                                          macro_f1_pp=float(100*f), accuracy_exact=str(a),
                                          wins=sum(v>0 for v, _, _ in changes), ties=sum(v==0 for v, _, _ in changes),
                                          losses=sum(v<0 for v, _, _ in changes)))
        return ranked

    for setting in frozen.cfg['counts']:
        keys = [k for k, r in roles.items() if r['setting'] == setting]
        overall = aggregate(setting, keys, 'all', None)
        by_source, leave = {}, {}
        for source in archive.scope['domains']:
            same = [k for k in keys if roles[k]['source'] == source]
            other = [k for k in keys if roles[k]['source'] != source]
            for selected, scope, store in ((same, 'source', by_source), (other, 'leave_source', leave)):
                values = aggregate(setting, selected, scope, source)
                store[source] = values[scale.MAIN] - values['Static']
        decision[setting] = gate(setting, overall, by_source, leave)

    tables = dict(endpoints=endpoints, pairs=pairs, summary=summary, contrasts=contrasts)
    records = {'decision': decision, 'score-execution': execution, 'e-label-access': port.actual}
    for name, rows in tables.items():
        write_table(output/f'{name}.csv', rows)
    for name, value in records.items():
        write_json(output/f'{name}.json', value)
    print(decision, flush=True)
    return prior_run.seal(output, 'FINAL',
                          [*(f'{k}.csv' for k in tables), *(f'{k}.json' for k in records), 'independent-verification.json'],
                          {'obs_sha256': sha})
