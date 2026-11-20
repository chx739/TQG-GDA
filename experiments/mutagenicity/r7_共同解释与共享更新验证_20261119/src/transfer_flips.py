"""Transfer frozen binary E decisions to strong source classifiers on the same H."""
from fractions import Fraction
import json

import numpy as np

from experiments.common.io import write_json
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.pilot import write_table
from . import head_crossover as cross, head_crossover_scoring as scoring

BASES = ('WL', 'Size')
META = ('executed-config.json', 'roles.json')
EXACT_CONTRASTS = tuple(f'{base}OwnMinus{control}' for base in BASES
                        for control in ('Base', 'OtherMean', 'CountOwn'))


def flip(b0, e, base):
    """On a disagreement with B0, use E; otherwise keep the strong source decision."""
    return np.where(e != b0, e, base)


def transfers(actions, source, domains):
    fixed = {name:actions[name] for name in cross.FIXED}
    heads = {cross.head_name('E', ref, count=count):actions[cross.head_name('E', ref, count=count)]
             for ref in sorted(set(domains)-{source}) for count in (False, True)}
    moved = {f'{base}_{name}':flip(fixed['B0'], action, fixed[base])
             for base in BASES for name,action in heads.items()}
    return {**fixed, **heads, **moved}


def freeze(output, parent, obs_sha, c, cfg):
    roles, _, actions = scoring.observed(parent, cfg, c, obs_sha)
    (output/'predictions').mkdir()
    write_json(output/'roles.json', roles)
    for key,role in roles.items():
        arrays = transfers(actions[key], role['source'], cfg['domains'])
        prior_run.save_predictions(output, f'predictions/{key}.npz',
                                   dict(ids=np.asarray(role['holdout_ids'], dtype=np.int64), **arrays))
    return prior_run.seal(output, 'OBS', [*META, *prior_run.inventory(roles, 'predictions')],
                          dict(crossover_obs_sha256=obs_sha))


def observed(output, parent, obs_sha, c, cfg, frozen_sha):
    roles, _, inputs_by_block = scoring.observed(parent, cfg, c, obs_sha)
    blob = prior_run.verified_members(output, 'OBS', frozen_sha,
                                      [*META, *prior_run.inventory(roles, 'predictions')],
                                      dict(crossover_obs_sha256=obs_sha))
    if json.loads(blob['roles.json']) != roles:
        raise PermissionError('Transfer roles differ from the original crossover')
    actions = {}
    for key,role in roles.items():
        expected = transfers(inputs_by_block[key], role['source'], cfg['domains'])
        arrays = inputs.npz_member(blob[f'predictions/{key}.npz'])
        ids = np.asarray(role['holdout_ids'], dtype=np.int64)
        if (set(arrays) != {'ids', *expected} or arrays['ids'].dtype != ids.dtype
                or not np.array_equal(arrays['ids'], ids)):
            raise PermissionError(f'Transfer inventory or H IDs changed for {key}')
        for name, action in expected.items():
            if arrays[name].dtype != action.dtype or not np.array_equal(arrays[name], action):
                raise PermissionError(f'Frozen transfer {key}/{name} differs from its definition')
        actions[key] = {name:arrays[name] for name in expected}
    return roles, actions


def partition(b0, e, base, moved, y):
    """Count gains and harms by old decision and whether the source base agrees."""
    changed = e != b0
    agree = base == b0
    e_correct = e == y
    rows = []
    for before in (0, 1):
        for agrees in (True, False):
            mask = changed & (b0 == before) & (agree == agrees)
            rows.append(dict(old_class=before, new_class=1-before, base_agrees_b0=agrees,
                             flips=int(mask.sum()), corrected=int((mask & e_correct).sum()),
                             harmed=int((mask & ~e_correct).sum())))
    raw_gain = int((changed & e_correct).sum())
    raw_harm = int((changed & ~e_correct).sum())
    moved_gain = int(((moved == y) & (base != y)).sum())
    moved_harm = int(((moved != y) & (base == y)).sum())
    if (sum(r['flips'] for r in rows) != int(changed.sum())
            or (sum(r['corrected'] for r in rows), sum(r['harmed'] for r in rows)) != (raw_gain, raw_harm)
            or (sum(r['corrected'] for r in rows if r['base_agrees_b0']),
                sum(r['harmed'] for r in rows if r['base_agrees_b0'])) != (moved_gain, moved_harm)
            or not np.array_equal(moved != base, changed & agree)):
        raise ArithmeticError('Binary decision-transfer partition does not reconcile')
    return rows


def views(role, domains):
    target,source = role['target'],role['source']
    other = sorted(set(domains)-{target, source})
    heads = {'Own':(target,), 'Other1':(other[0],), 'Other2':(other[1],), 'OtherMean':tuple(other)}
    e = {('Count' if count else '')+name:tuple(cross.head_name('E', t, count=count) for t in refs)
         for name,refs in heads.items() for count in (False, True)}
    aliases = {m:(m,) for m in cross.FIXED}
    aliases.update({f'E{name}':v for name,v in e.items()})
    pairs = {}
    for base in BASES:
        aliases.update({f'{base}{name}':tuple(f'{base}_{m}' for m in sources) for name,sources in e.items()})
        for trial,ref in (('Own','Base'),('Own','OtherMean'),('Own','CountOwn'),
                          ('CountOwn','Base'),('CountOwn','CountOtherMean'),
                          ('Other1','Base'),('Other2','Base'),('OtherMean','Base')):
            pairs[f'{base}{trial}Minus{ref}'] = (f'{base}{trial}', base if ref=='Base' else f'{base}{ref}')
    return aliases, pairs


def exact_curve(left, right, flags, counts, order):
    if counts is None:
        return None
    net = len(right)*sum(flags[m].astype(np.int64) for m in left)
    net -= len(left)*sum(flags[m].astype(np.int64) for m in right)
    return counts@net[order], len(left)*len(right)*len(order)


def score_blocks(roles, actions, cohorts, cfg):
    endpoints, contrasts, partitions, moments = [], [], [], {}
    for key,role in roles.items():
        identity = {k:role[k] for k in cross.IDENTITY}
        y, order, cohort = cohorts[role['setting'], role['target']]
        pred = actions[key]
        aliases, pairs = views(role, cfg['domains'])
        flags = {m:a==y for m,a in pred.items()}
        cells, points, draws = scoring.metric_tables(pred, y, cohort['weights'], order)
        common = dict(**identity, complete=role['complete'], n=len(y))
        for m,a in pred.items():
            tn,fp,fn,tp = cells[m]
            endpoints.append(dict(**common, method=m, tn=tn, fp=fp, fn=fn, tp=tp, positive=int(a.sum()),
                                  accuracy=float(points[m][0]), macro_f1=float(points[m][1])))
        for base in BASES:
            for ref in sorted(set(cfg['domains'])-{role['source']}):
                for count in (False, True):
                    name = cross.head_name('E', ref, count=count)
                    for row in partition(pred['B0'],pred[name],pred[base],pred[f'{base}_{name}'],y):
                        partitions.append(dict(**common, base=base, donor=ref, own=ref==role['target'],
                                               count_matched=count, **row))
        absolute = {m:(scoring.mean_values([points[a] for a in names]),
                       scoring.sampled_mean([draws.get(a) for a in names]), None)
                    for m,names in aliases.items()}
        effects, exact = {}, {}
        for name,(l,r) in pairs.items():
            left,right = aliases[l],aliases[r]
            delta = scoring.difference(absolute[l][0], absolute[r][0])
            gain,harm,change = scoring.paired_counts(left, right, pred, flags)
            if delta[0] != (gain-harm)/len(y):
                raise ArithmeticError('Net transfer accuracy does not match the paired counts')
            rates = gain,harm,change,change/len(y)
            sample = scoring.sampled_difference(absolute[l][1], absolute[r][1])
            effects[name] = delta, sample, rates
            contrasts.append(dict(**common, contrast=name, corrected=str(gain), harmed=str(harm),
                                  changed=str(change), accuracy_exact=str(delta[0]),
                                  accuracy_pp=float(100*delta[0]), macro_f1_pp=float(100*delta[1])))
            if name in EXACT_CONTRASTS:
                exact[name] = exact_curve(left,right,flags,cohort['integer_draws'],order)
        moments[key] = dict(absolute=absolute, contrasts=effects, exact_accuracy=exact)
    return endpoints, contrasts, partitions, moments


def gate(summary, plan):
    threshold = Fraction(*plan['minimum_accuracy_fraction'])
    details = {}
    for setting,base in plan['primary_bases'].items():
        rows = [r for r in summary if r['setting']==setting and r['scope']=='complete' and r['kind']=='contrasts']
        overall = {r['name']:r for r in rows if r['source'] is None}
        own, match = (overall[f'{base}OwnMinus{arm}'] for arm in ('Base', 'OtherMean'))
        directions = [Fraction(r['accuracy_exact']) for r in rows
                      if r['source'] is not None and r['name']==f'{base}OwnMinusBase']
        gain, matching = (Fraction(r['accuracy_exact']) for r in (own,match))
        low = own['accuracy_lo_exact']
        passed = gain >= threshold and low is not None and Fraction(low) > 0 and matching > 0
        details[setting] = dict(base=base, passed=passed, own_minus_base_pp=own['accuracy_pp'],
                                ci95=[own['accuracy_lo_pp'],own['accuracy_hi_pp']], ci95_lower_exact=low,
                                own_minus_base_exact=str(gain), own_minus_other_exact=str(matching),
                                own_minus_other_pp=match['accuracy_pp'],
                                positive=sum(x>0 for x in directions), tied=sum(x==0 for x in directions),
                                negative=sum(x<0 for x in directions))
    proceed = details['proteins_edge']['passed'] and any(
        details[s]['passed'] for s in ('mutagenicity_edge','mutagenicity_node'))
    return dict(settings=details, continue_cross_setting_transfer=proceed)


def score(output, parent, obs_sha, repo, c, cfg, plan, frozen_sha):
    roles, actions = observed(output, parent, obs_sha, c, cfg, frozen_sha)
    cohorts, access, percentile = scoring.read_h_cohorts(repo, c, roles)
    endpoints, contrasts, partitions, moments = score_blocks(roles, actions, cohorts, cfg)
    summary = scoring.aggregate(cfg, roles, moments, percentile, exact_contrasts=EXACT_CONTRASTS)
    tables = {'endpoints':endpoints, 'paired-blocks':contrasts, 'flip-partition':partitions, 'summary':summary}
    for name,rows in tables.items():
        write_table(output/f'{name}.csv', rows)
    write_json(output/'gate.json', gate(summary,plan))
    write_json(output/'cohort-audit.json', [item['audit'] for _,_,item in cohorts.values()])
    write_json(output/'h-label-access.json', dict(obs_sha256=frozen_sha, actual_reads=access))
    files = ['OBS.json','gate.json','cohort-audit.json','h-label-access.json',*[f'{m}.csv' for m in tables]]
    return prior_run.seal(output,'FINAL',files,dict(obs_sha256=frozen_sha))
