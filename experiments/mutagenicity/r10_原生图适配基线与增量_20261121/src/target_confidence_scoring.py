"""Pair ATC's fixed H actions using only published two-class vote counts."""
import csv
from fractions import Fraction
import io
import json
import runpy

import numpy as np

from experiments.common.io import digest, write_json
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.pilot import write_table
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run, roc_score, roc_transfer
from experiments.mutagenicity.r12_三种图读数与阈值识别_20261121.src import reverse_vote_scoring as vote_score
from . import target_confidence as atc


INCUMBENT = {'mutagenicity_edge':'WL', 'mutagenicity_node':'WL', 'proteins_edge':'Size'}
CURVES = ('Curve', 'CurvePool')
POLICY = (*atc.MODELS, *atc.CHOOSERS, 'OtherATCMean', 'Oracle')


def verify_blind(script, archive, decisions, blocks, obs_sha):
    if digest(script) != archive.cfg['blind_script_sha256']:
        raise PermissionError('Independent no-label script differs')
    stats = runpy.run_path(str(script))['stats']
    actual = {}
    verified = set()
    for setting, records in stats.items():
        for source,target,sel,vsel,anchor,changed,n,_,pr in records:
            key = inputs.key(setting=setting, source=source, target=target, seed=archive.original['pilot_seed'])
            decision = decisions[key]
            roles = blocks[key]
            own_model = decision['choices']['OwnATC']
            np.testing.assert_array_equal([float(Fraction(decision['estimates']['OwnATC'][m])) for m in atc.MODELS], pr)
            if (own_model != atc.MODELS[sel] or decision['choices']['SourceV'] != atc.MODELS[vsel]
                    or INCUMBENT[setting] != atc.MODELS[anchor] or n != len(roles['h_ids'])
                    or changed != sum(atc.BITS[own_model][z] != atc.BITS[INCUMBENT[setting]][z] for z in roles['h_bins'])
                    or key in verified):
                raise PermissionError(f'Independent ATC selections differ: {key}')
            verified.add(key)
            actual[setting] = actual.get(setting, 0) + int(changed>0)
    if verified != set(decisions) or actual != archive.cfg['registered_changed_directions']:
        raise PermissionError('Blind ATC direction inventory differs')
    return dict(obs_sha256=obs_sha, independent_script_sha256=archive.cfg['blind_script_sha256'],
                directions=len(verified), exact_own_estimates=3*len(verified),
                source_and_own_policies=len(verified)*2, directions_changed_vs_incumbent=actual)


def csv_member(path, files, name):
    return csv.DictReader(io.StringIO(prior_run.read_member(path, files, name).decode('utf-8')))


def parse_key(row):
    return inputs.key(setting=row['setting'], source=int(row['source']), target=int(row['target']), seed=int(row['seed']))


def confusion(cells, model):
    return vote_score.confusion(cells, atc.BITS[model])


def metrics(cells):
    tn, fp, fn, tp = cells
    return Fraction(tn+tp, sum(cells)), roc_score.f1(cells)


def published_cells(archive, blocks):
    cfg, root = archive.reference, archive.r12
    predecessor = {'obs_sha256': cfg['historical_obs_sha256']}
    final = roc_transfer.stage(root, 'FINAL', cfg['historical_final_sha256'])
    if final['predecessors'] != predecessor or final['files']['OBS.json'] != predecessor['obs_sha256']:
        raise PermissionError('Published H cells belong to a different R12 OBS')
    cells = {k: {} for k in blocks}
    for row in csv_member(root, final['files'], 'h-cells.csv'):
        key = parse_key(row)
        bits = [int(row[k]) for k in ('size','b0','wl')]
        n0,n1 = (int(row[k]) for k in ('true_0','true_1'))
        z = 4*bits[0] + 2*bits[1] + bits[2]
        if (any(b not in (0,1) for b in bits) or z in cells[key] or min(n0,n1)<0
                or n0+n1 != int(row['graphs']) or n0+n1 != int((blocks[key]['h_bins']==z).sum())):
            raise PermissionError(f'Published H vote cell differs: {key}/{z}')
        cells[key][z] = (n0,n1)
    if any(set(m) != set(range(8)) for m in cells.values()):
        raise PermissionError('Published H cell inventory incomplete')
    cells = {k: [m[i] for i in range(8)] for k,m in cells.items()}
    r10_root = archive.here / 'results/attempt-003-structural'
    r10 = roc_transfer.stage(r10_root, 'FINAL', archive.cfg['r10_final_sha256'])
    if r10['predecessors'] != {'obs_sha256': archive.original['r10_obs_sha256']}:
        raise PermissionError('Published R10 endpoints belong to a different holdout OBS')
    found = set()
    names = dict(B0='B0', WL_selected='WL', Size='Size')
    for row in csv_member(r10_root, r10['files'], 'endpoints.csv'):
        key = parse_key(row)
        if key not in cells or row['method'] not in names:
            continue
        arm = names[row['method']]
        result = confusion(cells[key], arm)
        a,f = metrics(result)
        if ((key,arm) in found or sum(result) != int(row['n']) or sum(result[2:]) != int(row['class_1_count'])
                or abs(float(a)-float(row['accuracy'])) > 1e-12 or abs(float(f)-float(row['macro_f1'])) > 1e-12
                or abs((result[1]+result[3])/sum(result)-float(row['predicted_class_1'])) > 1e-12):
            raise PermissionError(f'R12 vote cells and R10 endpoint disagree: {key}/{arm}')
        found.add((key,arm))
    if len(found) != 3*len(cells):
        raise PermissionError('R10 published endpoint grid does not cover this experiment')
    return cells, dict(r12_final_sha256=cfg['historical_final_sha256'], h_cells_sha256=final['files']['h-cells.csv'],
                       r10_final_sha256=archive.cfg['r10_final_sha256'], r10_endpoints_sha256=r10['files']['endpoints.csv'])


def published_curves(archive, cells):
    cfg = archive.cfg
    root = archive.here.parent / atc.R11 / 'results/attempt-004-protein-size-exposure'
    obs = roc_transfer.stage(root, 'OBS', cfg['r11_curve_obs_sha256'])
    final = roc_transfer.stage(root/'score', 'FINAL', cfg['r11_curve_final_sha256'])
    if (final['predecessors'] != {'obs_sha256': cfg['r11_curve_obs_sha256']}
            or obs['predecessors']['structural_obs_sha256'] != archive.original['r10_obs_sha256']):
        raise PermissionError('Published curves refer to different structural predictions')
    role_data = json.loads(prior_run.read_member(root, obs['files'], 'roles.json'))
    matches = {k: r for k,r in archive.roles.items() if r['setting']=='proteins_edge'}
    budget = []
    for key,r in matches.items():
        old = role_data[key]
        if old['holdout_ids'] != r['holdout_ids'] or old['manifest_sha256'] != r['manifest_sha256']:
            raise PermissionError('Published R11 curve does not have the same H IDs')
        budget.append({f:r[f] for f in atc.ID} | dict(atc_a=len(r['selected_a_ids']),
            curve_a=len(old['selected_a_ids']), exact_a_match=old['selected_a_ids']==r['selected_a_ids']))
    scores = {k:{} for k in matches}
    for row in csv_member(root/'score', final['files'], 'endpoints.csv'):
        key,method = parse_key(row),row['method']
        if key not in matches or method not in ('B0','Size',*CURVES):
            continue
        vals = tuple(int(row[k]) for k in ('tn','fp','fn','tp'))
        if (min(vals) < 0 or sum(vals)!=len(matches[key]['holdout_ids'])
                or sum(vals[2:])!=sum(c[1] for c in cells[key]) or method in scores[key]
                or (method in ('B0','Size') and vals != confusion(cells[key], method))):
            raise PermissionError(f'Published curve or R11 baseline endpoint differs: {key}/{method}')
        scores[key][method] = vals
    if any(set(v) != {'B0','Size',*CURVES} for v in scores.values()):
        raise PermissionError('Published R11 curves lack a complete pilot grid')
    return scores,budget,dict(r11_curve_obs_sha256=cfg['r11_curve_obs_sha256'],
        r11_curve_final_sha256=cfg['r11_curve_final_sha256'], r11_curve_endpoints_sha256=final['files']['endpoints.csv'])


def mean(values):
    kept = list(values)
    return sum(kept)/len(kept)


def score_policies(cells, selection):
    hard = {m:(m,) for m in atc.MODELS} | {n:(m,) for n,m in selection.items()}
    hard['OtherATCMean'] = (selection['OtherATC1'],selection['OtherATC2'])
    winner = max(atc.PRIORITY, key=lambda m: metrics(confusion(cells, m))[0])
    hard['Oracle'] = (winner,)
    scores = {p:tuple(mean(metrics(confusion(cells,m))[i] for m in hard[p]) for i in (0,1)) for p in hard}
    return hard,scores


def paired_counts(cells, left, right):
    # OtherATCMean averages two realized policy scores and their paired counts.
    combinations = [vote_score.pairs(cells, atc.BITS[l], atc.BITS[r]) for l in left for r in right]
    return tuple(mean(c[i] for c in combinations) for i in (0,1))


def evaluate(out, dest, archive, obs_sha, blind, execution):
    blocks, selected = atc.observed(out, archive, obs_sha)
    receipt = verify_blind(blind, archive, selected, blocks, obs_sha)
    print(f'Independent ATC check passed {receipt["directions"]} directions', flush=True)
    cells, sources = published_cells(archive, blocks)
    curves,budget,curve_sources = published_curves(archive, cells)
    sources |= curve_sources

    roles, points, pairs, scores, differences = archive.roles, [], [], {}, {}
    for key,r in roles.items():
        n = len(r['holdout_ids'])
        context = {f:r[f] for f in atc.ID} | dict(n=n)
        policy, sm = score_policies(cells[key], selected[key]['choices'])
        scores[key] = sm | {m:metrics(curves[key][m]) for m in CURVES if key in curves}
        for name,(a,f) in scores[key].items():
            choice = ';'.join(policy.get(name, (name,)))
            points.append(context | dict(method=name, selected=choice, accuracy=float(a), macro_f1=float(f),
                                         accuracy_exact=str(a), macro_f1_exact=str(f)))
        anchor = INCUMBENT[r['setting']]
        contrasts = dict.fromkeys((*atc.CHOOSERS,'OtherATCMean','Oracle'), anchor)
        comparisons = list(contrasts.items()) + [('OwnATC',m) for m in (*atc.MODELS,'SourceV','OtherATC1','OtherATC2','OtherATCMean')
                                                  if m != anchor]
        by_pair = {}
        for left,right in comparisons:
            corr,harm = paired_counts(cells[key],policy[left],policy[right])
            a,f = (x-y for x,y in zip(sm[left],sm[right],strict=True))
            if a != (corr-harm)/n:
                raise ArithmeticError('ATC paired errors disagree with Accuracy')
            by_pair[left,right] = (a,f,corr,harm)
            pairs.append(context | dict(left=left,right=right,corrected=float(corr),harmed=float(harm),
                                        accuracy_pp=float(100*a),accuracy_exact=str(a),macro_f1_pp=float(100*f)))
        differences[key] = by_pair

    summary, paired = [], []
    for setting in archive.original['settings']:
        keys = [k for k,r in roles.items() if r['setting']==setting]
        for removed in (None,*archive.original['domains']):
            kept = [k for k in keys if roles[k]['target'] != removed]
            identity = dict(setting=setting, omitted_target=removed, directions=len(kept),
                            graph_trials=sum(len(roles[k]['holdout_ids']) for k in kept))
            for m in (*POLICY, *(CURVES if setting=='proteins_edge' else ())):
                a,f = (mean(scores[k][m][i] for k in kept) for i in (0,1))
                summary.append(identity | dict(method=m,accuracy=float(a),macro_f1=float(f),accuracy_exact=str(a)))
            for left,right in differences[kept[0]]:
                joined = [differences[k][left,right] for k in kept]
                a,f = (mean(x[i] for x in joined) for i in (0,1))
                corr,harm = (sum(x[i] for x in joined) for i in (2,3))
                paired.append(identity | dict(left=left,right=right,corrected=float(corr),harmed=float(harm),
                    accuracy_pp=float(100*a),macro_f1_pp=float(100*f),accuracy_exact=str(a),
                    wins=sum(v>0 for v,_,_,_ in joined),ties=sum(v==0 for v,_,_,_ in joined),losses=sum(v<0 for v,_,_,_ in joined)))
    tables = {'endpoints':points,'pairs':pairs,'summary':summary,'paired-summary':paired,'curve-budgets':budget}
    records = {'independent-verification':receipt, 'score-execution':execution, 'aggregate-inputs':sources}
    for name,rows in tables.items():
        write_table(dest/f'{name}.csv',rows)
    for name,obj in records.items():
        write_json(dest/f'{name}.json',obj)
    return prior_run.seal(dest,'FINAL', [*(f'{name}.csv' for name in tables), *(f'{name}.json' for name in records)],
                          {'obs_sha256':obs_sha})
