"""Score the locked A-selected explanation, without selecting candidates on labels."""
import csv
import io
from statistics import fmean

import numpy as np

from experiments.common.io import write_json
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.pilot import write_table
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import prior_run
from . import evaluate, frozen_moment as attribution, frozen_moment_observation as obs
from .r5_inputs import evaluation_labels
from .seals import verified_seal_chain


def historical_rows(scope,filename,cfg,keys,fields):
    rows = {}
    payload = scope.read_bytes(filename,'R7_FINAL').decode('utf-8')
    for r in csv.DictReader(io.StringIO(payload)):
        key = f"{r['setting']}-M{r['source']}_to_M{r['target']}-seed-{r['seed']}"
        if key in keys and float(r['tau'])==cfg['tau']:
            index = (key, *(r[k] for k in fields))
            if index in rows:
                raise PermissionError(f'Historical endpoint has a duplicate: {index}')
            rows[index] = r
    return rows


def explain(normalizer, delta, q, y, tolerance):
    r = attribution.decomposition(normalizer,delta,q,y)
    if r['identity_error'] > tolerance:
        raise ArithmeticError('R7 expected versus true loss identity failed')
    r['mass_only_flips'] = r['expected'] < -tolerance and r['mass_only'] > tolerance
    r['assignment_only_flips'] = r['expected'] < -tolerance and r['assignment_only'] > tolerance
    return r


def mean_fields(rows, names):
    return {k:fmean(x[k] for x in rows) for k in names}


def evaluate_frozen(out, dest, root, cfg, repo, obs_sha, execution):
    cases = obs.observed(out, root, cfg, obs_sha)
    with verified_seal_chain(root,'R7_FINAL') as old:
        endpoints = historical_rows(old,'endpoints.csv',cfg,cases,('pool','method'))
        moments = historical_rows(old,'moments.csv',cfg,cases,('pool',))
    metrics, decomposition, warnings, sensitivity, requests = [],[],[],[],{}
    tol = cfg['absolute_replay_tolerance']
    sign_tol = cfg['worst_tie_tolerance']
    with evaluation_labels(root,repository=repo) as labels:
        for key,(arrays,selection,role,risk) in cases.items():
            ident = obs.tasks(cfg)[key] | dict(key=key)
            requests[key] = {}
            for pool in obs.POOLS:
                ids = arrays[f'{pool}_ids']
                y = labels.load(f'blocks/{key}/manifest.json',pool,authenticated_snapshot=True)
                if len(y)!=len(ids) or not np.isin(y,[0,1]).all():
                    raise PermissionError(f'Original R7 label scope differs: {key}/{pool}')
                requests[key][pool] = ids.tolist()
                logits = {m:arrays[f'{pool}_logits_{m}'] for m in obs.METHODS}
                margin = {m:v[:,1]-v[:,0] for m,v in logits.items()}
                scores = {}
                for m,z in logits.items():
                    sc = evaluate.classification(z,y,logits['B0'].argmax(1))
                    original = endpoints[key,pool,m]
                    if original['valid']!='True' or sc['n']!=int(original['n']):
                        raise PermissionError(f'Sealed historical R7 endpoint differs: {key}/{pool}/{m}')
                    for field in ('sampled_nll','accuracy','balanced_accuracy','macro_f1',
                                  'corrected','harmed','net_corrections','class_1_count'):
                        np.testing.assert_allclose(sc[field],float(original[field]),rtol=0.,atol=tol)
                    scores[m] = sc
                    metrics.append(ident | dict(pool=pool,method=m,**sc,
                         delta_nll=sc['sampled_nll']-scores['B0']['sampled_nll'],
                         accuracy_pp=100*(sc['accuracy']-scores['B0']['accuracy'])))

                q = arrays[f'{pool}_q1']
                by_method = {}
                for m in obs.MOVES:
                    vectors = attribution.risk_vector(margin['B0'],margin[m])
                    pick = selection[m]
                    selected = explain(*vectors,q[pick+1],y,sign_tol)
                    np.testing.assert_allclose(selected['actual'],scores[m]['sampled_nll']-scores['B0']['sampled_nll'],
                                               atol=tol,rtol=0.)
                    evaluated = [r for r in risk if r['pool']==pool and r['method']==m]
                    permitted = [r for r in evaluated if m=='J' and r['candidate_id']>=0 or m=='E' and r['candidate_id']==-1]
                    worst_bound = max(r['bound_high'] for r in permitted)
                    selected_bound = next(r for r in permitted if r['candidate_id']==pick)
                    np.testing.assert_allclose(selected['expected'],selected_bound['risk'],atol=tol,rtol=0.)
                    out_row = ident | dict(pool=pool,method=m,candidate_id=pick,n=len(y),**selected,
                         selected_bound_low=selected_bound['bound_low'],selected_bound_high=selected_bound['bound_high'],
                         active_bank_bound_high=worst_bound,
                         fixed_mass_rules_out_harm=worst_bound< -sign_tol and selected['actual']>sign_tol)
                    decomposition.append(out_row)
                    by_method[m] = (out_row,evaluated,vectors)

                original = moments[key,pool]
                j, j_rows, vectors = by_method['J']
                legacy_max = max(r['risk'] for r in j_rows if r['candidate_id']>=0)
                np.testing.assert_allclose([j['actual'],legacy_max],
                    [float(original['empirical_risk_change']),float(original['retained_worst_change'])],rtol=0.,atol=tol)
                for i in selection['J_same_worst_ids']:
                    entry = explain(*vectors,q[i+1],y,sign_tol)
                    sensitivity.append(ident | dict(pool=pool,candidate_id=i,selected=i==selection['J'],
                                                   n=len(y),**entry))
                e, e_rows, _ = by_method['E']
                alternative = [r for r in e_rows if r['candidate_id']>=0]
                worst_alternative = max(r['risk'] for r in alternative)
                upper = max(r['bound_high'] for r in e_rows)
                warnings.append(ident | dict(pool=pool,n=len(y),reference_risk=e['expected'],
                    retained_max_risk=worst_alternative,retained_ids_ge_zero=' '.join(
                        str(r['candidate_id']) for r in alternative if r['risk']>=-sign_tol),
                    all_e_masses_bound_high=upper,e_true=e['actual'],j_true=j['actual'],
                    reference_predicts_improvement=e['expected']< -sign_tol,
                    other_warns_no_improvement=worst_alternative>= -sign_tol,
                    e_true_harm=e['actual']>sign_tol,j_reduces_e_harm=j['actual']<e['actual']-sign_tol and e['actual']>sign_tol,
                    j_beats_b0=j['actual']< -sign_tol))
    if requests!={k:{p:entry[2][p] for p in obs.POOLS} for k,entry in cases.items()}:
        raise PermissionError('Original R7 evaluation role receipts differ')

    summary, endpoints_summary = [],[]
    for target in cfg['targets']:
        for pool in obs.POOLS:
            for m in obs.MOVES:
                rows = [r for r in decomposition if (r['target'],r['pool'],r['method'])==(target,pool,m)]
                if len(rows)!=len(cfg['seeds']):
                    raise PermissionError('Paired 5-seed attribution is incomplete')
                summary.append(dict(target=target,pool=pool,method=m,seed_count=len(rows),
                    **mean_fields(rows,('actual','expected','error_gap','mass_term','assignment_term',
                                       'selected_bound_low','selected_bound_high','active_bank_bound_high')),
                    fixed_mass_rules_out_harm=sum(r['fixed_mass_rules_out_harm'] for r in rows),
                    mass_only_sign_flips=sum(r['mass_only_flips'] for r in rows),
                    assignment_only_sign_flips=sum(r['assignment_only_flips'] for r in rows)))
            for m in obs.METHODS:
                rows = [r for r in metrics if (r['target'],r['pool'],r['method'])==(target,pool,m)]
                endpoints_summary.append(dict(target=target,pool=pool,method=m,seed_count=len(rows),
                    **mean_fields(rows,('sampled_nll','delta_nll','accuracy','accuracy_pp','macro_f1','net_corrections'))))
    tables = dict(endpoints=metrics, attribution=decomposition, summary=summary,
                  **{'endpoint-summary':endpoints_summary,'e-warnings':warnings,'a-tie-sensitivity':sensitivity})
    records = {'score-execution': execution,'label-access': requests}
    for name,rows in tables.items():
        write_table(dest/f'{name}.csv',rows)
    for name,r in records.items():
        write_json(dest/f'{name}.json',r)
    return prior_run.seal(dest,'FINAL',[*(f'{k}.csv' for k in tables),*(f'{k}.json' for k in records)],
                          {'obs_sha256':obs_sha})
