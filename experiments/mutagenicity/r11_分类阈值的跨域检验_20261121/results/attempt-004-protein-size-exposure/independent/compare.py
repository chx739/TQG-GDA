import csv
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def records(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def seq(xs):
    return '|'.join(map(str,xs))


def verify(out,blind,obs_sha):
    obs = out/'OBS.json'
    index = blind/'SHA256SUMS'
    assert sha(obs)==obs_sha
    assert sha(index)=='f7cb241d452b64079b7b17fd28c734b270f4e3d0aa2437f12280c7b34ca947a3'
    assert sha(blind/'blind-actions.csv')=='08d1cb879bfcf610e1b930fd366572e75b1d6050a6dae5feccf540d7c39fe3ca'
    expected = json.loads(obs.read_bytes())['files']
    assert all(sha(out/k)==v for k,v in expected.items())
    for line in index.read_text().splitlines():
        h,k = line.split()
        assert sha(blind/k)==h
    roles,decisions,models = (json.loads((out/f'{k}.json').read_bytes()) for k in ('roles','decisions','models'))
    diff_p = diff_rate = 0.
    min_a = min_h = 1.
    n_graphs=actions=thresholds=0
    for f in records(blind/'blind-fits.csv'):
        key = f'proteins_edge-M{f["source"]}-seed-{f["seed"]}'
        r = models[key]
        fit = r['fit']
        for o,b in [('n_source','n'),('positive','n1'),('negative','n0'),('min_nodes','nodes_min'),('max_nodes','nodes_max')]:
            assert fit[o]==int(f[b])
        diff_rate = max(diff_rate,abs(fit['rate']-float(f['lambda_mle'])))
        assert sha256(seq(r['identity']['source_ids']).encode()).hexdigest()==f['source_ids_sha256']
    columns = {
        'B0':'b0','SourceMajority':'source_majority','Curve':'single','CurvePool':'own_pool',
        'CurveDirectP':'own_prior_direct','PoolOther1':'other1_pool','PoolOther2':'other2_pool',
        'PriorOther1':'other1_prior_direct','PriorOther2':'other2_prior_direct',
        'QuantileOther1':'other1_quantile_direct','QuantileOther2':'other2_quantile_direct'}
    for f in records(blind/'blind-actions.csv'):
        key = f['key']
        r = roles[key]
        notes = decisions[key]
        with np.load(out/'predictions'/f'{key}.npz',allow_pickle=False) as x:
            assert seq(x['ids'])==f['h_ids'] and seq(x['a_ids'])==f['a_ids']
            assert r['holdout_ids']==x['ids'].tolist() and r['selected_a_ids']==x['a_ids'].tolist()
            assert f['own_prior_direct']==f['own_quantile_direct']
            for arm,column in columns.items():
                s = ''.join('1' if p else '0' for p in x[arm])
                assert s==f[column], (key,arm,s,f[column])
                actions += len(s)
            n_graphs += len(x['ids'])
            min_h = min(min_h,float(np.min(np.abs(x['h_curve_probability']-.5))))
            for arm,col in [('CurveDirectP','own_prior'),('PriorOther1','other1_prior'),('PriorOther2','other2_prior'),
                            ('QuantileOther1','other1_quantile'),('QuantileOther2','other2_quantile')]:
                cut = notes[arm]['cut']
                threshold = {'+inf':float('inf'),'-inf':-float('inf'),'Infinity':float('inf'),'-Infinity':-float('inf')}.get(str(cut))
                if threshold is None:
                    threshold=float(cut)
                assert threshold==float(f[col+'_cut']) and notes[arm]['a_positive']==int(f[col+'_a_count'])
                thresholds += 1
            for arm,col in [('CurvePool','own'),('PoolOther1','other1'),('PoolOther2','other2')]:
                p = float(Fraction(notes[arm]['p']))
                diff_p = max(diff_p,abs(p-float(f[col+'_p'])))
                min_a = min(min_a,abs(p-.5))
    assert diff_p<1e-14 and diff_rate<1e-14
    assert n_graphs==5100 and len(roles)==60 and actions==len(columns)*n_graphs and len(models)==20
    return dict(all_match=True,obs_sha256=obs_sha,obs_members=len(expected),blind_index_sha256=sha(index),
                blind_actions_sha256=sha(blind/'blind-actions.csv'),fitted_models=len(models),blocks=len(roles),
                n_h_graph_visits=n_graphs,action_columns=len(columns),action_decisions=actions,thresholds=thresholds,
                max_prior_abs_difference=diff_p,max_lambda_abs_difference=diff_rate,
                min_abs_a_prior_distance_from_half=min_a,min_abs_h_curve_probability_distance_from_half=min_h)


if __name__=='__main__':
    out,blind,obs_sha=sys.argv[1:]
    print(json.dumps(verify(Path(out),Path(blind),obs_sha),indent=2))
