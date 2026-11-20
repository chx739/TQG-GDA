"""Score frozen four-cell rules from the published R12 aggregates only."""
import csv
from fractions import Fraction as F
import json

import numpy as np

from experiments.common.io import write_json
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.pilot import write_table
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run, roc_score, roc_transfer
from . import conditional_votes as previous, reverse_vote_observation as experiment, reverse_votes as rules

METHODS = (*experiment.POLICIES,*previous.FIXED,'OtherMean','OracleDirection','OracleShared')
PAIRED = ('Anchor','Direct','PermMean','RowPerm','Other1','Other2','OtherMean','CountMatched',*previous.FIXED)
PHASE_GATE = ('Direct','PermMean','OtherMean','CountMatched',*previous.FIXED)


def mean(values):
    values = list(values)
    return sum(values,F(0))/len(values)


def identity(row):
    return {k:row[k] for k in previous.IDENTITY}


def task_key(row):
    return inputs.key(row['setting'],*[int(row[k]) for k in previous.IDENTITY[1:]])


def verify(out,root,cfg,old_cfg,sha):
    p = experiment.plan(cfg,old_cfg)
    keys = prior_run.registered_keys(p)
    obs = prior_run.verified_members(out,'OBS',sha,experiment.members(keys),experiment.parents(cfg))
    old,roles,sources = experiment.historical_obs(root,cfg,old_cfg)
    if json.loads(obs['roles.json'])!=roles or json.loads(obs['executed-config.json'])['plan']!=cfg:
        raise PermissionError('Current OBS has different roles or plan')
    fits = json.loads(obs['policies.json'])
    receipts = json.loads(obs['source-receipts.json'])
    if set(fits)!=set(keys) or set(receipts)!=set(sources):
        raise PermissionError('OBS is missing complete candidate tables or source role records')
    eight_cells = {}
    for key in keys:
        r = roles[key]
        data,a,h = experiment.historical_votes(old[f'predictions/{key}.npz'],r)
        frozen = inputs.npz_member(obs[f'inputs/{key}.npz'])
        group = r['group']
        s = sources[group]
        for field,expected in (('v_ids',s['v_ids']),('h_ids',data['ids']),('a_ids',data['a_ids']),
                               ('h_z',h),('a_z',a)):
            np.testing.assert_array_equal(frozen[field],expected)
        receipt = receipts[group]
        if any(receipt[k]!=s[k] for k in ('v_ids','v_label_sha256','r16_fit_member_sha256')):
            raise PermissionError('Current OBS source role differs from the published V')
        expected_v = [[sum(s['counts'][y][z%2][b*2+z//2] for b in (0,1)) for y in (0,1)]
                      for z in rules.CELLS]
        row = fits[key]
        if (row['v_counts']!=expected_v or rules.tally(a)!=row['a_counts']
                or rules.tally(frozen['v_z'])!=[sum(count) for count in expected_v]):
            raise PermissionError('Current OBS vote counts do not match the frozen source and target vote margins')
        for method in rules.LOCAL_ARMS:
            value = row['fits'][method]
            if len(value['brier'])!=16:
                raise PermissionError('Expected a loss for every candidate table')
            losses = [F(x) for x in value['brier']]
            if any(not 0<=x<=1 for x in losses):
                raise PermissionError('Invalid exact Brier loss')
            anchor = rules.anchor_number(r['setting'])
            if value!=rules.record(losses,anchor):
                raise PermissionError('Archived table violates exact selection rules')
        donor = [fits[inputs.key(r['setting'],r['source'],t,r['seed'])] for t in r['other_targets']]
        if row['deployment']!=rules.apply(row,donor,rules.tally(h),anchor):
            raise PermissionError('Frozen H actions do not match candidate tables')
        # Keep all eight margins to detect an accidental B0 or Size bit swap.
        eight_cells[key] = [int(np.count_nonzero(data['h_bins']==j)) for j in range(8)]
    return roles,fits,eight_cells


def published_rows(root,cfg,roles,expected):
    seal = roc_transfer.stage(root,'FINAL',cfg['historical_final_sha256'])
    parents = dict(obs_sha256=cfg['historical_obs_sha256'])
    if seal['predecessors']!=parents or seal['files']['OBS.json']!=parents['obs_sha256']:
        raise PermissionError('Historical aggregate scores have a different OBS ancestor')
    names = ('h-cells.csv','endpoints.csv')
    csv_data = {name:prior_run.read_member(root,seal['files'],name) for name in names}
    cells = {key:{} for key in roles}
    for row in csv.DictReader(csv_data['h-cells.csv'].decode('utf-8').splitlines()):
        key = task_key(row)
        size,b0,wl = [int(row[k]) for k in ('size','b0','wl')]
        idx = 4*size + 2*b0 + wl
        counts = [int(row[k]) for k in ('true_0','true_1')]
        if (any(j not in (0,1) for j in (size,b0,wl)) or idx in cells[key]
                or any(n<0 for n in counts) or sum(counts)!=int(row['graphs'])
                or sum(counts)!=expected[key][idx]):
            raise PermissionError('Historical H cell coverage or count differs from the OBS votes')
        cells[key][idx] = counts
    fixed = {key:{} for key in roles}
    for row in csv.DictReader(csv_data['endpoints.csv'].decode('utf-8').splitlines()):
        method = row['method']
        if method not in previous.FIXED:
            continue
        key = task_key(row)
        vals = tuple(int(row[k]) for k in ('tn','fp','fn','tp'))
        if method in fixed[key] or min(vals)<0 or sum(vals)!=len(roles[key]['holdout_ids']):
            raise PermissionError('Historical fixed endpoint differs from the H cohort')
        fixed[key][method] = vals
    for key in roles:
        src = cells[key]
        if set(src)!=set(range(8)) or set(fixed[key])!=set(previous.FIXED):
            raise PermissionError('Historical FINAL lacks complete H cells or endpoints')
        subtotal = sum(src[idx][1] for idx in src)
        for method,vals in fixed[key].items():
            if vals[2]+vals[3]!=subtotal:
                raise PermissionError('Published endpoint has a different positive label count')
        for method,bit in (('B0',1),('WL',0),('Size',2)):
            mask = [(idx>>bit)&1 for idx in range(8)]
            if confusion([src[i] for i in range(8)],mask)!=fixed[key][method]:
                raise PermissionError('Published H cells do not reconstruct historical hard decisions')
    return ({k:[[sum(cells[k][4*(z%2)+2*b+z//2][y] for b in (0,1))
                 for y in (0,1)] for z in rules.CELLS] for k in roles},
            fixed,{name:seal['files'][name] for name in names})


def confusion(n_by_y, q):
    return tuple(sum(n[y]*(p if action else 1-p) for n,p in zip(n_by_y,q,strict=True))
                 for y in (0,1) for action in (0,1))


def pairs(n_by_y,left,right):
    good = bad = F(0)
    for (n0,n1),p,q in zip(n_by_y,left,right,strict=True):
        if p>q:
            good += (p-q)*n1
            bad += (p-q)*n0
        elif p<q:
            good += (q-p)*n0
            bad += (q-p)*n1
    return good,bad


def accuracy(cells):
    return F(cells[0]+cells[-1],sum(cells))


def classification_metrics(cells):
    tn,fp,fn,tp = cells
    return (accuracy(cells),roc_score.f1(cells),
            F(tn,tn+fp) if tn+fp else None,F(tp,tp+fn) if tp+fn else None)


def metric_means(rows):
    return tuple(mean(column) if None not in column else None for column in zip(*rows,strict=True))


def recalls(recall_0,recall_1):
    return {key:float(value) if value is not None else None
            for key,value in (('recall_0',recall_0),('recall_1',recall_1))}


def oracle_losses(cells):
    return [1-accuracy(confusion(cells,q)) for q in rules.TABLES]


def score_blocks(roles,fits,cells,fixed):
    oracle = {key:oracle_losses(cells[key]) for key in roles}
    shared = {}
    for group in sorted({r['group'] for r in roles.values()}):
        keys = [k for k,r in roles.items() if r['group']==group]
        anchor = rules.anchor_number(roles[keys[0]]['setting'])
        losses = [mean(oracle[key][i] for key in keys) for i in range(16)]
        shared[group] = rules.choose(losses,anchor)[0]
    endpoints,paired,ceilings,points = [],[],[],{}
    for key,role in roles.items():
        r = identity(role)
        eligible = cells[key]
        n = sum(map(sum,eligible))
        deployment = fits[key]['deployment']
        p = {arm:[F(x) for x in q] for arm,q in deployment['policies'].items()}
        anchor = rules.anchor_number(role['setting'])
        best = rules.choose(oracle[key],anchor)[0]
        common = shared[role['group']]
        p.update(OracleDirection=rules.TABLES[best],OracleShared=rules.TABLES[common])
        all_cells = {**{arm:confusion(eligible,q) for arm,q in p.items()},**fixed[key]}
        scores = {arm:classification_metrics(counts) for arm,counts in all_cells.items()}
        scores['OtherMean'] = metric_means([scores['Other1'],scores['Other2']])
        points[key] = scores
        for arm in METHODS:
            a,f,r0,r1 = scores[arm]
            vals = all_cells.get(arm)
            endpoints.append(dict(**r,method=arm,n=n,accuracy_exact=str(a),accuracy=float(a),
                                  macro_f1=float(f),macro_f1_exact=str(f),**recalls(r0,r1),
                                  **({k:str(v) for k,v in zip(('tn','fp','fn','tp'),vals,strict=True)}
                                     if vals is not None else {})))
        for arm in PAIRED:
            delta = scores['Reverse'][0]-scores[arm][0]
            if arm=='OtherMean':
                good,bad = [mean(x) for x in zip(pairs(eligible,p['Reverse'],p['Other1']),
                                                pairs(eligible,p['Reverse'],p['Other2']),strict=True)]
            elif arm in p:
                good,bad = pairs(eligible,p['Reverse'],p[arm])
            else:
                # Only summary endpoint counts are authorized for these fixed arms.
                good = bad = None
            if good is not None and (good-bad)/n!=delta:
                raise ValueError('Paired corrected and harmed counts disagree with absolute endpoints')
            paired.append(dict(**r,left='Reverse',right=arm,n=n,corrected=good,harmed=bad,
                               accuracy_exact=str(delta),accuracy_pp=float(100*delta)))
        ceilings.append(dict(**r,reverse_table=deployment['tables']['Reverse'],
                             anchor_table=anchor,oracle_direction_table=best,oracle_shared_table=common,
                             oracle_direction_accuracy=float(scores['OracleDirection'][0]),
                             oracle_shared_accuracy=float(scores['OracleShared'][0]),
                             reverse_accuracy=float(scores['Reverse'][0]),
                             representation_headroom_pp=float(100*(scores['OracleDirection'][0]-scores['Anchor'][0])),
                             shared_headroom_pp=float(100*(scores['OracleShared'][0]-scores['Anchor'][0])),
                             selection_gap_pp=float(100*(scores['OracleDirection'][0]-scores['Reverse'][0]))))
    return endpoints,paired,ceilings,points


def gate(cfg,roles,points):
    decisions = {}
    for setting in cfg['settings']:
        keys = [key for key,r in roles.items() if r['setting']==setting]
        if len(keys)!=12:
            raise PermissionError('Need 12 paired directions per setting')
        gaps = {m:[points[k]['Reverse'][0]-points[k][m][0] for k in keys] for m in PAIRED}
        delta = {m:mean(xs) for m,xs in gaps.items()}
        best = min(previous.FIXED,key=delta.__getitem__)
        counts = [sum(np.sign(x)==s for x in gaps['Anchor']) for s in (1,0,-1)]
        checks = dict(anchor_gain=delta['Anchor']>=F(*cfg['minimum_accuracy_fraction']),
                      positive_directions=counts[0]>=cfg['minimum_positive_directions'])
        checks.update({m:delta[m]>0 for m in PHASE_GATE})
        decisions[setting] = dict(passed=all(checks.values()),checks=checks,best_fixed=best,
                                  wins=counts[0],ties=counts[1],losses=counts[2],
                                  accuracy_pp={m:float(100*d) for m,d in delta.items()},
                                  accuracy_exact={m:str(d) for m,d in delta.items()})
    return dict(settings=decisions,advance=decisions['proteins_edge']['passed'] and any(
        decisions[s]['passed'] for s in ('mutagenicity_edge','mutagenicity_node')))


def summary(cfg,roles,points):
    for setting in cfg['settings']:
        for scope,target in [('setting',None),*[(s,i) for s in ('target','omit_target') for i in cfg['domains']]]:
            keys = [k for k,r in roles.items() if r['setting']==setting and
                    (target is None or (r['target']==target)==(scope=='target'))]
            for arm in METHODS:
                acc,f1,r0,r1 = metric_means(points[k][arm] for k in keys)
                anchor = mean(points[k]['Anchor'][0] for k in keys)
                delta = acc-anchor
                yield dict(setting=setting,scope=scope,target=target,method=arm,directions=len(keys),
                           accuracy=float(acc),macro_f1=float(f1),**recalls(r0,r1),
                           delta_vs_anchor_pp=float(100*delta),delta_vs_anchor_exact=str(delta))


def paired_summary(cfg,paired):
    for setting in cfg['settings']:
        for arm in PAIRED:
            subset = [row for row in paired if row['setting']==setting and row['right']==arm]
            values = [F(row['accuracy_exact']) for row in subset]
            totals = {key:sum(r[key] for r in subset) if subset[0][key] is not None else None
                      for key in ('corrected','harmed')}
            yield dict(setting=setting,left='Reverse',right=arm,directions=len(subset),
                       wins=sum(x>0 for x in values),ties=sum(x==0 for x in values),losses=sum(x<0 for x in values),
                       accuracy_exact=str(mean(values)),accuracy_pp=float(100*mean(values)),**totals)


def score(out,root,cfg,old_cfg,sha,*,execution):
    roles,fits,expected = verify(out,root,cfg,old_cfg,sha)
    cells,fixed,evidence = published_rows(root,cfg,roles,expected)
    ends,paired,ceilings,points = score_blocks(roles,fits,cells,fixed)
    write_json(out/'score-execution.json',execution)
    write_json(out/'historical-aggregate-access.json',dict(obs_sha256=sha,
        historical_final_sha256=cfg['historical_final_sha256'],members=evidence,graph_label_reads=[]))
    write_json(out/'gate.json',gate(cfg,roles,points))
    data = dict(endpoints=ends,**{'paired-blocks':paired,'oracle-headroom':ceilings,
                                'summary':list(summary(cfg,roles,points)),
                                'paired-summary':list(paired_summary(cfg,paired))})
    for name,rows in data.items():
        write_table(out/f'{name}.csv',rows)
    files = ['OBS.json','score-execution.json','historical-aggregate-access.json',
             'gate.json',*[f'{n}.csv' for n in data]]
    return prior_run.seal(out,'FINAL',files,dict(obs_sha256=sha,historical_final_sha256=cfg['historical_final_sha256']))
