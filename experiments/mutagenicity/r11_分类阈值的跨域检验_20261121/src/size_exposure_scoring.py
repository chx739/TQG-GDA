"""Paired evaluation of protein size exposure on the original held-out cohorts."""
from collections import defaultdict
from fractions import Fraction as F
from statistics import stdev

import numpy as np

from experiments.common.io import write_json
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.pilot import write_table
from experiments.mutagenicity.r10_原生图适配基线与增量_20261121.src import structural_observation as so
from . import inputs, prior_run, roc_score, size_exposure as curve, size_exposure_observation as obs


BEST = ('OldDirectP','Curve','CurvePool','Size','B0')
SWAPS = ('PriorOtherMean','QuantileOtherMean')
COMPARISONS = {
    'Curve':('Size','CurvePool'),
    'CurvePool':('OldPool','PoolOther1','PoolOther2','PoolOtherMean','SourceMajority'),
    'CurveDirectP':(*BEST,'PriorOther1','PriorOther2','PriorOtherMean',
                    'QuantileOther1','QuantileOther2','QuantileOtherMean'),
}


def mean(xs):
    values = list(xs)
    return sum(values,F(0))/len(values)


def group(rows,*keys):
    out = defaultdict(list)
    for r in rows:
        out[tuple(r[k] for k in keys)].append(r)
    return out


def cohorts(archive,roles):
    grants = {}
    for key,r in roles.items():
        k,old,m,ctx = archive.context({i:r[i] for i in obs.ID})
        if key!=k or any(r[f]!=v for f,v in old.items()):
            raise PermissionError('Protein H role changed between freeze and scoring')
        t = r['target']
        grant = inputs.raw_grant(ctx,m,'holdout')
        if grants.setdefault(t,grant)!=grant:
            raise PermissionError('Physical H must share exactly one grant across seeds and sources')
    if set(grants)!=set(archive.plan['domains']):
        raise PermissionError('Missing one of four authorized PROTEINS target cohorts')
    y,reads = {},{}
    with inputs.GraphLabelSession('holdout') as reader:
        for t,grant in grants.items():
            value = reader.load(grant,grant.ids)
            if value.shape!=(len(grant.ids),) or not np.isin(value,[0,1]).all():
                raise PermissionError('Malformed protein H labels')
            y[t],reads[f'proteins_edge-M{t}'] = value,list(grant.ids)
    return y,reads


def validation_audit(archive,models,roles,validation):
    first = {}
    out = []
    for r in roles.values():
        first.setdefault(r['group'],r)
    for key,record in models.items():
        r = first[key]
        _,_,m,ctx = archive.context({i:r[i] for i in obs.ID})
        x = validation[key]
        v,sha = so.source_labels(ctx,m,r['source'],r['seed'],'validation')
        if sha!=record['identity']['v_label_sha256'] or v.shape!=x['ids'].shape:
            raise PermissionError('Protein V diagnostic does not match the frozen source validation')
        for arm in ('Curve','Size'):
            t = roc_score.confusion(x[arm],v)
            correct = t[0]+t[-1]
            out.append(dict(setting=r['setting'],source=r['source'],seed=r['seed'],method=arm,n=len(v),
                            correct=correct,accuracy=correct/len(v),macro_f1=float(roc_score.f1(t))))
    return out


def block(roles,pred,labels):
    endpoints,paired = [],[]
    for key,r in roles.items():
        flags,points = {},{}
        y = labels[r['target']]
        common = {k:r[k] for k in obs.ID} | {'n':len(y)}
        x = pred[key]
        for m in curve.ARMS:
            hard = x[m]
            if hard.dtype!=np.bool_ or hard.shape!=y.shape:
                raise ValueError('Expected one frozen hard action per protein graph')
            flags[m] = hard==y
            conf = roc_score.confusion(hard,y)
            a,f = F(conf[0]+conf[-1],len(y)),roc_score.f1(conf)
            constant = int(np.all(hard) or not np.any(hard))
            points[m] = a,f,constant,int(hard.sum())
            endpoints.append(dict(**common,method=m,accuracy=float(a),macro_f1=float(f),accuracy_exact=str(a),
                                  macro_f1_exact=str(f),constant=str(constant),positive=str(int(hard.sum())),
                                  **dict(zip(('tn','fp','fn','tp'),conf,strict=True))))
        for m,arms in curve.MEANS.items():
            a,f,constant,positive = tuple(mean(v) for v in zip(*(points[k] for k in arms),strict=True))
            points[m] = a,f,constant,positive
            endpoints.append(dict(**common,method=m,accuracy=float(a),macro_f1=float(f),accuracy_exact=str(a),
                                  macro_f1_exact=str(f),constant=str(constant),positive=str(positive)))
        for l,controls in COMPARISONS.items():
            for c in controls:
                compare = curve.MEANS.get(c,(c,))
                good = mean(int((flags[l]&~flags[j]).sum()) for j in compare)
                bad = mean(int((~flags[l]&flags[j]).sum()) for j in compare)
                da,df = [a-b for a,b in zip(points[l][:2],points[c][:2],strict=True)]
                if (good-bad)/len(y)!=da:
                    raise ArithmeticError('Paired changes do not match endpoint Accuracy')
                paired.append(dict(**common,left=l,right=c,corrected=str(good),harmed=str(bad),net=str(good-bad),
                                   accuracy_exact=str(da),macro_f1_exact=str(df),
                                   accuracy_pp=float(100*da),macro_f1_pp=float(100*df)))
    return endpoints,paired


def directions(rows,column):
    return {k:mean(F(r[column]) for r in rs) for k,rs in group(rows,'source','target').items()}


def grouped(cfg,endpoints,paired):
    summary,pair_summary,leave = [],[],[]
    seeds = cfg['seeds']
    for scope,keep in [('all',seeds),('reserve',seeds[1:]),*((str(s),[s]) for s in seeds)]:
        for (arm,),rows in group([r for r in endpoints if r['seed'] in keep],'method').items():
            a = mean(directions(rows,'accuracy_exact').values())
            f = mean(directions(rows,'macro_f1_exact').values())
            summary.append(dict(scope=scope,method=arm,directions=12,blocks=len(rows),accuracy=float(a),macro_f1=float(f),
                                accuracy_exact=str(a),macro_f1_exact=str(f),
                                constant_blocks=str(sum(F(r['constant']) for r in rows))))
        for (a,c),rows in group([r for r in paired if r['seed'] in keep],'left','right').items():
            da,df = [directions(rows,k) for k in ('accuracy_exact','macro_f1_exact')]
            if len(da)!=12:
                raise PermissionError('A comparison is missing a protein direction')
            xm,fm = mean(da.values()),mean(df.values())
            pair_summary.append(dict(scope=scope,left=a,right=c,blocks=len(rows),directions=len(da),
                accuracy_pp=float(100*xm),accuracy_exact=str(xm),macro_f1_pp=float(100*fm),
                wins=sum(v>0 for v in da.values()),ties=sum(v==0 for v in da.values()),losses=sum(v<0 for v in da.values()),
                **{k:str(sum(F(r[k]) for r in rows)) for k in ('corrected','harmed','net')}))
            for t in cfg['domains']:
                kept = [k for k in da if k[1]!=t]
                x = mean(da[k] for k in kept)
                leave.append(dict(scope=scope,left=a,right=c,omitted_target=t,directions=len(kept),accuracy_exact=str(x),
                                  accuracy_pp=float(100*x),macro_f1_pp=float(100*mean(df[k] for k in kept))))
    return summary,pair_summary,leave


def gate(cfg,paired):
    rows = {r['right']:r for r in paired if r['scope']=='reserve' and r['left']=='CurveDirectP'}
    gaps = {m:F(rows[m]['accuracy_exact']) for m in (*BEST,*SWAPS)}
    best = [m for m in BEST if gaps[m]==min(gaps[k] for k in BEST)]
    required = F(*cfg['minimum_accuracy_fraction'])
    checks = dict(minimum_gain=gaps[best[0]]>=required,
                  wins=all(rows[m]['wins']>=cfg['minimum_positive_directions'] for m in best),
                  **{m:gaps[m]>0 for m in SWAPS})
    return dict(advance=all(checks.values()),scope='3406–3409',strongest_references=best,checks=checks,
                gain_exact=str(gaps[best[0]]),gain_pp=float(100*gaps[best[0]]),
                wins={m:rows[m]['wins'] for m in best},gaps_pp={m:float(100*v) for m,v in gaps.items()})


def direction_statistics(rows):
    for (setting,s,t,m),items in group(rows,'setting','source','target','method').items():
        if len(items)!=5 or len({r['seed'] for r in items})!=5:
            raise PermissionError('Expected all five registered seeds per direction')
        fields = {}
        for name in ('accuracy','macro_f1'):
            values = [F(r[f'{name}_exact']) for r in items]
            fields[name+'_mean'] = float(mean(values))
            fields[name+'_sample_sd'] = stdev(map(float,values))
        yield dict(setting=setting,source=s,target=t,method=m,seeds=5,**fields)


def score(out,stage,archive,cfg,roc_cfg,sha,execution):
    roles,models,pred,v = obs.observed(out,archive,cfg,roc_cfg,sha)
    y,read = cohorts(archive,roles)
    base,pairs = block(roles,pred,y)
    val = validation_audit(archive,models,roles,v)
    endpoints,paired,leave = grouped(cfg,base,pairs)
    results = {'endpoints':base,'paired-blocks':pairs,'summary':endpoints,'paired-summary':paired,
               'leave-target':leave,'source-validation':val,'direction-statistics':list(direction_statistics(base))}
    for name,rows in results.items():
        write_table(stage/f'{name}.csv',rows)
    write_json(stage/'label-access.json',dict(obs_sha256=sha,holdout=read,adaptation_labels_read=False,
                validation={k:record['identity']['validation_ids'] for k,record in models.items()}))
    write_json(stage/'gate.json',gate(cfg,paired))
    write_json(stage/'score-execution.json',execution)
    (stage/'OBS.json').write_bytes((out/'OBS.json').read_bytes())
    names = ['OBS.json','gate.json','label-access.json','score-execution.json',*[f'{n}.csv' for n in results]]
    return prior_run.seal(stage,'FINAL',names,dict(obs_sha256=sha))
