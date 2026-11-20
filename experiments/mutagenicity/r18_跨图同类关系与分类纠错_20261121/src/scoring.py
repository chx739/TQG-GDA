"""Score frozen graph actions and pair probabilities on the original E role."""
from fractions import Fraction

import numpy as np

from experiments.common.io import write_json
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.pilot import write_table
from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import residual_inputs as data, true_rate_replay
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import prior_run, roc_score
from . import relations, screen


def mean(values):
    v = list(values)
    return sum(v)/len(v)


def brier_exact(q,y):
    """Use the frozen binary64 values as exact dyadic rationals for strict comparisons."""
    total = Fraction(0)
    for p,t in zip(q,y,strict=True):
        error = Fraction.from_float(float(p))-int(t)
        total += error*error
    return total/len(y)


def block(role,x,y):
    fields = {k:role[k] for k in data.IDENTITY}
    n = len(x['e_ids'])
    if y.shape!=(n,) or not np.isin(y,[0,1]).all():
        raise ValueError('Malformed authorized E labels')
    acc,ep,contrasts,paired = {},[],{},[]
    for m in screen.ACTIONS:
        tn,fp,fn,tp = c = roc_score.confusion(x[m],y)
        a,f = Fraction(tn+tp,n),roc_score.f1(c)
        acc[m] = (a,f)
        ep.append(dict(**fields,method=m,n=n,accuracy=float(a),macro_f1=float(f),positive=tp+fp,
                       tn=tn,fp=fp,fn=fn,tp=tp))
    for m in screen.CONTROLS:
        counts = true_rate_replay.paired_counts(x[screen.MAIN],x[m],y)
        a,f = acc[screen.MAIN][0]-acc[m][0],acc[screen.MAIN][1]-acc[m][1]
        if a!=Fraction(counts['net'],n) or counts['changed']!=counts['corrected']+counts['harmed']:
            raise ArithmeticError('R18 net correct differs from the paired decisions')
        contrasts[m] = a,f,counts,n
        paired.append(dict(**fields,left=screen.MAIN,right=m,n=n,**counts,accuracy_exact=str(a),
                           accuracy_pp=float(100*a),macro_f1_pp=float(100*f),
                           disagreement_pp=100*counts['changed']/n))
    pair_y = y[x['pair_i']]==y[x['pair_j']]
    brier = {}
    details = []
    for m in relations.PAIR_ARMS:
        p = x[f'pair_{m}']
        b = brier_exact(p,pair_y)
        brier[m] = b,len(pair_y),int(pair_y.sum())
        details.append(dict(**fields,method=m,n_pairs=len(pair_y),same_class=int(pair_y.sum()),
                            brier=float(b),brier_exact=str(b)))
    return ep,paired,details,acc,contrasts,brier


def group_tables(roles,acc,contrasts,brier):
    endpoints,paired,relations_rows = [],[],[]
    for setting in sorted({r['setting'] for r in roles.values()}):
        for source in (None,0,1,2,3):
            keys = [k for k,r in roles.items() if r['setting']==setting and (source is None or r['source']==source)]
            if len(keys)!=(8 if source is None else 2):
                raise PermissionError('R18 role aggregation requires the complete registered direction grid')
            for m in screen.ACTIONS:
                a,f = zip(*(acc[k][m] for k in keys),strict=True)
                endpoints.append(dict(setting=setting,source=source,method=m,directions=len(keys),
                                      accuracy=float(mean(a)),macro_f1=float(mean(f))))
            for m in screen.CONTROLS:
                a,f,p,n = zip(*(contrasts[k][m] for k in keys),strict=True)
                totals = {x:sum(r[x] for r in p) for x in ('net','corrected','harmed','changed')}
                rates = {x+'_pp':float(100*mean(Fraction(v[x],size) for v,size in zip(p,n,strict=True)))
                         for x in ('corrected','harmed','changed')}
                paired.append(dict(setting=setting,source=source,directions=len(keys),left=screen.MAIN,right=m,
                                   accuracy_exact=str(mean(a)),accuracy_pp=float(100*mean(a)),
                                   macro_f1_pp=float(100*mean(f)),**totals,**rates,
                                   positive_directions=sum(i>0 for i in a),tied=sum(i==0 for i in a),negative=sum(i<0 for i in a)))
            for m in relations.PAIR_ARMS:
                values,n,k = zip(*(brier[key][m] for key in keys),strict=True)
                score = mean(values)
                relations_rows.append(dict(setting=setting,source=source,method=m,directions=len(keys),
                                           n_pairs=sum(n),same_class=sum(k),brier=float(score),brier_exact=str(score)))
    return endpoints,paired,relations_rows


def gate(contrasts,brier):
    detail = {}
    for setting in sorted({r['setting'] for r in contrasts}):
        results = [r for r in contrasts if r['setting']==setting]
        overall = {r['right']:Fraction(r['accuracy_exact']) for r in results if r['source'] is None}
        source = {str(r['source']):Fraction(r['accuracy_exact']) for r in results
                  if r['source'] is not None and r['right']==screen.BASE}
        b = {r['method']:Fraction(r['brier_exact']) for r in brier
             if r['setting']==setting and r['source'] is None}
        groups = sum(v>0 for v in source.values())
        checks = dict(accuracy_at_least_half_pp=overall[screen.BASE]>=screen.MINIMUM,positive_groups=groups>=3)
        checks.update({f'accuracy_above_{m}':overall[m]>0 for m in screen.GATE_CONTROLS})
        checks.update({f'brier_below_{m}':b['Full']<v for m,v in b.items() if m!='Full'})
        detail[setting] = dict(passed=all(checks.values()),clauses=checks,positive_source_groups=groups,
                               group_accuracy_diff_exact={m:str(v) for m,v in source.items()},
                               accuracy_diff_exact={m:str(v) for m,v in overall.items()},
                               accuracy_diff_pp={m:float(100*v) for m,v in overall.items()},
                               relation_brier={m:float(v) for m,v in b.items()})
    advance = detail['proteins_edge']['passed'] and any(
        detail[s]['passed'] for s in ('mutagenicity_edge','mutagenicity_node'))
    return dict(settings=detail,continue_real_development=advance)


def score(out,cfg,prep,obs,*,execution):
    x,models = screen.observed(out,cfg,prep,obs)
    write_json(out/'score-execution.json',execution)
    orig = prep.original
    roles = orig.reference.roles
    endpoints,pairs,relations_blocks,diagnostics = [],[],[],[]
    acc,cont,brier = {},{},{}
    with data.ScopedLabels(orig.archive.by_setting,roles,'replay_E') as labels:
        for key,r in roles.items():
            with labels.direction(key):
                y = labels.load(key,'E',r['roles']['E'])
            ep,pa,br,acc[key],cont[key],brier[key] = block(r,x[key],y)
            endpoints.extend(ep)
            pairs.extend(pa)
            relations_blocks.extend(br)
            p = models[key]
            fields = {k:r[k] for k in data.IDENTITY}
            diagnostics.append(dict(**fields,anchor=p['choice'][0],anchor_C=p['choice'][1],
                                    D_count=len(r['roles']['D']),D_pairs=p['D_pairs'],
                                    reference_count=len(p['references']['U']),
                                    U_C_decision_disagreement=int((x[key][screen.MAIN]!=x[key]['SourceFull']).sum()),
                                    cal_b=p['calibrator']['coefficients'][0],cal_t=p['calibrator']['coefficients'][1],
                                    cal_kkt=p['calibrator']['gradient_inf'],Unary_kkt=p['Unary'].get('gradient_inf'),
                                    Full_kkt=p['Full'].get('gradient_inf')))
    data.validate_receipt(roles,labels.actual,'replay_E')
    ep,pa,br = group_tables(roles,acc,cont,brier)
    tables = {'endpoints':endpoints,'paired-blocks':pairs,'pair-brier-blocks':relations_blocks,
              'summary':ep,'paired-summary':pa,'pair-brier-summary':br,'diagnostics':diagnostics}
    for name,rows in tables.items():
        write_table(out/f'{name}.csv',rows)
    write_json(out/'e-label-access.json',labels.actual)
    write_json(out/'gate.json',gate(pa,br))
    files = ['OBS.json','score-execution.json','gate.json','e-label-access.json',*[f'{name}.csv' for name in tables]]
    return prior_run.seal(out,'FINAL',files,dict(obs_sha256=obs))
