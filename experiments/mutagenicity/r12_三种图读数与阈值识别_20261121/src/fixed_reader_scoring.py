"""Score H and audit class-conditional drift with the frozen V/A/H readouts."""
from fractions import Fraction as F

import numpy as np

from experiments.common.io import write_json
from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src.head_crossover_scoring import paired_counts
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.pilot import write_table
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run, roc_score
from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src import historical
from . import fixed_reader_transfer as experiment, reader_ranges as ranges

CONTROLS = tuple(m for m in (*experiment.ACTIONS,'OtherMean') if m!=experiment.MAIN)
DIAG_RULES = ('Anchor',*ranges.RULES,'Other1','Other2')
RANGES = ('SizeRange','HashRange')


def mean(values):
    xs = list(values)
    return sum(xs)/len(xs)


def exact(x):
    return None if x is None else str(x)


def numeric(x):
    return None if x is None else float(x)


def h_labels(repo,archive,roles):
    grants = {}
    for role in roles.values():
        task = {k:role[k] for k in experiment.ID}
        mf,sha,ctx = historical.context(archive.history.c,task,repo)
        if sha!=role['manifest_sha256'] or mf['ids']!=role['ids'] or mf['ids']['holdout']!=role['holdout_ids']:
            raise PermissionError('Frozen H role differs from the authorized source manifest')
        cohort = role['setting'],role['target']
        grant = inputs.raw_grant(ctx,mf,'holdout')
        if grants.setdefault(cohort,grant)!=grant:
            raise PermissionError('H graphs do not share one label grant per target')
    if len(grants)!=8:
        raise PermissionError('Expected eight distinct H roles')
    labels,access = {},{}
    with inputs.GraphLabelSession('holdout') as session:
        for (setting,target),grant in grants.items():
            y = session.load(grant,grant.ids)
            if y.shape!=(len(grant.ids),) or not np.isin(y,[0,1]).all():
                raise ValueError('Invalid H labels')
            labels[setting,target] = y
            access[f'{setting}-M{target}'] = list(grant.ids)
    return labels,access


def classification(role,x,y):
    fields = {k:role[k] for k in experiment.ID}
    names = experiment.ACTIONS
    acts = {m:x[m] for m in names}
    n = len(y)
    ok = {m:a==y for m,a in acts.items()}
    points,ends,pairs,stats = {},[],[],{}
    for m in names:
        tn,fp,fn,tp = cell = roc_score.confusion(acts[m],y)
        a,f = F(tn+tp,n),roc_score.f1(cell)
        points[m] = a,f
        ends.append(dict(**fields,method=m,n=n,accuracy=float(a),macro_f1=float(f),positive=tp+fp,
                         tn=tn,fp=fp,fn=fn,tp=tp))
    points['OtherMean'] = tuple(mean(x) for x in zip(points['Other1'],points['Other2'],strict=True))
    own = points[experiment.MAIN]
    for m in CONTROLS:
        members = ('Other1','Other2') if m=='OtherMean' else (m,)
        good,bad,changed = paired_counts((experiment.MAIN,),members,acts,ok)
        a,f = (i-j for i,j in zip(own,points[m],strict=True))
        if a!=(good-bad)/n or changed!=good+bad:
            raise ArithmeticError('Classification pair counts differ from the Accuracy difference')
        pairs.append(dict(**fields,left=experiment.MAIN,right=m,n=n,corrected=str(good),harmed=str(bad),
                          changed=str(changed),accuracy_exact=str(a),accuracy_pp=float(100*a),macro_f1_pp=float(100*f)))
        stats[m] = a,f,good,bad,changed
    return ends,pairs,points,stats


def diagnose(role,h,y,record,rules,donor):
    fields = {k:role[k] for k in experiment.ID}
    classes,bins,errors,drift = [],[],[],[]
    totals = {m:F(0) for m in DIAG_RULES}
    h_covered,ah_covered = 0,0
    for i,src in enumerate(record['bins']):
        b = i+1
        template = None if src['w'] is None else [F(w) for w in src['w']]
        mask = h['u']==b
        counts = [ranges.four_cells(h['x'][mask&(y==c)]) for c in (0,1)]
        n0,n1 = (sum(a) for a in counts)
        n = n0+n1
        p_h = None if not n else F(n1,n)
        n_a = rules[i]['n']
        extra = {m:item[i]['rules'][experiment.MAIN] for m,item in zip(('Other1','Other2'),donor,strict=True)}
        vals = dict(Anchor={'p':src['p0'],'reason':'source_prior'},**rules[i]['rules'],**extra)
        h_means = [None,None] if template is None else [ranges.readout_mean(a,template) for a in counts]
        eligible = bool(n and n_a and template is not None and any(template))
        for name in RANGES:
            interval = rules[i]['rules'][name]['feasible']
            span = None if interval is None else [F(x) for x in interval]
            contains = None if not eligible else span is not None and span[0]<=p_h<=span[1]
            width = None if span is None else span[1]-span[0]
            bins.append(dict(**fields,bin=b,rule=name,n_A=n_a,n_H=n,eligible=eligible,
                             reason=rules[i]['rules'][name]['reason'],p_H=numeric(p_h),contains_p_H=contains,
                             interval_lo=None if span is None else float(span[0]),
                             interval_hi=None if span is None else float(span[1]),interval_width=numeric(width)))
            for c in (0,1):
                mu = h_means[c]
                has = template is not None and any(template) and mu is not None
                r = src['ranges'].get(name)
                lo,hi = (None,None) if r is None else [F(x) for x in r[c]]
                status = ('missing_H_class' if not sum(counts[c]) else 'missing_V_class' if template is None
                          else 'zero_contrast' if not any(template) else 'observed')
                classes.append(dict(**fields,bin=b,label=c,rule=name,nH=sum(counts[c]),status=status,
                                    mean_H=numeric(mu),low=numeric(lo),high=numeric(hi),
                                    width=None if hi is None else float(hi-lo),
                                    covered=None if not has else lo<=mu<=hi,
                                    source_subgroup_empty=None if r is None else src['missing'][name][c]))
        if n:
            for m,r in vals.items():
                cost = abs(F(r['p'])-p_h)
                totals[m] += n*cost
                errors.append(dict(**fields,bin=b,method=m,n_H=n,estimated_p=float(F(r['p'])),
                                   p_H=float(p_h),error=float(cost),fallback=r['reason']))
        if n and template is not None:
            mu0,mu1 = (F(x) for x in src['means'])
            h_covered += n
            implied = (1-p_h)*mu0+p_h*mu1
            mean_h = ranges.readout_mean([sum(a) for a in zip(*counts,strict=True)],template)
            c0 = F(0) if not n0 else (1-p_h)*(h_means[0]-mu0)
            c1 = F(0) if not n1 else p_h*(h_means[1]-mu1)
            if c0+c1!=mean_h-implied:
                raise ArithmeticError('Classwise moment drift does not match the whole-bin drift')
            mean_a = None if rules[i]['u'] is None else F(rules[i]['u'])
            delta = None if mean_a is None else mean_a-mean_h
            ah_covered += n if delta is not None else 0
            drift.append(dict(**fields,bin=b,n_A=n_a,n_H=n,n_H0=n0,n_H1=n1,mu_V0=str(mu0),mu_V1=str(mu1),
                              mu_H0=exact(h_means[0]),mu_H1=exact(h_means[1]),p_H=str(p_h),mean_A=exact(mean_a),
                              mean_H=str(mean_h),source_mixture_at_p_H=str(implied),class0_weighted=str(c0),
                              class1_weighted=str(c1),H_moment_drift=str(c0+c1),A_minus_H=exact(delta)))
        else:
            drift.append(dict(**fields,bin=b,n_A=n_a,n_H=n,n_H0=n0,n_H1=n1,mu_V0=None,mu_V1=None,
                              mu_H0=None,mu_H1=None,p_H=exact(p_h),mean_A=rules[i]['u'],mean_H=None,
                              source_mixture_at_p_H=None,class0_weighted=None,class1_weighted=None,
                              H_moment_drift=None,A_minus_H=None))
    n = len(y)
    direction = [dict(**fields,method=m,n_H=n,mean_abs_error=float(totals[m]/n),exact_mean_abs_error=str(totals[m]/n),
                      H_supported_for_moment=h_covered,H_supported_for_AH=ah_covered) for m in DIAG_RULES]
    return classes,bins,errors,drift,direction


def summarize(roles,points,statistics,cfg):
    endpoints,pairs,decision = [],[],{}
    for setting in cfg['settings']:
        keys = [k for k,r in roles.items() if r['setting']==setting]
        if len(keys)!=12:
            raise PermissionError('Expected twelve full directions per setting')
        for m in (*experiment.ACTIONS,'OtherMean'):
            a,f = zip(*(points[k][m] for k in keys),strict=True)
            endpoints.append(dict(setting=setting,method=m,directions=12,accuracy=float(mean(a)),
                                  macro_f1=float(mean(f)),accuracy_exact=str(mean(a))))
        overall = {}
        for m in CONTROLS:
            a,f,good,bad,changed = zip(*(statistics[k][m] for k in keys),strict=True)
            score = mean(a)
            row = dict(setting=setting,left=experiment.MAIN,right=m,directions=12,accuracy_exact=str(score),
                       accuracy_pp=float(100*score),macro_f1_pp=float(100*mean(f)),
                       corrected=str(sum(good)),harmed=str(sum(bad)),changed=str(sum(changed)),
                       positive=sum(v>0 for v in a),tied=sum(v==0 for v in a),negative=sum(v<0 for v in a))
            pairs.append(row)
            overall[m] = row
        a = overall['Anchor']
        checks = dict(anchor_gain=F(a['accuracy_exact'])>=F(*cfg['minimum_accuracy_fraction']),
                      positive_directions=a['positive']>=cfg['minimum_positive_directions'])
        required = (*experiment.FIXED, 'Point', 'HashRange', 'EqualCount', 'OtherMean')
        checks.update({f'strictly_above_{m}': F(overall[m]['accuracy_exact']) > 0 for m in required})
        decision[setting] = dict(passed=all(checks.values()),clauses=checks,
                                 positive=a['positive'],tied=a['tied'],negative=a['negative'],
                                 accuracy_pp={m:r['accuracy_pp'] for m,r in overall.items()},
                                 accuracy_exact={m:r['accuracy_exact'] for m,r in overall.items()})
    return endpoints,pairs,dict(settings=decision,advance=all(d['passed'] for d in decision.values()))


def summarize_diagnostics(classes,bins,directions,cfg):
    summary=[]
    for setting in cfg['settings']:
        for rule in RANGES:
            cls = [r for r in classes if r['setting']==setting and r['rule']==rule]
            vals = [r for r in bins if r['setting']==setting and r['rule']==rule]
            observed = [r for r in cls if r['status']=='observed']
            eligible = [r for r in vals if r['eligible']]
            feasible = [r['interval_width'] for r in eligible if r['interval_width'] is not None]
            h_n = sum(r['n_H'] for r in eligible)
            covered = [r for r in eligible if r['contains_p_H']]
            summary.append(dict(setting=setting,rule=rule,class_combinations=len(cls),class_eligible=len(observed),
                                class_covered=sum(r['covered'] for r in observed),
                                class_missing_H=sum(r['status']=='missing_H_class' for r in cls),
                                class_missing_V=sum(r['status']=='missing_V_class' for r in cls),
                                class_zero_contrast=sum(r['status']=='zero_contrast' for r in cls),
                                full_range_fallback=sum(r['source_subgroup_empty'] for r in observed),
                                average_class_width=None if not observed else mean(r['width'] for r in observed),
                                eligible_Abins=len(eligible),p_H_covered=len(covered),p_H_eligible_graphs=h_n,
                                p_H_covered_graphs=sum(r['n_H'] for r in covered),
                                p_H_weighted_coverage=None if not h_n else sum(r['n_H'] for r in covered)/h_n,
                                feasible_intervals=len(feasible),average_p_interval_width=None if not feasible else mean(feasible)))
    proportion = [dict(setting=setting,method=m,mean_abs_error=float(mean(
                            F(r['exact_mean_abs_error']) for r in directions if r['setting']==setting and r['method']==m)))
                  for setting in cfg['settings'] for m in DIAG_RULES]
    return summary,proportion


def score(out,repo,cfg,historical_cfg,archive,obs,*,execution):
    roles,x,held,sources,rules = experiment.observed(out,cfg,historical_cfg,archive,obs)
    write_json(out/'score-execution.json',execution)
    ys,access = h_labels(repo,archive,roles)
    ends,pairs,counts,bins,errors,drift,directions = [],[],[],[],[],[],[]
    points,stats = {},{}
    for key,role in roles.items():
        y = ys[role['setting'],role['target']]
        ep,pa,points[key],stats[key] = classification(role,x[key],y)
        ends.extend(ep)
        pairs.extend(pa)
        donor = [rules[inputs.key(role['setting'],role['source'],t,role['seed'])] for t in role['other_targets']]
        for table,records in zip((counts,bins,errors,drift,directions),
                diagnose(role,held[key],y,sources[role['group']],rules[key],donor),strict=True):
            table.extend(records)
    endpoint,paired,gate = summarize(roles,points,stats,cfg)
    coverage,prop = summarize_diagnostics(counts,bins,directions,cfg)
    tables = {'endpoints':ends,'paired-blocks':pairs,'summary':endpoint,'paired-summary':paired,
              'class-coverage':counts,'prior-coverage':bins,'proportion-errors':errors,'moment-drift':drift,
              'proportion-by-direction':directions,'coverage-summary':coverage,'proportion-summary':prop}
    for name,rows in tables.items():
        write_table(out/f'{name}.csv',rows)
    write_json(out/'h-label-access.json',dict(obs_sha256=obs,actual_reads=access))
    write_json(out/'gate.json',gate)
    files = ['OBS.json','score-execution.json','h-label-access.json','gate.json',*[f'{n}.csv' for n in tables]]
    return prior_run.seal(out,'FINAL',files,dict(obs_sha256=obs))
