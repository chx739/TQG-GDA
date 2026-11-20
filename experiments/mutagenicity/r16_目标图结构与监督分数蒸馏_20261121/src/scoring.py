"""Validate all sealed predictions before H scoring and decide the registered gate."""
from fractions import Fraction
import json
from statistics import mean

import numpy as np
from scipy.special import expit

from experiments.common.io import write_json
from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import evaluate, paired
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.pilot import write_table
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs as r11, prior_run as storage
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src.run import summarize_pairs
from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src import historical, observation
from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src.scoring import auc
from . import experiment, inputs

GATES = ('B','Low','B2','B0','SourceCal','WL_selected','Size','LowTwice','VFull',
         'VResidual','OtherResidual','B_equal_positive')
PAIRS = tuple(('Own',m) for m in (*GATES,'OtherResidual1','OtherResidual2'))
PAIRS += tuple((arm,ref) for arm in ('B','B2') for ref in ('B0','SourceCal','WL_selected','Size'))
PAIRS += (('B2','B'),('Low','B'))
FINAL_MEMBERS = ('OBS.json','endpoints.csv','endpoint-summary.csv','paired-blocks.csv','paired-summary.csv',
                 'mechanism.csv','mechanism-summary.json','disagreement-summary.csv','gate.json','h-access.json','cohort-audit.json')


def gate(plan,cfg,results):
    fixed = cfg['phase1_seed']
    minimum = Fraction(*cfg['minimum_accuracy_fraction'])
    directions = [(s,t) for s in plan['domains'] for t in plan['domains'] if s!=t]
    table = {}
    for setting in plan['settings']:
        pairs = {arm:[results.get((setting,s,t,fixed,'Own',arm)) for s,t in directions] for arm in GATES}
        means = {arm:None if None in vals else sum(vals)/len(vals) for arm,vals in pairs.items()}
        positive = sum(v is not None and v>0 for v in pairs['B'])
        passed = (all(v is not None for v in means.values()) and means['B']>=minimum and means['Low']>=minimum
                  and all(means[m]>0 for m in GATES if m not in ('B','Low'))
                  and positive>=cfg['minimum_positive_directions'])
        table[setting] = dict(passed=passed,wins_vs_B=positive,
                              means_pp={a:None if v is None else float(100*v) for a,v in means.items()})
    extend = table['proteins_edge']['passed'] and any(
        row['passed'] for name,row in table.items() if name.startswith('mutagenicity_'))
    return dict(extend=extend,settings=table)


def contrast(lhs,rhs,actions, flags, sampled):
    refs = ('OtherResidual1','OtherResidual2') if rhs=='OtherResidual' else (rhs,)
    base = flags[lhs]
    correct,harm,disagreement = (sum(int(f(arm).sum()) for arm in refs) for f in (
        lambda arm:base&~flags[arm], lambda arm:~base&flags[arm], lambda arm:actions[lhs]!=actions[arm]))
    correction,damage,change = (Fraction(n,len(refs)) for n in (correct,harm,disagreement))
    net = correction-damage
    actual = int(base.sum()) - sum(Fraction(int(flags[arm].sum()),len(refs)) for arm in refs)
    if actual!=net:
        raise AssertionError('R16 exact paired counts differ')
    draws = sampled[lhs] - np.mean([sampled[arm] for arm in refs],axis=0) if sampled else None
    return net/len(base),draws,dict(corrected_exact=str(correction),harmed_exact=str(damage),net_exact=str(net),
                                   changed_exact=str(change),upper_bound_pp=float(100*change/len(base)))


def endpoints_by_setting(ends,plan):
    metrics = ('accuracy','macro_f1','balanced_accuracy','recall_0','recall_1','auroc','sampled_nll','predicted_class_1')
    counts = ('n','positive_count','corrected','harmed','net_corrections')
    for setting in plan['settings']:
        for m in experiment.PRIMARY:
            rows = [x for x in ends if x['setting']==setting and x['method']==m]
            yield dict(setting=setting,method=m,blocks=len(rows),
                       **{k:None if any(r[k] is None for r in rows) else mean(r[k] for r in rows) for k in metrics},
                       **{f'total_{k}':sum(r[k] for r in rows) for k in counts})
        foreign = [x for x in ends if x['setting']==setting and x['method'] in ('OtherResidual1','OtherResidual2')]
        yield dict(setting=setting,method='OtherResidual',blocks=len(foreign)//2,
                   **{k:None if any(r[k] is None for r in foreign) else mean(r[k] for r in foreign) for k in metrics},
                   **{f'total_{k}':sum(r[k] for r in foreign)/2 for k in counts})


def score(out,repo,cfg,seeds,fit_sha,obs_sha,*,pilot):
    history = inputs.Historical(cfg,repo)
    plan = history.plan
    fitted = experiment.verified_fit(out,plan,cfg,seeds,fit_sha)
    members = storage.verified_members(out,'OBS',obs_sha,experiment.obs_members(plan,seeds),{'fit_sha256':fit_sha})
    roles = json.loads(members['prediction-roles.json'])
    fit_roles = json.loads(fitted['fit-roles.json'])
    groups,tasks = observation.observations_expected(plan,seeds)
    if set(roles)!=set(tasks) or set(fit_roles)!=set(groups) or json.loads(fitted['historical-plan.json'])!=history.archive.plan:
        raise PermissionError('R16 sealed block inventory or C plan differs')
    history.held_role()
    predictions, grants = {}, {}
    for key, role in roles.items():
        task = {k:role[k] for k in experiment.IDENTITY}
        group = (task['setting'],task['target'])
        reg = history.roles[key]
        fit_role = next(r for r in fit_roles[role['group']]['tasks'] if r['target']==task['target'])
        mf,sha,ctx = historical.context(history.archive,task,repo)
        predictors = f'predictions/{key}.npz'
        if (r11.key(**task)!=key or sha!=role['manifest_sha256'] or sha!=reg['manifest_sha256']
                or sha!=fit_role['manifest_sha256'] or mf['ids']['holdout']!=role['ids']
                or reg['holdout_ids']!=role['ids'] or fit_role['ids']['holdout']!=role['ids']
                or role['r10_prediction_sha256']!=history.obs[predictors]):
            raise PermissionError(f'R16 held-out grant differs for {key}')
        x = r11.npz_member(members[predictors])
        experiment.checked_predictions(role,x)
        label = r11.raw_grant(ctx,mf,'holdout')
        if grants.setdefault(group,label)!=label:
            raise PermissionError('R16 shared H groups refer to different label grants')
        predictions[key] = task,x
    reps,seed,minimum,_ = paired._settings(history.archive.plan)
    cohorts, access, values, curves = {},{},{},{}
    ends, contrasts, mechanism = [],[],[]
    with r11.GraphLabelSession('holdout') as labels:
        for key,(task,x) in predictions.items():
            setting,s,t,model_seed = (task[k] for k in experiment.IDENTITY)
            group = setting,t
            ids = x['ids']
            if group not in cohorts:
                y = labels.load(grants[group],ids.tolist())
                order = np.argsort(ids,kind='stable')
                cohorts[group] = y,order,paired._cohort(setting,t,ids[order],y[order],reps,seed,minimum)
                access[key] = ids.tolist()
            y,order,cohort = cohorts[group]
            np.testing.assert_array_equal(ids[order],cohort['ids'])
            actions = {m:x[f'decision_{m}'] for m in experiment.PRIMARY}
            flags = {arm:a==y for arm,a in actions.items()}
            weights = cohort['weights']
            sampled = (dict(zip(experiment.PRIMARY,(weights@np.column_stack(
                [flags[m][order] for m in experiment.PRIMARY])).T)) if weights is not None else {})
            for m,a in actions.items():
                z = x[f'logits_{m}']
                metrics = evaluate.classification(z,y,actions['B'],decisions=a)
                if m=='B_equal_positive':
                    metrics['sampled_nll']=None
                ends.append(dict(**task,method=m,auroc=auc(z,y),positive_count=int(a.sum()),**metrics))
            for lhs,rhs in PAIRS:
                v,draws,counts = contrast(lhs,rhs,actions,flags,sampled)
                pair = (setting,s,t,model_seed,lhs,rhs)
                values[pair] = v
                if draws is not None:
                    curves[pair] = draws
                contrasts.append(dict(**task,left=lhs,right=rhs,n=len(ids),accuracy_pp=float(100*v),**counts))
            mechanism.append(dict(**task,mse_low=float(np.mean((x['g_low']-x['g'])**2)),
                mse_own=float(np.mean((x['g_own']-x['g'])**2)),
                nll_improvement_slope=float(np.mean((y-expit(inputs.margin(x['logits_Low'])))*x['own_residual_effect']))))
    storage.verified_members(out,'OBS',obs_sha,experiment.obs_members(plan,seeds),{'fit_sha256':fit_sha})
    write_table(out/'endpoints.csv',ends)
    write_table(out/'endpoint-summary.csv',list(endpoints_by_setting(ends,plan)))
    write_table(out/'paired-blocks.csv',contrasts)
    write_table(out/'paired-summary.csv',summarize_pairs(plan|{'seeds':seeds},values,curves,PAIRS))
    write_table(out/'disagreement-summary.csv',[
        dict(setting=s,right=arm,blocks=len(rows),changed_exact=str(sum(Fraction(r['changed_exact']) for r in rows)),
             corrected_exact=str(sum(Fraction(r['corrected_exact']) for r in rows)),
             harmed_exact=str(sum(Fraction(r['harmed_exact']) for r in rows)),
             upper_bound_pp=mean(r['upper_bound_pp'] for r in rows))
        for s in plan['settings'] for arm in GATES
        for rows in [[q for q in contrasts if q['setting']==s and (q['left'],q['right'])==('Own',arm)]]])
    write_table(out/'mechanism.csv',mechanism)
    fields = ('mse_low','mse_own','nll_improvement_slope')
    write_json(out/'mechanism-summary.json',{s:{name:mean(r[name] for r in mechanism if r['setting']==s)
                                               for name in fields} for s in plan['settings']})
    decision = gate(plan,cfg,values) if pilot else None
    write_json(out/'gate.json',dict(pilot=pilot,decision=decision))
    write_json(out/'h-access.json',dict(obs_sha256=obs_sha,actual_h_reads=access))
    write_json(out/'cohort-audit.json',[c[2]['audit'] for c in cohorts.values()])
    return storage.seal(out,'FINAL',FINAL_MEMBERS,{'obs_sha256':obs_sha}),decision
