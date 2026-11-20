"""Evaluate complete groups and B0 fallback with shared target-stratified draws."""
from collections import defaultdict
from fractions import Fraction
import csv
import io
import json
from math import lcm
from pathlib import Path
from statistics import fmean

import numpy as np

from experiments.common.io import write_json
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.pilot import write_table
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run, roc_score, roc_transfer
from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src import historical, observation
from . import head_crossover as cross, paired


def validate_actions(role, group, x, domains):
    ids = np.asarray(role['holdout_ids'],dtype=np.int64)
    targets = sorted(set(domains)-{role['source']})
    head = {cross.head_name(m,t):(m,t) for m in cross.HEADS for t in targets}
    names = {*cross.FIXED,*head,*(cross.head_name(m,t,count=True) for m,t in head.values())}
    if (role['complete']!=group['complete'] or set(role['head_records'])!=set(head)
            or set(x)!={'ids','margin_B0',*names,*(f'margin_{m}' for m in head)}):
        raise PermissionError('Frozen R7 prediction inventory differs')
    if x['ids'].dtype!=np.int64 or not np.array_equal(x['ids'],ids):
        raise PermissionError('Wrong H graph order')
    for m in names:
        if x[m].dtype!=np.int64 or x[m].shape!=ids.shape or not np.isin(x[m],[0,1]).all():
            raise PermissionError(f'Invalid frozen H action: {m}')
    for name in ('B0',*head):
        z = x[f'margin_{name}']
        if z.dtype!=np.float64 or z.shape!=ids.shape or not np.isfinite(z).all() or not np.array_equal(z>0.,x[name]):
            raise PermissionError(f'Invalid frozen H margin: {name}')
    for name,(method,target) in head.items():
        rec = role['head_records'][name]
        valid = group['status'][str(target)][method]['valid']
        action,tie = cross.equal_count(ids,x['margin_B0'],int(x[name].sum()))
        ep = cross.head_name(method,target,count=True)
        if (rec['reference']!=target or rec['valid']!=valid or rec['fallback']==valid
                or rec['positives']!=int(x[name].sum()) or any(rec[k]!=v for k,v in tie.items())
                or not np.array_equal(action,x[ep]) or (not valid and not np.array_equal(x[name],x['B0']))):
            raise PermissionError(f'Count matching or fallback was not preserved: {name}')
    return names


def observed(out, cfg, c, obs_sha):
    _, keys = observation.observations_expected(cfg,[cfg['phase1_seed'],*cfg['reserve_seeds']])
    blob = prior_run.verified_members(out,'OBS',obs_sha,cross.membership(keys),cross.parents(cfg))
    if json.loads(blob['historical-plan.json'])!=c.plan:
        raise PermissionError('Original R7 plan differs after freezing')
    roles = json.loads(blob['roles.json'])
    groups = json.loads(blob['groups.json'])
    if set(roles)!=set(keys) or set(groups)!={r['group'] for r in roles.values()}:
        raise PermissionError('R7 grid or source group coverage differs')
    actions, complete_counts = {},defaultdict(int)
    for key in keys:
        r = roles[key]
        task = {k:r[k] for k in cross.IDENTITY}
        group = groups[r['group']]
        if (inputs.key(**task)!=key or any(r[k]!=group[k] for k in ('setting','source','seed'))
                or group['complete']!=all(s['valid'] for t in group['status'].values() for s in t.values())):
            raise PermissionError(f'R7 group parity differs: {key}')
        x = inputs.npz_member(blob[f'predictions/{key}.npz'])
        names = validate_actions(r,group,x,cfg['domains'])
        actions[key] = {name:x[name] for name in sorted(names)}
        complete_counts[r['setting']] += r['complete']
    if dict(complete_counts)!={'mutagenicity_edge':60,'mutagenicity_node':60,'proteins_edge':57}:
        raise PermissionError('Complete cases do not match the originally registered failure')
    return roles,groups,actions


def metric_tables(actions, y, weights, order):
    names = list(actions)
    cells = {name:roc_score.confusion(x,y) for name,x in actions.items()}
    point = {name:(Fraction(v[0]+v[3],len(y)),roc_score.f1(v)) for name,v in cells.items()}
    if weights is None:
        return cells,point,{}
    pred = np.column_stack([actions[m][order] for m in names]).astype(float)
    counts = weights@np.column_stack((pred,pred*y[order,None]))
    pos,tp = np.split(counts,2,axis=1)
    prevalence = np.mean(y)
    tn = 1-prevalence-(pos-tp)
    dn,dp = 2.-prevalence-pos,prevalence+pos
    f1 = (np.divide(tn,dn,out=np.zeros_like(tn),where=dn!=0.)
          + np.divide(tp,dp,out=np.zeros_like(tp),where=dp!=0.))
    return cells,point,{m:np.column_stack((tp[:,i]+tn[:,i],f1[:,i])) for i,m in enumerate(names)}


def views(target, source, domains):
    foreign = sorted(set(domains)-{target,source})
    aliases = {name:(name,) for name in cross.FIXED}
    for method in cross.HEADS:
        for label,refs in (('Own',[target]),('Other1',foreign[:1]),('Other2',foreign[1:]),('OtherMean',foreign)):
            aliases[method+label] = tuple(cross.head_name(method,ref) for ref in refs)
            aliases[method+'Count'+label] = tuple(cross.head_name(method,ref,count=True) for ref in refs)
    return aliases


def comparison_names():
    comparisons = {}
    for m in cross.HEADS:
        comparisons[m+'Match'] = (m+'Own',m+'OtherMean')
        comparisons[m+'CountMatch'] = (m+'CountOwn',m+'CountOtherMean')
        comparisons[m+'WithinOwnBeyondCount'] = (m+'Own',m+'CountOwn')
        comparisons[m+'OtherMinusB0'] = (m+'OtherMean','B0')
        for base in cross.FIXED:
            comparisons[m+'OwnMinus'+base] = (m+'Own',base)
    comparisons['JMinusEOwn'] = ('JOwn','EOwn')
    return comparisons


def paired_counts(names_left, names_right, actions, flags):
    n = len(names_left)*len(names_right)
    gain = harm = changed = 0
    for l in names_left:
        for r in names_right:
            gain += int((flags[l]&~flags[r]).sum())
            harm += int((~flags[l]&flags[r]).sum())
            changed += int((actions[l]!=actions[r]).sum())
    return tuple(Fraction(v,n) for v in (gain,harm,changed))


def difference(left, right):
    return tuple(x-y for x,y in zip(left,right,strict=True))


def mean_values(values):
    return tuple(sum(v)/len(v) for v in zip(*values,strict=True))


def sampled_mean(arrays):
    return None if any(v is None for v in arrays) else np.mean(arrays,axis=0)


def sampled_difference(left,right):
    return None if left is None or right is None else left-right


def integer_draws(weights, n):
    if weights is None:
        return None
    counts = np.rint(weights*n).astype(np.int64)
    if not np.array_equal(counts.astype(float)/n,weights) or not np.all(counts.sum(axis=1)==n):
        raise ArithmeticError('Bootstrap weights are not the registered integer graph draws')
    return counts


def exact_match_curve(own, foreign, counts, order):
    if counts is None:
        return None
    n = len(own)
    net = 2*own.astype(np.int64)-foreign[0].astype(np.int64)-foreign[1].astype(np.int64)
    return counts@net[order],2*n


def exact_direction_curve(curves):
    if any(c is None for c in curves):
        return None
    denominator = curves[0][1]
    if any(d!=denominator for _,d in curves):
        raise ArithmeticError('Mixed H identities in one seeded direction')
    return np.sum([values for values,_ in curves],axis=0),denominator*len(curves)


def exact_accuracy_interval(curves, percentile):
    if any(c is None for c in curves):
        return None
    denominator = lcm(*(d for _,d in curves))
    factors = [denominator//d for _,d in curves]
    sums = sorted(sum(int(value)*factor for value,factor in zip(row,factors,strict=True))
                  for row in zip(*(values for values,_ in curves),strict=True))
    values = []
    for q in percentile:
        position = Fraction(str(q))*(len(sums)-1)/100
        row,remainder = divmod(position.numerator,position.denominator)
        mix = Fraction(remainder,position.denominator)
        upper = sums[row+1] if remainder else sums[row]
        values.append(((1-mix)*sums[row]+mix*upper)/(denominator*len(curves)))
    return values


def score_blocks(roles, actions, cohorts, domains):
    endpoints,contrasts,moments = [],[],{}
    pairs = comparison_names()
    for key,role in roles.items():
        identity = {k:role[k] for k in cross.IDENTITY}
        setting,source,target,seed = identity.values()
        y,order,cohort = cohorts[setting,target]
        pred = actions[key]
        c,point,sample = metric_tables(pred,y,cohort['weights'],order)
        aliases = views(target,source,domains)
        flags = {name:x==y for name,x in pred.items()}
        base = flags['B0']
        for m,x in pred.items():
            head = role['head_records'].get(m.replace('_count_','_'),{})
            tn,fp,fn,tp = c[m]
            gain,harm = int((flags[m]&~base).sum()),int((~flags[m]&base).sum())
            endpoints.append(dict(**identity,complete=role['complete'],method=m,n=len(y),
                tn=tn,fp=fp,fn=fn,tp=tp,accuracy=float(point[m][0]),macro_f1=float(point[m][1]),
                positive_count=int(x.sum()),corrected_vs_b0=gain,harmed_vs_b0=harm,
                changed_vs_b0=int((x!=pred['B0']).sum()),reference=head.get('reference'),
                was_fallback=head.get('fallback')))
        abs_point = {m:mean_values([point[v] for v in names]) for m,names in aliases.items()}
        abs_sample = {m:sampled_mean([sample.get(v) for v in names]) for m,names in aliases.items()}
        effects = {}
        for name,(l,r) in pairs.items():
            gain,harm,changed = paired_counts(aliases[l],aliases[r],pred,flags)
            delta = difference(abs_point[l],abs_point[r])
            if (gain-harm)/len(y)!=delta[0]:
                raise ArithmeticError('Exact correction counts differ from the paired accuracy')
            effects[name] = (delta,sampled_difference(abs_sample[l],abs_sample[r]),
                             (gain,harm,changed,Fraction(changed,len(y))))
        for name,l,r in (('ERerankMatch','EMatch','ECountMatch'),('JRerankMatch','JMatch','JCountMatch'),
                         ('JMinusEMatch','JMatch','EMatch')):
            effects[name] = (difference(effects[l][0],effects[r][0]),
                             sampled_difference(effects[l][1],effects[r][1]),None)
        for name,(delta,draws,counts) in effects.items():
            gain,harm,changed,upper = (None,)*4 if counts is None else counts
            contrasts.append(dict(**identity,complete=role['complete'],contrast=name,n=len(y),
                corrected=None if gain is None else str(gain),harmed=None if harm is None else str(harm),
                changed=None if changed is None else str(changed),disagreement_upper_pp=None if upper is None else float(100*upper),
                accuracy_pp=float(100*delta[0]),macro_f1_pp=float(100*delta[1])))
        exact = exact_match_curve(flags[aliases['EOwn'][0]],[flags[m] for m in aliases['EOtherMean']],
                                  cohort['integer_draws'],order)
        moments[key] = dict(absolute={name:(p,abs_sample[name],None) for name,p in abs_point.items()},contrasts=effects,
                            exact_E_match=exact)
    return endpoints,contrasts,moments


def slices(cfg, roles):
    for setting in cfg['settings']:
        universe = {k:r for k,r in roles.items() if r['setting']==setting}
        complete = {k:r for k,r in universe.items() if r['complete']}
        yield setting,'complete',complete
        yield setting,'fallback',universe
        if setting=='mutagenicity_edge':
            seen = lambda r:(r['source'],r['target']) in ((0,1),(0,3))
            yield setting,'beta_seen',{k:r for k,r in complete.items() if seen(r)}
            yield setting,'without_beta_seen',{k:r for k,r in complete.items() if not seen(r)}


def aggregate(cfg, roles, moments, percentile):
    rows = []
    for setting,scope,subset in slices(cfg,roles):
        by_direction = defaultdict(list)
        for k,r in subset.items():
            by_direction[r['source'],r['target']].append(k)
        for kind in ('absolute','contrasts'):
            first = moments[next(iter(subset))][kind]
            for name in first:
                directions = {}
                for pair,keys in by_direction.items():
                    values = [moments[k][kind][name] for k in keys]
                    points = mean_values([v[0] for v in values])
                    draws = sampled_mean([v[1] for v in values])
                    counts = [v[2] for v in values]
                    rates = None if counts[0] is None else mean_values([v[3:] for v in counts])[0]
                    exact = (exact_direction_curve([moments[k]['exact_E_match'] for k in keys])
                             if name=='EMatch' else None)
                    directions[pair] = points,draws,rates,len(keys),exact
                for pair in (*directions,None):
                    selected = list(directions.values()) if pair is None else [directions[pair]]
                    p = mean_values([v[0] for v in selected])
                    d = sampled_mean([v[1] for v in selected])
                    ci = np.percentile(100*d,percentile,axis=0) if d is not None else [[None,None],[None,None]]
                    lower_exact = None
                    if name=='EMatch':
                        exact_ci = exact_accuracy_interval([v[4] for v in selected],percentile)
                        if exact_ci is not None:
                            lower_exact = str(exact_ci[0])
                            ci[0][0],ci[1][0] = (float(100*v) for v in exact_ci)
                    rates = [v[2] for v in selected]
                    rows.append(dict(setting=setting,scope=scope,kind=kind,name=name,source=None if pair is None else pair[0],
                        target=None if pair is None else pair[1],directions=len(selected),blocks=sum(v[3] for v in selected),
                        accuracy_exact=str(p[0]),accuracy_pp=float(100*p[0]),accuracy_lo_pp=ci[0][0],accuracy_hi_pp=ci[1][0],
                        accuracy_lo_exact=lower_exact,
                        macro_f1_pp=float(100*p[1]),macro_f1_lo_pp=ci[0][1],macro_f1_hi_pp=ci[1][1],
                        disagreement_upper_pp=None if rates[0] is None else float(100*sum(rates)/len(rates))))
    return rows


def matrix_summary(endpoints):
    grouping = defaultdict(list)
    for row in endpoints:
        ref = row['reference']
        if ref is None:
            continue
        for scope in ('fallback','complete') if row['complete'] else ('fallback',):
            ix = (row['setting'],row['source'],row['target'],ref,row['method'],scope)
            grouping[ix].append(row)
    return [dict(setting=a,source=b,target=c,reference=d,method=m,scope=s,
                 seeds=len(rows),accuracy=fmean(r['accuracy'] for r in rows),macro_f1=fmean(r['macro_f1'] for r in rows))
            for (a,b,c,d,m,s),rows in grouping.items()]


def gate(summary,cfg):
    details = {}
    cutoff = Fraction(*cfg['minimum_match_accuracy_fraction'])
    for setting in cfg['settings']:
        current = {r['name']:r for r in summary if r['scope']=='complete' and r['source'] is None
                   and r['setting']==setting and r['kind']=='contrasts'}
        m,b = current['EMatch'],current['EOwnMinusB0']
        low = m['accuracy_lo_pp']
        exact_lower = m['accuracy_lo_exact']
        passed = (Fraction(m['accuracy_exact'])>=cutoff and exact_lower is not None
                  and Fraction(exact_lower)>0 and Fraction(b['accuracy_exact'])>0)
        details[setting] = dict(passed=passed,own_minus_other_pp=m['accuracy_pp'],ci95_low_pp=low,
                                ci95_high_pp=m['accuracy_hi_pp'],own_minus_b0_pp=b['accuracy_pp'])
    proceed = details['proteins_edge']['passed'] and any(
        details[s]['passed'] for s in ('mutagenicity_edge','mutagenicity_node'))
    return dict(settings=details,continue_cross_setting_target_match=proceed)


def score(out,repo,cfg,c,obs_sha):
    roles,groups,actions = observed(out,cfg,c,obs_sha)
    grants,cohorts,access = {},{},{}
    for key,r in roles.items():
        task = {k:r[k] for k in cross.IDENTITY}
        mf,checksum,ctx = historical.context(c,task,repo)
        if checksum!=r['manifest_sha256'] or mf['ids']['holdout']!=r['holdout_ids'] or mf['ids']['adapt']!=r['adapt_ids']:
            raise PermissionError(f'H grant differs from frozen roles: {key}')
        cohort = r['setting'],r['target']
        grant = inputs.raw_grant(ctx,mf,'holdout')
        if grants.setdefault(cohort,grant)!=grant:
            raise PermissionError('Shared target graph cohort changed across sources or seeds')
    reps,draw_seed,minimum,percentile = paired._settings(c.plan)
    with inputs.GraphLabelSession('holdout') as reader:
        for (setting,target),grant in grants.items():
            ids = np.asarray(grant.ids)
            y = reader.load(grant,grant.ids)
            if y.shape!=ids.shape or not np.isin(y,[0,1]).all():
                raise ValueError('Malformed binary H labels')
            order = np.argsort(ids,kind='stable')
            cohort = paired._cohort(setting,target,ids[order],y[order],reps,draw_seed,minimum)
            cohort['integer_draws'] = integer_draws(cohort['weights'],len(ids))
            cohorts[setting,target] = y,order,cohort
            access[f'{setting}-M{target}'] = ids.tolist()
    if len(access)!=12:
        raise PermissionError('Unexpected number of authorized H cohorts')
    endpoints,contrasts,moments = score_blocks(roles,actions,cohorts,cfg['domains'])
    parity = historical_endpoints(repo,cfg,c,endpoints,roles)
    summary = aggregate(cfg,roles,moments,percentile)
    tables = {'endpoints':endpoints,'paired-blocks':contrasts,'summary':summary,
              'crossover-matrix':matrix_summary(endpoints)}
    for name, rows in tables.items():
        write_table(out/f'{name}.csv',rows)
    write_json(out/'gate.json',gate(summary,cfg))
    write_json(out/'cohort-audit.json',[c['audit'] for _,_,c in cohorts.values()])
    write_json(out/'h-label-access.json',dict(obs_sha256=obs_sha,actual_reads=access))
    write_json(out/'historical-parity.json',parity)
    files = ['OBS.json','gate.json','cohort-audit.json','h-label-access.json','historical-parity.json',
             *[f'{name}.csv' for name in tables]]
    return prior_run.seal(out,'FINAL',files,dict(obs_sha256=obs_sha))


def historical_endpoints(repo,cfg,c,endpoints,roles):
    roots = {'r10':Path(cfg['r10']),'r11':repo/cfg['r11']}
    final10 = roc_transfer.stage(roots['r10'],'FINAL',cfg['r10_final_sha256'])
    blob = (roots['r11']/'FINAL.json').read_bytes()
    final11 = json.loads(blob)
    if (final10['predecessors']!={'obs_sha256':cfg['r10_obs_sha256']}
            or inputs.sha(blob)!=cfg['r11_final_sha256'] or final11['stage']!='R11_FINAL'
            or final11['obs_sha256']!=cfg['r11_obs_sha256']):
        raise PermissionError('Historical baseline evaluation provenance differs')
    files = {'r7':c.read('R7_FINAL','endpoints.csv'),
             'r10':prior_run.read_member(roots['r10'],final10['files'],'endpoints.csv'),
             'r11':prior_run.read_member(roots['r11'],final11['files'],'endpoints.csv')}
    indices = {}
    for label,data in files.items():
        indices[label] = {(r['setting'],int(r['source']),int(r['target']),int(r['seed']),r['method']):r
                          for r in csv.DictReader(io.StringIO(data.decode())) if label!='r7' or r['pool']=='holdout'}
    parity = defaultdict(int)
    for r in endpoints:
        task = {k:r[k] for k in cross.IDENTITY}
        key = inputs.key(**task)
        tgt,m = r['target'],r['method']
        donor = {'B0':('r7','B0'),'SourceV':('r11','SourceV'),'WL':('r10','WL_selected'),'Size':('r10','Size')}
        for head in cross.HEADS:
            name = cross.head_name(head,tgt)
            if m==name and roles[key]['head_records'][name]['valid']:
                donor[name] = ('r7',head)
        if m not in donor:
            continue
        label,arm = donor[m]
        old = indices[label][(*task.values(),arm)]
        if int(old['n'])!=r['n'] or abs(float(old['accuracy'])-r['accuracy'])>1e-14:
            raise ArithmeticError(f'Historical H accuracy does not replay: {key}/{m}')
        if 'macro_f1' in old and abs(float(old['macro_f1'])-r['macro_f1'])>1e-14:
            raise ArithmeticError(f'Historical H Macro-F1 does not replay: {key}/{m}')
        parity[label] += 1
    gate_old = json.loads(c.read('R7_FINAL','gate.json'))
    return dict(checked_endpoints=dict(parity),original_c_gate=gate_old)
