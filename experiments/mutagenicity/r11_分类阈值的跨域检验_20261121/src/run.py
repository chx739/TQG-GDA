"""Verify, fit the grouped thresholds, then score a separately sealed H."""
from fractions import Fraction
import json
from pathlib import Path
import statistics

import numpy as np

from experiments.common.io import digest, write_json
from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import paired, r5_inputs, seals
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.pilot import write_table
from . import inputs, thresholds

METHODS = ('B0','SourceV','OwnA','OtherA1','OtherA2')
PAIRS = (('OwnA','SourceV'),('OwnA','OtherA1'),('OwnA','OtherA2'),('OwnA','OtherMean'),
         ('OwnA','B0'),('SourceV','B0'),('OtherA1','B0'),('OtherA2','B0'))


def fit_group(setting, validation, group, adapt_labels):
    ordered = sorted(group)
    y = validation['v_port'].load()
    fit_v = thresholds.choose(validation['margin'], y, 0.)
    models, access = {}, {}
    n = min(len(item['ids']['adapt']) for item in group.values())
    for target in ordered:
        item = group[target]
        ids = item['ids']['adapt']
        chosen = thresholds.subset_ids(setting,ids,n)
        index = {gid:row for row,gid in enumerate(ids)}
        labels = adapt_labels.load(item['adapt_port'],chosen)
        margins = item['margins']['adapt'][[index[i] for i in chosen]]
        models[target] = thresholds.choose(margins,labels,fit_v['threshold'])
        access[str(target)] = chosen
    return fit_v,models,access


def freeze(output,repo,plan):
    archive = Path(plan['archive'])
    groups,missing,pools = {},[],{}
    with seals.verified_seal_chain(archive,'R7_FINAL') as scope:
        final = json.loads((archive/'seals/R7_FINAL.json').read_bytes())
        if final['digest'] != plan['r7_final_digest']:
            raise PermissionError('R11 historical FINAL differs')
        original = json.loads(scope.read_bytes('executed-plan.json','R7_INPUT'))
        c = original['phases']['C']
        if (c['settings']!=plan['settings'] or c['domains']!=plan['domains']
                or original['real_model_seeds']!=plan['seeds'] or c['primary_tau']!=plan['tau']):
            raise ValueError('Historical design and registered R11 grid differ')
        for setting,source,seed,targets in inputs.registered(plan):
            try:
                validation,group = inputs.prepare_group(scope,repo,plan,setting,source,seed,targets)
            except (FileNotFoundError,ValueError,KeyError,AssertionError) as error:
                missing.extend(dict(setting=setting,source=source,target=t,seed=seed,reason=str(error)) for t in targets)
                continue
            for item in group.values():
                dataset = item['dataset']
                counts = pools.setdefault(dataset,{'adapt':set(),'holdout':set()})
                for pool in counts:
                    counts[pool].update(item['ids'][pool])
            groups[setting,source,seed] = validation,group
        if any(x['adapt']&x['holdout'] for x in pools.values()):
            raise PermissionError('A and H are not globally disjoint within the same dataset')
        (output/'predictions').mkdir()
        roles, fit_rows, phase = {},[],[]
        with inputs.GraphLabelSession('adapt') as adapt_labels:
            for (setting,source,seed),(v,group) in groups.items():
                fit_v, learned, a_access = fit_group(setting,v,group,adapt_labels)
                phase.append(dict(setting=setting,source=source,seed=seed,validation_ids=v['ids'],
                                  source_checkpoint=v['lineage'],selected_adaptation_ids=a_access))
                for target,item in group.items():
                    others = sorted(t for t in group if t!=target)
                    fit_specs = {'B0':(None,dict(threshold=0.,correct=None,n=0,candidate_count=1)),
                             'SourceV':(None,fit_v),'OwnA':(target,learned[target]),
                             'OtherA1':(others[0],learned[others[0]]),'OtherA2':(others[1],learned[others[1]])}
                    x = item['margins']['holdout']
                    rows = {}
                    for arm,(fit_target,spec) in fit_specs.items():
                        rows[arm] = thresholds.decisions(x,spec['threshold'])
                        fit_rows.append(dict(**item['task'],method=arm,fit_target=fit_target,
                                             **(spec|{'threshold':thresholds.encode_threshold(spec['threshold'])})))
                    path = output/'predictions'/f"{item['key']}.npz"
                    with path.open('xb') as stream:
                        np.savez_compressed(stream,margin=x,ids=np.asarray(item['ids']['holdout']),**rows)
                    roles[item['key']] = dict(**item['task'],holdout_ids=item['ids']['holdout'],
                                              manifest_sha256=item['manifest_sha256'],other_targets=others)
    write_json(output/'roles.json',roles)
    write_json(output/'fit-label-access.json',phase)
    write_json(output/'inputs-verified.json',dict(historical_obs_validated=True,
        full_r7_final_digest=plan['r7_final_digest'],expected_blocks=180,available_blocks=len(roles),
        missing=missing,datasets={k:{name:len(ids) for name,ids in x.items()} for k,x in pools.items()}))
    write_table(output/'thresholds.csv',fit_rows)
    files = ['executed-config.json','roles.json','fit-label-access.json','inputs-verified.json','thresholds.csv']
    files.extend(f'predictions/{name}.npz' for name in roles)
    obs = dict(stage='R11_OBS',r7_final_digest=plan['r7_final_digest'],files={name:digest(output/name) for name in files})
    write_json(output/'OBS.json',obs)
    return digest(output/'OBS.json'),original


def observations(output, checksum,plan):
    payload = (output/'OBS.json').read_bytes()
    seal = json.loads(payload)
    if inputs.sha(payload)!=checksum or seal['stage']!='R11_OBS' or seal['r7_final_digest']!=plan['r7_final_digest']:
        raise PermissionError('R11 OBS identity differs')
    files = {}
    for name,sha in seal['files'].items():
        member = (output/name).read_bytes()
        if inputs.sha(member)!=sha:
            raise PermissionError('R11 frozen observation changed')
        files[name] = member
    roles = json.loads(files['roles.json'])
    if {name for name in files if name.startswith('predictions/')} != {f'predictions/{name}.npz' for name in roles}:
        raise PermissionError('R11 frozen complete-block registry differs')
    return roles,files


def paired_changes(accuracy,weights):
    which = dict(accuracy)
    which['OtherMean'] = (accuracy['OtherA1'].astype(float)+accuracy['OtherA2'])/2
    values = np.column_stack([which[left].astype(float)-which[right].astype(float) for left,right in PAIRS])
    # Compose the mean of the two alternative-domain contrasts inside each
    # shared graph draw, before computing any quantile.
    return weights@values if weights is not None else None


def summarize_pairs(plan,pairs,curves,comparisons):
    rows = []
    fixed = [(s,t) for s in plan['domains'] for t in plan['domains'] if s!=t]
    for setting in plan['settings']:
        for held in [[d] for d in fixed]+[fixed]:
            for left,right in comparisons:
                by_seed,boot,used = [],[],0
                for seed in plan['seeds']:
                    keys = [(setting,s,t,seed,left,right) for s,t in held]
                    present = [key for key in keys if key in pairs]
                    used += len(present)
                    by_seed.append(sum(pairs[key] for key in keys)/len(keys) if len(present)==len(keys) else None)
                    if all(key in curves for key in keys):
                        boot.append(np.mean([curves[key] for key in keys],axis=0))
                complete = all(x is not None for x in by_seed)
                value = sum(by_seed)/len(by_seed) if complete else None
                lo,hi = (100*np.percentile(np.mean(boot,axis=0),[2.5,97.5])).tolist() if len(boot)==len(by_seed) else [None,None]
                rows.append(dict(setting=setting,scope='setting' if len(held)>1 else 'direction',
                    source=None if len(held)>1 else held[0][0],target=None if len(held)>1 else held[0][1],
                    left=left,right=right,pairs=used,expected=len(held)*len(by_seed),
                    mean_pp=float(100*value) if complete else None, exact_fraction=str(value) if complete else None,
                    sd_pp=100*statistics.stdev(float(x) for x in by_seed) if complete and len(by_seed)>1 else None,
                    lower_pp=lo,upper_pp=hi, per_seed_pp=[None if val is None else float(100*val) for val in by_seed]))
    return rows


def aggregate(plan,pairs,curves):
    rows = summarize_pairs(plan,pairs,curves,PAIRS)
    gates = {}
    fixed = [(s,t) for s in plan['domains'] for t in plan['domains'] if s!=t]
    for setting in plan['settings']:
        summary = {r['right']:r for r in rows if r['setting']==setting and r['scope']=='setting' and r['left']=='OwnA'}
        positive = sum(all(all((setting,s,t,seed,'OwnA',arm) in pairs for seed in plan['seeds'])
                           and sum(pairs[setting,s,t,seed,'OwnA',arm] for seed in plan['seeds'])>0
                           for arm in ('SourceV','OtherMean')) for s,t in fixed)
        cutoff = Fraction(*plan['gates']['minimum_mean_accuracy_fraction'])
        value = {arm: Fraction(summary[arm]['exact_fraction']) if summary[arm]['exact_fraction'] is not None else None
                 for arm in ('SourceV','OtherMean','B0')}
        passed = (all(value[arm] is not None and value[arm]>=cutoff for arm in ('SourceV','OtherMean'))
                  and value['B0'] is not None and value['B0']>0
                  and positive>=plan['gates']['positive_directions_per_setting'])
        gates[setting] = dict(passed=passed,simultaneously_positive_directions=positive,
                             comparison_pp={arm: None if x is None else float(100*x) for arm,x in value.items()})
    return rows,dict(settings=gates,passed=sum(x['passed'] for x in gates.values())>=plan['gates']['settings_required'])


def endpoint_summary(plan,endpoints):
    values = {(r['setting'],r['source'],r['target'],r['seed'],r['method']):Fraction(r['correct'],r['n'])
              for r in endpoints}
    output = []
    scopes = [(s,t) for s in plan['domains'] for t in plan['domains'] if s!=t]
    for setting in plan['settings']:
        for subset in [[d] for d in scopes]+[scopes]:
            for name in (*METHODS,'HUpper'):
                rows,count = [],0
                for seed in plan['seeds']:
                    keys = [(setting,s,t,seed,name) for s,t in subset]
                    matched = [values[k] for k in keys if k in values]
                    count += len(matched)
                    rows.append(sum(matched)/len(matched) if len(matched)==len(keys) else None)
                complete = all(x is not None for x in rows)
                output.append(dict(setting=setting,scope='direction' if len(subset)==1 else 'setting',
                    source=subset[0][0] if len(subset)==1 else None,target=subset[0][1] if len(subset)==1 else None,
                    method=name,n_blocks=count,expected_blocks=len(subset)*len(plan['seeds']),
                    mean_accuracy=float(sum(rows)/len(rows)) if complete else None,
                    accuracy_sd=statistics.stdev(float(x) for x in rows) if complete else None))
    return output


def score(output,repo,checksum,plan,original):
    roles,files = observations(output,checksum,plan)
    archive = Path(plan['archive'])
    endpoints, contrasts, pair_values, curves, cohorts, held = [],[],{},{},{},{}
    h_access = {}
    reps,seed,minimum,_ = paired._settings(original)
    with seals.verified_seal_chain(archive,'R7_FINAL') as scope, inputs.GraphLabelSession('holdout') as h_labels:
        if json.loads((archive/'seals/R7_FINAL.json').read_bytes())['digest']!=plan['r7_final_digest']:
            raise PermissionError('Observed R11 refers to a different R7 FINAL')
        for key,task in roles.items():
            s,src,t,model_seed = (task[k] for k in ('setting','source','target','seed'))
            x = inputs.npz_member(files[f'predictions/{key}.npz'])
            np.testing.assert_array_equal(x['ids'],task['holdout_ids'])
            role_blob = scope.read_bytes(f'blocks/{key}/manifest.json','R7_INPUT')
            if inputs.sha(role_blob)!=task['manifest_sha256']:
                raise PermissionError('H authorization references a different role manifest')
            manifest = json.loads(role_blob)
            if manifest['ids']['holdout']!=task['holdout_ids']:
                raise PermissionError('Sealed H predictions and historical IDs do not agree')
            cohort_key = s,t
            if cohort_key not in cohorts:
                context = r5_inputs._context(manifest['r7_plan'],s,src,t,model_seed,repo)
                values = h_labels.load(inputs.raw_grant(context,manifest,'holdout'),task['holdout_ids'])
                order = np.argsort(x['ids'],kind='stable')
                cohorts[cohort_key] = (values,order,paired._cohort(s,t,x['ids'][order],values[order],reps,seed,minimum))
                h_access[f'{s}-M{t}']=task['holdout_ids']
            if h_access[f'{s}-M{t}']!=task['holdout_ids']:
                raise PermissionError('H graph order or identity varies within a shared target cohort')
            y,order,cohort = cohorts[cohort_key]
            correct = {name:x[name]==y for name in METHODS}
            scores = {name:int(a.sum()) for name,a in correct.items()}
            identity = {name:task[name] for name in ('setting','source','target','seed')}
            maxima = thresholds.choose(x['margin'],y,0.)['correct']
            n = len(y)
            held.setdefault((s,src,model_seed),{})[t]=(x['margin'],y,scores['SourceV'],maxima)
            for name in METHODS:
                endpoints.append(dict(**identity,method=name,n=n,correct=scores[name],accuracy=scores[name]/n,
                                      recall_0=float(correct[name][y==0].mean()) if np.any(y==0) else None,
                                      recall_1=float(correct[name][y==1].mean()) if np.any(y==1) else None))
            endpoints.append(dict(**identity,method='HUpper',n=n,correct=maxima,accuracy=maxima/n))
            boot = paired_changes({name:a[order] for name,a in correct.items()},cohort['weights'])
            for i,(left,right) in enumerate(PAIRS):
                k = (s,src,t,model_seed,left,right)
                if right=='OtherMean':
                    net = Fraction(2*scores[left]-scores['OtherA1']-scores['OtherA2'],2)
                    extra = dict(corrected=None,harmed=None,disagreement=None)
                else:
                    a,b = correct[left],correct[right]
                    net = Fraction(scores[left]-scores[right])
                    extra = dict(corrected=int((a&~b).sum()),harmed=int((~a&b).sum()),
                                 disagreement=int((x[left]!=x[right]).sum()))
                pair_values[k]=net/n
                if boot is not None:
                    curves[k]=boot[:,i]
                contrasts.append(dict(**identity,left=left,right=right,n=n,net_correct_exact=str(net),
                                      accuracy_pp=float(100*pair_values[k]),**extra))
    observations(output,checksum,plan)
    upper = []
    for (setting,source,seed),pools in held.items():
        if len(pools)!=3:
            continue
        ordered = [pools[t] for t in sorted(pools)]
        individual,separate,shared = thresholds.h_upper_bounds([(x,y) for x,y,_,_ in ordered])
        if individual!=[maximum for _,_,_,maximum in ordered] or shared>separate:
            raise ValueError('H empirical upper bound arithmetic differs')
        v = sum(Fraction(v,len(y)) for _,y,v,_ in ordered)/3
        upper.append(dict(setting=setting,source=source,seed=seed,directions=3,separate=float(separate),
                    shared=float(shared),v=float(v),separate_minus_shared_pp=float(100*(separate-shared)),
                    exact_separate_minus_shared=str(separate-shared),separate_minus_v_pp=float(100*(separate-v))))
    rows,gate = aggregate(plan,pair_values,curves)
    write_table(output/'endpoints.csv',endpoints)
    write_table(output/'endpoint-summary.csv',endpoint_summary(plan,endpoints))
    write_table(output/'paired-blocks.csv',contrasts)
    write_table(output/'h-upper.csv',upper)
    write_table(output/'paired-summary.csv',rows)
    write_json(output/'gate.json',gate)
    write_json(output/'h-label-access.json',dict(obs_sha256=checksum,roles=h_access))
    finals = ['OBS.json','endpoints.csv','endpoint-summary.csv','paired-blocks.csv','h-upper.csv','paired-summary.csv','gate.json','h-label-access.json']
    write_json(output/'FINAL.json',dict(stage='R11_FINAL',obs_sha256=checksum,
                                       files={name:digest(output/name) for name in finals}))
    return gate
