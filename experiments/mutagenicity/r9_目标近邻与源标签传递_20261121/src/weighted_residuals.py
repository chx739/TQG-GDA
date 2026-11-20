"""Freeze all twenty-four source-only predictions, then score the opposite-size pools."""
from fractions import Fraction
import json
from statistics import fmean

import numpy as np
from scipy.special import expit

from experiments.common.io import write_json
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.pilot import write_table
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run, roc_rate, roc_score
from . import residual_inputs as data, residual_models as models


BASE_FILES = ('executed-config.json','scope.json','source-topology.json','roles.json','fit-label-access.json','estimates.json',
              'source-provenance.json')
PAIRS = tuple(dict.fromkeys([
    *((method,'WL') for method in models.METHODS if method!='WL'),
    *(('WeightedResidual',method) for method in models.METHODS if method!='WeightedResidual'),
]))
POOL_METRICS = ('strat_mae','pool_bias','pool_abs_error')


def members(keys):
    return [*BASE_FILES,*prior_run.inventory(keys,'predictions'),*(f'models/{key}.json' for key in keys)]


def parents(cfg):
    return {'r10_fit_sha256':cfg['r10_fit_sha256']}


def id_fields(role):
    return {name:role[name] for name in data.IDENTITY}


def freeze(out, cfg, archive, roles, features, topology):
    for folder in ('models','predictions'):
        (out/folder).mkdir()
    estimates = {}
    with data.ScopedLabels(archive.by_setting,roles,'fit') as port:
        for key,region in roles.items():
            ids = region['roles']
            arrays = {f'{name.lower()}_ids':np.asarray(ids[name],dtype=np.int64) for name in ('C','U','E')}
            if region['status']=='ok':
                with port.direction(key):
                    yd = port.load(key,'D',ids['D'])
                    yc = port.load(key,'C',ids['C'])
                subset = features[region['setting'],region['source']]
                model,estimate,arrays = models.fitted_predictions(subset,ids,yd,yc,cfg,archive.plan['solver'])
                estimates[key] = dict(status='ok',**estimate)
            else:
                model = dict(status='unavailable',reason=region['reason'])
                estimates[key] = model
            write_json(out/f'models/{key}.json',model)
            prior_run.save_predictions(out,f'predictions/{key}.npz',arrays)
            print(f"R9 size residuals {key} " + " ".join(f"{r}={len(ids[r])}" for r in data.ROLES)
                  + f" status={estimates[key]['status']}",flush=True)
    data.validate_receipt(roles,port.actual,'fit')
    scope = dict(settings=cfg['settings'],seed=cfg['seed'],directions=len(roles),
                 available=sum(row['status']=='ok' for row in roles.values()),
                 mean_matching_reads_labels=False,fit_label_roles=['D','C'],post_observation_label_roles=['U','E'],
                 historical_cross_domain_input_roles_used=[],fresh_models=2*len(port.actual))
    provenance = {setting:record.receipt for setting,record in archive.by_setting.items()}
    for name,obj in (('source-provenance',provenance),('scope',scope),('source-topology',topology),('roles',roles),
                     ('fit-label-access',port.actual),('estimates',estimates)):
        write_json(out/f'{name}.json',obj)
    return prior_run.seal(out,'OBS',members(roles),parents(cfg))


def verify_observation(out, cfg, expected, topology, obs_sha):
    keys = data.grid(cfg)
    raw = prior_run.verified_members(out,'OBS',obs_sha,members(keys),parents(cfg))
    roles = json.loads(raw['roles.json'])
    estimates = json.loads(raw['estimates.json'])
    if roles != expected or json.loads(raw['source-topology.json'])!=topology or set(estimates)!=set(keys):
        raise PermissionError('Sealed directional roles differ from the registered source split')
    data.validate_receipt(roles,json.loads(raw['fit-label-access.json']),'fit')
    reloaded = {}
    for key,row in roles.items():
        identity = id_fields(row)
        if identity != keys[key]:
            raise PermissionError(f'Direction identity differs: {key}')
        fit_est = estimates[key]
        if row['status']!=fit_est['status']:
            raise PermissionError(f'Direction status differs: {key}')
        x = inputs.npz_member(raw[f'predictions/{key}.npz'])
        for name in ('C','U','E'):
            actual = x[f'{name.lower()}_ids']
            if actual.dtype!=np.int64:
                raise PermissionError(f'Malformed graph IDs: {key}/{name}')
            np.testing.assert_array_equal(actual,row['roles'][name])
        if row['status']!='ok':
            if set(x)!={'c_ids','u_ids','e_ids'} or fit_est['reason']!=row['reason']:
                raise PermissionError(f'Unavailable direction has unexpected predictions: {key}')
            continue
        if set(x)!={'u_ids','e_ids','c_ids','wl_c','wl_u','wl_e','size_u','size_e','weights',*models.METHODS}:
            raise PermissionError(f'Frozen prediction columns differ: {key}')
        for stem,pool in (('wl','c'),('wl','u'),('wl','e'),('size','u'),('size','e')):
            a = x[f'{stem}_{pool}']
            if a.dtype!=np.float64 or a.shape!=x[f'{pool}_ids'].shape or not np.isfinite(a).all():
                raise PermissionError(f'Malformed {stem} margin in {pool}: {key}')
        for method in models.METHODS:
            a = x[method]
            if a.dtype!=np.int64 or a.shape!=x['e_ids'].shape or not np.isin(a,[0,1]).all():
                raise PermissionError(f'Malformed E action: {key}/{method}')
        np.testing.assert_array_equal(x['WL'],x['wl_e']>0.)
        np.testing.assert_array_equal(x['Size'],x['size_e']>0.)
        groups = fit_est['groups']
        n_c,pos_c = fit_est['calibration_n'],fit_est['calibration_positive']
        if n_c!=len(x['c_ids']) or type(pos_c)!=int or not 0<=pos_c<=n_c:
            raise PermissionError(f'Invalid C aggregate label count: {key}')
        calib = Fraction(pos_c,n_c)
        model = json.loads(raw[f'models/{key}.json'])
        constant = Fraction(model['prior_exact']) if model['status']=='single_class_D' else None
        f_c,f_u = (tuple(np.full(len(x[f'wl_{p}']),float(constant)) for p in ('c','u'))
                   if constant is not None else (expit(x['wl_c']),expit(x['wl_u'])))
        if len(groups)!=2 or x['weights'].shape!=(2,len(x['c_ids'])) or x['weights'].dtype!=np.float64:
            raise PermissionError(f'Size layers or C weights differ: {key}')
        for z,g in enumerate(groups):
            u = (x['size_u']>0.)==z
            e = (x['size_e']>0.)==z
            if (g['size_action']!=z or g['u_n']!=int(u.sum()) or g['e_n']!=int(e.sum())
                    or g['status']!=('fit' if u.any() else 'fallback_WL')):
                raise PermissionError(f'Size layer coverage differs: {key}/{z}')
            w = x['weights'][z]
            if u.any():
                checks = g['diagnostics']
                cap = cfg['max_relative_weight']/len(w)
                if (not np.isfinite(w).all() or np.any(w<0.) or np.any(w>cap) or abs(w.sum()-1.)>1e-12
                        or abs(g['decomposition_error'])>1e-12
                        or not -1e-11<=checks['first_order_gap']<=cfg['weight_solver']['max_first_order_gap']):
                    raise PermissionError(f'Matching or decomposition check failed: {key}/{z}')
            elif np.any(w!=0.):
                raise PermissionError(f'Empty U stratum has nonzero C weights: {key}/{z}')
            uniform = bool(u.any()) and np.array_equal(w,np.full(len(w),1./len(w)))
            if (g['matching_uniform']!=uniform or type(g['neighbors_uniform'])!=bool
                    or (not u.any() and g['neighbors_uniform'])):
                raise PermissionError(f'Invalid uniform-weight dispatch: {key}/{z}')
            shared_wl = models.common_wl_probability(f_c,f_u[u])
            if shared_wl!=g['shared_wl_probability']:
                raise PermissionError(f'Invalid shared WL probability: {key}/{z}')
            rates = models.clipped_rates(g['raw'],source_prior=calib,constant=constant,has_u=bool(u.any()),
                                         matching_uniform=uniform,neighbors_uniform=g['neighbors_uniform'],
                                         shared_wl_probability=shared_wl)
            if (g['exact_rate']!={k:str(v) for k,v in rates.items()}
                    or g['clipped']!={k:float(v) for k,v in rates.items()}):
                raise PermissionError(f'Wrong clipping or exact threshold rate: {key}/{z}')
            for name in models.ESTIMATORS:
                cut,n = roc_rate.quantile(x['wl_u'][u],rates[name]) if u.any() else (0.,0)
                stored = g['thresholds'][name]
                if (float(stored['cut'])!=cut or n!=stored['u_predicted_positive']):
                    raise PermissionError(f'Frozen threshold differs: {key}/{z}/{name}')
                np.testing.assert_array_equal(x[name][e],x['wl_e'][e]>cut)
        reloaded[key] = x
    return roles,estimates,reloaded


def proportion_errors(records, z_calls, truth, estimator):
    signed,mae,strata = Fraction(0),Fraction(0),[]
    for z,row in enumerate(records):
        within = truth[z_calls==z]
        if not len(within):
            continue
        obs = Fraction(int(within.sum()),len(within))
        estimate = Fraction(row['exact_rate'][estimator])
        delta = estimate-obs
        scale = Fraction(len(within),len(truth))
        signed += scale*delta
        mae += scale*abs(delta)
        strata.append(dict(size_action=z,n=len(within),positive=int(within.sum()),observed=float(obs),
                           estimate=float(estimate),signed_error=float(delta),abs_error=float(abs(delta)),
                           did_fallback=row['status']=='fallback_WL',raw_estimate=row['raw'][estimator]))
    return dict(strat_mae=mae,pool_bias=signed,pool_abs_error=abs(signed)),strata


def summarize(cfg, roles, rates, label_scores, paired):
    summary, pool_summary, paired_summary, settings_gate = [],[],[],{}
    for setting in cfg['settings']:
        all_keys = [key for key,row in roles.items() if row['setting']==setting]
        keys = [key for key in all_keys if roles[key]['status']=='ok']
        complete = len(keys)==len(all_keys)==8
        if complete:
            accuracy = {m:sum(label_scores[key,m][0] for key in keys)/8 for m in models.METHODS}
            errors = {(p,m):sum(rates[key,p,m]['strat_mae'] for key in keys)/8
                      for p in ('U','E') for m in models.ESTIMATORS}
            main = 'WeightedResidual'
            best = max(('WL','Size'),key=accuracy.__getitem__)
            loss_cut = Fraction(*cfg['minimum_u_mae_gain_fraction'])
            classification_cut = Fraction(*cfg['minimum_e_accuracy_gain_fraction'])
            u_control = {m:errors['U',m]-errors['U',main]
                         for m in ('WLMean','UniformResidual','WeightedLabels','Neighbors8')}
            e_control = {m:errors['E',m]-errors['E',main] for m in ('WLMean','UniformResidual')}
            accuracy_gain = {m:accuracy[main]-accuracy[m] for m in ('WLMean','UniformResidual','WL','Size')}
            u_pass = all(u_control[m]>=loss_cut for m in ('WLMean','UniformResidual')) and all(
                u_control[m]>0 for m in ('WeightedLabels','Neighbors8'))
            e_pass = all(v>0 for v in e_control.values())
            accuracy_pass = all(accuracy_gain[m]>=classification_cut for m in ('WLMean','UniformResidual')) and accuracy_gain[best]>0
            settings_gate[setting] = dict(passed=u_pass and e_pass and accuracy_pass,complete=True,
                u_proportion_pass=u_pass,e_proportion_pass=e_pass,e_accuracy_pass=accuracy_pass,
                u_strat_mae_reduction_pp={m:float(100*d) for m,d in u_control.items()},
                e_strat_mae_reduction_pp={m:float(100*d) for m,d in e_control.items()},
                e_accuracy_gain_pp={m:float(100*d) for m,d in accuracy_gain.items()},strongest_d_classifier=best)
        else:
            settings_gate[setting] = dict(passed=False,complete=False,available=len(keys),expected=len(all_keys))
        for m in models.METHODS:
            scores = [label_scores[k,m] for k in keys]
            summary.append(dict(setting=setting,method=m,available=len(keys),expected=len(all_keys),
                accuracy=fmean(a for a,_ in scores) if complete else None,
                macro_f1=fmean(f for _,f in scores) if complete else None))
        for p in ('U','E'):
            for m in models.ESTIMATORS:
                pool_summary.append(dict(setting=setting,pool=p,method=m,available=len(keys),expected=len(all_keys),
                    **{f:fmean(rates[k,p,m][f] for k in keys) if complete else None for f in POOL_METRICS}))
        for l,r in PAIRS:
            values = [paired[k,l,r] for k in keys]
            paired_summary.append(dict(setting=setting,left=l,right=r,available=len(keys),expected=len(all_keys),
                **{f:sum(v[f] for v in values) if complete else None for f in ('n','corrected','harmed','net','changed')},
                accuracy_pp=fmean(v['accuracy_pp'] for v in values) if complete else None,
                **{f'{f}_percent':fmean(100*v[f]/v['n'] for v in values) if complete else None
                   for f in ('corrected','harmed','changed')}))
    promote = settings_gate['proteins_edge']['passed'] and any(
        settings_gate[s]['passed'] for s in ('mutagenicity_edge','mutagenicity_node'))
    return summary,pool_summary,paired_summary,dict(settings=settings_gate,register_real_domain_validation=promote)


def score(out, cfg, archive, expected, topology, obs_sha):
    roles,estimate,rows = verify_observation(out,cfg,expected,topology,obs_sha)
    rates, label_scores, pairs = {},{},{}
    pool_rows,strata_rows,end_rows,pair_rows,directions = [],[],[],[],[]
    with data.ScopedLabels(archive.by_setting,roles,'score') as port:
        for key,role in roles.items():
            identity = id_fields(role)
            directions.append(dict(**identity,status=role['status'],cut=role['cut'],reason=role['reason'],
                                   **{r:len(role['roles'][r]) for r in data.ROLES}))
            if role['status']!='ok':
                continue
            region = role['roles']
            x = rows[key]
            with port.direction(key):
                y_u = port.load(key,'U',region['U'])
                y_e = port.load(key,'E',region['E'])
            for pool,y in (('U',y_u),('E',y_e)):
                calls = x[f'size_{pool.lower()}']>0.
                for m in models.ESTIMATORS:
                    errors,strata = proportion_errors(estimate[key]['groups'],calls,y,m)
                    rates[key,pool,m] = errors
                    pool_rows.append(dict(**identity,pool=pool,method=m,n=len(y),positives=int(y.sum()),
                                          **{f:float(v) for f,v in errors.items()}))
                    strata_rows.extend(dict(**identity,pool=pool,method=m,**r) for r in strata)
            flags = {m:x[m]==y_e for m in models.METHODS}
            for m in models.METHODS:
                tn,fp,fn,tp = roc_score.confusion(x[m],y_e)
                a = Fraction(tn+tp,len(y_e))
                f = roc_score.f1((tn,fp,fn,tp))
                label_scores[key,m] = a,f
                end_rows.append(dict(**identity,method=m,n=len(y_e),correct=tn+tp,accuracy=float(a),macro_f1=float(f),
                                     predicted_positives=int(x[m].sum()),tn=tn,fp=fp,fn=fn,tp=tp))
            for left,right in PAIRS:
                l,r = flags[left],flags[right]
                corrected,harmed = int((l&~r).sum()),int((~l&r).sum())
                paired = dict(n=len(y_e),corrected=corrected,harmed=harmed,net=corrected-harmed,
                              changed=int((x[left]!=x[right]).sum()),accuracy_pp=100*(corrected-harmed)/len(y_e))
                pairs[key,left,right] = paired
                pair_rows.append(dict(**identity,left=left,right=right,**paired))
    data.validate_receipt(roles,port.actual,'score')
    summary, pool_summary, paired_summary, gate = summarize(cfg,roles,rates,label_scores,pairs)
    tables = {'directions':directions,'proportion-blocks':pool_rows,'proportion-strata':strata_rows,
              'proportion-summary':pool_summary,'endpoints':end_rows,'endpoint-summary':summary,
              'paired-blocks':pair_rows,'paired-summary':paired_summary}
    for name,vals in tables.items():
        write_table(out/f'{name}.csv',vals)
    write_json(out/'gate.json',gate)
    write_json(out/'score-label-access.json',port.actual)
    return prior_run.seal(out,'FINAL',['OBS.json','gate.json','score-label-access.json',
                                       *[f'{n}.csv' for n in tables]],dict(obs_sha256=obs_sha))
