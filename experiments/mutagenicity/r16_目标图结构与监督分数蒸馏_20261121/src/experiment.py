"""Seal source stacks and four equal-budget mappings before predicting H."""
import json

import numpy as np

from experiments.common.io import write_json
from experiments.mutagenicity.r10_原生图适配基线与增量_20261121.src import structural
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs as r11, prior_run as storage
from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src import observation
from . import inputs, models

IDENTITY = structural.IDENTITY
PRIMARY = ('B0','SourceCal','WL_selected','Size','B','B2','Low','Own','LowTwice','VFull',
           'VResidual','OtherResidual1','OtherResidual2','B_equal_positive')
DIAGNOSTICS = ('g','g_low','g_own','own_residual_effect')


def parents(cfg):
    return dict(r10_fit_sha256=cfg['structural_fit_sha256'], r14_fit_sha256=cfg['r14_fit_sha256'])


def fit_members(plan,seeds):
    groups, _ = observation.observations_expected(plan,seeds)
    return ['executed-config.json','fit-roles.json','historical-plan.json',*[f'fits/{g}.json' for g in groups]]


def obs_members(plan,seeds):
    _, tasks = observation.observations_expected(plan,seeds)
    return ['FIT.json','prediction-roles.json',*storage.inventory(tasks,'predictions')]


def as_margins(g, features, fitted):
    predicted = structural.predict(fitted,features)
    return np.column_stack((g,predicted['WL_selected'],predicted['Size'])), np.stack([x[1] for x in features])


def fit(output,repo,cfg,seeds,*,preflight=False):
    history = inputs.Historical(cfg,repo)
    plan = history.plan
    roles = {}
    if not preflight:
        (output/'fits').mkdir()
    for setting, source, seed, _ in observation.group_tasks(plan,seeds):
        role,mf,ctx,structural_fit,lineage = history.group(setting,source,seed)
        name = f'{setting}-M{source}-seed-{seed}'
        roles[name] = role
        if preflight:
            continue
        s,v,pools = role['source_fit'],role['validation'],role['pools']
        source_points = history.features(setting,ctx,s,lineage['graph_sha256'])
        validation_points = history.features(setting,ctx,v,lineage['graph_sha256'])
        g_raw,ys,yv,label_sha = history.source_scores(role['tasks'],s,v,mf,ctx)
        candidate = inputs.verify_structural(structural_fit,source_points,validation_points,ys,yv,plan)
        v_margins,v_sizes = as_margins(g_raw,validation_points,structural_fit)
        stack = models.fit_stack(v_margins,v_sizes,yv,plan['solver'])
        mappings = {}
        v_index = {i:n for n,i in enumerate(v)}
        tasks = {f'A{row["target"]}':row for row in role['tasks']}
        for pool,ids in pools.items():
            if pool=='V':
                rows = [v_index[i] for i in ids]
                features = [validation_points[i] for i in rows]
                margins,sizes = v_margins[rows],v_sizes[rows]
            else:
                features = history.features(setting,ctx,ids,lineage['graph_sha256'])
                margins,sizes = as_margins(history.a_scores(tasks[pool],ids),features,structural_fit)
            g,x,_,_ = models.design(margins,sizes,stack)
            mappings[pool] = models.fit_pool(x,g,[f[0] for f in features],cfg['ridge_numerical_tolerance'])
        write_json(output/f'fits/{name}.json',
                   dict(setting=setting, source=source, seed=seed,lineage=lineage,source_label_sha256=label_sha,
                        structural_candidate_max_gradient=candidate,stack=stack,reference=mappings,
                        validation_ids=v,validation_margins=v_margins.tolist(),validation_sizes=v_sizes.tolist()))
        print(f'R16 fit {name} V={len(v)} K={len(pools["V"])} wG={stack["B"]["weights"][0]:.5g}',flush=True)
    if preflight:
        return {k: len(v['pools']['V']) for k,v in roles.items()}
    write_json(output/'fit-roles.json',roles)
    write_json(output/'historical-plan.json',history.archive.plan)
    return storage.seal(output,'FIT',fit_members(plan,seeds),parents(cfg))


def verified_fit(out,plan,cfg,seeds,fit_sha):
    return storage.verified_members(out,'FIT',fit_sha,fit_members(plan,seeds),parents(cfg))


def predict(output,repo,cfg,seeds,fit_sha):
    history = inputs.Historical(cfg,repo)
    plan = history.plan
    members = verified_fit(output,plan,cfg,seeds,fit_sha)
    roles = json.loads(members['fit-roles.json'])
    history.held_role()
    (output/'predictions').mkdir()
    outcomes = {}
    for setting,source,seed,_ in observation.group_tasks(plan,seeds):
        name = f'{setting}-M{source}-seed-{seed}'
        role,_,ctx,r10,lineage = history.group(setting,source,seed)
        model = json.loads(members[f'fits/{name}.json'])
        if role!=roles[name] or lineage!=model['lineage']:
            raise PermissionError('Frozen fit registry differs before H prediction')
        fit = model['stack']
        wg = fit['B']['weights'][0]
        references = {pool:[z[0] for z in history.features(setting,ctx,ids,lineage['graph_sha256'])]
                      for pool,ids in role['pools'].items()}
        for task in role['tasks']:
            target = task['target']
            own = f'A{target}'
            other = sorted(p for p in references if p not in ('V',own))
            ids = task['ids']['holdout']
            held = history.features(setting,ctx,ids,lineage['graph_sha256'])
            archived, sha = history.held_scores(task,held,r10)
            margins = np.column_stack([inputs.margin(archived[m]) for m in ('B0','WL_selected','Size')])
            g,x,b,b2 = models.design(margins,np.stack([z[1] for z in held]),fit)
            mapped = {p:models.predict_pool(model['reference'][p],x,[z[0] for z in held],rows)
                      for p,rows in references.items()}
            l,k,_ = mapped[own]
            predicted = models.compose(b,g,wg,mapped[own],mapped['V'],[mapped[p] for p in other],cfg['blend'])
            logits = archived | {m:np.column_stack((np.zeros(len(ids)),z))
                                 for m,z in {'B':b,'B2':b2,**predicted}.items()}
            arrays = dict(ids=np.asarray(ids),g=g,g_low=l,g_own=l+k,own_residual_effect=cfg['blend']*wg*k)
            for arm,z in logits.items():
                arrays[f'logits_{arm}'] = z
                arrays[f'decision_{arm}'] = np.asarray(inputs.margin(z)>0., dtype=np.int64)
            arrays['logits_B_equal_positive'] = logits['B']
            arrays['decision_B_equal_positive'] = observation.equal_positive(
                arrays['ids'],logits['B'],int(arrays['decision_Own'].sum()))
            key = r11.key(**{k:task[k] for k in IDENTITY})
            storage.save_predictions(output,f'predictions/{key}.npz',arrays)
            outcomes[key] = {k:task[k] for k in IDENTITY} | dict(group=name,ids=ids,manifest_sha256=task['manifest_sha256'],
                      r10_prediction_sha256=sha,**{'own':own,'other':other})
        print(f'R16 frozen-ready H {name}',flush=True)
    write_json(output/'prediction-roles.json',outcomes)
    return storage.seal(output,'OBS',obs_members(plan,seeds),{'fit_sha256':fit_sha})


def checked_predictions(role,arrays):
    n = len(role['ids'])
    expected = {'ids',*DIAGNOSTICS,*[f'{stem}_{m}' for stem in ('logits','decision') for m in PRIMARY]}
    if set(arrays)!=expected:
        raise PermissionError('R16 prediction columns differ from registration')
    np.testing.assert_array_equal(arrays['ids'],role['ids'])
    for m in PRIMARY:
        z, a = arrays[f'logits_{m}'],arrays[f'decision_{m}']
        if z.shape!=(n,2) or a.shape!=(n,) or a.dtype!=np.int64 or not np.isfinite(z).all():
            raise ValueError('Malformed frozen classification arrays')
        pred = (observation.equal_positive(arrays['ids'],z,int(arrays['decision_Own'].sum()))
                if m=='B_equal_positive' else np.asarray(inputs.margin(z)>0.,dtype=np.int64))
        np.testing.assert_array_equal(a,pred)
    for name in DIAGNOSTICS:
        if arrays[name].shape!=(n,) or not np.isfinite(arrays[name]).all():
            raise ValueError('Nonfinite or malformed frozen R16 mechanism values')
    np.testing.assert_array_equal(arrays['logits_B_equal_positive'],arrays['logits_B'])
    np.testing.assert_allclose(inputs.margin(arrays['logits_Own'])-inputs.margin(arrays['logits_Low']),
                               arrays['own_residual_effect'],atol=1e-12,rtol=0.)
