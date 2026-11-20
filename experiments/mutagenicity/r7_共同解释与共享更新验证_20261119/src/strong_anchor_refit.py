"""Freeze refits of the original R7 moment on the historical WL and Size anchors."""
import json

import numpy as np
from scipy.special import expit

from experiments.common.io import write_json
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run
from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src import observation
from experiments.mutagenicity.r16_目标图结构与监督分数蒸馏_20261121.src import inputs as r16
from . import anchor_solver as solver, decisions, fit, head_crossover as cross

ANCHORS = {'WL':'WL_selected','Size':'Size'}
UPDATES = ('Full','B0Teacher','CalTeacher','ScoreOnly')
CONTROLS = (*UPDATES,'OriginalE','EqualCount')
META = ('executed-config.json','roles.json','fit-notes.json','input-access.json')


def parents(cfg):
    return {k:cfg[k] for k in ('r7_final_digest','structural_fit_sha256','structural_obs_sha256','r14_fit_sha256')}


def members(roles):
    return [*META,*prior_run.inventory(roles,'predictions'),*prior_run.inventory(roles,'fits')]


def arm(anchor, name, target=None):
    return f'{anchor}_{name}' if target is None else f'{anchor}_{name}_A{target}'


def read_reference(c, task, cfg):
    ref = cross.read_task(c,task,cfg)
    key = inputs.key(**task)
    prefix = f'blocks/{key}'
    def read(stage,name,fields):
        return inputs.npz_member(c.read(stage,f'{prefix}/{name}'),fields)

    first = read('R7_INPUT','fit_input.npz',['adapt_ids','adapt_z0','adapt_h'])
    std = read('R7_OBS','standardization.npz',['adapt_u','adapt_z0'])
    width = fit.width_key(cfg['tau'])
    old = read('R7_OBS',f'{width}-observed.npz',
               ['theta','stats','ref_theta','adapt_logits_B0','adapt_logits_E'])
    ids = ref['ids']['adapt']
    np.testing.assert_array_equal(first['adapt_ids'],ids)
    u = fit.features(first['adapt_h'],ref['std']['mean'],ref['std']['scale'])
    np.testing.assert_array_equal(u,std['adapt_u'])
    np.testing.assert_array_equal(first['adapt_z0'],std['adapt_z0'])
    z = cross.check_logits(first['adapt_z0'],len(ids))
    g = inputs.margin(z)
    v0 = ref['std']['v0']
    np.testing.assert_allclose(u@(v0[1]-v0[0]),g,rtol=1e-10,atol=1e-12)
    for m in ('B0','E'):
        if not ref['valid'].get(m,{'valid':True})['valid']:
            raise ArithmeticError(f'Unverified historical {m} in the registered seed: {key}')
        archived = cross.check_logits(old[f'adapt_logits_{m}'],len(ids))
        replay,actions = cross.project(z,u,ref['ws'][f'w_{m}'],True)
        np.testing.assert_array_equal(actions,archived.argmax(1))
        np.testing.assert_allclose(inputs.margin(replay),inputs.margin(archived),rtol=0.,atol=1e-12)
    s, index = solver.reference_moment(old['theta'],old['stats'],old['ref_theta'],u)
    return dict(**ref, U_A=u, g_A=g, moment=s, reference_index=index,
                archived_A_E=inputs.margin(old['adapt_logits_E']))


def solve_reference(item, scores_a, cfg):
    u,g_a,s = (item[k] for k in ('U_A','g_A','moment'))
    v0 = item['std']['v0']
    teacher = u.T@expit(g_a)/len(g_a)
    cal,calibration = solver.calibrated_teacher(g_a,u,s,v0)
    q = solver.score_basis(v0)
    kwargs = dict(maxiter=cfg.master_maxiter,tolerance=cfg.optimality_tol)
    lam = cfg.regularization
    raw,original = solver.solve_binary(g_a,u,s,lam,**kwargs)
    a_replay = g_a+u@raw
    z0 = inputs.margin(item['archived']['holdout_logits_B0'])
    h_replay = z0+item['u']@raw
    old_h = inputs.margin(item['archived']['holdout_logits_E'])
    np.testing.assert_array_equal(a_replay>0,item['archived_A_E']>0)
    np.testing.assert_array_equal(h_replay>0,old_h>0)
    np.testing.assert_array_equal(item['archived']['holdout_logits_E'].argmax(1), old_h>0)
    original.update(A_margin_max_abs_error=float(np.max(abs(a_replay-item['archived_A_E']))),
                    H_margin_max_abs_error=float(np.max(abs(h_replay-old_h))),
                    original_weight_max_abs_error=float(np.max(abs(raw-(item['ws']['w_E'][1]-item['ws']['w_E'][0])))))
    residuals = {}
    arrays = dict(A_ids=np.asarray(item['ids']['adapt'],dtype=np.int64),v0=v0,Q=q,
                  original_moment=s,teacher_moment=teacher,calibrated_moment=cal,
                  original_B0_A=g_a,refit_E_update=raw,original_E_A=item['archived_A_E'])
    zero = np.zeros(u.shape[1])
    for anchor,c in scores_a.items():
        arrays[f'{anchor}_A'] = c
        full_at_zero = solver.terms(c,u,s,lam,zero)[1]
        errors = [float(np.max(abs(full_at_zero-solver.terms(c,u,t,lam,zero)[1]-(t-s))))
                  for t in (teacher,cal)]
        if max(errors)>1e-12:
            raise ArithmeticError('Binary master gradient difference identity did not replay')
        fit_log = {}
        for name,target,basis in (('Full',s,None),('B0Teacher',teacher,None),('CalTeacher',cal,None),('ScoreOnly',s,q)):
            arrays[arm(anchor,name)],fit_log[name] = solver.solve_binary(c,u,target,lam,q=basis,**kwargs)
        residuals[anchor] = dict(solutions=fit_log,gradient_identity_inf=errors)
    return arrays,dict(original_E=original,ref_index=item['reference_index'],A_n=len(g_a),
                       regularization=lam,calibrated_teacher=calibration,
                       teacher_moment_difference_inf=float(np.max(abs(cal-s))),anchors=residuals)


def predict_h(item, all_fits, anchors_h, target):
    ids = np.asarray(item['ids']['holdout'],dtype=np.int64)
    old = inputs.margin(item['archived']['holdout_logits_E'])
    b0 = inputs.margin(item['archived']['holdout_logits_B0'])
    np.testing.assert_array_equal(item['archived']['holdout_logits_B0'].argmax(1),b0>0)
    z = {'B0':b0,'E':old,**anchors_h}
    flags = {}
    for anchor,base in anchors_h.items():
        for donor,arrays in all_fits.items():
            z[arm(anchor,'Full',donor)] = base+item['u']@arrays[arm(anchor,'Full')]
        for name in UPDATES[1:]:
            z[arm(anchor,name)] = base+item['u']@all_fits[target][arm(anchor,name)]
        z[arm(anchor,'OriginalE')] = base+(old-b0)
        full = z[arm(anchor,'Full',target)]
        flags[arm(anchor,'EqualCount')],_ = cross.equal_count(ids,base,int((full>0).sum()))
    actions = {m:(v>0).astype(np.int64) for m,v in z.items()}
    if any(not np.isfinite(v).all() for v in z.values()):
        raise ArithmeticError('Nonfinite H margin')
    return dict(ids=ids,**actions,**flags,**{f'margin_{m}':v for m,v in z.items()})


def freeze(output,repo,cfg,c):
    if (c.tau!=cfg['tau'] or decisions.decision_policy(c.plan) is not None
            or list(cfg['primary_anchors'])!=cfg['settings']):
        raise PermissionError('Registered reference width or policy differs')
    historical = r16.Historical(cfg,repo)
    if historical.archive.plan!=c.plan or historical.plan['r7_final_digest']!=cfg['r7_final_digest']:
        raise PermissionError('Structural FIT uses incompatible R7 coordinates or roles')
    historical.held_role()
    for name in ('fits','predictions'):
        (output/name).mkdir()

    roles,notes,pools = {},{},{}
    for setting,source,seed,tasks in observation.group_tasks(cfg,[cfg['phase1_seed']]):
        record,mf,context,model,lineage = historical.group(setting,source,seed)
        maps = {t['target']:t for t in record['tasks']}
        group = {t['target']:read_reference(c,t,cfg) for t in tasks}
        first = next(iter(group.values()))
        refits,predictions = {},{}
        cfg_shared = fit.config(c.plan,seed,cfg['tau'])
        for task in tasks:
            target = task['target']
            key = inputs.key(**task)
            item = group[target]
            if (item['identity']!=first['identity'] or
                    any(not np.array_equal(v,first['std'][name]) for name,v in item['std'].items()) or
                    maps[target]['ids']!=item['ids'] or maps[target]['manifest_sha256']!=item['manifest_sha']):
                raise PermissionError(f'Cross-reference source coordinates or roles differ: {key}')
            cross.check_pool(pools,setting,target,item['ids'])
            def features(pool):
                return historical.features(setting,context,item['ids'][pool],lineage['graph_sha256'])

            pred_a = r16.structural.predict(model,features('adapt'))
            archive_h, h_sha = historical.held_scores(maps[target],features('holdout'),model)
            anchors_h = {anchor:inputs.margin(archive_h[name]) for anchor,name in ANCHORS.items()}
            arrays,notes[key] = solve_reference(item,{a:pred_a[m] for a,m in ANCHORS.items()},cfg_shared)
            refits[target],predictions[target] = arrays,anchors_h
            prior_run.save_predictions(output,f'fits/{key}.npz',arrays)
            roles[key] = dict(**task,manifest_sha256=item['manifest_sha'],adapt_ids=item['ids']['adapt'],
                              holdout_ids=item['ids']['holdout'],R10_H_prediction_sha256=h_sha,
                              source_lineage=lineage,reference_targets=sorted(group))
        for task in tasks:
            target = task['target']
            key = inputs.key(**task)
            x = predict_h(group[target],refits,predictions[target],target)
            prior_run.save_predictions(output,f'predictions/{key}.npz',x)
        print(f'R7 strong anchors {setting} M{source}: 3 references x 2 anchors',flush=True)
    if len(roles)!=36 or len(notes)!=36:
        raise PermissionError('The complete 36-block first-seed grid is required')
    for setting,target_pools in pools.items():
        flat = [i for p in target_pools.values() for ids in p.values() for i in ids]
        if len(flat)!=len(set(flat)):
            raise PermissionError(f'Overlapping A/H graph IDs: {setting}')
    for name,value in (('roles',roles),('fit-notes',notes),('input-access',c.digests)):
        write_json(output/f'{name}.json',value)
    return prior_run.seal(output,'OBS',members(roles),parents(cfg))
