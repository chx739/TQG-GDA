"""Build frozen cross-target actions using R7's original shared linear heads."""
import json
from pathlib import Path

import numpy as np

from experiments.common.io import write_json
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run, roc_transfer
from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src import historical, observation
from . import decisions, fit


IDENTITY = ('setting','source','target','seed')
HEADS = ('E','J')
FIXED = ('B0','SourceV','WL','Size')
METADATA = ('executed-config.json','roles.json','groups.json','historical-plan.json','input-access.json')


def parents(cfg):
    return {k:cfg[k] for k in ('r7_final_digest','r10_obs_sha256','r11_obs_sha256')}


def head_name(method, reference, count=False):
    return f'{method}_{"count_" if count else ""}A{reference}'


def membership(keys):
    return [*METADATA,*prior_run.inventory(keys,'predictions')]


def check_logits(x, n):
    if x.dtype!=np.float64 or x.shape!=(n,2) or not np.isfinite(x).all():
        raise PermissionError('Expected frozen binary float64 logits')
    return x


def equal_count(ids, margin, positive):
    n = len(ids)
    if (not 0<=positive<=n or len(set(ids))!=n or margin.shape!=(n,) or not np.isfinite(margin).all()):
        raise ValueError('Invalid historical rank or positive count')
    order = np.lexsort((ids,-margin))
    actions = np.zeros(n,dtype=np.int64)
    actions[order[:positive]] = 1
    boundary = dict(tied=0,tied_selected=0)
    if 0<positive<n and margin[order[positive-1]]==margin[order[positive]]:
        v = margin[order[positive-1]]
        boundary = dict(tied=int((margin==v).sum()),tied_selected=int((margin[order[:positive]]==v).sum()))
    return actions,boundary


def project(z, u, correction, valid):
    result = fit.logits(z,u,correction) if valid else fit.logits(z,u,np.zeros_like(correction))
    check_logits(result,len(z))
    return result,result.argmax(1)


class Baselines:
    def __init__(self, repo, cfg):
        self.roots = {'r10':Path(cfg['r10']),'r11':repo/cfg['r11']}
        x = roc_transfer.stage(self.roots['r10'],'OBS',cfg['r10_obs_sha256'])
        blob = (self.roots['r11']/'OBS.json').read_bytes()
        y = json.loads(blob)
        if (inputs.sha(blob)!=cfg['r11_obs_sha256'] or y['stage']!='R11_OBS'
                or y['r7_final_digest']!=cfg['r7_final_digest']):
            raise PermissionError('R11 SourceV OBS identity differs')
        self.files = {'r10':x['files'],'r11':y['files']}
        self.roles = {label:json.loads(self.read(label,'roles.json')) for label in self.roots}

    def read(self, scope, relative):
        return prior_run.read_member(self.roots[scope],self.files[scope],relative)

    def actions(self, key, task, manifest_sha, ids, b0_logits, b0):
        for roles in self.roles.values():
            role = roles[key]
            if (any(role[k]!=v for k,v in task.items()) or role['manifest_sha256']!=manifest_sha
                    or role['holdout_ids']!=ids):
                raise PermissionError(f'Historical strong baseline H role differs: {key}')
        path = f'predictions/{key}.npz'
        r10 = inputs.npz_member(self.read('r10',path),['ids','logits_B0','action_B0',
                                                   'logits_WL_selected','action_WL_selected','logits_Size','action_Size'])
        r11 = inputs.npz_member(self.read('r11',path),['ids','margin','B0','SourceV'])
        np.testing.assert_array_equal(r10['ids'],ids)
        np.testing.assert_array_equal(r11['ids'],ids)
        np.testing.assert_array_equal(r10['action_B0'],b0)
        np.testing.assert_array_equal(r11['B0'],b0)
        np.testing.assert_allclose(inputs.margin(r10['logits_B0']),inputs.margin(b0_logits),rtol=0.,atol=1e-12)
        np.testing.assert_allclose(r11['margin'],inputs.margin(b0_logits),rtol=0.,atol=1e-12)
        for m in ('WL_selected','Size'):
            np.testing.assert_array_equal(r10[f'action_{m}'],check_logits(r10[f'logits_{m}'],len(ids)).argmax(1))
        if r11['SourceV'].dtype!=np.bool_ or r11['SourceV'].shape!=(len(ids),):
            raise PermissionError(f'Invalid SourceV actions for {key}')
        return dict(B0=b0,SourceV=r11['SourceV'].astype(np.int64),
                    WL=r10['action_WL_selected'],Size=r10['action_Size'])


def read_task(c, task, cfg):
    mf, manifest_sha = c.roles(task)
    ids = mf['ids']
    if mf['r7_plan']!=c.plan or mf['provenance']['target_labels_loaded'] is not False:
        raise PermissionError('Frozen R7 manifest does not refer to the pinned label-blind execution')
    key = inputs.key(**task)
    prefix = f'blocks/{key}'
    width = fit.width_key(cfg['tau'])

    def read(stage, name, keys=None):
        return inputs.npz_member(c.read(stage,f'{prefix}/{name}'),keys)

    std = read('R7_OBS','standardization.npz',['mean','scale','v0'])
    h = read('R7_INPUT','holdout_input.npz',['holdout_ids','holdout_z0','holdout_h'])
    held = read('R7_OBS','holdout_standardization.npz',['holdout_ids','holdout_z0','holdout_u'])
    for set_ in (h,held):
        np.testing.assert_array_equal(set_['holdout_ids'],ids['holdout'])
    np.testing.assert_array_equal(h['holdout_z0'],held['holdout_z0'])
    z = check_logits(held['holdout_z0'],len(ids['holdout']))
    dim = h['holdout_h'].shape[-1]
    if (any(std[name].dtype!=np.float64 or std[name].shape!=(dim,) or not np.isfinite(std[name]).all()
            for name in ('mean','scale')) or np.any(std['scale']<=0.)
            or std['v0'].shape!=(2,dim+1) or not np.isfinite(std['v0']).all()):
        raise PermissionError(f'Invalid frozen source coordinates: {key}')
    u = fit.features(h['holdout_h'],std['mean'],std['scale'])
    np.testing.assert_array_equal(u,held['holdout_u'])
    np.testing.assert_allclose(u@std['v0'].T,z-z.mean(axis=1,keepdims=True),rtol=1e-10,atol=1e-12)
    archived = read('R7_OBS',f'{width}-heldout.npz',[f'holdout_logits_{m}' for m in ('B0',*HEADS)])
    ws = read('R7_OBS',f'{width}-observed.npz',[f'w_{m}' for m in ('B0',*HEADS)])
    status = json.loads(c.read('R7_OBS',f'{prefix}/{width}-observed.json'))
    if np.any(ws['w_B0']!=0.) or status['B0']['valid'] is not True:
        raise PermissionError(f'Original B0 is not a fixed zero update: {key}')
    lineage = mf['provenance']
    checkpoints = {p:h for p,h in lineage['r5_files'].items() if p.endswith(('/source.pt','/frozen.pt'))}
    if len(checkpoints)!=2 or {Path(p).name for p in checkpoints}!={'source.pt','frozen.pt'}:
        raise PermissionError(f'Missing source checkpoint lineage: {key}')
    identity = dict(source_ids=ids['source'],validation_ids=ids['validation'],checkpoints=checkpoints,
                    selected_source_epoch=lineage['selected_source_epoch'],source_config_hash=lineage['source_config_hash'])
    valid = {}
    for m in HEADS:
        if (type(status[m]['valid'])!=bool or ws[f'w_{m}'].shape!=(2,dim+1) or ws[f'w_{m}'].dtype!=np.float64
                or not np.isfinite(ws[f'w_{m}']).all()):
            raise PermissionError(f'Invalid frozen head certificate: {key}/{m}')
        valid[m] = {field:status[m][field] for field in ('valid','status','stop_reason')}
    return dict(ids=ids,manifest_sha=manifest_sha,z=z,u=u,std=std,identity=identity,archived=archived,ws=ws,valid=valid)


def group_actions(group, target, fixed):
    current = group[target]
    ids = np.asarray(current['ids']['holdout'],dtype=np.int64)
    z,u = current['z'],current['u']
    b0_margin = inputs.margin(current['archived']['holdout_logits_B0'])
    actions = dict(ids=ids,margin_B0=b0_margin,**fixed)
    report = {}
    for reference, donor in group.items():
        for method in HEADS:
            name = head_name(method,reference)
            valid = donor['valid'][method]['valid']
            logits, predicted = project(z,u,donor['ws'][f'w_{method}'],valid)
            error = None
            if target==reference and valid:
                archived = check_logits(current['archived'][f'holdout_logits_{method}'],len(ids))
                np.testing.assert_array_equal(predicted,archived.argmax(1))
                np.testing.assert_allclose(inputs.margin(logits),inputs.margin(archived),rtol=0.,atol=1e-12)
                error = float(np.max(np.abs(inputs.margin(logits)-inputs.margin(archived))))
            if not valid:
                np.testing.assert_array_equal(predicted,fixed['B0'])
            ep,tie = equal_count(ids,b0_margin,int(predicted.sum()))
            actions.update({name:predicted,f'margin_{name}':inputs.margin(logits),head_name(method,reference,count=True):ep})
            report[name] = dict(reference=reference,valid=valid,fallback=not valid,diagonal_margin_error=error,
                                positives=int(predicted.sum()),**tie)
    return actions,report


def check_pool(registry, setting, target, ids):
    pools = registry.setdefault(setting,{})
    pair = dict(A=ids['adapt'],H=ids['holdout'])
    if pools.setdefault(target,pair)!=pair:
        raise PermissionError('Historical target pool or graph order varies by source or seed')


def freeze(out, repo, cfg, c):
    if c.tau!=cfg['tau'] or decisions.decision_policy(c.plan) is not None:
        raise PermissionError('R7 reference must be the registered plain-argmax primary width')
    baseline = Baselines(repo,cfg)
    (out/'predictions').mkdir()
    roles,groups,pools = {},{},{}
    for setting,source,seed,tasks in observation.group_tasks(cfg,[cfg['phase1_seed'],*cfg['reserve_seeds']]):
        group = {task['target']:read_task(c,task,cfg) for task in tasks}
        source_key = f'{setting}-M{source}-seed-{seed}'
        first = next(iter(group.values()))
        if any(item['identity']!=first['identity'] or any(not np.array_equal(v,first['std'][name])
                for name,v in item['std'].items()) for item in group.values()):
            raise PermissionError(f'Source order, checkpoint or coordinate system differs: {source_key}')
        complete = all(r['valid'] for item in group.values() for r in item['valid'].values())
        groups[source_key] = dict(setting=setting,source=source,seed=seed,complete=complete,
            checkpoints=first['identity']['checkpoints'],selected_source_epoch=first['identity']['selected_source_epoch'],
            source_standardization={n:x.tolist() for n,x in first['std'].items()},
            adaptation_counts={str(t):len(item['ids']['adapt']) for t,item in group.items()},
            status={str(t):item['valid'] for t,item in group.items()})
        for task in tasks:
            target = task['target']
            item = group[target]
            ids = item['ids']
            check_pool(pools,setting,target,ids)
            key = inputs.key(**task)
            b0 = item['archived']['holdout_logits_B0']
            saved = check_logits(b0,len(ids['holdout']))
            b0_replay,a0 = project(item['z'],item['u'],item['ws']['w_B0'],True)
            np.testing.assert_array_equal(a0,saved.argmax(1))
            np.testing.assert_allclose(inputs.margin(b0_replay),inputs.margin(saved),rtol=0.,atol=1e-12)
            fixed = baseline.actions(key,task,item['manifest_sha'],ids['holdout'],saved,a0)
            actions,report = group_actions(group,target,fixed)
            path = f'predictions/{key}.npz'
            prior_run.save_predictions(out,path,actions)
            roles[key] = dict(**task,group=source_key,complete=complete,manifest_sha256=item['manifest_sha'],
                              holdout_ids=ids['holdout'],adapt_ids=ids['adapt'],head_records=report,
                              r10_prediction_sha256=baseline.files['r10'][path],r11_prediction_sha256=baseline.files['r11'][path])
        print(f'R7 crossover {source_key}: complete={complete}',flush=True)
    if len(roles)!=c.plan['phases']['C']['primary_blocks_total']:
        raise PermissionError('The full R7 grid was not predicted')
    for setting,targets in pools.items():
        merged = [i for roles_ in targets.values() for pool in roles_.values() for i in pool]
        if len(merged)!=len(set(merged)):
            raise PermissionError(f'Cross-target A/H overlap: {setting}')
    for name,value in (('roles',roles),('groups',groups),('historical-plan',c.plan),('input-access',c.digests)):
        write_json(out/f'{name}.json',value)
    return prior_run.seal(out,'OBS',membership(roles),parents(cfg))
