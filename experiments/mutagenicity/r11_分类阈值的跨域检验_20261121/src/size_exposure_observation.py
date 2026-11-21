"""Authenticate the original protein cohorts and seal the 60 fixed deployments."""
from fractions import Fraction as F
import json

import numpy as np
from scipy.special import expit

from experiments.common.io import write_json
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src import composition
from experiments.mutagenicity.r10_原生图适配基线与增量_20261121.src import structural, structural_observation as so
from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src import historical
from . import inputs, prior_run, roc_transfer, size_exposure as curve, thresholds

ID = prior_run.ID_FIELDS
META = ('executed-config.json','roles.json','models.json','decisions.json')


def parents(cfg,roc_cfg):
    return {k:roc_cfg[k] for k in ('prior_unlabeled_sha256','structural_fit_sha256','structural_obs_sha256')} | {
        'pilot_roc_unlabeled_sha256':cfg['pilot_roc_unlabeled_sha256']}


def members(keys,groups):
    return [*META,*prior_run.inventory(keys,'predictions'),*prior_run.inventory(groups,'validation')]


class Archive:
    def __init__(self,repo,here,cfg,roc_cfg):
        if (cfg['settings']!=['proteins_edge'] or cfg['domains']!=list(range(4))
                or cfg['seeds']!=list(range(3405,3410)) or cfg['minimum_accuracy_fraction']!=[1,200]
                or cfg['minimum_positive_directions']!=7):
            raise PermissionError('Protein size-exposure grid or gate differs from registration')
        self.repo = repo
        (main,self.main_groups,self.main_roles,self.prior,self.u_files,self.structural,
         self.s_fit,self.s_obs) = roc_transfer.historical_inputs(repo,here,roc_cfg)
        self.plan = dict(settings=cfg['settings'],domains=cfg['domains'],seeds=cfg['seeds'])
        self.c = historical.CArchive(main | dict(r7_archive=main['archive'],phase1_seed=cfg['seeds'][0],
                                                reserve_seeds=cfg['seeds'][1:]))
        self.prior_roles = json.loads(prior_run.read_member(self.prior,self.u_files,'roles.json'))
        self.s_roles = json.loads(prior_run.read_member(self.structural,self.s_obs,'roles.json'))
        self.fit_access = json.loads(prior_run.read_member(self.structural,self.s_fit,'fit-access.json'))
        self.pilot = here/cfg['pilot_roc_attempt']
        pilot = roc_transfer.stage(self.pilot,'UNLABELED',cfg['pilot_roc_unlabeled_sha256'])
        expected = {k:v for k,v in parents(cfg,roc_cfg).items() if k!='pilot_roc_unlabeled_sha256'}
        if pilot['predecessors']!=expected:
            raise PermissionError('Historical pilot ROC carries different graph inputs')
        self.pilot_files = pilot['files']
        self.pilot_roles = json.loads(prior_run.read_member(self.pilot,self.pilot_files,'roles.json'))
        self.store = None

    def context(self,task):
        key = inputs.key(**task)
        manifest,sha,ctx = historical.context(self.c,task,self.repo)
        r = self.prior_roles[key]
        if (any(r[k]!=v for k,v in self.main_roles[key].items()) or any(r[k]!=v for k,v in task.items())
                or r['manifest_sha256']!=sha or r['holdout_ids']!=manifest['ids']['holdout']
                or not set(r['selected_a_ids'])<=set(manifest['ids']['adapt'])):
            raise PermissionError('A/H exposure roles differ from the historically registered R11 roles')
        return key,r,manifest,ctx

    def graphs(self,ctx,access,ids,authorized):
        if not set(ids)<=set(authorized):
            raise PermissionError('Protein graphs outside the authorized S/V/A/H role')
        if self.store is None:
            self.store = historical.GraphStore(ctx)
        if (ctx['root']!=self.store_root or self.store.graph_sha!=access['graph_container_sha256']
                or access['class_mapping']!=ctx['label_meta']['mapping']):
            raise PermissionError('R10 and R11 refer to different proteins graph or label identities')
        return self.store.take(ids)

    def group(self,setting,source,seed,targets):
        group = so.group_key(setting,source,seed)
        access = self.fit_access[group]
        size_path = f'fits/{group}.json'
        fitted = json.loads(prior_run.read_member(self.structural,self.s_fit,size_path))
        if (fitted['setting'],fitted['source'],fitted['seed'])!=(setting,source,seed):
            raise PermissionError('R10 Size wrong source model identity')
        registry = self.main_groups[setting,source,seed]
        rows = {}
        roles = []
        for t in targets:
            key,r,m,ctx = self.context(dict(setting=setting,source=source,target=t,seed=seed))
            if self.store is None:
                self.store_root = ctx['root']
            fits = m['ids']
            checkpoint = registry['source_checkpoint']
            if (fits['source']!=access['source_ids'] or fits['validation']!=access['validation_ids']
                    or registry['validation_ids']!=fits['validation']
                    or checkpoint['sha256']!=m['provenance']['r5_files'][checkpoint['path']]
                    or checkpoint['selected_epoch']!=m['provenance']['selected_source_epoch']
                    or r['other_targets']!=sorted(tt for tt in targets if tt!=t)):
                raise PermissionError('Protein source role or checkpoint differs from previous fit')
            rows[t] = key,r,m,ctx
            roles.append(fits)
        structural.check_roles(roles)
        budget = min(len(r['adapt']) for r in roles)
        for t,(_,r,m,_) in rows.items():
            a = r['selected_a_ids']
            if registry['selected_adaptation_ids'][str(t)]!=a or thresholds.subset_ids(setting,m['ids']['adapt'],budget)!=a:
                raise PermissionError('Selected protein A pool differs from R11 equal-budget version')
        return group,rows,access,fitted['size']

    def original_target(self,key,role,manifest,ctx,access,size_model):
        path = f'blocks/{key}'
        a_ids,h_ids = role['selected_a_ids'],role['holdout_ids']
        full = manifest['ids']['adapt']
        a = inputs.npz_member(self.c.read('R7_OBS',f'{path}/tau-{self.c.tau:.3f}-observed.npz'),
                              ['adapt_logits_B0','w_B0'])
        origin = inputs.npz_member(self.c.read('R7_INPUT',f'{path}/fit_input.npz'),['adapt_ids','adapt_z0'])
        np.testing.assert_array_equal(origin['adapt_ids'],full)
        a_margin = inputs.margin(a['adapt_logits_B0'])
        np.testing.assert_allclose(a_margin,inputs.margin(origin['adapt_z0']),rtol=0,atol=1e-12)
        if np.any(a['w_B0']):
            raise PermissionError('Protein threshold transfer is not at frozen B0')
        rows = {g:i for i,g in enumerate(full)}
        a_margin = a_margin[[rows[g] for g in a_ids]]
        hist = inputs.npz_member(prior_run.read_member(self.prior,self.u_files,f'unlabeled/{key}.npz'),
                                 ['ids','margin','B0'])
        r10 = inputs.npz_member(prior_run.read_member(self.structural,self.s_obs,f'predictions/{key}.npz'))
        fit_path = f'fits/{so.group_key(role["setting"],role["source"],role["seed"])}.json'
        s = self.s_roles[key]
        if (any(s[k]!=role[k] for k in (*ID,'manifest_sha256','holdout_ids')) or s['fit_sha256']!=self.s_fit[fit_path]):
            raise PermissionError('Previous Size actions have different block identity')
        so.verify_predictions(r10,s)
        np.testing.assert_array_equal(hist['ids'],h_ids)
        np.testing.assert_array_equal(r10['ids'],h_ids)
        h_margin = hist['margin']
        np.testing.assert_allclose(h_margin,inputs.margin(self.c.b0({k:role[k] for k in ID},h_ids)),rtol=0,atol=1e-12)
        np.testing.assert_allclose(h_margin,inputs.margin(r10['logits_B0']),rtol=0,atol=1e-12)
        np.testing.assert_array_equal(h_margin>0.,hist['B0'])
        np.testing.assert_array_equal(hist['B0'],r10['action_B0'])
        a_graphs = self.graphs(ctx,access,a_ids,full)
        h_graphs = self.graphs(ctx,access,h_ids,h_ids)
        a_size = size_margin(size_model,a_graphs)
        return dict(ids=hist['ids'],a_ids=np.asarray(a_ids,dtype=np.int64),a_margin=a_margin,h_margin=h_margin,
                    a_nodes=node_counts(a_graphs),h_nodes=node_counts(h_graphs),a_old_size=a_size,
                    B0=hist['B0'].astype(bool),Size=r10['action_Size'].astype(bool))

    def check_pilot(self,key,role,x):
        if role['seed']!=3405:
            return
        r = self.pilot_roles[key]
        if any(r[k]!=role[k] for k in (*ID,'manifest_sha256','holdout_ids','selected_a_ids','other_targets')):
            raise PermissionError('Primary protein pilot replay does not match the archived R11 ROC cohort')
        wanted = ('ids','a_margin','h_margin','a_size_margin','B0','Size','PoolMajority','DirectP')
        prev = inputs.npz_member(prior_run.read_member(self.pilot,self.pilot_files,f'deployment/{key}.npz'),wanted)
        for k in wanted:
            field = {'a_size_margin':'a_old_size','PoolMajority':'OldPool','DirectP':'OldDirectP'}.get(k,k)
            np.testing.assert_array_equal(prev[k],x[field])


def node_counts(graphs):
    return np.asarray([len(g.x) for g in graphs],dtype=np.int64)


def size_margin(model,graphs):
    raw = np.asarray([structural.size(g) for g in graphs])
    return composition.scores((raw-model['mean'])/model['scale'],structural.select_model(model))


def calculate(archive,*,replay=None):
    roles,models,deploy,validation,decisions = {},{},{},{},{}
    for setting,source,seed,targets in inputs.registered(archive.plan):
        group,rows,access,old_size = archive.group(setting,source,seed,targets)
        _,_,m,ctx = next(iter(rows.values()))
        s_ids,v_ids = access['source_ids'],access['validation_ids']
        v_graphs = archive.graphs(ctx,access,v_ids,m['ids']['validation'])
        identity = dict(source_ids=s_ids,validation_ids=v_ids,s_label_sha256=access['s_label_sha256'],
                        v_label_sha256=access['v_label_sha256'],graph_container_sha256=access['graph_container_sha256'],
                        class_mapping=access['class_mapping'])
        if replay is None:
            s_graphs = archive.graphs(ctx,access,s_ids,m['ids']['source'])
            y,sha = so.source_labels(ctx,m,source,seed,'fit')
            if sha!=identity['s_label_sha256']:
                raise PermissionError('Exposure must fit only the original R10 S labels')
            fitted = curve.fit(node_counts(s_graphs),y)
        else:
            recorded = replay[group]
            if recorded['identity']!=identity or recorded['fit']['n_source']!=len(s_ids):
                raise PermissionError('Frozen fit refers to a different S graph or label identity')
            fitted = recorded['fit']
        models[group] = dict(setting=setting,source=source,seed=seed,identity=identity,fit=fitted)
        v_nodes = node_counts(v_graphs)
        validation[group] = dict(ids=np.asarray(v_ids,dtype=np.int64),nodes=v_nodes,
                                 Curve=curve.predict(v_nodes,fitted)>.5,Size=size_margin(old_size,v_graphs)>0.)
        base = {t:archive.original_target(*r,access,old_size) for t,r in rows.items()}
        p = {t:curve.exposure_prior(x['a_nodes'],fitted) for t,x in base.items()}
        a = {t:x['a_margin'] for t,x in base.items()}
        for t,(key,r,_,_) in rows.items():
            x = base[t]
            old_p = curve.pool_probability(expit(x['a_old_size']))
            p_h = curve.predict(x['h_nodes'],fitted)
            actions,notes = curve.policies(t,p,a,x['h_margin'],old_p,p_h,fitted)
            roles[key] = r | {'group':group}
            deploy[key] = dict(**x,**actions,h_curve_probability=p_h,
                               a_curve_probability=curve.predict(x['a_nodes'],fitted))
            decisions[key] = notes
            archive.check_pilot(key,r,deploy[key])
        if replay is None:
            print(f'R11 exposure {group} S={len(s_ids)} lambda={fitted["rate"]}',flush=True)
    if set(deploy)!=set(prior_run.registered_keys(archive.plan)) or len(deploy)!=60 or len(models)!=20:
        raise PermissionError('Incomplete 20-model, 60-direction/seed exposure grid')
    return roles,models,deploy,validation,decisions


def freeze(out,archive,cfg,roc_cfg):
    roles,models,deploy,validation,details = calculate(archive)
    for name,rows in (('predictions',deploy),('validation',validation)):
        (out/name).mkdir()
        for key,x in rows.items():
            prior_run.save_predictions(out,f'{name}/{key}.npz',x)
    for n,data in (('roles',roles),('models',models),('decisions',details)):
        write_json(out/f'{n}.json',data)
    return prior_run.seal(out,'OBS',members(roles,models),parents(cfg,roc_cfg))


def observed(out,archive,cfg,roc_cfg,sha):
    keys = prior_run.registered_keys(archive.plan)
    groups = [so.group_key(setting,source,seed) for setting,source,seed,_ in inputs.registered(archive.plan)]
    raw = prior_run.verified_members(out,'OBS',sha,members(keys,groups),parents(cfg,roc_cfg))
    registered = json.loads(raw['models.json'])
    roles,models,deploy,validation,details = calculate(archive,replay=registered)
    if (registered!=models or roles!=json.loads(raw['roles.json']) or details!=json.loads(raw['decisions.json'])
            or json.loads(raw['executed-config.json'])['plan']!=cfg):
        raise PermissionError('Sealed exposure choices or role lineage differ')
    for name,rows in (('predictions',deploy),('validation',validation)):
        for key,x in rows.items():
            frozen = inputs.npz_member(raw[f'{name}/{key}.npz'])
            if set(frozen)!=set(x) or any(frozen[k].dtype!=v.dtype or not np.array_equal(frozen[k],v) for k,v in x.items()):
                raise PermissionError(f'Frozen protein exposure predictions differ: {key}')
    return roles,models,deploy,validation
