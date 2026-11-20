"""Authenticate and seal R12's source-only and unlabeled-target vote inputs."""
import json
from pathlib import Path

import numpy as np

from experiments.common.io import write_json
from experiments.mutagenicity.r10_原生图适配基线与增量_20261121.src import structural, structural_observation as so
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run, roc_transfer, thresholds
from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src import historical, observation
from . import vote_mixture


FIXED = ('B0', 'WL', 'Size', 'B', 'B2', 'PoolMajority', 'DirectP')
FITTED = ('Static', 'Global', 'Own', 'Other1', 'Other2', 'OwnLock')
METHODS = (*FIXED, *FITTED)
IDENTITY = structural.IDENTITY
FILES = ('executed-config.json','scope.json','roles.json','source-counts.json','rules.json','input-access.json')


def parents(cfg):
    return {k:cfg[k] for k in ('r10_fit_sha256','r10_obs_sha256','r11_unlabeled_sha256','r16_fit_sha256','r16_obs_sha256')}


def stage_files(root, stage, sha, predecessors=None):
    seal = roc_transfer.stage(root,stage,sha)
    if predecessors is not None and seal['predecessors'] != predecessors:
        raise PermissionError(f'Wrong {stage} predecessors in {root}')
    return seal


class Archive:
    def __init__(self, repo, cfg):
        p = parents(cfg)
        self.roots = (Path(cfg['r10']),repo/cfg['r11'],repo/cfg['r16'])
        r10,r11,r16 = self.roots
        sf = stage_files(r10,'FIT',p['r10_fit_sha256'])
        self.native = json.loads(prior_run.read_member(r10,sf['files'],'executed-config.json'))['plan']
        structural.validate_plan(self.native)
        self.plan = dict(self.native,seeds=[cfg['pilot_seed']])
        if ((self.native['settings'],self.native['domains'],self.native['seeds']) !=
                (cfg['settings'],cfg['domains'],[cfg['pilot_seed'],*cfg['reserve_seeds']])
                or sf['predecessors'] != dict(r7_final_digest=self.native['r7_final_digest'])):
            raise PermissionError('Structural models do not match the pilot grid')
        self.c = so.open_c(self.native)
        if json.loads(prior_run.read_member(r10,sf['files'],'historical-plan.json')) != self.c.plan:
            raise PermissionError('R10 FIT and R7 C plan differ')
        s = stage_files(r10,'OBS',p['r10_obs_sha256'],dict(fit_sha256=p['r10_fit_sha256']))
        u = stage_files(r11,'UNLABELED',p['r11_unlabeled_sha256'])
        f = stage_files(r16,'FIT',p['r16_fit_sha256'])
        b = stage_files(r16,'OBS',p['r16_obs_sha256'],dict(fit_sha256=p['r16_fit_sha256']))
        if (u['predecessors']['structural_fit_sha256']!=p['r10_fit_sha256']
                or u['predecessors']['structural_obs_sha256']!=p['r10_obs_sha256']
                or f['predecessors']['r10_fit_sha256']!=p['r10_fit_sha256']):
            raise PermissionError('Historical comparators do not share R10 FIT/OBS')
        self.files = dict(sf=sf['files'],s=s['files'],u=u['files'],f=f['files'],b=b['files'])
        self.roles = {stage:json.loads(self.member(stage,name)) for stage,name in (
            ('sf','fit-roles.json'),('s','roles.json'),('u','roles.json'),('b','prediction-roles.json'))}
        self.fit_access = json.loads(self.member('sf','fit-access.json'))
        self.stores, self.features = {}, {}

    def member(self, step, name):
        root = self.roots[0 if step in ('sf','s') else 1 if step=='u' else 2]
        return prior_run.read_member(root,self.files[step],name)

    def features_for(self, setting, ctx, ids, authorized, graph_sha):
        if not ids or len(set(ids)) != len(ids) or not set(ids) <= set(authorized):
            raise PermissionError('Topology request is outside approved V/A')
        if setting not in self.stores:
            self.stores[setting] = ctx['root'], historical.GraphStore(ctx)
        root,store = self.stores[setting]
        if root != ctx['root'] or store.graph_sha != graph_sha:
            raise PermissionError('R5 graph topology used for R12 differs from R10 FIT')
        cache = self.features.setdefault(setting,{})
        missing = [gid for gid in ids if gid not in cache]
        if missing:
            cache.update((gid,structural.features(graph,setting))
                         for gid,graph in zip(missing,store.take(missing),strict=True))
        return [cache[gid] for gid in ids]


def source_v(history, task, model, model_sha, ctx, manifest, group):
    key = inputs.key(**task)
    setting, source, seed = (task[k] for k in structural.GROUP)
    path = f'fits/{group}.json'
    fit_access = history.fit_access[group]
    v = manifest['ids']['validation']
    labels, v_sha = so.source_labels(ctx,manifest,source,seed,'validation')
    expected = json.loads(history.member('f',path))
    if ([model[k] for k in structural.GROUP] != [setting,source,seed] or
            [expected[k] for k in structural.GROUP] != [setting,source,seed] or
            expected['lineage']['model_sha256'] != model_sha or
            expected['lineage']['graph_sha256'] != fit_access['graph_container_sha256'] or
            expected['validation_ids'] != v or fit_access['validation_ids'] != v or
            expected['source_label_sha256']['V'] != v_sha or fit_access['v_label_sha256'] != v_sha):
        raise PermissionError('R10 and R16 source template inputs differ')
    original = inputs.npz_member(history.c.read('R7_INPUT',f'blocks/{key}/validation_input.npz'),
                                 ['validation_ids','validation_z0'])
    np.testing.assert_array_equal(original['validation_ids'],v)
    points = history.features_for(setting,ctx,v,v,fit_access['graph_container_sha256'])
    structural_margins = structural.predict(model,points)
    v_margins = np.column_stack((inputs.margin(original['validation_z0']),
                                  structural_margins['WL_selected'],structural_margins['Size']))
    np.testing.assert_array_equal(expected['validation_margins'],v_margins)
    vote = vote_mixture.encode(*(v_margins.T > 0.))
    return vote_mixture.source_counts(vote,labels), dict(v_ids=v,v_label_sha256=v_sha,
        structure_sha256=model_sha,r16_fit_member_sha256=history.files['f'][path],
        source_head_sha256=expected['source_label_sha256']['source_head'])


def target_a(history, task, model, ctx, role, selected, u):
    key = inputs.key(**task)
    path = f'blocks/{key}'
    original = inputs.npz_member(history.c.read('R7_INPUT',f'{path}/fit_input.npz'),
                                 ['adapt_ids','adapt_z0'])
    observed = inputs.npz_member(history.c.read('R7_OBS',f'{path}/tau-{history.c.tau:.3f}-observed.npz'),
                                 ['adapt_logits_B0','w_B0'])
    a = role['ids']['adapt']
    np.testing.assert_array_equal(original['adapt_ids'],a)
    np.testing.assert_allclose(inputs.margin(original['adapt_z0']),inputs.margin(observed['adapt_logits_B0']),
                               rtol=0.,atol=1e-12)
    if np.any(observed['w_B0'] != 0):
        raise PermissionError('R7 B0 has a nonzero target update')
    graph_sha = history.fit_access[role['group']]['graph_container_sha256']
    points = history.features_for(task['setting'],ctx,selected,a,graph_sha)
    z = structural.predict(model,points)
    position = {gid:i for i,gid in enumerate(a)}
    b0 = inputs.margin(observed['adapt_logits_B0'])[[position[i] for i in selected]]
    np.testing.assert_array_equal(u['a_margin'],b0)
    np.testing.assert_array_equal(u['a_size_margin'],z['Size'])
    return vote_mixture.encode(b0>0.,z['WL_selected']>0.,z['Size']>0.)


def target_h(history, task, h, u, s, b, s_role, b_role):
    key = inputs.key(**task)
    rel = f'predictions/{key}.npz'
    if (b_role['r10_prediction_sha256']!=history.files['s'][rel] or b_role['ids']!=h):
        raise PermissionError('R16 H predictions do not match immutable R10 H')
    so.verify_predictions(s,s_role)
    np.testing.assert_array_equal(s['ids'],h)
    np.testing.assert_array_equal(b['ids'],h)
    prior_run.verified_predictions(u,h,('B0','DirectP','PoolMajority'),('B0','DirectP','PoolMajority'))
    originals = ('B0','WL_selected','Size')
    for name in originals:
        np.testing.assert_array_equal(s[f'action_{name}'],b[f'decision_{name}'])
        np.testing.assert_array_equal(s[f'logits_{name}'],b[f'logits_{name}'])
    np.testing.assert_allclose(u['h_margin'],inputs.margin(s['logits_B0']),rtol=0.,atol=1e-12)
    np.testing.assert_array_equal(u['B0'],s['action_B0'])
    for name in ('B','B2'):
        a = b[f'decision_{name}']
        if a.dtype!=np.int64 or a.shape!=(len(h),):
            raise PermissionError(f'Malformed historical R16 {name} decision')
        np.testing.assert_array_equal(a,inputs.margin(b[f'logits_{name}'])>0.)
    return dict(B0=s['action_B0'],WL=s['action_WL_selected'],Size=s['action_Size'],
                B=b['decision_B'],B2=b['decision_B2'],PoolMajority=u['PoolMajority'].astype(np.int64),
                DirectP=u['DirectP'].astype(np.int64))


def members(keys):
    return [*FILES,*prior_run.inventory(keys,'predictions')]


def freeze(out, repo, cfg, history):
    plan = history.plan
    seed = cfg['pilot_seed']
    (out/'predictions').mkdir()
    roles, sources, fitted = {},{},{}
    for setting, source, _, tasks in observation.group_tasks(plan,[seed]):
        group = so.group_key(setting,source,seed)
        path = f'fits/{group}.json'
        model_sha = history.files['sf'][path]
        model = json.loads(history.member('sf',path))
        bound = {}
        manifests = [historical.context(history.c,t,repo) for t in tasks]
        budget = min(len(m['ids']['adapt']) for m,_,_ in manifests)
        structural.check_roles([m['ids'] for m,_,_ in manifests])
        for task,(manifest,manifest_sha,ctx) in zip(tasks,manifests,strict=True):
            key = inputs.key(**task)
            r = {step:row[key] for step,row in history.roles.items()}
            h = manifest['ids']['holdout']
            a = r['u']['selected_a_ids']
            if (any(any(row[k]!=v for k,v in task.items()) or row['manifest_sha256']!=manifest_sha
                    for row in r.values()) or any(r[step]['holdout_ids']!=h for step in ('sf','s','u'))
                    or any(r[step]['group']!=group for step in ('s','u','b')) or r['u']['size_fit_sha256']!=model_sha
                    or r['s']['fit_sha256']!=model_sha or r['u']['r10_h_member_sha256']!=history.files['s'][f'predictions/{key}.npz']
                    or a!=thresholds.subset_ids(setting,manifest['ids']['adapt'],budget)
                    or history.fit_access[group]['source_ids']!=manifest['ids']['source']
                    or history.fit_access[group]['class_mapping']!=ctx['label_meta']['mapping']):
                raise PermissionError(f'R11, R10 and R16 task identities or cohort roles differ: {key}')
            role = dict(**task,group=group,manifest_sha256=manifest_sha,holdout_ids=h,selected_a_ids=a,
                        other_targets=sorted(t['target'] for t in tasks if t!=task),ids=manifest['ids'])
            if role['other_targets']!=r['u']['other_targets']:
                raise PermissionError('Historical same-source swap ordering differs')
            roles[key] = role
            upath, rel = f'deployment/{key}.npz',f'predictions/{key}.npz'
            u = inputs.npz_member(history.member('u',upath))
            s = inputs.npz_member(history.member('s',rel))
            b = inputs.npz_member(history.member('b',rel),['ids',*[f'{stem}_{arm}' for stem in
                ('logits','decision') for arm in ('B0','WL_selected','Size','B','B2')]])
            a_votes = target_a(history,task,model,ctx,role,a,u)
            h_fixed = target_h(history,task,h,u,s,b,r['s'],r['b'])
            bound[task['target']] = key,a_votes,h_fixed
            role['evidence'] = dict(r11_sha256=history.files['u'][upath],r10_sha256=history.files['s'][rel],
                                    r16_sha256=history.files['b'][rel],model_sha256=model_sha)
        source_manifest,_,source_ctx = manifests[0]
        counts,receipt = source_v(history,tasks[0],model,model_sha,source_ctx,source_manifest,group)
        sources[group] = dict(**receipt,counts=counts.tolist(),size_C=model['size']['selected_C'],
                              wl_C=model['wl']['selected_C'])
        for target,(key,a_votes,_) in bound.items():
            a_counts = np.bincount(a_votes,minlength=8).reshape(2,4)
            fitted[key] = vote_mixture.rules(counts,a_counts)
        for target,(key,a_votes,base) in bound.items():
            bins = vote_mixture.encode(base['B0'],base['WL'],base['Size'])
            other = [fitted[bound[t][0]] for t in roles[key]['other_targets']]
            rules = vote_mixture.predict(fitted[key],other,bins)
            prior_run.save_predictions(out,f'predictions/{key}.npz',dict(
                ids=np.asarray(roles[key]['holdout_ids']),a_ids=np.asarray(roles[key]['selected_a_ids']),
                a_bins=a_votes,h_bins=bins,**base,**rules))
        print(f'R12 votes {group} V={sum(counts.ravel())} A per target={budget}',flush=True)
    keys = prior_run.registered_keys(plan)
    if set(roles)!=set(keys) or len(keys)!=36:
        raise PermissionError('Registered R12 36-block grid is incomplete')
    scope = dict(settings=plan['settings'],seed=seed,blocks=len(roles),source_groups=len(sources),
                 template_label_role='source_validation',adaptation_role='R11 selected unlabeled A',
                 heldout_role='H scored only after OBS',source_training_updates=0,
                 target_label_roles_used_in_freeze=[],r7_archive=plan['r7_archive'],r7_final_digest=plan['r7_final_digest'])
    write_json(out/'scope.json',scope)
    write_json(out/'roles.json',roles)
    write_json(out/'source-counts.json',sources)
    write_json(out/'rules.json',fitted)
    write_json(out/'input-access.json',history.c.digests)
    return prior_run.seal(out,'OBS',members(keys),parents(cfg))
