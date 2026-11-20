"""Authenticate the reused structural classifiers, GNN scores, and exact cohort roles."""
from pathlib import Path
import json

import numpy as np
from scipy.special import expit

from experiments.common.io import digest
from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import wl
from experiments.mutagenicity.r10_原生图适配基线与增量_20261121.src import structural, structural_observation as so
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs as r11, prior_run as storage
from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src import historical
from experiments.mutagenicity.r14_节点类型与分类更新_20261121.src.experiment import match_roles


class Historical:
    def __init__(self, cfg, repo):
        self.cfg, self.repo = cfg, repo
        self.root = Path(cfg['structural_archive'])
        self.plan = json.loads((self.root/'executed-config.json').read_bytes())['plan']
        if self.plan['seeds'] != [cfg['phase1_seed'], *cfg['reserve_seeds']]:
            raise PermissionError('R16 seed grid differs from original structural fits')
        structural.validate_plan(self.plan)
        self.fit, self.roles, self.access = so.verify_fit(self.root, self.plan, cfg['structural_fit_sha256'])
        self.archive = so.open_c(self.plan)
        if json.loads(self.fit['historical-plan.json']) != self.archive.plan:
            raise PermissionError('Original R10 FIT is from a different R7 C')
        self.r14 = repo / cfg['r14_attempt']
        if digest(self.r14/'FIT.json') != cfg['r14_fit_sha256']:
            raise PermissionError('R14 FIT identity differs')
        self.r14_fit = json.loads((self.r14/'FIT.json').read_bytes())
        if self.r14_fit['stage']!='FIT' or self.r14_fit['predecessors'] != {
            'r7_final_digest':self.plan['r7_final_digest']
        }:
            raise PermissionError('R14 reference cohorts come from a different R7 C')
        self.r14_roles = json.loads(storage.read_member(self.r14, self.r14_fit['files'], 'fit-roles.json'))
        self.stores, self.feature_cache = {}, {}

    def group(self, setting, source, seed):
        group = so.group_key(setting, source, seed)
        data = self.fit[f'fits/{group}.json']
        model = json.loads(data)
        if [model[k] for k in ('setting','source','seed')] != [setting,source,seed]:
            raise PermissionError('Structural FIT group identity differs')
        rows, manifests, contexts = [], [], []
        for t in self.plan['domains']:
            if t==source:
                continue
            task = dict(setting=setting, source=source,target=t,seed=seed)
            key = r11.key(**task)
            role = self.roles[key]
            mf, ctx = so.bind(self.archive,self.repo,task,role,self.access[group])
            rows.append(dict(**task,ids=mf['ids'],manifest_sha256=role['manifest_sha256']))
            manifests.append(mf)
            contexts.append(ctx)
        s, v, pools = match_roles(setting, rows)
        record = dict(setting=setting, source=source, seed=seed, source_fit=s, validation=v, pools=pools, tasks=rows)
        if seed==self.cfg['phase1_seed'] and self.r14_roles.get(group)!=record:
            raise PermissionError('R16 source, validation, and reference subset IDs differ from sealed R14')
        access = self.access[group]
        lineage = dict(model_sha256=r11.sha(data), graph_sha256=access['graph_container_sha256'],
                       wl_C=model['wl']['selected_C'], size_C=model['size']['selected_C'])
        return record, manifests[0], contexts[0], model, lineage

    def features(self, setting, ctx, ids, sha):
        if setting not in self.stores:
            self.stores[setting] = historical.GraphStore(ctx)
        store = self.stores[setting]
        if store.graph_sha != sha:
            raise PermissionError('R5 R16 topology differs from R10 FIT')
        cache = self.feature_cache.setdefault(setting,{})
        missing = [i for i in ids if i not in cache]
        if missing:
            cache.update((i,structural.features(g,setting)) for i,g in zip(missing,store.take(missing),strict=True))
        return [cache[i] for i in ids]

    def source_scores(self, rows, s, v, mf, ctx):
        key = r11.key(**{k:rows[0][k] for k in structural.IDENTITY})
        reader = self.archive
        fit = r11.npz_member(reader.read('R7_INPUT',f'blocks/{key}/fit_input.npz'),
            ['source_h','source_y','source_ids','source_z0','source_head_weight','source_head_bias'])
        val = r11.npz_member(reader.read('R7_INPUT',f'blocks/{key}/validation_input.npz'),
                             ['validation_h','validation_ids','validation_z0'])
        np.testing.assert_array_equal(fit['source_ids'],s)
        np.testing.assert_array_equal(val['validation_ids'],v)
        source, seed = rows[0]['source'],rows[0]['seed']
        y_s, sha_s = so.source_labels(ctx,mf,source,seed,'fit')
        y_v, sha_v = so.source_labels(ctx,mf,source,seed,'validation')
        access = self.access[so.group_key(rows[0]['setting'],source,seed)]
        if sha_s!=access['s_label_sha256'] or sha_v!=access['v_label_sha256']:
            raise PermissionError('R16 S/V labels differ from those used by R10')
        np.testing.assert_array_equal(y_s,fit['source_y'])
        original, ckpt = historical.checkpoint(mf,ctx)
        w, b = original.head.weight.double().detach().numpy(),original.head.bias.double().detach().numpy()
        np.testing.assert_array_equal(w,fit['source_head_weight'])
        np.testing.assert_array_equal(b,fit['source_head_bias'])
        for x in (fit,val):
            prefix = 'source' if x is fit else 'validation'
            np.testing.assert_allclose(x[f'{prefix}_h']@w.T+b,x[f'{prefix}_z0'],atol=1e-12,rtol=0.)
        for task in rows[1:]:
            tkey = r11.key(**{k:task[k] for k in structural.IDENTITY})
            other = r11.npz_member(reader.read('R7_INPUT',f'blocks/{tkey}/validation_input.npz'),
                                   ['validation_ids','validation_z0'])
            np.testing.assert_array_equal(other['validation_ids'],v)
            np.testing.assert_array_equal(other['validation_z0'],val['validation_z0'])
        return margin(val['validation_z0']), y_s, y_v, dict(S=sha_s,V=sha_v,source_head=ckpt['sha256'])

    def a_scores(self, task, chosen):
        key = r11.key(**{k:task[k] for k in structural.IDENTITY})
        fit = r11.npz_member(self.archive.read('R7_INPUT',f'blocks/{key}/fit_input.npz'),['adapt_ids','adapt_z0'])
        observed = r11.npz_member(self.archive.read('R7_OBS',f'blocks/{key}/tau-{self.archive.tau:.3f}-observed.npz'),
                                  ['adapt_logits_B0'])
        ids = task['ids']['adapt']
        np.testing.assert_array_equal(fit['adapt_ids'],ids)
        g = margin(fit['adapt_z0'])
        np.testing.assert_allclose(g,margin(observed['adapt_logits_B0']),atol=1e-12,rtol=0.)
        np.testing.assert_array_equal(g>0.,margin(observed['adapt_logits_B0'])>0.)
        positions = {i:n for n,i in enumerate(ids)}
        return g[[positions[i] for i in chosen]]

    def held_role(self):
        relative = 'OBS.json'
        if digest(self.root/relative)!=self.cfg['structural_obs_sha256']:
            raise PermissionError('Reused R10 H predictions differ')
        obs = json.loads((self.root/relative).read_bytes())
        if obs['stage']!='OBS' or obs['predecessors']!={'fit_sha256':self.cfg['structural_fit_sha256']}:
            raise PermissionError('R10 H predictions bind a different FIT')
        self.obs = obs['files']
        self.held_roles = json.loads(storage.read_member(self.root,self.obs,'roles.json'))

    def held_scores(self, task, features, model):
        key = r11.key(**{k:task[k] for k in structural.IDENTITY})
        role = self.held_roles[key]
        for field, expected in self.roles[key].items():
            if role[field]!=expected:
                raise PermissionError('R10 historical H role differs from FIT')
        if role['fit_sha256'] != r11.sha(self.fit[f'fits/{role["group"]}.json']):
            raise PermissionError('R10 historical H references another classifier')
        rel = f'predictions/{key}.npz'
        names = ('B0','SourceCal','WL_selected','Size')
        x = r11.npz_member(storage.read_member(self.root,self.obs,rel), ['ids',*[
            f'{stem}_{name}' for stem in ('logits','action') for name in names]])
        ids = task['ids']['holdout']
        np.testing.assert_array_equal(x['ids'],ids)
        np.testing.assert_array_equal(x['logits_B0'],self.archive.b0({k:task[k] for k in structural.IDENTITY},ids))
        computed = structural.predict(model,features)
        for name in ('WL_selected','Size'):
            np.testing.assert_array_equal(margin(x[f'logits_{name}']),computed[name])
        for name in names:
            np.testing.assert_array_equal(x[f'action_{name}'],x[f'logits_{name}'].argmax(1))
        return {m:x[f'logits_{m}'] for m in names}, self.obs[rel]


def margin(z):
    return z[:,1]-z[:,0]


def verify_structural(model, source, validation, y_s, y_v, plan):
    s_wl, s_size = zip(*source)
    v_wl, v_size = zip(*validation)
    vocab = wl.vocabulary(s_wl,range(len(s_wl)))
    if [[depth,color.hex()] for depth,color in vocab] != model['wl']['vocabulary']:
        raise PermissionError('Rebuilt S-only WL vocabulary differs from R10')
    x_s = wl.sparse_rows(s_wl,range(len(s_wl)),vocab)
    x_v = wl.sparse_rows(v_wl,range(len(v_wl)),vocab)
    sizes = np.asarray(s_size)
    average, scale = sizes.mean(0), sizes.std(0)
    scale[np.ptp(sizes,axis=0)==0.] = 1.
    np.testing.assert_array_equal(average,model['size']['mean'])
    np.testing.assert_array_equal(scale,model['size']['scale'])
    maximum = {}
    for kind, a, b in [('wl',x_s,x_v),('size',(sizes-average)/scale,(np.asarray(v_size)-average)/scale)]:
        branch = model[kind]
        if set(branch['candidates'])!={str(c) for c in plan['C']} or [r['C'] for r in branch['validation']]!=plan['C']:
            raise PermissionError('Structural candidate C set differs')
        gradient_errors = []
        for entry in branch['validation']:
            strength = entry['C']
            fitted = branch['candidates'][str(strength)]
            z = structural.scores(b,fitted)
            v = structural.validation_metrics(strength,z,y_v)
            if v['correct']!=entry['correct'] or v['n']!=entry['n']:
                raise PermissionError('Restored structural V labels or score ordering differ')
            np.testing.assert_allclose(v['nll'],entry['nll'],atol=1e-12,rtol=0.)
            w = np.asarray(fitted['weights'])
            m = structural.scores(a,fitted)
            error = expit(m)-y_s
            residual = float(max(np.max(np.abs(a.T@error + w/strength)),abs(error.sum()))/len(y_s))
            loss = np.logaddexp(0.,(1-2*y_s)*m).mean() + (w@w)/(2*len(y_s)*strength)
            if residual > plan['solver']['stationarity_tolerance']+1e-12:
                raise ValueError('Original structural source solution is not stationary')
            np.testing.assert_allclose(loss,fitted['training_objective'],atol=1e-12,rtol=0.)
            gradient_errors.append(residual)
        if structural.choose(branch['validation']) != branch['selected_C']:
            raise PermissionError('Structural V candidate selection differs')
        maximum[kind] = max(gradient_errors)
    return maximum
