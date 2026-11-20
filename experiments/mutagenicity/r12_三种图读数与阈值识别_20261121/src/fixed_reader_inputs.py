"""Read authenticated R12 probes, and apply them at their original scale."""
import json

import numpy as np

from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src import composition
from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import wl
from experiments.mutagenicity.r10_原生图适配基线与增量_20261121.src import structural, structural_observation as so
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run, thresholds
from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src import historical, observation
from . import conditional_votes as previous, readouts


def parents(cfg,historical_cfg):
    return {'r12_source_final_sha256':cfg['source_final_sha256'],**previous.parents(historical_cfg)}


def encode(ids,margin,cuts,probes,nodes,topology):
    names = {(depth,bytes.fromhex(color)):i for i,(depth,color) in enumerate(probes['T']['vocabulary'])}
    l = composition.scores(np.stack([nodes[i] for i in ids]),probes['L'])
    t = composition.scores(wl.sparse_rows(topology,ids,names),probes['T'])
    u = np.searchsorted(cuts,margin,side='left').astype(np.int64)+1
    lb,tb = (value>probes[k]['threshold'] for k,value in (('L',l),('T',t)))
    return dict(ids=np.asarray(ids,dtype=np.int64),b0_margin=margin,u=u,l_score=l,t_score=t,
                l=lb.astype(np.int64),t=tb.astype(np.int64),x=2*lb.astype(np.int64)+tb.astype(np.int64))


class Archive:
    def __init__(self,repo,here,cfg,historical_cfg):
        self.repo,self.cfg = repo,cfg
        self.history = previous.Archive(repo,historical_cfg)
        self.plan = dict(self.history.plan, settings=cfg['settings'])
        if (cfg['settings']!=['mutagenicity_edge','proteins_edge'] or cfg['domains']!=self.plan['domains']
                or self.plan['seeds']!=[cfg['pilot_seed']]):
            raise PermissionError('Fixed source-reader grid differs from the registered 24 directions')
        self.root = here/cfg['source_attempt']
        blob = (self.root/'FINAL.json').read_bytes()
        seal = json.loads(blob)
        if inputs.sha(blob)!=cfg['source_final_sha256'] or seal['stage']!='R12_SOURCE_SCREEN_FINAL':
            raise PermissionError('Historical R12 source probe archive identity differs')
        self.files = seal['files']
        records = json.loads(self.member('source-roles.json'))
        self.roles = {so.group_key(r['setting'],r['source'],r['seed']):r for r in records}
        expected = {so.group_key(s,src,seed) for s,src,seed,_ in observation.group_tasks(self.plan,self.plan['seeds'])}
        if set(self.roles)!=expected or len(records)!=len(expected):
            raise PermissionError('Expected the eight original R12 readers')
        prior = json.loads(self.member('executed-config.json'))['plan']
        if prior['r7_final_digest']!=self.history.native['r7_final_digest'] or prior['seed']!=cfg['pilot_seed']:
            raise PermissionError('R12 readers were calibrated on a different R7 archive')
        self.receipts = json.loads(self.member('input-access.json'))['r5_sources']
        self.stores,self.cached = {},{}

    def member(self,name):
        return prior_run.read_member(self.root,self.files,name)

    def group(self,setting,source,seed,tasks):
        history = self.history
        group = so.group_key(setting,source,seed)
        role = self.roles[group]
        model_path = f'fits/{group}.json'
        structural_model = json.loads(history.member('sf',model_path))
        structural_sha = history.files['sf'][model_path]
        fit = history.fit_access[group]
        if (role['fit_ids']!=fit['source_ids'] or role['validation_ids']!=fit['validation_ids']
                or any(role[k]!=v for k,v in zip(('setting','source','seed'),(setting,source,seed),strict=True))
                or any(structural_model[k]!=role[k] for k in structural.GROUP)):
            raise PermissionError('R12 reader source roles differ from historical R10')
        receipt = self.receipts[setting]
        if (receipt['members']['inputs/graphs.pt']!=fit['graph_container_sha256']
                or receipt['source_label_reads'][f'M{source}-validation']!=role['validation_ids']):
            raise PermissionError('R12 source graph or V lineage differs')
        manifests = [historical.context(history.c,task,self.repo) for task in tasks]
        structural.check_roles([m['ids'] for m,_,_ in manifests])
        budget = min(len(m['ids']['adapt']) for m,_,_ in manifests)
        data = {}
        for task,(m,m_sha,ctx) in zip(tasks,manifests,strict=True):
            key = inputs.key(**task)
            fs,s,u,b = (history.roles[stage][key] for stage in ('sf','s','u','b'))
            other = sorted(t['target'] for t in tasks if t!=task)
            ids = m['ids']
            a,h = u['selected_a_ids'],ids['holdout']
            rel,upath = f'predictions/{key}.npz',f'deployment/{key}.npz'
            cp = role['b0_checkpoint']
            if (ids['source']!=role['fit_ids'] or ids['validation']!=role['validation_ids']
                    or fit['class_mapping']!=ctx['label_meta']['mapping']
                    or m['provenance']['r5_files'][cp['path']]!=cp['sha256']
                    or any(any(r[k]!=v for k,v in task.items()) or r['manifest_sha256']!=m_sha for r in (fs,s,u,b))
                    or any(r['holdout_ids']!=h for r in (fs,s,u))
                    or any(r['group']!=group for r in (s,u,b)) or s['fit_sha256']!=structural_sha
                    or u['size_fit_sha256']!=structural_sha or u['r10_h_member_sha256']!=history.files['s'][rel]
                    or a!=thresholds.subset_ids(setting,ids['adapt'],budget) or u['other_targets']!=other):
                raise PermissionError(f'R12, R10, R11 or R16 role is different for {key}')
            current = dict(**task,group=group,manifest_sha256=m_sha,ids=ids,holdout_ids=h,
                           selected_a_ids=a,other_targets=other,
                           evidence=dict(R10=history.files['s'][rel],R11=history.files['u'][upath],
                                         R16=history.files['b'][rel],R10_FIT=structural_sha))
            ar = inputs.npz_member(history.member('u',upath))
            hr = inputs.npz_member(history.member('s',rel))
            bc = inputs.npz_member(history.member('b',rel),['ids',*[f'{stem}_{arm}' for stem in ('logits','decision')
                                                             for arm in ('B0','WL_selected','Size','B','B2')]])
            previous.target_a(history,task,structural_model,ctx,current,a,ar)
            fixed = previous.target_h(history,task,h,ar,hr,bc,s,b)
            source_anchor = 'WL_selected' if self.cfg['anchor'][setting]=='WL' else 'Size'
            anchor = inputs.margin(hr[f'logits_{source_anchor}'])
            np.testing.assert_array_equal(anchor>0.,fixed[self.cfg['anchor'][setting]])
            data[task['target']] = (key,current,ctx,ar['a_margin'],inputs.margin(hr['logits_B0']),fixed,anchor)
        return group,manifests[0],data

    def graph_features(self,setting,ctx,ids,permitted,graph_sha):
        if not ids or len(set(ids))!=len(ids) or not set(ids)<=set(permitted):
            raise PermissionError('R12 probe request is outside the authenticated V/A/H role')
        if setting not in self.stores:
            self.stores[setting] = (ctx['root'],historical.GraphStore(ctx))
        root,store = self.stores[setting]
        if root!=ctx['root'] or store.graph_sha!=graph_sha:
            raise PermissionError('R12 reader input graph container changed')
        cache = self.cached.setdefault(setting,{})
        missing = [i for i in ids if i not in cache]
        graphs = store.take(missing) if missing else ()
        for gid,g in zip(missing,graphs,strict=True):
            raw = dict(x=g.x,edge_index=g.edge_index,edge_attr=g.edge_attr)
            nodes = readouts.node_frequency(raw,structural.SETTINGS[setting][0])
            words = wl.normalize(readouts.topology_counts(raw))
            edges = len({(min(a,b),max(a,b)) for a,b in g.edge_index.T.tolist() if a!=b})
            cache[gid] = (nodes,words,edges)
        return {i:cache[i][0] for i in ids},{i:cache[i][1] for i in ids},np.asarray([cache[i][2] for i in ids],dtype=np.int64)

    def source(self,group,first,key):
        role = self.roles[group]
        setting,src,seed = (role[k] for k in structural.GROUP)
        manifest,_,ctx = first
        model_path,view_path = f'probes/{group}.json',f'views/{group}-frozen.npz'
        probe = json.loads(self.member(model_path))
        old = inputs.npz_member(self.member(view_path))
        x = inputs.npz_member(self.history.c.read('R7_INPUT',f'blocks/{key}/validation_input.npz'),
                              ['validation_ids','validation_z0'])
        ids = role['validation_ids']
        margin = inputs.margin(x['validation_z0'])
        cuts = old['cuts']
        np.testing.assert_array_equal(x['validation_ids'],ids)
        np.testing.assert_array_equal(old['validation_ids'],ids)
        np.testing.assert_allclose(margin,old['margin'],rtol=0,atol=1e-9)
        np.testing.assert_array_equal(cuts,role['u_cutpoints'])
        features = self.graph_features(setting,ctx,ids,ids,self.history.fit_access[group]['graph_container_sha256'])
        data = encode(ids,margin,cuts,probe,*features[:2])
        for m in ('l_score','t_score'):
            np.testing.assert_allclose(data[m],old[m],rtol=0,atol=1e-9)
        for m in ('u','l','t'):
            np.testing.assert_array_equal(data[m],old[m])
        y,sha = so.source_labels(ctx,manifest,src,seed,'validation')
        label_name = f'source_labels/M{src}/seed{seed}_validation.pt'
        if (sha!=self.receipts[setting]['members'][label_name]
                or sha!=self.history.fit_access[group]['v_label_sha256']):
            raise PermissionError('Validation labels differ from the original source reader')
        evidence = dict(v_label_sha256=sha,probe_sha256=self.files[model_path],view_sha256=self.files[view_path],
                        l_error=float(np.max(abs(data['l_score']-old['l_score']))),
                        t_error=float(np.max(abs(data['t_score']-old['t_score']))))
        return data,features[2],y,cuts,probe,evidence
