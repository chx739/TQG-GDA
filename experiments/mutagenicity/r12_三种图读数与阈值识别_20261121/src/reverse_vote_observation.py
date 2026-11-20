"""Authenticate the historical vote pilot and freeze 36 reverse-check policies."""
from fractions import Fraction as F
import json

import numpy as np

from experiments.common.io import write_json
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.pilot import write_table
from experiments.mutagenicity.r10_原生图适配基线与增量_20261121.src import structural, structural_observation as so
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run
from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src import historical, observation
from . import conditional_votes as previous, reverse_votes as rules, vote_mixture

FILES = ('executed-config.json','roles.json','source-receipts.json','policies.json',
         'candidate-losses.csv','action-bounds.csv','action-bounds-summary.csv')
POLICIES = ('Anchor',*rules.TABLE_ARMS,'CountMatched')
BOUNDS = ('Anchor','Direct','PermMean','RowPerm','Other1','Other2')


def parents(cfg):
    return dict(historical_obs_sha256=cfg['historical_obs_sha256'])


def plan(cfg,historical_cfg):
    expected = (list(structural.SETTINGS),[0,1,2,3],3405)
    keys = ('settings','domains','pilot_seed')
    if (tuple(cfg[k] for k in keys)!=expected or tuple(historical_cfg[k] for k in keys)!=expected
            or cfg['neighbors']!=8 or cfg['minimum_accuracy_fraction']!=[1,200]
            or cfg['minimum_positive_directions']!=7):
        raise PermissionError('Reverse-vote grid or registered gate changed')
    return dict(settings=cfg['settings'],domains=cfg['domains'],seeds=[cfg['pilot_seed']])


def members(keys):
    return [*FILES,*prior_run.inventory(keys,'inputs')]


def historical_obs(root,cfg,old_cfg):
    p = plan(cfg,old_cfg)
    keys = prior_run.registered_keys(p)
    blobs = prior_run.verified_members(root,'OBS',cfg['historical_obs_sha256'],
                                      previous.members(keys),previous.parents(old_cfg))
    if json.loads(blobs['executed-config.json'])['plan']!=old_cfg:
        raise PermissionError('Published pilot execution differs from the original plan')
    roles = json.loads(blobs['roles.json'])
    sources = json.loads(blobs['source-counts.json'])
    if set(roles)!=set(keys) or set(sources)!={r['group'] for r in roles.values()} or len(sources)!=12:
        raise PermissionError('Historical pilot is missing a source or direction')
    return blobs,roles,sources


def historical_votes(blob,role):
    x = inputs.npz_member(blob,['ids','a_ids','a_bins','h_bins',*previous.FIXED])
    for archived,expected in (('ids','holdout_ids'),('a_ids','selected_a_ids')):
        np.testing.assert_array_equal(x[archived],role[expected])
    h = vote_mixture.encode(x['B0'],x['WL'],x['Size'])
    np.testing.assert_array_equal(x['h_bins'],h)
    a = x['a_bins']
    if (a.dtype!=np.int64 or a.shape!=x['a_ids'].shape or not np.isin(a,np.arange(8)).all()
            or any(x[m].shape!=h.shape or x[m].dtype!=np.int64 or not np.isin(x[m],[0,1]).all()
                   for m in previous.FIXED)):
        raise PermissionError('Invalid frozen original vote vectors')
    # The old index was 4*Size + 2*B0 + WL. Reverse uses 2*WL + Size.
    return x,2*(a%2)+a//4,rules.votes(x['WL'],x['Size'])


class FreezeInputs:
    def __init__(self,repo,old_cfg):
        self.repo = repo
        self.history = previous.Archive(repo,old_cfg)
        self.stores = {}

    def manifest(self,task,role):
        m,sha,ctx = historical.context(self.history.c,task,self.repo)
        if (any(role[k]!=v for k,v in task.items()) or role['ids']!=m['ids']
                or role['manifest_sha256']!=sha or role['holdout_ids']!=m['ids']['holdout']
                or not set(role['selected_a_ids'])<=set(m['ids']['adapt'])):
            raise PermissionError('Frozen pilot roles differ from authorized V/A/H')
        return m,ctx

    def histograms(self,setting,group,ctx,ids,authorized):
        if not ids or len(set(ids))!=len(ids) or not set(ids)<=set(authorized):
            raise PermissionError('Histogram request outside V or selected A')
        if setting not in self.stores:
            self.stores[setting] = (ctx['root'],historical.GraphStore(ctx),{})
        root,store,cache = self.stores[setting]
        if root!=ctx['root'] or store.graph_sha!=self.history.fit_access[group]['graph_container_sha256']:
            raise PermissionError('Graph records differ from frozen source fitting')
        width = structural.SETTINGS[setting][0]
        missing = [gid for gid in ids if gid not in cache]
        if missing:
            for gid,graph in zip(missing,store.take(missing),strict=True):
                cache[gid] = rules.edge_histogram(graph,width)
        return np.stack([cache[i] for i in ids])

    def source(self,group,manifest,ctx,receipt):
        setting,source,seed = (manifest[k] for k in structural.GROUP)
        fit_name = f'fits/{group}.json'
        data = json.loads(self.history.member('f',fit_name))
        sha = self.history.files['f'][fit_name]
        v = manifest['ids']['validation']
        lineage = self.history.fit_access[group]
        evidence = dict(r16_fit_member_sha256=sha,graph_container_sha256=lineage['graph_container_sha256'],
                        v_label_sha256=receipt['v_label_sha256'])
        margins = np.asarray(data['validation_margins'],dtype=np.float64)
        if (any(data[k]!=manifest[k] for k in structural.GROUP)
                or sha!=receipt['r16_fit_member_sha256']
                or receipt['structure_sha256']!=self.history.files['sf'][fit_name]
                or data['lineage']['model_sha256']!=receipt['structure_sha256']
                or data['lineage']['graph_sha256']!=evidence['graph_container_sha256']
                or data['validation_ids']!=v or receipt['v_ids']!=v or lineage['validation_ids']!=v
                or lineage['class_mapping']!=ctx['label_meta']['mapping']
                or data['source_label_sha256']['source_head']!=receipt['source_head_sha256']
                or data['source_label_sha256']['V']!=evidence['v_label_sha256']
                or lineage['v_label_sha256']!=evidence['v_label_sha256']
                or margins.shape!=(len(v),3) or not np.isfinite(margins).all()):
            raise PermissionError('Source V predictions differ from historical R12')
        y,label_sha = so.source_labels(ctx,manifest,source,seed,'validation')
        if label_sha!=evidence['v_label_sha256'] or y.shape!=(len(v),) or not np.isin(y,[0,1]).all():
            raise PermissionError('Source V labels differ from historical R12')
        b0,wl,size = margins.T>0
        np.testing.assert_array_equal(vote_mixture.source_counts(vote_mixture.encode(b0,wl,size),y),
                                      receipt['counts'])
        hist = self.histograms(setting,group,ctx,v,v)
        return hist,rules.votes(wl,size),y,dict(**evidence,v_ids=v)


def detail_rows(roles,fitted):
    for key,role in roles.items():
        record = fitted[key]
        identity = {k:role[k] for k in previous.IDENTITY}
        for method in rules.LOCAL_ARMS:
            selected = record['fits'][method]
            for number,loss in enumerate(selected['brier']):
                yield dict(**identity,method=method,table=number,brier_exact=loss,brier=float(F(loss)),
                           selected=int(number==selected['table']),second_table=selected['second_table'],
                           gap_exact=selected['gap_exact'],gap=float(F(selected['gap_exact'])))


def action_bounds(roles,fitted):
    for key,role in roles.items():
        record = fitted[key]['deployment']
        identity = {k:role[k] for k in previous.IDENTITY}
        h = record['h_counts']
        p = record['policies']
        for m in BOUNDS:
            changed = sum(n*int(a!=b) for n,a,b in zip(h,p['Reverse'],p[m],strict=True))
            yield dict(**identity,right=m,n=sum(h),changed=changed,max_accuracy_pp=100*changed/sum(h))


def summarize_bounds(rows,settings):
    for setting in settings:
        for arm in BOUNDS:
            match = [r for r in rows if r['setting']==setting and r['right']==arm]
            upper = sum(F(r['changed'],r['n']) for r in match)/len(match)
            yield dict(setting=setting,right=arm,directions=len(match),changed=sum(r['changed'] for r in match),
                       max_positive_directions=sum(r['changed']>0 for r in match),
                       max_accuracy_exact=str(upper),max_accuracy_pp=float(100*upper))


def freeze(out,repo,root,cfg,old_cfg):
    p = plan(cfg,old_cfg)
    obs,roles,sources = historical_obs(root,cfg,old_cfg)
    access = FreezeInputs(repo,old_cfg)
    fitted,receipts = {},{}
    (out/'inputs').mkdir()
    for setting,source,seed,tasks in observation.group_tasks(p,[cfg['pilot_seed']]):
        group = so.group_key(setting,source,seed)
        prepared = {}
        for task in tasks:
            key = inputs.key(**task)
            role = roles[key]
            if role['group']!=group or role['other_targets']!=sorted(t['target'] for t in tasks if t!=task):
                raise PermissionError(f'Same-source swap identity differs: {key}')
            manifest,ctx = access.manifest(task,role)
            arrays,a_z,h_z = historical_votes(obs[f'predictions/{key}.npz'],role)
            prepared[task['target']] = (key,role,manifest,ctx,arrays,a_z,h_z)
        _,_,m,ctx,*_ = next(iter(prepared.values()))
        if any(m['ids']['validation']!=z[2]['ids']['validation'] or ctx['root']!=z[3]['root']
               for z in prepared.values()):
            raise PermissionError(f'Same-source V membership differs: {group}')
        v_hist,v_z,y,receipts[group] = access.source(group,m,ctx,sources[group])
        anchor = rules.anchor_number(setting)
        for key,role,_,ctx,arrays,a_z,h_z in prepared.values():
            a_hist = access.histograms(setting,group,ctx,role['selected_a_ids'],role['ids']['adapt'])
            weights = rules.reverse_weights(v_hist,a_hist,a_z,cfg['neighbors'])
            fitted[key] = rules.fit(v_z,y,a_z,weights,anchor)
            numerators = np.array([[w.numerator for w in row] for row in weights],dtype=np.int64)
            denominators = np.array([[w.denominator for w in row] for row in weights],dtype=np.int64)
            prior_run.save_predictions(out,f'inputs/{key}.npz',dict(
                a_ids=arrays['a_ids'],h_ids=arrays['ids'],v_ids=np.asarray(receipts[group]['v_ids'],dtype=np.int64),
                a_z=a_z,h_z=h_z,v_z=v_z,weight_numerator=numerators,weight_denominator=denominators))
        for key,role,_,_,_,_,h_z in prepared.values():
            donors = [fitted[prepared[t][0]] for t in role['other_targets']]
            fitted[key]['deployment'] = rules.apply(fitted[key],donors,rules.tally(h_z),anchor)
        print(f'R12 reverse {group} V={len(v_z)} tables={[fitted[z[0]]["fits"]["Reverse"]["table"] for z in prepared.values()]}',
              flush=True)
    if set(fitted)!=set(roles) or len(roles)!=36:
        raise PermissionError('Incomplete 36-direction reverse check')
    write_json(out/'roles.json',roles)
    write_json(out/'source-receipts.json',receipts)
    write_json(out/'policies.json',fitted)
    write_table(out/'candidate-losses.csv',list(detail_rows(roles,fitted)))
    bounds = list(action_bounds(roles,fitted))
    write_table(out/'action-bounds.csv',bounds)
    write_table(out/'action-bounds-summary.csv',list(summarize_bounds(bounds,p['settings'])))
    return prior_run.seal(out,'OBS',members(prior_run.registered_keys(p)),parents(cfg))
