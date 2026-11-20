"""Authenticate continuous historic margins and freeze all rank policies before H scoring."""
import json
from fractions import Fraction

import numpy as np

from experiments.common.io import write_json
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.pilot import write_table
from experiments.mutagenicity.r10_原生图适配基线与增量_20261121.src import structural, structural_observation as so
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run
from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src import historical, observation
from . import conditional_rank as rank, conditional_votes as previous, reverse_vote_observation as reverse, vote_mixture

ID = previous.IDENTITY
FIXED = (*previous.FIXED,'Static','Centered')
RANKS = ('OwnRank','SourceRank','SignMix','Other1','Other2')
METHODS = ('Anchor',*RANKS,*FIXED)
META = ('executed-config.json','roles.json','source-rules.json','source-candidates.csv',
        'action-bounds.csv','action-bounds-summary.csv')


def plan(cfg,old):
    expected = (list(structural.SETTINGS),[0,1,2,3],3405)
    keys = ('settings','domains','pilot_seed')
    if (tuple(cfg[k] for k in keys)!=expected or tuple(old[k] for k in keys)!=expected
            or cfg['minimum_accuracy_fraction']!=[1,200] or cfg['minimum_positive_directions']!=7):
        raise PermissionError('Conditional-rank grid or registered gate differs')
    return dict(settings=cfg['settings'],domains=cfg['domains'],seeds=[cfg['pilot_seed']])


def parents(cfg):
    return {k:cfg[k] for k in ('historical_obs_sha256','centered_obs_sha256')}


def members(keys,groups):
    return [*META,*prior_run.inventory(keys,'predictions'),*prior_run.inventory(groups,'v-inputs')]


class Archive:
    def __init__(self,repo,here,cfg,old_cfg):
        self.repo = repo
        self.h = previous.Archive(repo,old_cfg)
        self.plan = plan(cfg,old_cfg)
        keys = prior_run.registered_keys(self.plan)
        self.prior = prior_run.verified_members(here/cfg['historical_attempt'],'OBS',cfg['historical_obs_sha256'],
                    previous.members(keys),previous.parents(old_cfg))
        if json.loads(self.prior['executed-config.json'])['plan']!=old_cfg:
            raise PermissionError('Historical R12 execution differs from the registered plan')
        self.roles = json.loads(self.prior['roles.json'])
        self.v = json.loads(self.prior['source-counts.json'])
        if set(self.roles)!=set(keys) or set(self.v)!={r['group'] for r in self.roles.values()} or len(self.v)!=12:
            raise PermissionError('Historical R12 lacks the 36 directions or 12 V groups')
        self.centered = prior_run.verified_members(repo/cfg['centered_attempt'],'OBS',cfg['centered_obs_sha256'],
                        ['executed-config.json','roles.json','weight-differences.json',
                         *prior_run.inventory(keys,'predictions')],dict(anchor_obs_sha256=cfg['centered_parent_obs_sha256']))
        self.centered_roles = json.loads(self.centered['roles.json'])
        if set(self.centered_roles)!=set(keys):
            raise PermissionError('Historical Centered has a different grid')

    def context(self,task):
        key = inputs.key(**task)
        r = self.roles[key]
        m,sha,ctx = historical.context(self.h.c,task,self.repo)
        if (any(r[k]!=v for k,v in task.items()) or r['ids']!=m['ids']
                or r['manifest_sha256']!=sha or r['holdout_ids']!=m['ids']['holdout']
                or not set(r['selected_a_ids'])<=set(m['ids']['adapt'])):
            raise PermissionError('Conditional-rank cohort differs from the original V/A/H roles')
        return key,r,m,ctx

    def v_scores(self,key,role,m,ctx):
        group = role['group']
        saved = self.v[group]
        name = f'fits/{group}.json'
        fitted = json.loads(self.h.member('f',name))
        raw = inputs.npz_member(self.h.c.read('R7_INPUT',f'blocks/{key}/validation_input.npz'),
                                ['validation_ids','validation_z0'])
        y_sha = saved['v_label_sha256']
        v = m['ids']['validation']
        history = self.h.fit_access[group]
        values = np.asarray(fitted['validation_margins'],dtype=np.float64)
        if (any(fitted[k]!=role[k] for k in structural.GROUP)
                or saved['v_ids']!=v or fitted['validation_ids']!=v or history['validation_ids']!=v
                or history['class_mapping']!=ctx['label_meta']['mapping']
                or history['v_label_sha256']!=y_sha or fitted['source_label_sha256']['V']!=y_sha
                or saved['source_head_sha256']!=fitted['source_label_sha256']['source_head']
                or saved['r16_fit_member_sha256']!=self.h.files['f'][name]
                or saved['structure_sha256']!=self.h.files['sf'][name]
                or fitted['lineage']['model_sha256']!=saved['structure_sha256']
                or fitted['lineage']['graph_sha256']!=history['graph_container_sha256']
                or values.shape!=(len(v),3) or not np.isfinite(values).all()):
            raise PermissionError('Historical V source, score, or supervision identity differs')
        np.testing.assert_array_equal(raw['validation_ids'],v)
        np.testing.assert_array_equal(inputs.margin(raw['validation_z0']),values[:,0])
        return values

    def target_scores(self,key,role):
        path = f'predictions/{key}.npz'
        u_path = f'deployment/{key}.npz'
        a_id,h_id = role['selected_a_ids'],role['holdout_ids']
        previous_x,a_z,h_z = reverse.historical_votes(self.prior[path],role)
        np.testing.assert_array_equal(h_z,2*previous_x['WL']+previous_x['Size'])
        a = inputs.npz_member(self.h.member('u',u_path),['a_margin','a_size_margin','ids','h_margin'])
        arms = ('B0','WL_selected','Size','B','B2')
        h = inputs.npz_member(self.h.member('b',path),['ids',*[f'logits_{m}' for m in arms]])
        ar,hr,c = self.h.roles['u'][key],self.h.roles['b'][key],self.centered_roles[key]
        sources = (('r11_sha256','u',u_path),('r16_sha256','b',path),('r10_sha256','s',path))
        if (any(any(row[k]!=role[k] for k in (*ID,'manifest_sha256')) for row in (ar,hr,c))
                or any(role['evidence'][ev]!=self.h.files[stage][p] for ev,stage,p in sources)
                or c['R10_H_prediction_sha256']!=role['evidence']['r10_sha256']
                or hr['r10_prediction_sha256']!=role['evidence']['r10_sha256']
                or ar['r10_h_member_sha256']!=role['evidence']['r10_sha256']
                or ar['size_fit_sha256']!=role['evidence']['model_sha256']
                or ar['selected_a_ids']!=a_id or ar['holdout_ids']!=h_id or hr['ids']!=h_id or c['holdout_ids']!=h_id):
            raise PermissionError('R7, R11 or R16 A/H identity differs from R12')
        np.testing.assert_array_equal(a['ids'],h_id)
        np.testing.assert_array_equal(h['ids'],h_id)
        for name in ('a_margin','a_size_margin'):
            if a[name].dtype!=np.float64 or a[name].shape!=(len(a_id),) or not np.isfinite(a[name]).all():
                raise PermissionError(f'Invalid authenticated A scores: {name}')
        np.testing.assert_array_equal(a['a_margin']>0.,(previous_x['a_bins']//2)%2)
        np.testing.assert_array_equal(a['a_size_margin']>0.,previous_x['a_bins']//4)
        raw_a = inputs.npz_member(self.h.c.read('R7_INPUT',f'blocks/{key}/fit_input.npz'),['adapt_ids','adapt_z0'])
        np.testing.assert_array_equal(raw_a['adapt_ids'],role['ids']['adapt'])
        index = {int(i):p for p,i in enumerate(raw_a['adapt_ids'])}
        np.testing.assert_allclose(a['a_margin'],inputs.margin(raw_a['adapt_z0'])[[index[i] for i in a_id]],rtol=0,atol=1e-12)
        for arm,orig in zip(arms,('B0','WL','Size','B','B2'),strict=True):
            x = h[f'logits_{arm}']
            if x.dtype!=np.float64 or x.shape!=(len(h_id),2) or not np.isfinite(x).all():
                raise PermissionError(f'Invalid authenticated H scores: {arm}')
            np.testing.assert_array_equal(inputs.margin(x)>0.,previous_x[orig])
        g = inputs.margin(h['logits_B0'])
        np.testing.assert_allclose(g,a['h_margin'],rtol=0,atol=1e-12)
        anchor = rank.anchor_name(role['setting'])
        name = f'{anchor}_Centered_A{role["target"]}'
        cen = inputs.npz_member(self.centered[path],['ids',anchor,name,f'margin_{name}'])
        np.testing.assert_array_equal(cen['ids'],h_id)
        np.testing.assert_array_equal(cen[anchor],previous_x[anchor])
        np.testing.assert_array_equal(cen[name],cen[f'margin_{name}']>0.)
        historic = inputs.npz_member(self.prior[path],['Static'])
        static = historic['Static']
        if (static.shape!=g.shape or static.dtype!=np.int64 or not np.isin(static,[0,1]).all()
                or cen[name].shape!=g.shape or cen[name].dtype!=np.int64 or not np.isfinite(cen[f'margin_{name}']).all()):
            raise PermissionError('Malformed historical Static or Centered actions')
        return dict(ids=h['ids'],a_ids=previous_x['a_ids'],g=g,z=h_z,a_g=a['a_margin'],a_z=a_z,
                    b2_margin=inputs.margin(h['logits_B2']),Static=static,Centered=cen[name],
                    **{arm:previous_x[arm] for arm in previous.FIXED})

    def group(self,setting,source,seed,tasks):
        group = so.group_key(setting,source,seed)
        rows = {}
        for task in tasks:
            key,role,m,ctx = self.context(task)
            if role['group']!=group or role['other_targets']!=sorted(t['target'] for t in tasks if t!=task):
                raise PermissionError('Historical same-source A swaps differ')
            rows[task['target']] = (key,role,m,ctx,self.target_scores(key,role))
        _,_,m0,c0,_ = next(iter(rows.values()))
        if any(m['ids']['validation']!=m0['ids']['validation'] or c['root']!=c0['root']
               for _,_,m,c,_ in rows.values()):
            raise PermissionError('Same-source V roles differ across the target tasks')
        return group,rows


def deployment(role,base,quotas,source,a,others):
    anchor = rank.anchor_name(role['setting'])
    anchors = rank.anchor_cells(role['setting'])
    held = base['g'],base['z']
    pred = {name:rank.predict(quotas,source,ref,held,anchors,sign_mix=(name=='SignMix'))
            for name,ref in zip(RANKS,(a,source,a,*others),strict=True)}
    cp_num,cp_den = rank.count_matched(base['b2_margin'],int(pred['OwnRank'].sum()))
    return dict(**base,Anchor=base[anchor],**pred,CountMatched_num=cp_num,CountMatched_den=cp_den,
                halves=rank.half_split(role['setting'],role['target'],base['ids']))


def candidate_rows(groups):
    for key,record in groups.items():
        for cell,rule in enumerate(record['quotas']):
            for k,correct in rule['candidates'] or [(None,None)]:
                yield dict(group=key,z=cell,n_V=rule['n'],k=k,correct=correct,selected=k==rule['k'])


def bounds(roles,predictions):
    rows = []
    for key,role in roles.items():
        x = predictions[key]
        for m in ('Anchor','SourceRank','SignMix','Other1','Other2',*FIXED):
            n = len(x['ids'])
            d = int(np.count_nonzero(x['OwnRank']!=x[m]))
            rows.append(dict(**{k:role[k] for k in ID},control=m,n=n,changed=d,
                             max_possible_accuracy_pp=100*d/n))
    summary = []
    for setting in dict.fromkeys(r['setting'] for r in rows):
        for control in dict.fromkeys(r['control'] for r in rows):
            match = [r for r in rows if r['setting']==setting and r['control']==control]
            bound = sum(Fraction(r['changed'],r['n']) for r in match)/len(match)
            summary.append(dict(setting=setting,control=control,changed=sum(r['changed'] for r in match),
                                max_possible_wins=sum(r['changed']>0 for r in match),
                                directions=len(match),
                                max_possible_accuracy_exact=str(bound),max_possible_accuracy_pp=float(100*bound)))
    return rows,summary


def run(out,archive,cfg,*,frozen=None):
    predictions,groups,v_inputs = {},{},{}
    for setting,source,seed,tasks in observation.group_tasks(archive.plan,[cfg['pilot_seed']]):
        group, rows = archive.group(setting,source,seed,tasks)
        key,role,m,ctx,_ = next(iter(rows.values()))
        margins = archive.v_scores(key,role,m,ctx)
        b0,wl,size = margins.T>0.
        g,z = margins[:,0],2*wl.astype(np.int64)+size
        v_ids = m['ids']['validation']
        receipt = archive.v[group]
        data = dict(ids=np.asarray(v_ids,dtype=np.int64),g=g,z=z,b0=b0.astype(np.int64))
        record = dict(v_ids=v_ids,v_label_sha256=receipt['v_label_sha256'],
                      r16_fit_member_sha256=receipt['r16_fit_member_sha256'])
        if frozen is None:
            y,sha = so.source_labels(ctx,m,source,seed,'validation')
            if y.shape!=(len(v_ids),) or not np.isin(y,[0,1]).all() or sha!=record['v_label_sha256']:
                raise PermissionError('Conditional-rank V supervision differs from historical R12')
            np.testing.assert_array_equal(vote_mixture.source_counts(vote_mixture.encode(b0,wl,size),y),
                                          receipt['counts'])
            quotas = rank.fit(g,z,y,rank.anchor_cells(setting))
        else:
            prior = frozen[group]
            if any(prior[k]!=v for k,v in record.items()):
                raise PermissionError('Frozen V role or hash differs from authorized sources')
            quotas = prior['quotas']
        groups[group] = dict(**record,quotas=quotas)
        v_inputs[group] = data
        pools = {t:(entry[-1]['a_g'],entry[-1]['a_z']) for t,entry in rows.items()}
        for target,(key,role,_,_,base) in rows.items():
            predictions[key] = deployment(role,base,quotas,(g,z),pools[target],
                                           [pools[t] for t in role['other_targets']])
        if frozen is None:
            print(f'R12 rank {group} V={len(g)} quotas={[(q["k"],q["n"]) for q in quotas]}',flush=True)
    if len(predictions)!=36 or set(predictions)!=set(archive.roles) or len(groups)!=12:
        raise PermissionError('Conditional-rank records do not cover 36 directions and 12 V sets')
    return predictions,groups,v_inputs


def freeze(out,archive,cfg):
    predictions,groups,v_inputs = run(out,archive,cfg)
    for name,values in (('predictions',predictions),('v-inputs',v_inputs)):
        (out/name).mkdir()
        for key,x in values.items():
            prior_run.save_predictions(out,f'{name}/{key}.npz',x)
    write_json(out/'roles.json',archive.roles)
    write_json(out/'source-rules.json',groups)
    write_table(out/'source-candidates.csv',list(candidate_rows(groups)))
    detail,summary = bounds(archive.roles,predictions)
    write_table(out/'action-bounds.csv',detail)
    write_table(out/'action-bounds-summary.csv',summary)
    return prior_run.seal(out,'OBS',members(predictions,groups),parents(cfg))


def observed(out,archive,cfg,sha):
    raw = prior_run.verified_members(out,'OBS',sha,members(archive.roles,archive.v),parents(cfg))
    roles = json.loads(raw['roles.json'])
    groups = json.loads(raw['source-rules.json'])
    if roles!=archive.roles or set(groups)!=set(archive.v) or json.loads(raw['executed-config.json'])['plan']!=cfg:
        raise PermissionError('Wrong sealed 36-direction roles or protocol')
    predicted,groups_replay,v_inputs = run(out,archive,cfg,frozen=groups)
    if groups_replay!=groups:
        raise PermissionError('Wrong frozen V role records')
    for name,values in (('predictions',predicted),('v-inputs',v_inputs)):
        for key,x in values.items():
            seen = inputs.npz_member(raw[f'{name}/{key}.npz'])
            if set(seen)!=set(x) or any(seen[k].dtype!=v.dtype or not np.array_equal(seen[k],v) for k,v in x.items()):
                raise PermissionError(f'Conditional-rank replay differs: {name}/{key}')
    return roles,predicted
