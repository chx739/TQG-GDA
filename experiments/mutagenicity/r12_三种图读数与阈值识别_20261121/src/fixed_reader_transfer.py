"""Freeze the fixed-reader adaptation and replay all decisions before H access."""
from fractions import Fraction as F
import json

import numpy as np

from experiments.common.io import write_json
from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src.head_crossover import equal_count
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run
from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src import observation
from . import conditional_votes, fixed_reader_inputs as readers, reader_ranges as ranges

ANCHOR = 'Anchor'
MAIN = 'SizeRange'
FIXED = conditional_votes.FIXED
ACTIONS = (ANCHOR,*FIXED,*ranges.RULES,'Other1','Other2','EqualCount')
META = ('executed-config.json','roles.json','source-tables.json','rules.json','action-bounds.json')
READOUT = ('ids','b0_margin','u','l_score','t_score','l','t','x')
ID = conditional_votes.IDENTITY


def members(keys,groups):
    return [*META,*prior_run.inventory(keys,'predictions'),*prior_run.inventory(keys,'adapt-readouts'),
            *prior_run.inventory(keys,'held-readouts'),*prior_run.inventory(groups,'source-readouts')]


def actions(role,h,base,anchor,rules_by_target):
    own = rules_by_target[role['target']]
    projected = {m:ranges.margins(anchor,h['u'],own,m) for m in ranges.RULES}
    for arm,t in zip(('Other1','Other2'),role['other_targets'],strict=True):
        projected[arm] = ranges.margins(anchor,h['u'],rules_by_target[t],MAIN)
    bits = {name:(z>0).astype(np.int64) for name,z in projected.items()}
    equal,_ = equal_count(h['ids'],anchor,int(bits[MAIN].sum()))
    return dict(ids=h['ids'],**{ANCHOR:(anchor>0).astype(np.int64)},**base,**bits,EqualCount=equal,
                margin_Anchor=anchor,**{f'margin_{name}':z for name,z in projected.items()})


def bound_rows(roles,acts):
    summary,details = {},[]
    for setting in sorted({r['setting'] for r in roles.values()}):
        rows = [k for k,r in roles.items() if r['setting']==setting]
        bands = {}
        for m in (ANCHOR,'EqualCount'):
            fractions = []
            for key in rows:
                x = acts[key]
                n = len(x['ids'])
                changed = int(np.count_nonzero(x[MAIN]!=x[m]))
                fractions.append(F(changed,n))
                details.append(dict(**{k:roles[key][k] for k in ID},against=m,n=n,changed=changed,
                                    accuracy_upper_exact=str(fractions[-1])))
            high = sum(fractions)/len(fractions)
            bands[m] = dict(accuracy_upper_exact=str(high),accuracy_upper_pp=float(100*high),
                            max_positive_directions=sum(x>0 for x in fractions))
        summary[setting] = bands
    return dict(settings=summary,per_block=details)


def readouts(archive,info,source):
    _,role,ctx,a_margin,h_margin,_,_ = info
    graph_sha = archive.history.fit_access[role['group']]['graph_container_sha256']
    result = []
    for ids,margin,pool in ((role['selected_a_ids'],a_margin,'adapt'),(role['holdout_ids'],h_margin,'holdout')):
        profile = archive.graph_features(role['setting'],ctx,ids,role['ids'][pool],graph_sha)
        result.append(readers.encode(ids,margin,source['cuts'],source['probes'],*profile[:2]))
    return result


def freeze(out,cfg,historical_cfg,archive):
    for name in ('predictions','held-readouts','adapt-readouts','source-readouts'):
        (out/name).mkdir()
    roles,source_tables,fitted,predicted = {},{},{},{}
    for setting,source,seed,tasks in observation.group_tasks(archive.plan,[cfg['pilot_seed']]):
        group,first,infos = archive.group(setting,source,seed,tasks)
        v,edges,y,cuts,probes,evidence = archive.source(group,first,infos[tasks[0]['target']][0])
        table,low,hashed = ranges.source_tables(v['ids'],y,v['u'],v['x'],edges,setting,source)
        record = dict(**table,bins=ranges.templates(table),cuts=cuts.tolist(),probes=probes,
                      v_label_ids=v['ids'].tolist(),evidence=evidence)
        source_tables[group] = record
        prior_run.save_predictions(out,f'source-readouts/{group}.npz',dict(**v,edges=edges,
                                    size_low=low.astype(np.int64),hash_low=hashed.astype(np.int64)))
        stage = {}
        for t,info in infos.items():
            key,role,*_ = info
            a,h = readouts(archive,info,record)
            roles[key] = role
            rules = ranges.fit_rules(record['bins'],ranges.count_vector(a['u'],a['x']))
            fitted[key] = rules
            stage[t] = a,h,rules
        rules_by_target = {t:x[2] for t,x in stage.items()}
        for t,(key,role,_,_,_,base,anchor) in infos.items():
            a,h,_ = stage[t]
            predicted[key] = actions(role,h,base,anchor,rules_by_target)
            for folder,x in (('predictions',predicted[key]),('held-readouts',h),('adapt-readouts',a)):
                prior_run.save_predictions(out,f'{folder}/{key}.npz',x)
        print(f'R12 fixed reader {group} V={len(v["ids"])} cut={table["cut"]} '
              f'A sizes={[len(x[0]["ids"]) for x in stage.values()]}',flush=True)
    if len(roles)!=24 or len(source_tables)!=8:
        raise PermissionError('All 24 directions and eight source readers are required')
    for name,value in (('roles',roles),('source-tables',source_tables),('rules',fitted),
                       ('action-bounds',bound_rows(roles,predicted))):
        write_json(out/f'{name}.json',value)
    return prior_run.seal(out,'OBS',members(roles,source_tables),readers.parents(cfg,historical_cfg))


def validate_readout(x,ids,record):
    if (set(x)!=set(READOUT) or x['ids'].dtype!=np.int64 or x['ids'].tolist()!=ids
            or any(v.shape!=x['ids'].shape for v in x.values())):
        raise PermissionError('Frozen readout arrays do not match their role')
    for name in ('b0_margin','l_score','t_score'):
        if x[name].dtype!=np.float64 or not np.isfinite(x[name]).all():
            raise PermissionError('Malformed frozen continuous score')
    expected = dict(u=np.searchsorted(record['cuts'],x['b0_margin'],side='left')+1,
                    l=(x['l_score']>record['probes']['L']['threshold']).astype(np.int64),
                    t=(x['t_score']>record['probes']['T']['threshold']).astype(np.int64))
    expected['x'] = 2*expected['l']+expected['t']
    for name,v in expected.items():
        if x[name].dtype!=np.int64 or not np.array_equal(x[name],v):
            raise PermissionError(f'Wrong frozen ULT readout: {name}')


def equal_arrays(got,want):
    return set(got)==set(want) and all(got[k].dtype==v.dtype and np.array_equal(got[k],v) for k,v in want.items())


def validate_source(raw,archive,group,data):
    role = archive.roles[group]
    record = data[group]
    v = inputs.npz_member(raw[f'source-readouts/{group}.npz'])
    actual = {k:v[k] for k in READOUT}
    validate_readout(actual,role['validation_ids'],record)
    a = inputs.npz_member(archive.member(f'views/{group}-frozen.npz'))
    p = f'probes/{group}.json'
    if (json.loads(archive.member(p))!=record['probes'] or a['cuts'].tolist()!=record['cuts']
            or record['v_label_ids']!=role['validation_ids'] or record['bins']!=ranges.templates(record)):
        raise PermissionError('V template differs from the immutable source probes or count witnesses')
    for key in ('u','l','t'):
        np.testing.assert_array_equal(a[key],v[key])
    for key in ('l_score','t_score'):
        np.testing.assert_allclose(a[key],v[key],rtol=0,atol=1e-9)
    counts = np.asarray(record['counts'])
    if counts.shape!=(2,4,4) or counts.sum(axis=0).tolist()!=ranges.count_vector(v['u'],v['x']):
        raise PermissionError('V contingency counts do not cover the original source records')
    size = np.ones(len(v['ids']),dtype=np.int64) if record['cut'] is None else (v['edges']<=record['cut']).astype(np.int64)
    if record['cut']!=ranges.size_cut(v['edges']) or not np.array_equal(size,v['size_low']):
        raise PermissionError('V size grouping differs')
    for name,which in (('SizeRange','size_low'),('HashRange','hash_low')):
        g = np.asarray(record['groups'][name])
        m = [v[which]==1] if record['cut'] is None else [v[which]==1,v[which]==0]
        if (not np.array_equal(g.sum(axis=0),counts) or
                any(grouped.sum(axis=0).tolist()!=ranges.count_vector(v['u'][part],v['x'][part])
                    for grouped,part in zip(g,m,strict=True))):
            raise PermissionError('V strata no longer reconcile with the frozen group assignments')
    if (not np.array_equal(np.asarray(record['groups']['SizeRange']).sum(axis=-1),
                           np.asarray(record['groups']['HashRange']).sum(axis=-1))
            or record['evidence']['v_label_sha256']!=archive.history.fit_access[group]['v_label_sha256']):
        raise PermissionError('Hash grouping does not preserve the true group sizes or V label authority')


def observed(out,cfg,historical_cfg,archive,obs):
    _,keys = observation.observations_expected(archive.plan,[cfg['pilot_seed']])
    groups = archive.roles
    raw = prior_run.verified_members(out,'OBS',obs,members(keys,groups),readers.parents(cfg,historical_cfg))
    roles,sources,rules = (json.loads(raw[f'{name}.json']) for name in ('roles','source-tables','rules'))
    if set(roles)!=set(keys) or set(rules)!=set(keys) or set(sources)!=set(groups):
        raise PermissionError('Wrong sealed R12 grid')
    predicted,held = {},{}
    for setting,source,seed,tasks in observation.group_tasks(archive.plan,[cfg['pilot_seed']]):
        group,_,infos = archive.group(setting,source,seed,tasks)
        validate_source(raw,archive,group,sources)
        rules_by_target = {}
        for t,info in infos.items():
            key,role,*_ = info
            if role!=roles[key]:
                raise PermissionError('R12 task role does not replay')
            a,h = (inputs.npz_member(raw[f'{folder}/{key}.npz']) for folder in ('adapt-readouts','held-readouts'))
            validate_readout(a,role['selected_a_ids'],sources[group])
            validate_readout(h,role['holdout_ids'],sources[group])
            ea,eh = readouts(archive,info,sources[group])
            if not equal_arrays(a,ea) or not equal_arrays(h,eh):
                raise PermissionError('R12 frozen A/H readouts do not replay')
            estimate = ranges.fit_rules(sources[group]['bins'],ranges.count_vector(a['u'],a['x']))
            if estimate!=rules[key]:
                raise PermissionError('R12 fitted ranges or updates do not replay')
            rules_by_target[t],held[key] = estimate,h
        for t,(key,role,_,_,_,base,anchor) in infos.items():
            got = inputs.npz_member(raw[f'predictions/{key}.npz'])
            if not equal_arrays(got,actions(role,held[key],base,anchor,rules_by_target)):
                raise PermissionError(f'Frozen classifier does not replay: {key}')
            predicted[key] = got
    if json.loads(raw['action-bounds.json'])!=bound_rows(roles,predicted):
        raise PermissionError('R12 action-only bounds do not replay')
    return roles,predicted,held,sources,rules
