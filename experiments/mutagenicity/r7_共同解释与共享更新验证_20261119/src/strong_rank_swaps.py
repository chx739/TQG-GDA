"""Freeze and score equal-positive swaps of already sealed R7 ranking actions."""
from collections import defaultdict
import csv
from fractions import Fraction as F
import json

import numpy as np

from experiments.common.io import digest, write_json
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.pilot import write_table
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run, roc_score, roc_transfer
from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src import historical
from . import head_crossover as cross, head_crossover_scoring as old

PARENTS = ('crossover_obs_sha256','centered_obs_sha256','votes_obs_sha256')
MEMBERS = ('executed-config.json','blocks.csv','graphs.csv')
LATE_BASES = ('Static','B2','DirectP','PoolMajority')
BINS = ('count_0_to_e_1_anchor_count','count_1_to_e_0_anchor_count',
        'count_0_to_e_1_anchor_e','count_1_to_e_0_anchor_e')
IDENTITY = (*cross.IDENTITY,'reference','own','complete','valid_head','anchor')
REFS = ('Own','Other1','Other2')
EFFECTS = ('OwnMMinusBase','OwnMMinusOtherMean','OwnLMinusOtherMean')


def link(plan):
    return {k:plan[k] for k in PARENTS}


def check_plan(plan,cfg):
    if (plan['primary_bases']!={'mutagenicity_edge':'WL','mutagenicity_node':'WL','proteins_edge':'Size'}
            or cfg['settings']!=list(plan['primary_bases']) or cfg['domains']!=list(range(4))
            or [cfg['phase1_seed'],*cfg['reserve_seeds']]!=list(range(3405,3410))):
        raise PermissionError('Strong-rank swap grid or primary comparators changed')


def load_supplement(root,sha,keys):
    seal = roc_transfer.stage(root,'OBS',sha)
    roles = json.loads(prior_run.read_member(root,seal['files'],'roles.json'))
    if set(roles)!=set(keys):
        raise PermissionError('Supplementary prediction grid differs from the original seed')
    return roles,seal['files']


def load_actions(repo,here,plan,cfg,c):
    check_plan(plan,cfg)
    roles,_,pred = old.observed(here/plan['crossover'],cfg,c,plan['crossover_obs_sha256'])
    seed = cfg['phase1_seed']
    keys = [k for k,r in roles.items() if r['seed']==seed]
    roots = (repo/plan['votes'],here/plan['centered'])
    supp = [load_supplement(root,plan[f'{tag}_obs_sha256'],keys)
            for root,tag in zip(roots,('votes','centered'),strict=True)]
    bases = {}
    for key,role in roles.items():
        primary = plan['primary_bases'][role['setting']]
        base = {m:pred[key][m] for m in (primary,'SourceV')}
        if role['seed']==seed:
            centered_name = f'{primary}_Centered_A{role["target"]}'
            fields = (('B0','WL','Size',*LATE_BASES),('WL','Size',centered_name))
            for root,(registered,checksums),names in zip(roots,supp,fields,strict=True):
                r = registered[key]
                if any(r[k]!=role[k] for k in (*cross.IDENTITY,'manifest_sha256','holdout_ids')):
                    raise PermissionError(f'Supplementary H identity differs: {key}')
                raw = prior_run.read_member(root,checksums,f'predictions/{key}.npz')
                data = inputs.npz_member(raw,['ids',*names])
                np.testing.assert_array_equal(data.pop('ids'),role['holdout_ids'])
                for m,a in data.items():
                    if a.shape!=(len(role['holdout_ids']),) or a.dtype!=np.int64 or not np.isin(a,[0,1]).all():
                        raise PermissionError(f'Supplementary action is not binary: {key}/{m}')
                    if m in ('B0','WL','Size'):
                        np.testing.assert_array_equal(a,pred[key][m])
                    else:
                        base['Centered' if m==centered_name else m] = a
        bases[key] = base
    return roles,pred,bases


def membership(e, c, b):
    change = e!=c
    remaining = change & (b==c)
    absorbed = change & (b==e)
    masks = tuple(group & (c==before) for group in (remaining,absorbed) for before in (0,1))
    counts = tuple(int(q.sum()) for q in masks)
    if sum(counts)!=int(change.sum()) or counts[0]+counts[2]!=counts[1]+counts[3]:
        raise ArithmeticError('E vs C partition or equal-positive invariant fails')
    return masks,counts


def observations(roles,pred,bases,cfg):
    blocks,graphs = [],[]
    for key,r in sorted(roles.items()):
        ids = r['holdout_ids']
        own = r['target']
        donors = sorted(set(cfg['domains'])-{r['source']})
        local = []
        for ref in donors:
            name = cross.head_name('E',ref)
            e = pred[key][name]
            c = pred[key][cross.head_name('E',ref,count=True)]
            for anchor,b in bases[key].items():
                common = {k:r[k] for k in cross.IDENTITY} | dict(reference=ref,own=int(ref==own),
                         complete=int(r['complete']),valid_head=int(r['head_records'][name]['valid']),anchor=anchor)
                _,counts = membership(e,c,b)
                block = dict(**common,n=len(ids),e_positive=int(e.sum()),n_swap=sum(counts),
                             **dict(zip(BINS,counts,strict=True)),m=min(counts[:2]))
                local.append(block)
                for i in np.flatnonzero(e!=c):
                    graphs.append(dict(**common,graph_id=ids[i],e=int(e[i]),count=int(c[i]),anchor_action=int(b[i]),
                                       stratum='anchor_count' if b[i]==c[i] else 'anchor_e'))
        budgets = {anchor:min(r['m'] for r in local if r['anchor']==anchor) for anchor in bases[key]}
        blocks.extend(dict(**r,L=budgets[r['anchor']]) for r in local)
    by_block = {tuple(r[k] for k in IDENTITY):r for r in blocks}
    for row in graphs:
        block = by_block[tuple(row[k] for k in IDENTITY)]
        available = block[BINS[row['count']]] if row['stratum']=='anchor_count' else 0
        for budget in ('m','L'):
            row[f'select_p_{budget}'] = str(F(block[budget],available) if available else 0)
    return blocks,graphs


def freeze(out,roles,pred,bases,cfg,plan):
    blocks,graphs = observations(roles,pred,bases,cfg)
    write_table(out/'blocks.csv',blocks)
    write_table(out/'graphs.csv',graphs)
    return prior_run.seal(out,'OBS',MEMBERS,link(plan))


def csv_rows(data):
    return list(csv.DictReader(data.decode('utf-8').splitlines()))


def check_records(frozen,expected):
    if ([{k:str(v) for k,v in r.items()} for r in expected]!=frozen):
        raise PermissionError('Frozen rows fail exact replay against crossover actions')


def blind_check(out,sha,blind,plan):
    raw = prior_run.verified_members(out,'OBS',sha,MEMBERS,link(plan))
    index = blind/'SHA256SUMS'
    if digest(index)!=plan['independent_index_sha256']:
        raise PermissionError('Wrong independent pre-score checksum index')
    files = {}
    for line in index.read_text().splitlines():
        h,name = line.split()
        if digest(blind/name)!=h:
            raise PermissionError(f'Independent blind member differs: {name}')
        files[name] = h
    strict_index = blind/'STRICT_SHA256SUMS'
    if digest(strict_index)!=plan['independent_strict_index_sha256']:
        raise PermissionError('Wrong independent strict-budget checksum index')
    for line in strict_index.read_text().splitlines():
        h,name = line.split()
        if digest(blind/name)!=h:
            raise PermissionError(f'Independent strict-budget member differs: {name}')
        files[name] = h
    graphs = csv_rows(raw['graphs.csv'])
    bare = [{k:v for k,v in r.items() if k not in ('select_p_m','select_p_L')} for r in graphs]
    if bare!=csv_rows((blind/'graphs.csv').read_bytes()) or graphs!=csv_rows((blind/'strict-graphs.csv').read_bytes()):
        raise PermissionError('Independent graph membership or exact selection probability differs')
    independent = csv_rows((blind/'blocks.csv').read_bytes())
    official = csv_rows(raw['blocks.csv'])
    if [{k:v for k,v in r.items() if k not in ('m','L')} for r in official]!=independent:
        raise PermissionError('Independent block memberships differ')
    groups = defaultdict(list)
    for row in independent:
        key = tuple(row[k] for k in (*cross.IDENTITY,'anchor'))
        groups[key].append(min(int(row[BINS[0]]),int(row[BINS[1]])))
    strict = csv_rows((blind/'strict-blocks.csv').read_bytes())
    for row,ref in zip(official,strict,strict=True):
        key = tuple(row[k] for k in (*cross.IDENTITY,'anchor'))
        m = min(int(row[BINS[0]]),int(row[BINS[1]]))
        if (len(groups[key])!=3 or (int(row['m']),int(row['L']))!=(m,min(groups[key]))
                or any(ref[k]!=v for k,v in row.items())):
            raise PermissionError('Independent group counts imply different m or L')
    receipt = dict(obs_sha256=sha,independent_index_sha256=plan['independent_index_sha256'],
                   independent_strict_index_sha256=plan['independent_strict_index_sha256'],
                   graph_rows=len(graphs),graph_probability_entries=2*len(graphs),block_reference_anchor_rows=len(official),
                   budget_groups=len(groups),budget_pairs=len(official),all_match=True,files=files)
    write_json(out/'blind-verification.json',receipt)
    return receipt


def authorized_h(repo,c,roles):
    grants = {}
    for key,r in roles.items():
        task = {k:r[k] for k in cross.IDENTITY}
        m,sha,ctx = historical.context(c,task,repo)
        if sha!=r['manifest_sha256'] or m['ids']['holdout']!=r['holdout_ids'] or m['ids']['adapt']!=r['adapt_ids']:
            raise PermissionError(f'H role differs from the original crossover: {key}')
        cohort = r['setting'],r['target']
        grant = inputs.raw_grant(ctx,m,'holdout')
        if grants.setdefault(cohort,grant)!=grant:
            raise PermissionError('Shared H cohort differs across seeds or sources')
    if len(grants)!=12:
        raise PermissionError('Unexpected H label role count')
    labels,reads = {},{}
    with inputs.GraphLabelSession('holdout') as reader:
        for cohort,grant in grants.items():
            y = reader.load(grant,grant.ids)
            if y.shape!=(len(grant.ids),) or not np.isin(y,[0,1]).all():
                raise PermissionError('Invalid authorized binary H label vector')
            labels[cohort] = y
            reads[f'{cohort[0]}-M{cohort[1]}'] = list(grant.ids)
    return labels,reads


def swap_expectation(base,n_add,p_add,n_drop,p_drop,budget):
    if not 0<=budget<=min(n_add,n_drop):
        raise ValueError('Swap budget exceeds one eligible group')
    init = tuple(map(F,base))
    qa = F(p_add,n_add) if n_add else None
    qd = F(p_drop,n_drop) if n_drop else None
    dtp = budget*(qa-qd) if budget else F(0)
    good = budget*(qa+1-qd) if budget else F(0)
    bad = 2*budget-good
    tn,fp,fn,tp = init
    conf = tn+dtp,fp-dtp,fn-dtp,tp+dtp
    if conf[1]+conf[3]!=fp+tp or good-bad!=2*dtp or min(conf)<0:
        raise ArithmeticError('Expected equal-mass swap confusion does not reconcile')
    return conf,good,bad


def metric(conf):
    return ((conf[0]+conf[-1])/sum(conf),roc_score.f1(conf))


def score_blocks(roles,pred,bases,labels,blocks,cfg):
    attributed,swaps = [],[]
    for row in blocks:
        key = inputs.key(**{k:row[k] for k in cross.IDENTITY})
        role = roles[key]
        name = cross.head_name('E',row['reference'])
        b = bases[key][row['anchor']]
        e = pred[key][name]
        c = pred[key][cross.head_name('E',row['reference'],count=True)]
        y = labels[role['setting'],role['target']]
        masks,counts = membership(e,c,b)
        ns = dict(zip(BINS,counts,strict=True))
        common = {k:row[k] for k in (*IDENTITY,'n')}
        foreign = sorted(set(cfg['domains'])-{row['source'],row['target']})
        common['head'] = 'Own' if row['own'] else f'Other{1+foreign.index(row["reference"])}'
        for group,mask in zip(BINS,masks,strict=True):
            good = int(np.count_nonzero(mask & (e==y)))
            n = ns[group]
            attributed.append(dict(**common,group=group,flips=n,corrected=good,harmed=n-good,net=2*good-n))
        p_add,p_drop = (int(np.count_nonzero(mask & (y==1))) for mask in masks[:2])
        original = tuple(map(F,roc_score.confusion(b,y)))
        acc0,f1_0 = metric(original)
        n_add,n_drop = counts[:2]
        for kind in ('m','L'):
            budget = row[kind]
            conf,good,bad = swap_expectation(original,n_add,p_add,n_drop,p_drop,budget)
            a,f = metric(conf)
            swaps.append(dict(**common,kind=kind,budget=budget,n_add=n_add,n_drop=n_drop,p_add=p_add,p_drop=p_drop,
                              add_positive_rate='' if not n_add else str(F(p_add,n_add)),
                              drop_positive_rate='' if not n_drop else str(F(p_drop,n_drop)),
                              base_positive=str(original[1]+original[3]),base_accuracy_exact=str(acc0),
                              base_macro_f1_exact=str(f1_0),accuracy_exact=str(a),macro_f1_exact=str(f),
                              accuracy=float(a),macro_f1=float(f),delta_accuracy_exact=str(a-acc0),
                              delta_macro_f1_exact=str(f-f1_0),corrected=str(good),harmed=str(bad),net=str(good-bad),
                              **dict(zip(('tn','fp','fn','tp'),map(str,conf),strict=True))))
    return attributed,swaps


def group(rows,*keys):
    result = defaultdict(list)
    for row in rows:
        result[tuple(row[k] for k in keys)].append(row)
    return result


def mean(values):
    xs = list(values)
    return sum(xs,F(0))/len(xs)


def directions(rows,fn):
    return {key:mean(fn(r) for r in items) for key,items in group(rows,'source','target').items()}


def effect_row(ident,swap):
    own,other1,other2 = (swap[m] for m in REFS)
    own_m,own_L = (own[k] for k in ('m','L'))
    other_m = [r['m'] for r in (other1,other2)]
    other_L = [r['L'] for r in (other1,other2)]
    if len({r['budget'] for r in [own_L,*other_L]})!=1:
        raise ArithmeticError('Within-block comparison uses mixed L budgets')
    def signed_gain(field):
        return (F(own_m[f'delta_{field}_exact']),
                F(own_m[f'{field}_exact'])-mean(F(r[f'{field}_exact']) for r in other_m),
                F(own_L[f'{field}_exact'])-mean(F(r[f'{field}_exact']) for r in other_L))
    return [dict(**ident,effect=m,n=own_m['n'],m=own_m['budget'],L=own_L['budget'],
                 accuracy_exact=str(a),accuracy_pp=float(100*a),macro_f1_exact=str(f),macro_f1_pp=float(100*f))
            for m,a,f in zip(EFFECTS,signed_gain('accuracy'),signed_gain('macro_f1'),strict=True)]


def summaries(attributed,swaps,plan,seed):
    out = defaultdict(list)
    for setting,primary in plan['primary_bases'].items():
        for scope in ('complete','seed3405'):
            selected = lambda r: (r['complete'] if scope=='complete' else r['seed']==seed)
            aa = [r for r in attributed if r['setting']==setting and selected(r)]
            ss = [r for r in swaps if r['setting']==setting and selected(r)]
            anchors = (primary,'SourceV') if scope=='complete' else (primary,'SourceV',*LATE_BASES,'Centered')
            for anchor in anchors:
                common = dict(setting=setting,scope=scope,anchor=anchor)
                at = [r for r in aa if r['anchor']==anchor]
                sw = [r for r in ss if r['anchor']==anchor]
                for (head,b),items in group(at,'head','group').items():
                    d = directions(items,lambda r:F(r['net'],r['n']))
                    sums = {key:sum(r[key] for r in items) for key in ('n','flips','corrected','harmed','net')}
                    out['attribution-summary'].append(dict(**common,head=head,group=b,blocks=len(items),
                                                          directions=len(d),**sums,
                                                          delta_accuracy_exact=str(mean(d.values())),
                                                          delta_accuracy_pp=float(100*mean(d.values()))))
                for (head,kind),items in group(sw,'head','kind').items():
                    metrics = {m:mean(directions(items,lambda r:F(r[f'{m}_exact'])).values())
                               for m in ('base_accuracy','base_macro_f1','accuracy','macro_f1','delta_accuracy','delta_macro_f1')}
                    sums = {key:str(sum(F(r[key]) for r in items)) for key in ('corrected','harmed','net','budget')}
                    out['swap-summary'].append(dict(**common,head=head,kind=kind,blocks=len(items),
                        directions=len(group(items,'source','target')),graph_visits=sum(r['n'] for r in items),
                        positive_budget_blocks=sum(r['budget']>0 for r in items),**sums,
                        **{m:float(100*v) for m,v in metrics.items()},
                        delta_accuracy_exact=str(metrics['delta_accuracy'])))
                effects = []
                for (s,t,seed_id),items in group(sw,'source','target','seed').items():
                    refs = {head:{r['kind']:r for r in runs} for (head,),runs in group(items,'head').items()}
                    if any(set(refs[m])!={'m','L'} for m in REFS):
                        raise PermissionError('Missing one donor/budget arm')
                    effects.extend(effect_row(dict(**common,source=s,target=t,seed=seed_id),refs))
                out['block-contrasts'].extend(effects)
                for (effect,),items in group(effects,'effect').items():
                    ac = directions(items,lambda r:F(r['accuracy_exact']))
                    mf = directions(items,lambda r:F(r['macro_f1_exact']))
                    for (s,t),a in ac.items():
                        here = [r for r in items if (r['source'],r['target'])==(s,t)]
                        out['direction-contrasts'].append(dict(**common,effect=effect,source=s,target=t,
                            seeds=len(here),positive_L_blocks=sum(r['L']>0 for r in here),
                            accuracy_exact=str(a),accuracy_pp=float(100*a),macro_f1_pp=float(100*mf[s,t])))
                    omissions = {}
                    for role,i in (('source',0),('target',1)):
                        candidates = []
                        for domain in range(4):
                            kept = [key for key in ac if key[i]!=domain]
                            val = mean(ac[k] for k in kept)
                            candidates.append(val)
                            out['leave-domain'].append(dict(**common,effect=effect,omit_role=role,omit_domain=domain,
                                directions=len(kept),accuracy_exact=str(val),accuracy_pp=float(100*val),
                                macro_f1_pp=float(100*mean(mf[k] for k in kept))))
                        omissions[f'min_leave_{role}_out_pp'] = float(100*min(candidates))
                    a = mean(ac.values())
                    out['contrast-summary'].append(dict(**common,effect=effect,directions=len(ac),blocks=len(items),
                        positive_m_blocks=sum(r['m']>0 for r in items),positive_L_blocks=sum(r['L']>0 for r in items),
                        accuracy_exact=str(a),accuracy_pp=float(100*a),macro_f1_pp=float(100*mean(mf.values())),
                        wins=sum(v>0 for v in ac.values()),ties=sum(v==0 for v in ac.values()),
                        losses=sum(v<0 for v in ac.values()),**omissions))
    return out


def score(out,stage,repo,c,loaded,cfg,plan,sha,execution):
    raw = prior_run.verified_members(out,'OBS',sha,MEMBERS,link(plan))
    roles,pred,bases = loaded
    blocks,graphs = observations(*loaded,cfg)
    check_records(csv_rows(raw['blocks.csv']),blocks)
    check_records(csv_rows(raw['graphs.csv']),graphs)
    receipt = (out/'blind-verification.json').read_bytes()
    observed = json.loads(receipt)
    if (observed['obs_sha256']!=sha or observed['all_match'] is not True
            or observed['independent_index_sha256']!=plan['independent_index_sha256']
            or observed['independent_strict_index_sha256']!=plan['independent_strict_index_sha256']
            or observed['block_reference_anchor_rows']!=len(blocks) or observed['graph_rows']!=len(graphs)):
        raise PermissionError('The current OBS has not passed its independent comparison')
    labels,reads = authorized_h(repo,c,roles)
    at,sw = score_blocks(roles,pred,bases,labels,blocks,cfg)
    tables = dict(summaries(at,sw,plan,cfg['phase1_seed']))
    tables.update({'attribution-blocks':at,'swap-blocks':sw})
    for name,rows in tables.items():
        write_table(stage/f'{name}.csv',rows)
    write_json(stage/'h-label-access.json',dict(obs_sha256=sha,actual_reads=reads,adaptation_labels_read=False))
    write_json(stage/'score-execution.json',execution)
    (stage/'OBS.json').write_bytes((out/'OBS.json').read_bytes())
    (stage/'blind-verification.json').write_bytes(receipt)
    files = ['OBS.json','h-label-access.json','score-execution.json','blind-verification.json',
             *[f'{m}.csv' for m in tables]]
    return prior_run.seal(stage,'FINAL',files,dict(obs_sha256=sha))
