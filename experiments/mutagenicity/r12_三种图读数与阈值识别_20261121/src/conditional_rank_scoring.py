"""Score frozen rank rules, then run the two registered H diagnostics."""
from fractions import Fraction as F
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import numpy as np

from experiments.common.io import write_json
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.pilot import write_table
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run, roc_score
from . import conditional_rank as rank, conditional_rank_observation as experiment


DIAGNOSTICS = ('HalfRank','HalfFour','HalfEight')
METHODS = (*experiment.METHODS,'CountMatched','OtherMean',*DIAGNOSTICS)
REQUIRED = ('SourceRank','SignMix','OtherMean','CountMatched',*experiment.FIXED)
CONTROLS = ('Anchor',*REQUIRED,'Other1','Other2')
ERROR_METHODS = ('Anchor','OwnRank','SourceRank','SignMix','CountMatched',*DIAGNOSTICS)


def mean(values):
    xs = list(values)
    return sum(xs,F(0))/len(xs)


def ident(role):
    return {k:role[k] for k in experiment.ID}


def h_labels(archive,roles):
    grants = {}
    for role in roles.values():
        _,r,m,ctx = archive.context(ident(role))
        if r!=role:
            raise PermissionError('H role changed after replay')
        key = role['setting'],role['target']
        grant = inputs.raw_grant(ctx,m,'holdout')
        if grants.setdefault(key,grant)!=grant:
            raise PermissionError('The same physical H has different source grants')
    if len(grants)!=12:
        raise PermissionError('Expected 12 original target roles only')
    labels,receipt = {},{}
    with inputs.GraphLabelSession('holdout') as reader:
        for key,grant in grants.items():
            y = reader.load(grant,grant.ids)
            if y.shape!=(len(grant.ids),) or not np.isin(y,[0,1]).all():
                raise PermissionError('Expected binary H labels in authenticated role order')
            labels[key] = y
            receipt[f'{key[0]}-M{key[1]}'] = list(grant.ids)
    return labels,receipt


def expectation(num,den):
    return [F(int(n),int(d)) for n,d in zip(num,den,strict=True)]


def expected_confusion(probability,y):
    fp = sum((q for q,t in zip(probability,y,strict=True) if t==0),F(0))
    tp = sum(probability,F(0))-fp
    p = int(y.sum())
    return len(y)-p-fp,fp,p-tp,tp


def metrics(cells,n):
    tn,fp,fn,tp = cells
    return F(tn+tp,n),roc_score.f1(cells)


def correctness(arm,y):
    return [q if t else 1-q for q,t in zip(arm,y,strict=True)]


def pair_counts(left,right):
    d = [a-b for a,b in zip(left,right,strict=True)]
    return sum((v for v in d if v>0),F(0)),sum((-v for v in d if v<0),F(0))


def endpoint_rows(role,y,q):
    n = len(y)
    matrices = {m:expected_confusion(p,y) for m,p in q.items()}
    points = {m:metrics(c,n) for m,c in matrices.items()}
    points['OtherMean'] = tuple(mean(xs) for xs in zip(points['Other1'],points['Other2'],strict=True))
    rows = []
    for m in METHODS:
        a,f = points[m]
        cells = matrices.get(m)
        rows.append(dict(**ident(role),method=m,n=n,accuracy=float(a),micro_f1=float(a),macro_f1=float(f),
                         accuracy_exact=str(a),macro_f1_exact=str(f),**(
                             {k:str(x) for k,x in zip(('tn','fp','fn','tp'),cells,strict=True)} if cells else {})))
    return rows,points


def compare(role,y,q,points):
    flags = {m:correctness(p,y) for m,p in q.items()}
    pairs = [('OwnRank',m) for m in CONTROLS]
    pairs += [(m,'Anchor') for m in DIAGNOSTICS] + [('HalfRank',m) for m in ('HalfFour','HalfEight')]
    for arm,ref in pairs:
        if ref=='OtherMean':
            components = [pair_counts(flags[arm],flags[x]) for x in ('Other1','Other2')]
            good,bad = map(mean,zip(*components,strict=True))
        else:
            good,bad = pair_counts(flags[arm],flags[ref])
        d,f = [a-b for a,b in zip(points[arm],points[ref],strict=True)]
        if (good-bad)/len(y)!=d:
            raise ArithmeticError('Corrected minus harmed is not the exact Accuracy difference')
        yield dict(**ident(role),left=arm,right=ref,n=len(y),corrected=str(good),harmed=str(bad),
                   changed=str(good+bad),accuracy_pp=float(100*d),accuracy_exact=str(d),macro_f1_pp=float(100*f))


def errors(role,z,y,q):
    for arm in ERROR_METHODS:
        probability = q[arm]
        for cell in range(4):
            inside = np.flatnonzero(z==cell)
            v = y[inside]
            tn,fp,fn,tp = expected_confusion([probability[i] for i in inside],v)
            k,p = fp+tp,fn+tp
            count_gap = abs(p-k)
            mix = 2*min(fp,fn)
            if fp+fn!=count_gap+mix:
                raise ArithmeticError('Cell error partition differs from the confusion counts')
            yield dict(**ident(role),method=arm,z=cell,n=len(inside),K=str(k),P=str(p),
                       fp=str(fp),fn=str(fn),errors=str(fp+fn),
                       count_gap=str(count_gap),ranking_errors=str(mix),
                       expectation_confusion=arm=='CountMatched')


def sorting(role,g,z,y):
    hard = (g>0.).astype(np.int64)
    for scope,cells,n in (('same_WL_Size',z,4),('same_all_three',2*z+hard,8)):
        for method,values in (('continuous_B0',g),('hard_B0',hard)):
            row = rank.concordance(values,cells,y,n)
            denom = 2*row['pairs']
            a = F(row['credit_twice'],denom) if denom else None
            yield dict(**ident(role),scope=scope,method=method,**row,positive=int(y.sum()),negative=len(y)-int(y.sum()),
                       concordance=None if a is None else float(a),concordance_exact=None if a is None else str(a))


def block_score(role,x,y):
    q = {m:list(map(int,x[m])) for m in experiment.METHODS}
    supervised = rank.supervised(x['g'],x['z'],y,x['halves'],rank.anchor_cells(role['setting']))
    q.update((m,values.tolist()) for m,values in supervised.items())
    q['CountMatched'] = expectation(x['CountMatched_num'],x['CountMatched_den'])
    if sum(q['CountMatched'])!=int(x['OwnRank'].sum()):
        raise ArithmeticError('CountMatched must assign the same number of positives')
    rows,points = endpoint_rows(role,y,q)
    return (rows,list(compare(role,y,q,points)),list(errors(role,x['z'],y,q)),
            list(sorting(role,x['g'],x['z'],y)),supervised)


def gate(cfg,paired):
    settings = {}
    for setting in cfg['settings']:
        groups = {m:[F(r['accuracy_exact']) for r in paired
                    if r['setting']==setting and r['left']=='OwnRank' and r['right']==m] for m in CONTROLS}
        if any(len(xs)!=12 for xs in groups.values()):
            raise PermissionError('Need exactly twelve directions for each gate comparison')
        gaps = {m:mean(v) for m,v in groups.items()}
        count = groups['Anchor']
        positive,tied,negative = (sum(fn(x) for x in count) for fn in (
            lambda x:x>0,lambda x:x==0,lambda x:x<0))
        checks = dict(anchor_gain=gaps['Anchor']>=F(*cfg['minimum_accuracy_fraction']),
                      positive_directions=positive>=cfg['minimum_positive_directions'])
        checks.update({m:gaps[m]>0 for m in REQUIRED})
        c = groups['Centered']
        centered = dict(accuracy_pp=float(100*gaps['Centered']),accuracy_exact=str(gaps['Centered']),
                        wins=sum(x>0 for x in c),ties=sum(x==0 for x in c),losses=sum(x<0 for x in c))
        centered['passes_gain_and_wins'] = (gaps['Centered']>=F(*cfg['minimum_accuracy_fraction'])
                                           and centered['wins']>=cfg['minimum_positive_directions'])
        settings[setting] = dict(passed=all(checks.values()),checks=checks,best_fixed=min(experiment.FIXED,key=gaps.__getitem__),
                                wins=positive,ties=tied,losses=negative,
                                centered_sensitivity=centered,
                                accuracy_pp={m:float(100*v) for m,v in gaps.items()},
                                accuracy_exact={m:str(v) for m,v in gaps.items()})
    return dict(settings=settings,advance=settings['proteins_edge']['passed'] and any(
        settings[s]['passed'] for s in ('mutagenicity_edge','mutagenicity_node')))


def endpoint_summary(cfg,rows):
    for setting in cfg['settings']:
        here = [r for r in rows if r['setting']==setting]
        for method in METHODS:
            selected = [r for r in here if r['method']==method]
            a = mean(F(r['accuracy_exact']) for r in selected)
            b = mean(F(r['macro_f1_exact']) for r in selected)
            yield dict(setting=setting,method=method,directions=len(selected),
                       accuracy=float(a),micro_f1=float(a),macro_f1=float(b),accuracy_exact=str(a))


def pair_summary(cfg,rows):
    specs = list(dict.fromkeys((r['left'],r['right']) for r in rows))
    for setting in cfg['settings']:
        here = [r for r in rows if r['setting']==setting]
        for a,b in specs:
            selected = [r for r in here if r['left']==a and r['right']==b]
            dif = [F(r['accuracy_exact']) for r in selected]
            sums = {name:str(sum(F(r[name]) for r in selected)) for name in ('corrected','harmed','changed')}
            gap = mean(dif)
            yield dict(setting=setting,left=a,right=b,directions=len(selected),
                       accuracy_pp=float(100*gap),accuracy_exact=str(gap),
                       macro_f1_pp=sum(r['macro_f1_pp'] for r in selected)/len(selected),
                       wins=sum(x>0 for x in dif),ties=sum(x==0 for x in dif),losses=sum(x<0 for x in dif),
                       changed_directions=sum(F(r['changed'])>0 for r in selected),**sums)


def sort_summary(cfg,rows):
    for setting in cfg['settings']:
        here = [r for r in rows if r['setting']==setting]
        for scope in ('same_WL_Size','same_all_three'):
            for method in ('continuous_B0','hard_B0'):
                selected = [r for r in here if r['scope']==scope and r['method']==method]
                valid = [F(r['concordance_exact']) for r in selected if r['concordance_exact'] is not None]
                counts = {k:sum(r[k] for r in selected) for k in ('pairs','covered_0','covered_1','negative','positive')}
                yield dict(setting=setting,scope=scope,method=method,directions=len(selected),
                           defined_directions=len(valid),concordance=None if not valid else float(mean(valid)),
                           coverage_0=counts['covered_0']/counts['negative'] if counts['negative'] else None,
                           coverage_1=counts['covered_1']/counts['positive'] if counts['positive'] else None,**counts)


def evaluate(out,archive,cfg,obs,roles,x,execution):
    labels,access = h_labels(archive,roles)
    rows,pairs,by_cell,sorting = [],[],[],[]
    (out/'half-diagnostic').mkdir()
    for key,r in roles.items():
        y = labels[r['setting'],r['target']]
        e,p,c,s,a = block_score(r,x[key],y)
        rows.extend(e)
        pairs.extend(p)
        by_cell.extend(c)
        sorting.extend(s)
        prior_run.save_predictions(out,f'half-diagnostic/{key}.npz',dict(ids=x[key]['ids'],halves=x[key]['halves'],**a))
    tables = {'endpoints':rows,'paired-blocks':pairs,'cell-errors':by_cell,'ranking-directions':sorting,
              'summary':list(endpoint_summary(cfg,rows)),'paired-summary':list(pair_summary(cfg,pairs)),
              'ranking-summary':list(sort_summary(cfg,sorting))}
    for name,result in tables.items():
        write_table(out/f'{name}.csv',result)
    write_json(out/'gate.json',gate(cfg,pairs))
    write_json(out/'score-execution.json',execution)
    write_json(out/'h-label-access.json',dict(obs_sha256=obs,actual_reads=access,a_labels_read=False))
    files = ['OBS.json','gate.json','score-execution.json','h-label-access.json',*[f'{n}.csv' for n in tables],
             *prior_run.inventory(roles,'half-diagnostic')]
    return prior_run.seal(out,'FINAL',files,dict(obs_sha256=obs))


def score(out,archive,cfg,obs,*,execution):
    final = out/'score'
    if final.exists():
        raise FileExistsError('This OBS already has a published score')
    roles,x = experiment.observed(out,archive,cfg,obs)
    with TemporaryDirectory(prefix=f'.{out.name}-score-',dir=out.parent) as tmp:
        stage = Path(tmp)/'score'
        stage.mkdir()
        copyfile(out/'OBS.json',stage/'OBS.json')
        if prior_run.digest(stage/'OBS.json')!=obs:
            raise PermissionError('OBS identity changed before scoring')
        digest = evaluate(stage,archive,cfg,obs,roles,x,execution)
        stage.rename(final)
    return digest
