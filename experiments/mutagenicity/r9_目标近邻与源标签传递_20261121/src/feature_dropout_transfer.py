"""Rebuild sealed FLDA ridge learners and change only their evaluation graphs."""
from collections import defaultdict
from hashlib import sha256
from zipfile import ZipFile
import json

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from experiments.common.io import digest, write_json
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.pilot import write_table
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import (
    inputs, prior_run, roc_transfer, size_exposure as protein)
from . import reference_geometry as geom, residual_inputs as data, wl

ORIGINAL = ('Target','Static','Source','Perm')
PROTEIN = ('Curve','CurvePool')
META = ('executed-config.json','roles.json','fit-label-access.json','model-replay.csv','fitted.json')
CANDIDATES = (.01,.1,1.,10.)
PERMUTATION_SEED = 20261121


def parents(cfg):
    return {k:cfg[k] for k in ('geometry_obs_sha256','zip_sha256','freeze_sha256')}


def members(keys):
    return [*META,*prior_run.inventory(keys,'predictions'),*prior_run.inventory(keys,'models')]


def sample(pool,count,key):
    return sorted(pool,key=lambda i:sha256(f'flda1|{key}|{i}'.encode('utf-8')).digest())[:count]


def scope(reference,cfg):
    rows = {}
    for key,r in reference.roles.items():
        role = r['roles']
        m = min(len(role['C']),len(role['U']))
        target = sample(role['U'],m,key)
        source = sample(role['C'],m,key)
        supp = sorted(set(role['U'])-set(target))
        if not supp or set(supp)&(set(role['D'])|set(role['C'])|set(role['E'])|set(target)):
            raise PermissionError(f'Supplement overlaps the fitting or initial score roles: {key}')
        rows[key] = r | dict(target_reference=target,source_reference=source,supplement=supp,m=m)
    for setting,n in cfg['supplement_counts'].items():
        chosen = [r for r in rows.values() if r['setting']==setting]
        supp = [i for r in chosen for i in r['supplement']]
        old = {i for r in chosen for i in r['roles']['E']}
        if len(chosen)!=8 or len(supp)!=n or len(set(supp))!=n or set(supp)&old:
            raise PermissionError(f'Supplement count or initial E identity differs: {setting}')
    return rows


class HistoricalScreen:
    def __init__(self,repo,here,cfg,geom_cfg,reference):
        self.root = here/cfg['previous_screen']
        self.archive,features = geom.prepare(repo,reference)
        self.features = {group:{gid:item[0] for gid,item in rows.items()} for group,rows in features.items()}
        self.roles = scope(reference,cfg)
        meta = roc_transfer.stage(here/cfg['geometry'],'OBS',cfg['geometry_obs_sha256'])
        if meta['predecessors']!=dict(source_obs_sha256=geom_cfg['source_obs_sha256']):
            raise PermissionError('Geometry role snapshot has a different source OBS')
        for name,expected in (('roles',reference.roles),('source-topology',reference.topology),
                              ('source-provenance',reference.provenance)):
            if json.loads(prior_run.read_member(here/cfg['geometry'],meta['files'],f'{name}.json'))!=expected:
                raise PermissionError('Original geometry identities differ')
        self.package = self.root/'frozen-run.zip'
        manifest = self.root/'freeze.json'
        if digest(self.package)!=cfg['zip_sha256'] or digest(manifest)!=cfg['freeze_sha256']:
            raise PermissionError('Initial FLDA source/prediction bytes changed')
        registry = json.loads(manifest.read_bytes())
        self.initial = registry['rows']
        with ZipFile(self.package) as z:
            if (set(z.namelist())!={'probe.py',*(f'{k}.npz' for k in self.roles)}
                    or len(z.namelist())!=25 or set(self.initial)!=set(self.roles)
                    or inputs.sha(z.read('probe.py'))!=registry['source_program']):
                raise PermissionError('Initial FLDA archive inventory differs')

    def original_arrays(self,package,key):
        raw = package.read(f'{key}.npz')
        if inputs.sha(raw)!=self.initial[key]['prediction_sha']:
            raise PermissionError(f'Initial E margin artifact differs: {key}')
        x = inputs.npz_member(raw,['e',*ORIGINAL,'Count'])
        np.testing.assert_array_equal(x['e'],self.roles[key]['roles']['E'])
        if any(x[k].shape!=x['e'].shape or x[k].dtype!=np.float64 or not np.isfinite(x[k]).all()
               for k in (*ORIGINAL,'Count')):
            raise PermissionError('Initial E margin inventory is malformed')
        return x

    def nodes(self,setting,ids):
        return np.asarray([len(self.archive.by_setting[setting].graphs[i]['x']) for i in ids],dtype=np.int64)


def lex_hash(lex):
    return inputs.sha(json.dumps([[rad,word.hex()] for rad,word in lex],separators=(',',':')).encode('ascii'))


def dropout_ratios(x_d,features,lex,row):
    support = np.asarray(x_d.getnnz(axis=0))
    f_d = support/x_d.shape[0]

    def ratio(ids):
        x = wl.sparse_rows(features,ids,lex)
        return np.minimum(1.,(np.asarray(x.getnnz(axis=0))/row['m'])/f_d)

    target = ratio(row['target_reference'])
    perm = target.copy()
    strata = defaultdict(list)
    for (rad,_),j in lex.items():
        strata[rad,int(np.floor(np.log2(support[j])))].append(j)
    rng = np.random.default_rng(PERMUTATION_SEED)
    for js in strata.values():
        perm[js] = rng.permutation(target[js])
    return dict(Target=target,Static=np.ones(len(lex)),Source=ratio(row['source_reference']),Perm=perm)


def ridge(x,y,ratio,penalty,lam):
    inv = np.zeros_like(ratio)
    keep = ratio>0
    inv[keep] = 1/(lam+penalty[keep]*(1/ratio[keep]-1))
    weighted = x.multiply(inv).tocsr()
    gram = (weighted@x.T).toarray()
    gram.flat[::len(gram)+1] += 1
    sol = cho_solve(cho_factor(gram),np.c_[y,np.ones(len(y))])
    intercept = sol[:,0].sum()/sol[:,1].sum()
    coef = weighted.T@(sol[:,0]-intercept*sol[:,1])
    return np.asarray(coef,dtype=np.float64),float(intercept)


def fit(features,row,y_d,y_c):
    role = row['roles']
    lex = wl.vocabulary(features,role['D'])
    xd = wl.sparse_rows(features,role['D'],lex)
    xc = wl.sparse_rows(features,role['C'],lex)
    y_d,y_c = (2*np.asarray(y,dtype=np.int64)-1 for y in (y_d,y_c))
    ridges = dropout_ratios(xd,features,lex,row)
    penalty = np.asarray(xd.power(2).sum(axis=0)).ravel()
    params,coefs,intercepts,selection = {},{},{},{}
    for m,r in ridges.items():
        options = []
        for lam in CANDIDATES:
            w,b = ridge(xd,y_d,r,penalty,lam)
            z = xc@w+b
            right = int(np.count_nonzero((z>0)==(y_c>0)))
            mse = float(np.mean((z-y_c)**2))
            options.append(((-right,mse,-lam),w,b))
        best,w,b = min(options,key=lambda op:op[0])
        params[m],coefs[m],intercepts[m] = -best[2],w,b
        selection[m] = [dict(lam=-c,n_correct=-a,mse=bm) for (a,bm,c),_,_ in options]
    return lex,coefs,intercepts,dict(m=row['m'],n_features=len(lex),zero=int((ridges['Target']==0).sum()),
                                    params=params,selection=selection,vocabulary_sha256=lex_hash(lex))


def count_matched(ids,z,n_positive):
    result = np.zeros(len(ids),dtype=np.int64)
    result[np.lexsort((ids,-z))[:n_positive]] = 1
    return result


def predictions(screen,row,lex,coefs,intercepts,exposure):
    setting,source = (row[k] for k in ('setting','source'))
    features = screen.features[setting,source]
    ids = np.asarray(row['supplement'],dtype=np.int64)
    x = wl.sparse_rows(features,ids,lex)
    margins = {name:x@coefs[name]+intercepts[name] for name in ORIGINAL}
    actions = {m:(z>0).astype(np.int64) for m,z in margins.items()}
    actions['Count'] = count_matched(ids,margins['Static'],int(actions['Target'].sum()))
    if setting=='proteins_edge':
        margins['Curve'] = protein.predict(screen.nodes(setting,ids),exposure)-.5
        p = protein.exposure_prior(screen.nodes(setting,row['target_reference']),exposure)
        margins['CurvePool'] = np.full(len(ids),float(p)-.5)
        actions.update((m,(margins[m]>0).astype(np.int64)) for m in PROTEIN)
    return dict(ids=ids,**actions,**{f'margin_{m}':z for m,z in margins.items()})


def validate_initial(screen,key,old,row,lex,coefs,bias,fit_record,tol):
    metadata = screen.initial[key]
    for name in ('m','n_features','zero','params'):
        if metadata[name]!=fit_record[name]:
            raise PermissionError(f'Initial FLDA candidate or feature space differs: {key}/{name}')
    for role in ('D','C','E'):
        if metadata[role]!=len(row['roles'][role]):
            raise PermissionError('Original FLDA sample count differs')
    x = wl.sparse_rows(screen.features[row['setting'],row['source']],old['e'],lex)
    rows = []
    for name in ORIGINAL:
        rebuilt = x@coefs[name]+bias[name]
        np.testing.assert_allclose(rebuilt,old[name],rtol=0,atol=tol)
        np.testing.assert_array_equal(rebuilt>0.,old[name]>0.)
        rows.append({k:row[k] for k in data.IDENTITY} | dict(
            method=name,n=len(rebuilt),max_abs_error=float(np.max(abs(rebuilt-old[name]))),hard_equal=True))
    np.testing.assert_array_equal(count_matched(old['e'],old['Static'],int(np.count_nonzero(old['Target']>0))),
                                  old['Count']>0)
    return rows


def freeze(out,screen,cfg):
    fitted,replay = {},[]
    for sub in ('predictions','models'):
        (out/sub).mkdir()
    with ZipFile(screen.package) as previous, data.ScopedLabels(screen.archive.by_setting,screen.roles,'fit') as port:
        for key,row in screen.roles.items():
            role = row['roles']
            with port.direction(key):
                yd = port.load(key,'D',role['D'])
                yc = port.load(key,'C',role['C'])
            features = screen.features[row['setting'],row['source']]
            lex,w,b,fit_record = fit(features,row,yd,yc)
            old = screen.original_arrays(previous,key)
            replay.extend(validate_initial(screen,key,old,row,lex,w,b,fit_record,cfg['absolute_replay_tolerance']))
            model = None
            if row['setting']=='proteins_edge':
                model = protein.fit(screen.nodes(row['setting'],role['D']),yd)
            fitted[key] = dict(**fit_record,protein=model,initial_member_sha256=screen.initial[key]['prediction_sha'])
            prior_run.save_predictions(out,f'models/{key}.npz',dict(**w,intercepts=np.asarray(list(b.values()))))
            pred = predictions(screen,row,lex,w,b,model)
            prior_run.save_predictions(out,f'predictions/{key}.npz',pred)
            print(f'R9 dropout replay {key} E={len(old["e"])} supplement={len(pred["ids"])}',flush=True)
    data.validate_receipt(screen.roles,port.actual,'fit')
    for name,obj in (('roles',screen.roles),('fit-label-access',port.actual),('fitted',fitted)):
        write_json(out/f'{name}.json',obj)
    write_table(out/'model-replay.csv',replay)
    return prior_run.seal(out,'OBS',members(screen.roles),parents(cfg))


def observed(out,screen,cfg,sha):
    raw = prior_run.verified_members(out,'OBS',sha,members(screen.roles),parents(cfg))
    fit = json.loads(raw['fitted.json'])
    if (json.loads(raw['roles.json'])!=screen.roles or set(fit)!=set(screen.roles)
            or cfg!=json.loads(raw['executed-config.json'])['plan']):
        raise PermissionError('Original source dropout roles or plan differ from new OBS')
    data.validate_receipt(screen.roles,json.loads(raw['fit-label-access.json']),'fit')
    result = {}
    with ZipFile(screen.package) as previous:
        for key,row in screen.roles.items():
            features = screen.features[row['setting'],row['source']]
            lex = wl.vocabulary(features,row['roles']['D'])
            record = fit[key]
            one = inputs.npz_member(raw[f'models/{key}.npz'])
            if (set(one)!={'intercepts',*ORIGINAL} or one['intercepts'].shape!=(4,)
                    or any(one[m].shape!=(len(lex),) for m in ORIGINAL)
                    or lex_hash(lex)!=record['vocabulary_sha256']
                    or record['initial_member_sha256']!=screen.initial[key]['prediction_sha']
                    or any(a.dtype!=np.float64 or not np.isfinite(a).all() for a in one.values())):
                raise PermissionError(f'Model replay certificate differs: {key}')
            bias = dict(zip(ORIGINAL,map(float,one['intercepts']),strict=True))
            validate_initial(screen,key,screen.original_arrays(previous,key),row,lex,one,bias,record,
                             cfg['absolute_replay_tolerance'])
            expected = predictions(screen,row,lex,one,bias,record['protein'])
            actual = inputs.npz_member(raw[f'predictions/{key}.npz'])
            if set(actual)!=set(expected) or any(actual[k].dtype!=v.dtype or not np.array_equal(actual[k],v)
                                               for k,v in expected.items()):
                raise PermissionError(f'Supplemental predictions differ from the frozen D/C model: {key}')
            result[key] = actual
    return result
