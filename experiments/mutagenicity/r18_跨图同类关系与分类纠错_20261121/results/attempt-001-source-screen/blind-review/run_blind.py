"""Blind reproduction from the R18 protocol. Only historical R9 geometry and D/C are used."""
import csv, hashlib, json, sys
from contextlib import ExitStack
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from scipy.special import expit, logit
from numpy.testing import assert_array_equal

repo = Path('/home/hxchen/research/GDA/TSBA/repository')
out = Path(__file__).parent
sys.path.insert(0, str(repo))
sys.path.insert(0, '/tmp')
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.composition import scores
from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import reference_geometry as rg, residual_inputs as ri, wl
from experiments.mutagenicity.r10_原生图适配基线与增量_20261121.src import structural

bases = ('WL', 'Size', 'Base')
action_names = ('Strong','WL','Size','Base','WL_C1','CalAnchor','Product','TargetUnary','TargetFull','SourceFull','PermMean','EqualCount','RelationOnly')
r9 = Path(rg.__file__).parents[1]
r9orig = r9/'results/reference-geometry-001'
sha = lambda payload: hashlib.sha256(payload).hexdigest()
def json_file(path):
    return json.loads(path.read_bytes())
def put(name, value):
    (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csv_out(name, rows):
    with (out/name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0],delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)

def order(prefix, row, ids):
    return sorted(ids,key=lambda i:(hashlib.sha256(f"r18-{prefix}|{row['setting']}|{row['source']}|{i}".encode()).digest(), i))

def margin_m(a):
    return np.clip(np.tanh(a/2),-1+1e-12,1-1e-12)

def features3(rows, size_by_id, training, query):
    lex = wl.vocabulary(rows, training)
    F = wl.sparse_rows(rows,query,lex)
    train_size = np.stack([size_by_id[i] for i in training])
    mean = train_size.mean(axis=0)
    sd = train_size.std(axis=0)
    sd[np.ptp(train_size,axis=0)==0] = 1.
    size = (np.stack([size_by_id[i] for i in query])-mean)/sd
    return (F,size,sparse.hstack((F,size),format='csr'))

def candidate_rank(family,validation):
    winner = min(validation,key=lambda x:(-x['correct'],x['nll'],x['C']))
    return (-winner['correct'],winner['nll'],winner['C'],bases.index(family))

def refit(training, c, held, yd,yc,rows,sz,solver):
    if np.all(yd==yd[0]):
        return np.full(len(held),logit((yd.sum()+.5)/(len(yd)+1))),('constant',None)
    mats = features3(rows,sz,training,[*training,*c,*held])
    n,k = len(training),len(c)
    fits = {}
    for name,mat in zip(bases,mats,strict=True):
        train,cal = mat[:n],mat[n:n+k]
        fitted = structural.fit_candidates(train,cal,yd,yc,dict(C=structural.STRENGTHS,solver=solver))
        fits[name]=fitted
    fam = min(bases,key=lambda name:candidate_rank(name,fits[name]['validation']))
    return scores(mats[bases.index(fam)][n+k:],structural.select_model(fits[fam])),(fam,fits[fam]['selected_C'])

def original_margins(original, query, d,c,y_d,y_c,rows,sz,solver, model_bytes):
    model = json.loads(model_bytes)
    if model['status']=='single_class_D':
        arrays = {name:np.full(len(query),model['constant']) for name in [*bases,'WL_C1']}
        return arrays, 'WL'
    candidates = model['candidates']
    key = min(bases,key=lambda n:candidate_rank(n,candidates[n]['validation']))
    vocab = {(layer,bytes.fromhex(word)):i for i,(layer,word) in enumerate(model['vocabulary'])}
    f = wl.sparse_rows(rows,query,vocab)
    z = (np.stack([sz[i] for i in query])-model['size_mean'])/model['size_scale']
    mats = (f,z,sparse.hstack((f,z),format='csr'))
    arrays = {name:scores(mat,candidates[name]['coefficients']) for name,mat in zip(bases,mats,strict=True)}
    for name in bases:
        assert_array_equal(arrays[name][:len(original['e_ids'])],original['margin_'+name])
    fd, fc = (wl.sparse_rows(rows,ids,vocab) for ids in (d,c))
    fixed = structural.fit_candidates(fd,fc,y_d,y_c,dict(C=[1.],solver=solver))
    arrays['WL_C1'] = scores(f,structural.select_model(fixed))
    return arrays, key

class Relations:
    def __init__(self, rows, sizes):
        ids=list(rows)
        self.loc = {gid:i for i,gid in enumerate(ids)}
        words=wl.vocabulary(rows,ids)
        all_wl=wl.sparse_rows(rows,ids,words)
        self.cos=[]
        for depth in range(3):
            cols=[i for (d,_),i in words.items() if d==depth]
            slice_=all_wl[:,cols]
            self.cos.append((slice_@slice_.T).toarray()*3)
        self.size = np.stack([sizes[i] for i in ids])

    def inputs(self, ia,ib, map_margin):
        from r18_blind_relation import pair_features
        idx=np.array([self.loc[i] for i in ia])
        jdx=np.array([self.loc[i] for i in ib])
        sim=np.column_stack([k[idx,jdx] for k in self.cos])
        sa=np.array([map_margin[i] for i in ia])
        sb=np.array([map_margin[i] for i in ib])
        return pair_features(sa,sb,sim,np.abs(self.size[idx]-self.size[jdx]))

def one(key,row,models,solver, rawfeat, yd,yc, rel):
    from r18_blind_relation import calib,head_fit
    d,c,u,e = (row['roles'][q] for q in 'DCUE')
    assert len(set(d+c+u+e))==sum(map(len,(d,c,u,e)))
    rows={i:xy[0] for i,xy in rawfeat.items()}
    sz={i:xy[1] for i,xy in rawfeat.items()}
    key_model=f'models/{key}.json'
    key_pred=f'predictions/{key}.npz'
    original={}
    with np.load(r9orig/key_pred,allow_pickle=False) as stored:
        for name in ('e_ids',*('margin_'+n for n in bases)):
            original[name]=stored[name]
    assert_array_equal(original['e_ids'],e)
    model_bytes=(r9orig/key_model).read_bytes()
    assert sha(model_bytes)==models[key_model] and sha((r9orig/key_pred).read_bytes())==models[key_pred]
    nm=min(len(c),len(u))
    uref,cref=(order('ref',row,l)[:nm] for l in (u,c))
    query=e+uref+cref
    pred,fam=original_margins(original,query,d,c,yd,yc,rows,sz,solver,model_bytes)
    chosen=pred[fam]
    m_query=margin_m(chosen)
    eval_m=dict(zip(query,m_query,strict=True))

    y_d=dict(zip(d,yd,strict=True))
    folds=[order('fold',row,d)[j::4] for j in range(4)]
    oof={}
    fold_choices=[]
    blocks=[]
    for fold in folds:
        skip=set(fold)
        training=[i for i in d if i not in skip]
        y_train=np.array([y_d[i] for i in training])
        a, selection=refit(training,c,fold,y_train,yc,rows,sz,solver)
        oof.update(zip(fold,a,strict=True))
        fold_choices.append(selection)
        if len(fold)>=2:
            fi=np.asarray(fold)
            left,right=np.triu_indices(len(fold),k=1)
            labels=np.array([y_d[i] for i in fold])
            b,x=rel.inputs(fi[left],fi[right],dict(zip(fold,margin_m(a),strict=True)))
            same=(labels[left]==labels[right]).astype(float)
            blocks.append((b,x,same,np.full(len(left),len(fold)/len(left))))
    b,t,cgrad = calib(np.array([oof[i] for i in d]),yd)
    if blocks:
        offset,x,same,w=(np.concatenate(fields) for fields in zip(*blocks,strict=True))
        w=w/w.sum()
        unary,ugrad=head_fit(x[:,:3],offset,same,w,len(d))
        full,fgrad=head_fit(x,offset,same,w,len(d))
    else:
        unary=full=lambda x,off:expit(off)
        ugrad=fgrad=0.

    def queries(ref):
        a=np.repeat(e,len(ref))
        r=np.tile(ref,len(e))
        offset,design=rel.inputs(a,r,eval_m)
        shape=(len(e),len(ref))
        un=unary(design[:,:3],offset).reshape(shape)
        fu=full(design,offset).reshape(shape)
        return un,fu

    mi=m_query[:len(e)]
    def voting(q,ref):
        S=2*q-1
        mref=np.array([eval_m[j] for j in ref])
        relsum=np.mean(S*mref[None,:],axis=1)
        denom=1+np.mean(mref*mref)
        combined=(mi+relsum)/denom
        pure=np.where(relsum==0,mi,relsum)
        perm=(mi+np.mean(S,axis=1)*mref.mean())/denom
        return combined,pure,perm

    unq,fuq=queries(uref)
    ft,pt,perm=voting(fuq,uref)
    ut=voting(unq,uref)[0]
    fs=voting(queries(cref)[1],cref)[0]
    K=int((ft>0).sum())
    equal=np.zeros(len(e),dtype=np.int64)
    equal[np.lexsort((np.asarray(e),-chosen[:len(e)]))[:K]]=1
    q0=(1+mi[:,None]*m_query[len(e):len(e)+len(uref)][None,:])/2
    product_e,_,_=voting(q0,uref)
    product_max=float(np.max(np.abs(product_e-mi)))
    assert product_max<1e-14
    actions={k:pred[k][:len(e)]>0 for k in (*bases,'WL_C1')}
    actions.update(Strong=chosen[:len(e)]>0, CalAnchor=b+t*chosen[:len(e)]>0,Product=mi>0,TargetUnary=ut>0,TargetFull=ft>0,
                   SourceFull=fs>0,PermMean=perm>0,EqualCount=equal,RelationOnly=pt>0)
    assert_array_equal(actions['Product'],actions['Strong'])
    recs=[dict(key=key,graph_id=e[i],**{k:int(actions[k][i]) for k in action_names}) for i in np.argsort(e)]
    ii,jj=np.triu_indices(len(e),1)
    ids=np.array(e)
    off,xx=rel.inputs(ids[ii],ids[jj],eval_m)
    mc=margin_m(b+t*chosen[:len(e)])
    probs=dict(q0=(1+mi[ii]*mi[jj])/2,CalProduct=(1+mc[ii]*mc[jj])/2,
               Unary=unary(xx[:,:3],off),Full=full(xx,off))
    for v in probs.values():
        assert len(v)==len(e)*(len(e)-1)//2 and np.all(np.isfinite(v)&(v>=0)&(v<=1))
    np.savez_compressed(out/'pairs'/f'{key}.npz',e_ids=ids,**probs)
    diagnostics=dict(strong=fam,sizes={k:len(v) for k,v in row['roles'].items()},fold_selections=fold_choices,
        cgrad=float(cgrad),ugrad=float(ugrad),fgrad=float(fgrad),max_product_identity_error=product_max,
        pair_count=len(ii),cal_intercept=float(b),cal_slope=float(t),
        probability_hashes={k:sha(v.astype('<f8').tobytes()) for k,v in probs.items()},
        pairs_graph_ids_sha256=sha(np.column_stack((ids[ii],ids[jj])).astype('<i8').tobytes()),
        fold_ids=folds,U_reference_ids=uref,C_reference_ids=cref)
    return recs,diagnostics

def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cfg=json_file(r9/'configs/reference-geometry.json')
    ref=rg.authenticate(r9,cfg)
    archive,features=rg.prepare(repo,ref)
    print('R9 roles and topology verified',flush=True)
    original_seal=sha((r9orig/'OBS.json').read_bytes())
    assert original_seal=='823ff737f381767bf445c034e18983b2eed8e817e7095fe2bfbf82a2ea1dbee2'
    models=json_file(r9orig/'OBS.json')['files']
    results=[]
    diag={}
    last_group=None
    with ri.ScopedLabels(archive.by_setting,ref.roles,'fit') as port:
        for key,row in ref.roles.items():
            ids=row['roles']
            with port.direction(key):
                yd=port.load(key,'D',ids['D'])
                yc=port.load(key,'C',ids['C'])
            group=row['setting'],row['source']
            feat=features[group]
            if last_group!=group:
                relation=Relations({k:v[0] for k,v in feat.items()},{k:v[1] for k,v in feat.items()})
                last_group=group
            recs,diagnostics=one(key,row,models,archive.plan['solver'],feat,yd,yc,relation)
            diag[key]=diagnostics
            results.extend(recs)
            print(key,diagnostics['strong'],[diagnostics[j] for j in ('pair_count','cgrad','ugrad','fgrad')],flush=True)
    ri.validate_receipt(ref.roles,port.actual,'fit')
    put('fit-label-access.json',port.actual)
    put('diagnostics.json',diag)
    results.sort(key=lambda r:(r['key'],r['graph_id']))
    csv_out('actions.tsv',results)
    from collections import defaultdict
    bounds=defaultdict(lambda: defaultdict(list))
    for key,row in ref.roles.items():
        rr=[r for r in results if r['key']==key]
        for method in action_names:
            changed=sum(r[method]!=r['TargetFull'] for r in rr)
            bounds[row['setting'],method][row['source']].append(Fraction(changed,len(rr)))
    boundrows=[]
    for (setting,method),by_group in sorted(bounds.items()):
        grp={k:sum(v)/len(v) for k,v in by_group.items()}
        full_bound=sum(grp.values())/len(grp)
        boundrows.append(dict(setting=setting,target='TargetFull',comparator=method,
                              n_potential_winning_sources=sum(v>0 for v in grp.values()),
                              max_accuracy_gain_exact=str(full_bound),max_gain_pp=float(100*full_bound),
                              **{f'M{k}_upper_bound_exact':str(v) for k,v in grp.items()}))
    csv_out('blind-bounds.tsv',boundrows)
    import r18_blind_relation as rm
    put('receipt.json',dict(R9_source_obs_sha256=original_seal,directions=len(diag),graph_occurrences=len(results),
        methods=action_names,rows_order='key lexical ascending, then integer graph_id ascending',
        tsv='UTF-8, tab-delimited header, newline LF including the final row',actions_sha256=sha((out/'actions.tsv').read_bytes()),
        probabilities_order='Each direction retains its exact original e_ids. Pairs are native e_ids indices (i,j) with i ascending, j=i+1 .. n-1. Indexed arrays q0, CalProduct, Unary, Full contain respective float64 probabilities.',
        raw_float_hashes='per probability method, little-endian float64 C-order raw bytes in diagnostics.json',
        pair_count=sum(r['pair_count'] for r in diag.values()), impl_sha256=sha(Path(__file__).read_bytes()),math_sha256=sha(Path(rm.__file__).read_bytes())))

if __name__=='__main__': main()
