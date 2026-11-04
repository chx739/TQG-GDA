"""Query-label versus prediction decomposition and fixed-bandwidth D7 helpers."""
import hashlib
import torch
from torch.nn import functional as F
from .p0_diagnostics import normalized,cosine
from .d34_diagnostics import solve_observed


def label_prediction_vectors(geo,labels,block_seed,permutation_offsets,random_offsets,epsilon):
    y=F.one_hot(labels,geo.source_probabilities.shape[1]).to(geo.source_probabilities)
    p=geo.source_probabilities;q=geo.contrasts
    parts={'GLOBAL':((y.mean(0)-p.mean(0))[None].expand_as(p)@q).reshape(-1),
           'LABEL':((y-y.mean(0))@q).reshape(-1),
           'PRED':(-(p-p.mean(0))@q).reshape(-1)}
    apply=lambda x:-(geo.basis@(geo.readout@x))
    v={k:apply(x) for k,x in parts.items()}
    v['BASE']=v['GLOBAL']+v['PRED']
    v['FULL']=apply(((y-p)@q).reshape(-1))
    error=float((v['FULL']-v['BASE']-v['LABEL']).norm())
    if error>1e-10*(1+float(v['FULL'].norm())):raise ValueError('Label/prediction additivity failed')
    label_norm=float(v['LABEL'].norm());valid={k:float(x.norm())>epsilon for k,x in v.items()}
    for rep,offset in enumerate(permutation_offsets,1):
        perm=torch.randperm(len(labels),generator=torch.Generator().manual_seed(block_seed+offset))
        raw=apply(((y[perm]-y.mean(0))@q).reshape(-1))
        matched,ok=normalized(raw,label_norm,epsilon)
        v[f'PL-{rep}']=raw;v[f'CPL-{rep}']=v['BASE']+matched
        valid[f'PL-{rep}']=ok;valid[f'CPL-{rep}']=ok and label_norm>epsilon
    for rep,offset in enumerate(random_offsets,1):
        u=torch.randn(geo.basis.shape[1],generator=torch.Generator().manual_seed(block_seed+offset),dtype=torch.float64)
        vector,ok=normalized(geo.basis@u,label_norm,epsilon)
        v[f'RL-{rep}']=vector;v[f'CRL-{rep}']=v['BASE']+vector
        valid[f'RL-{rep}']=ok;valid[f'CRL-{rep}']=ok and label_norm>epsilon
    norms={name+'_norm':float(v[name].norm()) for name in ['GLOBAL','LABEL','PRED','BASE','FULL']}
    denominator=sum(norms[k+'_norm'] for k in ['GLOBAL','LABEL','PRED'])
    stats=dict(**norms,additivity_error=error,label_prediction_cosine=cosine(v['LABEL'],v['PRED'],epsilon),
        cancellation_ratio=norms['FULL_norm']/denominator if denominator>epsilon else None)
    return v,valid,stats


def label_prediction_steps(vectors,g_source,g_a,beta,epsilon):
    fullnorm=float(vectors['FULL'].norm())
    base_m,base_ok=normalized(vectors['BASE'],fullnorm,epsilon)
    oracle,oracle_ok=normalized(g_a,fullnorm,epsilon)
    steps={'S':g_source,'A':g_source+beta*vectors['BASE'],'B':g_source+beta*vectors['FULL'],
        'A-M':g_source+beta*base_m,'LABEL':g_source+beta*vectors['LABEL'],
        'PRED':g_source+beta*vectors['PRED'],'ORACLE-M':g_source+beta*oracle}
    valid={name:True for name in steps};valid['A-M']=base_ok;valid['ORACLE-M']=oracle_ok
    labelnorm=float(vectors['LABEL'].norm())
    for kind in ['PL','RL']:
        for rep in [1,2,3]:
            name=f'{kind}-{rep}';composite=vectors['C'+name]
            steps[name]=g_source+beta*composite
            valid[name]=labelnorm>epsilon and float(vectors[name].norm())>epsilon
            matched,ok=normalized(composite,fullnorm,epsilon)
            steps[name+'-M']=g_source+beta*matched
            valid[name+'-M']=valid[name] and ok
    return steps,valid


def query_order(query_ids,seed,rep):
    return sorted(query_ids,key=lambda i:hashlib.sha256(f'D7|{seed}|{rep}|{i}'.encode()).hexdigest())


def query_lambda(n,total,convention,base):
    if convention=='FIXED':return base
    if convention=='NORMALIZED':return base*(n/total)
    raise ValueError('Unknown lambda convention')


def duplicate_check(kernels,response,mode,lambda_w,config):
    n=len(kernels[0]);q=response.shape[1];r=response.shape[2]
    idx=torch.arange(n).repeat_interleave(2)
    duplicate=(kernels[0][idx][:,idx],kernels[1][idx],kernels[2])
    original,io=solve_observed(kernels,response,mode,lambda_w,config)
    half,ih=solve_observed(kernels,response,mode,lambda_w/2,config)
    fixed,iff=solve_observed(duplicate,response,mode,lambda_w,config)
    scaled,iss=solve_observed(duplicate,response,mode,2*lambda_w,config)
    collapse=lambda w:w.reshape(r,n,2,q).sum(2).reshape(r,n*q)
    rel=lambda a,b:float((a-b).norm())/max(float(b.norm()),1e-12)
    return dict(mode=mode,query=n,q=q,rank=r,scaled_collapse_error=rel(collapse(scaled),original),
        fixed_collapse_vs_half_lambda_error=rel(collapse(fixed),half),
        scaled_original_converged=io['converged'] and iss['converged'],
        fixed_half_converged=ih['converged'] and iff['converged'],
        all_solver_info=[io,ih,iff,iss])
