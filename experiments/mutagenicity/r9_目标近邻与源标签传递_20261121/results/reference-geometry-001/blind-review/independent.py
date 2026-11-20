from pathlib import Path
import importlib.metadata
import json
import pickle
import hashlib
import numpy as np
from scipy.linalg import solve
from scipy.special import expit,logit
from prepare import ROOT, DIR, SHA, inputs, residual_inputs, wl
from torch import set_num_threads, set_num_interop_threads

METHODS = ('WL','Size','Base','TargetFull','SourceFull','TargetDiag','BaseSameCount')
CANDIDATES = (0.1,1.,10.,100.)

def optimum(k, y, c):
    # Represent w in D's row span. Solve IRLS for the intercept and
    # coefficients in that span, with zero intercept penalty.
    n=len(y)
    a=np.zeros(n)
    bias=logit(y.mean())
    def objective(coeff, intercept):
        raw=k@coeff
        return np.logaddexp(0.,(1-2*y)*(raw+intercept)).sum()+(raw@coeff)/(2*c)
    old=objective(a,bias)
    for iteration in range(60):
        current=k@a+bias
        p=expit(current)
        q=c*p*(1-p)
        # Solve for fixed intercept first. sum(a)=0 gives free intercept.
        lhs=q[:,None]*k+np.eye(n)
        rhs=np.column_stack((c*(y-p)+q*current,q))
        v,t=solve(lhs,rhs,check_finite=False).T
        b2=v.sum()/t.sum()
        a2=v-t*b2
        size=np.max(np.abs(k@(a2-a)+b2-bias))
        if size<1e-9:
            a,bias=a2,b2
            break
        step=1.
        while objective(a+step*(a2-a),bias+step*(b2-bias)) > old+3e-11:
            step*=.5
            if step<1e-7: raise ArithmeticError('No IRLS step')
        a+=step*(a2-a)
        bias+=step*(b2-bias)
        old=objective(a,bias)
    else: raise ArithmeticError('IRLS did not converge')
    residual=c*(y-expit(k@a+bias))
    kernel_grad=a-residual
    assert max(abs(residual.sum()),abs(kernel_grad).max()) < 2e-7
    return a,bias

def run_direction(row, feature, yd, yc):
    r=row['roles']
    d,c,u,e=map(r.__getitem__,('D','C','U','E'))
    order=d+c+e+u
    n=len(d); nc=len(c); qn=n+nc+len(e)
    d_slice=slice(0,n); c_slice=slice(n,n+nc); e_slice=slice(n+nc,qn)
    where={g:i for i,g in enumerate(order)}
    assert len(where)==len(order)
    rows={g:feature[g][0] for g in order}
    X=wl.sparse_rows(rows,order,wl.vocabulary(rows,order))
    Z=np.stack([feature[g][1] for g in order])
    sd=Z[:n].std(axis=0); sd[np.ptp(Z[:n],axis=0)==0]=1
    Z=(Z-Z[:n].mean(axis=0))/sd
    F=(X[:qn]@X[:n].T).toarray()
    size=Z[:qn]@Z[:n].T
    base=F+size
    scores={}
    candidates={}
    info={}
    if len(np.unique(yd))==1:
        for name in METHODS[:-1]:
            scores[name]=np.full(len(e),logit((yd.sum()+.5)/(len(d)+1)))
        info['constant']=True
    else:
        for name, kernel in [('WL',F),('Size',size),('Base',base)]:
            models={}
            records=[]
            for strength in CANDIDATES:
                a,b=optimum(kernel[d_slice],yd,strength)
                cal=kernel[c_slice]@a+b
                records.append((int(np.count_nonzero((cal>0)!=yc)), float(np.logaddexp(0.,(1-2*yc)*cal).mean()),strength))
                models[strength]=(a,b)
            _,_,selected=min(records)
            a,b=models[selected]
            scores[name]=kernel[e_slice]@a+b
            candidates[name]=dict(selected=selected,records=records)
    def trace_variance(k):
        gram=k[:n]
        return np.trace(gram)/n-gram.mean()
    tf=trace_variance(F)
    def order_refs(pool):
        def key(g):
            return (hashlib.sha256(f"r9-geometry-v1|{row['setting']}|{row['source']}|{g}".encode()).digest(),g)
        return sorted(pool,key=key)[:min(len(c),len(u))]
    for name,pool in [('TargetFull',u),('SourceFull',c),('TargetDiag',u)]:
        ids=order_refs(pool)
        ref=X[[where[g] for g in ids]]
        if name.endswith('Full'):
            # Each query has its own mean over the entire frozen reference pool.
            sim=(X[:qn]@ref.T).toarray()
            sim-=sim.mean(axis=1,keepdims=True)
            v_gram=(sim@sim[:n].T)/len(ids)
        else:
            mean=np.asarray(ref.mean(axis=0)).ravel()
            svar=np.asarray(ref.power(2).mean(axis=0)).ravel()-mean**2
            assert svar.min()>=-1e-13
            v_gram=(X[:qn].multiply(np.maximum(0,svar))@X[:n].T).toarray()
        tv=trace_variance(v_gram)
        scale=np.sqrt(tf/tv) if min(tf,tv)>1e-14 else None
        info[name]=dict(refs=ids,tf=tf,tv=tv,scale=scale)
        if info.get('constant'): continue
        if scale is None:
            scores[name]=scores['Base']; continue
        joint=.5*(F+(scale*scale)*v_gram)+size
        a,b=optimum(joint[d_slice],yd,candidates['Base']['selected'])
        scores[name]=joint[e_slice]@a+b
    actions={m:(s>0).astype(int) for m,s in scores.items()}
    order_positive=np.lexsort((e,-scores['Base']))
    same=np.zeros(len(e),dtype=int)
    same[order_positive[:int(actions['TargetFull'].sum())]]=1
    actions['BaseSameCount']=same
    return dict(ids=e,actions={k:v.tolist() for k,v in actions.items()},margins={k:v.tolist() for k,v in scores.items()},selection=candidates,geometry=info)

def main():
    set_num_threads(1)
    set_num_interop_threads(1)
    cfg,roles=inputs()
    archive=residual_inputs.Archive(ROOT,cfg,load_graphs=False)
    rows={}
    with residual_inputs.ScopedLabels(archive.by_setting,roles,'fit') as labels:
        for task,row in roles.items():
            assert row['status']=='ok'
            with labels.direction(task):
                d,c=(labels.load(task,r,row['roles'][r]) for r in ('D','C'))
                with (DIR/f"{row['setting']}-M{row['source']}.pkl").open('rb') as f:
                    feat=pickle.load(f)
                rows[task]=run_direction(row,feat,d,c)
                del feat,d,c
            print(task, {m:t['selected'] for m,t in rows[task]['selection'].items()},flush=True)
        residual_inputs.validate_receipt(roles,labels.actual,'fit')
    (DIR/'fit-access.json').write_text(json.dumps(labels.actual,separators=(',',':'))+'\n')
    lines=[]; by_block={}
    for task in sorted(rows):
        result=rows[task]
        block=[]
        for i in np.argsort(result['ids']):
            cols=[task,str(result['ids'][i]),*[str(result['actions'][m][i]) for m in METHODS]]
            block.append('\t'.join(cols)+'\n')
        by_block[task]=SHA(''.join(block).encode('utf-8'))
        lines.extend(block)
    (DIR/'actions.tsv').write_bytes(''.join(lines).encode('utf-8'))
    (DIR/'blind.json').write_text(json.dumps(rows,separators=(',',':'))+'\n')
    note=dict(directory=str(DIR),canonical_hash=SHA((DIR/'actions.tsv').read_bytes()),evaluation_graph_occurrences=len(lines),direction_count=len(by_block),columns=['task','graph_id',*METHODS],canonicalization='No header, UTF-8; sort task by ASCII, graph_id by numeric ascending; join fields with TAB, each row ends with LF; integer actions 0 or 1.',by_direction_sha256=by_block,files={f:SHA((DIR/f).read_bytes()) for f in ['actions.tsv','blind.json','fit-access.json','inputs.json','prepare.py','independent.py']},versions={pkg:importlib.metadata.version(pkg) for pkg in ['scipy','numpy','torch','torch-geometric']})
    (DIR/'seal.json').write_text(json.dumps(note,indent=2)+'\n')
    print('SEALED',note['canonical_hash'],'rows',len(lines),flush=True)
if __name__=='__main__':
    main()
