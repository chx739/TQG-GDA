"""Label-boundary-safe numerical components for the bounded optimization study."""
from dataclasses import dataclass
import numpy as np
import torch

from .assumption_diagnostics import metrics, split_folds, fit_grid, ProbeConfig


def bandwidth_kernel(distance, multiplier):
    if multiplier <= 0 or not np.isfinite(distance).all():
        raise ValueError('Invalid distance or bandwidth')
    return np.exp(-np.maximum(distance, 0)/(2*multiplier**2))


def effective_lambda(base, count, reference, convention):
    if min(base, count, reference) <= 0:
        raise ValueError('Positive regularization and sample sizes required')
    if convention == 'FIXED':
        return float(base)
    if convention == 'NORMALIZED':
        return float(base)*count/reference
    raise ValueError(convention)


def kernel_factor(kernel):
    values, vectors = np.linalg.eigh((kernel+kernel.T)/2)
    if values.min() < -1e-8:
        raise ValueError('Indefinite kernel')
    return values.clip(0), vectors


@dataclass
class RidgeFit:
    probabilities: np.ndarray  # [label variant, lambda, evaluation sample]
    alpha: np.ndarray          # probability residual coefficients [variant, lambda, train]
    eigenvalues: np.ndarray
    effective_dimensions: np.ndarray
    rkhs_squared: np.ndarray   # probability-coordinate norm; multiply by ||Q1-Q0||^2 for contrast


def ridge_grid(kernel, evaluation_kernel, labels, train_probability, evaluation_probability,
               penalties, factor=None):
    """Evaluation labels are deliberately absent from the fitting interface."""
    y = np.asarray(labels, dtype=float)
    penalties = np.asarray(penalties, dtype=float)
    if y.ndim != 2 or y.shape[1] != len(kernel) or np.any(penalties <= 0):
        raise ValueError('Invalid regression dimensions or regularization')
    values, vectors = kernel_factor(kernel) if factor is None else factor
    residual = y-np.asarray(train_probability)[None, :]
    projected = residual@vectors
    scaled = projected[:, None, :]/(values[None, None, :]+penalties[None, :, None])
    alpha = scaled@vectors.T
    probabilities = np.asarray(evaluation_probability)[None, None, :]+alpha@evaluation_kernel.T
    if not np.isfinite(probabilities).all():
        raise FloatingPointError('Nonfinite ridge predictions')
    return RidgeFit(probabilities, alpha, values,
        (values[None, :]/(values[None, :]+penalties[:, None])).sum(1),
        (scaled**2*values[None, None, :]).sum(2))


def choose_kernel(scores, multipliers, penalties):
    """scores[bandwidth, lambda]; exact ties favor stronger/wider regularization."""
    if not np.isfinite(scores).all():
        raise ValueError('All CV scores must be finite')
    return min(np.ndindex(scores.shape), key=lambda ij: (scores[ij], -penalties[ij[1]], -multipliers[ij[0]]))


def mean_readout(kernel, cross_kernel, target_response, penalty):
    """Binary specialization, used only where q=1 is asserted."""
    if target_response.shape[1] != 1:
        raise ValueError('Binary reader requires q=1')
    hst = cross_kernel@target_response[:, 0, :]/len(target_response)
    return np.linalg.solve(kernel+penalty*np.eye(len(kernel)), hst).T


def closest_bank(source_ids, cross_kernel, count):
    if not 0 < count <= len(source_ids):
        raise ValueError('Invalid bank size')
    maximum = np.asarray(cross_kernel).max(1)
    order = sorted(range(len(source_ids)), key=lambda i: (-maximum[i], int(source_ids[i])))
    return np.array(sorted(order[:count]), dtype=int)


def check_mean_identity(kss, kst, ktt, target_response, residual, penalty, calibration):
    """Independent multi-class Kronecker oracle, not the production mean solver."""
    from .d34_diagnostics import solve_observed
    q = target_response.shape[1]
    bt = target_response.reshape(-1, target_response.shape[-1])/len(target_response)
    block_k = torch.kron(kss.contiguous(), torch.eye(q, dtype=kss.dtype))
    block_ts = torch.kron(kst.mT.contiguous(), torch.eye(q, dtype=kss.dtype))
    alpha = torch.linalg.solve(block_k+penalty*torch.eye(len(block_k), dtype=kss.dtype), residual)
    direct = bt.mT@(block_ts@alpha)
    w, _ = solve_observed((kss, kst, ktt), target_response, 'MEAN', penalty, calibration)
    iso, _ = solve_observed((kss, kst, ktt), target_response, 'ISO', 2*penalty, calibration)
    return dict(mean_vs_pointwise_max=float((w@residual-direct).abs().max()),
                mean_vs_iso_max=float((w-iso).abs().max()))


def select_kernel(source_ids, distances, label_variants, probabilities, settings):
    """Source-only CV. One independently selected configuration per label variant."""
    ids=list(map(int,source_ids));n=len(ids)
    folds=split_folds(ids,3,'A4-inner',0)
    variants=np.asarray(label_variants);records=[]
    options=[];scores=[]
    for sig in settings['signatures']:
        for c in settings['bandwidth_multipliers']:
            kernel=bandwidth_kernel(distances[sig][:n,:n],c)
            cv=np.zeros((len(settings['lambda_conventions']),len(variants),len(settings['kernel_lambdas'])))
            for f,validation in enumerate(folds):
                va=[ids.index(i) for i in validation];tr=sorted(set(range(n))-set(va))
                kt=kernel[np.ix_(tr,tr)];ke=kernel[np.ix_(va,tr)];factor=kernel_factor(kt)
                for co,convention in enumerate(settings['lambda_conventions']):
                    penalties=[effective_lambda(l,len(tr),n,convention) for l in settings['kernel_lambdas']]
                    fitted=ridge_grid(kt,ke,variants[:,tr],probabilities[tr],probabilities[va],penalties,factor)
                    for v in range(len(variants)):
                        for li,base in enumerate(settings['kernel_lambdas']):
                            score=metrics(variants[v,va],fitted.probabilities[v,li])['nll'];cv[co,v,li]+=score/3
                            records.append(dict(variant=v,signature=sig,bandwidth=c,convention=convention,
                                lambda_base=base,lambda_actual=penalties[li],fold=f,nll=score,training_ids=[ids[i] for i in tr],validation_ids=validation))
            for co,convention in enumerate(settings['lambda_conventions']):
                for li,base in enumerate(settings['kernel_lambdas']):
                    options.append(dict(signature=sig,bandwidth=c,convention=convention,lambda_base=base))
                    scores.append(cv[co,:,li])
    scores=np.array(scores)
    chosen=[]
    for v in range(len(variants)):
        j=min(range(len(options)),key=lambda j:(scores[j,v],-options[j]['lambda_base'],-options[j]['bandwidth'],
            options[j]['signature']!='H',options[j]['convention']!='NORMALIZED'))
        chosen.append(dict(options[j],cv_nll=float(scores[j,v]),option_index=j))
    return chosen,records


def select_local(source_ids, kernels, label_variants, probabilities, settings):
    """Choose a local bank using source audit NLL; target inputs only rank neighbors."""
    ids=list(map(int,source_ids));n=len(ids);ys=np.asarray(label_variants)
    options=[dict(selector=s,n=count,convention=convention) for count in settings['bank_sizes']
             for s in settings['signatures'] for convention in settings['lambda_conventions']]
    cv=np.zeros((len(options),len(ys)));records=[]
    h=kernels['H']
    for f,validation in enumerate(split_folds(ids,4,'outer')):
        va=[ids.index(i) for i in validation];tr=sorted(set(range(n))-set(va))
        for j,option in enumerate(options):
            selected=closest_bank([ids[i] for i in tr],kernels[option['selector']][np.ix_(tr,range(n,kernels[option['selector']].shape[1]))],option['n'])
            ix=np.array(tr)[selected];count=len(ix)
            lam=effective_lambda(settings['base_joint_lambda'],count,n,option['convention'])/2
            fit=ridge_grid(h[np.ix_(ix,ix)],h[np.ix_(va,ix)],ys[:,ix],probabilities[ix],probabilities[va],[lam])
            for v in range(len(ys)):
                score=metrics(ys[v,va],fit.probabilities[v,0])['nll'];cv[j,v]+=score/4
                records.append(dict(option,variant=v,fold=f,nll=score,proxy_mean_lambda=lam,
                    bank_ids=[ids[i] for i in ix],validation_ids=validation))
    selected=[]
    for v in range(len(ys)):
        j=min(range(len(options)),key=lambda j:(cv[j,v],-options[j]['n'],options[j]['selector']!='H',options[j]['convention']!='NORMALIZED'))
        selected.append(dict(options[j],cv_nll=float(cv[j,v]),option_index=j))
    return selected,records


def teacher_fit(source_ids, features, label_variants, evaluation_features, penalties, seed):
    """LR soft-label teacher. Entire CV/refit repeated for every shuffled label set."""
    ids=list(map(int,source_ids));ys=np.asarray(label_variants);n=len(ids);records=[]
    scores=np.zeros((len(ys),len(penalties)));configuration=ProbeConfig()
    for f,validation in enumerate(split_folds(ids,3,'teacher-inner')):
        va=[ids.index(i) for i in validation];tr=sorted(set(range(n))-set(va))
        pred,_,info=fit_grid('LR',features[tr],ys[:,tr],np.zeros(len(tr)),features[va],np.zeros(len(va)),
            None,None,None,penalties,seed,configuration)
        for v in range(len(ys)):
            for j,lam in enumerate(penalties):
                loss=metrics(ys[v,va],pred[v,j])['nll'];scores[v,j]+=loss/3
                records.append(dict(variant=v,fold=f,penalty=lam,nll=loss,optimization=info[v*len(penalties)+j],
                    train_ids=[ids[i] for i in tr],validation_ids=validation))
    chosen=[min(range(len(penalties)),key=lambda j:(scores[v,j],-penalties[j])) for v in range(len(ys))]
    selected=[penalties[j] for j in chosen]
    pred,states,info=fit_grid('LR',features,ys,np.zeros(n),evaluation_features,np.zeros(len(evaluation_features)),
                            None,None,None,selected,seed,configuration,paired=True)
    metadata=[dict(variant=v,penalty=selected[v],cv_nll=float(scores[v,chosen[v]]),optimization=info[v]) for v in range(len(ys))]
    return pred[:,0],states,metadata,records


def feedback_controls(full, base, fixed_permutations, pipeline_permutations, basis, block_seed, extra=None, epsilon=1e-12):
    """Separate conditional label controls from whole-selection permutation controls."""
    from .p0_diagnostics import normalized
    label=full-base;fn=float(full.norm());ln=float(label.norm())
    bm,bok=normalized(base,fn,epsilon)
    vectors=dict(FULL=full,BASE=base,BASE_M=bm,LABEL=label,**(extra or {}))
    valid={k:float(v.norm())>epsilon for k,v in vectors.items()};valid['BASE_M']=bok
    for i,(conditional,pipeline) in enumerate(zip(fixed_permutations,pipeline_permutations),1):
        delta,ok=normalized(conditional-base,ln,epsilon)
        rand=torch.randn(basis.shape[1],generator=torch.Generator().manual_seed(block_seed+9200+i),dtype=torch.float64)
        rlabel,rok=normalized(basis@rand,ln,epsilon)
        cp,cr=base+delta,base+rlabel
        vectors[f'CPL_{i}']=cp;vectors[f'CRL_{i}']=cr
        valid[f'CPL_{i}']=ok and ln>epsilon;valid[f'CRL_{i}']=rok and ln>epsilon
        for key,g in [(f'CPL_M_{i}',cp),(f'CRL_M_{i}',cr),(f'FULL_PERM_M_{i}',pipeline),(f'FULL_RAND_M_{i}',basis@rand)]:
            value,is_valid=normalized(g,fn,epsilon);vectors[key]=value;valid[key]=is_valid and fn>epsilon
            if key.startswith('CPL_M'):valid[key]&=valid[f'CPL_{i}']
            if key.startswith('CRL_M'):valid[key]&=valid[f'CRL_{i}']
    return vectors,valid


def binary_logit_metrics(margin, labels):
    margin=np.asarray(margin,dtype=float);y=np.asarray(labels,dtype=int)
    ce=float(np.mean(np.logaddexp(0,margin)-y*margin))
    recalls=[np.mean((margin[y==k]>=0)==k) for k in [0,1] if np.any(y==k)]
    return dict(ce=ce,balanced_accuracy=float(np.mean(recalls)) if len(recalls)==2 else None)
