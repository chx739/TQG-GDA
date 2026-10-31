"""Fixed-reader kernels and calibration observers for frozen D3/D4 plans."""
from dataclasses import replace
import time

import torch
from torch.nn import functional as F

from .calibration_v2 import CalibrationConfig, check_psd, solve_calibration
from .feedback_v2 import V2Config, query_matrices
from .p0_diagnostics import append_json, fixed_isotropic


@torch.no_grad()
def kernel_family(source_h, target_h, source_a, target_a, permutation, *, scale_epsilon=1e-8):
    n=len(source_h)
    if n<2 or sorted(permutation.tolist())!=list(range(n)):
        raise ValueError('Invalid source signature permutation')
    if any(not torch.isfinite(x).all() for x in (source_h,target_h,source_a,target_a)):
        raise FloatingPointError('Nonfinite signatures')
    a_s,a_t=source_a.flatten(1),target_a.flatten(1)
    h,a=torch.cat((source_h,target_h)),torch.cat((a_s,a_t))
    ds_h,ds_a=torch.pdist(source_h).square(),torch.pdist(a_s).square()
    floor=scale_epsilon**2
    scale_h=torch.quantile(ds_h,.5).clamp_min(floor)
    scale_a=torch.quantile(ds_a,.5).clamp_min(floor)
    dh=torch.cdist(h,h,compute_mode='donot_use_mm_for_euclid_dist').square()/(2*scale_h)
    da=torch.cdist(a,a,compute_mode='donot_use_mm_for_euclid_dist').square()/(2*scale_a)
    ha=torch.exp(-(dh+da))
    all_indices=torch.cat((permutation,torch.arange(n,len(h))))
    grams={'H':torch.exp(-dh),'A':torch.exp(-da),'HA':ha,
           'CONST':torch.ones_like(ha),'HA_PERM':ha[all_indices][:,all_indices]}
    for name,gram in grams.items(): check_psd(gram,'D4 full Gram '+name)
    statistics=dict(scale_h_squared=float(scale_h),scale_a_squared=float(scale_a),
        scale_h_at_floor=bool(scale_h<=floor),scale_a_at_floor=bool(scale_a<=floor),
        distance_h_source_quantiles=torch.quantile(ds_h,torch.tensor([.05,.5,.95],dtype=ds_h.dtype)).tolist(),
        distance_a_source_quantiles=torch.quantile(ds_a,torch.tensor([.05,.5,.95],dtype=ds_a.dtype)).tolist(),
        distance_h_source_CV=float(ds_h.std(unbiased=False)/ds_h.mean()) if ds_h.mean()>0 else None,
        distance_a_source_CV=float(ds_a.std(unbiased=False)/ds_a.mean()) if ds_a.mean()>0 else None)
    return {name:(gram[:n,:n],gram[:n,n:],gram[n:,n:]) for name,gram in grams.items()},statistics


def query_error(kss,hst,htt,w):
    n=len(kss);r=w.shape[0];q=w.shape[1]//n
    wk=torch.einsum('aic,ij->ajc',w.reshape(r,n,q),kss).reshape(r,n*q)
    cross=w@hst
    e=htt-cross-cross.mT+wk@w.mT
    return (e+e.mT)/2


@torch.no_grad()
def solve_observed(kernels,target_response,mode,lambda_w,base_config):
    kss,kst,ktt=kernels
    hst,htt=query_matrices(kst,ktt,target_response)
    r=hst.shape[1]
    if not r:
        return hst.mT,dict(mode=mode,converged=True,iterations=0,termination='zero_rank',gap=0.,
            primal=None,dual=None,backtracks=0,mean_error=None,max_error=None,regularization=0.,
            weight_norm=0.,omega_eigenvalues=None,omega_deviation=None,delta_min=None,delta_max=None,
            effective_lambda=[],solve_seconds=0.)
    started=time.perf_counter()
    if mode=='ISO':
        w=fixed_isotropic(kss,hst,lambda_w)
        e=query_error(kss,hst,htt,w)
        omega=torch.eye(r,dtype=w.dtype)/r
        objective=2*e.trace()/r+lambda_w*w.square().sum()/r
        info=dict(mode=mode,converged=True,iterations=1,termination='fixed_isotropic_direct_solve',
                  gap=0.,primal=float(objective),dual=float(objective),backtracks=0)
    else:
        cfg=replace(CalibrationConfig(**base_config),mode='joint' if mode=='JOINT' else 'mean_only',lambda_w=lambda_w)
        if mode not in ('MEAN','JOINT'): raise ValueError('Unknown observed calibration mode')
        w,info=solve_calibration(kss,hst,htt,cfg,return_details=True)
        details=info.pop('_details')
        e,omega=details['error'],details['omega']
    elapsed=time.perf_counter()-started
    reg=lambda_w*float(w.square().sum())/r
    if omega is None:
        omega_eigenvalues=None;omega_deviation=None
        delta=torch.ones(r,dtype=w.dtype)
    else:
        omega_eigenvalues=torch.linalg.eigvalsh(omega).tolist()
        omega_deviation=float((omega-torch.eye(r,dtype=w.dtype)/r).norm())
        delta=torch.linalg.eigvalsh(torch.eye(r,dtype=w.dtype)+r*omega)
    info.update(mode=mode,mean_error=float(e.trace()/r),max_error=float(torch.linalg.eigvalsh(e)[-1]),
        regularization=reg,weight_norm=float(w.norm()),omega_eigenvalues=omega_eigenvalues,
        omega_deviation=omega_deviation,delta_min=float(delta.min()),delta_max=float(delta.max()),
        effective_lambda=(lambda_w/delta).tolist(),solve_seconds=elapsed)
    if not torch.isfinite(w).all(): raise FloatingPointError('Nonfinite observed W')
    return w,info


def source_residual(probabilities,labels,contrasts):
    return ((F.one_hot(labels,probabilities.shape[1]).double()-probabilities)@contrasts).reshape(-1)


@torch.no_grad()
def kernel_statistics(kernels,bank_labels,bank_residual,audit_labels=None,audit_residual=None):
    kss,kst,ktt=kernels
    eig=torch.linalg.eigvalsh((kss+kss.mT)/2)
    positive=eig.clamp_min(0)
    tolerance=max(kss.shape)*torch.finfo(kss.dtype).eps*max(float(eig.max()),0.)
    numerical_rank=int((eig>tolerance).sum())
    if positive.sum()>0:
        p=positive/positive.sum()
        effective_rank=float(torch.exp(-(p[p>0]*p[p>0].log()).sum()))
    else: effective_rank=0.
    nn_kernel=kss.clone();nn_kernel.fill_diagonal_(-float('inf'))
    neighbors=nn_kernel.argmax(1)
    counts=torch.bincount(bank_labels,minlength=2).double()
    n=len(bank_labels)
    chance=float((counts*(counts-1)).sum()/(n*(n-1)))
    r_s=bank_residual.reshape(n,-1)
    result=dict(K_SS_rank=numerical_rank,K_SS_effective_rank=effective_rank,
        K_ST_mean=float(kst.mean()),K_ST_max=float(kst.max()),
        K_ST_quantiles=torch.quantile(kst.reshape(-1),torch.tensor([.05,.5,.95],dtype=kst.dtype)).tolist(),
        offdiagonal_std=float(kss[~torch.eye(n,dtype=torch.bool)].std(unbiased=False)),
        source_nn_label_agreement=float((bank_labels==bank_labels[neighbors]).double().mean()),
        source_prior_agreement=chance,
        source_nn_residual_MSE=float((r_s-r_s[neighbors]).square().sum(1).mean()))
    if audit_labels is not None:
        neighbors=kst.argmax(0)
        result['target_nn_label_agreement']=float((audit_labels==bank_labels[neighbors]).double().mean())
        result['target_nn_residual_MSE']=float((audit_residual.reshape(len(audit_labels),-1)-r_s[neighbors]).square().sum(1).mean())
    return result


class SourceAuditOracle:
    """Source-query holdout labels, separated from bank, fit and target roles."""
    def __init__(self,task,bank_ids,audit_ids,log_path,*,enabled=False):
        self.task=task;self.bank=set(bank_ids);self.allowed=set(audit_ids)
        if self.bank&self.allowed or self.bank|self.allowed!=set(task.query_ids):
            raise ValueError('Source audit must partition the original query pool')
        self.enabled=enabled;self.seals=set();self.log_path=log_path

    def seal(self,token,a_ids,**hashes):
        if not set(a_ids)<=self.allowed: raise PermissionError('Source audit IDs outside holdout')
        self.seals.add(token)
        append_json(self.log_path,dict(event='unlabeled_query_sealed',scope='source_query_audit',
            token=token,ids=list(a_ids),time=time.time(),**hashes))

    def labels(self,ids,*,token,purpose):
        ids=list(map(int,ids))
        valid=self.enabled and token in self.seals and bool(ids) and set(ids)<=self.allowed and not set(ids)&self.bank
        append_json(self.log_path,dict(event='label_access' if valid else 'access_denied',
            scope='source_query_audit',ids=ids,token=token,purpose=purpose,time=time.time()))
        if not valid: raise PermissionError('Enabled sealed source-audit role required')
        return torch.tensor([self.task.labels[i] for i in ids],dtype=torch.long)
