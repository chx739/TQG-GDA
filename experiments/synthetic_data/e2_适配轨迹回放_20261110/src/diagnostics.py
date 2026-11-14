"""Read-only checkpoint diagnostics on known distributions and fixed probes.

No training, optimizer, score fitting, parameter proposal, or model selection.
Only input gradients are used, to measure the boundary normal at a saved state.
"""
from collections import Counter
import math
import torch
from torch.nn import functional as F
from experiments.common.metrics import classification


def posterior_log_prob(raw, shift):
    # Equal-prior means (-1+shift, 1+shift), variance along x1 = 1/4.
    # log p(y=1|x)/p(y=0|x) = 8*(x1-shift).
    half = 4 * (raw[:, 0] - shift)
    return torch.stack((-half, half), dim=1).log_softmax(-1)


def mean_se(values):
    return dict(mean=float(values.mean()), mc_se=float(values.std(unbiased=True)/math.sqrt(len(values))))


@torch.no_grad()
def source_metrics(logits, labels, true_logp, initial_logits):
    logp = logits.log_softmax(-1); p = true_logp.exp()
    conditional_ce = -(p*logp).sum(-1)
    kl = (p*(true_logp-logp)).sum(-1)
    initial_logp = initial_logits.log_softmax(-1)
    excess = (p*(initial_logp-logp)).sum(-1)
    return dict(**classification(logits,labels), conditional_ce=mean_se(conditional_ce),
                posterior_kl=mean_se(kl), conditional_ce_delta=mean_se(excess),
                posterior_probability_mae=float((logp.exp()[:,1]-p[:,1]).abs().mean()))


@torch.no_grad()
def target_metrics(logits, labels, initial_logits):
    loss = F.cross_entropy(logits,labels,reduction='none')
    initial_loss = F.cross_entropy(initial_logits,labels,reduction='none')
    correct = logits.argmax(-1)==labels
    initial_correct = initial_logits.argmax(-1)==labels
    confidence = logits.softmax(-1).max(-1).values
    wrong = ~correct; n=len(labels)
    def average(values,mask):return float(values[mask].mean()) if mask.any() else None
    groups={}
    for name,mask in [('correct_correct',initial_correct & correct),
                      ('correct_wrong',initial_correct & wrong),
                      ('wrong_correct',~initial_correct & correct),
                      ('wrong_wrong',~initial_correct & wrong)]:
        groups[name]=dict(count=int(mask.sum()),nll_delta_contribution=float((loss-initial_loss)[mask].sum()/n),
                          initial_mean_nll=average(initial_loss,mask),current_mean_nll=average(loss,mask),
                          mean_confidence=average(confidence,mask))
    delta=float((loss-initial_loss).mean())
    if abs(sum(v['nll_delta_contribution'] for v in groups.values())-delta)>1e-12:
        raise ValueError('NLL cohort identity failed')
    return dict(**classification(logits,labels),nll_delta=delta,error_count=int(wrong.sum()),
                mean_wrong_confidence=average(confidence,wrong),mean_correct_confidence=average(confidence,correct),
                wrong_mean_nll=average(loss,wrong),correct_mean_nll=average(loss,correct),
                wrong_nll_contribution=float(loss[wrong].sum()/n),
                correct_nll_contribution=float(loss[correct].sum()/n),
                high_confidence_error_fraction_90=float((wrong & (confidence>=.9)).double().mean()),
                high_confidence_error_fraction_99=float((wrong & (confidence>=.99)).double().mean()),
                # Baseline confidence within each current cohort avoids composition ambiguity.
                cohorts=groups)


def boundary_probe(model, coordinates, nuisance, lower=-3., upper=3., grid_size=513, bisections=32):
    """Find detected roots along raw x1 at fixed nuisance values.

    Multiple/no detected roots are retained as failures, never resolved by
    choosing a convenient root. A finite scan cannot certify global uniqueness
    or detect every tangency/sub-grid crossing. Geometry uses unique detections.
    Normal orientation is grad(f1-f0), so the ideal normal is +e1.
    """
    device=nuisance.device;dtype=nuisance.dtype;n,rest=nuisance.shape;dim=rest+1
    grid=torch.linspace(lower,upper,grid_size,dtype=dtype,device=device)
    with torch.no_grad():
        raw=torch.empty(n,grid_size,dim,dtype=dtype,device=device)
        raw[:,:,0]=grid;raw[:,:,1:]=nuisance[:,None,:]
        vals=[]
        for block in raw.reshape(-1,dim).split(8192):
            logits=model(coordinates(block));vals.append(logits[:,1]-logits[:,0])
        margin=torch.cat(vals).reshape(n,grid_size)
        roots=torch.full((n,),float('nan'),dtype=dtype,device=device)
        counts=torch.zeros(n,dtype=torch.long,device=device);status=[];indices=[];lo=[];hi=[]
        for i,row in enumerate(margin):
            exact=torch.where(row==0)[0];cross=torch.where(row[:-1]*row[1:]<0)[0]
            counts[i]=len(exact)+len(cross)
            if not torch.isfinite(row).all():status.append('nonfinite')
            elif counts[i]==0:status.append('no_detected_root')
            elif counts[i]>1:status.append('multiple_detected_roots')
            elif len(exact):roots[i]=grid[exact[0]];status.append('unique_detected')
            else:
                j=int(cross[0]);indices.append(i);lo.append(grid[j]);hi.append(grid[j+1]);status.append('unique_detected')
        def margins(t,ids):
            x=torch.cat((t[:,None],nuisance[ids]),dim=1)
            out=model(coordinates(x));return out[:,1]-out[:,0]
        if indices:
            ids=torch.tensor(indices,device=device);left=torch.stack(lo);right=torch.stack(hi);ml=margins(left,ids)
            for _ in range(bisections):
                mid=(left+right)/2;mm=margins(mid,ids);inside=ml*mm<=0
                right=torch.where(inside,mid,right);left=torch.where(inside,left,mid);ml=torch.where(inside,ml,mm)
            roots[ids]=(left+right)/2
    valid=torch.isfinite(roots);ids=torch.where(valid)[0]
    normals=torch.full((n,dim),float('nan'),dtype=dtype,device=device)
    angles=torch.full((n,),float('nan'),dtype=dtype,device=device)
    residuals=torch.full((n,),float('nan'),dtype=dtype,device=device)
    info=dict(probes=n,unique_detected=int(valid.sum()),status_counts=dict(Counter(status)),
              root_mean=None,root_std=None,target_root_rms=None,source_root_rms=None,
              target_root_mae=None,normal_angle_mean_deg=None,normal_angle_p90_deg=None,
              positive_orientation_fraction=None,plane_intercept=None,plane_slope_norm=None,
              nonplanarity_rms=None,max_root_margin_residual=None,valid_normals=0)
    fit=torch.full((dim,),float('nan'),dtype=dtype,device=device)
    if len(ids):
        x=torch.cat((roots[ids,None],nuisance[ids]),dim=1).detach().requires_grad_(True)
        with torch.enable_grad():
            out=model(coordinates(x));m=out[:,1]-out[:,0]
            g=torch.autograd.grad(m.sum(),x)[0].detach()
        norm=g.norm(dim=1);good=torch.isfinite(g).all(-1)&(norm>1e-12)
        normals[ids[good]]=g[good]/norm[good,None]
        angles[ids[good]]=torch.acos(normals[ids[good],0].clamp(-1,1))*180/math.pi
        residuals[ids]=m.detach().abs();values=roots[ids]
        info.update(root_mean=float(values.mean()),root_std=float(values.std(unbiased=False)),
                    target_root_rms=float(values.square().mean().sqrt()),source_root_rms=float((values-.5).square().mean().sqrt()),
                    target_root_mae=float(values.abs().mean()),max_root_margin_residual=float(residuals[ids].max()),
                    valid_normals=int(good.sum()))
        if good.any():
            theta=angles[ids[good]]
            info.update(normal_angle_mean_deg=float(theta.mean()),normal_angle_p90_deg=float(torch.quantile(theta,.9)),
                        positive_orientation_fraction=float((g[good,0]>0).double().mean()))
        if len(ids)>=dim:
            design=torch.cat((torch.ones(len(ids),1,dtype=dtype,device=device),nuisance[ids]),dim=1)
            solution=torch.linalg.lstsq(design,values,driver='gelsd')
            if int(solution.rank)==dim:
                fit=solution.solution
                info.update(plane_intercept=float(fit[0]),plane_slope_norm=float(fit[1:].norm()),
                            nonplanarity_rms=float((values-design@fit).square().mean().sqrt()))
    arrays=dict(roots=roots,normals=normals,angles_deg=angles,root_margin_residual=residuals,
                detected_root_counts=counts,status=status,plane_coefficients=fit)
    return info,arrays
