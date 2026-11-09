"""Paired measurements at identical queries; no training decisions here."""
import torch
from torch.nn import functional as F
from .geometry import rms


@torch.no_grad()
def score_audit(points, normals, scores, sigma):
    if not len(points):
        return dict(count=0, comparisons={}), {}
    values = {name: model(points).detach() for name, model in scores.items()}
    if not all(torch.isfinite(v).all() for v in values.values()):
        raise ValueError('nonfinite score observer output')
    comparisons = {}
    for estimate, reference in [('learned','empirical'),('empirical','population'),
                                ('learned','population'),('zero','population')]:
        error = sigma * (values[estimate]-values[reference])
        row = dict(scaled_vector_mse=float(error.square().mean()))
        if normals is not None:
            a = sigma * (values[estimate]*normals).sum(-1)
            r = sigma * (values[reference]*normals).sum(-1)
            mask = r.abs() > 1e-3
            row.update(normal_rmse=rms(a-r), reference_nonstationary=int(mask.sum()),
                       excluded_fraction=1-float(mask.double().mean()),
                       sign_agreement=float((a[mask].sign()==r[mask].sign()).double().mean()) if mask.any() else None)
            recalls=[]
            for sign, name in [(1,'positive'),(-1,'negative')]:
                eligible = mask & (r.sign() == sign)
                value = float((a[eligible].sign()==sign).double().mean()) if eligible.any() else None
                row[name+'_count']=int(eligible.sum());row[name+'_recall']=value
                recalls.append(value)
            row['balanced_sign_agreement'] = sum(recalls)/2 if all(v is not None for v in recalls) else None
        comparisons[estimate+'_vs_'+reference]=row
    arrays = {'score_'+k:v for k,v in values.items()}
    return dict(count=len(points),comparisons=comparisons), arrays


def classification(logits, labels):
    prediction=logits.argmax(-1)
    confusion=torch.bincount(2*labels+prediction,minlength=4).reshape(2,2)
    counts=confusion.sum(1)
    recalls=confusion.diag()/counts
    return dict(accuracy=float((prediction==labels).double().mean()),
                balanced_accuracy=float(recalls.mean()) if (counts>0).all() else None,
                nll=float(F.cross_entropy(logits, labels)),
                per_class_recall=recalls.tolist(),confusion_matrix=confusion.tolist(),
                predicted_proportions=(torch.bincount(prediction,minlength=2)/len(labels)).tolist())
