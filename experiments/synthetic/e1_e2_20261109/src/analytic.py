"""E1 closed-form risks and a restricted scalar source/density conflict probe."""
import math
import numpy as np
import torch
from torch import nn
from tsba.boundary import BoundaryBatch
from experiments.synthetic.e1_e2_20261109.src.distributions import GaussianMixtureScore
from experiments.common.geometry import normal_roots


def cdf(x):
    return .5*(1+math.erf(x/math.sqrt(2)))


def mixture_risk(b):
    return .5*(1-cdf((b+1)/.5))+.5*cdf((b-1)/.5)


class Threshold(nn.Module):
    def __init__(self, b):
        super().__init__(); self.b=b
    def forward(self,x):
        m=3*(x[:,0]-self.b)
        return torch.stack([m/2,-m/2],-1)


def run_analytic():
    sigma=.25; tau=.1
    mixture=GaussianMixtureScore(torch.tensor([[-1.],[1.]],dtype=torch.float64),torch.tensor([[.25]],dtype=torch.float64),sigma)
    normal=GaussianMixtureScore(torch.tensor([[0.]],dtype=torch.float64),torch.tensor([[1.]],dtype=torch.float64),sigma)
    rows=[]
    for case, b, score, risk in [('compatible',-.5,mixture,mixture_risk),
                                ('high_density_correct_boundary',.25,normal,lambda t:abs(cdf(t)-.5)),
                                ('semantics_permutation',-.5,mixture,lambda t:1-mixture_risk(t))]:
        x=torch.tensor([[b]],dtype=torch.float64)
        s=float(score(x)[0,0])
        shift=-max(-.1*sigma,min(.1*sigma,tau*sigma*sigma*s))
        for alpha in [1.,.5,.25,.125,.0625]:
            moved=b+alpha*shift
            delta_density=float(score.log_prob(torch.tensor([[moved]],dtype=torch.float64))-score.log_prob(x))
            delta_risk=risk(moved)-risk(b)
            rows.append(dict(case=case,alpha=alpha,old_boundary=b,new_boundary=moved,
                             score=s,delta_log_density=delta_density,risk_before=risk(b),risk_after=risk(moved),delta_risk=delta_risk))
            assert delta_density<0
            assert delta_risk<0 if case=='compatible' else delta_risk>0
    # Deliberately nonzero old root residual: observer must subtract it.
    point=torch.tensor([[.5000001]],dtype=torch.float64)
    batch=BoundaryBatch(point,torch.tensor([[0,1]]),torch.tensor([3.],dtype=torch.float64),
                        torch.tensor([3e-7],dtype=torch.float64),torch.tensor([-.01],dtype=torch.float64),
                        torch.ones_like(point),{})
    old,_=normal_roots(Threshold(.5),batch,sigma)
    new,_=normal_roots(Threshold(.51),batch,sigma)
    linear_error=abs(float(new-old)-.01)
    assert linear_error<=1e-8
    # Fixed slope, source population. Stationarity here does not describe
    # the nonlinear TSBA head, changing anchor sets, skips or rejections.
    nodes, weights=np.polynomial.hermite.hermgauss(128)
    x=np.concatenate([-.5+math.sqrt(2)*.5*nodes,1.5+math.sqrt(2)*.5*nodes])
    y=np.concatenate([np.zeros(128),np.ones(128)])
    w=np.tile(weights/math.sqrt(math.pi)/2,2)
    def g_source(b):
        p=1/(1+np.exp(-8*(x-b)))
        return float(np.sum(w*8*(y-p)))
    def balance(b):
        score=float(mixture(torch.tensor([[b]],dtype=torch.float64))[0,0])
        return g_source(b)+tau*sigma*sigma*score
    grid=np.linspace(0,.5,1001)
    brackets=[(float(a),float(b)) for a,b in zip(grid[:-1],grid[1:]) if balance(a)*balance(b)<0]
    roots=[]
    for lo,hi in brackets:
        for _ in range(60):
            mid=(lo+hi)/2
            if balance(lo)*balance(mid)<=0:hi=mid
            else:lo=mid
        b=(lo+hi)/2
        roots.append(dict(b=b,source_gradient=g_source(b),balance=balance(b),target_risk=mixture_risk(b)))
    return dict(cases=rows,linear_geometry_error=linear_error,
                scalar_conflict=dict(sigma=sigma,source_boundary=.5,target_valley=0.,roots=roots,
                                     risk_at_source_boundary=mixture_risk(.5),pure_translation_30_step_cap=30*.1*sigma,
                                     scope='fixed-slope scalar population probe; not a general bound or TSBA equilibrium theorem'))
