"""Known-truth graph generators and stopped residual decomposition diagnostics."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data, Batch

from .p0_diagnostics import normalized,cosine


@dataclass
class SyntheticPool:
    graphs: list
    eta: torch.Tensor
    labels: torch.Tensor
    latents: list
    input_seed: int

    def batch(self,ids):
        return Batch.from_data_list([self.graphs[i] for i in ids])


def graph_pool(count,seed,*,target=False,scenario='S1'):
    g=torch.Generator().manual_seed(seed)
    sizes=torch.tensor([18,24,30] if target else [12,18,24])
    ns=sizes[torch.randint(3,(count,),generator=g)]
    ts=torch.randint(2,(count,),generator=g)
    noise=torch.rand(count,generator=g,dtype=torch.float64)*.6-.3
    eta=torch.sigmoid(4*(2*ts.double()-1))
    if target and scenario=='S3':eta=1-eta
    labels=torch.bernoulli(eta,generator=torch.Generator().manual_seed(seed+1000)).long()
    graphs=[];latents=[]
    for n,t,eps in zip(ns.tolist(),ts.tolist(),noise.tolist()):
        u=(2*t-1)+eps
        if target and scenario=='S2':u=-(2*t-1)+eps
        if target and scenario=='S4':u+=10
        edges=[]
        for i in range(n):edges.extend([(i,(i+1)%n),((i+1)%n,i)])
        if t:edges.extend([(0,2),(2,0)])
        ei=torch.tensor(edges,dtype=torch.long).mT.contiguous()
        graphs.append(Data(x=torch.tensor([[1.,u]]*n,dtype=torch.float64),edge_index=ei,
                           edge_attr=torch.ones(len(edges),1,dtype=torch.float64)))
        latents.append(dict(n=n,t=t,noise=eps,u=u))
    return SyntheticPool(graphs,eta,labels,latents,seed)


class LinearBinary(nn.Module):
    def __init__(self,theta):
        super().__init__();self.theta=nn.Parameter(torch.tensor(theta,dtype=torch.float64));self.encoder=nn.Identity()
    def forward(self,x):
        z=x@self.theta
        return torch.stack((-z/2,z/2),dim=-1)


def linear_pool(count,seed,*,target=False):
    g=torch.Generator().manual_seed(seed)
    x=torch.randn(count,2,generator=g,dtype=torch.float64)*torch.tensor([2.,.5] if target else [.5,2.],dtype=torch.float64)
    eta=torch.sigmoid(x[:,0]-x[:,1])
    y=torch.bernoulli(eta,generator=torch.Generator().manual_seed(seed+1000)).long()
    return x,eta,y


def split_residual(probabilities,labels,contrasts):
    """Center in query-graph space, retaining per-graph model predictions."""
    residual=(F.one_hot(labels,probabilities.shape[1]).to(probabilities)-probabilities)@contrasts
    average=residual.mean(0,keepdim=True).expand_as(residual)
    return residual.reshape(-1),average.reshape(-1),(residual-average).reshape(-1)


def soft_residual(probabilities,eta,contrasts):
    soft=torch.stack((1-eta,eta),dim=-1)
    return ((soft-probabilities)@contrasts).reshape(-1)


def soft_gradient(snapshot,batches,eta):
    result=torch.zeros_like(snapshot.flat);offset=0
    for batch in batches:
        count=batch.num_graphs if hasattr(batch,'num_graphs') else len(batch)
        variable=snapshot.flat.detach().requires_grad_(True)
        logits=snapshot.logits(batch,variable)
        probabilities=torch.stack((1-eta[offset:offset+count],eta[offset:offset+count]),dim=-1)
        loss=-(probabilities*F.log_softmax(logits,dim=-1)).sum()/len(eta)
        gradient,=torch.autograd.grad(loss,variable,allow_unused=True)
        if gradient is not None:result+=gradient.detach()
        offset+=count
    if offset!=len(eta) or not torch.isfinite(result).all():raise ValueError('Invalid soft gradient')
    return result


@torch.no_grad()
def soft_ce(snapshot,flat,batches,eta):
    total=0.;offset=0
    for batch in batches:
        logits=snapshot.logits(batch,flat);count=len(logits)
        probabilities=torch.stack((1-eta[offset:offset+count],eta[offset:offset+count]),dim=-1)
        total+=float(-(probabilities*F.log_softmax(logits,dim=-1)).sum())
        offset+=count
    if offset!=len(eta):raise ValueError('Soft CE count mismatch')
    return total/len(eta)


def decomposition_vectors(geo,labels,block_seed,permutation_offsets,random_offsets,epsilon):
    r,average,centered=split_residual(geo.source_probabilities,labels,geo.contrasts)
    apply=lambda v:-(geo.basis@(geo.readout@v))
    vectors={'FULL':apply(r),'GLOBAL':apply(average),'CENTERED':apply(centered)}
    reference=float(vectors['CENTERED'].norm())
    valid={k:float(v.norm())>epsilon for k,v in vectors.items()}
    for rep,offset in enumerate(permutation_offsets,1):
        indices=torch.randperm(len(labels),generator=torch.Generator().manual_seed(block_seed+offset))
        _,pmean,pc=split_residual(geo.source_probabilities,labels[indices],geo.contrasts)
        torch.testing.assert_close(pmean,average,atol=1e-14,rtol=1e-12)
        vectors[f'PC-{rep}']=apply(pc)
        valid[f'PC-{rep}']=float(vectors[f'PC-{rep}'].norm())>epsilon
    for rep,offset in enumerate(random_offsets,1):
        u=torch.randn(geo.basis.shape[1],generator=torch.Generator().manual_seed(block_seed+offset),dtype=torch.float64)
        vectors[f'RC-{rep}'],valid[f'RC-{rep}']=normalized(geo.basis@u,reference,epsilon)
    error=float((vectors['FULL']-vectors['GLOBAL']-vectors['CENTERED']).norm())
    tolerance=1e-10+1e-10*float(vectors['FULL'].norm())
    if error>tolerance:raise ValueError('Residual additivity failed')
    return vectors,valid,dict(additivity_absolute_error=error,
        full_norm=float(vectors['FULL'].norm()),global_norm=float(vectors['GLOBAL'].norm()),centered_norm=reference,
        global_centered_cosine=cosine(vectors['GLOBAL'],vectors['CENTERED'],epsilon),
        global_centered_cross_term=2*float(torch.dot(vectors['GLOBAL'],vectors['CENTERED'])),
        residual_mean_norm=float(average.reshape(len(labels),-1)[0].norm()),
        residual_centered_norm=float(centered.norm()))


def decomposition_steps(vectors,g_source,g_oracle,*,beta,epsilon):
    full,global_,center=vectors['FULL'],vectors['GLOBAL'],vectors['CENTERED']
    norm=float(full.norm());cnorm=float(center.norm())
    gm,vg=normalized(global_,norm,epsilon);cm,vc=normalized(center,norm,epsilon)
    om,vo=normalized(g_oracle,norm,epsilon)
    arms={'S':g_source,'S+FULL':g_source+beta*full,'S+GLOBAL':g_source+beta*global_,
        'S+CENTERED':g_source+beta*center,'S+GLOBAL-M':g_source+beta*gm,
        'S+CENTERED-M':g_source+beta*cm,'S+ORACLE-M':g_source+beta*om}
    valid={k:True for k in arms};valid.update({'S+GLOBAL-M':vg,'S+CENTERED-M':vc,'S+ORACLE-M':vo})
    for kind in ['PC','RC']:
        for rep in [1,2,3]:
            v,ok=normalized(vectors[f'{kind}-{rep}'],cnorm,epsilon)
            name=f'S+GLOBAL+{kind}-{rep}'
            arms[name]=g_source+beta*(global_+v);valid[name]=ok and cnorm>epsilon
    return arms,valid
