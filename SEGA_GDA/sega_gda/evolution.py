"""§3.9 content/space block updates, exact dual acceptance, round-boundary recovery."""
import copy
from dataclasses import dataclass
import math
import random
import torch
from torch.nn import functional as F

from .relations import encode_graph_ids, relation_loss_from_coordinates, RelationMemory
from .transport import (sample_source_references,class_balanced_masses,uniform_target_masses,
                        solve_semi_relaxed,fixed_plan_objective,OTTolerances)
from .objectives import alignment_loss,learning_objective
from .train_source import rng_state,restore_rng,atomic_save


@dataclass(frozen=True)
class EvolutionBatch:
    query: tuple
    reference: tuple
    target: tuple


@dataclass(frozen=True)
class FrozenFeedback:
    batch: EvolutionBatch
    coordinates: dict
    q: torch.Tensor
    beta: torch.Tensor
    plan: object
    a: torch.Tensor
    b: torch.Tensor
    pairs: dict
    content_version: int
    geometry_version: int
    solver_config: dict
    memories: tuple=None  # (source, target) relation views used by this space step


def acceptance_conditions(phi_old,phi_new,delta,delta_space,c_A,alpha,step_squared):
    """No acceptance tolerance: use both inequalities exactly as specified."""
    rhs=phi_old-(c_A/alpha)*step_squared
    return delta<=delta_space**2,phi_new<=rhs,rhs


def trust_change(previous,current,weights):
    """Mean SQUARED change within each group, then the fixed group combination."""
    groups={k:float((current[k]-previous[k]).square().mean()) for k in ('ss','tt','st')}
    return sum(weights[k]*groups[k] for k in groups),groups


def _unique_pairs(ids,count,generator,cross=None):
    """Sample pair ranks without constructing an N² candidate list."""
    ids=sorted(ids)
    n=len(ids)
    total=n*len(cross) if cross is not None else n*(n-1)//2
    ranks=generator.sample(range(total),min(count,total))
    pairs=[]
    for rank in ranks:
        if cross is not None:
            i,j=divmod(rank,len(cross));pairs.append((ids[i],sorted(cross)[j]))
        else:
            lo,hi=0,n-2
            while lo<hi:
                mid=(lo+hi+1)//2
                if mid*(2*n-mid-1)//2<=rank: lo=mid
                else: hi=mid-1
            i=lo;j=i+1+rank-i*(2*n-i-1)//2
            pairs.append((ids[i],ids[j]))
    return tuple(pairs)


class EvolutionTrainer:
    """Current GIN/GINE has no BN/dropout. Deterministic encoding in both blocks.

    Graph lookup owns input access control; source_labels is SOURCE TRAIN only.
    No distance/path/scale cache is retained. Pi records are diagnostic/checkpoint
    state tagged with content/geometry versions, never reused by the next step.
    """
    def __init__(self,model,graph_lookup,source_labels,target_ids,source_memory,target_memory,config,data_identity):
        self.model=model
        self.lookup=graph_lookup
        self.source_labels=dict(source_labels)
        self.target_ids=tuple(target_ids)
        self.source_memory=source_memory
        self.target_memory=target_memory
        self.config=copy.deepcopy(config)
        self.data_identity=copy.deepcopy(data_identity)
        self.metric=model.distance.solver.metric
        self.content_parameters=[*model.encoder.parameters(),model.q,model.beta]
        self._validate()
        self.optimizer=torch.optim.Adam(self.content_parameters,lr=config['content_lr'],weight_decay=0.)
        self.tolerances=OTTolerances(**config['ot_tolerances'])
        self.batch_rng=torch.Generator().manual_seed(config['batch_seed'])
        self.probe_rng=random.Random(config['probe_seed'])
        self.round=0;self.content_step_count=0;self.content_version=0;self.geometry_version=0
        self.ot_solve_count=0;self.pi_state=None;self.phase='boundary'
        self._flags(False)

    def _validate(self):
        c=self.config
        if set(self.source_labels.values())!=set(range(len(self.model.q))):
            raise ValueError('Source train labels must use the existing contiguous class mapping')
        if (not self.target_ids or len(set(self.target_ids))!=len(self.target_ids)
                or set(self.target_ids)&set(self.source_labels)):
            raise ValueError('Need distinct allowed target IDs and disjoint domain ID namespaces')
        if self.source_memory.domain!='source' or self.target_memory.domain!='target':
            raise ValueError('Memory domains reversed')
        for memory,allowed in ((self.source_memory,set(self.source_labels)),(self.target_memory,set(self.target_ids))):
            if any(g not in allowed for t in memory.triplets for g in t):
                raise PermissionError('Relation ID outside allowed input pool')
        for key in ('content_steps','query_count','target_count','references_per_class','backtrack_attempts'):
            if not isinstance(c[key],int) or c[key]<1: raise ValueError(f'Invalid {key}')
        for key in ('content_lr','space_alpha','delta_space'):
            if not math.isfinite(c[key]) or c[key]<=0: raise ValueError(f'Invalid {key}')
        if not 0<c['c_A']<.5 or not 0<c['backtrack_factor']<1:
            raise ValueError('Require 0<c_A<1/2 and 0<backtrack_factor<1')
        weights=c['trust_weights']
        if set(weights)!=set(('ss','tt','st')) or any(not math.isfinite(v) or v<=0 for v in weights.values()) or abs(sum(weights.values())-1)>1e-12:
            raise ValueError('Three positive fixed trust weights must sum to one')
        if any(not isinstance(c['trust_pairs'][k],int) or c['trust_pairs'][k]<1 for k in weights):
            raise ValueError('Positive trust pair budgets required')
        budget=c.get('relation_triplets_per_step')
        if budget is not None and (isinstance(budget,bool) or not isinstance(budget,int) or budget<1):
            raise ValueError('relation_triplets_per_step must be null (full fixed memory) or a positive integer')
        if any(isinstance(m,(torch.nn.modules.batchnorm._BatchNorm,torch.nn.modules.dropout._DropoutNd)) for m in self.model.encoder.modules()):
            raise ValueError('This implementation supports the existing deterministic GIN/GINE; BN/dropout policy needs a separate decision')
        learning_objective(torch.tensor(0.),torch.tensor(0.),torch.tensor(0.),c['alignment_weight'],c['relation_weight'])

    def _flags(self,space):
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        for p in self.content_parameters: p.requires_grad_(not space)
        self.metric.B.requires_grad_(space)

    def make_batch(self):
        def sample(ids,count):
            ids=sorted(ids)
            return tuple(ids[i] for i in torch.randperm(len(ids),generator=self.batch_rng)[:count].tolist())
        seed=int(torch.randint(2**31,(1,),generator=self.batch_rng))
        refs,_=sample_source_references(self.source_labels,self.config['references_per_class'],seed)
        return EvolutionBatch(sample(self.source_labels,self.config['query_count']),refs,
                              sample(self.target_ids,self.config['target_count']))

    def _validate_batch(self,batch):
        for ids,allowed in ((batch.query,self.source_labels),(batch.reference,self.source_labels),(batch.target,self.target_ids)):
            if not ids or len(set(ids))!=len(ids) or any(g not in allowed for g in ids):
                raise PermissionError('Invalid/unauthorized batch IDs')
        class_balanced_masses(batch.reference,[self.source_labels[g] for g in batch.reference],list(range(len(self.model.q))))

    def _relation_views(self,generator_factory):
        """Relation memories used by one step: the FULL fixed memories by default.

        `relation_triplets_per_step` (absent/null = design default) optionally
        draws that many triplets per domain without replacement from the fixed
        memory, using a generator from `generator_factory` (called only when the
        option is active, so default RNG consumption is unchanged). Identities,
        ranking rules and the per-domain mean are untouched; the per-step
        reduction then becomes a uniform minibatch estimate of the full mean.
        """
        budget=self.config.get('relation_triplets_per_step')
        if budget is None: return self.source_memory,self.target_memory
        generator=generator_factory();views=[]
        for memory in (self.source_memory,self.target_memory):
            count=len(memory.triplets)
            if count<=budget: views.append(memory);continue
            chosen=sorted(torch.randperm(count,generator=generator)[:budget].tolist())
            views.append(RelationMemory(memory.domain,tuple(memory.triplets[i] for i in chosen),memory.metadata))
        return tuple(views)

    def _encode(self,batch,pairs=None,memories=None):
        self._validate_batch(batch)
        memories=memories or (self.source_memory,self.target_memory)
        ids=set(batch.query+batch.reference+batch.target)
        ids.update(g for m in memories for t in m.triplets for g in t)
        ids.update(g for group in (pairs or {}).values() for pair in group for g in pair)
        if not ids<=set(self.source_labels)|set(self.target_ids): raise PermissionError('Probe outside allowed pool')
        ids=sorted(ids)
        return dict(zip(ids,encode_graph_ids(ids,self.lookup,self.model.encoder)))

    def _masses(self,batch):
        a=class_balanced_masses(batch.reference,[self.source_labels[g] for g in batch.reference],list(range(len(self.model.q))),
                                dtype=self.model.q.dtype,device=self.model.q.device)
        b=uniform_target_masses(len(batch.target),dtype=a.dtype,device=a.device)
        return a,b

    @staticmethod
    def _stack(coordinates,ids): return torch.stack([coordinates[g] for g in ids])

    def _terms(self,coordinates,batch,alignment,q=None,beta=None,memories=None):
        source_memory,target_memory=memories or (self.source_memory,self.target_memory)
        query=self._stack(coordinates,batch.query)
        labels=torch.tensor([self.source_labels[g] for g in batch.query],device=query.device)
        ce=F.cross_entropy(self.model.logits(query,q=q,beta=beta),labels)
        rel=relation_loss_from_coordinates(source_memory,target_memory,coordinates,self.model.distance,self.config['relation_margin'])
        return dict(classification=ce,alignment=alignment,relation=rel['total'])

    def _total(self,terms):
        return learning_objective(terms['classification'],terms['alignment'],terms['relation'],
                                  self.config['alignment_weight'],self.config['relation_weight'])

    def _component_gradients(self,terms,parameters):
        result={}
        if self.config.get('gradient_diagnostics',False):
            for name,value in terms.items():
                gradients=torch.autograd.grad(value,parameters,retain_graph=True,allow_unused=True)
                if any(g is not None and not torch.isfinite(g).all() for g in gradients):
                    raise FloatingPointError('Nonfinite component gradient')
                result[name]=sum(float(g.detach().square().sum()) for g in gradients if g is not None)**.5
        return result

    def _plan_record(self,result,batch,event):
        self.ot_solve_count+=1
        return dict(result=result,batch=batch,event=event,solve_id=self.ot_solve_count,
                    content_version=self.content_version,geometry_version=self.geometry_version)

    @torch.no_grad()
    def _solve(self,coordinates,batch,event):
        a,b=self._masses(batch)
        cost=self.model.distance.pairwise(self._stack(coordinates,batch.reference),self._stack(coordinates,batch.target))
        result=solve_semi_relaxed(cost,a,b,self.config['epsilon'],self.config['rho'],self.tolerances)
        return self._plan_record(result,batch,event)

    def initialize_correspondence(self):
        with torch.no_grad():
            batch=self.make_batch();coordinates=self._encode(batch)
            self.pi_state=self._solve(coordinates,batch,'initial')

    def content_step(self,batch=None):
        self.phase='content'
        self._flags(False)
        batch=batch or self.make_batch()
        memories=self._relation_views(lambda:self.batch_rng)
        coordinates=self._encode(batch,memories=memories)
        a,b=self._masses(batch)
        align,result,_=alignment_loss(self._stack(coordinates,batch.reference),self._stack(coordinates,batch.target),
                                     self.model.distance,a,b,self.config['epsilon'],self.config['rho'],self.tolerances)
        record=self._plan_record(result,batch,'content')
        terms=self._terms(coordinates,batch,align['total'],memories=memories)
        component=self._component_gradients(terms,tuple(self.model.encoder.parameters()))
        total=self._total(terms)
        total.backward()
        if self.metric.B.grad is not None: raise RuntimeError('Content phase leaked a B gradient')
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.content_parameters):
            raise FloatingPointError('Nonfinite content gradient; optimizer not stepped')
        self.optimizer.step()
        if any(not torch.isfinite(p).all() for p in self.content_parameters):
            raise FloatingPointError('Nonfinite content parameters; abort without checkpoint')
        self.content_step_count+=1;self.content_version+=1
        # The record belongs to PRE-update coordinates and cannot be current state.
        self.pi_state=None
        return dict(step=self.content_step_count,correspondence_id=record['solve_id'],
                    correspondence_content_version=record['content_version'],content_version_after=self.content_version,
                    geometry_version=self.geometry_version,losses={k:float(v.detach()) for k,v in terms.items()},
                    total=float(total.detach()),encoder_component_gradient_norms=component,
                    pi_detached=not result.pi.requires_grad,ot=result.diagnostics,
                    relation_triplets=[len(m.triplets) for m in memories],
                    batch=dict(query=batch.query,reference=batch.reference,target=batch.target))

    def freeze_feedback(self,batch=None,pairs=None):
        self.phase='feedback';self._flags(True)
        batch=batch or self.make_batch()
        if pairs is None:
            counts=self.config['trust_pairs']
            pairs=dict(ss=_unique_pairs(self.source_labels,counts['ss'],self.probe_rng),
                       tt=_unique_pairs(self.target_ids,counts['tt'],self.probe_rng),
                       st=_unique_pairs(self.source_labels,counts['st'],self.probe_rng,self.target_ids))
        if set(pairs)!=set(('ss','tt','st')): raise ValueError('Need all three trust groups')
        for name,group in pairs.items():
            for i,j in group:
                left=self.source_labels if name!='tt' else self.target_ids
                right=self.target_ids if name!='ss' else self.source_labels
                if i not in left or j not in right or i==j: raise PermissionError('Invalid trust pair domain/identity')
        # One relation view per space step: baseline and every candidate use it.
        memories=self._relation_views(lambda:torch.Generator().manual_seed(self.probe_rng.getrandbits(62)))
        with torch.no_grad():
            coordinates={g:u.detach().clone() for g,u in self._encode(batch,pairs,memories).items()}
            plan=self._solve(coordinates,batch,'feedback')
        a,b=self._masses(batch)
        return FrozenFeedback(batch,coordinates,self.model.q.detach().clone(),self.model.beta.detach().clone(),
                              plan,a,b,{k:tuple(v) for k,v in pairs.items()},self.content_version,self.geometry_version,
                              copy.deepcopy(self.model.distance.solver.config),memories)

    def space_terms(self,feedback):
        cost=self.model.distance.pairwise(self._stack(feedback.coordinates,feedback.batch.reference),
                                          self._stack(feedback.coordinates,feedback.batch.target))
        align=fixed_plan_objective(feedback.plan['result'],cost,feedback.a,feedback.b)['total']
        return self._terms(feedback.coordinates,feedback.batch,align,feedback.q,feedback.beta,feedback.memories)

    def _trust_distances(self,feedback):
        pairs=[pair for group in feedback.pairs.values() for pair in group]
        if not pairs: return {}
        values=self.model.distance(torch.stack([feedback.coordinates[i] for i,j in pairs]),
                                    torch.stack([feedback.coordinates[j] for i,j in pairs]))
        result={};offset=0
        for key,group in feedback.pairs.items():
            result[key]=values[offset:offset+len(group)];offset+=len(group)
        return result

    def space_update(self,feedback):
        if (feedback.content_version!=self.content_version or feedback.geometry_version!=self.geometry_version
                or feedback.solver_config!=self.model.distance.solver.config):
            raise RuntimeError('Stale feedback/changed solver specification')
        if any(u.requires_grad for u in feedback.coordinates.values()) or feedback.q.requires_grad or feedback.beta.requires_grad:
            raise RuntimeError('Spatial feedback must be detached')
        if not torch.equal(feedback.q,self.model.q) or not torch.equal(feedback.beta,self.model.beta):
            raise RuntimeError('Class references changed after feedback freeze')
        self.phase='space';self._flags(True)
        old=self.metric.B.detach().clone()
        log=dict(accepted=False,attempts=[],feedback_correspondence_id=feedback.plan['solve_id'],
                 content_version=self.content_version,geometry_version_before=self.geometry_version,
                 pairs=feedback.pairs,trust_weights=copy.deepcopy(self.config['trust_weights']),
                 spectral_bound=self.metric.spectral_bound,
                 relation_triplets=[len(m.triplets) for m in (feedback.memories or (self.source_memory,self.target_memory))])
        completed=False
        try:
            if any(not v for v in feedback.pairs.values()):
                log['reason']='INSUFFICIENT_PROBE_PAIRS';completed=True
                return log
            terms=self.space_terms(feedback)
            phi=self._total(terms)
            # The gradient of the three-term SUM, accumulated in one explicit
            # order. Each term still fully differentiates paths and scale.
            # This avoids scheduler-dependent accumulation into the shared B
            # leaf across independent loss branches on CUDA.
            names=('classification','alignment','relation')
            gradients=[torch.autograd.grad(terms[name],self.metric.B,retain_graph=i<2)[0]
                       for i,name in enumerate(names)]
            gradient=gradients[0]+self.config['alignment_weight']*gradients[1]+self.config['relation_weight']*gradients[2]
            component={name:float(g.norm()) for name,g in zip(names,gradients)} if self.config.get('gradient_diagnostics',False) else {}
            if not torch.isfinite(gradient).all(): raise FloatingPointError('Invalid baseline spatial gradient')
            old_phi=float(phi.detach())
            with torch.no_grad():
                baseline={k:v.detach().clone() for k,v in self._trust_distances(feedback).items()}
                old_logits=self.model.logits(self._stack(feedback.coordinates,feedback.batch.query),q=feedback.q,beta=feedback.beta)
                old_scale=float(self.model.distance.scale2())
            log.update(phi_before=old_phi,component_B_gradient_norms=component,gradient_norm=float(gradient.norm()),scale2_before=old_scale)
            alpha=self.config['space_alpha']
            for attempt in range(self.config['backtrack_attempts']):
                raw=old-alpha*gradient
                entry=dict(attempt=attempt+1,alpha=alpha)
                try:
                    self.metric.project_(raw)
                    self.metric.validate()
                    difference=self.metric.B.detach()-old
                    step_squared=float(difference.square().sum())
                    with torch.no_grad():
                        new_phi=float(self._total(self.space_terms(feedback)))
                        current=self._trust_distances(feedback)
                        delta,groups=trust_change(baseline,current,self.config['trust_weights'])
                        new_scale=float(self.model.distance.scale2())
                    if not math.isfinite(new_phi) or not math.isfinite(delta): raise FloatingPointError('Nonfinite candidate')
                    trust_ok,armijo_ok,rhs=acceptance_conditions(old_phi,new_phi,delta,self.config['delta_space'],
                                                                self.config['c_A'],alpha,step_squared)
                    entry.update(phi=new_phi,delta=delta,delta_groups=groups,scale2=new_scale,
                                 trust_ok=trust_ok,armijo_ok=armijo_ok,armijo_rhs=rhs,step_squared=step_squared,
                                 raw_step_norm=float((raw-old).norm()),projected_step_norm=step_squared**.5,
                                 projection_correction_norm=float((self.metric.B.detach()-raw).norm()),
                                 max_trace_error=float(self.metric.B.detach().diagonal(dim1=-2,dim2=-1).sum(-1).abs().max()),
                                 max_abs_eigenvalue=float(torch.linalg.eigvalsh(self.metric.B.detach()).abs().max()),
                                 reason='ACCEPTED' if trust_ok and armijo_ok else
                                     '+'.join(k for k,ok in [('TRUST_REGION',trust_ok),('ARMIJO',armijo_ok)] if not ok))
                    if trust_ok and armijo_ok:
                        log['accepted']=True
                except (FloatingPointError,ValueError) as error:
                    entry.update(reason='INVALID_CANDIDATE',error=str(error))
                log['attempts'].append(entry)
                if log['accepted']: break
                with torch.no_grad(): self.metric.B.copy_(old)
                alpha*=self.config['backtrack_factor']
            if not log['accepted']:
                with torch.no_grad(): self.metric.B.copy_(old)
            changed=not torch.equal(self.metric.B,old)
            if changed: self.geometry_version+=1
            with torch.no_grad():
                logits=self.model.logits(self._stack(feedback.coordinates,feedback.batch.query),q=feedback.q,beta=feedback.beta)
            log.update(reason='ACCEPTED' if log['accepted'] else 'ALL_CANDIDATES_REJECTED',
                       geometry_changed=changed,geometry_version_after=self.geometry_version,
                       final_alpha=log['attempts'][-1]['alpha'],
                       phi_after=log['attempts'][-1]['phi'] if log['accepted'] else old_phi,
                       classification_logit_max_change=float((logits-old_logits).abs().max()))
            completed=True
            return log
        finally:
            if not completed:
                with torch.no_grad(): self.metric.B.copy_(old)
            self._flags(False)

    def run_round(self):
        if self.phase!='boundary': raise RuntimeError('Start a round only at a complete boundary')
        content=[self.content_step() for _ in range(self.config['content_steps'])]
        feedback=self.freeze_feedback()
        space=self.space_update(feedback)
        with torch.no_grad():
            self.pi_state=self._solve(feedback.coordinates,feedback.batch,'next')
        pi_change=float((self.pi_state['result'].pi-feedback.plan['result'].pi).abs().max())
        self.round+=1;self.phase='boundary'
        return dict(round=self.round,content=content,space=space,next_correspondence_id=self.pi_state['solve_id'],
                    next_pi_max_change=pi_change,next_ot=self.pi_state['result'].diagnostics,
                    content_version=self.content_version,geometry_version=self.geometry_version)

    def save_checkpoint(self,path):
        if self.phase!='boundary' or self.pi_state is None: raise RuntimeError('Checkpoint only at a complete round boundary with current Pi')
        atomic_save(path,dict(kind='sega_s4_evolution',schema_version=1,model=self.model.state_dict(),
            encoder_spec=self.model.encoder.spec,solver=self.model.distance.solver.config,spectral_bound=self.metric.spectral_bound,
            optimizer=self.optimizer.state_dict(),config=self.config,source_labels=self.source_labels,target_ids=self.target_ids,
            source_memory=self.source_memory.to_dict(),target_memory=self.target_memory.to_dict(),data_identity=self.data_identity,
            round=self.round,content_steps=self.content_step_count,content_version=self.content_version,
            geometry_version=self.geometry_version,ot_solve_count=self.ot_solve_count,pi_state=self.pi_state,
            batch_rng=self.batch_rng.get_state(),probe_rng=self.probe_rng.getstate(),rng=rng_state(),
            device=str(self.model.q.device),phase='boundary'))

    @classmethod
    def load_checkpoint(cls,path,graph_lookup,data_identity,device='cpu',expected_config=None):
        from .models import GraphEncoder
        from .metric import LocalMetricField
        from .geodesic import GeodesicSolver,NormalizedGeodesicDistance
        from .geometry_head import RiemannianClassifier
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state['kind']!='sega_s4_evolution' or state['schema_version']!=1 or state['phase']!='boundary':
            raise ValueError('Unsupported evolution checkpoint/boundary')
        if state['data_identity']!=data_identity: raise ValueError('Input identity changed')
        if expected_config is not None and state['config']!=expected_config: raise ValueError('Training rules changed')
        if state['device']!=str(torch.device(device)): raise ValueError('Exact recovery requires the same device type/index')
        saved=state['model']
        field=LocalMetricField(saved['distance.solver.metric.centers'],saved['distance.solver.metric.bandwidths'],state['spectral_bound'])
        distance=NormalizedGeodesicDistance(GeodesicSolver(field,**state['solver']),saved['distance.reference_pairs'])
        model=RiemannianClassifier(GraphEncoder(**state['encoder_spec']).to(saved['q']),saved['q'],saved['beta'],saved['temperature'],distance).to(device)
        model.load_state_dict(saved)
        engine=cls(model,graph_lookup,state['source_labels'],state['target_ids'],RelationMemory.from_dict(state['source_memory']),
                   RelationMemory.from_dict(state['target_memory']),state['config'],data_identity)
        engine.optimizer.load_state_dict(state['optimizer'])
        engine.round=state['round'];engine.content_step_count=state['content_steps'];engine.content_version=state['content_version']
        engine.geometry_version=state['geometry_version'];engine.ot_solve_count=state['ot_solve_count'];engine.pi_state=state['pi_state']
        # Plans are checkpointed on CPU; only their tensors require device mapping.
        plan=engine.pi_state['result'];plan.pi=plan.pi.to(device);plan.log_pi=plan.log_pi.to(device)
        engine.batch_rng.set_state(state['batch_rng']);engine.probe_rng.setstate(state['probe_rng'])
        restore_rng(state['rng'])
        return engine
