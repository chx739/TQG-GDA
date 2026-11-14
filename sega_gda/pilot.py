"""Real-task initialization, behind an explicit input protocol. No target evaluator."""
import hashlib
from types import SimpleNamespace
import torch
from torch_geometric.loader import DataLoader
from .real_data import RealTaskGraphs
from .train_source import load_source_checkpoint
from .initialization import InitializedClassifier,fixed_scale_pairs,IdentityGeometry,verify_conversion
from .metric import LocalMetricField,fit_fixed_coverage
from .geometry_head import RiemannianClassifier
from .relations import encode_graph_ids,build_source_memory,build_target_memory,RelationRules
from .evolution import EvolutionTrainer
from .data import sha256


def verify_geometric_handoff(source,model,loader,device):
    # Reuse S1's established coordinate/probability/margin checks. Only adapt
    # the verification interface: the real head is model.logits, not .head.
    view=SimpleNamespace(eval=model.eval,encoder=model.encoder,head=model.logits)
    return verify_conversion(source,view,loader,device)


class ExactInitialAssetDistance:
    """Exact D0 specialization for fixed-ID memory construction ONLY.

    At B=0 the S2 path energy is exactly squared Euclidean distance for every
    L/Q/J. No derivative is requested during asset construction. Training,
    OT, classification and all space steps keep the full S2 engine.
    """
    def __init__(self,distance):
        self.solver=distance.solver
        self.identity=IdentityGeometry(self.solver.metric.dim)
        pairs=distance.reference_pairs.detach().cpu()
        self.scale2=self.identity.squared_distance(pairs[:,0],pairs[:,1]).mean()
        if not torch.isfinite(self.scale2) or self.scale2<=0: raise FloatingPointError('Invalid initial asset scale')

    @torch.no_grad()
    def __call__(self,x,y):
        if bool((self.solver.metric.B.detach()!=0).any()):
            raise ValueError('Exact initial asset specialization requires B=0')
        return self.identity.squared_distance(x,y)/self.scale2.to(x)


@torch.no_grad()
def coordinates(ids,lookup,encoder,batch_size=32):
    return torch.cat([encode_graph_ids(ids[i:i+batch_size],lookup,encoder).cpu()
                      for i in range(0,len(ids),batch_size)])


def initialize_real_task(checkpoint,task_manifest,config,device):
    source,state=load_source_checkpoint(checkpoint,device)
    source.eval()
    data=RealTaskGraphs(task_manifest,state['preprocessing'])
    if state['identity']['source_ids']!=data.source.manifest:
        raise ValueError('Source checkpoint and task use different source splits')
    sids,tids=tuple(data.source_ids),tuple(data.target_ids)
    us=coordinates(sids,data.graph,source.encoder)
    ut=coordinates(tids,data.graph,source.encoder)
    pairs,pair_ids,scale2=fixed_scale_pairs(us,sids,config['scale_pairs'],config['scale_seed'])
    centers,h,coverage=fit_fixed_coverage(us,ut,config['coverage_centers'],config['bandwidth_floor'],config['coverage_seed'])
    metric=LocalMetricField(centers,h,config['spectral_bound'])
    converted=InitializedClassifier(source,scale2.to(device),config['temperature'],config['coverage_centers'])
    model=RiemannianClassifier.from_s1(converted,pairs,metric,**config['solver']).to(device).eval()
    initial_distance=ExactInitialAssetDistance(model.distance)
    # Check the exact asset specialization against the ACTUAL path engine.
    with torch.no_grad():
        exact=initial_distance(us[:4],us[4:8])
        actual=model.distance(us[:4].to(device),us[4:8].to(device)).cpu()
    torch.testing.assert_close(actual,exact,atol=1e-10,rtol=1e-8)
    rules=RelationRules(**config['relation_rules'])
    provenance=dict(initial_encoder_sha256=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in source.encoder.state_dict().values())).hexdigest(),
                    source_checkpoint_sha256=sha256(checkpoint),task_manifest_sha256=sha256(task_manifest),
                    initial_distance_implementation='exact_B0_specialization_checked_against_S2')
    sm=build_source_memory(sids,us,data.source_labels,initial_distance,rules,config['source_relation_seed'],provenance)
    tm=build_target_memory(tids,ut,initial_distance,rules,config['target_relation_seed'],provenance)
    # Source validation participates ONLY in this frozen equivalence/eval check.
    graphs=data.source.materialize(data.source.train_ids+data.source.val_ids,state['preprocessing'])
    proof=verify_geometric_handoff(source,model,DataLoader(graphs,batch_size=16,shuffle=False,
        generator=torch.Generator().manual_seed(804)),device)
    identity=dict(task=data.manifest,source_checkpoint_sha256=sha256(checkpoint),scale_pair_ids=pair_ids,
                  coverage=coverage,initial_scale2=float(scale2),target_label_access=False)
    engine=EvolutionTrainer(model,data.graph,data.source_labels,tids,sm,tm,config,identity)
    engine.initialize_correspondence()
    return engine,data,dict(equivalence=proof,coverage=coverage,scale_pair_ids=pair_ids,
                            scale2=float(scale2),source_relations=len(sm.triplets),target_relations=len(tm.triplets),
                            exact_asset_distance_max_error=float((exact-actual).abs().max()))
