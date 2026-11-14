"""Reconstruct tiny synthetic test assets in memory; no historical run required."""
import hashlib
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
from test_s3 import graph_fixture,encoder_hash
from sega_gda.models import SourceClassifier
from sega_gda.initialization import InitializedClassifier,fixed_scale_pairs
from sega_gda.metric import LocalMetricField,fit_fixed_coverage
from sega_gda.geometry_head import RiemannianClassifier
from sega_gda.relations import RelationRules,encode_graph_ids,build_source_memory,build_target_memory
from sega_gda.evolution import EvolutionTrainer


def make_fixture(config=None,device='cpu',initialize=True):
    config=json.loads((ROOT/'configs/s4_evolution_debug.json').read_text()) if config is None else config
    assets=json.loads((ROOT/'configs/s3_ot_relation_debug.json').read_text())
    graphs,labels,encoder=graph_fixture()
    source_ids,target_ids=tuple(labels),tuple(g for g in graphs if g.startswith('t:'))
    with torch.no_grad():
        us=encode_graph_ids(source_ids,graphs.__getitem__,encoder)
        ut=encode_graph_ids(target_ids,graphs.__getitem__,encoder)
    pairs,_,scale=fixed_scale_pairs(us,source_ids,assets['scale_pairs'],assets['scale_seed'])
    centers,h,_=fit_fixed_coverage(us,ut,assets['coverage_centers'],assets['bandwidth_floor'],assets['coverage_seed'])
    source=SourceClassifier(encoder.spec,2).double();source.encoder=encoder
    with torch.no_grad():
        source.head.weight.copy_(torch.tensor([[.4,-.25],[-.3,.5]],dtype=torch.float64))
        source.head.bias.copy_(torch.tensor([.1,-.1],dtype=torch.float64))
    field=LocalMetricField(centers,h,config['spectral_bound'])
    model=RiemannianClassifier.from_s1(InitializedClassifier(source,scale,1.,2),pairs,field,**config['solver']).eval()
    if config['initial_geometry']!='identity':raise ValueError('Synthetic tests start at B=0')
    rules=RelationRules(**assets['relation_rules'])
    provenance=dict(initial_encoder_sha256=encoder_hash(encoder),dataset='synthetic_test_only')
    sm=build_source_memory(source_ids,us,labels,model.distance,rules,assets['source_relation_seed'],provenance)
    tm=build_target_memory(target_ids,ut,model.distance,rules,assets['target_relation_seed'],provenance)
    model.to(device)
    def lookup(g):
        if g not in graphs:raise PermissionError(g)
        return graphs[g]
    digest=hashlib.sha256()
    for g in sorted(graphs):
        digest.update(g.encode());digest.update(graphs[g].x.numpy().tobytes());digest.update(graphs[g].edge_index.numpy().tobytes())
    identity=dict(kind='synthetic_test_only',graph_input_sha256=digest.hexdigest(),source_train_labels=labels,allowed_target_ids=list(target_ids))
    engine=EvolutionTrainer(model,lookup,labels,target_ids,sm,tm,config,identity)
    if initialize:engine.initialize_correspondence()
    return engine
