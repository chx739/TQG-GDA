import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import torch
from torch.nn import functional as F
from torch_geometric.data import Data, Batch
from tqg_gda.config import Config
from tqg_gda.classifier import GraphClassifier
from tqg_gda.predict import model_checkpoint, load_predictor, predict_graphs
from tqg_gda.v1.query import ProbeConfig, build_feedback, feedback_loss
from tqg_gda.v2.query import V2Config, build_gradient_feedback, parameter_feedback_loss

ROOT = Path(__file__).resolve().parents[3]
torch.set_num_threads(1)
rows = []

def graphs(count, offset=0):
    result=[]
    for i in range(count):
        n=3+(i+offset)%4
        x=torch.randn(n,14)
        edge=torch.tensor([(j,(j+1)%n) for j in range(n)]+[((j+1)%n,j) for j in range(n)]).T
        result.append(Data(x=x,edge_index=edge,edge_attr=F.one_hot(torch.arange(2*n)%3,3).float()))
    return result

for dtype in (torch.float32,torch.float64):
    for conv in ('gin','gine'):
        torch.manual_seed(787)
        model=GraphClassifier(dict(in_dim=14,edge_dim=3,hidden=64,num_layers=3,out_dim=16,dropout=.2,conv=conv),3,dtype=dtype).eval()
        source=Batch.from_data_list(graphs(5))
        target=Batch.from_data_list(graphs(3,5))
        labels=torch.tensor([0,1,2,0,1])
        original=copy.deepcopy(model.state_dict())
        v1=build_feedback(model,source,labels,target,ProbeConfig(),torch.Generator().manual_seed(31))
        p=tuple(model.parameters())
        g1=torch.cat([g.flatten() for g in torch.autograd.grad(feedback_loss(model,target,v1),p)])
        expected=torch.cat([g.flatten() for g in torch.autograd.grad(-(model(target)*v1.correction).sum()/3,p)])
        relative=float((g1-expected).norm()/expected.norm())
        assert relative<(1e-4 if dtype==torch.float32 else 1e-10),relative
        repeated=build_feedback(model,source,labels,target,ProbeConfig(),torch.Generator().manual_seed(31))
        assert torch.equal(v1.correction,repeated.correction)
        assert not any(t.requires_grad for t in (v1.reference,v1.coefficients,v1.source_features,v1.target_features))
        v2=build_gradient_feedback(model,source,labels,target,V2Config())
        assert v2.statistics['status']=='gradient_feedback',v2.statistics
        g2=torch.cat([g.flatten() for g in torch.autograd.grad(parameter_feedback_loss(model,v2),p)])
        e2=float((g2-v2.gradient).norm())
        assert torch.equal(g2,v2.gradient),e2
        assert not any(t.requires_grad for t in (v2.snapshot,v2.basis,v2.coefficients,v2.readout,v2.source_response))
        for name,value in model.state_dict().items():
            assert torch.equal(value,original[name]),name
        rows.append(dict(conv=conv,dtype=str(dtype),parameters=sum(v.numel() for v in p),v1_gradient_relative_error=relative,v1_directions=len(v1.directions),v2_rank=v2.statistics['rank'],v2_gradient_error=e2,v2_converged=v2.statistics['calibration']['converged'],v2_gap=v2.statistics['calibration']['gap']))

# Raw inference strips TU continuous columns exactly once when configured false.
with tempfile.TemporaryDirectory(prefix='tqg-review-inference-') as temp:
    torch.manual_seed(44)
    model=GraphClassifier(dict(in_dim=3,edge_dim=0,hidden=5,num_layers=2,out_dim=4,conv='gin'),2,dtype=torch.float32).eval()
    task=SimpleNamespace(classes=[3,8],preprocessing=dict(mean=torch.zeros(3,dtype=torch.float64),scale=torch.ones(3,dtype=torch.float64),fit_ids=[0,1]),sizes={'num_node_attributes':2})
    config=SimpleNamespace(use_node_attributes=False,dtype='float32',version='v1')
    checkpoint=model_checkpoint(model,task,config)
    path=Path(temp)/'model.pt';torch.save(checkpoint,path)
    restored,state=load_predictor(path)
    raw=[Data(x=torch.randn(4,5),edge_index=torch.tensor([[0,1,2,3],[1,2,3,0]]),y=torch.tensor([999])) for _ in range(2)]
    actual,original_classes=predict_graphs(restored,state,raw)
    clean=[g.clone() for g in raw]
    for g in clean:g.x=g.x[:,2:];g.y=None
    expected=model(Batch.from_data_list(clean)).softmax(-1)
    assert torch.equal(actual,expected)
    assert all(g.y.item()==999 for g in raw)
    rows.append(dict(check='raw_checkpoint_prediction_without_node_attributes',max_error=float((actual-expected).abs().max()),class_labels=original_classes.tolist()))

manifest=json.loads((ROOT/'configs/sega_style/minimal_20261106/core_tasks.json').read_text())
configs=[Config.from_dict(x) for x in manifest]
individuals=[Config.from_json(x) for x in sorted((ROOT/'configs/sega_style/minimal_20261106').glob('*_seed*.json'))]
assert len(configs)==len(individuals)==15
assert sorted(json.dumps(x.to_dict(),sort_keys=True) for x in configs)==sorted(json.dumps(x.to_dict(),sort_keys=True) for x in individuals)
assert all((ROOT/x.source_split).is_file() and (ROOT/x.target_index).is_file() for x in configs)
rows.append(dict(check='all_core_configs_parse_and_match_individuals',count=15,versions={v:sum(x.version==v for x in configs) for v in ('v1','v2','source_only')}))
print(json.dumps(rows,indent=2))
