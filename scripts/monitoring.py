"""Single real direction. Explicit target protocol required before any access."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from torch.nn import functional as F
from torch_geometric.loader import DataLoader
from sega_gda.pilot import initialize_real_task
from sega_gda.real_data import InputProtocolRequired
from sega_gda.train_source import seed_everything,write_json


class RuntimeMeasurements:
    """Wall times include synchronization; distance forward excludes backward.

    Instrumentation only: no tensor cache, optimizer or numerical rule changes.
    """
    def __init__(self,engine,output):
        self.engine,self.output=engine,output
        self.phase='initialization';self.events=[];self.distances=[];self.ot=[];self.mechanisms=[]
        solver=engine.model.distance.solver
        original=solver.forward
        def forward(x,y,**kwargs):
            self.sync();start=time.perf_counter()
            result=original(x,y,**kwargs)
            self.sync()
            self.distances.append(dict(phase=self.phase,pairs=int(torch.broadcast_shapes(x.shape[:-1],y.shape[:-1]).numel()),seconds=time.perf_counter()-start))
            return result
        solver.forward=forward
        import sega_gda.evolution as evolution
        import sega_gda.objectives as objectives
        solve=evolution.solve_semi_relaxed
        def timed_solve(*args,**kwargs):
            self.sync();start=time.perf_counter();result=solve(*args,**kwargs);self.sync()
            self.ot.append(dict(phase=self.phase,seconds=time.perf_counter()-start,shape=list(args[0].shape)))
            return result
        evolution.solve_semi_relaxed=objectives.solve_semi_relaxed=timed_solve
        for name in ('content_step','freeze_feedback','space_update'):
            self.wrap(name)

    def sync(self):
        if self.engine.model.q.is_cuda:torch.cuda.synchronize()

    @torch.no_grad()
    def classification_probe(self,feedback):
        model=self.engine.model
        u=self.engine._stack(feedback.coordinates,feedback.batch.query)
        distance=model.distance.pairwise(u,feedback.q)
        logits=-distance/model.temperature+feedback.beta
        values=logits.topk(2,dim=1).values
        return dict(distances=distance.cpu().tolist(),probabilities=logits.softmax(-1).cpu().tolist(),
                    top_two_margin=(values[:,0]-values[:,1]).cpu().tolist())

    def wrap(self,name):
        original=getattr(self.engine,name)
        def measured(*args,**kwargs):
            self.phase=name
            print(json.dumps(dict(event='begin',phase=name,round=self.engine.round+1)),flush=True)
            before=self.classification_probe(args[0]) if name=='space_update' else None
            self.sync();start=time.perf_counter();result=original(*args,**kwargs);self.sync()
            event=dict(phase=name,round=self.engine.round+1,seconds=time.perf_counter()-start,
                       peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if self.engine.model.q.is_cuda else None)
            self.events.append(event)
            if before is not None:
                self.mechanisms.append(dict(round=self.engine.round+1,before=before,after=self.classification_probe(args[0]),
                                            accepted=result['accepted'],fixed_content=True,fixed_class_parameters=True))
            write_json(self.output/'runtime_measurements.json',dict(events=self.events,distances=self.distances,ot=self.ot,mechanisms=self.mechanisms))
            print(json.dumps(dict(event='end',**event)),flush=True)
            return result
        setattr(self.engine,name,measured)


@torch.no_grad()
def source_validation(model,data):
    logits,labels=[],[]
    for batch in DataLoader(data.validation_graphs(),batch_size=8,shuffle=False,
                            generator=torch.Generator().manual_seed(805)):
        logits.append(model(batch.to(model.q.device)));labels.append(batch.y.to(model.q.device))
    z,y=torch.cat(logits),torch.cat(labels);pred=z.argmax(-1)
    cm=torch.zeros(len(model.q),len(model.q),dtype=torch.long)
    for a,b in zip(y.cpu(),pred.cpu()):cm[a,b]+=1
    return dict(ce=float(F.cross_entropy(z,y)),accuracy=float((pred==y).double().mean()),
                confusion=cm.tolist(),class_recall=(cm.diag().double()/cm.sum(1)).tolist())

