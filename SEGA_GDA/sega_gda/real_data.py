"""Real TU task boundary: preserve source split; never separate a target label."""
import json
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.data.separate import separate
from .data import SourceGraphs,sha256


class InputProtocolRequired(ValueError):
    pass


class RealTaskGraphs:
    def __init__(self,task_manifest,preprocessing):
        self.manifest=json.loads(Path(task_manifest).read_text())
        m=self.manifest
        if m.get('status')!='LOCKED_DEVELOPMENT' or not m.get('protocol_authority'):
            raise InputProtocolRequired('Real target input policy is not locked; no target graphs accessed')
        if m.get('target_policy')!='transductive_unlabeled':
            raise InputProtocolRequired('This pilot adapter requires an explicitly declared transductive input pool')
        self.source=SourceGraphs(m['data_cache'],m['source_split_manifest'],m['use_node_attributes'])
        if sha256(m['source_split_manifest'])!=m['source_split_sha256']:
            raise ValueError('Source split changed')
        if sha256(m['target_index_file'])!=m['target_index_sha256']:
            raise ValueError('Target domain index changed')
        ids=np.load(m['target_index_file'],allow_pickle=False)
        if ids.ndim!=1 or ids.dtype.kind not in 'iu' or len(set(ids.tolist()))!=len(ids):
            raise ValueError('Invalid target domain identities')
        if set(ids.tolist())&self.source.allowed:
            raise ValueError('Source/target graph IDs overlap')
        if not len(ids) or ids.min()<0 or ids.max()>=len(self.source.slices['x'])-1:
            raise ValueError('Target index out of dataset range')
        if preprocessing['fit_ids']!=self.source.train_ids:
            raise ValueError('Preprocessing was not fitted on this source train split')
        self.preprocessing=preprocessing
        dataset=m['dataset']
        self.source_ids={f'{dataset}:{i:08d}':i for i in self.source.train_ids}
        self.target_ids={f'{dataset}:{int(i):08d}':int(i) for i in ids}
        # Remove the y FIELD from the collated view BEFORE separating any target.
        # SourceGraphs alone retains the labeled cache for allowed source eval.
        keys=('x','edge_index','edge_attr')
        self.inputs=Data(**{k:getattr(self.source.data,k) for k in keys if getattr(self.source.data,k,None) is not None})
        self.input_slices={k:v for k,v in self.source.slices.items() if k in keys}
        self.source_labels={g:self.source.class_map[int(self.source.graph(i).y.item())] for g,i in self.source_ids.items()}
        # Separated, preprocessed input graphs are immutable for the task: cache
        # them so every content step does not re-run separate()+checks for the
        # whole relation memory. Callers build fresh Data/Batch objects from them.
        self._graph_cache={}

    def graph(self,graph_id):
        cached=self._graph_cache.get(graph_id)
        if cached is not None: return cached
        indices={**self.source_ids,**self.target_ids}
        if graph_id not in indices: raise PermissionError('Graph outside source-train / target-unlabeled pool')
        g=separate(Data,self.inputs,indices[graph_id],self.input_slices,decrement=False).clone()
        g.x=(g.x.double()-self.preprocessing['mean'])/self.preprocessing['scale']
        if g.edge_attr is not None: g.edge_attr=g.edge_attr.double()
        if g.y is not None: raise RuntimeError('Label crossed the training input boundary')
        if g.x.ndim!=2 or g.num_nodes<1 or not torch.isfinite(g.x).all():
            raise FloatingPointError('Invalid real graph features')
        if g.edge_index.ndim!=2 or g.edge_index.shape[0]!=2:
            raise ValueError('Invalid real graph edge shape')
        if g.edge_index.numel() and (g.edge_index.min()<0 or g.edge_index.max()>=g.num_nodes):
            raise ValueError('Invalid real graph edge endpoints')
        if g.edge_attr is not None and (g.edge_attr.shape[0]!=g.num_edges or not torch.isfinite(g.edge_attr).all()):
            raise ValueError('Invalid real graph edge attributes')
        self._graph_cache[graph_id]=g
        return g

    def validation_graphs(self):
        # Explicit separate role; validation IDs cannot enter graph().
        return self.source.materialize(self.source.val_ids,self.preprocessing)
