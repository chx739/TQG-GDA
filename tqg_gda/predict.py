"""Standalone GNN inference. No source bank, queries or kernels are retained."""
import copy

import torch
from torch_geometric.data import Batch

from .classifier import GraphClassifier
from .utils import cpu_state


def model_checkpoint(model, task, config, **metadata):
    return dict(format='tqg-sega-gnn', schema_version=1,
        encoder_spec=copy.deepcopy(model.encoder.spec), class_labels=list(task.classes),
        model=cpu_state(model), preprocessing=copy.deepcopy(task.preprocessing),
        use_node_attributes=config.use_node_attributes,
        num_node_attributes=int(task.sizes['num_node_attributes']),
        dtype=config.dtype, version=config.version, **metadata)


def load_predictor(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('format') != 'tqg-sega-gnn' or state.get('schema_version') != 1:
        raise ValueError('Expected a new SEGA-style TQG checkpoint; legacy checkpoints use training.load_model')
    model = GraphClassifier(state['encoder_spec'], len(state['class_labels']),
                            dtype=getattr(torch, state['dtype'])).to(
        device=device, dtype=getattr(torch, state['dtype']))
    model.load_state_dict(state['model'])
    return model.eval(), state


@torch.no_grad()
def predict_graphs(model, state, graphs):
    """Raw complete graphs -> probabilities and original class IDs.

    This adapter applies the stored fit-only preprocessing exactly once.
    Use predict_task for already prepared GraphDomainTask inputs.
    """
    if not graphs:
        raise ValueError('Need at least one graph')
    inputs = []
    for original in graphs:
        graph = original.clone()
        graph.y = None
        if not state['use_node_attributes']:
            graph.x = graph.x[:, state['num_node_attributes']:]
        mean, scale = state['preprocessing']['mean'], state['preprocessing']['scale']
        graph.x = (graph.x.double() - mean.to(graph.x.device)) / scale.to(graph.x.device)
        if graph.x.shape[1] != state['encoder_spec']['in_dim'] or not torch.isfinite(graph.x).all():
            raise ValueError('Inference node features do not match training')
        inputs.append(graph)
    p = next(model.parameters())
    batch = Batch.from_data_list(inputs).to(p.device)
    model.eval()
    probability = model(batch).softmax(-1).cpu()
    classes = torch.tensor(state['class_labels'])
    return probability, classes[probability.argmax(-1)]


@torch.no_grad()
def predict_task(model, task, batch_size=128):
    model.eval()
    parameter = next(model.parameters())
    probabilities = []
    for start in range(0, len(task.target_ids), batch_size):
        inputs = task.batch(task.target_ids[start:start+batch_size], 'target', parameter.device, parameter.dtype)
        probabilities.append(model(inputs).softmax(-1).cpu())
    return torch.cat(probabilities)
