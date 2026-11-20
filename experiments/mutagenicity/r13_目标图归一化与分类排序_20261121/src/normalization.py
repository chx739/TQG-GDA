"""Sequential weighted population BN moments in the frozen encoder's call order."""
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Batch

from tsba.pretrain import encode, graph_batches


class PopulationMoments:
    """Merge centered weighted moments in float64 across batches."""

    def __init__(self, weighting):
        if weighting not in ('node', 'graph'):
            raise ValueError('Unknown BN weighting')
        self.weighting = weighting
        self.graphs = self.nodes = self.mass = 0
        self.mean = self.m2 = None

    def update(self, activation, ptr):
        sizes = ptr.diff()
        n, g = len(activation), len(sizes)
        if activation.ndim != 2 or n != int(ptr[-1]) or not g or bool((sizes <= 0).any()):
            raise ValueError('BN input must have one row per node of each complete graph')
        values = activation.to(torch.float64)
        if not torch.isfinite(values).all():
            raise ValueError('Nonfinite BN input')
        if self.weighting == 'graph':
            weights = sizes.to(torch.float64).reciprocal().repeat_interleave(sizes)[:, None]
            batch_mass = g
            mean = (values * weights).sum(0) / batch_mass
            m2 = ((values - mean).square() * weights).sum(0)
        else:
            batch_mass = n
            mean = values.mean(0)
            m2 = (values - mean).square().sum(0)
        if self.mass:
            delta = mean - self.mean
            joined = self.mass + batch_mass
            self.m2 = self.m2 + m2 + delta.square() * (self.mass * batch_mass / joined)
            self.mean = self.mean + delta * (batch_mass / joined)
        else:
            self.mean, self.m2 = mean, m2
        self.mass += batch_mass
        self.graphs += g
        self.nodes += n

    def finish(self):
        if not self.mass:
            raise ValueError('No complete graph contributed BN statistics')
        var = self.m2 / self.mass
        if not torch.isfinite(self.mean).all() or not torch.isfinite(var).all() or bool((var < 0).any()):
            raise ValueError('Invalid weighted population mean or variance')
        return self.mean, var


@torch.inference_mode()
def discover(encoder, batch):
    modules = {name: layer for name, layer in encoder.named_modules() if isinstance(layer, nn.BatchNorm1d)}
    names = []
    handles = [layer.register_forward_pre_hook(lambda _m, _x, name=name: names.append(name))
               for name, layer in modules.items()]
    try:
        encoder(batch)
    finally:
        for handle in handles:
            handle.remove()
    interleaved = [name for layer in range(encoder.spec['num_layers'])
                   for name in (f'convs.{layer}.nn.1', f'norms.{layer}')]
    if names != interleaved or set(modules) != set(names):
        raise ValueError(f'Unexpected encoder BN forward order: {names}')
    return [(name, modules[name]) for name in names]


def verify_frozen(original, candidate, bn_names):
    if any(m.training for m in candidate.modules()) or any(p.requires_grad or p.grad is not None
                                                          for p in candidate.parameters()):
        raise ValueError('A frozen R13 inference arm entered training')
    original_state, fitted_state = original.state_dict(), candidate.state_dict()
    if original_state.keys() != fitted_state.keys() or original.encoder.spec != candidate.encoder.spec:
        raise ValueError('The frozen source architecture changed')
    allowed = {f'encoder.{name}.{suffix}' for name in bn_names for suffix in ('running_mean', 'running_var')}
    changed = {name for name in original_state if not torch.equal(original_state[name], fitted_state[name])}
    if not changed <= allowed:
        raise ValueError(f'Unexpected frozen-state changes: {changed - allowed}')
    own = dict(original.encoder.named_modules())
    fitted = dict(candidate.encoder.named_modules())
    if any(own[name].eps != fitted[name].eps for name in bn_names):
        raise ValueError('BN epsilon changed')
    return sorted(changed)


@torch.inference_mode()
def recalibrate(original, graphs, weighting, batch_size):
    """Fit one arm starting from the original buffers, without labels or gradients."""
    if not graphs or any(graph.y is not None for graph in graphs):
        raise ValueError('Expected a nonempty unlabeled graph cohort')
    # PyG's copy of GINEConv recreates its edge projection in training mode.
    model = deepcopy(original).eval()
    batches = [Batch.from_data_list([graphs[i] for i in ids]) for ids in graph_batches(graphs, batch_size)]
    ordered = discover(model.encoder, batches[0])
    summaries = {}
    for name, layer in ordered:
        accumulator = PopulationMoments(weighting)
        if layer.running_mean is None or layer.running_var is None:
            raise ValueError('R13 requires BN running buffers')
        calls = 0

        def collect(_layer, args):
            nonlocal calls
            calls += 1
            accumulator.update(args[0], batch.ptr)

        handle = layer.register_forward_pre_hook(collect)
        try:
            for batch in batches:
                model.encoder(batch)
        finally:
            handle.remove()
        if calls != len(batches) or accumulator.graphs != len(graphs):
            raise ValueError('Every complete graph must enter a BN statistic exactly once')
        mean, var = accumulator.finish()
        layer.running_mean.copy_(mean)
        layer.running_var.copy_(var)
        summaries[name] = dict(mean=mean.tolist(), variance=var.tolist(), graphs=accumulator.graphs,
                               nodes=accumulator.nodes, effective_weight=accumulator.mass,
                               epsilon=layer.eps, buffer_dtype=str(layer.running_mean.dtype))
    changed = verify_frozen(original, model, list(summaries))
    return model, summaries, changed


@torch.inference_mode()
def logits(model, graphs, batch_size):
    if any(graph.y is not None for graph in graphs):
        raise PermissionError('R13 inference graphs must not carry labels')
    embeddings = encode(model.encoder, graphs, batch_size, 'cpu').double()
    # The historical R7 B0 used a float32 encoder, followed by the original
    # R5 affine head promoted to float64. Use that exact convention for all arms.
    return (embeddings @ model.head.weight.double().T + model.head.bias.double()).numpy()


def compare_logits(actual, expected, tolerance):
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('Prediction replay acquired nonfinite or mismatched logits')
    np.testing.assert_allclose(actual, expected, **tolerance)
    np.testing.assert_array_equal(actual.argmax(axis=1), expected.argmax(axis=1))
    return float(np.max(np.abs(actual - expected)))
