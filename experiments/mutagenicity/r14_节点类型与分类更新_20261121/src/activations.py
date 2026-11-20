"""Cache the exact last outer BN inputs and fixed source readout."""
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool

from tsba.pretrain import graph_batches


def batches(graphs, batch_size):
    if not graphs or any(g.y is not None for g in graphs):
        raise PermissionError('The graph forward requires a nonempty graph-only pool')
    for order in graph_batches(graphs, batch_size):
        yield order, Batch.from_data_list([graphs[i] for i in order])


def tensor(array):
    return array.detach().double().numpy()


def theta(model):
    bn = model.encoder.norms[-1]
    return np.r_[tensor(bn.weight), tensor(bn.bias)]


def normalize(pre_bn, bn):
    return (tensor(pre_bn) - tensor(bn.running_mean)) / np.sqrt(tensor(bn.running_var) + bn.eps)


def margin(logits):
    return logits[:,1] - logits[:,0]


def theta_split(parameters, width):
    if parameters.shape != (2 * width,) or not np.isfinite(parameters).all():
        raise ValueError('Expected finite final BN gamma and beta')
    return np.split(parameters, 2)


def relu_right(x, speed):
    return np.where(x > 0, speed, np.where(x == 0, np.maximum(speed, 0), 0.))


def relu_change(x, delta):
    """Stable one-sided finite perturbation without subtracting positive activations."""
    return np.where(x > 0, np.maximum(-x, delta), np.maximum(x + delta, 0.))


@dataclass
class HeldBatch:
    model: object
    pre_bn: torch.Tensor
    bn_output: np.ndarray
    prefix: torch.Tensor
    readout_input: np.ndarray
    node_batch: torch.Tensor
    n: int
    base_logits: np.ndarray

    def local(self, direction, fd_step):
        bn = self.model.encoder.norms[-1]
        readout = self.model.encoder.readout
        width = bn.num_features
        affine_weights = tensor(readout[0].weight)[:, -width:]
        terminal = (tensor(self.model.head.weight)[1] - tensor(self.model.head.weight)[0]) @ tensor(readout[2].weight)
        gamma, beta = theta_split(direction, width)
        dbn = normalize(self.pre_bn, bn) * gamma + beta
        dp = self._sum_nodes(relu_right(self.bn_output, dbn))
        d_readout = dp @ affine_weights.T
        velocity = relu_right(self.readout_input, d_readout) @ terminal
        # Keep the verification inside the first linear segment. Exactly
        # zero activations stay in the right-hand segment for this direction.
        for values, speed in ((self.bn_output, dbn), (self.readout_input, d_readout)):
            moving = ((values > 0) & (speed < 0)) | ((values < 0) & (speed > 0))
            if np.any(moving):
                fd_step = min(fd_step, .25 * float(np.min(-values[moving] / speed[moving])))
        delta_pool = self._sum_nodes(relu_change(self.bn_output, fd_step * dbn))
        delta_post = relu_change(self.readout_input, delta_pool @ affine_weights.T)
        fd = (delta_post @ terminal) / fd_step
        return velocity, fd, fd_step

    def _sum_nodes(self, matrix):
        return global_add_pool(torch.from_numpy(matrix), self.node_batch, size=self.n).numpy()

    @torch.inference_mode()
    def finite(self, parameters):
        bn = self.model.encoder.norms[-1]
        gamma, beta = (torch.as_tensor(a, dtype=bn.weight.dtype)
                       for a in theta_split(parameters, bn.num_features))
        x = F.batch_norm(self.pre_bn, bn.running_mean, bn.running_var, gamma, beta,
                         training=False, momentum=0., eps=bn.eps)
        last = global_add_pool(torch.relu(x), self.node_batch, size=self.n)
        h = self.model.encoder.readout(torch.cat((self.prefix, last), dim=1)).double()
        return tensor(h @ self.model.head.weight.double().T + self.model.head.bias.double())


@torch.inference_mode()
def held_batches(model, graphs, batch_size):
    """Capture native BN and ReLU base points, including exactly-zero activations."""
    encoder, bn = model.encoder, model.encoder.norms[-1]
    width = bn.num_features
    captured = {}
    handles = (
        bn.register_forward_hook(lambda _m, args, z: captured.update(pre=args[0], bn=z)),
        encoder.readout[0].register_forward_hook(lambda _m, args, z: captured.update(pooled=args[0], readout=z)),
    )
    output = []
    try:
        for _, batch in batches(graphs, batch_size):
            h = encoder(batch).double()
            base = tensor(h @ model.head.weight.double().T + model.head.bias.double())
            cache = HeldBatch(model, captured['pre'], tensor(captured['bn']), captured['pooled'][:, :-width],
                              tensor(captured['readout']), batch.batch, batch.num_graphs, base)
            np.testing.assert_array_equal(cache.finite(theta(model)), base)
            output.append(cache)
    finally:
        for handle in handles:
            handle.remove()
    return output


def residualize(z, v):
    x = np.column_stack((np.ones(len(z)), z))
    coefficient, _, rank, _ = np.linalg.lstsq(x, v, rcond=None)
    return v - x @ coefficient, coefficient, int(rank)
