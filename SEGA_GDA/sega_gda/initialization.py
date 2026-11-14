"""Exact §3.5.2 handoff, restricted explicitly to g=I (S1 only).

Squared Euclidean distance is the exact geodesic distance at identity. This
module refuses nonzero B; it is NOT a substitute for S2's learned-field solver.
"""
import copy
import random
import torch
from torch import nn


class IdentityGeometry(nn.Module):
    def __init__(self, dim, num_regions=1):
        super().__init__()
        self.register_buffer('B', torch.zeros(num_regions, dim, dim, dtype=torch.float64))

    def assert_identity(self):
        if not torch.isfinite(self.B).all() or torch.count_nonzero(self.B).item():
            raise ValueError('S1 identity distance cannot evaluate a nonzero metric field')

    def forward(self, points):
        self.assert_identity()
        return torch.eye(points.shape[-1], dtype=points.dtype, device=points.device).expand(
            *points.shape[:-1], points.shape[-1], points.shape[-1])

    def squared_distance(self, x, y):
        self.assert_identity()
        return (x - y).square().sum(-1)


def fixed_scale_pairs(coordinates, graph_ids, pair_count, seed):
    if coordinates.ndim != 2 or not torch.isfinite(coordinates).all():
        raise ValueError('Invalid initial source coordinates')
    if len(set(graph_ids)) != len(graph_ids) or len(graph_ids) != len(coordinates):
        raise ValueError('Scale references require distinct source graph IDs')
    if pair_count < 1 or len(coordinates) < 2:
        raise ValueError('No valid scale reference budget')
    rng = random.Random(seed)
    n = len(coordinates)
    requested = min(pair_count, n * (n - 1) // 2)
    chosen = set()
    for _ in range(max(1000, 100 * requested)):
        a, b = sorted(rng.sample(range(n), 2))
        if (coordinates[a] - coordinates[b]).square().sum() > 0:
            chosen.add((a, b))
        if len(chosen) == requested:
            break
    if len(chosen) != requested:
        raise ValueError('Insufficient distinct noncoincident source scale pairs; no epsilon fallback')
    indices = torch.tensor(sorted(chosen), dtype=torch.long, device=coordinates.device)
    pairs = coordinates.detach()[indices].clone()
    scale2 = (pairs[:, 0] - pairs[:, 1]).square().sum(-1).mean()
    if not torch.isfinite(scale2) or scale2 <= 0:
        raise ValueError('Degenerate source scale')
    ids = [[graph_ids[a], graph_ids[b]] for a, b in sorted(chosen)]
    return pairs, ids, scale2


class GeometricHead(nn.Module):
    def __init__(self, q, beta, scale2, temperature, num_regions=1):
        super().__init__()
        if not temperature > 0 or not torch.isfinite(torch.tensor(temperature)):
            raise ValueError('Temperature must be fixed, finite and positive')
        if not torch.isfinite(scale2) or scale2 <= 0:
            raise ValueError('Scale must be finite and positive, without epsilon flooring')
        if not torch.isfinite(q).all() or not torch.isfinite(beta).all():
            raise ValueError('Nonfinite converted classifier')
        self.q = nn.Parameter(q.detach().clone())
        self.beta = nn.Parameter(beta.detach().clone())
        self.register_buffer('scale2', scale2.detach().clone())
        self.register_buffer('temperature', q.new_tensor(temperature))
        self.geometry = IdentityGeometry(q.shape[1], num_regions).to(q)

    @classmethod
    def from_linear(cls, head, scale2, temperature, num_regions=1):
        # scale2 is s0^2, not s0 and not s0^4.
        q = temperature * scale2 * head.weight.detach() / 2
        beta = head.bias.detach() + q.square().sum(-1) / (temperature * scale2)
        return cls(q, beta, scale2, temperature, num_regions)

    def forward(self, coordinates):
        energy = self.geometry.squared_distance(coordinates[:, None, :], self.q[None, :, :])
        return -energy / (self.scale2 * self.temperature) + self.beta


class InitializedClassifier(nn.Module):
    def __init__(self, source, scale2, temperature, num_regions=1):
        super().__init__()
        self.encoder = copy.deepcopy(source.encoder)
        self.head = GeometricHead.from_linear(source.head, scale2, temperature, num_regions)

    def forward(self, batch):
        return self.head(self.encoder(batch))


@torch.no_grad()
def verify_conversion(source, converted, loader, device, atol=1e-10, rtol=1e-8):
    source.eval()
    converted.eval()
    errors = dict(max_coordinate_error=0.0, max_probability_error=0.0,
                  max_margin_error=0.0, prediction_mismatches=0, graphs=0)
    for batch in loader:
        batch = batch.to(device)
        u = source.encoder(batch)
        v = converted.encoder(batch)
        linear = source.head(u)
        geometric = converted.head(v)
        p, q = linear.softmax(-1), geometric.softmax(-1)
        margin = linear - linear[:, :1]
        geo_margin = geometric - geometric[:, :1]
        if not torch.equal(u, v):
            raise AssertionError('Conversion changed source coordinates')
        torch.testing.assert_close(p, q, atol=atol, rtol=rtol)
        torch.testing.assert_close(margin, geo_margin, atol=atol, rtol=rtol)
        mismatches = int((linear.argmax(-1) != geometric.argmax(-1)).sum())
        if mismatches:
            raise AssertionError('Conversion changed predicted classes')
        errors['max_coordinate_error'] = max(errors['max_coordinate_error'], float((u-v).abs().max()))
        errors['max_probability_error'] = max(errors['max_probability_error'], float((p-q).abs().max()))
        errors['max_margin_error'] = max(errors['max_margin_error'], float((margin-geo_margin).abs().max()))
        errors['prediction_mismatches'] += mismatches
        errors['graphs'] += batch.num_graphs
    return errors
