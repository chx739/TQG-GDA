"""Geometric classifier using the SAME normalized path distance as future OT/relations."""
import copy
import torch
from torch import nn
from .geodesic import GeodesicSolver, NormalizedGeodesicDistance


class RiemannianClassifier(nn.Module):
    def __init__(self, encoder, q, beta, temperature, distance):
        super().__init__()
        self.encoder = encoder
        self.q = nn.Parameter(q.detach().clone())
        self.beta = nn.Parameter(beta.detach().clone())
        self.register_buffer('temperature', temperature.detach().clone())
        if not torch.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError('Fixed temperature must be finite and positive')
        self.distance = distance

    @classmethod
    def from_s1(cls, initialized, reference_pairs, metric, **solver_config):
        # No new coordinate map, no normalization, no new class references.
        distance = NormalizedGeodesicDistance(GeodesicSolver(metric, **solver_config), reference_pairs)
        distance.to(initialized.head.q)
        return cls(copy.deepcopy(initialized.encoder), initialized.head.q,
                   initialized.head.beta, initialized.head.temperature, distance)

    def logits(self, coordinates, *, q=None, beta=None):
        # Optional frozen references support §3.9.3 without a second classifier.
        q = self.q if q is None else q
        beta = self.beta if beta is None else beta
        return -self.distance.pairwise(coordinates, q)/self.temperature+beta

    def forward(self, batch):
        return self.logits(self.encoder(batch))
