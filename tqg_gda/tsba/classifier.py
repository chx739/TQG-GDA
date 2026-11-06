"""Fixed coordinates and probability-preserving residual classifier (§3.2–3.3)."""
import torch
from torch import nn
from torch.func import functional_call
from .utils import finite


class Coordinates(nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer("mean", mean.detach().clone())
        self.register_buffer("scale", scale.detach().clone())

    @classmethod
    def fit(cls, source_h, eps):
        finite(source_h)
        mean = source_h.mean(0)
        scale = ((source_h - mean).square().mean() + eps ** 2).sqrt()
        finite(scale)
        return cls(mean, scale)

    def forward(self, h):
        return (h.to(self.mean) - self.mean) / self.scale


class ResidualClassifier(nn.Module):
    def __init__(self, dim, classes, width, *, device=None, dtype=None):
        super().__init__()
        self.linear = nn.Linear(dim, classes, device=device, dtype=dtype)
        self.first = nn.Linear(dim, width, device=device, dtype=dtype)
        self.last = nn.Linear(width, classes, bias=False, device=device, dtype=dtype)
        nn.init.zeros_(self.first.bias)
        nn.init.zeros_(self.last.weight)

    @classmethod
    def from_source(cls, linear, coordinates, width, seed):
        # CPU initialization isolates random adapter features from prior training's RNG usage.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = cls(linear.in_features, linear.out_features, width, dtype=coordinates.mean.dtype)
        model = model.to(coordinates.mean.device)
        with torch.no_grad():
            w = linear.weight.to(coordinates.mean)
            model.linear.weight.copy_(coordinates.scale * w)
            model.linear.bias.copy_(w @ coordinates.mean + linear.bias.to(w))
        return model

    def forward(self, x):
        logits = self.linear(x) + self.last(nn.functional.silu(self.first(x)))
        return logits - logits.mean(-1, keepdim=True)


class ParameterMap:
    """Functional candidate evaluation: rejected steps never mutate the live model."""
    def __init__(self, model):
        self.model = model
        self.entries = [(name, p.shape, p.numel()) for name, p in model.named_parameters()]

    def vector(self):
        return torch.cat([p.detach().reshape(-1) for p in self.model.parameters()]).clone()

    def mapping(self, vector):
        result, offset = {}, 0
        for name, shape, size in self.entries:
            result[name] = vector[offset:offset + size].view(shape)
            offset += size
        return result

    def logits(self, vector, x):
        return functional_call(self.model, self.mapping(vector), (x,))

    def commit(self, vector):
        finite(vector)
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                p.copy_(self.mapping(vector)[name])
