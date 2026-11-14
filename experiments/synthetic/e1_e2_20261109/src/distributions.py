"""Known input distributions and exact smoothed scores; diagnostic use only."""
import math
import torch
from torch import nn
from tsba.classifier import Coordinates, ResidualClassifier


def generator(seed):
    return torch.Generator().manual_seed(seed)


def sample_gmm(n, dim, seed, shift=0.0):
    rng = generator(seed)
    y = torch.randint(2, (n,), generator=rng)
    x = torch.randn(n, dim, generator=rng, dtype=torch.float64)
    x[:, 0] = .5 * x[:, 0] + 2 * y - 1 + shift
    return x, y


class GaussianMixtureScore(nn.Module):
    """Diagonal Gaussian mixture after convolution with N(0, sigma²I).

    Population and empirical oracles use the same log-sum-exp implementation.
    Empirical components are target inputs with zero unsmoothed variance.
    """
    def __init__(self, means, variances, sigma):
        super().__init__()
        self.register_buffer('means', means.detach().clone())
        self.register_buffer('variance', torch.broadcast_to(variances, means.shape).clone() + sigma ** 2)

    def components(self, x):
        diff = self.means[None] - x[:, None]
        logs = -.5 * (diff.square() / self.variance + self.variance.log() + math.log(2 * math.pi)).sum(-1)
        return diff, logs

    def forward(self, x):
        parts = []
        for block in x.split(256):
            diff, logs = self.components(block)
            parts.append((logs.softmax(-1)[..., None] * diff / self.variance).sum(1))
        return torch.cat(parts) if parts else torch.empty_like(x)

    def log_prob(self, x):
        return torch.cat([self.components(b)[1].logsumexp(-1) - math.log(len(self.means))
                          for b in x.split(256)])


class ZeroScore(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


def setup_cell(dim, seed, cfg):
    source_raw, source_y = sample_gmm(512, dim, seed + 1000, .5)
    target_raw, _ = sample_gmm(512, dim, seed + 2000)
    evaluation_raw, evaluation_y = sample_gmm(10000, dim, seed + 3000)
    coordinates = Coordinates.fit(source_raw, cfg.coordinate_eps)
    # Shared-covariance source Bayes logits: W_c=Σ⁻¹ μ_c,
    # b_c=log π_c−0.5 μ_cᵀΣ⁻¹μ_c. Common quadratic term cancels.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 7000)
        linear = nn.Linear(dim, 2, dtype=torch.float64)
    means = torch.zeros(2, dim, dtype=torch.float64)
    means[:, 0] = torch.tensor([-.5, 1.5])
    variance = torch.ones(dim, dtype=torch.float64)
    variance[0] = .25
    with torch.no_grad():
        linear.weight.copy_(means / variance)
        linear.bias.copy_(-.5 * (means.square() / variance).sum(-1) - math.log(2))
    head = ResidualClassifier.from_source(linear, coordinates, min(16, dim), seed)
    source, target, evaluation = [coordinates(x).detach() for x in (source_raw, target_raw, evaluation_raw)]
    # Probabilities must survive the source-only scalar coordinate transform.
    with torch.no_grad():
        torch.testing.assert_close(linear(source_raw).softmax(-1), head(source).softmax(-1), atol=1e-12, rtol=1e-12)
    return dict(source=source, source_y=source_y, target=target, evaluation=evaluation,
                evaluation_y=evaluation_y, coordinates=coordinates, head=head)


def oracles(target, coordinates, sigma):
    dim = target.shape[1]
    means = torch.zeros(2, dim, dtype=target.dtype)
    means[:, 0] = torch.tensor([-1., 1.])
    variance = torch.ones(dim, dtype=target.dtype)
    variance[0] = .25
    pop = GaussianMixtureScore(coordinates(means), variance / coordinates.scale.square(), sigma)
    emp = GaussianMixtureScore(target, torch.zeros_like(target), sigma)
    return pop.eval(), emp.eval()


def query_banks(cell, sigma, seed):
    target, coordinates = cell['target'], cell['coordinates']
    dim = target.shape[1]
    rng = generator(seed + 4000)
    ids = torch.randint(len(target), (1024,), generator=rng)
    pool = target[ids] + sigma * torch.randn(1024, dim, generator=rng, dtype=target.dtype)
    raw, _ = sample_gmm(1024, dim, seed + 5000)
    # Fresh noise follows the same generator's GMM draws (not reusing noise).
    rng = generator(seed + 5000)
    torch.randint(2, (1024,), generator=rng)
    torch.randn(1024, dim, generator=rng, dtype=target.dtype)
    population = coordinates(raw) + sigma * torch.randn(1024, dim, generator=rng, dtype=target.dtype)
    planes = []
    for threshold in (-.5, .5):
        points = target.clone()
        points[:, 0] = (threshold - coordinates.mean[0]) / coordinates.scale
        normals = torch.zeros_like(points)
        normals[:, 0] = -1  # Production pair (0,1), margin decreases along +x₁.
        planes.append((points, normals))
    return dict(pool_noise=dict(points=pool, pool_ids=ids),
                population_noise=dict(points=population, generation_ids=torch.arange(1024)),
                auxiliary_planes=dict(points=torch.cat([p for p, _ in planes]),
                                      normals=torch.cat([n for _, n in planes]),
                                      raw_thresholds=torch.repeat_interleave(torch.tensor([-.5, .5]),512)))
