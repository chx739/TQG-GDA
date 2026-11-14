"""Design §3.3: fixed coverage, bounded trace-free local log metrics."""
import math
import torch
from torch import nn


@torch.no_grad()
def project_log_metrics(matrices, spectral_bound, iterations=80):
    """Frobenius projection onto symmetric, trace-zero, spectral-box matrices.

    Eigenvalues are shifted by ONE common root, then clipped. Subtracting a
    mean before/after clipping is not this projection. No eigenderivatives.
    """
    if not math.isfinite(spectral_bound) or spectral_bound < 0:
        raise ValueError('spectral_bound must be finite and nonnegative')
    if matrices.ndim < 2 or matrices.shape[-1] != matrices.shape[-2]:
        raise ValueError('Expected square matrices')
    if not torch.isfinite(matrices).all():
        raise FloatingPointError('Nonfinite log metric candidate')
    symmetric = (matrices + matrices.transpose(-1, -2)) / 2
    values, vectors = torch.linalg.eigh(symmetric)
    lo = values.min(-1, keepdim=True).values - spectral_bound
    hi = values.max(-1, keepdim=True).values + spectral_bound
    for _ in range(iterations):
        shift = (lo + hi) / 2
        projected = (values - shift).clamp(-spectral_bound, spectral_bound)
        positive = projected.sum(-1, keepdim=True) > 0
        lo = torch.where(positive, shift, lo)
        hi = torch.where(positive, hi, shift)
    projected = (values - (lo + hi) / 2).clamp(-spectral_bound, spectral_bound)
    return (vectors * projected.unsqueeze(-2)) @ vectors.transpose(-1, -2)


def taylor_degree(half_norm, tolerance=2.0 ** -53):
    """Smallest N such that 2 e^{4h} h^{N+1}/(N+1)! <= tolerance, h = ||A/2||_2.

    This bounds the RELATIVE error of ||T_N(A/2) delta||^2 against
    delta^T expm(A) delta for symmetric A with ||A||_2 <= 2h: the truncation
    error of the Taylor polynomial T_N is at most e^h h^{N+1}/(N+1)! in
    operator norm, the squared norm is Lipschitz with constant 2 e^h ||delta||
    on that neighbourhood, and the exact value is at least e^{-2h} ||delta||^2.
    With the default tolerance the polynomial is expm to double precision
    (c_g = 1 gives N = 15; c_g = 0 gives N = 0, i.e. exactly the identity).
    """
    if not math.isfinite(half_norm) or half_norm < 0:
        raise ValueError('Taylor degree needs a finite nonnegative norm bound')
    factor = 2 * math.exp(4 * half_norm)
    term, degree = 1.0, 0  # term = h^degree / degree!
    while True:
        term = term * half_norm / (degree + 1)  # h^{degree+1} / (degree+1)!
        if factor * term <= tolerance:
            return degree
        degree += 1
        if degree > 1000:
            raise ValueError('Spectral bound too large for a Taylor evaluation of the metric')


class LocalMetricField(nn.Module):
    def __init__(self, centers, bandwidths, spectral_bound=1.0):
        super().__init__()
        if centers.ndim != 2 or min(centers.shape) < 1:
            raise ValueError('centers must have shape [K,d], K,d >= 1')
        if bandwidths.shape != centers.shape[:1]:
            raise ValueError('One bandwidth is required per center')
        if (not torch.isfinite(centers).all() or not torch.isfinite(bandwidths).all()
                or (bandwidths <= 0).any()):
            raise ValueError('Centers must be finite and bandwidths finite/positive')
        if not math.isfinite(spectral_bound) or spectral_bound < 0:
            raise ValueError('Invalid spectral bound')
        self.spectral_bound = float(spectral_bound)
        # Degree of the matrix-vector Taylor polynomial used by quadratic_form();
        # derived once from the feasible set, not a tunable approximation level.
        # At least one (exactly zero-valued) term keeps B in the autograd graph
        # for the degenerate spectral_bound = 0 field.
        self.taylor_degree = max(1, taylor_degree(self.spectral_bound / 2))
        self.register_buffer('centers', centers.detach().clone())
        self.register_buffer('bandwidths', bandwidths.detach().clone().to(centers))
        self.B = nn.Parameter(centers.new_zeros(centers.shape[0], centers.shape[1], centers.shape[1]))

    @property
    def dim(self):
        return self.centers.shape[1]

    @torch.no_grad()
    def validate(self):
        """Check stored feasible parameters; call once per distance request."""
        tolerance = 1e-9 if self.B.dtype == torch.float64 else 2e-5
        if not torch.isfinite(self.B).all():
            raise FloatingPointError('Nonfinite metric parameter')
        if (self.B-self.B.transpose(-1, -2)).abs().max() > tolerance:
            raise ValueError('B must be symmetric; project the candidate before using it')
        if self.B.diagonal(dim1=-2, dim2=-1).sum(-1).abs().max() > tolerance:
            raise ValueError('B must be trace-free; project the candidate before using it')
        if torch.linalg.eigvalsh(self.B).abs().max() > self.spectral_bound + tolerance:
            raise ValueError('B exceeds its spectral bound; project the candidate')

    @torch.no_grad()
    def project_(self, candidate=None):
        self.B.copy_(project_log_metrics(self.B if candidate is None else candidate, self.spectral_bound))
        return self

    def weights(self, points):
        radii = ((points[..., None, :] - self.centers) / self.bandwidths[:, None]).square().sum(-1)
        r = torch.exp(-radii / 2)
        denominator = 1 + r.sum(-1, keepdim=True)
        # Index zero is the fixed identity-background atom B0=0.
        return torch.cat((1 / denominator, r / denominator), dim=-1)

    def log_metric(self, points, B=None):
        # An explicit B lets a recomputing chunk evaluate the field at the exact
        # primitives it saved, independent of the module's current parameter.
        B = self.B if B is None else B
        return torch.einsum('...k,kij->...ij', self.weights(points)[..., 1:], B)

    def forward(self, points, B=None):
        """Full matrix g(z) = expm(A(z)); reference/diagnostic use only."""
        return torch.matrix_exp(self.log_metric(points, B))

    def quadratic_form(self, points, delta, B=None):
        """delta^T g(points) delta without forming a matrix exponential.

        For symmetric A, delta^T e^A delta = ||e^{A/2} delta||^2, and e^{A/2} delta
        is evaluated as a Taylor polynomial of matrix-VECTOR products whose degree
        is fixed from the spectral bound to double precision (taylor_degree).
        This is the same quantity the path energy (§3.10.1) needs, at O(N d^2)
        per point instead of torch.matrix_exp, whose first and second
        derivatives (needed by the unrolled relaxation) exponentiate 2d x 2d and
        4d x 4d block matrices. `delta` broadcasts over the leading dims of
        `points` (e.g. one segment increment shared by its quadrature points).
        """
        half = self.log_metric(points, B) / 2
        vector = delta.expand(*half.shape[:-1])
        total, term = vector, vector
        for n in range(1, self.taylor_degree + 1):
            term = torch.matmul(half, term.unsqueeze(-1)).squeeze(-1) / n
            total = total + term
        return total.square().sum(-1)


@torch.no_grad()
def fit_fixed_coverage(source_coordinates, target_coordinates, num_centers,
                       bandwidth_floor, seed=0, iterations=30):
    """Equal-domain weighted Lloyd fit, with fixed seeded weighted k-means++.

    Inputs are one coordinate per ORIGINAL allowed graph, not resampled copies.
    This is a numerical local solution of the design's coverage objective,
    not a claim of a globally optimal k-means solution. No labels are accepted.
    """
    source = source_coordinates.detach()
    target = target_coordinates.detach()
    if (source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]
            or len(source) == 0 or len(target) == 0):
        raise ValueError('Both domains need nonempty coordinates in the same space')
    if not math.isfinite(bandwidth_floor) or bandwidth_floor <= 0:
        raise ValueError('A positive finite bandwidth floor is required')
    points = torch.cat((source, target)).cpu()
    if not torch.isfinite(points).all() or not 1 <= num_centers <= len(points) or iterations < 1:
        raise ValueError('Invalid coverage inputs or budget')
    weights = torch.cat((points.new_full((len(source),), .5/len(source)),
                         points.new_full((len(target),), .5/len(target))))
    generator = torch.Generator().manual_seed(seed)
    chosen = [int(torch.multinomial(weights, 1, generator=generator))]
    for _ in range(num_centers-1):
        nearest = (points[:, None]-points[chosen][None]).square().sum(-1).min(-1).values
        preference = weights * nearest
        preference[chosen] = 0
        if preference.sum() == 0:
            preference = weights.clone()
            preference[chosen] = 0
        chosen.append(int(torch.multinomial(preference, 1, generator=generator)))
    centers = points[chosen].clone()
    for _ in range(iterations):
        assignment = (points[:, None]-centers[None]).square().sum(-1).argmin(-1)
        for k in range(num_centers):
            mask = assignment == k
            if mask.any():
                centers[k] = (points[mask]*weights[mask, None]).sum(0)/weights[mask].sum()
    distances = (points[:, None]-centers[None]).square().sum(-1)
    assignment = distances.argmin(-1)
    bandwidths = points.new_full((num_centers,), bandwidth_floor)
    empty = []
    for k in range(num_centers):
        mask = assignment == k
        if mask.any():
            rms = ((distances[mask, k]*weights[mask]).sum()/weights[mask].sum()).sqrt()
            bandwidths[k] = rms.clamp_min(bandwidth_floor)
        else:
            empty.append(k)  # Keep last center; bandwidth uses the declared floor.
    metadata = dict(seed=seed, iterations=iterations, source_count=len(source), target_count=len(target),
                    source_weight=.5, target_weight=.5, empty_clusters=empty,
                    weighted_objective=float((distances.min(-1).values*weights).sum()),
                    bandwidth='domain-weighted cluster RMS with fixed positive floor')
    return centers.to(source), bandwidths.to(source), metadata
