"""TQG design §§3.4–3.11: stopped, stateless response queries.

Only encoder parameters are probed. No optimizer/model parameter is mutated;
only the final ordinary classification and feedback losses train the GNN.
"""
from dataclasses import dataclass, field
import math

import torch
from torch.func import functional_call
from torch.nn import functional as F

from .models import center_logits


@dataclass(frozen=True)
class ProbeConfig:
    count: int = 4
    radius: float = .05
    ridge: float = .01
    epsilon: float = 1e-8
    direction: str = 'target'
    responses: str = 'odd_even'

    def __post_init__(self):
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 1:
            raise ValueError('Probe count must be a positive integer')
        if any(not math.isfinite(v) or v <= 0 for v in (self.radius, self.ridge, self.epsilon)):
            raise ValueError('radius, ridge and epsilon must be finite and positive')
        if self.direction not in ('target', 'random', 'source'):
            raise ValueError('direction must be target, random or source')
        if self.responses not in ('odd_even', 'odd', 'linear'):
            raise ValueError('responses must be odd_even, odd or linear')


@dataclass
class Feedback:
    source_features: torch.Tensor
    target_features: torch.Tensor
    source_residual: torch.Tensor
    coefficients: torch.Tensor
    correction: torch.Tensor
    reference: torch.Tensor
    scales: torch.Tensor
    directions: tuple = ()
    statistics: dict = field(default_factory=dict)
    intercept: torch.Tensor | None = None


class Snapshot:
    def __init__(self, model):
        self.model = model
        self.parameters = {n: p.detach().clone() for n, p in model.named_parameters()}
        self.buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
        self.names = tuple(n for n in self.parameters if n.startswith('encoder.'))
        if not self.names:
            raise ValueError('Expected a model with encoder parameters')
        self.values = tuple(self.parameters[n] for n in self.names)
        if any(p.dtype not in (torch.float32, torch.float64) for p in self.parameters.values()):
            raise ValueError('TQG probes require at least FP32 floating parameters')

    def logits(self, batch, values=None):
        parameters = dict(self.parameters)
        if values is not None:
            parameters.update(zip(self.names, values))
        # Clone buffers to protect even the snapshot from a custom forward's mutation.
        buffers = {n: b.clone() for n, b in self.buffers.items()}
        return functional_call(self.model, (parameters, buffers), (batch,), strict=True)

    def jvp(self, batch, direction):
        # Reverse-over-reverse JVP supports PyG scatter operators. Its internal
        # AD graph is temporary: no derivative through directions enters training.
        _, tangent = torch.autograd.functional.jvp(
            lambda *values: self.logits(batch, values), self.values, direction,
            create_graph=False, strict=False)
        return tangent.detach()


def shared_response_scale(source, target, epsilon):
    """Inputs [graphs, classes, responses]; equal domain contribution, no centering."""
    if source.ndim != 3 or target.ndim != 3 or source.shape[1:] != target.shape[1:]:
        raise ValueError('Incompatible response matrices')
    d = (.5 * source.square().sum(1).mean(0) + .5 * target.square().sum(1).mean(0)
         + epsilon ** 2).sqrt()
    return source / d, target / d, d


@torch.no_grad()
def fit_residual(source_features, residual, ridge):
    if ridge <= 0 or not math.isfinite(ridge):
        raise ValueError('Ridge must be positive')
    if source_features.shape[:2] != residual.shape:
        raise ValueError('Residual shape does not match response features')
    m = source_features.shape[-1]
    if m == 0:
        return source_features.new_empty(0)
    gram = torch.einsum('bcm,bcn->mn', source_features, source_features) / len(residual)
    correlation = torch.einsum('bcm,bc->m', source_features, residual) / len(residual)
    coefficients = torch.linalg.solve(gram + ridge * torch.eye(m, device=gram.device, dtype=gram.dtype), correlation)
    if not torch.isfinite(coefficients).all():
        raise FloatingPointError('Nonfinite ridge coefficients')
    return coefficients.detach()


@torch.no_grad()
def fit_residual_with_intercept(source_features, residual, ridge):
    """Unpenalized class intercept, ridge-penalized shared response weights.

    Center across SOURCE FITTING GRAPHS only, after the usual shared RMS.
    The evaluation graph means never enter the fit.
    """
    if source_features.shape[:2] != residual.shape or not len(residual):
        raise ValueError('Residual shape does not match response features')
    feature_mean = source_features.mean(0)
    residual_mean = residual.mean(0)
    coefficients = fit_residual(source_features-feature_mean, residual-residual_mean, ridge)
    intercept = residual_mean-feature_mean@coefficients
    if not torch.isfinite(intercept).all():
        raise FloatingPointError('Nonfinite residual intercept')
    return coefficients.detach(), intercept.detach()


def build_feedback(model, source_batch, source_labels, target_batch, config, generator,
                   predictor='responses'):
    """No target label argument, no gradients retained by any returned feedback tensor."""
    if predictor not in ('responses', 'responses_with_intercept'):
        raise ValueError('Unknown response predictor')
    model.eval()  # Freeze dropout and BN running statistics; parameters stay trainable.
    snap = Snapshot(model)
    with torch.no_grad():
        source_base, target_base = snap.logits(source_batch), snap.logits(target_batch)
        if (source_base.ndim != 2 or target_base.ndim != 2 or source_base.shape[1] < 2
                or source_base.shape[1] != target_base.shape[1]):
            raise ValueError('Expected compatible multiclass logits, including two-logit binary tasks')
        if not torch.isfinite(source_base).all() or not torch.isfinite(target_base).all():
            raise FloatingPointError('Nonfinite snapshot logits')
        labels = source_labels.to(device=source_base.device, dtype=torch.long).reshape(-1)
        if len(labels) != len(source_base):
            raise ValueError('Source query label count differs from batch size')
        residual = F.one_hot(labels, source_base.shape[1]).to(source_base) - source_base.softmax(-1)
    source_columns, target_columns, scales, directions, discarded = [], [], [], [], []
    forward_evaluations = 2
    for k in range(config.count):
        if config.direction == 'random':
            raw = tuple(torch.randn(p.shape, generator=generator, dtype=p.dtype).to(p.device) for p in snap.values)
        else:
            direction_batch = target_batch if config.direction == 'target' else source_batch
            variables = tuple(p.detach().requires_grad_(True) for p in snap.values)
            logits = snap.logits(direction_batch, variables)
            xi = torch.randn(logits.shape, generator=generator, dtype=logits.dtype).to(logits.device)
            xi = center_logits(xi)
            scalar = (xi * logits).sum() / math.sqrt(len(logits))
            gradients = torch.autograd.grad(scalar, variables, create_graph=False, allow_unused=True)
            raw = tuple((g if g is not None else torch.zeros_like(p)).detach() for p, g in zip(variables, gradients))
            forward_evaluations += 1
        if any(not torch.isfinite(v).all() for v in raw):
            discarded.append(dict(probe=k, reason='nonfinite_direction')); continue
        tangent = snap.jvp(target_batch, raw)
        sensitivity = tangent.square().sum(-1).mean().sqrt()
        if not torch.isfinite(sensitivity) or sensitivity <= config.epsilon:
            discarded.append(dict(probe=k, reason='zero_or_nonfinite_target_sensitivity')); continue
        direction = tuple((v / (sensitivity + config.epsilon)).detach() for v in raw)
        if config.responses == 'linear':
            source_response = [snap.jvp(source_batch, direction)]
            target_response = [snap.jvp(target_batch, direction)]
        else:
            with torch.no_grad():
                plus = tuple(p + config.radius * v for p, v in zip(snap.values, direction))
                minus = tuple(p - config.radius * v for p, v in zip(snap.values, direction))
                sp, sm = snap.logits(source_batch, plus), snap.logits(source_batch, minus)
                tp, tm = snap.logits(target_batch, plus), snap.logits(target_batch, minus)
                source_response = [(sp - sm) / (2 * config.radius)]
                target_response = [(tp - tm) / (2 * config.radius)]
                if config.responses == 'odd_even':
                    source_response.append((sp + sm - 2 * source_base) / config.radius ** 2)
                    target_response.append((tp + tm - 2 * target_base) / config.radius ** 2)
            forward_evaluations += 4
        # Reject the complete direction in BOTH domains when any of its responses is invalid.
        if any(not torch.isfinite(v).all() for v in source_response + target_response):
            discarded.append(dict(probe=k, reason='nonfinite_response')); continue
        with torch.no_grad():
            s = torch.stack(source_response, dim=-1)
            t = torch.stack(target_response, dim=-1)
            s, t, d = shared_response_scale(s, t, config.epsilon)
        if any(not torch.isfinite(v).all() for v in (s, t, d)):
            discarded.append(dict(probe=k, reason='nonfinite_scaling')); continue
        source_columns.append(s); target_columns.append(t); scales.append(d)
        directions.append({n: v for n, v in zip(snap.names, direction)})
    with torch.no_grad():
        source_features = torch.cat(source_columns, -1) if source_columns else source_base.new_empty(*source_base.shape, 0)
        target_features = torch.cat(target_columns, -1) if target_columns else target_base.new_empty(*target_base.shape, 0)
        scale = torch.cat(scales) if scales else source_base.new_empty(0)
        intercept = None
        if predictor == 'responses_with_intercept':
            coefficients, intercept = fit_residual_with_intercept(source_features, residual, config.ridge)
        else:
            coefficients = fit_residual(source_features, residual, config.ridge)
        correction = target_features @ coefficients
        if intercept is not None:
            correction = correction+intercept
        reference = target_base + correction
        if not torch.isfinite(reference).all():
            raise FloatingPointError('Nonfinite target feedback reference')
        fitted = source_features @ coefficients
        if intercept is not None:
            fitted = fitted+intercept
        fit_error = (fitted - residual).square().sum(-1).mean()
        statistics = dict(candidate_directions=config.count, valid_directions=len(directions),
            response_columns=source_features.shape[-1], discarded=discarded,
            source_residual_rms=float(residual.square().sum(-1).mean().sqrt()),
            source_residual_fit_mse=float(fit_error), coefficient_norm=float(coefficients.norm()),
            target_correction_rms=float(correction.square().sum(-1).mean().sqrt()),
            explicit_forward_evaluations=forward_evaluations,
            status='feedback' if directions else 'source_only_fallback')
        if intercept is not None:
            statistics.update(predictor=predictor, intercept_norm=float(intercept.norm()),
                status='feedback' if directions else 'intercept_only_feedback')
    return Feedback(source_features.detach(), target_features.detach(), residual.detach(), coefficients.detach(),
                    correction.detach(), reference.detach(), scale.detach(), tuple(directions), statistics, intercept)


@torch.no_grad()
def build_mean_feedback(model, source_batch, source_labels, target_batch):
    """One source residual mean for every target graph; no response probes."""
    model.eval()
    snapshot = Snapshot(model)
    source_base, target_base = snapshot.logits(source_batch), snapshot.logits(target_batch)
    labels = source_labels.to(device=source_base.device, dtype=torch.long).reshape(-1)
    if (source_base.ndim != 2 or target_base.ndim != 2 or source_base.shape[1] < 2
            or source_base.shape[1] != target_base.shape[1] or len(labels) != len(source_base)):
        raise ValueError('Incompatible source labels and graph logits')
    if not torch.isfinite(source_base).all() or not torch.isfinite(target_base).all():
        raise FloatingPointError('Nonfinite mean-feedback logits')
    residual = F.one_hot(labels, source_base.shape[1]).to(source_base)-source_base.softmax(-1)
    intercept = residual.mean(0).detach()
    correction = intercept.expand_as(target_base)
    reference = target_base+correction
    statistics = dict(candidate_directions=0, valid_directions=0, response_columns=0, discarded=[],
        source_residual_rms=float(residual.square().sum(-1).mean().sqrt()),
        source_residual_fit_mse=float((intercept-residual).square().sum(-1).mean()),
        coefficient_norm=0., intercept_norm=float(intercept.norm()),
        target_correction_rms=float(correction.square().sum(-1).mean().sqrt()),
        explicit_forward_evaluations=2, status='mean_residual_feedback', predictor='mean_residual')
    return Feedback(source_base.new_empty(*source_base.shape,0), target_base.new_empty(*target_base.shape,0),
        residual.detach(), source_base.new_empty(0), correction.detach(), reference.detach(),
        source_base.new_empty(0), (), statistics, intercept)


def feedback_loss(model, target_batch, feedback):
    """Exactly 1/(2 b_T) times the sum across graphs AND class coordinates."""
    return .5 * (model(target_batch) - feedback.reference.detach()).square().sum(-1).mean()
