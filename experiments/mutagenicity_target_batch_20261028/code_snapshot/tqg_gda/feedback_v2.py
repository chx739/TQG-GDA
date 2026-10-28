"""Full target tangent space and direct parameter feedback, design v2.

No finite differences, target labels, trainable kernel, or differentiation
through the calibration solve. Only parameter_feedback_loss trains the GNN.
"""
from dataclasses import dataclass, field
import math

import torch
from torch.func import functional_call
from torch.nn import functional as F

from .calibration_v2 import CalibrationConfig, check_psd, solve_calibration


@dataclass(frozen=True)
class V2Config:
    scale_epsilon: float = 1e-8
    signature: str = 'representation_response'
    calibration: dict = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.scale_epsilon, bool) or not math.isfinite(self.scale_epsilon) or self.scale_epsilon <= 0:
            raise ValueError('scale_epsilon must be finite and positive')
        if self.signature not in ('representation_response', 'representation_only'):
            raise ValueError('Unknown v2 signature; representation_only is an ablation')
        CalibrationConfig(**self.calibration)


def contrast_basis(classes, *, dtype, device):
    """Fixed Helmert contrasts Q_C, with Q.T Q=I and Q Q.T=P_C."""
    if classes < 2:
        raise ValueError('At least two classes are required')
    q = torch.zeros(classes, classes-1, dtype=dtype, device=device)
    for column in range(classes-1):
        scale = math.sqrt((column+1) * (column+2))
        q[:column+1, column] = 1/scale
        q[column+1, column] = -(column+1)/scale
    return q


class ParameterSnapshot:
    """A functional all-trainable-parameter snapshot, including the head."""
    def __init__(self, model):
        self.model = model
        self.all_parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
        active = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
        if not active:
            raise ValueError('The model has no trainable parameters')
        self.names = tuple(name for name, _ in active)
        self.shapes = tuple(value.shape for _, value in active)
        self.counts = tuple(value.numel() for _, value in active)
        self.flat = torch.cat([value.detach().reshape(-1) for _, value in active]).clone()
        if self.flat.dtype not in (torch.float32, torch.float64):
            raise ValueError('v2 Jacobians require float32 or float64 parameters')
        if any(value.dtype != self.flat.dtype or value.device != self.flat.device for _, value in active):
            raise ValueError('All trainable parameters must share dtype and device')
        self.buffers = {name: value.detach().clone() for name, value in model.named_buffers()}

    def parameters(self, flat):
        values = dict(self.all_parameters)
        offset = 0
        for name, shape, count in zip(self.names, self.shapes, self.counts):
            values[name] = flat[offset:offset+count].reshape(shape)
            offset += count
        return values

    def logits(self, batch, flat=None):
        parameters = self.parameters(self.flat if flat is None else flat)
        buffers = {name: value.clone() for name, value in self.buffers.items()}
        return functional_call(self.model, (parameters, buffers), (batch,), strict=True)

    def representation(self, batch):
        parameters = {name[len('encoder.'):]: value for name, value in self.all_parameters.items()
                      if name.startswith('encoder.')}
        buffers = {name[len('encoder.'):]: value.clone() for name, value in self.buffers.items()
                   if name.startswith('encoder.')}
        return functional_call(self.model.encoder, (parameters, buffers), (batch,), strict=True)

    def jacobian(self, batch, contrasts):
        variable = self.flat.detach().requires_grad_(True)
        output = (self.logits(batch, variable) @ contrasts).reshape(-1)
        rows = []
        for index, value in enumerate(output):
            gradient = None
            if value.requires_grad:
                gradient, = torch.autograd.grad(value, variable, retain_graph=index+1 < len(output),
                                               create_graph=False, allow_unused=True)
            rows.append(torch.zeros_like(variable) if gradient is None else gradient.detach())
        return torch.stack(rows)

    def response(self, batch, contrasts, basis):
        columns = []
        for direction in basis.mT:
            _, response = torch.autograd.functional.jvp(
                lambda flat: self.logits(batch, flat) @ contrasts,
                self.flat, direction.contiguous(), create_graph=False, strict=False)
            columns.append(response.detach())
        return torch.stack(columns, dim=-1)


@torch.no_grad()
def target_space(jacobian):
    """Keep all numerically nonzero singular directions; no rank budget."""
    if not bool(torch.isfinite(jacobian).all()):
        raise FloatingPointError('Nonfinite target Jacobian')
    _, singular, vh = torch.linalg.svd(jacobian, full_matrices=False)
    tolerance = max(jacobian.shape) * torch.finfo(jacobian.dtype).eps * float(singular[0])
    rank = int((singular > tolerance).sum())
    return vh[:rank].mT.contiguous().detach(), dict(
        rank=rank, rank_tolerance=tolerance, largest_singular_value=float(singular[0]),
        smallest_retained_singular_value=float(singular[rank-1]) if rank else None,
        rank_policy='all_numerically_nonzero', target_jacobian_rows=jacobian.shape[0],
        parameter_count=jacobian.shape[1])


@torch.no_grad()
def signature_kernels(source_h, target_h, source_a, target_a, config=V2Config()):
    """Source-only median bandwidths, shared by SS, ST and TT kernels."""
    source_h, target_h, source_a, target_a = [
        value.detach().to(dtype=torch.float64) for value in (source_h, target_h, source_a, target_a)]
    if len(source_h) < 2:
        raise ValueError('At least two source query graphs are required for v2 scales')
    if any(not bool(torch.isfinite(x).all()) for x in (source_h, target_h, source_a, target_a)):
        raise FloatingPointError('Nonfinite complete-graph signature')
    source_a, target_a = source_a.flatten(1), target_a.flatten(1)
    floor = config.scale_epsilon**2
    if floor == 0 or not math.isfinite(floor):
        raise ValueError('scale_epsilon squared must be representable and positive')
    scale_h = torch.quantile(torch.pdist(source_h).square(), .5).clamp_min(floor)
    scale_a = torch.quantile(torch.pdist(source_a).square(), .5).clamp_min(floor)
    h, a = torch.cat((source_h, target_h)), torch.cat((source_a, target_a))
    distances = torch.cdist(h, h, compute_mode='donot_use_mm_for_euclid_dist').square() / (2*scale_h)
    if config.signature == 'representation_response':
        distances += torch.cdist(a, a, compute_mode='donot_use_mm_for_euclid_dist').square() / (2*scale_a)
    kernel = torch.exp(-distances)
    check_psd(kernel, 'Complete source-target kernel')
    ns = len(source_h)
    return kernel[:ns, :ns], kernel[:ns, ns:], kernel[ns:, ns:], dict(
        scale_h_squared=float(scale_h), scale_a_squared=float(scale_a),
        scale_h_at_floor=bool(scale_h <= floor), scale_a_at_floor=bool(scale_a <= floor),
        signature=config.signature, source_target_kernel_mean=float(kernel[:ns, ns:].mean()),
        source_target_kernel_max=float(kernel[:ns, ns:].max()))


@torch.no_grad()
def query_matrices(k_st, k_tt, target_response):
    """Compute H_ST and H_TT in graph-major order without expanded block kernels."""
    b = target_response.to(dtype=torch.float64) / len(target_response)
    h_st = torch.einsum('ij,jca->ica', k_st, b).flatten(0, 1)
    h_tt = torch.einsum('ica,ij,jcb->ab', b, k_tt, b)
    return h_st, h_tt


@dataclass
class GradientFeedback:
    parameter_names: tuple
    snapshot: torch.Tensor
    basis: torch.Tensor
    coefficients: torch.Tensor
    contrasts: torch.Tensor
    source_response: torch.Tensor
    target_response: torch.Tensor
    source_residual: torch.Tensor
    readout: torch.Tensor
    statistics: dict

    @property
    def gradient(self):
        return -(self.basis.double() @ self.coefficients).to(self.snapshot.dtype)


def build_gradient_feedback(model, source_batch, source_labels, target_batch, config=V2Config()):
    """Build a stopped gradient estimate; target labels are not an argument."""
    if getattr(target_batch, 'y', None) is not None:
        raise PermissionError('Target training inputs must not contain labels')
    model.eval()
    snapshot = ParameterSnapshot(model)
    if len(source_labels) < 2 or source_labels.ndim != 1 or source_labels.dtype != torch.long:
        raise ValueError('v2 requires at least two integer source query labels')
    # Neither source nor target labels are exposed to functional query forwards.
    source_inputs = source_batch.clone()
    if hasattr(source_inputs, 'y'):
        source_inputs.y = None
    basis = snapshot.flat.new_zeros((len(snapshot.flat), 0))
    empty = snapshot.flat.new_zeros(0, dtype=torch.float64)
    contrasts = snapshot.flat.new_zeros((0, 0))
    source_a = target_a = snapshot.flat.new_zeros((0, 0, 0))
    residual, readout = empty, empty.reshape(0, 0)
    statistics = dict(version='v2', status='initializing', target_labels_used=False)
    try:
        with torch.no_grad():
            source_logits = snapshot.logits(source_inputs)
            target_logits = snapshot.logits(target_batch)
            if (source_logits.ndim != 2 or target_logits.ndim != 2
                    or len(source_logits) != len(source_labels) or not len(target_logits)
                    or source_logits.shape[1] != target_logits.shape[1]):
                raise ValueError('Incompatible source/target classifier output dimensions')
            if bool((source_labels < 0).any()) or bool((source_labels >= source_logits.shape[1]).any()):
                raise ValueError('Source label outside the shared class space')
            if not all(bool(torch.isfinite(x).all()) for x in (source_logits, target_logits)):
                raise FloatingPointError('Nonfinite snapshot logits')
            contrasts = contrast_basis(source_logits.shape[1], dtype=source_logits.dtype, device=source_logits.device)
        # Needs first-order AD even when the caller only wants stopped results.
        with torch.enable_grad():
            jacobian = snapshot.jacobian(target_batch, contrasts)
            basis, rank_statistics = target_space(jacobian)
            statistics.update(rank_statistics)
            if basis.shape[1]:
                source_a = snapshot.response(source_inputs, contrasts, basis)
                target_a = (jacobian @ basis).reshape(len(target_logits), contrasts.shape[1], -1).detach()
        if not basis.shape[1]:
            statistics['status'] = 'zero_rank_source_only'
        else:
            with torch.no_grad():
                source_h, target_h = snapshot.representation(source_inputs), snapshot.representation(target_batch)
                k_ss, k_st, k_tt, kernel_statistics = signature_kernels(source_h, target_h, source_a, target_a, config)
                statistics.update(kernel_statistics)
                h_st, h_tt = query_matrices(k_st, k_tt, target_a)
                readout, calibration = solve_calibration(k_ss, h_st, h_tt, CalibrationConfig(**config.calibration))
                # The labels enter the readout value only after W has been solved.
                residual = ((F.one_hot(source_labels, source_logits.shape[1]).to(source_logits)
                             - source_logits.softmax(-1)) @ contrasts).double().reshape(-1).detach()
                coefficients = readout @ residual  # No additional division by b_S.
                if not bool(torch.isfinite(coefficients).all()):
                    raise FloatingPointError('Nonfinite target gradient coefficients')
                statistics.update(calibration=calibration, status='gradient_feedback',
                    source_query_graphs=len(source_logits), target_graphs=len(target_logits),
                    residual_norm=float(residual.norm()), coefficient_norm=float(coefficients.norm()))
                return GradientFeedback(snapshot.names, snapshot.flat.detach(), basis.detach(), coefficients.detach(),
                    contrasts.detach(), source_a.detach(), target_a.detach(), residual, readout.detach(), statistics)
    except (FloatingPointError, torch.linalg.LinAlgError) as error:
        statistics.update(status='numerical_fallback_source_only', numerical_error=str(error))
    coefficients = snapshot.flat.new_zeros(basis.shape[1], dtype=torch.float64)
    return GradientFeedback(snapshot.names, snapshot.flat.detach(), basis.detach(), coefficients,
        contrasts.detach(), source_a.detach(), target_a.detach(), residual, readout.detach(), statistics)


def parameter_feedback_loss(model, feedback):
    """Section 3.13: at the snapshot the gradient equals -V d, without /r."""
    parameters = dict(model.named_parameters())
    current_names = tuple(name for name, value in model.named_parameters() if value.requires_grad)
    if current_names != feedback.parameter_names:
        raise ValueError('Trainable parameter layout changed after v2 query construction')
    flat = torch.cat([parameters[name].reshape(-1) for name in feedback.parameter_names])
    displacement = flat.double() - feedback.snapshot.detach().double()
    coordinates = feedback.basis.detach().double().mT @ displacement
    return .5 * (coordinates - feedback.coefficients.detach()).square().sum()
