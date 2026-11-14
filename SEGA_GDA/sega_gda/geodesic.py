"""Design §3.10: finite piecewise-linear path energy, FULL unrolled derivatives."""
import math
import warnings
import torch
from torch import nn


class _RecomputedChunk(torch.autograd.Function):
    """Memory-bounded chunk evaluation with ONE recompute, in backward only.

    torch.utils.checkpoint (non-reentrant) discards every tensor saved inside
    the region, so each of the J_path inner `torch.autograd.grad` calls of the
    relaxation triggered a recompute of the chunk prefix already during the
    forward pass (O(J_path^2) extra solves per chunk). Here the forward pass
    runs the solve without any outer graph and stores only the chunk inputs;
    the backward pass re-runs the solve once with the complete unrolled graph
    (§3.10.2 derivative contract) and differentiates it. The metric primitives
    are an explicit input, so the recompute uses exactly the saved B. Outer
    derivatives are first order; create_graph through this function is not
    supported (the unrolled solver itself is still second order inside).
    """
    @staticmethod
    def forward(ctx, solver, x, y, B):
        ctx.solver = solver
        ctx.save_for_backward(x, y, B)
        return solver._solve(x, y, False, B=B)

    @staticmethod
    def backward(ctx, grad_output):
        x, y, B = ctx.saved_tensors
        needs = ctx.needs_input_grad[1:]
        with torch.enable_grad():
            leaves = [t.detach().requires_grad_(bool(n)) for t, n in zip((x, y, B), needs)]
            energy = ctx.solver._solve(leaves[0], leaves[1], True, B=leaves[2])
            wanted = [leaf for leaf, n in zip(leaves, needs) if n]
            gradients = (torch.autograd.grad(energy, wanted, grad_outputs=grad_output, allow_unused=True)
                         if wanted else ())
        results, iterator = [], iter(gradients)
        for leaf, n in zip(leaves, needs):
            gradient = next(iterator) if n else None
            results.append(torch.zeros_like(leaf) if n and gradient is None else gradient)
        return (None, *results)


def symmetric_quadrature(order, dtype=torch.float64):
    """Gauss-Legendre nodes/positive weights on [0,1], native Torch setup."""
    if not isinstance(order, int) or order < 1:
        raise ValueError('Quadrature order must be a positive integer')
    jacobi = torch.zeros(order, order, dtype=dtype)
    i = torch.arange(1, order, dtype=dtype)
    off = i / (4*i.square()-1).sqrt()
    jacobi += torch.diag(off, 1) + torch.diag(off, -1)
    eigenvalues, vectors = torch.linalg.eigh(jacobi)
    nodes = (eigenvalues+1)/2
    weights = vectors[0].square()
    nodes = (nodes + 1-nodes.flip(0))/2
    weights = (weights+weights.flip(0))/2
    return nodes, weights/weights.sum()


class GeodesicSolver(nn.Module):
    def __init__(self, metric, segments=4, quadrature=3, path_steps=8,
                 path_lr=.01, chunk_size=32, checkpoint_chunks=False):
        super().__init__()
        if (not all(isinstance(v, int) for v in [segments, path_steps, chunk_size])
                or segments < 1 or path_steps < 0 or chunk_size < 1):
            raise ValueError('Invalid path/chunk budget')
        if not math.isfinite(path_lr) or path_lr <= 0:
            raise ValueError('Path step size must be finite and positive')
        self.metric = metric
        self.config = dict(segments=segments, quadrature=quadrature, path_steps=path_steps,
                           path_lr=path_lr, chunk_size=chunk_size, checkpoint_chunks=checkpoint_chunks)
        # The inner relaxation is explicit gradient descent on the discrete energy.
        # For a locally constant metric g its Hessian w.r.t. the interior nodes
        # is 2L (T kron g) with T = tridiag(-1, 2, -1), so lambda_max < 8 L e^{c_g}
        # and the fixed step is stable only when path_lr * 8 L e^{c_g} < 2. The
        # identity background keeps the realised field below e^{c_g}, so this is
        # a conservative heuristic warning, not a hard error: a violated bound
        # shows up as non-finite candidate energies (rejected as INVALID), i.e.
        # a silently stalled space update rather than a wrong accepted one.
        bound = float(getattr(metric, 'spectral_bound', 0.))
        if path_steps and path_lr * 8 * segments * math.exp(bound) >= 2:
            warnings.warn(f'path_lr={path_lr} may be unstable for segments={segments}, '
                          f'spectral_bound={bound}: need path_lr < {2 / (8 * segments * math.exp(bound)):.4g} '
                          f'for the worst-case metric eigenvalue e^{{{bound}}}', RuntimeWarning, stacklevel=2)
        nodes, weights = symmetric_quadrature(quadrature)
        self.register_buffer('nodes', nodes)
        self.register_buffer('weights', weights)

    def energy(self, path, B=None):
        """One energy per path; never average paths inside the inner update.

        §3.10.1 discrete energy. The metric enters only through the quadratic
        form Delta^T g Delta at the quadrature points, evaluated by
        LocalMetricField.quadratic_form (matrix-vector Taylor polynomial of
        expm to double precision) rather than by materialising expm itself.
        """
        segments = path.shape[-2]-1
        delta = path[:, 1:]-path[:, :-1]
        points = path[:, :-1, None, :] + self.nodes[None, None, :, None]*delta[:, :, None, :]
        local = self.metric.quadratic_form(points, delta[:, :, None, :], B)
        return segments*(local*self.weights).sum(dim=(-1, -2))

    def _solve(self, x, y, track_grad, diagnostics=False, B=None):
        segments = self.config['segments']
        B = self.metric.B if B is None else B
        # Calling under no_grad is supported: internal optimization still needs AD.
        with torch.enable_grad():
            if not track_grad:
                # Evaluation: no outer graph through endpoints or primitives.
                x, y, B = x.detach(), y.detach(), B.detach()
            t = torch.arange(1, segments, dtype=x.dtype, device=x.device)/segments
            interior = x[:, None]+t[None, :, None]*(y-x)[:, None]
            if segments > 1 and not interior.requires_grad:
                interior.requires_grad_(True)
            history = []
            for _ in range(self.config['path_steps'] if segments > 1 else 0):
                path = torch.cat((x[:, None], interior, y[:, None]), dim=1)
                energy = self.energy(path, B)
                if diagnostics:
                    history.append(energy.detach())
                gradient, = torch.autograd.grad(energy.sum(), interior, create_graph=track_grad)
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError('Nonfinite internal path gradient')
                interior = interior-self.config['path_lr']*gradient
                # Evaluation alone discards inner graphs. Trainable calls MUST NOT.
                if not track_grad:
                    interior = interior.detach().requires_grad_(True)
            path = torch.cat((x[:, None], interior, y[:, None]), dim=1)
            energy = self.energy(path, B)
            if not torch.isfinite(energy).all():
                raise FloatingPointError('Nonfinite geodesic energy; no substitute distance')
            if diagnostics:
                history.append(energy.detach())
                if segments > 1:
                    residual, = torch.autograd.grad(energy.sum(), interior, retain_graph=track_grad)
                    residual = residual.detach().norm(dim=(-1, -2))
                else:
                    residual = energy.detach()*0
                return (energy if track_grad else energy.detach(), path.detach(),
                        torch.stack(history), residual)
            return energy if track_grad else energy.detach()

    def forward(self, x, y, *, diagnostics=False, track_grad=None):
        if torch.is_inference_mode_enabled():
            raise RuntimeError('Use no_grad, not inference_mode: inner path relaxation needs autograd')
        if x.shape[-1] != y.shape[-1] or x.shape[-1] != self.metric.dim:
            raise ValueError('Endpoint dimensions differ from the metric')
        if x.dtype != self.nodes.dtype or x.device != self.nodes.device:
            raise ValueError('Move solver and endpoints to the same dtype/device before use')
        if x.dtype != y.dtype or x.device != y.device or not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError('Endpoints must be finite and share dtype/device')
        self.metric.validate()
        x, y = torch.broadcast_tensors(x, y)
        shape = x.shape[:-1]
        x, y = x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1])
        if len(x) == 0:
            raise ValueError('No point pairs requested')
        if track_grad is None:
            track_grad = torch.is_grad_enabled() and (x.requires_grad or y.requires_grad
                         or any(p.requires_grad for p in self.metric.parameters()))
        if track_grad and not torch.is_grad_enabled():
            raise ValueError('Cannot request outer gradients inside no_grad')
        # Deterministic lexicographic orientation; reverse queries use reversed paths.
        different = x.detach() != y.detach()
        first = different.long().argmax(-1, keepdim=True)
        reverse = (x.detach().gather(-1, first) > y.detach().gather(-1, first)).squeeze(-1)
        a, b = torch.where(reverse[:, None], y, x), torch.where(reverse[:, None], x, y)
        energies, paths, histories, residuals = [], [], [], []
        chunk_size = self.config['chunk_size']
        for start in range(0, len(x), chunk_size):
            xc, yc = a[start:start+chunk_size], b[start:start+chunk_size]
            if diagnostics:
                if self.config['checkpoint_chunks']:
                    raise ValueError('Diagnostics retain paths; request separately from checkpointed execution')
                e, p, history, residual = self._solve(xc, yc, track_grad, True, B=self.metric.B)
                r = reverse[start:start+chunk_size]
                paths.append(torch.where(r[:, None, None], p.flip(1), p))
                histories.append(history)
                residuals.append(residual)
            elif track_grad and self.config['checkpoint_chunks']:
                # Explicit reentrant recompute: forward keeps only the chunk inputs,
                # backward re-runs the full unrolled solve with its graph once.
                e = _RecomputedChunk.apply(self, xc, yc, self.metric.B)
            else:
                e = self._solve(xc, yc, track_grad, B=self.metric.B)
            energies.append(e)
        values = torch.cat(energies).reshape(shape)
        if not diagnostics:
            return values
        return values, dict(path=torch.cat(paths).reshape(*shape, self.config['segments']+1, x.shape[-1]),
                            energy_history=torch.cat(histories, dim=1).reshape(-1, *shape),
                            stationarity=torch.cat(residuals).reshape(shape))


class NormalizedGeodesicDistance(nn.Module):
    """§3.4.2: fixed initial SOURCE coordinates, current differentiable scale."""
    def __init__(self, solver, reference_pairs):
        super().__init__()
        if (reference_pairs.ndim != 3 or reference_pairs.shape[1:] != (2, solver.metric.dim)
                or len(reference_pairs) < 1 or not torch.isfinite(reference_pairs).all()):
            raise ValueError('Expected finite fixed coordinate pairs [P,2,d]')
        if ((reference_pairs[:, 0]-reference_pairs[:, 1]).square().sum(-1) <= 0).any():
            raise ValueError('Scale reference coordinates must be noncoincident')
        self.solver = solver
        self.register_buffer('reference_pairs', reference_pairs.detach().clone())

    def scale2(self):
        value = self.solver(self.reference_pairs[:, 0], self.reference_pairs[:, 1]).mean()
        if not torch.isfinite(value) or value <= 0:
            raise FloatingPointError('Invalid current geodesic scale; no epsilon fallback')
        return value

    def forward(self, x, y):
        # Recompute every call: no stale cache, no detached denominator in space steps.
        scale = self.scale2()
        return self.solver(x, y)/scale

    def pairwise(self, x, y):
        return self(x[:, None, :], y[None, :, :])
