"""Independent finite-grid normal-ray observer. Never used for acceptance."""
import torch
from torch.func import jvp
from tsba.classifier import ParameterMap


@torch.no_grad()
def margin_on_points(model, points, pairs):
    logits = model(points)
    pair_logits = logits.gather(1, pairs)
    return pair_logits[:, 0] - pair_logits[:, 1]


@torch.no_grad()
def normal_roots(model, b, sigma):
    n = len(b.points)
    roots = b.old_margin.new_full((n,), float('nan'))
    if not n:
        return roots, []
    grid = torch.linspace(-.5 * sigma, .5 * sigma, 257, dtype=b.points.dtype)
    points = b.points[:, None] + grid[None, :, None] * b.normals[:, None]
    pairs = b.pairs[:, None].expand(-1, len(grid), -1)
    values = margin_on_points(model, points.reshape(-1, points.shape[-1]), pairs.reshape(-1, 2)).reshape(n, -1)
    statuses, brackets, ids = [], [], []
    for i, row in enumerate(values):
        exact = torch.where(row == 0)[0].tolist()
        crossings = torch.where(row[:-1] * row[1:] < 0)[0].tolist()
        count = len(exact) + len(crossings)
        if not torch.isfinite(row).all():
            statuses.append('nonfinite')
        elif count == 0:
            statuses.append('no_bracket')
        elif count > 1:
            statuses.append('ambiguous_multiple_roots')
        elif exact:
            roots[i] = grid[exact[0]]
            statuses.append('ok')
        else:
            j = crossings[0]
            brackets.append((grid[j], grid[j+1]))
            ids.append(i)
            statuses.append('ok')
    if ids:
        ids_t = torch.tensor(ids)
        intervals = torch.tensor(brackets, dtype=grid.dtype)
        lo, hi = intervals[:, 0].clone(), intervals[:, 1].clone()
        at_lo = margin_on_points(model, b.points[ids_t] + lo[:, None] * b.normals[ids_t], b.pairs[ids_t])
        for _ in range(60):
            mid = (lo + hi) / 2
            at_mid = margin_on_points(model, b.points[ids_t] + mid[:, None] * b.normals[ids_t], b.pairs[ids_t])
            left = (at_lo * at_mid <= 0)
            hi = torch.where(left, mid, hi)
            lo = torch.where(left, lo, mid)
            at_lo = torch.where(left, at_lo, at_mid)
            if float((hi-lo).max()) <= 1e-8 * sigma:
                break
        roots[ids_t] = (lo + hi) / 2
    valid = torch.where(torch.isfinite(roots))[0]
    if len(valid):
        logits = model(b.points[valid] + roots[valid, None] * b.normals[valid])
        for index, out in zip(valid.tolist(), logits):
            pair = b.pairs[index]
            if len(out) > 2:
                others = torch.ones_like(out, dtype=torch.bool)
                others[pair] = False
                if (out[others] > out[pair].min() + 1e-7).any():
                    roots[index] = float('nan')
                    statuses[index] = 'inactive_pair'
    return roots, statuses


def rms(x):
    return float(x.square().mean().sqrt()) if len(x) else None


def compare_geometry(model, b, before, after, sigma):
    """Restores the caller's model even when the observer fails."""
    params = ParameterMap(model)
    original = params.vector()
    try:
        params.commit(before)
        old, old_status = normal_roots(model, b, sigma)
        def normalized(vector):
            logits = params.logits(vector, b.points).gather(1, b.pairs)
            return (logits[:, 0] - logits[:, 1]) / b.normal_norm
        with torch.no_grad():
            v = normalized(after) - b.old_margin / b.normal_norm
        linear = jvp(normalized, (before,), (after - before,))[1].detach()
        params.commit(after)
        new, new_status = normal_roots(model, b, sigma)
        geom = new - old
        valid = torch.isfinite(geom)
        measured = geom[valid]
        nonzero = valid & (v.abs() > 1e-12 * sigma) & (geom.abs() > 1e-12 * sigma)
        metrics = dict(anchors=len(b), measured=int(valid.sum()),
                       E_lin=rms((v-linear)/sigma), E_geom=rms((geom[valid]+v[valid])/sigma),
                       E_request=rms((geom[valid]+b.zeta[valid])/sigma),
                       displacement_sign_agreement=float(((geom[nonzero]*v[nonzero]) < 0).double().mean()) if nonzero.any() else None,
                       old_status=old_status, new_status=new_status)
        return metrics, dict(v=v, linear=linear, old_root=old, new_root=new, geometric=geom, valid=valid)
    finally:
        params.commit(original)
