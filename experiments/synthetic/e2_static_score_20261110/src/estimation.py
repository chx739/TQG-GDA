"""Static score diagnostics: fixed queries, paired noise, exactly 400 updates.

Research-only helpers. Production score training and classifier code are unchanged.
"""
import copy
import hashlib
import math
import torch
from tsba.score import ScoreNetwork, denoising_loss
from tsba.utils import finite
from experiments.synthetic.e1_e2_20261109.src.distributions import generator, sample_gmm

SETTINGS = (('n512_s1', 512, 1.), ('n2048_s1', 2048, 1.),
            ('n8192_s1', 8192, 1.), ('n512_s05', 512, .5), ('n512_s2', 512, 2.))


def tensor_hash(x):
    return hashlib.sha256(str((tuple(x.shape), x.dtype)).encode() +
                          x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def enlarged_pool(target, coordinates, seed, maximum=8192):
    """Retain the archived 512 inputs exactly; append one independent nested pool."""
    raw, _ = sample_gmm(maximum - len(target), target.shape[1], seed + 9100)
    return torch.cat((target, coordinates(raw).detach()))


def schedules(dim, seed, updates=400, batch=128, baseline_n=512):
    """Replay production's baseline stream; all arms share these exact noises.

    Larger pools use a separate shuffle stream, so changing N cannot change noise.
    Without-replacement epochs truncate at the same update budget in every arm.
    """
    if baseline_n % batch:
        raise ValueError('baseline pool must divide into complete paired batches')
    rng = generator(seed + 102)
    ids, noise = [], []
    while len(ids) < updates:
        for part in torch.randperm(baseline_n, generator=rng).split(batch):
            if len(ids) == updates:
                break
            ids.append(part)
            noise.append(torch.randn(batch, dim, generator=rng, dtype=torch.float64))
    result = {baseline_n: torch.stack(ids)}
    for n in (2048, 8192):
        rng = generator(seed + 9102)
        blocks = []
        while len(blocks) < updates:
            blocks.extend(torch.randperm(n, generator=rng).split(batch))
        result[n] = torch.stack(blocks[:updates])
    val_noise = torch.randn(baseline_n, dim, generator=generator(seed + 103), dtype=torch.float64)
    return dict(ids=result, noise=torch.stack(noise), validation_noise=val_noise)


def new_model(dim, cfg):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg.seed + 101)
        model = ScoreNetwork(dim, cfg.score_hidden).double()
    return model


def fit(target, sigma, cfg, schedule, log=lambda _: None, checkpoint=None):
    """Same DSM/Adam as production; primary output is the last fixed-budget state.

    A secondary checkpoint follows production's validation rule on the original
    512 inputs and noise. Its validation cadence is always four updates.
    """
    model = new_model(target.shape[1], cfg)
    initial = copy.deepcopy(model.state_dict())
    opt = torch.optim.Adam(model.parameters(), lr=cfg.score_lr)
    ids = schedule['ids'][len(target)]
    best, best_step, best_state = math.inf, None, None
    history, validations = [], []
    try:
        for step, (batch_ids, noise) in enumerate(zip(ids, schedule['noise']), 1):
            loss = denoising_loss(model, target[batch_ids], sigma, noise)
            finite(loss)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            finite(*(p.grad for p in model.parameters()))
            opt.step()
            finite(*model.parameters())
            history.append(dict(update=step, loss=float(loss.detach())))
            if step % 4 == 0:
                with torch.no_grad():
                    value = sum(float(denoising_loss(model, xb, sigma, eb)) * len(xb)
                                for xb, eb in zip(target[:512].split(cfg.score_batch),
                                                  schedule['validation_noise'].split(cfg.score_batch))) / 512
                finite(value)
                validations.append(dict(update=step, noise_validation_loss=value))
                if value < best:
                    best, best_step, best_state = value, step, copy.deepcopy(model.state_dict())
            if step % 100 == 0:
                log(f'update={step} DSM={float(loss.detach()):.8g} validation={value:.8g}')
    finally:
        # On failure preserve the last live weights, optimizer and partial log.
        if checkpoint is not None:
            checkpoint(dict(final_state=copy.deepcopy(model.state_dict()), initial_state=initial,
                            selected_state=best_state, optimizer=opt.state_dict(),
                            updates=len(history), selected_update=best_step,
                            train_history=history, validation_history=validations))
    model.requires_grad_(False).eval()
    return model, dict(updates=len(history), selected_update=best_step,
                       validation_loss=best, train_history=history, validation_history=validations), best_state


@torch.no_grad()
def evaluate(model, points, chunk=32):
    # Keep exact empirical KDE memory bounded even for 8192 components.
    return torch.cat([model(b) for b in points.split(chunk)]) if len(points) else torch.empty_like(points)


def sign_metrics(a, r, mask):
    recalls, result = [], dict(eligible=int(mask.sum()), excluded=len(mask)-int(mask.sum()))
    result['agreement'] = float((a[mask].sign() == r[mask].sign()).double().mean()) if mask.any() else None
    for sign, label in ((1, 'positive'), (-1, 'negative')):
        chosen = mask & (r.sign() == sign)
        recall = float((a[chosen].sign() == sign).double().mean()) if chosen.any() else None
        result[label+'_count'], result[label+'_recall'] = int(chosen.sum()), recall
        recalls.append(recall)
    result['balanced'] = sum(recalls)/2 if all(v is not None for v in recalls) else None
    return result


def rmse(x):
    return float(x.square().mean().sqrt())


def audit_scores(points, normals, values, sigma, scale, cfg, common_mask=None, raw_x1=None):
    """§3.6 normal score and §3.7 clipped request, with explicit physical units."""
    if not all(torch.isfinite(x).all() for x in values.values()):
        raise ValueError('nonfinite score')
    arrays = {'score_'+k: v for k, v in values.items()}
    metrics, signals = {}, {}
    if normals is not None:
        gamma = {k: (v*normals).sum(-1) for k, v in values.items()}
        zeta = {k: (cfg.tau*sigma**2*g).clamp(-cfg.request_multiplier*sigma,
                                             cfg.request_multiplier*sigma) for k, g in gamma.items()}
        arrays.update({'gamma_'+k: v for k, v in gamma.items()})
        arrays.update({'zeta_'+k: v for k, v in zeta.items()})
        for k, g in gamma.items():
            # Boundary displacement is -zeta*n; the correct raw x1 direction is toward zero.
            desired = (raw_x1*normals[:, 0]).sign()
            eligible = (raw_x1.abs() > 1e-10) & (normals[:, 0].abs() > 1e-10)
            signals[k] = dict(scaled_normal_rms=rmse(sigma*g), raw_normal_rms=rmse(g/scale),
                              request_raw_rms=rmse(scale*zeta[k]),
                              clipped_fraction=float((cfg.tau*sigma**2*g.abs() >= cfg.request_multiplier*sigma).double().mean()),
                              toward_target_boundary=float((g[eligible].sign() == desired[eligible]).double().mean()) if eligible.any() else None)
    for a, r in (('learned', 'empirical'), ('empirical', 'population'),
                 ('learned', 'population'), ('zero', 'population'), ('selected', 'population')):
        diff = values[a]-values[r]
        row = dict(vector_mse_scaled=float((sigma*diff).square().mean()),
                   vector_mse_raw=float((diff/scale).square().mean()))
        if normals is not None:
            da, dr = gamma[a], gamma[r]
            row.update(normal_rmse_scaled=rmse(sigma*(da-dr)), normal_rmse_raw=rmse((da-dr)/scale),
                       request_rmse_sigma=rmse((zeta[a]-zeta[r])/sigma),
                       request_rmse_raw=rmse(scale*(zeta[a]-zeta[r])),
                       request_magnitude_rmse_raw=rmse(scale*(zeta[a].abs()-zeta[r].abs())),
                       sign=sign_metrics(da, dr, (sigma*dr).abs() > 1e-3))
            if r == 'population':
                row['common_sign'] = sign_metrics(da, dr, common_mask)
        metrics[a+'_vs_'+r] = row
    return dict(count=len(points), comparisons=metrics, signals=signals), arrays
