"""Read-only E1 mechanism measurements; never imported by the trainer.

Feedback is constructed without target labels in feedback.py. This module
receives already constructed feedback and separately supplied diagnostic labels.
Every tentative update is stateless and starts from the same model snapshot.
"""
import hashlib
import math
import statistics

import torch
from torch.func import functional_call
from torch.nn import functional as F

from .feedback import feedback_loss


def heldout_response_features(snapshot, batch, feedback, config):
    """Extend the SAME response functions to new graphs; reuse fitted scales.

    Never re-fit directions or recompute RMS on held-out source/target inputs.
    This is diagnostic-only and never reads labels.
    """
    with torch.no_grad():
        base = snapshot.logits(batch)
    columns = []
    for values in feedback.directions:
        direction = tuple(values[name] for name in snapshot.names)
        if config.responses == 'linear':
            columns.append(snapshot.jvp(batch,direction).unsqueeze(-1))
        else:
            with torch.no_grad():
                plus=tuple(p+config.radius*v for p,v in zip(snapshot.values,direction))
                minus=tuple(p-config.radius*v for p,v in zip(snapshot.values,direction))
                zp,zm=snapshot.logits(batch,plus),snapshot.logits(batch,minus)
                response=[(zp-zm)/(2*config.radius)]
                if config.responses=='odd_even':response.append((zp+zm-2*base)/config.radius**2)
                columns.append(torch.stack(response,dim=-1))
    if not columns:
        return base.new_empty(*base.shape,0)
    with torch.no_grad():
        features=torch.cat(columns,-1)/feedback.scales
    if not torch.isfinite(features).all():
        raise FloatingPointError('Nonfinite held-out diagnostic response')
    return features.detach()


def reserve_target_ids(ids, fraction, seed):
    """Label-blind, persistent diagnostic/final split, independent of model seed."""
    if not 0 < fraction < 1 or len(ids) < 3 or len(set(ids)) != len(ids):
        raise ValueError('Need unique target IDs and at least three graphs')
    order = torch.randperm(len(ids), generator=torch.Generator().manual_seed(seed)).tolist()
    n = math.floor(len(ids) * fraction)
    if n < 2 or n >= len(ids):
        raise ValueError('Reserve >=2 diagnostic graphs and >=1 final graph')
    return tuple(sorted(ids[i] for i in order[:n])), tuple(sorted(ids[i] for i in order[n:]))


def feedback_digest(feedback):
    """Evidence that changing the diagnostic labels cannot change the feedback."""
    digest = hashlib.sha256()
    for value in (feedback.source_features, feedback.target_features,
                  feedback.coefficients, feedback.correction, feedback.reference):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _gradients(loss, parameters):
    values = torch.autograd.grad(loss, parameters, allow_unused=True)
    result = tuple(torch.zeros_like(p) if v is None else v.detach()
                   for p, v in zip(parameters, values))
    if any(not torch.isfinite(v).all() for v in result):
        raise FloatingPointError('Nonfinite diagnostic gradient')
    return result


def _norm(values):
    value = torch.stack([v.double().square().sum() for v in values]).sum().sqrt()
    if not torch.isfinite(value):
        raise FloatingPointError('Nonfinite diagnostic vector norm')
    return value


def measure_feedbacks(model, target_query, target_eval, eval_labels, feedbacks,
                      step_fractions=(1e-4, 1e-3), gradient_epsilon=1e-12):
    """Compare ALL methods to the same held-out CE gradient and equal-length steps.

    No optimizer step, parameter assignment, or .backward(). Zero feedback counts
    as cosine=0 and CE decrease=0. If the oracle gradient vanishes, all methods in
    this paired block get cosine=None (the block is uninformative for alignment).
    Positive cosine means -feedback_gradient locally decreases held-out CE.
    """
    if (not step_fractions or any(not math.isfinite(x) or x <= 0 for x in step_fractions)
            or not math.isfinite(gradient_epsilon) or gradient_epsilon <= 0):
        raise ValueError('Step fractions and gradient epsilon must be finite and positive')
    model.eval()
    named = tuple(model.named_parameters())
    parameters = tuple(p for _, p in named)
    logits = model(target_eval)
    labels = eval_labels.to(device=logits.device, dtype=torch.long).reshape(-1)
    oracle_loss = F.cross_entropy(logits, labels)
    if not torch.isfinite(oracle_loss):
        raise FloatingPointError('Nonfinite diagnostic CE')
    oracle = _gradients(oracle_loss, parameters)
    oracle_norm = float(_norm(oracle))
    baseline_ce = float(oracle_loss.detach())
    informative = oracle_norm > gradient_epsilon
    parameter_scale = max(float(_norm(tuple(p.detach() for p in parameters))), 1.)
    snapshot = {n: p.detach().clone() for n, p in named}
    buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
    rows = {}
    for name, feedback in feedbacks.items():
        gradient = _gradients(feedback_loss(model, target_query, feedback), parameters)
        norm = float(_norm(gradient))
        zero = norm <= gradient_epsilon
        cosine = None
        if informative:
            dot = sum(float((a.double() * b.double()).sum()) for a, b in zip(gradient, oracle))
            cosine = max(-1., min(1., dot / (norm * oracle_norm))) if not zero else 0.
        steps = []
        for fraction in step_fractions:
            length = fraction * parameter_scale
            if zero:
                after = baseline_ce
            else:
                candidate = {n: snapshot[n] - (length / norm) * g for (n, _), g in zip(named, gradient)}
                with torch.no_grad():
                    candidate_logits = functional_call(model, (candidate, {n: b.clone() for n, b in buffers.items()}),
                                                       (target_eval,), strict=True)
                    after = float(F.cross_entropy(candidate_logits, labels))
                if not math.isfinite(after):
                    raise FloatingPointError('Nonfinite CE after tentative diagnostic step')
            steps.append(dict(fraction=fraction, parameter_step_length=0. if zero else length,
                              ce_after=after, ce_decrease=baseline_ce - after))
        rows[name] = dict(cosine=cosine, feedback_gradient_norm=norm, zero_feedback=zero,
                          feedback_sha256=feedback_digest(feedback), steps=steps,
                          query_statistics=feedback.statistics)
    # The zero-feedback reference is exact; it does not consume an optimizer step.
    rows['no_feedback'] = dict(cosine=0. if informative else None, feedback_gradient_norm=0.,
        zero_feedback=True, feedback_sha256=None, query_statistics={},
        steps=[dict(fraction=f, parameter_step_length=0., ce_after=baseline_ce, ce_decrease=0.)
               for f in step_fractions])
    return dict(oracle_gradient_norm=oracle_norm, informative=informative,
                baseline_eval_ce=baseline_ce, methods=rows)


def summarize_blocks(blocks):
    """Average blocks WITHIN one model seed; blocks are not independent seed runs."""
    if not blocks:
        raise ValueError('No diagnostic blocks')
    methods = list(blocks[0]['methods'])
    summary = dict(blocks=len(blocks), informative_blocks=sum(b['informative'] for b in blocks), methods={})
    summary['uninformative_fraction'] = 1 - summary['informative_blocks'] / len(blocks)
    for method in methods:
        rows = [b['methods'][method] for b in blocks]
        cosines = [r['cosine'] for r in rows if r['cosine'] is not None]
        steps = []
        for i, step in enumerate(rows[0]['steps']):
            values = [r['steps'][i]['ce_decrease'] for r in rows]
            steps.append(dict(fraction=step['fraction'], mean_ce_decrease=statistics.mean(values),
                              fraction_improved=sum(x > 0 for x in values) / len(values)))
        summary['methods'][method] = dict(mean_cosine=statistics.mean(cosines) if cosines else None,
            zero_feedback_fraction=sum(r['zero_feedback'] for r in rows) / len(rows), steps=steps)
    target = summary['methods']['target']; random = summary['methods']['random']
    summary['target_minus_random'] = dict(
        mean_cosine=(target['mean_cosine'] - random['mean_cosine'])
                    if target['mean_cosine'] is not None else None,
        steps=[dict(fraction=t['fraction'], mean_ce_decrease_gain=t['mean_ce_decrease'] - r['mean_ce_decrease'])
               for t, r in zip(target['steps'], random['steps'])])
    return summary
