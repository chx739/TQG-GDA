"""Stages B/C: fresh stopped query, then exactly one joint GNN update.

The organization follows SEGA's problem/inner-solve/execute separation. The
execution rule is TQG's: v1 §§3.10–3.11 or v2 §3.13, without geometry backtracking.
"""
import time

import torch
from torch.nn import functional as F

from .pretrain import check_finite
from .v1.query import build_feedback, feedback_loss
from .v2.query import build_gradient_feedback, parameter_feedback_loss


def _flat(values, parameters):
    return torch.cat([(v if v is not None else torch.zeros_like(p)).reshape(-1)
                      for v, p in zip(values, parameters)])


class Adapter:
    def __init__(self, model, config):
        self.model, self.config, self.steps = model, config, 0
        if not all(p.requires_grad for p in model.parameters()):
            raise ValueError('Canonical TQG adapts both the encoder and linear head')
        factory = torch.optim.SGD if config.adapt.optimizer == 'sgd' else torch.optim.Adam
        self.optimizer = factory(model.parameters(), lr=config.adapt.lr, weight_decay=0.)
        self.generators = {role: torch.Generator().manual_seed(config.seed+offset)
                           for role, offset in (('fit', 201), ('query', 202), ('target', 203), ('probes', 301))}

    def step(self, task):
        started = time.perf_counter()
        cfg, model = self.config, self.model
        model.eval()  # deterministic probes AND loss forwards; affine BN remains trainable
        p = next(model.parameters())
        fit, fit_ids = task.sample('fit', cfg.adapt.fit_batch_size, self.generators['fit'], p.device, p.dtype)
        fit_labels, fit.y = fit.y, None
        query_ids, target_ids = (), ()
        feedback, query_info = None, dict(status='source_only_baseline')
        self.optimizer.zero_grad(set_to_none=True)
        if cfg.version != 'source_only':
            query, query_ids = task.sample('query', cfg.adapt.query_batch_size, self.generators['query'], p.device, p.dtype)
            target, target_ids = task.sample('target', cfg.adapt.target_batch_size, self.generators['target'], p.device, p.dtype)
            labels, query.y = query.y, None
            try:
                if cfg.version == 'v1':
                    feedback = build_feedback(model, query, labels, target, cfg.v1, self.generators['probes'])
                else:
                    feedback = build_gradient_feedback(model, query, labels, target, cfg.v2)
                query_info = feedback.statistics
            except (FloatingPointError, torch.linalg.LinAlgError) as error:
                # A finite source CE can still be executed after a failed query.
                query_info = dict(status='numerical_fallback_source_only', numerical_error=str(error))

        source_loss = F.cross_entropy(model(fit), fit_labels)
        parameters = tuple(model.parameters())
        before = _flat([v.detach() for v in parameters], parameters).clone()
        if feedback is None:
            auxiliary = source_loss * 0.
        elif cfg.version == 'v1':
            auxiliary = feedback_loss(model, target, feedback)
        else:
            auxiliary = parameter_feedback_loss(model, feedback)
        total = source_loss + cfg.adapt.feedback_weight*auxiliary
        check_finite(model, total)
        auxiliary_parts = torch.autograd.grad(auxiliary, parameters, retain_graph=True, allow_unused=True)
        auxiliary_gradient = _flat(auxiliary_parts, parameters).detach()
        execution = dict(optimizer=cfg.adapt.optimizer, feedback_gradient_norm=float(auxiliary_gradient.norm()),
                         updates_per_query=1, weight_decay=0.)
        if cfg.version == 'v2' and feedback is not None:
            execution['feedback_gradient_identity_error'] = float((auxiliary_gradient-feedback.gradient).norm())
        total.backward()
        check_finite(model, total)
        gradient = _flat([v.grad for v in parameters], parameters).detach().clone()
        self.optimizer.step()
        if any(not bool(torch.isfinite(v).all()) for v in model.state_dict().values()):
            raise FloatingPointError('Nonfinite GNN state after the joint update')
        increment = _flat([v.detach() for v in parameters], parameters)-before
        execution['actual_step_norm'] = float(increment.norm())
        if cfg.adapt.optimizer == 'sgd':
            execution.update(momentum=0., sgd_step_identity_error=float((increment+cfg.adapt.lr*gradient).norm()))
        self.steps += 1
        return dict(step=self.steps, source_ce=float(source_loss.detach()),
            feedback_loss=float(auxiliary.detach()), total_loss=float(total.detach()),
            gradient_norm=float(gradient.norm()), query=query_info, execution=execution,
            batches=dict(fit=list(fit_ids), query=list(query_ids), target=list(target_ids)),
            seconds=time.perf_counter()-started)


def adapt_model(model, task, config, history_path):
    import json
    adapter = Adapter(model, config)
    with open(history_path, 'x') as stream:
        for _ in range(config.adapt.steps):
            row = adapter.step(task)
            stream.write(json.dumps(row, allow_nan=False)+'\n')
            stream.flush()
    return model.eval()
