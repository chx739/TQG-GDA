"""Stateless query/heldout gradient and exact cloned-Adam diagnostic trials."""
import copy
import math

import torch
from torch.func import functional_call
from torch.nn import functional as F

from tqg_gda.utils import cpu_state


def flatten(values):
    return torch.cat([value.detach().reshape(-1) for value in values])


def gradient(loss, parameters, retain_graph=False):
    values = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    return flatten([torch.zeros_like(p) if g is None else g.detach() for p, g in zip(parameters, values)])


def dense_centered_jacobian(model, batch):
    """Independent full-parameter Jacobian; exploit only exact class centering."""
    parameters = tuple(model.parameters())
    logits = model(batch)
    rows = []
    for i in range(len(logits)):
        independent = [gradient(logits[i, c], parameters, retain_graph=True)
                       for c in range(logits.shape[1]-1)]
        rows.append(torch.stack(independent + [-torch.stack(independent).sum(0)]))
    return logits.detach(), torch.stack(rows).detach()


class ExecutionDiagnostics:
    def __init__(self, model, source_fit, source_bank, query, held, labels_query,
                 labels_held, *, historical_optimizer=None):
        self.model = model
        self.names = tuple(n for n, _ in model.named_parameters())
        self.parameters = tuple(model.parameters())
        self.base = cpu_state(model)
        self.query, self.held, self.source = query, held, source_bank
        self.yq, self.yh = labels_query, labels_held
        self.zq, self.jq = dense_centered_jacobian(model, query)
        self.zs, self.js = dense_centered_jacobian(model, source_bank)
        self.z_h = model(held).detach()
        self.g_fit = gradient(F.cross_entropy(model(source_fit), source_fit.y), self.parameters)
        self.g_h = gradient(F.cross_entropy(model(held), labels_held), self.parameters)
        self.g_q = gradient(F.cross_entropy(model(query), labels_query), self.parameters)
        self.encoder_mask = torch.cat([torch.full((p.numel(),), n.startswith('encoder.'), dtype=torch.bool)
                                       for n, p in model.named_parameters()])
        self.scale = max(1., float(flatten(self.parameters).norm()))
        self.base_ce_h = float(F.cross_entropy(self.z_h, labels_held))
        self.base_ce_q = float(F.cross_entropy(self.zq, labels_query))
        self.trial = copy.deepcopy(model)
        self.optimizer = torch.optim.Adam(self.trial.parameters(), lr=.0003,
            betas=(.9, .999), eps=1e-8, weight_decay=0, amsgrad=False)
        if historical_optimizer is not None:
            self.optimizer.load_state_dict(copy.deepcopy(historical_optimizer))
        self.optimizer_base = copy.deepcopy(self.optimizer.state_dict())
        self.source_trial = self.adam_trial(self.g_fit)

    def adam_trial(self, gradient_vector):
        self.trial.load_state_dict(self.base)
        self.optimizer.load_state_dict(copy.deepcopy(self.optimizer_base))
        offset = 0
        for p in self.trial.parameters():
            p.grad = gradient_vector[offset:offset+p.numel()].reshape_as(p).clone()
            offset += p.numel()
        self.optimizer.step()
        with torch.no_grad():
            z_h, z_q = self.trial(self.held), self.trial(self.query)
        return dict(ce_h=float(F.cross_entropy(z_h, self.yh)),
                    ce_q=float(F.cross_entropy(z_q, self.yq)),
                    step_norm=float((flatten(self.trial.parameters())-flatten(self.parameters)).norm()))

    def score(self, item):
        pred_q, pred_h = item['query'], item['held']
        gate = pred_q.gate.double()
        if pred_q.source_ce_weights is not None:
            y_s = item['source_labels']
            coefficient = self.zs.softmax(-1)-F.one_hot(y_s, self.zs.shape[1]).to(self.zs)
            g = torch.einsum('ncp,nc,n->p', self.js, coefficient, pred_q.source_ce_weights)
        elif pred_q.soft_target is not None:
            coefficient = self.zq.softmax(-1)-pred_q.soft_target
            g = torch.einsum('ncp,nc,n->p', self.jq, coefficient, gate)/len(gate)
        else:
            g = -torch.einsum('ncp,nc,n->p', self.jq, pred_q.correction, gate)/len(gate)
        if not torch.isfinite(g).all():
            raise FloatingPointError('Nonfinite full-parameter feedback gradient')
        gn = float(g.norm())
        def cosine(other, mask=None):
            left, right = (g, other) if mask is None else (g[mask], other[mask])
            denominator = float(left.norm()*right.norm())
            return float(torch.clamp(left @ right / denominator, -1, 1)) if denominator > 1e-12 else None
        joint = self.adam_trial(self.g_fit+.1*g)
        result = dict(gradient_cosine=cosine(self.g_h), gradient_cosine_encoder=cosine(self.g_h, self.encoder_mask),
                      query_gradient_cosine=cosine(self.g_q), feedback_gradient_norm=gn,
                      feedback_active=gn > 1e-12,
                      feedback_source_gradient_ratio=.1*gn/max(1e-12, float(self.g_fit.norm())),
                      actual_Adam_CE_gain_vs_source_only=self.source_trial['ce_h']-joint['ce_h'],
                      actual_Adam_CE_gain_from_base=self.base_ce_h-joint['ce_h'],
                      actual_Adam_query_CE_gain=self.base_ce_q-joint['ce_q'],
                      actual_Adam_step_norm=joint['step_norm'])
        # For source-CE-only controls the nominal d is zero: direct-output
        # scores are explicitly inapplicable rather than a misleading zero win.
        if pred_q.source_ce_weights is None:
            result['direct_logit_CE_gain'] = self.base_ce_h-float(F.cross_entropy(
                self.z_h+.1*pred_h.correction*pred_h.gate[:, None], self.yh))
            result['direct_query_logit_CE_gain'] = self.base_ce_q-float(F.cross_entropy(
                self.zq+.1*pred_q.correction*pred_q.gate[:, None], self.yq))
        else:
            result['direct_logit_CE_gain'] = result['direct_query_logit_CE_gain'] = None
        length = 1e-4*self.scale
        candidate = {}
        offset = 0
        for n, p in self.model.named_parameters():
            displacement = g[offset:offset+p.numel()].reshape_as(p)*(length/gn) if gn > 1e-12 else torch.zeros_like(p)
            candidate[n] = p.detach()-displacement
            offset += p.numel()
        buffers = {n: b.detach().clone() for n, b in self.model.named_buffers()}
        with torch.no_grad():
            sh = functional_call(self.model, (candidate, buffers), (self.held,), strict=True)
            sq = functional_call(self.model, (candidate, buffers), (self.query,), strict=True)
        result['normalized_SGD_CE_gain'] = self.base_ce_h-float(F.cross_entropy(sh, self.yh))
        result['normalized_SGD_query_CE_gain'] = self.base_ce_q-float(F.cross_entropy(sq, self.yq))
        if any(not math.isfinite(v) for v in result.values() if isinstance(v, float)):
            raise FloatingPointError('Nonfinite execution metric')
        if any(not torch.equal(self.base[n], p) for n, p in self.model.state_dict().items()):
            raise RuntimeError('Diagnostic mutated original model')
        return result
