"""Frozen-round source CE and true-forward movement objective (§3.10)."""
import torch
from torch.nn import functional as F
from .classifier import ParameterMap
from .utils import finite


class TsbaProblem:
    def __init__(self, model, source, labels, boundaries, cfg):
        self.params = ParameterMap(model)
        self.base = self.params.vector()
        self.source, self.labels = source.detach(), labels.detach()
        self.boundaries, self.cfg = boundaries, cfg

    def source_logits(self, vector):
        return self.params.logits(vector, self.source)

    def normalized_margin(self, vector):
        b = self.boundaries
        logits = self.params.logits(vector, b.points)
        pair_logits = logits.gather(1, b.pairs)
        return (pair_logits[:, 0] - pair_logits[:, 1]) / b.normal_norm

    def movement(self, vector):
        b = self.boundaries
        return self.normalized_margin(vector) - b.old_margin / b.normal_norm

    def source_loss(self, vector):
        return F.cross_entropy(self.source_logits(vector), self.labels)

    def objective(self, vector):
        movement = self.movement(vector)
        return self.source_loss(vector) + self.cfg.lambda_boundary * (movement - self.boundaries.zeta).square().mean() / 2

    @torch.no_grad()
    def evaluate(self, vector):
        logits = self.source_logits(vector)
        ce = F.cross_entropy(logits, self.labels, reduction="none")
        movement = self.movement(vector)
        move_loss = (movement - self.boundaries.zeta).square().mean() / 2
        objective = ce.mean() + self.cfg.lambda_boundary * move_loss
        finite(logits, ce, movement, objective)
        risks = {str(c): float(ce[self.labels == c].mean()) for c in self.labels.unique().tolist()}
        return dict(objective=float(objective), source_ce=float(ce.mean()), source_class_ce=risks,
                    move_loss=float(move_loss), max_movement=float(movement.abs().max()),
                    movement_rmse=float((movement - self.boundaries.zeta).square().mean().sqrt()))
