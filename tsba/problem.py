"""Frozen-round source CE and true-forward movement objective (§3.10)."""
import torch
from torch.nn import functional as F
from .classifier import ParameterMap
from .utils import finite


# §3.6、§3.10：一轮的固定目标；base=ψ_bar，boundaries 为本轮旧参考。
# source:[BS,d]、labels:[BS]；候选 vector:[P] 仅改变分类器输出。
class TsbaProblem:
    def __init__(self, model, source, labels, boundaries, cfg):
        self.params = ParameterMap(model)
        self.base = self.params.vector()
        self.source, self.labels = source.detach(), labels.detach()
        self.boundaries, self.cfg = boundaries, cfg

    def source_logits(self, vector):
        return self.params.logits(vector, self.source)

    # §3.9：返回 m_ψ,j(ξ_j)/κ_j。对 vector 求 Jacobian 就得到 D_j。
    # κ_j 和 ξ_j 来自旧快照，不随候选分类器重新求导或归一化。
    def normalized_margin(self, vector):
        b = self.boundaries
        logits = self.params.logits(vector, b.points)
        pair_logits = logits.gather(1, b.pairs)
        return (pair_logits[:, 0] - pair_logits[:, 1]) / b.normal_norm

    # §3.10.3：v_j(ψ)=[m_ψ,j(ξ_j)−m_bar,j(ξ_j)]/κ_j。
    # 减去真实旧 margin，而不是假设投影得到精确的零；v 是位移的一阶代理。
    def movement(self, vector):
        b = self.boundaries
        return self.normalized_margin(vector) - b.old_margin / b.normal_norm

    # §3.10.1：L_sup=−(1/BS)Σ_i log softmax(f_ψ(x_i^S))[y_i^S]。
    def source_loss(self, vector):
        return F.cross_entropy(self.source_logits(vector), self.labels)

    # §3.10.3–3.10.4：L_move=(1/(2M))Σ_j(v_j−ζ_j)²；
    # J_t=L_sup+lambda_B·L_move。ζ 已固定，候选 margin 保留参数响应。
    def objective(self, vector):
        movement = self.movement(vector)
        return self.source_loss(vector) + self.cfg.lambda_boundary * (movement - self.boundaries.zeta).square().mean() / 2

    @torch.no_grad()
    # §3.12：真实前向复算 J_t 和 max_j|v_j|，而不是只看线性系统的预测。
    # 这里无需反向；逐类源 CE 只记录，不额外加入 SEGA 的硬风险预算。
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
