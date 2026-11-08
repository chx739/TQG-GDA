"""Damped GGN proposals: direct or matrix-free JVP/VJP CG (§3.11)."""
import torch
from torch.func import grad, jacrev, jvp, vjp
from .utils import finite


# §3.11：Aδ=b，A=F_S+(λB/M)DᵀD+μI，b=−gS+(λB/M)Dᵀζ。
# F_S 是源 CE 的广义 Gauss–Newton 近似，不是非线性模型的完整 Hessian。
# P 为可训练分类器参数数，M 为本轮有效边界数。
class GGNSystem:
    def __init__(self, problem):
        self.problem = problem
        self.base = problem.base
        # 本轮固定 ψ_bar：p_i=softmax(f_bar(x_i^S))，gS=∇_ψ L_sup(ψ_bar)。
        self.p = problem.source_logits(self.base).softmax(-1).detach()
        self.gs = grad(problem.source_loss)(self.base).detach()
        self.coefficient = problem.cfg.lambda_boundary / len(problem.boundaries)
        # VJP 是 Jacobian 转置乘向量：source_pullback 对应 J_Sᵀ，
        # margin_pullback 对应 Dᵀ；normalized_margin 已包含 1/κ。
        _, self.source_pullback = vjp(problem.source_logits, self.base)
        _, self.margin_pullback = vjp(problem.normalized_margin, self.base)
        # ζ=−ν，因此边界项右端为正号 +(λB/M)Dᵀζ；rhs=−∇J_t(ψ_bar)。
        self.rhs = (-self.gs + self.coefficient * self.margin_pullback(problem.boundaries.zeta)[0]).detach()
        finite(self.p, self.gs, self.rhs)

    # §3.11：H_i u=[Diag(p_i)−p_i p_iᵀ]u=p_i⊙(u−Σ_c p_ic u_c)。
    def categorical_product(self, value):
        return self.p * (value - (self.p * value).sum(-1, keepdim=True))

    # 矩阵无关乘法：JVP 得 J_S q 与 Dq，随后 VJP 得 J_SᵀH J_S q 与 DᵀDq。
    # 实现 Aq，不显式存整个 Jacobian；CG 与 dense 求解同一个线性系统。
    def matvec(self, direction):
        source_jv = jvp(self.problem.source_logits, (self.base,), (direction,))[1]
        margin_jv = jvp(self.problem.normalized_margin, (self.base,), (direction,))[1]
        # F_S q=(1/BS)Σ_i J_iᵀH_iJ_i q；源批次平均与边界点平均不要混用。
        fs = self.source_pullback(self.categorical_product(source_jv) / len(self.p))[0]
        db = self.margin_pullback(margin_jv)[0]
        result = (fs + self.coefficient * db + self.problem.cfg.damping * direction).detach()
        finite(result)
        return result

    # 显式路径：source_j:[BS,C,P]，d=D:[M,P]，最终 A:[P,P]。
    def dense(self):
        source_j = jacrev(self.problem.source_logits)(self.base).detach()
        d = jacrev(self.problem.normalized_margin)(self.base).detach()
        hj = self.p[..., None] * (source_j - (self.p[..., None] * source_j).sum(1, keepdim=True))
        fs = torch.einsum("bcp,bcq->pq", source_j, hj) / len(self.p)
        matrix = fs + self.coefficient * d.T @ d
        # 对称化消除浮点不对称；μ>0 阻尼使理论半正定系统变为正定。
        matrix = (matrix + matrix.T) / 2 + self.problem.cfg.damping * torch.eye(
            self.base.numel(), dtype=self.base.dtype, device=self.base.device)
        finite(matrix)
        return matrix


# §3.11：仅生成候选 δ；直接解或 CG 后交给 adapt.backtrack 验收。
# 不在此提交参数，不通过求解器取元梯度，也不再交给 Adam。
def solve(problem):
    cfg = problem.cfg
    system = GGNSystem(problem)
    rhs = system.rhs
    rhs_norm = float(rhs.norm())
    method = cfg.solver
    if method == "auto":
        method = "direct" if rhs.numel() <= cfg.direct_max_params else "cg"
    if rhs_norm == 0:
        return torch.zeros_like(rhs), dict(valid=False, reason="stationary", solver=method)
    if method == "direct":
        try:
            delta = torch.linalg.solve(system.dense(), rhs)
        except torch.linalg.LinAlgError as exc:
            return None, dict(valid=False, reason=str(exc), solver=method)
        iterations = 1
    else:
        delta = torch.zeros_like(rhs)
        residual, direction = rhs.clone(), rhs.clone()
        rr = residual @ residual
        iterations = 0
        for iterations in range(1, cfg.cg_steps + 1):
            product = system.matvec(direction)
            denom = direction @ product
            finite(denom, rr)
            if float(denom) <= 0:
                return None, dict(valid=False, reason="cg_nonpositive_curvature", solver=method)
            step = rr / denom
            delta = delta + step * direction
            residual = residual - step * product
            new_rr = residual @ residual
            finite(delta, new_rr)
            if float(new_rr.sqrt()) <= cfg.cg_tolerance * rhs_norm:
                break
            direction = residual + (new_rr / rr) * direction
            rr = new_rr
    finite(delta)
    # 数值验收：真实相对残差 ||Aδ−b||/||b||；
    # 因为 ∇J_t=−b，还要求 directional=∇J_tᵀδ<0。这不是目标风险下降证明。
    relative = float((system.matvec(delta) - rhs).norm()) / rhs_norm
    directional = float((-rhs) @ delta)
    valid = relative <= cfg.cg_tolerance and directional < 0
    return delta.detach(), dict(valid=valid, reason="solved" if valid else "inaccurate_or_non_descent",
                               solver=method, iterations=iterations, relative_residual=relative,
                               directional_derivative=directional)
