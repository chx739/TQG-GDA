"""Finite-radius projection and detached normal displacement requests (§3.7–3.9)."""
from collections import Counter
from dataclasses import dataclass
import math
import torch
from .utils import finite


@dataclass
# §3.6–3.9：本轮固定参考集合。M 为有效边界点数，d 为表示维数。
# points=ξ:[M,d]；pairs=(c,d):[M,2]；normal_norm=κ:[M]；
# old_margin=m_bar(ξ)、zeta=−ν 均为 [M]；normals=n:[M,d]。
class BoundaryBatch:
    points: torch.Tensor
    pairs: torch.Tensor
    normal_norm: torch.Tensor
    old_margin: torch.Tensor
    zeta: torch.Tensor
    normals: torch.Tensor
    rejected: dict
    # 可选审计映射：每个有效锚点来自传入 target 批次的哪个位置。
    input_indices: torch.Tensor | None = None

    def __len__(self):
        return len(self.points)


# §3.7、§3.13：m(x)=f_bar,c(x)−f_bar,d(x)，求的是输入梯度 ∇_x m。
# 返回的 normal 此时尚未单位化；这里不累计分类器参数的训练梯度。
def margin_and_normal(model, point, pair):
    with torch.enable_grad():
        point = point.detach().requires_grad_(True)
        out = model(point.unsqueeze(0))[0]
        margin = out[pair[0]] - out[pair[1]]
        normal, = torch.autograd.grad(margin, point)
    finite(margin, normal, out)
    return margin.detach(), normal.detach(), out.detach()


# §3.7：固定本轮类对，在 B(x_target,R_proj) 内寻找 m(ξ)≈0。
# 这里只找附近根，不保证全局最近边界；ξ 是辅助坐标，不是真实生成图。
def project(model, x, pair, radius, tolerance, cfg):
    point = x.detach().clone()
    for _ in range(cfg.projection_steps):
        margin, gradient, _ = margin_and_normal(model, point, pair)
        norm = gradient.norm()
        finite(norm)
        if float(norm) <= cfg.min_normal:
            return None, "small_normal"
        # 几何残差 |m(ξ)|/||∇_x m(ξ)||，以坐标距离近似度量根精度。
        residual = margin.abs() / norm
        finite(residual)
        if float(residual) <= tolerance:
            break
        # §3.7 投影步：Δξ=−m(ξ)·∇m(ξ)/(||∇m(ξ)||²+ε_proj)。
        # 范数有限不代表其平方有限；分母溢出会把方向伪装成零，必须停止。
        denominator = norm.square() + cfg.projection_eps
        finite(denominator)
        direction = -margin * gradient / denominator
        finite(direction)
        accepted = False
        for back in range(cfg.projection_backtracks):
            candidate = point + (0.5 ** back) * direction
            offset = candidate - x
            offset_norm = offset.norm()
            finite(offset_norm)
            # Π_B：超出半径时沿径向缩回球内；回溯要求几何残差下降。
            candidate = x + offset * torch.clamp(radius / offset_norm.clamp_min(1e-30), max=1)
            cm, cg, _ = margin_and_normal(model, candidate, pair)
            candidate_norm = cg.norm()
            finite(candidate_norm)
            if float(candidate_norm) > cfg.min_normal:
                candidate_residual = cm.abs() / candidate_norm
                finite(candidate_residual)
            else:
                continue
            if float(candidate_residual) < float(residual):
                point, accepted = candidate.detach(), True
                break
        if not accepted:
            return None, "projection_stalled"
    margin, gradient, out = margin_and_normal(model, point, pair)
    norm = gradient.norm()
    finite(norm)
    if float(norm) <= cfg.min_normal:
        return None, "small_normal"
    residual = margin.abs() / norm
    finite(residual)
    if float(residual) > tolerance:
        return None, "root_tolerance"
    if float((point - x).norm()) > radius * (1 + 1e-6):
        return None, "radius"
    # §3.7 有效类别面：c,d 必须仍为最高两类。
    # 第三类若显著更高，m_c,d=0 也不是当前实际决策边界。
    if out.numel() > 2:
        mask = torch.ones_like(out, dtype=torch.bool)
        mask[pair] = False
        if bool((out[mask] > out[pair].min() + tolerance * norm).any()):
            return None, "inactive_class_pair"
    # §3.8：κ=||∇m||，n=∇m/κ 是指向 margin 增大方向的单位法向。
    normal = gradient / norm
    finite(normal)
    return (point, margin, normal, norm), None


# §3.6–3.8：在当前分类器快照上构造请求，接受更新后下一轮全部重建。
def build_boundaries(model, target, score, sigma, cfg):
    radius = cfg.radius_multiplier * sigma * math.sqrt(target.shape[1])
    tolerance = cfg.root_tolerance * sigma
    rows, indices, rejected = [], [], Counter()
    for input_index, x in enumerate(target.detach()):
        with torch.no_grad():
            logits = model(x.unsqueeze(0))[0]
            finite(logits)
            # Fix binary orientation; multi-class ties resolve by class index.
            # §3.7：二分类固定 (0,1)，多分类取当前 top-two 后整轮保持不变。
            # 类对用于定位决策面，不当作目标真标签。
            pair = (torch.tensor([0, 1], device=x.device) if len(logits) == 2 else
                    torch.argsort(logits, descending=True, stable=True)[:2])
        projected, reason = project(model, x, pair, radius, tolerance, cfg)
        if projected is None:
            rejected[reason] += 1
            continue
        point, old_margin, normal, norm = projected
        with torch.no_grad():
            value = score(point.unsqueeze(0))[0]
            finite(value)
            # §3.8：γ=nᵀs_ω*(ξ)，只保留 score 的法向分量。
            gamma = normal @ value
            # zeta = -nu = clip(tau sigma^2 gamma), with frozen reference quantities.
            # §3.8–3.9：ν=−clip(τ·sigma²·γ,−Δmax,Δmax)，ζ=−ν。
            # 代码直接计算 ζ：γ>0 请求边界沿负法向移动，对应旧边界点 margin 增大。
            zeta = (cfg.tau * sigma ** 2 * gamma).clamp(-cfg.request_multiplier * sigma,
                                                      cfg.request_multiplier * sigma)
            finite(gamma, zeta)
        rows.append((point, pair, norm, old_margin, zeta, normal))
        indices.append(input_index)
    if rows:
        # §3.13：所有边界参考停止梯度；参数更新不穿过投影器或 score。
        columns = [torch.stack(column).detach() for column in zip(*rows)]
        return BoundaryBatch(*columns, dict(rejected), torch.tensor(indices, device=target.device))
    d = target.shape[1]
    return BoundaryBatch(target.new_empty((0, d)), torch.empty((0, 2), device=target.device, dtype=torch.long),
                         target.new_empty(0), target.new_empty(0), target.new_empty(0),
                         target.new_empty((0, d)), dict(rejected),
                         torch.empty(0, dtype=torch.long, device=target.device))
