"""Predeclared training and numerical budgets (§3.15); no target-label selection."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
# §3.15：运行前固定的工程配置。下列数值是默认选择，不是文档证明的最优值。
# 公式符号沿用 docs/tsba/TSBA_Graph_Level_GDA_Design.md。
class Config:
    seed: int = 0
    encoder: str = "gin"
    hidden: int = 64
    num_layers: int = 3
    emb_dim: int = 16
    dropout: float = 0.0
    source_epochs: int = 100
    source_lr: float = 1e-3
    source_batch: int = 128
    adapter_width: int = 0  # 0 means min(16, d)
    coordinate_eps: float = 1e-8
    score_hidden: int = 64
    score_epochs: int = 100
    score_batch: int = 128
    score_lr: float = 1e-3
    score_validation: bool = True
    kappa_sigma: float = 1.0
    adapt_rounds: int = 30
    source_adapt_batch: int = 128
    target_batch: int = 128
    # §3.10：lambda_boundary=λB，控制源 CE 与边界位移拟合的相对权重。
    lambda_boundary: float = 1.0
    # §3.8：τ 为无量纲移动系数；ν=−clip(τ·sigma²·γ)。
    tau: float = 0.1
    # §3.7、§3.12：三个不同预算分别限制投影距离、请求位移、执行时的归一化变化。
    radius_multiplier: float = 1.0  # R_proj = multiplier * sigma * sqrt(d)
    request_multiplier: float = 0.1  # Delta_max = multiplier * sigma
    execution_multiplier: float = 0.1  # Delta_exec = multiplier * sigma
    root_tolerance: float = 1e-5  # epsilon_B = tolerance * sigma
    min_normal: float = 1e-8
    projection_eps: float = 1e-12
    projection_steps: int = 20
    projection_backtracks: int = 10
    movement_zero: float = 1e-12  # zero-request tolerance, in units of sigma
    # §3.11：damping=μ_t，本版本固定常数；它是数值阻尼，不是目标风险置信度。
    damping: float = 1e-2  # constant mu_t, fixed before adaptation
    solver: str = "auto"
    direct_max_params: int = 2048
    # §3.11：None 在构造时按精度确定默认值；显式容差不会被静默放宽。
    # float32 的 Jacobian/矩阵乘法舍入误差高于 float64；两条求解路径共用验收值。
    cg_tolerance: float | None = None
    cg_steps: int = 200
    # §3.12：armijo=cA∈(0,1)，backtracks 限制 α=2^(−k) 的尝试次数。
    armijo: float = 1e-4
    backtracks: int = 15
    adaptation_dtype: str = "float64"
    device: str = "cpu"

    def __post_init__(self):
        if self.adaptation_dtype not in ("float32", "float64"):
            raise ValueError("adaptation_dtype must be float32 or float64")
        if self.cg_tolerance is None:
            object.__setattr__(self, "cg_tolerance", 1e-5 if self.adaptation_dtype == "float32" else 1e-7)
        integers = ("seed", "hidden", "num_layers", "emb_dim", "source_epochs", "source_batch",
                    "adapter_width", "score_hidden", "score_epochs", "score_batch", "adapt_rounds",
                    "source_adapt_batch", "target_batch", "projection_steps", "projection_backtracks",
                    "direct_max_params", "cg_steps", "backtracks")
        for name in integers:
            v = getattr(self, name)
            minimum = 0 if name in ("seed", "adapter_width", "adapt_rounds") else 1
            if type(v) is not int or v < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name, v in asdict(self).items():
            if isinstance(v, float) and (not math.isfinite(v) or v < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        positive = ("source_lr", "coordinate_eps", "score_lr", "kappa_sigma", "lambda_boundary", "tau",
                    "radius_multiplier", "request_multiplier", "execution_multiplier", "root_tolerance",
                    "min_normal", "projection_eps", "damping", "cg_tolerance")
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.dropout < 1 or not 0 < self.armijo < 1:
            raise ValueError("invalid dropout or Armijo coefficient")
        if self.encoder not in ("gin", "gine") or self.solver not in ("auto", "direct", "cg"):
            raise ValueError("unknown encoder or solver")
        if type(self.score_validation) is not bool:
            raise ValueError("score_validation must be boolean")

    def to_dict(self):
        return asdict(self)
