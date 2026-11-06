"""Predeclared training and numerical budgets (§3.15); no target-label selection."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
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
    lambda_boundary: float = 1.0
    tau: float = 0.1
    radius_multiplier: float = 1.0  # R_proj = multiplier * sigma * sqrt(d)
    request_multiplier: float = 0.1  # Delta_max = multiplier * sigma
    execution_multiplier: float = 0.1  # Delta_exec = multiplier * sigma
    root_tolerance: float = 1e-5  # epsilon_B = tolerance * sigma
    min_normal: float = 1e-8
    projection_eps: float = 1e-12
    projection_steps: int = 20
    projection_backtracks: int = 10
    movement_zero: float = 1e-12  # zero-request tolerance, in units of sigma
    damping: float = 1e-2  # constant mu_t, fixed before adaptation
    solver: str = "auto"
    direct_max_params: int = 2048
    cg_tolerance: float = 1e-7
    cg_steps: int = 200
    armijo: float = 1e-4
    backtracks: int = 15
    adaptation_dtype: str = "float64"
    device: str = "cpu"

    def __post_init__(self):
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
        if self.adaptation_dtype not in ("float32", "float64"):
            raise ValueError("adaptation_dtype must be float32 or float64")
        if type(self.score_validation) is not bool:
            raise ValueError("score_validation must be boolean")

    def to_dict(self):
        return asdict(self)
