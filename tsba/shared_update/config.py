"""Fixed parameter ranges and numerical budgets for shared logit adaptation."""

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SharedUpdateConfig:
    seed: int = 0
    coordinate_floor: float = 1e-6
    variance_floor: float = 1e-3
    mean_radius: float = 2.0
    scale_min: float = 0.25
    scale_max: float = 4.0
    shift_bound: float = 4.0
    prior_floor: float = 1e-4
    tau: float = 0.01
    regularization: float = 0.1
    inner_starts: int = 5
    inner_maxiter: int = 150
    master_maxiter: int = 200
    search_rounds: int = 10
    feasibility_tol: float = 1e-8
    optimality_tol: float = 1e-6
    separation_tol: float = 1e-6

    def __post_init__(self):
        for name in ("seed", "inner_starts", "inner_maxiter", "master_maxiter", "search_rounds"):
            value = getattr(self, name)
            minimum = 0 if name == "seed" else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in ("coordinate_floor", "variance_floor", "mean_radius", "scale_min", "scale_max",
                     "shift_bound", "prior_floor", "tau", "regularization", "feasibility_tol",
                     "optimality_tol", "separation_tol"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (name != "tau" and value == 0):
                raise ValueError(f"{name} must be finite and {'nonnegative' if name == 'tau' else 'positive'}")
        if not self.scale_min <= 1 <= self.scale_max or self.scale_min >= self.scale_max:
            raise ValueError("scale range must contain the identity and have positive width")
        if self.prior_floor >= 0.5:
            raise ValueError("prior_floor must be less than 1/2")
        if self.feasibility_tol >= 1 or self.optimality_tol >= 1:
            raise ValueError("feasibility_tol and optimality_tol must be less than 1")
        if self.separation_tol < self.feasibility_tol:
            raise ValueError("separation_tol must be at least feasibility_tol")

    def to_dict(self):
        return asdict(self)
