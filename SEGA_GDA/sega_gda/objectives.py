"""S3 building blocks only: no optimizer, evolution loop, or model selection."""
import math
import torch
from .transport import solve_semi_relaxed, fixed_plan_objective


def alignment_loss(source_coordinates, target_coordinates, distance, a, b, epsilon, rho, tolerances=None):
    """lambda-term of §3.8: F(Pi*; C) with Pi* solved on the STOPPED cost.

    The pairwise path cost is evaluated exactly once, with gradients (encoder
    positions in content steps, metric primitives in space steps). The
    correspondence solver reads a detached view of that same tensor, so the
    plan and the differentiated cost are the same numbers by construction:
    there is no second path solve and no floating-point equality check.
    The solver itself never differentiates the OT iterations (§3.7.3).
    """
    cost = distance.pairwise(source_coordinates, target_coordinates)
    result = solve_semi_relaxed(cost.detach(), a, b, epsilon, rho, tolerances)
    return fixed_plan_objective(result, cost, a, b), result, cost


def learning_objective(classification, alignment, relation, alignment_weight, relation_weight):
    """§3.8 only: CE + lambda full F + gamma relation. No additional objective."""
    if (not math.isfinite(alignment_weight) or alignment_weight < 0
            or not math.isfinite(relation_weight) or relation_weight <= 0):
        raise ValueError('Require lambda >= 0 and gamma > 0')
    total = classification+alignment_weight*alignment+relation_weight*relation
    if not torch.isfinite(total):
        raise FloatingPointError('Nonfinite learning objective')
    return total
