"""Opt-in R7 separation with compatible cuts recovered from failed endpoints.

The primary SLSQP launches use the stock starts, iteration budget, inequality,
and objective. A rejected finite endpoint can supply a cut by bisection from
its compatible launch point. If every primary launch fails to converge to a
compatible point, one feasible trust-region solve starts at the frozen
reference. The likelihood threshold and parameter bounds are shared across
both solvers. These remain local searches.
"""

import numpy as np
from scipy.optimize import Bounds, NonlinearConstraint, minimize

from tsba.shared_update.explanations import ExplanationSet


class RestoredExplanationSet(ExplanationSet):
    """Keep the stock reference fit and opt into R7 numerical recovery."""

    restoration_bisections = 42
    fallback_maxiter = 300
    fallback_gtol = 1e-8
    fallback_xtol = 1e-10
    fallback_barrier_tol = 1e-10

    def _restore_endpoint(self, start, endpoint):
        # Keep a strictly compatible bracket even on a nonlinear path.
        anchor = start if start.observation_loss <= self.threshold else self._reference
        if anchor.observation_loss > self.threshold:
            return None
        direction = endpoint.parameters - anchor.parameters
        feasible, infeasible = 0., 1.
        for _ in range(self.restoration_bisections):
            middle = (feasible + infeasible) / 2
            trial = anchor.parameters + middle * direction
            if self._observation(trial)[0] <= self.threshold:
                feasible = middle
            else:
                infeasible = middle
        candidate = self._candidate(anchor.parameters + feasible * direction, True)
        if (candidate is None or candidate.observation_loss > self.threshold
                or np.array_equal(candidate.parameters, anchor.parameters)):
            return None
        return candidate

    def _fallback(self, objective):
        reference = self._candidate(self._reference.parameters)
        unavailable = dict(launched=False, success=False, verified=False, feasible_candidate=False)
        if reference is None or reference.observation_loss > self.threshold:
            return None, unavailable | {"reason": "infeasible_frozen_reference"}
        if reference.observation_loss == self.threshold:
            reason = "zero_width_reference_on_boundary" if self.config.tau == 0 else "reference_on_boundary"
            return None, unavailable | {"reason": reason}
        constraints = NonlinearConstraint(
            lambda parameters: self._observation(parameters)[0], -np.inf, self.threshold,
            jac=lambda parameters: self._observation(parameters)[1], keep_feasible=True,
        )
        try:
            result = minimize(
                objective, reference.parameters.copy(), jac=True, method="trust-constr",
                bounds=Bounds(self._lower, self._upper, keep_feasible=True), constraints=constraints,
                options={"maxiter": self.fallback_maxiter, "gtol": self.fallback_gtol,
                         "xtol": self.fallback_xtol, "barrier_tol": self.fallback_barrier_tol},
            )
            candidate = self._candidate(result.x, True)
            optimality = float(result.optimality)
            verified = bool(result.success and candidate is not None and optimality <= self.fallback_gtol)
            return candidate, dict(
                launched=True, success=bool(result.success), status=int(result.status), message=str(result.message),
                reason="converged_verified" if verified else "solver_unverified",
                iterations=int(result.nit), optimality=optimality, verified=verified,
                feasible_candidate=candidate is not None,
                observation_loss=None if candidate is None else candidate.observation_loss,
            )
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
            return None, unavailable | {"launched": True, "reason": "solver_exception", "message": str(error)}

    def search(self, correction_logits, starts=()):
        """Find a compatible cut and record how the local search terminated."""
        if self._reference is None:
            raise RuntimeError("Call fit() before searching the fixed explanation set")
        correction = np.asarray(correction_logits, dtype=np.float64)
        if correction.shape != (len(self.target), self.classes) or not np.isfinite(correction).all():
            raise ValueError("correction_logits must be a finite [n_target, classes] matrix")
        centered = correction - correction.mean(axis=1, keepdims=True)
        magnitude = float(np.max(np.abs(centered)))
        objective_correction = centered / magnitude if magnitude else centered

        def objective(parameters):
            return self._separation(parameters, objective_correction)

        supplied = list(starts)
        known = self._unique([self._candidate(item.parameters, True)
                              for item in [self._reference, *self._fit_candidates, *supplied]])
        if not known:
            self.last_search = {"status": "no_feasible_start", "global_optimality_certified": False}
            return None
        known.sort(key=lambda candidate: objective(candidate.parameters)[0])
        previous_best = objective(known[0].parameters)[0]
        choices = [(candidate, "known") for candidate in known]
        optimizer_starts = known[:int(self.config.inner_starts)]
        for _ in range(4 * int(self.config.inner_starts)):
            if len(optimizer_starts) >= self.config.inner_starts:
                break
            proposed = self._perturb(self._reference.parameters)
            for contraction in range(20):
                reference = self._reference.parameters
                parameters = reference + 0.5 ** contraction * (proposed - reference)
                candidate = self._candidate(parameters, True)
                if candidate is not None:
                    optimizer_starts = self._unique([*optimizer_starts, candidate])
                    break

        reports, fallback = [], None
        if magnitude:
            bounds = list(zip(self._lower, self._upper))
            constraint = {"type": "ineq",
                          "fun": lambda p: self.threshold - self._observation(p)[0],
                          "jac": lambda p: -self._observation(p)[1]}
            for start in optimizer_starts:
                choices.append((start, "primary_start"))
                try:
                    result = minimize(
                        objective, start.parameters.copy(), jac=True, method="SLSQP",
                        bounds=bounds, constraints=[constraint],
                        options={"maxiter": int(self.config.inner_maxiter), "ftol": 1e-10},
                    )
                    endpoint = self._candidate(result.x)
                    compatible = (endpoint is not None
                                  and endpoint.observation_loss <= self.threshold + self.config.feasibility_tol)
                    restored = None
                    if compatible:
                        choices.append((endpoint, "primary_slsqp"))
                    elif endpoint is not None:
                        restored = self._restore_endpoint(start, endpoint)
                        if restored is not None:
                            choices.append((restored, "restored_endpoint"))
                    reports.append(dict(
                        success=bool(result.success), status=int(result.status), message=str(result.message),
                        iterations=int(result.nit), feasible_candidate=compatible,
                        endpoint_observation_loss=None if endpoint is None else endpoint.observation_loss,
                        restored_observation_loss=None if restored is None else restored.observation_loss,
                        restored_feasible_candidate=restored is not None,
                    ))
                except (FloatingPointError, OverflowError) as error:
                    reports.append(dict(success=False, message=str(error), feasible_candidate=False,
                                        restored_feasible_candidate=False))

        primary_successes = sum(row["success"] and row["feasible_candidate"] for row in reports)
        if magnitude and not primary_successes:
            candidate, fallback = self._fallback(objective)
            if candidate is not None:
                choices.append((candidate, "trust_constr"))
        qualified_fallback = fallback is not None and fallback["verified"]
        restored_count = sum(row["restored_feasible_candidate"] for row in reports)
        best, origin = min(choices, key=lambda choice: objective(choice[0].parameters)[0])
        best_objective = objective(best.parameters)[0]
        minimum_loss = min(candidate.observation_loss for candidate, _ in choices)
        if not magnitude:
            outcome = "flat_objective"
        elif primary_successes:
            outcome = "primary_slsqp"
        elif qualified_fallback:
            outcome = "trust_constr"
        else:
            outcome = "restored_only" if restored_count else "unverified"
        completed = primary_successes or qualified_fallback
        self.last_search = dict(
            status=("flat_objective" if not magnitude else
                    "local_search_completed" if completed else "local_search_failed"),
            outcome=outcome, selected_candidate_origin=origin,
            known_feasible_candidates=len(known), supplied_starts=len(supplied),
            optimizer_runs=reports, successful_feasible_runs=primary_successes + int(qualified_fallback),
            primary_successful_feasible_runs=primary_successes, restored_endpoints=restored_count,
            fallback_run=fallback, objective=self._separation(best.parameters, correction)[0],
            improvement_over_known=magnitude * (previous_best - best_objective),
            new_candidate_found=best_objective < previous_best - 1e-10,
            observation_violation=max(0., best.observation_loss - self.threshold),
            min_observation_loss=minimum_loss,
            better_reference_found=minimum_loss < self.reference_loss - self.config.feasibility_tol,
            global_optimality_certified=False,
        )
        return best
