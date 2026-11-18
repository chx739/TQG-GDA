"""Optional R7 joint Gaussian explanations in a source-task subspace.

The Gaussian observation model sees two source-standardized projected
coordinates. Its posterior still supplies moments of *all* original update
features, so this does not restrict the shared logit correction. Projection
selection uses source-fit features, source-fit labels, and the archived raw
source head normal expressed in those features' coordinates. No target label
or holdout feature enters projection selection.

This changes the explanation family; it is not a claim that a two-dimensional
posterior covers the target task. Source inputs of dimension <= 2 pass through
unchanged, preserving the existing synthetic model and numerical behavior.
"""

import hashlib
import json

import numpy as np

from .restored_separation import RestoredExplanationSet


def _readonly(values):
    values = np.array(values, dtype=np.float64, copy=True)
    values.setflags(write=False)
    return values


def _fingerprint(values):
    array = np.ascontiguousarray(values)
    state = hashlib.sha256()
    state.update(str((array.shape, array.dtype.str)).encode("ascii"))
    state.update(array.tobytes())
    return state.hexdigest()


def _orthogonal(vector, basis):
    vector = np.array(vector, dtype=np.float64, copy=True)
    # Reorthogonalize to avoid accepting a roundoff-only Fisher residual.
    for _ in range(2):
        for axis in basis:
            vector -= axis * (axis @ vector)
    return vector


def _unit(vector):
    """Normalize without squaring the original vector's possibly tiny scale."""
    maximum = float(np.max(np.abs(vector)))
    if not maximum:
        return None
    scaled = vector / maximum
    return scaled / np.linalg.norm(scaled)


def _canonical_sign(vector, tolerance):
    absolute = np.abs(vector)
    pivot = int(np.flatnonzero(absolute >= absolute.max() * (1 - tolerance))[0])
    return vector if vector[pivot] >= 0 else -vector


def _canonical_projector_axis(projector, basis, tolerance):
    """Fix rotations within a tied eigenspace using the first best coordinate."""
    diagonal = np.maximum(np.diag(projector), 0.)
    maximum = float(diagonal.max())
    candidates = np.flatnonzero(diagonal >= maximum * (1 - tolerance))
    for index in candidates:
        axis = _unit(_orthogonal(projector[:, index], basis))
        if axis is not None:
            return _canonical_sign(axis, tolerance), int(index)
    raise ValueError("Could not construct a deterministic complementary axis")


def _residual_pca(covariance, basis, scale_floor, tolerance):
    dimension = len(covariance)
    projector = np.eye(dimension)
    for axis in basis:
        projector -= np.outer(axis, axis)
    residual = projector @ covariance @ projector
    residual = (residual + residual.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(residual)
    reference_scale = max(float(np.trace(covariance) / dimension), scale_floor ** 2)
    largest = max(0., float(eigenvalues[-1]))
    if largest <= tolerance * reference_scale:
        axis, pivot = _canonical_projector_axis(projector, basis, tolerance)
        return axis, dict(kind="canonical_complement", pivot=pivot,
                          residual_largest_eigenvalue=largest, top_eigenspace_size=0)
    tied = largest - eigenvalues <= tolerance * max(largest, scale_floor ** 2)
    eigenspace = eigenvectors[:, tied]
    axis, pivot = _canonical_projector_axis(eigenspace @ eigenspace.T, basis, tolerance)
    return axis, dict(kind="residual_pca", pivot=pivot,
                      residual_largest_eigenvalue=largest, top_eigenspace_size=int(tied.sum()))


def _source_projection(source, labels, normal, *, coordinate_floor,
                       fisher_regularization, rank_tolerance):
    dimension = source.shape[1]
    report = dict(
        version="source_task_head_fisher_2d_v1",
        full_dimension=int(dimension), projected_dimension=min(2, int(dimension)),
        source_count=len(source), source_class_counts=np.bincount(labels).tolist(),
        coordinate_floor=coordinate_floor, fisher_regularization=fisher_regularization,
        rank_tolerance=rank_tolerance, source_head_normal=normal.tolist(),
        source_inputs_sha256=dict(features=_fingerprint(source), labels=_fingerprint(labels),
                                  head_normal=_fingerprint(normal)),
        input_coordinates="full source-standardized features without intercept",
        correction_coordinates="full original features plus intercept",
        target_used_to_select_projection=False, source_label_role="source_fit_only",
    )
    if dimension <= 2:
        basis, mean, scale = np.eye(dimension), np.zeros(dimension), np.ones(dimension)
        report.update(mode="identity", axis_origins=["identity"] * dimension,
                      axis_diagnostics=[], fisher_ridge=None, raw_scale=scale.tolist(),
                      scale_floored=[False] * dimension,
                      standardization="none; exact identity for dimension <= 2")
    else:
        if not np.array_equal(np.unique(labels), [0, 1]):
            raise ValueError("Source-task projection above dimension two requires binary source labels")
        centered = source - source.mean(axis=0)
        covariance = centered.T @ centered / len(source)
        basis, origins, diagnostics = [], [], []
        first = _unit(normal)
        if first is None:
            first, diagnostic = _residual_pca(covariance, [], coordinate_floor, rank_tolerance)
            origins.append("zero_head_" + diagnostic["kind"])
            diagnostics.append(diagnostic)
        else:
            origins.append("source_head_normal")
            diagnostics.append(dict(kind="source_head_normal", orientation="class_1_minus_class_0"))
        basis.append(first)

        class_means = np.stack([source[labels == k].mean(axis=0) for k in (0, 1)])
        within = source - class_means[labels]
        scatter = within.T @ within / len(source)
        mean_trace = float(np.trace(scatter) / dimension)
        ridge = fisher_regularization * max(mean_trace, coordinate_floor ** 2)
        fisher = np.linalg.solve(scatter + ridge * np.eye(dimension), class_means[1] - class_means[0])
        # Normalize before the projection so its rank threshold is relative
        # to the full Fisher direction, not to arbitrary coefficient units.
        fisher_unit = _unit(fisher)
        residual = np.zeros(dimension) if fisher_unit is None else _orthogonal(fisher_unit, basis)
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm <= rank_tolerance:
            second, diagnostic = _residual_pca(covariance, basis, coordinate_floor, rank_tolerance)
            origins.append("degenerate_fisher_" + diagnostic["kind"])
            diagnostics.append(diagnostic | {"fisher_relative_residual_norm": residual_norm})
        else:
            second = residual / residual_norm
            origins.append("orthogonal_source_fisher")
            diagnostics.append(dict(kind="orthogonal_source_fisher",
                                    fisher_relative_residual_norm=residual_norm,
                                    orientation="projected_class_1_minus_class_0_fisher"))
        basis = np.column_stack([first, second])
        if not np.allclose(basis.T @ basis, np.eye(2), rtol=0, atol=1e-12):
            raise ValueError("Source-task axes are not orthonormal")
        projected = source @ basis
        mean = projected.mean(axis=0)
        raw_scale = np.sqrt(np.mean((projected - mean) ** 2, axis=0))
        scale = np.maximum(raw_scale, coordinate_floor)
        report.update(mode="source_head_fisher", axis_origins=origins, axis_diagnostics=diagnostics,
                      fisher_ridge=float(ridge), fisher_within_mean_trace=mean_trace,
                      fisher_scatter_denominator="n_source", raw_scale=raw_scale.tolist(),
                      scale_floored=(raw_scale < coordinate_floor).tolist(),
                      standardization="source projection population mean and RMS; coordinate floor",
                      fallback_rule="residual PCA; tied top eigenspace first maximum-leverage coordinate; "
                                    "zero variance uses canonical orthogonal complement",
                      pca_sign_rule="first maximum-absolute coordinate is positive")
    report.update(basis=basis.tolist(), mean=mean.tolist(), scale=scale.tolist())
    # A report is saved verbatim by the runner and must contain no NaN/Infinity.
    json.dumps(report, allow_nan=False)
    return _readonly(basis), _readonly(mean), _readonly(scale), report


class ProjectedExplanationSet(RestoredExplanationSet):
    """Fit projected likelihoods and retain full-dimensional update moments.

    ``source_head_normal`` is the binary class-1 minus class-0 raw head normal
    in the supplied full standardized coordinates, e.g. ``v0[1, :-1] -
    v0[0, :-1]``. It is not a residual-adapted TSBA head. Arrays contain no
    intercept. Recreate this class from the same source arrays and normal for
    A/H posterior replay; ``target`` then holds projected query features and
    ``features`` holds the original query features with an intercept.

    ``project(full_x)`` applies the saved source-only map to another feature
    matrix, including the features consumed by a projected CDF diagnostic.
    """

    def __init__(self, source_x, source_y, target_x, config, *, source_head_normal,
                 fisher_regularization=0.1, rank_tolerance=1e-10):
        source = np.array(source_x, dtype=np.float64, copy=True)
        target = np.array(target_x, dtype=np.float64, copy=True)
        labels = np.asarray(source_y)
        normal = np.array(source_head_normal, dtype=np.float64, copy=True)
        if (source.ndim != 2 or target.ndim != 2 or not len(source) or not len(target)
                or not source.shape[1] or source.shape[1] != target.shape[1]
                or normal.shape != (source.shape[1],)):
            raise ValueError("Expected nonempty full source/target matrices and one full source head normal")
        if not all(np.isfinite(values).all() for values in (source, target, normal)):
            raise ValueError("Source-task projection inputs must be finite")
        if (labels.ndim != 1 or len(labels) != len(source)
                or not np.issubdtype(labels.dtype, np.integer)):
            raise ValueError("Source labels must be one integer per source-fit row")
        unique = np.unique(labels)
        if len(unique) < 2 or not np.array_equal(unique, np.arange(len(unique))):
            raise ValueError("Source labels must contain all classes in contiguous range 0..C-1")
        labels = np.array(labels, dtype=np.int64, copy=True)
        coordinate_floor = float(getattr(config, "coordinate_floor", 1e-6))
        fisher_regularization, rank_tolerance = float(fisher_regularization), float(rank_tolerance)
        if (not np.isfinite([coordinate_floor, fisher_regularization, rank_tolerance]).all()
                or coordinate_floor <= 0 or fisher_regularization <= 0 or not 0 < rank_tolerance < 1):
            raise ValueError("Projection floors, Fisher regularization, and rank tolerance must be positive")
        (self.projection_basis, self.projection_mean, self.projection_scale,
         self.projection_report) = _source_projection(
            source, labels, normal, coordinate_floor=coordinate_floor,
            fisher_regularization=fisher_regularization, rank_tolerance=rank_tolerance)
        self.full_source, self.full_target = _readonly(source), _readonly(target)
        self.full_dimension = source.shape[1]
        super().__init__(self.project(source), labels, self.project(target), config)
        # The parent _candidate contracts posterior with this matrix. Its
        # likelihood and analytic separation derivatives use source/target.
        self.features = _readonly(np.column_stack((target, np.ones(len(target)))))

    def project(self, full_x):
        values = np.asarray(full_x, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.projection_basis.shape[0] or not np.isfinite(values).all():
            raise ValueError("Projection requires a finite matrix in full source-standardized coordinates")
        if self.projection_report["mode"] == "identity":
            return values.copy()
        projected = (values @ self.projection_basis - self.projection_mean) / self.projection_scale
        if not np.isfinite(projected).all():
            raise ValueError("Non-finite projected features")
        return projected


def check_projected_explanations():
    """Small deterministic engineering fixture; uses only generated toy data."""
    from tsba.shared_update.config import SharedUpdateConfig
    from tsba.shared_update.solver import relative_risks, solve_master

    rng = np.random.default_rng(37)
    labels = np.repeat([0, 1], 20)
    source = rng.normal(size=(40, 5)) + (2 * labels[:, None] - 1) * np.array([.7, .5, .2, 0., 0.])
    target = rng.normal(size=(17, 5))
    normal = np.array([1., .2, -.1, 0., .05])
    cfg = SharedUpdateConfig(seed=9, inner_starts=1, inner_maxiter=100, tau=.02)
    family = ProjectedExplanationSet(source, labels, target, cfg, source_head_normal=normal)
    other = ProjectedExplanationSet(source, labels, rng.normal(size=(7, 5)), cfg, source_head_normal=normal)
    assert family.projection_report == other.projection_report
    np.testing.assert_allclose(family.source.mean(0), 0., rtol=0, atol=1e-14)
    np.testing.assert_allclose(family.source.var(0), 1., rtol=0, atol=1e-14)
    np.testing.assert_allclose(family.projection_basis[:, 0], normal / np.linalg.norm(normal), rtol=0, atol=1e-15)
    reference = family.fit()
    posterior = family._target_terms(reference.parameters)[-1]
    assert reference.statistic.shape == (2, 6) and family.dimension == 2
    np.testing.assert_allclose(reference.statistic, posterior.T @ family.features / len(target), rtol=0, atol=0)
    np.testing.assert_allclose(reference.statistic.sum(0), family.features.mean(0), rtol=0, atol=1e-14)
    np.testing.assert_array_equal(other._target_terms(reference.parameters)[-1].sum(1).round(12), np.ones(7))
    correction = rng.normal(size=(17, 2))
    value, analytic = family._separation(reference.parameters, correction)
    np.testing.assert_allclose(value, (posterior * correction).sum(1).mean(), rtol=0, atol=1e-14)
    finite_difference = []
    for index in range(len(reference.parameters)):
        direction = np.eye(len(reference.parameters))[index] * 1e-5
        finite_difference.append((family._separation(reference.parameters + direction, correction)[0]
                                  - family._separation(reference.parameters - direction, correction)[0]) / 2e-5)
    np.testing.assert_allclose(analytic, finite_difference, rtol=2e-5, atol=2e-7)
    z = np.column_stack((-target @ normal, target @ normal)) / 2
    w, detail = solve_master(z, family.features, [reference.statistic], regularization=.1,
                             feasibility_tol=1e-8, optimality_tol=1e-6, master_maxiter=200)
    assert detail["valid"] and w.shape == (2, 6)
    from scipy.special import logsumexp
    delta = family.features @ w.T
    empirical = np.mean(logsumexp(z + delta, axis=1) - logsumexp(z, axis=1) - (posterior * delta).sum(1))
    np.testing.assert_allclose(relative_risks(z, family.features, w, reference.statistic[None]),
                               [empirical], rtol=0, atol=1e-12)

    # Identity must preserve the full original fit/search path exactly.
    identity = ProjectedExplanationSet(source[:, :2], labels, target[:, :2], cfg,
                                       source_head_normal=normal[:2])
    stock = RestoredExplanationSet(source[:, :2], labels, target[:, :2], cfg)
    ref_identity, ref_stock = identity.fit(), stock.fit()
    for key in ("parameters", "statistic", "observation_loss"):
        np.testing.assert_array_equal(getattr(ref_identity, key), getattr(ref_stock, key))
    cut_identity, cut_stock = identity.search(correction), stock.search(correction)
    for key in ("parameters", "statistic", "observation_loss"):
        np.testing.assert_array_equal(getattr(cut_identity, key), getattr(cut_stock, key))
    assert identity.last_search == stock.last_search

    # Rank-zero sources and repeated top eigenvalues still define two axes.
    constant = ProjectedExplanationSet(np.ones((8, 4)), np.tile([0, 1], 4), np.zeros((3, 4)), cfg,
                                       source_head_normal=np.zeros(4))
    np.testing.assert_array_equal(constant.projection_basis, np.eye(4)[:, :2])
    assert constant.projection_report["scale_floored"] == [True, True]
    tied_source = np.vstack((np.eye(4), -np.eye(4), np.eye(4), -np.eye(4)))
    tied = ProjectedExplanationSet(tied_source, np.repeat([0, 1], 8), np.zeros((3, 4)), cfg,
                                   source_head_normal=np.array([1., 0., 0., 0.]))
    np.testing.assert_array_equal(tied.projection_basis, np.eye(4)[:, :2])
    assert tied.projection_report["axis_origins"][1] == "degenerate_fisher_residual_pca"
    json.dumps(family.projection_report, allow_nan=False)
    return dict(status="passed", full_statistic_shape=list(reference.statistic.shape),
                projected_dimension=family.dimension, source_only_projection=True,
                identity_fit_and_search_exact=True, rank_deficient_fallback=True,
                full_master_verified=bool(detail["valid"]))
