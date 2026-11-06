"""Numerical sanity checks of the SEGA-GDA implementation on a small synthetic problem.

Run:  python -m pytest -q tests/test_sanity.py      or      python tests/test_sanity.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sega.classifier import scale_weights, sq_euclidean  # noqa: E402
from sega.config import Config  # noqa: E402
from sega.coverage import build_coverage  # noqa: E402
from sega.geodesic import GeodesicSolver  # noqa: E402
from sega.geometry import FieldParam  # noqa: E402
from sega.gram import GramAccumulator  # noqa: E402
from sega.problem import SegaProblem  # noqa: E402
from sega.subproblem import ProxSubproblem  # noqa: E402

DT = torch.float64


def _setup(d=4, K=3, rho=0.5, kappa=2.0, seed=0, L=4, J=6, Q=2):
    g = torch.Generator().manual_seed(seed)
    C = 2
    z_ref = torch.randn(24, d, generator=g, dtype=DT)
    y_ref = torch.tensor([0] * 12 + [1] * 12)
    z_ref[y_ref == 1] += 1.5
    z_fit = torch.randn(20, d, generator=g, dtype=DT)
    y_fit = torch.tensor([0] * 10 + [1] * 10)
    z_fit[y_fit == 1] += 1.5
    z_tp = torch.randn(16, d, generator=g, dtype=DT) + 0.4
    yhat = torch.tensor([0] * 8 + [1] * 8)
    z_tp[yhat == 1] += 1.2
    w = torch.rand(16, generator=g, dtype=DT) * 0.5 + 0.5
    cov = build_coverage(torch.cat([z_ref, z_fit]), z_tp, K, 0.7, 0.1, seed)
    field = FieldParam(d, cov.nu, kappa, rho, dtype=DT)
    solver = GeodesicSolver(cov, L, Q, J, 0.5 / (4 * L * field.m_plus), dtype=DT)
    cfg = Config(emb_dim=d, K=K, rho=rho, kappa=kappa, L_path=L, J_path=J, Q_quad=Q, tau=0.2,
                 pair_chunk_jac=64, pair_chunk_eval=4096)
    v = scale_weights(y_ref, C, dtype=DT)
    m = np.array([0.3, 0.3])
    prob = SegaProblem(cfg, field, solver, z_ref, y_ref, v, z_fit, y_fit, z_tp, yhat, w, [0, 1], m, C)
    return cfg, cov, field, solver, prob


def _random_feasible_b(field, scale=1.0, seed=1):
    rng = np.random.RandomState(seed)
    u = rng.randn(field.P)
    u /= np.linalg.norm(u)
    t = scale
    while not field.is_feasible(t * u):
        t *= 0.7
    return torch.as_tensor(t * u, dtype=DT)


def test_identity_metric_gives_euclidean():
    cfg, cov, field, solver, prob = _setup()
    b0 = torch.zeros(field.P, dtype=DT)
    for mode in ("eval", "unrolled", "envelope"):
        D = solver.solve(prob.z_fit, prob.z_ref, field.mats(b0), mode).detach()
        assert torch.allclose(D, sq_euclidean(prob.z_fit, prob.z_ref), atol=1e-12), mode


def test_analytic_path_gradient():
    cfg, cov, field, solver, prob = _setup(Q=2)
    b = _random_feasible_b(field, 3.0, seed=7)
    mats = field.mats(b)
    start, end = solver._endpoints(prob.z_fit[:5], prob.z_tp[:6])
    pts = start[..., None, :] + solver.frac[:, None] * (end - start)[..., None, :]
    pts = pts + 0.3 * torch.randn_like(pts)
    pts[..., 0, :], pts[..., -1, :] = start, end
    inner = pts[..., 1:-1, :].clone().requires_grad_(True)
    E = solver.energy(torch.cat([start[..., None, :], inner, end[..., None, :]], -2), mats)
    g_auto = torch.autograd.grad(E.sum(), inner)[0]
    E2, g_an = solver.energy_and_grad(pts, mats)
    assert torch.allclose(E.detach(), E2, atol=1e-12)
    assert torch.allclose(g_auto, g_an, atol=1e-10)
    # forward-only (analytic) and unrolled (autograd) relaxations give the same distances
    D_eval = solver.solve(prob.z_fit, prob.z_tp, mats, "eval")
    D_unr = solver.solve(prob.z_fit, prob.z_tp, mats, "unrolled").detach()
    assert torch.allclose(D_eval, D_unr, atol=1e-11)


def test_symmetry_and_bounds():
    cfg, cov, field, solver, prob = _setup()
    b = _random_feasible_b(field, 3.0)
    mats = field.mats(b)
    D_xy = solver.solve(prob.z_fit, prob.z_tp, mats, "eval")
    D_yx = solver.solve(prob.z_tp, prob.z_fit, mats, "eval")
    assert torch.allclose(D_xy, D_yx.T, atol=1e-12)
    B = mats[0]
    diff = prob.z_fit[:, None] - prob.z_tp[None]
    DB = torch.einsum("nmd,de,nme->nm", diff, B, diff)
    assert bool((D_xy <= (1 + field.rho) * DB + 1e-10).all())
    assert bool((D_xy >= (1 - field.rho) * DB - 1e-10).all())
    # identical endpoints -> zero distance
    assert float(solver.solve(prob.z_fit[:3], prob.z_fit[:3], mats, "eval").diagonal().abs().max()) == 0.0


def test_parameterisation():
    cfg, cov, field, solver, prob = _setup(d=5, K=4)
    assert field.P == 4 * 5 * 6 // 2 - 1
    b = torch.randn(field.P, dtype=DT) * 0.1
    M = field.mats(b).numpy()
    assert abs(np.trace(M[0]) - 5) < 1e-12                                   # tr B = d
    assert np.abs(np.tensordot(field.nu, M[1:], axes=1)).max() < 1e-12       # sum nu_k E_k = 0
    # orthonormal basis: ||dB||_F^2 + sum_k ||E_k||_F^2 == ||b||^2
    nrm = ((M[0] - np.eye(5)) ** 2).sum() + (M[1:] ** 2).sum()
    assert abs(nrm - float(b @ b)) < 1e-10
    # linear maps used by the sub-problem agree with mats()
    A_B, A_E = field.linear_maps()
    assert np.allclose(np.eye(5).ravel() + A_B @ b.numpy(), M[0].ravel())
    for k in range(field.K):
        assert np.allclose(A_E[k] @ b.numpy(), M[1 + k].ravel())


def test_jacobian_matches_finite_differences():
    for mode in ("unrolled", "envelope"):
        cfg, cov, field, solver, prob = _setup()
        cfg.jac_mode = mode
        b = _random_feasible_b(field, 2.0, seed=3)
        jp = prob.jacobian_pass(b)
        rng = np.random.RandomState(0)
        u = torch.as_tensor(rng.randn(field.P), dtype=DT)
        u /= u.norm()
        eps = 1e-5
        ep, em = prob.evaluate(b + eps * u), prob.evaluate(b - eps * u)
        for key, J in (("S_fit", "J_fit"), ("S_fitT", "J_fitT"), ("S_tp", "J_tp")):
            fd = (ep[key] - em[key]) / (2 * eps)
            an = jp[J] @ u
            err = float((fd - an).abs().max() / (fd.abs().max() + 1e-12))
            tol = 1e-5 if mode == "unrolled" else 5e-2       # envelope is exact only at stationary paths
            assert err < tol, (mode, key, err)
        # Jacobian pass reproduces the forward evaluation exactly
        e0 = prob.evaluate(b)
        assert torch.allclose(e0["S_fit"], jp["S_fit"], atol=1e-12)


def test_gram_matches_direct_integral():
    cfg, cov, field, solver, prob = _setup(d=3, K=3, L=3, J=4, Q=1)
    for b in (torch.zeros(field.P, dtype=DT), _random_feasible_b(field, 2.0, seed=5)):
        acc = GramAccumulator(field, res=400, dtype=DT)
        prob.jacobian_pass(b, acc)
        G = acc.finalize(b)
        # direct: collect the same nodes and weights, evaluate tr(g^-1 dg g^-1 dg)
        mats = field.mats(b)
        nodes, wts = [], []
        for X, Y in ((prob.z_fit, prob.z_ref), (prob.z_tp, prob.z_ref), (prob.z_fit, prob.z_tp)):
            _, path = solver.solve(X, Y, mats, "eval", return_path=True)
            dp = path[..., 1:, :] - path[..., :-1, :]
            nd = path[..., :-1, None, :] + solver.t[:, None] * dp[..., None, :]
            nodes.append(nd.reshape(-1, field.d))
            wts.append(((0.5 / prob.n_paths) / solver.L * solver.w).expand(nd.shape[:-1]).reshape(-1))
        nodes.append(cov.mu_points); wts.append(0.5 * cov.mu_weights)
        Z, W = torch.cat(nodes), torch.cat(wts)
        assert abs(float(W.sum()) - 1.0) < 1e-10
        rng = np.random.RandomState(1)
        for _ in range(3):
            D = rng.randn(field.P)
            Dm = field.mats(torch.as_tensor(D)) - field.mats(torch.zeros(field.P, dtype=DT))
            Dm[0] = Dm[0]                                                     # delta B (I cancels)
            al = cov.alpha(Z)
            dg = Dm[0] + torch.einsum("nk,kde->nde", al, Dm[1:])
            g = mats[0] + torch.einsum("nk,kde->nde", al, mats[1:])
            Gi = torch.linalg.inv(g)
            direct = float((W * torch.einsum("nab,nbc,ncd,nda->n", Gi, dg, Gi, dg)).sum())
            quad = float(D @ G @ D)
            assert abs(direct - quad) / abs(direct) < 2e-3, (direct, quad)


def test_subproblem_zero_step_and_validity():
    cfg, cov, field, solver, prob = _setup()
    b = torch.zeros(field.P, dtype=DT)
    acc = GramAccumulator(field, res=16, dtype=DT)
    jp = prob.jacobian_pass(b, acc)
    M = acc.finalize(b) + 1e-4 * np.eye(field.P)
    cur = {k: (None if jp[k] is None else jp[k].numpy()) for k in ("S_fit", "S_fitT", "S_tp", "J_fit", "J_fitT", "J_tp")}
    budgets = jp["R_c"] + 0.05
    sub = ProxSubproblem(cfg, field, prob.numpy_meta(), cur, budgets, M, b.numpy())
    lv0 = sub.local_values(np.zeros(field.P))
    assert abs(lv0["L_DA"] - jp["L_DA"]) < 1e-10
    assert np.allclose(lv0["R_c"], jp["R_c"])
    res = sub.solve(1.0)
    assert res["Delta"] is not None and res["valid"], res["status"]
    assert field.is_feasible(res["Delta"])
    assert res["obj"] <= res["obj0"] + 1e-12


def test_edge_cases_rho0_kappa1_largeK():
    for kw in ({"rho": 0.0}, {"kappa": 1.0}, {"d": 2, "K": 17}):
        cfg, cov, field, solver, prob = _setup(**kw)
        b = torch.zeros(field.P, dtype=DT)
        acc = GramAccumulator(field, res=16, dtype=DT)
        jp = prob.jacobian_pass(b, acc)
        G = acc.finalize(b)
        assert G.shape == (field.P, field.P) and np.all(np.linalg.eigvalsh(G) > -1e-10), kw
        if kw.get("rho") == 0.0:
            assert field.P == field.s - 1
        if kw.get("kappa") == 1.0:
            assert field.P == (field.K - 1) * field.s
        cur = {k: (None if jp[k] is None else jp[k].numpy()) for k in ("S_fit", "S_fitT", "S_tp", "J_fit", "J_fitT", "J_tp")}
        sub = ProxSubproblem(cfg, field, prob.numpy_meta(), cur, jp["R_c"] + 0.05, G + 1e-4 * np.eye(field.P), b.numpy())
        res = sub.solve(1.0)
        assert res["valid"] and field.is_feasible(res["Delta"]), (kw, res["status"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
