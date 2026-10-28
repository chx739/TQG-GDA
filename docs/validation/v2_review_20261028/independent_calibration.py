"""Independent dense primal check; does not use real graph data or labels."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tqg_gda.calibration_v2 import CalibrationConfig, solve_calibration


def check(seed, q, r):
    rng = np.random.default_rng(seed)
    ns, nt = 4, 3
    features = rng.normal(size=(ns+nt, 5))
    kernel = np.exp(-((features[:, None]-features[None, :])**2).sum(-1)/5)
    kss = np.kron(kernel[:ns, :ns], np.eye(q))
    b = rng.normal(size=(nt*q, r))/nt
    hst = np.kron(kernel[:ns, ns:], np.eye(q)) @ b
    htt = b.T @ np.kron(kernel[ns:, ns:], np.eye(q)) @ b
    lam = .17
    n = ns*q

    def error(w):
        return htt-w@hst-hst.T@w.T+w@kss@w.T

    def objective(x):
        w = x[:-1].reshape(r, n)
        return np.trace(error(w))/r+lam*np.sum(w*w)/r+x[-1]

    def gradient(x):
        w = x[:-1].reshape(r, n)
        return np.r_[(2*(w@kss-hst.T+lam*w)/r).ravel(), 1.]

    def slack(x):
        return x[-1]-np.linalg.eigvalsh(error(x[:-1].reshape(r, n)))

    def slack_jac(x):
        w = x[:-1].reshape(r, n)
        _, vectors = np.linalg.eigh(error(w))
        rows = []
        for v in vectors.T:
            rows.append(np.r_[(-2*np.outer(v, v@(w@kss-hst.T))).ravel(), 1.])
        return np.stack(rows)

    initial = np.linalg.solve(kss+lam*np.eye(n), hst).T
    x0 = np.r_[initial.ravel(), np.linalg.eigvalsh(error(initial))[-1]+.1]
    solved = minimize(objective, x0, jac=gradient, method='SLSQP',
                      constraints=[dict(type='ineq', fun=slack, jac=slack_jac)],
                      options=dict(ftol=1e-12, maxiter=2000))
    w, info = solve_calibration(
        torch.from_numpy(kernel[:ns, :ns]), torch.from_numpy(hst), torch.from_numpy(htt),
        CalibrationConfig(lambda_w=lam, max_iterations=200, gap_atol=1e-8, gap_rtol=1e-6))
    actual = w.numpy()
    e = error(actual)
    dense_value = np.trace(e)/r+np.linalg.eigvalsh(e)[-1]+lam*np.sum(actual**2)/r
    gap_limit = 1e-8+1e-6*max(abs(info['primal']), abs(info['dual']))
    row = dict(seed=seed, q=q, r=r, independent_success=bool(solved.success),
               independent_feasibility=float(np.min(slack(solved.x))),
               independent_objective=float(solved.fun), production_objective=float(dense_value),
               absolute_difference=float(abs(dense_value-solved.fun)),
               logged_primal_difference=float(abs(dense_value-info['primal'])),
               production_converged=info['converged'], production_gap=info['gap'],
               production_tolerance=gap_limit)
    assert solved.success and row['independent_feasibility'] > -1e-8, row
    assert row['logged_primal_difference'] < 1e-11, row
    assert row['absolute_difference'] < info['gap']+1e-7, row
    return row


if __name__ == '__main__':
    torch.set_num_threads(1)
    rows = [check(seed, q, r) for seed, q, r in (
        (11, 1, 2), (22, 2, 3), (33, 3, 2), (44, 2, 3), (55, 1, 3))]
    output = dict(passed=True, kind='independent_dense_primal_slsqp',
                  target_labels_read=False, real_data_training=False, cases=rows)
    Path(__file__).with_suffix('.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output, indent=2))
