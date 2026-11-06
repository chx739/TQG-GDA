"""Self-evolving shared geometry: proposal, non-linear re-evaluation, acceptance, back-tracking, stop (Sec. 3.9)."""
from __future__ import annotations

import time
from typing import Dict

import numpy as np
import torch

from .gram import GramAccumulator
from .subproblem import ProxSubproblem


def _np(x):
    return None if x is None else x.detach().cpu().numpy()


def adapt_geometry(problem, field, cfg, logger) -> Dict:
    dtype, device = problem.dtype, problem.device
    P = field.P
    b = torch.zeros(P, dtype=dtype, device=device)                      # b_0 = 0  <=>  g = I_d
    hist = []
    t0 = time.time()

    ev0 = problem.evaluate(b)
    budgets = ev0["R_c"] + cfg.eps_src if cfg.use_risk_budget else None   # b_c^risk = R_0c + eps_c
    logger.log(f"  b_0: L_F {ev0['L_F']:.5f}  L_R {ev0['L_R']:.5f}  L_DA {ev0['L_DA']:.5f}  "
               f"R_c {np.round(ev0['R_c'], 4).tolist()}  P={P}")
    result = {"R0_c": ev0["R_c"].tolist(), "budgets": None if budgets is None else budgets.tolist(),
              "L_DA_0": ev0["L_DA"], "L_F_0": ev0["L_F"], "L_R_0": ev0["L_R"]}
    stop_reason, accepted = "R_max", 0

    if P == 0:
        stop_reason = "P=0"
    else:
        while accepted < cfg.R_max:
            tr = time.time()
            gram = GramAccumulator(field, cfg.gram_alpha_res, dtype=dtype, device=device)
            jp = problem.jacobian_pass(b, gram)
            L_cur = jp["L_DA"]
            if L_cur <= cfg.eps_loss:
                stop_reason = "loss<=eps_loss"
                break
            G = gram.finalize(b)
            M = G + cfg.eps_M * np.eye(P)
            cur = {k: _np(jp[k]) for k in ("S_fit", "S_fitT", "S_tp", "J_fit", "J_fitT", "J_tp")}
            sub = ProxSubproblem(cfg, field, problem.numpy_meta(), cur, budgets, M, _np(b))
            t_build = time.time() - tr
            eta = cfg.eta_init
            done, acc_this = False, False
            for trial in range(cfg.J_back + 1):
                if eta < cfg.eta_min:
                    stop_reason, done = "eta<eta_min", True
                    break
                ts = time.time()
                sol = sub.solve(eta)
                t_solve = time.time() - ts
                rec = {"round": accepted, "trial": trial, "eta": eta, "status": sol["status"],
                       "valid": sol["valid"], "L_DA_cur": L_cur, "t_solve": t_solve}
                if not sol["valid"]:
                    why = sol["status"]
                    if sol.get("local") is not None:
                        why += (f"; local R_c {np.round(sol['local']['R_c'], 5).tolist()} vs budget, "
                                f"obj {sol.get('obj', float('nan')):.6g} vs zero-step {sol.get('obj0', float('nan')):.6g}")
                    logger.log(f"  r{accepted:02d} t{trial} eta {eta:.2e}: no valid candidate ({why})")
                    hist.append(rec)
                    eta *= cfg.beta_step
                    continue
                D = sol["Delta"]
                lv = sol["local"]
                pred = L_cur - lv["L_DA"]
                rec.update({"pred": pred, "normM": lv["normM"]})
                if pred <= cfg.eps_pred:
                    hist.append(rec)
                    stop_reason, done = "pred<=eps_pred", True
                    break
                if lv["normM"] <= cfg.eps_step:
                    hist.append(rec)
                    stop_reason, done = "step<=eps_step", True
                    break
                cand = b + torch.as_tensor(D, dtype=dtype, device=device)
                feas = field.is_feasible(_np(cand))
                ev = problem.evaluate(cand)
                ared = L_cur - ev["L_DA"]
                risk_ok = True if budgets is None else bool(np.all(ev["R_c"] <= budgets))
                ok = (pred > cfg.eps_pred) and (ared >= cfg.c_acc * pred) and risk_ok and feas
                rec.update({"ared": ared, "risk_ok": risk_ok, "feasible": feas, "accepted": ok,
                            "L_F": ev["L_F"], "L_R": ev["L_R"], "L_DA": ev["L_DA"], "R_c": ev["R_c"].tolist()})
                hist.append(rec)
                logger.log(f"  r{accepted:02d} t{trial} eta {eta:.2e}  pred {pred:.5f}  ared {ared:.5f}  "
                           f"|D|_M {lv['normM']:.4f}  L_F {ev['L_F']:.4f} L_R {ev['L_R']:.4f}  "
                           f"R_c {np.round(ev['R_c'], 4).tolist()}  feas {feas}  -> {'ACCEPT' if ok else 'reject'}"
                           f"  (build {t_build:.1f}s, solve {t_solve:.1f}s)")
                if ok:
                    b = cand
                    accepted += 1
                    acc_this = True
                    break
                eta *= cfg.beta_step
            if done:
                break
            if not acc_this:
                stop_reason = "backtracking exhausted"
                break

    ev = problem.evaluate(b)
    result.update({"b_star": _np(b), "accepted": accepted, "stop_reason": stop_reason, "history": hist,
                   "L_DA_star": ev["L_DA"], "L_F_star": ev["L_F"], "L_R_star": ev["L_R"],
                   "R_c_star": ev["R_c"].tolist(), "feasibility_star": field.feasibility(_np(b)),
                   "time_adapt": time.time() - t0})
    logger.log(f"  stop: {stop_reason}; accepted {accepted}; L_DA {result['L_DA_0']:.5f} -> {ev['L_DA']:.5f}; "
               f"R_c {np.round(ev['R_c'], 4).tolist()}")
    return result
