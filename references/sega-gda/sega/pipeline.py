"""End-to-end SEGA-GDA for one source -> target task (Sec. 3.11, stages A-D)."""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Dict, Optional

import numpy as np
import torch

from .adapt import adapt_geometry
from .classifier import scale_weights
from .coverage import build_coverage
from .data import graph_labels, load_domain, load_tu_dataset, split_source_roles
from .encoder import GraphEncoder
from .geodesic import GeodesicSolver
from .geometry import FieldParam
from .predict import predict
from .pretrain import encode, pretrain_encoder
from .problem import SegaProblem
from .target_refs import select_target_refs
from .utils import Logger, ensure_dir, get_device, get_dtype, set_seed

ENC_KEYS = ("dataset", "split_type", "source", "seed", "frac_ref", "frac_fit", "encoder", "hidden", "num_layers",
            "emb_dim", "dropout", "epochs", "lr", "weight_decay", "query_batch", "tau")


def encoder_cache_path(cfg) -> str:
    key = json.dumps({k: getattr(cfg, k) for k in ENC_KEYS}, sort_keys=True)
    h = hashlib.md5(key.encode()).hexdigest()[:10]
    return os.path.join(cfg.out_dir, "encoders",
                        f"{cfg.dataset}_{cfg.split_type}_src{cfg.source}_seed{cfg.seed}_{h}.pt")


def _to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    return x


def task_name(cfg) -> str:
    return f"{cfg.dataset}_{cfg.split_type}_M{cfg.source}toM{cfg.target}_seed{cfg.seed}{('_' + cfg.tag) if cfg.tag else ''}"


def run_task(cfg, ds=None, logger: Optional[Logger] = None) -> Dict:
    t_start = time.time()
    ensure_dir(cfg.out_dir)
    name = task_name(cfg)
    own_logger = logger is None
    if own_logger:
        logger = Logger(os.path.join(ensure_dir(os.path.join(cfg.out_dir, "logs")), name + ".log"), cfg.verbose)
    device = get_device(cfg.device)
    gdt = get_dtype(cfg.geo_dtype)
    set_seed(cfg.seed)
    logger.log(f"=== {name}  (device {device}) ===")

    # ------------------------------------------------------------ stage A
    if ds is None:
        ds = load_tu_dataset(cfg.data_root, cfg.dataset)
    y_all = graph_labels(ds)
    C = int(y_all.max() + 1)
    src_ids = load_domain(cfg.split_dir, cfg.split_type, cfg.source, cfg.dataset)
    tgt_ids = load_domain(cfg.split_dir, cfg.split_type, cfg.target, cfg.dataset)
    if len(np.intersect1d(src_ids, tgt_ids)):
        logger.log("  WARNING: source and target domains overlap")
    roles = split_source_roles(src_ids, y_all, cfg.frac_ref, cfg.frac_fit, C, cfg.seed)
    logger.log(f"  source {len(src_ids)} graphs: |ref| {len(roles.ref)}  |fit| {len(roles.fit)}  "
               f"|cal| {len(roles.cal)};  target {len(tgt_ids)} graphs")

    cache = encoder_cache_path(cfg)
    model = GraphEncoder(ds.num_node_features, ds.num_edge_features, cfg.hidden, cfg.num_layers, cfg.emb_dim,
                         cfg.dropout, cfg.encoder).to(device)
    if cfg.cache_encoders and os.path.exists(cache):
        ck = torch.load(cache, map_location=device, weights_only=False)
        model.load_state_dict(ck["state"])
        pre_info = ck["info"]
        logger.log(f"  loaded source encoder from {cache}")
    else:
        model, pre_info = pretrain_encoder(cfg, ds, roles, y_all, C, device, logger)
        if cfg.cache_encoders:
            ensure_dir(os.path.dirname(cache))
            torch.save({"state": model.state_dict(), "info": pre_info}, cache)
    model.eval()

    # frozen coordinates (encoder never changes after this point)
    enc = lambda ids: encode(model, ds, ids, device).to(gdt)
    z_ref, z_fit, z_cal, z_tgt = enc(roles.ref), enc(roles.fit), enc(roles.cal), enc(tgt_ids)
    tt = lambda a: torch.as_tensor(a, device=device)
    y_ref, y_fit, y_cal = tt(y_all[roles.ref]), tt(y_all[roles.fit]), tt(y_all[roles.cal])
    y_tgt = tt(y_all[tgt_ids])                       # used ONLY for reporting accuracy
    v = scale_weights(y_ref, C, dtype=gdt).to(device)

    # ------------------------------------------------------------ stage B
    pos_in_ref = {int(g): i for i, g in enumerate(roles.ref)}
    sub1 = np.array([pos_in_ref[int(g)] for g in roles.ref_sub1])
    sub2 = np.array([pos_in_ref[int(g)] for g in roles.ref_sub2])
    tr = select_target_refs(z_ref, y_ref, sub1, sub2, z_cal, y_cal, z_tgt, v, C, cfg.tau, cfg.q_m, cfg.q_R,
                            cfg.n_cal_min, cfg.target_filter)
    acc_idx = torch.as_tensor(tr.accepted, dtype=torch.long, device=device)
    pl_acc = float((y_tgt[acc_idx].cpu().numpy() == tr.yhat).mean()) if len(tr.accepted) else float("nan")
    logger.log(f"  target refs: |J+| {len(tr.accepted)}/{len(tgt_ids)}  C_T {tr.C_T}  per class "
               f"{tr.info['accepted_per_class']}  m_c {np.round(tr.m, 4).tolist()}  R_c {np.round(tr.R, 4).tolist()}"
               f"  [diagnostic pseudo-label acc {pl_acc:.4f}]")

    cov = build_coverage(torch.cat([z_ref, z_fit], 0), z_tgt, cfg.K, cfg.c_h, cfg.c_min, cfg.seed, cfg.kmeans_n_init)
    field = FieldParam(cfg.emb_dim, cov.nu, cfg.kappa, cfg.rho, dtype=gdt, device=device)
    eta_path = cfg.c_path / (4.0 * cfg.L_path * field.m_plus)
    solver = GeodesicSolver(cov, cfg.L_path, cfg.Q_quad, cfg.J_path, eta_path, dtype=gdt, device=device)
    logger.log(f"  coverage: K {cov.K}  sigma {cov.sigma:.4f}  nu {np.round(cov.nu, 3).tolist()}  P {field.P}  "
               f"eta_path {eta_path:.4g}")

    b0 = torch.zeros(field.P, dtype=gdt, device=device)
    yhat0, prob0, _ = predict(z_tgt, z_ref, y_ref, v, field, solver, b0, cfg.tau, C, cfg.pair_chunk_eval)
    acc0 = float((yhat0 == y_tgt).double().mean())

    # ------------------------------------------------------------ stage C
    if len(tr.C_T) < 2:
        logger.log("  |C_T| < 2: no bidirectional adaptation, returning the initial source classifier")
        adapt_res = {"accepted": 0, "stop_reason": "|C_T|<2", "b_star": np.zeros(field.P), "history": []}
        b_star = b0
    else:
        z_tp = z_tgt[acc_idx]
        problem = SegaProblem(cfg, field, solver, z_ref, y_ref, v, z_fit, y_fit, z_tp,
                              tt(tr.yhat), torch.as_tensor(tr.w, dtype=gdt, device=device), tr.C_T, tr.m, C)
        adapt_res = adapt_geometry(problem, field, cfg, logger)
        b_star = torch.as_tensor(adapt_res["b_star"], dtype=gdt, device=device)

    # ------------------------------------------------------------ stage D
    yhat, prob, _ = predict(z_tgt, z_ref, y_ref, v, field, solver, b_star, cfg.tau, C, cfg.pair_chunk_eval)
    acc = float((yhat == y_tgt).double().mean())
    ycal0, _, _ = predict(z_cal, z_ref, y_ref, v, field, solver, b0, cfg.tau, C, cfg.pair_chunk_eval)
    ycal, _, _ = predict(z_cal, z_ref, y_ref, v, field, solver, b_star, cfg.tau, C, cfg.pair_chunk_eval)
    cal0 = float((ycal0 == y_cal).double().mean())
    cal1 = float((ycal == y_cal).double().mean())
    changed = int((yhat != yhat0).sum())
    logger.log(f"  RESULT {name}: target acc  source-only {acc0:.4f}  ->  SEGA-GDA {acc:.4f}   "
               f"(source cal acc {cal0:.4f} -> {cal1:.4f}; {changed} target predictions changed)")

    # save everything final prediction needs (Sec. 3.11): encoder cache path, reference coords/labels,
    # tau, coverage centres + bandwidth, final B, E_k, numerical spec.
    B, E = field.BE_numpy(adapt_res["b_star"]) if field.P else (np.eye(cfg.emb_dim), np.zeros((cov.K, cfg.emb_dim, cfg.emb_dim)))
    ensure_dir(os.path.join(cfg.out_dir, "models"))
    torch.save({"encoder_state": model.state_dict(), "z_ref": z_ref.cpu(), "y_ref": y_ref.cpu(), "v": v.cpu(),
                "tau": cfg.tau, "centers": cov.centers.cpu(), "sigma": cov.sigma, "nu": cov.nu, "B": B, "E": E,
                "b_star": adapt_res["b_star"],
                "numeric_spec": {"L_path": cfg.L_path, "Q_quad": cfg.Q_quad, "J_path": cfg.J_path,
                                 "eta_path": eta_path, "tie_rule": "strict < keeps earliest iterate; "
                                 "lexicographic unordered pairs; argmax ties -> first class"},
                "config": cfg.to_dict()}, os.path.join(cfg.out_dir, "models", name + ".pt"))

    result = {
        "task": name, "split_type": cfg.split_type, "source": cfg.source, "target": cfg.target, "seed": cfg.seed,
        "acc_source_only": acc0, "acc_sega": acc, "acc_cal_before": cal0, "acc_cal_after": cal1,
        "n_pred_changed": changed, "n_target": len(tgt_ids),
        "n_accepted_refs": int(len(tr.accepted)), "C_T": tr.C_T, "pseudo_label_acc_diag": pl_acc,
        "m_c": tr.m, "R_c": tr.R, "K_eff": cov.K, "P": field.P,
        "pretrain": {kk: vv for kk, vv in pre_info.items() if kk != "history"},
        "adapt": {kk: vv for kk, vv in adapt_res.items() if kk not in ("b_star",)},
        "time_total": time.time() - t_start, "config": cfg.to_dict(),
    }
    with open(os.path.join(ensure_dir(os.path.join(cfg.out_dir, "json")), name + ".json"), "w") as f:
        json.dump(_to_jsonable(result), f, indent=1)
    if own_logger:
        logger.close()
    return result
