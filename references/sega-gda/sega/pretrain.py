"""Source pre-training with the same reference classifier (Sec. 3.3).

L_src^warm(phi) = - mean_{i in I_fit} log p^{A_S(phi)}_{I_d}(y_i | f_phi(G_i)).
Query coordinates, all source-reference coordinates and the normaliser are
recomputed by the CURRENT encoder at every step; gradients flow through all three.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch_geometric.data import Batch

from .classifier import RefLibrary, class_scores, cross_entropy, normaliser, scale_weights, sq_euclidean
from .encoder import GraphEncoder


def _ref_variance(z_ref: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """V_A(phi) = sum_i v_i || a_i - abar ||^2."""
    abar = (v[:, None] * z_ref).sum(0)
    return (v * ((z_ref - abar) ** 2).sum(1)).sum()


@torch.no_grad()
def encode(model: GraphEncoder, dataset, ids: np.ndarray, device, batch_size: int = 512) -> torch.Tensor:
    model.eval()
    out = []
    for s in range(0, len(ids), batch_size):
        b = Batch.from_data_list([dataset[int(i)] for i in ids[s:s + batch_size]]).to(device)
        out.append(model(b))
    return torch.cat(out, 0)


def _ref_ce(z_q, y_q, z_ref, y_ref, v, num_classes, tau):
    lib = RefLibrary.build(z_ref, y_ref, classes=list(range(num_classes)))
    D = sq_euclidean(z_q, z_ref)
    S = class_scores(D, normaliser(D, v), lib, tau)
    return cross_entropy(S, y_q), S


def pretrain_encoder(cfg, dataset, roles, y_all: np.ndarray, num_classes: int, device, logger):
    torch.manual_seed(cfg.seed)
    in_dim = dataset.num_node_features
    edge_dim = dataset.num_edge_features
    model = GraphEncoder(in_dim, edge_dim, cfg.hidden, cfg.num_layers, cfg.emb_dim,
                         cfg.dropout, cfg.encoder).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    ref_list = [dataset[int(i)] for i in roles.ref]
    fit_ids = roles.fit
    y_ref = torch.as_tensor(y_all[roles.ref], device=device)
    v = scale_weights(y_ref, num_classes, dtype=torch.float32).to(device)
    n_ref = len(ref_list)
    rng = np.random.RandomState(cfg.seed)

    best = {"ce": float("inf"), "state": None, "epoch": -1}
    # most recent state VERIFIED to have a non-degenerate source head (model + optimizer moments)
    last_valid = None
    history = []
    for epoch in range(cfg.epochs):
        model.train()
        perm = fit_ids[rng.permutation(len(fit_ids))]
        tot, cnt, rejected = 0.0, 0, 0
        for s in range(0, len(perm), cfg.query_batch):
            q_ids = perm[s:s + cfg.query_batch]
            batch = Batch.from_data_list(ref_list + [dataset[int(i)] for i in q_ids]).to(device)
            z = model(batch)
            z_ref, z_q = z[:n_ref], z[n_ref:]
            if float(_ref_variance(z_ref, v)) <= cfg.degenerate_eps:
                # the previous update produced a degenerate source head: reject it and return to the
                # most recent verified-valid state, including the optimizer moments (3.3)
                if last_valid is None:
                    raise RuntimeError("encoder initialisation gives coincident source references (V_A = 0)")
                model.load_state_dict(last_valid[0])
                opt.load_state_dict(last_valid[1])
                rejected += 1
                continue
            # current parameters verified non-degenerate -> snapshot before stepping
            last_valid = (copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict()))
            y_q = torch.as_tensor(y_all[q_ids], device=device)
            ce, _ = _ref_ce(z_q, y_q, z_ref, y_ref, v, num_classes, cfg.tau)
            loss = ce.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(q_ids)
            cnt += len(q_ids)

        # checkpoint selection on I_cal with the same classifier (eval mode)
        z_ref_e = encode(model, dataset, roles.ref, device)
        if float(_ref_variance(z_ref_e, v)) <= cfg.degenerate_eps:
            continue
        z_cal = encode(model, dataset, roles.cal, device)
        y_cal = torch.as_tensor(y_all[roles.cal], device=device)
        with torch.no_grad():
            ce_cal, S_cal = _ref_ce(z_cal, y_cal, z_ref_e, y_ref, v, num_classes, cfg.tau)
        ce_cal = float(ce_cal.mean())
        acc_cal = float((S_cal.argmax(1) == y_cal).float().mean())
        history.append({"epoch": epoch, "train_ce": tot / max(cnt, 1), "cal_ce": ce_cal,
                        "cal_acc": acc_cal, "rejected": rejected})
        if ce_cal < best["ce"]:
            best = {"ce": ce_cal, "state": copy.deepcopy(model.state_dict()), "epoch": epoch, "acc": acc_cal}
        if logger is not None and (epoch % 10 == 0 or epoch == cfg.epochs - 1):
            logger.log(f"  warm-up ep {epoch:3d}  train CE {tot / max(cnt, 1):.4f}  "
                       f"cal CE {ce_cal:.4f}  cal acc {acc_cal:.4f}")
    if best["state"] is None:
        raise RuntimeError("no valid (non-degenerate) checkpoint found")
    model.load_state_dict(best["state"])
    model.eval()
    if logger is not None:
        logger.log(f"  selected checkpoint epoch {best['epoch']} (cal CE {best['ce']:.4f}, cal acc {best['acc']:.4f})")
    return model, {"best_epoch": best["epoch"], "best_cal_ce": best["ce"], "best_cal_acc": best["acc"],
                   "history": history}
