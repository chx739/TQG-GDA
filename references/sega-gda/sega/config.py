"""Configuration for SEGA-GDA.

Every hyper-parameter of the design document (Sec. 3.12) appears here with the
same meaning. All values are fixed before adaptation and never selected with
target labels.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass, field, fields
from typing import Optional


@dataclass
class Config:
    # ------------------------------------------------------------------ data
    data_root: str = "./data"            # PyG TUDataset root (contains Mutagenicity/raw)
    split_dir: str = "./splits"          # folder with {idx,node}_Mutagenicity_{0..3}.npy
    dataset: str = "Mutagenicity"
    split_type: str = "idx"              # "idx" or "node"
    source: int = 0
    target: int = 1
    # source role split (Sec. 3.1.2), stratified by class, by original graph id
    frac_ref: float = 0.4
    frac_fit: float = 0.4                # cal = remaining fraction
    seed: int = 0

    # --------------------------------------------------------------- encoder
    encoder: str = "gin"                 # "gin" or "gine" (uses bond-type edge attributes)
    hidden: int = 64
    num_layers: int = 3
    emb_dim: int = 16                    # d, graph representation dimension
    dropout: float = 0.0

    # ------------------------------------------------- source pre-training (3.3)
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 0.0
    query_batch: int = 128
    tau: float = 0.1                     # kernel temperature, fixed during adaptation
    degenerate_eps: float = 1e-10        # V_A(phi) > eps required

    # ------------------------------------------ trusted target references (3.4)
    q_m: float = 0.50
    q_R: float = 0.95
    n_cal_min: int = 5
    target_filter: str = "robust"        # "robust" (design) | "all" (ablation: every non-tied argmax, w=1)

    # ------------------------------------------------------- coverage (3.5.1)
    K: int = 4
    c_h: float = 0.5
    c_min: float = 0.1
    kmeans_n_init: int = 10

    # ------------------------------------------------- feasible set (3.5.2)
    kappa: float = 3.0
    rho: float = 0.5

    # ------------------------------------------------------ source budget (3.7)
    eps_src: float = 0.05                # per-class CE tolerance epsilon_c (same for all classes)

    # ------------------------------------------- prox-linear step (3.8, 3.9)
    eps_M: float = 1e-4
    eta_init: float = 10.0
    eta_min: float = 1e-4
    beta_step: float = 0.5
    c_acc: float = 0.1
    eps_pred: float = 1e-6
    eps_loss: float = 1e-6
    eps_step: float = 1e-6
    J_back: int = 8
    R_max: int = 30
    sdp_margin: float = 1e-6             # tightening of the LMIs inside the sub-problem
    local_feas_tol: float = 1e-7         # tolerance for local CE feasibility of a solver output
    solver: str = "CLARABEL"             # cvxpy solver; falls back to SCS on failure
    solver_verbose: bool = False

    # ------------------------------------------ numerical geodesic (3.10)
    L_path: int = 4
    Q_quad: int = 1                      # Gauss-Legendre nodes per segment (1 = midpoint)
    J_path: int = 8
    c_path: float = 0.5                  # eta_path = c_path / (4 * L_path * m_plus)
    jac_mode: str = "unrolled"           # "unrolled" (differentiate the finite solver) | "envelope"
    gram_alpha_res: int = 16             # alpha-space resolution for the deformation Gram (3.8.4)

    # --------------------------------------------------------------- compute
    device: str = "auto"
    geo_dtype: str = "float64"
    pair_chunk_eval: int = 131072        # pairs per chunk in forward-only evaluation
    pair_chunk_jac: int = 8192           # pairs per chunk in the Jacobian pass

    # ------------------------------------------------------------- ablations
    use_forward: bool = True             # L_F term
    use_reverse: bool = True             # L_R term
    use_risk_budget: bool = True         # per-class source CE budget

    # ----------------------------------------------------------------- output
    out_dir: str = "./results"
    cache_encoders: bool = True
    tag: str = ""
    verbose: int = 1

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def _str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("1", "true", "t", "yes", "y"):
        return True
    if v.lower() in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"boolean expected, got {v}")


def add_config_args(parser: argparse.ArgumentParser, skip: Optional[set] = None) -> None:
    """Expose every Config field as --name on the command line."""
    skip = skip or set()
    for f in fields(Config):
        if f.name in skip:
            continue
        default = f.default
        if isinstance(default, bool):
            parser.add_argument(f"--{f.name}", type=_str2bool, default=None)
        else:
            parser.add_argument(f"--{f.name}", type=type(default), default=None)
    parser.add_argument("--config", type=str, default=None, help="optional JSON file with overrides")


def config_from_args(args: argparse.Namespace) -> Config:
    cfg = Config()
    if getattr(args, "config", None):
        with open(args.config) as f:
            for k, v in json.load(f).items():
                if not hasattr(cfg, k):
                    raise KeyError(f"unknown config key {k}")
                setattr(cfg, k, v)
    for f in fields(Config):
        v = getattr(args, f.name, None)
        if v is not None:
            setattr(cfg, f.name, v)
    return cfg
