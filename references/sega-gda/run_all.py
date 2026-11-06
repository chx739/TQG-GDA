"""Run every ordered task M_s -> M_t (s != t) for the chosen split types and seeds, then summarise.

Example:
    python run_all.py --split_types idx,node --seeds 0,1,2
    python run_all.py --split_types idx --seeds 0 --tasks 0-1,1-0 --epochs 50
"""
import argparse
import csv
import json
import os
import traceback
from collections import defaultdict

import numpy as np

from sega.config import add_config_args, config_from_args
from sega.data import load_tu_dataset
from sega.pipeline import run_task, task_name
from sega.utils import Logger, ensure_dir


def summarise(rows, out_dir):
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[(r["split_type"], f"M{r['source']}→M{r['target']}")]["src"].append(100 * r["acc_source_only"])
        by[(r["split_type"], f"M{r['source']}→M{r['target']}")]["sega"].append(100 * r["acc_sega"])
    lines = []
    for st in sorted({k[0] for k in by}):
        tasks = sorted([k[1] for k in by if k[0] == st])
        fmt = lambda xs: f"{np.mean(xs):.2f}" + (f"±{np.std(xs):.2f}" if len(xs) > 1 else "")
        lines.append(f"\n### Mutagenicity ({st} split), target accuracy (%)\n")
        lines.append("| Method | " + " | ".join(tasks) + " | Avg |")
        lines.append("|---" * (len(tasks) + 2) + "|")
        for key, label in (("src", "Source-only (Euclidean ref. classifier)"), ("sega", "SEGA-GDA")):
            vals = [by[(st, t)][key] for t in tasks]
            # average over tasks within each seed, then mean±std over seeds
            n_seed = min(len(v) for v in vals)
            avg = [np.mean([v[s] for v in vals]) for s in range(n_seed)]
            lines.append(f"| {label} | " + " | ".join(fmt(v) for v in vals) + f" | {fmt(avg)} |")
    text = "\n".join(lines)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(text + "\n")
    print(text)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p, skip={"split_type", "source", "target", "seed"})
    p.add_argument("--split_types", type=str, default="idx,node")
    p.add_argument("--seeds", type=str, default="0")
    p.add_argument("--tasks", type=str, default="all", help="'all' or comma list like 0-1,2-3")
    args = p.parse_args()
    base = config_from_args(args)
    ensure_dir(base.out_dir)
    ds = load_tu_dataset(base.data_root, base.dataset)
    if args.tasks == "all":
        pairs = [(s, t) for s in range(4) for t in range(4) if s != t]
    else:
        pairs = [tuple(int(x) for x in tok.split("-")) for tok in args.tasks.split(",")]
    rows = []
    csv_path = os.path.join(base.out_dir, "results.csv")
    logger = Logger(os.path.join(base.out_dir, "run_all.log"), base.verbose)
    for st in args.split_types.split(","):
        for seed in [int(s) for s in args.seeds.split(",")]:
            for (s, t) in pairs:
                cfg = config_from_args(args)
                cfg.split_type, cfg.source, cfg.target, cfg.seed = st, s, t, seed
                jpath = os.path.join(cfg.out_dir, "json", task_name(cfg) + ".json")
                if os.path.exists(jpath):
                    with open(jpath) as f:
                        res = json.load(f)
                    logger.log(f"skip finished {task_name(cfg)}")
                else:
                    try:
                        res = run_task(cfg, ds=ds, logger=logger)
                    except Exception:
                        logger.log(f"FAILED {task_name(cfg)}\n{traceback.format_exc()}")
                        continue
                row = {k: res[k] for k in ("split_type", "source", "target", "seed", "acc_source_only", "acc_sega",
                                           "n_accepted_refs", "pseudo_label_acc_diag", "acc_cal_before",
                                           "acc_cal_after", "n_pred_changed")}
                row["accepted_updates"] = res["adapt"].get("accepted")
                row["stop_reason"] = res["adapt"].get("stop_reason")
                rows.append(row)
                with open(csv_path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
    if rows:
        summarise(rows, base.out_dir)


if __name__ == "__main__":
    main()
