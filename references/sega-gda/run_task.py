"""Run SEGA-GDA on one source -> target task.

Example:
    python run_task.py --split_type idx --source 0 --target 1 --seed 0
"""
import argparse
import json

from sega.config import add_config_args, config_from_args
from sega.pipeline import run_task


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    cfg = config_from_args(p.parse_args())
    res = run_task(cfg)
    print(json.dumps({k: res[k] for k in ("task", "acc_source_only", "acc_sega", "n_accepted_refs",
                                          "pseudo_label_acc_diag")}, indent=1))
    print("adapt:", {k: res["adapt"].get(k) for k in ("accepted", "stop_reason", "L_DA_0", "L_DA_star")})


if __name__ == "__main__":
    main()
