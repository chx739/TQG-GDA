"""Command-line entry for this experiment."""
import argparse
import json
from pathlib import Path
import torch
from experiments.common.io import write_json
from experiments.synthetic_data.e2_适配轨迹回放_20261110.src.analysis import write_csv
from experiments.synthetic_data.e2_静态分数估计_20261110.src.analysis import aggregate, audit, figures, paired

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    audit_result, metrics, signals, fits, gates = audit(args.run)
    pairs = paired(metrics)
    summary = dict(metrics=aggregate(metrics, ('dim','setting','bank','comparison')),
                   signals=aggregate(signals, ('dim','setting','bank','estimator')),
                   paired=aggregate(pairs, ('dim','setting','bank','comparison')),
                   gates=aggregate(gates, ('dim','setting')))
    write_json(args.output/'checkpoint-audit.json', audit_result)
    write_json(args.output/'summary.json', summary)
    for name, rows in (('metrics',metrics),('signals',signals),('fits',fits),('paired',pairs),('gates',gates)):
        write_csv(args.output/(name+'.csv'), rows)
    figures(metrics, signals, args.output)
    write_json(args.output/'analysis-notes.json', dict(primary_checkpoint='final400',
        secondary_checkpoint='minimum fixed same-pool noise validation loss',
        units='normal_rmse_raw is raw-coordinate score; request_rmse_raw is raw displacement',
        paired='setting minus n512_s1 for each seed; no factorial interaction estimate',
        gate='descriptive conjunction: oracle toward-target fraction=1; learned fraction>0.5; others in frozen plan; no closed-loop run',
        label_access=False, public_svg_whitespace_normalization='public copy only; XML-equivalent; archived original preserved'))
    print(json.dumps(audit_result, indent=2))
