"""Command-line entry for this experiment."""
import argparse
import json
from pathlib import Path
import torch
from experiments.common.io import write_json
from experiments.synthetic.e2_replay_20261110.src.analysis import audit, plots, read, report, summarize, write_csv

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1)
    a=audit(args.run);rows=read(args.run/'metrics.json');s,flat_rows=summarize(rows)
    s.update(audit=a,execution=read(args.run/'execution.json'),execution_commit=read(args.run/'provenance.json')['git_commit'],failures=read(args.run/'failures.json'))
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'summary.json',s);write_json(args.output/'checkpoint-audit.json',a)
    write_csv(args.output/'trajectory.csv',flat_rows);write_csv(args.output/'paired.csv',s['paired']);write_csv(args.output/'onsets.csv',s['onsets'])
    plots(flat_rows,args.output)
    with (args.output/'report.md').open('x') as f:f.write(report(s))
    print(json.dumps(a,indent=2))
