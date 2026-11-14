"""Command-line entry for this experiment."""
import argparse
import json
from pathlib import Path
from experiments.common.io import write_json
from experiments.synthetic.e1_e2_20261109.src.analysis import plots, summarize, write_csv

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();summary=summarize(args.run.resolve())
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'summary.json',summary)
    for field in ('evaluation','score_queries','trajectories','decompositions','nonlinear_geometry'):
        write_csv(args.output/(field+'.csv'),summary[field])
    plots(summary,args.output)
    print(json.dumps(summary['grouped'],indent=2))
