"""Validate R10 process lineage without opening holdout labels, or resume exact code."""
import argparse
import json
from pathlib import Path

from experiments.common.io import write_json
from ..src import jobs, validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--receipt',type=Path)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    repo=Path(__file__).resolve().parents[4]
    root=args.root.resolve()
    _,report=validation.validate_jobs(root,repo,seeds=[args.seed] if args.seed is not None else None,
                                     allow_missing=args.resume)
    if args.receipt:
        write_json(args.receipt,report)
    print(json.dumps(report,ensure_ascii=False),flush=True)
    if args.resume:
        jobs.supervise(root,validation.MODULE)


if __name__ == '__main__':
    main()
