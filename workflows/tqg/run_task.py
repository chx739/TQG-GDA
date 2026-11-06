"""Run one explicit, label-free TQG adaptation task."""
import argparse
import json

from tqg_gda.config import Config
from tqg_gda.pipeline import run_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='New typed pipeline JSON; see configs/sega_style/')
    parser.add_argument('--version', choices=('v1', 'v2', 'source_only'))
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device')
    parser.add_argument('--out-dir')
    parser.add_argument('--cache-path')
    parser.add_argument('--source-split')
    parser.add_argument('--target-index')
    args = parser.parse_args()
    with open(args.config) as stream:
        values = json.load(stream)
    values.update({k: v for k, v in vars(args).items() if k != 'config' and v is not None})
    print(json.dumps(run_task(Config.from_dict(values)), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
