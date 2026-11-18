"""Command-line entry point for R7 engineering checks and phase S."""
import argparse
import os
from pathlib import Path
from tempfile import TemporaryDirectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('check', 'S'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.command == 'S' and args.output is None:
        parser.error('S requires a new --output directory')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    from ..src import stage_s

    round_dir = Path(__file__).resolve().parents[1]
    plan = round_dir / 'configs' / 'r7-plan.json'
    if args.command == 'check':
        stage_s.controls.check_controls()
        with TemporaryDirectory() as folder:
            stage_s.check_seals(Path(folder) / 'seals')
            stage_s.check_replay(stage_s.json_read(plan), Path(folder) / 'replay')
        if not stage_s.check_known_moment()['passed']:
            raise RuntimeError('Known-moment fixture failed')
        print('R7 engineering fixtures passed')
    else:
        stage_s.run(plan, args.output, args.workers, round_dir.parents[2])


if __name__ == '__main__':
    main()
