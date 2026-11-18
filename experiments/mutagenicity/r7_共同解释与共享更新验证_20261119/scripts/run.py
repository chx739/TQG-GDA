"""Command-line entry point for R7 engineering checks, synthetic confirmation, and gated real evaluation."""
import argparse
import os
from pathlib import Path
from tempfile import TemporaryDirectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('check', 'S', 'S-audit', 'P', 'C'))
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--s-output', type=Path)
    parser.add_argument('--p-output', type=Path)
    parser.add_argument('--original', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.command in ('S', 'S-audit', 'P', 'C') and args.output is None:
        parser.error('Research commands require a new --output directory')
    if args.command == 'P' and args.s_output is None:
        parser.error('P requires the passed --s-output directory')
    if args.command == 'C' and args.p_output is None:
        parser.error('C requires the passed --p-output directory')
    if args.command == 'S-audit' and args.original is None:
        parser.error('S-audit requires the sealed --original S directory')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    from ..src import stage_s

    round_dir = Path(__file__).resolve().parents[1]
    plan = args.plan or round_dir / 'configs' / 'r7-plan.json'
    if args.command == 'check':
        from ..src import anchored_explanations, matched_tsba, mixture_explanations, paired, posterior_anchored_explanations, posterior_mass_explanations, projected_explanations, stage_p

        stage_s.controls.check_controls()
        paired.check_paired()
        matched_tsba.check_matched_tsba()
        projected_explanations.check_projected_explanations()
        anchored_explanations.check_anchored_explanations()
        mixture_explanations.check_mixture_explanations()
        posterior_anchored_explanations.check_posterior_anchored_explanations()
        posterior_mass_explanations.check_posterior_mass_explanations()
        posterior_mass_explanations.check_mixture_mass_explanations()
        stage_p.check_gate(stage_s.json_read(plan))
        with TemporaryDirectory() as folder:
            stage_s.check_seals(Path(folder) / 'seals')
            stage_s.check_replay(stage_s.json_read(plan), Path(folder) / 'replay')
        if not stage_s.check_known_moment()['passed']:
            raise RuntimeError('Known-moment fixture failed')
        print('R7 engineering fixtures passed')
    elif args.command == 'S':
        stage_s.run(plan, args.output, args.workers, round_dir.parents[2])
    elif args.command == 'S-audit':
        from ..src import separation_audit
        separation_audit.run(args.original, args.output, round_dir.parents[2])
    elif args.command == 'P':
        from ..src import stage_p
        stage_p.run(plan, args.output, args.s_output, args.workers, round_dir.parents[2])
    else:
        from ..src import stage_c
        stage_c.run(plan, args.output, args.p_output, args.workers, round_dir.parents[2])


if __name__ == '__main__':
    main()
