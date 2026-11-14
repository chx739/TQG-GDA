"""Run the sealed Mutagenicity R1 protocol, without score training/adaptation.

Example: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m
scripts.mutagenicity.r1_representation_boundary_support_20261111.run --r0-log /tmp/r1-r0.txt
"""
from experiments.common.paths import repository_root
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
import torch

from experiments.common.io import digest, manifest, provenance, write_json
from experiments.common.runtime import RunLog, seal, time_limit


def assert_phase(root, expected):
    seals = [stage for stage in ('S1', 'S2', 'S3') if (Path(root) / 'seals' / f'{stage}.json').exists()]
    actual = seals[-1] if seals else 'R0'
    if actual != expected:
        raise PermissionError(f'Expected phase {expected}; actual {actual}. No training/geometry after S2.')


def execute(repo, root, plan, r0_log):
    from experiments.mutagenicity.r1_representation_boundary_support_20261111.src.data import prepare_inputs
    from experiments.mutagenicity.r1_representation_boundary_support_20261111.src.source import run_source
    from experiments.mutagenicity.r1_representation_boundary_support_20261111.src.geometry import run_source_coverage, run_cell
    from experiments.mutagenicity.r1_representation_boundary_support_20261111.src.probes import run_probes
    from experiments.mutagenicity.r1_representation_boundary_support_20261111.src.evaluate import evaluate_run
    from experiments.mutagenicity.r1_representation_boundary_support_20261111.src.audit import replay_predictions
    repo, root = Path(repo), Path(root)
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True).strip():
        raise RuntimeError('Commit implementation and protocol before reading real labels')
    if not Path(r0_log).is_file() or '\nOK\n' not in Path(r0_log).read_text():
        raise RuntimeError('A passing R0 unittest log is required')
    if os.environ.get('OMP_NUM_THREADS') != '1' or os.environ.get('MKL_NUM_THREADS') != '1':
        raise RuntimeError('Set OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1 before launch')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    root.mkdir(parents=True, exist_ok=False)
    for directory in ('logs', 'protocol', 'failures', 'source', 'geometry', 'probes', 'evaluation'):
        (root / directory).mkdir()
    shutil.copyfile(r0_log, root / 'logs/R0-tests.txt')
    for path in (repo / 'experiments/mutagenicity/r1_representation_boundary_support_20261111/configs/stage1-plan.json', repo / 'experiments/mutagenicity/r1_representation_boundary_support_20261111/results/stage1-design-checks.json', repo / 'experiments/mutagenicity/r1_representation_boundary_support_20261111/docs/stage1-report-template.md', repo / 'experiments/mutagenicity/stage1-inputs.json'):
        shutil.copyfile(path, root / "protocol" / path.name)
    shutil.copyfile(repo / "experiments/mutagenicity/r1_representation_boundary_support_20261111/docs/mutagenicity-stage1-protocol.md", root / "protocol/protocol.md")
    write_json(root / 'execution-plan.json', plan)
    write_json(root / 'provenance.json', provenance(repo))
    with (root / 'protocol/source.tar').open('xb') as stream:
        subprocess.run(['git', 'archive', '--format=tar', 'HEAD'], cwd=repo, stdout=stream, check=True)
    packages = sorted({(d.metadata['Name'], d.version) for d in importlib.metadata.distributions() if d.metadata['Name']})
    with (root / 'protocol/environment.txt').open('x') as stream:
        stream.write('OMP_NUM_THREADS=1\nMKL_NUM_THREADS=1\nTorch deterministic algorithms=True\n')
        stream.write('\n'.join(f'{name}=={version}' for name, version in packages) + '\n')
    log = RunLog(root / 'logs/run.log', plan['resources']['max_total_seconds'], plan['resources']['max_rss_bytes'])
    failures = []
    outcomes = []
    started = time.monotonic()
    def failed(stage, exc):
        record = dict(stage=stage, error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        failures.append(record)
        write_json(root / 'failures' / f'{stage}.json', record)
        log(f'FAILURE {stage} {record["error"]}')
    try:
        log('R0 prepare raw inputs, historical target roles and cfg.seed source roles')
        prepare_inputs(root, repo, plan)
        write_json(root / 'R0.json', dict(status='passed', tests_sha256=digest(root / 'logs/R0-tests.txt'),
            input_verification_sha256=digest(root / 'inputs/verification.json'), final_values_parsed=False))
        for seed in plan['seeds']:
            assert_phase(root, 'R0')
            try:
                with time_limit(plan['resources']['max_source_seconds']):
                    run_source(root, seed, plan, log)
            except Exception as exc:
                failed(f'source-seed{seed}', exc)
                continue
            for _, domain in plan['tasks']:
                try:
                    with time_limit(plan['resources']['max_geometry_density_cell_seconds']):
                        run_source_coverage(root, seed, domain, plan, log)
                except Exception as exc:
                    failed(f'initial-coverage-seed{seed}-domain{domain}', exc)
        s1 = seal(root, 'S1', ['source', 'geometry', 'inputs', 'roles', 'roles.json', 'protocol',
                                  'execution-plan.json', 'provenance.json', 'R0.json', 'logs/R0-tests.txt', 'failures'])
        log(f'S1 sealed digest={s1["digest"]}; diagnostic source/A access opens')
        assert_phase(root, 'S1')
        probe_records = run_probes(root, plan, log=log)
        probe_failures = [dict(path=r['path'], status=r['status'], failure=r.get('failure'))
                          for r in probe_records if r['status'] != 'success']
        write_json(root / 'probe-outcomes.json', dict(total=len(probe_records), failed=probe_failures))
        for seed in plan['seeds']:
            if not (root / 'source' / f'seed{seed}' / 'frozen.pt').exists():
                continue
            for _, domain in plan['tasks']:
                assert_phase(root, 'S1')
                try:
                    with time_limit(plan['resources']['max_geometry_density_cell_seconds']):
                        result = run_cell(root, seed, domain, plan, log)
                        outcomes.append(dict(seed=seed, domain=domain, status=result['status']))
                except Exception as exc:
                    failed(f'geometry-seed{seed}-domain{domain}', exc)
        replay = replay_predictions(root, plan)
        write_json(root / 'geometry-outcomes.json', outcomes)
        log(f'All saved checkpoint probabilities reloaded before B access: {replay}')
        s2 = seal(root, 'S2', ['seals/S1.json', 'source', 'geometry', 'probes', 'roles', 'roles.json',
                               'inputs', 'failures', 'execution-plan.json', 'protocol', 'provenance.json', 'prediction-replay.json',
                               'probe-outcomes.json', 'geometry-outcomes.json'])
        log(f'S2 sealed digest={s2["digest"]}; B evaluation opens with frozen predictions/neighborhoods')
        assert_phase(root, 'S2')
        evaluate_run(root, plan)
        # Recheck all pre-evaluation evidence after the independent evaluator.
        # Label audit/log additions are intentionally outside S2's immutable set.
        for relative, expected in s2['files'].items():
            if digest(root / relative) != expected:
                raise RuntimeError(f'Pre-evaluation evidence changed after S2: {relative}')
        log('S3 evaluation complete; final class values remain closed')
        incomplete = bool(failures or probe_failures or any(r['status'] != 'completed' for r in outcomes))
        write_json(root / 'completion.json', dict(status='completed_with_failures_or_missing' if incomplete else 'completed',
                   failures=failures, probe_failures=len(probe_failures), geometry_outcomes=outcomes,
                   elapsed_seconds=time.monotonic()-started, score_trained=False,
                   classifier_adapted=False, final_values_parsed=False))
        log.close()
        seal(root, 'S3', ['evaluation', 'completion.json', 'seals/S1.json', 'seals/S2.json', 'logs'])
        write_json(root / 'manifest.json', manifest(root))
    except BaseException as exc:
        if not (root / 'interrupted.json').exists():
            write_json(root / 'interrupted.json', dict(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc(),
                elapsed_seconds=time.monotonic()-started, note='Preserve partial artifacts; repairs require a new attempt.'))
        log.close()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=repository_root())
    parser.add_argument('--root', type=Path)
    parser.add_argument('--r0-log', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((args.repo / 'experiments/mutagenicity/r1_representation_boundary_support_20261111/configs/stage1-plan.json').read_text())
    execute(args.repo, args.root or plan['output_root'], plan, args.r0_log)
