"""Serial, budgeted training entry point for complete predeclared A/B phases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from revision_common import Context, Budget, STAGE, atomic_json, read
from training_revision import run_job
from tqg_gda.utils import sha256


def phase_arms(context, phase):
    arms = [name for name, cfg in context.arms.items()
            if cfg.get('training') and cfg.get('phase') == phase]
    expected = {'A': 8, 'B': 12}
    if phase not in expected or len(arms) != expected[phase]:
        raise ValueError('Predeclared phase must contain all eight A or twelve B arms')
    return arms


def require_phase_gate(phase, stage=STAGE):
    if phase == 'B':
        path = Path(stage) / 'diagnostics' / 'D2' / 'decision.json'
        if not path.exists() or read(path).get('allow_B') is not True:
            raise PermissionError('Phase B requires diagnostics/D2/decision.json allow_B=true')


def job_keys(context, phase):
    arms = phase_arms(context, phase)
    # Profiling is the prefix of the first *real* T5/T6 job, later resumed.
    profile_arm = 'T5' if phase == 'A' else 'T6'
    arms = [profile_arm] + [name for name in arms if name != profile_arm]
    domains = [int(task.rsplit('_', 1)[-1]) for task in context.plan['datasets'][0]['tasks']]
    return [(domain, seed, arm) for domain in domains for seed in context.plan['seeds'] for arm in arms]


def run_phase(context, phase, *, profile=False, resume=False, budget=None, stage=STAGE):
    stage = Path(stage)
    require_phase_gate(phase, stage)
    keys = job_keys(context, phase)
    expected = 80 if phase == 'A' else 120
    if len(keys) != expected:
        raise ValueError('Planned tasks/seeds do not form the complete registered phase')
    selected = keys[:1] if profile else keys
    manifest_path = stage / f'phase_{phase}_training_manifest.json'
    if manifest_path.exists():
        if not resume:
            raise FileExistsError('Existing phase requires --resume')
        manifest = read(manifest_path)
        if manifest.get('git_commit') != context.commit or manifest.get('phase') != phase:
            raise ValueError('Phase recovery changed code identity')
    else:
        manifest = dict(phase=phase, git_commit=context.commit, expected_jobs=expected,
                        records=[], status='in_progress', target_labels_used=False)
    known = {(row['domain'], row['seed'], row['arm']): row for row in manifest['records']}
    for domain, seed, arm in selected:
        if budget is not None:
            budget.check()
        output = stage / 'jobs' / f'0_to_{domain}' / str(seed) / arm
        result = run_job(context, domain, seed, arm, output=output,
                         stop_after=10 if profile else None, resume=resume, budget=budget)
        known[(domain, seed, arm)] = dict(domain=domain, task=f'0_to_{domain}', seed=seed, arm=arm,
            output=str(output), status=result['status'], completed_updates=result['completed_updates'],
            result_sha256=sha256(output / 'result.json'), config_sha256=result['config_sha256'],
            source_checkpoint_sha256=result['source_checkpoint_sha256'],
            final_checkpoint=result['last_checkpoint'], selected_checkpoint=result['selected_checkpoint'])
        manifest['records'] = [known[key] for key in keys if key in known]
        atomic_json(manifest_path, manifest)
    complete = (len(known) == expected and all(row['status'] == 'completed' for row in known.values()))
    manifest['status'] = 'completed' if complete else 'profile_paused' if profile else 'incomplete'
    atomic_json(manifest_path, manifest)
    if complete:
        hashes = {str(manifest_path): sha256(manifest_path)}
        sealed_jobs = []
        for record in manifest['records']:
            output = Path(record['output'])
            result = read(output / 'result.json')
            outputs = {}
            for name, expected_sha in result['artifact_hashes'].items():
                path = output / name
                if sha256(path) != expected_sha:
                    raise ValueError('Cannot seal altered training artifact: ' + str(path))
                hashes[str(path)] = expected_sha
                outputs[name] = dict(path=str(path), sha256=expected_sha)
            hashes[str(output / 'result.json')] = sha256(output / 'result.json')
            outputs['result.json'] = dict(path=str(output / 'result.json'),
                                          sha256=hashes[str(output / 'result.json')])
            for path in sorted(output.glob('step_*.pt')):
                hashes[str(path)] = sha256(path)
                outputs[path.name] = dict(path=str(path), sha256=hashes[str(path)])
            sealed_jobs.append(dict(task=record['task'], seed=record['seed'], arm=record['arm'],
                                    path=str(output), outputs=outputs))
        atomic_json(stage / f'phase_{phase}_training_seal.json', dict(
            phase=phase, complete=True, complete_jobs=expected, git_commit=context.commit,
            commit=context.commit, jobs=sealed_jobs,
            target_labels_used=False, final_labels_used=False, artifact_hashes=hashes))
    records = []
    for candidate in ('A', 'B'):
        path = stage / f'phase_{candidate}_training_manifest.json'
        if path.exists():
            records.extend(read(path)['records'])
    atomic_json(stage / 'runs_manifest.json', dict(git_commit=context.commit,
                target_labels_used=False, records=records))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('A', 'B'), required=True)
    parser.add_argument('--profile', action='store_true', help='Continue first T5/T6 job only through update 10')
    parser.add_argument('--resume', action='store_true', help='Recover exact state from the same committed job')
    args = parser.parse_args(argv)
    context = Context()
    require_phase_gate(args.phase)
    budget = Budget('training_' + args.phase + ('_profile' if args.profile else ''))
    status = 'failed'
    try:
        result = run_phase(context, args.phase, profile=args.profile, resume=args.resume, budget=budget)
        status = result['status']
        print(json.dumps(dict(phase=args.phase, status=status, jobs=len(result['records']))), flush=True)
    finally:
        budget.finish(status)


if __name__ == '__main__':
    main()
