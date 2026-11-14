"""Resume evaluation in a new attempt from verified, immutable S2 evidence.

This entry point does not train models, select checkpoints, construct queries,
or estimate density. It preserves the interrupted attempt and checks that
previously written scalar results are byte-identical after a plotting repair.
"""
from experiments.common.paths import repository_root
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import torch

from experiments.common.synthetic.archive import digest, manifest, provenance, write_json
from experiments.common.real_data.evaluate import evaluate_run
from experiments.common.real_data.runtime import RunLog, seal


def resume(source, root, repo, tests):
    source, root, repo = Path(source), Path(root), Path(repo)
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True).strip():
        raise RuntimeError('Commit the evaluation repair before execution')
    if not (source / 'interrupted.json').is_file() or not (source / 'seals/S2.json').is_file():
        raise ValueError('Requires an interrupted attempt with frozen S2 observations')
    if (source / 'seals/S3.json').exists():
        raise ValueError('Completed runs are not repair inputs')
    if '\nOK\n' not in Path(tests).read_text():
        raise ValueError('Passing repair tests required')
    if os.environ.get('OMP_NUM_THREADS') != '1' or os.environ.get('MKL_NUM_THREADS') != '1':
        raise ValueError('Single-thread environment required')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    plan = json.loads((source / 'execution-plan.json').read_text())
    s1 = json.loads((source / 'seals/S1.json').read_text())
    s2 = json.loads((source / 'seals/S2.json').read_text())
    files = {**s1['files'], **s2['files']}
    files.update({'seals/S1.json': digest(source / 'seals/S1.json'), 'seals/S2.json': digest(source / 'seals/S2.json')})
    for relative, expected in files.items():
        if digest(source / relative) != expected:
            raise ValueError(f'Dependency integrity failure: {relative}')
    root.mkdir(parents=True, exist_ok=False)
    for relative in sorted(files):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, destination)
    (root / 'failures').mkdir(exist_ok=True)
    (root / 'logs').mkdir(exist_ok=True)
    for original, destination in [('logs/run.log', 'logs/original-run.log'),
                                   ('logs/label-access.jsonl', 'logs/original-label-access.jsonl'),
                                   ('interrupted.json', 'original-interruption.json')]:
        shutil.copy2(source / original, root / destination)
    shutil.copy2(tests, root / 'logs/repair-tests.txt')
    write_json(root / 'evaluation-repair-provenance.json', dict(**provenance(repo),
        source_attempt=str(source), source_S2_digest=s2['digest'],
        reason='Plot-only Wilson interval floating-point endpoint repair',
        models_retrained=False, queries_reconstructed=False, checkpoints_reselected=False,
        original_scalar_outputs={str(p.relative_to(source)): digest(p) for p in sorted((source / 'evaluation/public').glob('*.json'))}))
    with (root / 'protocol/evaluation-repair-source.tar').open('xb') as stream:
        subprocess.run(['git', 'archive', '--format=tar', 'HEAD'], cwd=repo, stdout=stream, check=True)
    log = RunLog(root / 'logs/run.log')
    started = time.monotonic()
    try:
        log('Verified and copied immutable S1/S2 dependencies; evaluation-only repair begins')
        result = evaluate_run(root, plan)
        comparisons = []
        for path in sorted((source / 'evaluation/public').glob('*.json')):
            relative = path.relative_to(source)
            equal = digest(path) == digest(root / relative)
            comparisons.append(dict(path=str(relative), byte_identical=equal, sha256=digest(path)))
            if not equal:
                raise ValueError(f'Plot repair changed pre-existing scalar output: {relative}')
        for relative, expected in files.items():
            if digest(root / relative) != expected:
                raise ValueError(f'Reused evidence changed: {relative}')
        write_json(root / 'repair-verification.json', dict(scalar_comparisons=comparisons,
                   all_scalar_outputs_byte_identical=True, reused_files_verified=len(files)))
        write_json(root / 'completion.json', dict(status='completed_evaluation_only_repair',
                   source_attempt=str(source), source_S2_digest=s2['digest'], original_interruption_retained=True,
                   elapsed_seconds=time.monotonic()-started, score_trained=False, classifier_adapted=False,
                   models_retrained=False, final_values_parsed=False))
        log('Evaluation/figures completed; scalar outputs unchanged and reused evidence hashes intact')
        log.close()
        seal(root, 'S3', ['evaluation', 'completion.json', 'seals/S1.json', 'seals/S2.json', 'logs',
                         'repair-verification.json', 'evaluation-repair-provenance.json',
                         'original-interruption.json', 'protocol/evaluation-repair-source.tar'])
        write_json(root / 'manifest.json', manifest(root))
        return result
    except BaseException as exc:
        import traceback
        write_json(root / 'interrupted.json', dict(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()))
        log.close()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=repository_root())
    parser.add_argument('--tests', type=Path, required=True)
    args = parser.parse_args()
    resume(args.source, args.root, args.repo, args.tests)


if __name__ == '__main__':
    main()
