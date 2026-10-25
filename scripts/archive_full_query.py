"""Copy completed full-query evidence into a new archive without training models."""
import argparse
import csv
import json
from pathlib import Path
import re
import shutil

from verify_experiment_archive import sha256, verify


STAGES = ('stage3_query_full_20261025', 'stage3_query_full_domain1_20261026')


def read(path):
    return json.loads(path.read_text())


def archive(run, destination):
    run = run.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    files, source_map, jobs, records = [], {}, [], []

    def copy(source, relative, category):
        source = source.resolve()
        relative = Path(relative)
        target = destination / relative
        if target.exists() or not target.resolve().is_relative_to(destination.resolve()):
            raise ValueError('Unsafe or duplicate archive destination: ' + str(relative))
        target.parent.mkdir(parents=True, exist_ok=True)
        before = sha256(source)
        shutil.copy2(source, target)
        if sha256(target) != before or sha256(source) != before:
            raise ValueError('Source changed or copy mismatch: ' + str(source))
        item = dict(path=relative.as_posix(), bytes=target.stat().st_size,
                    sha256=before, source=str(source), category=category)
        files.append(item)
        source_map[str(source)] = item
        return relative.as_posix()

    def copy_job(source, key):
        paths = {}
        for path in sorted(source.iterdir()):
            if not path.is_file():
                continue
            if path.suffix == '.pt':
                relative = Path('checkpoints') / key / path.name
                category = 'checkpoint'
            elif path.suffix == '.jsonl':
                relative = Path('logs') / key / path.name
                category = 'history' if path.name == 'history.jsonl' else 'monitor'
            elif path.suffix == '.json':
                relative = Path('metadata') / 'jobs' / key / path.name
                category = 'job_metadata'
            else:
                continue
            paths[path.name] = copy(path, relative, category)
        for required in ('history.jsonl', 'last.pt', 'config.json', 'result.json'):
            if required not in paths:
                raise ValueError('Incomplete source job: ' + str(source / required))
        return paths

    # Copy shared source warmups once; the fifteen frozen references reuse fit models.
    old = read(run / 'stage3_execution' / 'training_manifest.json')
    warmups = sorted({str(Path(row['warmup_checkpoint']).parent) for row in old['records']})
    source_models = {}
    for folder in warmups:
        source = Path(folder)
        match = re.fullmatch(r'warmup_(fit|all_source)_(\d+)', source.parent.name)
        if match is None:
            raise ValueError('Unexpected warmup directory: ' + folder)
        pool, seed = match.group(1), int(match.group(2))
        key = Path('source_pretraining') / pool / str(seed)
        paths = copy_job(source, key)
        source_models[(pool, seed)] = paths['best.pt']
        jobs.append(dict(phase='source_pretraining', task='source_0', seed=seed,
                         arm=pool, reused='shared', training_commit=old['code_git_commit'],
                         history=paths['history.jsonl'], last_checkpoint=paths['last.pt'],
                         selected_checkpoint=paths['best.pt'], source_checkpoint='',
                         original_output=folder))

    for stage in STAGES:
        source_stage = run / stage
        stage_records = read(source_stage / 'training_manifest.json')['records']
        records.extend(stage_records)
        for path in sorted(source_stage.glob('*.json')):
            copy(path, Path('metadata') / 'stages' / stage / path.name, 'stage_metadata')
        for row in stage_records:
            if row['arm'] == 'B_frozen':
                continue
            task = '0_to_' + str(row['domain'])
            seed, arm = int(row['seed']), row['arm']
            source = Path(row['output'])
            key = Path('adaptation') / task / str(seed) / arm
            paths = copy_job(source, key)
            pool = 'all_source' if arm == 'B_full' else 'fit'
            jobs.append(dict(phase='adaptation', task=task, seed=seed, arm=arm,
                             reused=row['reused'], training_commit=row['training_commit'],
                             history=paths['history.jsonl'], last_checkpoint=paths['last.pt'],
                             selected_checkpoint=paths['selected.pt'],
                             source_checkpoint=source_models[(pool, seed)],
                             original_output=str(source)))

    with (destination / 'job_index.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(jobs[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(jobs)
    results = run / 'stage3_query_full_summary_20261026' / 'results.csv'
    with results.open(newline='', encoding='utf-8-sig') as stream:
        evaluations = list(csv.DictReader(stream))
    for row in evaluations:
        checkpoint = source_map[str(Path(row['checkpoint']).resolve())]
        if checkpoint['sha256'] != row['checkpoint_sha256']:
            raise ValueError('Evaluation checksum differs from archived checkpoint')
        row['archived_checkpoint'] = checkpoint['path']
        row['archived_history'] = source_map[
            str((Path(row['checkpoint']).parent / 'history.jsonl').resolve())]['path']
    with (destination / 'evaluation_index.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(evaluations[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(evaluations)

    files.sort(key=lambda item: item['path'])
    manifest = dict(schema_version=1, experiment='proteins_full_query_20261026',
                    archive_date='2026-10-26', source_run=str(run),
                    scope='Latest three-domain results, reused baselines, and shared source warmups',
                    new_real_data_training_runs_during_export=0,
                    statistics=dict(indexed_jobs=len(jobs), adaptation_jobs=105,
                                    new_adaptation_jobs=85, reused_adaptation_jobs=20,
                                    shared_source_warmup_jobs=10, frozen_evaluation_references=15,
                                    evaluation_records=len(evaluations),
                                    history_files=sum(x['category'] == 'history' for x in files),
                                    checkpoint_files=sum(x['category'] == 'checkpoint' for x in files),
                                    total_bytes=sum(x['bytes'] for x in files)),
                    files=files)
    (destination / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    (destination / 'SHA256SUMS').write_text(''.join(
        item['sha256'] + '  ' + item['path'] + '\n' for item in files))
    result = verify(destination)
    if (result['jobs'], result['evaluations'], result['history_files'], result['checkpoint_files']) != (115, 225, 115, 755):
        raise ValueError('Unexpected completed-experiment scope')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(archive(args.run_root, args.destination), indent=2))


if __name__ == '__main__':
    main()
