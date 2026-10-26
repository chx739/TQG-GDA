"""Export a completed v2 run, without loading graph labels or training models."""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from verify_experiment_archive import verify


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def export(run, previous_archive, destination):
    run, previous_archive, destination = map(
        lambda p: Path(p).resolve(), (run, previous_archive, destination))
    stage = run / 'stage3_execution'
    manifest = json.loads((stage / 'experiment_manifest.json').read_text())
    seal = json.loads((stage / 'training_seal.json').read_text())
    if not (run / 'stage_3_done').is_file() or seal['count'] != 105:
        raise ValueError('Expected a completed 105-job experiment')
    code = stage / 'code'
    commit = subprocess.check_output(
        ['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(
        ['git', '-C', str(code), 'status', '--porcelain'], text=True).strip()
    if dirty or commit != manifest['git_commit'] or commit != seal['git_commit']:
        raise ValueError('Execution snapshot differs from the sealed commit')
    destination.mkdir(parents=True, exist_ok=False)
    entries, source_map, jobs = [], {}, []

    def copy(source, relative, category, expected=None):
        source = Path(source).resolve()
        target = destination / relative
        if not target.resolve().is_relative_to(destination) or target.exists():
            raise ValueError('Unsafe or duplicate archive path: ' + str(relative))
        digest = sha256(source)
        if expected is not None and digest != expected:
            raise ValueError('Source differs from sealed hash: ' + str(source))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha256(target) != digest or sha256(source) != digest:
            raise ValueError('Copy mismatch: ' + str(source))
        entry = dict(path=str(relative), bytes=target.stat().st_size,
                     sha256=digest, source=str(source), category=category)
        entries.append(entry)
        source_map[str(source)] = entry
        return entry['path']

    old_sources = {}
    with (previous_archive / 'job_index.csv').open(newline='', encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            if row['phase'] == 'source_pretraining':
                old_sources[(row['arm'], int(row['seed']))] = row
    for model in manifest['source_models']:
        pool, seed = model['pool'], model['seed']
        old = old_sources[(pool, seed)]
        if old['training_commit'] != model['original_training_commit']:
            raise ValueError('Source training provenance mismatch')
        row = dict(phase='source_pretraining', task='source_0', seed=seed,
                   arm=pool, reused=True, training_commit=old['training_commit'],
                   source_checkpoint='')
        for key in ('history', 'last_checkpoint', 'selected_checkpoint'):
            relative = old[key]
            row[key] = copy(previous_archive / relative, relative,
                            'history' if key == 'history' else 'checkpoint',
                            model['sha256'] if key == 'selected_checkpoint' else None)
        if sha256(Path(model['path'])) != model['sha256']:
            raise ValueError('Reused source snapshot hash mismatch')
        source_map[str(Path(model['path']).resolve())] = source_map[
            str((previous_archive / old['selected_checkpoint']).resolve())]
        metadata = Path('metadata/jobs/source_pretraining') / pool / str(seed)
        for source in sorted((previous_archive / metadata).iterdir()):
            if source.is_file():
                copy(source, metadata / source.name, 'job_metadata')
        jobs.append(row)

    for record in sorted(seal['jobs'], key=lambda r: (
            r['spec']['domain'], r['spec']['seed'], r['spec']['arm'])):
        if record['status'] != 'completed' or record['attempts'] != 1:
            raise ValueError('Unexpected incomplete or retried job')
        spec = record['spec']
        task, seed, arm = '0_to_' + str(spec['domain']), spec['seed'], spec['arm']
        output = Path(record['output'])
        relative = Path(task) / str(seed) / arm
        for name, artifact in record['artifacts'].items():
            if sha256(output / name) != artifact['sha256']:
                raise ValueError('Sealed artifact changed: ' + str(output / name))
        for source in sorted(output.iterdir()):
            if not source.is_file():
                continue
            if source.suffix == '.pt':
                target, category = Path('checkpoints') / relative / source.name, 'checkpoint'
            elif source.name == 'history.jsonl':
                target, category = Path('logs') / relative / source.name, 'history'
            else:
                target, category = Path('metadata/jobs') / relative / source.name, 'job_metadata'
            copy(source, target, category)
        for name in ('completion.json', 'RUN_INFO_attempt_1.json', 'stdout_attempt_1.log'):
            copy(output.parent / name, Path('metadata/jobs') / relative / name, 'job_metadata')
        pool = 'all_source' if arm == 'B_full' else 'fit'
        source_best = stage / 'source_checkpoints' / pool / str(seed) / 'best.pt'
        jobs.append(dict(
            phase='adaptation', task=task, seed=seed, arm=arm, reused=False,
            training_commit=record['git_commit'],
            history=source_map[str((output / 'history.jsonl').resolve())]['path'],
            last_checkpoint=source_map[str((output / 'last.pt').resolve())]['path'],
            selected_checkpoint=source_map[str((output / 'selected.pt').resolve())]['path'],
            source_checkpoint=source_map[str(source_best.resolve())]['path']))

    for source in sorted(stage.iterdir()):
        if source.is_file():
            folder = 'results' if source.name in (
                'results.csv', 'results_summary.json', 'run_report.md', '结果解读.md') else 'metadata/stage'
            copy(source, Path(folder) / source.name, 'stage_metadata')
    for folder, prefix in ((stage / 'figures', 'results/figures'),
                           (run / 'stage2_method', 'protocol')):
        for source in sorted(folder.rglob('*')):
            if source.is_file():
                copy(source, Path(prefix) / source.relative_to(folder), 'protocol_or_figure')
    copy(run / 'stage_3_done', Path('metadata/stage_3_done.json'), 'stage_metadata')
    tracked = subprocess.check_output(
        ['git', '-C', str(code), 'ls-files', '-z']).decode().split('\0')
    for name in filter(None, tracked):
        copy(code / name, Path('code_snapshot') / name, 'code_snapshot')

    with (stage / 'results.csv').open(newline='', encoding='utf-8-sig') as stream:
        evaluations = list(csv.DictReader(stream))
    for row in evaluations:
        checkpoint = source_map[str(Path(row['checkpoint']).resolve())]
        if checkpoint['sha256'] != row['checkpoint_sha256']:
            raise ValueError('Evaluation checkpoint hash mismatch')
        row['archived_checkpoint'] = checkpoint['path']
        if row['arm'] == 'B_frozen':
            history = Path('logs/source_pretraining/fit') / row['seed'] / 'history.jsonl'
        else:
            history = Path('logs') / row['task'] / row['seed'] / row['arm'] / 'history.jsonl'
        row['archived_history'] = str(history)
    for name, rows in (('job_index.csv', jobs), ('evaluation_index.csv', evaluations)):
        with (destination / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    statistics = dict(
        indexed_jobs=len(jobs), adaptation_jobs=105, new_adaptation_jobs=105,
        reused_adaptation_jobs=0, shared_source_warmup_jobs=10,
        frozen_evaluation_references=15, evaluation_records=len(evaluations),
        history_files=sum(e['category'] == 'history' for e in entries),
        checkpoint_files=sum(e['category'] == 'checkpoint' for e in entries),
        total_bytes=sum(e['bytes'] for e in entries))
    if (len(jobs), len(evaluations), statistics['history_files'],
            statistics['checkpoint_files']) != (115, 225, 115, 650):
        raise ValueError('Unexpected archive counts')
    result = dict(schema_version=1, experiment='mutagenicity_v2_20261026',
                  archive_date='2026-10-26', source_run=str(run),
                  execution_commit=commit, plan_sha256=manifest['plan_sha256'],
                  new_real_data_training_runs_during_export=0,
                  scope='105 new adaptations and 10 reused source warmups; TensorBoard files and graph data excluded; JSONL metrics preserved',
                  statistics=statistics, files=entries)
    (destination / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    (destination / 'SHA256SUMS').write_text(''.join(
        e['sha256'] + '  ' + e['path'] + '\n' for e in entries))
    return verify(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--previous-archive', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.run, args.previous_archive, args.destination), indent=2))


if __name__ == '__main__':
    main()
