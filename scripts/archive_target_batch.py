"""Archive the completed target-batch ablation without training or label access."""
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
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def export(run, destination):
    run, destination = Path(run).resolve(), Path(destination).resolve()
    stage, code = run/'stage3_execution', run/'stage3_execution/code'
    if not (run/'stage_3_done').exists():
        raise ValueError('Do not archive an incomplete experiment')
    manifest, seal = read(stage/'experiment_manifest.json'), read(stage/'training_seal.json')
    plan = read(run/'stage2_method/experiment_plan.yaml')
    commit = subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(code), 'status', '--porcelain'], text=True).strip()
    if dirty or commit != manifest['git_commit'] or commit != seal['git_commit']:
        raise ValueError('Frozen execution snapshot changed')
    new_count, reused_count = len(seal['jobs']), len(seal['references'])
    source_count = len(manifest['source_models'])
    if seal['count'] != new_count or new_count != len(manifest['jobs']):
        raise ValueError('Sealed experiment count differs')
    if not new_count or not reused_count or not source_count:
        raise ValueError('Empty experiment scope')
    destination.mkdir(parents=True, exist_ok=False)
    entries, source_map, jobs = [], {}, []

    def copy(source, relative, category, expected=None):
        source, relative = Path(source).resolve(), Path(relative)
        target = destination/relative
        if not target.resolve().is_relative_to(destination) or target.exists():
            raise ValueError('Unsafe or duplicate archive path: '+str(relative))
        digest = sha256(source)
        if expected is not None and digest != expected:
            raise ValueError('Source differs from sealed bytes: '+str(source))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha256(source) != digest or sha256(target) != digest:
            raise ValueError('Copy mismatch: '+str(source))
        entry = dict(path=str(relative), bytes=target.stat().st_size, sha256=digest,
                     source=str(source), category=category)
        entries.append(entry)
        source_map[str(source)] = entry
        return entry['path']

    previous = Path(plan['source_archive'])
    with (previous/'job_index.csv').open(newline='', encoding='utf-8-sig') as stream:
        sources = {(row['arm'], int(row['seed'])):row for row in csv.DictReader(stream)
                   if row['phase'] == 'source_pretraining'}
    for model in manifest['source_models']:
        pool, seed = model['pool'], model['seed']
        old = sources[(pool, seed)]
        if old['training_commit'] != model['original_training_commit']:
            raise ValueError('Source training provenance mismatch')
        row = dict(phase='source_pretraining', task='source_0', seed=seed, arm=pool,
                   reused=True, training_commit=old['training_commit'], source_checkpoint='')
        for key in ('history', 'last_checkpoint', 'selected_checkpoint'):
            relative = old[key]
            row[key] = copy(previous/relative, relative,
                            'history' if key == 'history' else 'checkpoint',
                            model['sha256'] if key == 'selected_checkpoint' else None)
        if sha256(model['path']) != model['sha256']:
            raise ValueError('Reused source model changed')
        source_map[str(Path(model['path']).resolve())] = source_map[
            str((previous/old['selected_checkpoint']).resolve())]
        metadata = Path('metadata/jobs/source_pretraining')/pool/str(seed)
        for source in sorted((previous/metadata).iterdir()):
            if source.is_file():
                copy(source, metadata/source.name, 'job_metadata')
        jobs.append(row)

    all_records = [(record, False) for record in seal['jobs']]
    all_records += [(record, True) for record in seal['references']]
    for record, reused in all_records:
        spec = record['spec']
        task, seed, arm = '0_to_'+str(spec['domain']), spec['seed'], spec['arm']
        relative = Path(task)/str(seed)/arm
        artifacts = record['artifacts']
        for item in artifacts.values():
            if sha256(item['path']) != item['sha256']:
                raise ValueError('Sealed job artifact changed')
        if reused:
            sources_to_copy = [Path(item['path']) for item in artifacts.values()]
        else:
            output = Path(record['output'])
            sources_to_copy = [p for p in sorted(output.iterdir()) if p.is_file()]
            # A resumed final attempt may not contain early periodic checkpoints.
            known_names = {p.name for p in sources_to_copy}
            for earlier in sorted(output.parent.glob('attempt_*'), reverse=True):
                for checkpoint in sorted(earlier.glob('step_*.pt')):
                    if checkpoint.name not in known_names:
                        sources_to_copy.append(checkpoint)
                        known_names.add(checkpoint.name)
        for source in sources_to_copy:
            if source.suffix == '.pt':
                target, category = Path('checkpoints')/relative/source.name, 'checkpoint'
            elif source.name == 'history.jsonl':
                target, category = Path('logs')/relative/source.name, 'history'
            else:
                target, category = Path('metadata/jobs')/relative/source.name, 'job_metadata'
            expected = artifacts.get(source.name, {}).get('sha256')
            copy(source, target, category, expected)
        if not reused:
            for source in sorted(output.parent.iterdir()):
                if source.is_file():
                    copy(source, Path('metadata/jobs')/relative/source.name, 'job_metadata')
        source_best = stage/'source_checkpoints/fit'/str(seed)/'best.pt'
        jobs.append(dict(phase='adaptation', task=task, seed=seed, arm=arm, reused=reused,
            training_commit=record['git_commit'],
            source_checkpoint=source_map[str(source_best.resolve())]['path'],
            history=source_map[str(Path(artifacts['history.jsonl']['path']).resolve())]['path'],
            last_checkpoint=source_map[str(Path(artifacts['last.pt']['path']).resolve())]['path'],
            selected_checkpoint=source_map[str(Path(artifacts['selected.pt']['path']).resolve())]['path']))

    for source in sorted(stage.iterdir()):
        if source.is_file():
            folder = 'results' if source.name in ('results.csv', 'results_summary.json', 'run_report.md', '结果解读.md', 'diagnostics_summary.csv', 'descriptive_analysis.json') else 'metadata/stage'
            copy(source, Path(folder)/source.name, 'stage_metadata')
    for source in sorted(run.iterdir()):
        if source.is_file():
            copy(source, Path('metadata/run')/source.name, 'run_metadata')
    for folder, prefix in ((stage/'figures', 'results/figures'), (run/'stage2_method', 'protocol')):
        for source in sorted(folder.rglob('*')):
            if source.is_file():
                copy(source, Path(prefix)/source.relative_to(folder), 'protocol_or_figure')
    tracked = subprocess.check_output(['git', '-C', str(code), 'ls-files', '-z']).decode().split('\0')
    for name in filter(None, tracked):
        copy(code/name, Path('code_snapshot')/name, 'code_snapshot')

    with (stage/'results.csv').open(newline='', encoding='utf-8-sig') as stream:
        evaluations = list(csv.DictReader(stream))
    for row in evaluations:
        checkpoint = source_map[str(Path(row['checkpoint']).resolve())]
        if checkpoint['sha256'] != row['checkpoint_sha256']:
            raise ValueError('Evaluation model hash mismatch')
        row['archived_checkpoint'] = checkpoint['path']
        row['archived_history'] = str(
            Path('logs/source_pretraining/fit')/row['seed']/'history.jsonl'
            if row['arm'] == 'B_frozen' else Path('logs')/row['task']/row['seed']/row['arm']/'history.jsonl')
    for name, rows in (('job_index.csv', jobs), ('evaluation_index.csv', evaluations)):
        with (destination/name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    adaptation_count = new_count + reused_count
    frozen_count = len(plan['domains']) * len(plan['seeds'])
    statistics = dict(indexed_jobs=len(jobs), adaptation_jobs=adaptation_count, new_adaptation_jobs=new_count,
        reused_adaptation_jobs=reused_count, shared_source_warmup_jobs=source_count,
        frozen_evaluation_references=frozen_count, evaluation_records=len(evaluations),
        history_files=sum(e['category'] == 'history' for e in entries),
        checkpoint_files=sum(e['category'] == 'checkpoint' for e in entries),
        total_bytes=sum(e['bytes'] for e in entries))
    expected_counts = (adaptation_count + source_count, 2*adaptation_count + frozen_count,
                       adaptation_count + source_count, 6*adaptation_count + 2*source_count)
    actual_counts = (len(jobs), len(evaluations), statistics['history_files'], statistics['checkpoint_files'])
    if actual_counts != expected_counts:
        raise ValueError('Unexpected archive counts')
    result = dict(schema_version=1, experiment=destination.name,
        archive_date=read(run/'stage_3_done')['finished_at'][:10], source_run=str(run), execution_commit=commit,
        plan_sha256=manifest['plan_sha256'], new_real_data_training_runs_during_export=0,
        scope=f'{new_count} new jobs, {reused_count} reused adaptations, {source_count} reused source warmups; graph cache and TensorBoard events excluded; JSONL retained',
        statistics=statistics, files=entries)
    (destination/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    (destination/'SHA256SUMS').write_text(''.join(e['sha256']+'  '+e['path']+'\n' for e in entries))
    return verify(destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.run, args.destination), indent=2))
