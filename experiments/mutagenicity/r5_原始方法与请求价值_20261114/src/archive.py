"""Read sealed R5 results, independently audit summaries and publish byte copies."""
import argparse
import csv
from fractions import Fraction
import json
import math
from pathlib import Path
import shutil
import statistics
from .shared import io, output, p4, digest

SETTINGS = ('mutagenicity_edge', 'mutagenicity_node', 'proteins_edge')
METRICS = ('accuracy', 'balanced_accuracy', 'macro_f1', 'nll')


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def close(actual, expected, context):
    if not math.isclose(float(actual), float(expected), abs_tol=1e-12, rel_tol=1e-10):
        raise ValueError(f'{context}: {actual} != {expected}')


def audit_evaluation(directory, plan):
    endpoints = read_csv(directory / 'endpoints.csv')
    final = {(int(r['source']), int(r['target']), int(r['seed']), r['method']): r for r in endpoints
             if r['pool'] == 'target' and int(r['attempt']) == 30}
    if len(final) != 180:
        raise ValueError('Missing final method/task/seed endpoint')
    original = {(r['method_or_contrast'], r['scope']): r
                for r in json.loads((directory / 'summary.json').read_text())}
    recomputed, seeds = [], []
    for left, right in [('B0', None), ('B1', None), ('A0', None), *plan['pairs']]:
        name = left if right is None else f'{left}−{right}'
        values = {metric: [] for metric in METRICS}
        for seed in plan['seeds']:
            cells = {metric: [] for metric in METRICS}
            for source, target in plan['directions']:
                a = final[source, target, seed, left]
                b = final[source, target, seed, right] if right else None
                if b and a['n'] != b['n']:
                    raise ValueError('Unpaired target cardinality')
                for metric in METRICS:
                    if metric == 'accuracy':
                        value = Fraction(int(a['correct']) - (int(b['correct']) if b else 0), int(a['n']))
                    else:
                        value = float(a[metric]) - (float(b[metric]) if b else 0.)
                    cells[metric].append(value)
            row = dict(setting=plan['setting']['id'], method_or_contrast=name, seed=seed)
            for metric in METRICS:
                total = sum(cells[metric]) if metric == 'accuracy' else math.fsum(cells[metric])
                value = total / len(plan['directions'])
                values[metric].append(value)
                row[metric] = float(value)
            seeds.append(row)
        for metric in METRICS:
            raw = values[metric]
            mean = float(sum(raw) / len(raw))
            sd = statistics.stdev(map(float, raw))
            expected = original[name, 'Avg'][metric]
            close(mean, expected['mean'], f'{name}/{metric}/mean')
            close(sd, expected['sd'], f'{name}/{metric}/sd')
            signs = [(-v if metric == 'nll' else v) for v in raw]
            row = dict(setting=plan['setting']['id'], method_or_contrast=name, metric=metric,
                       mean=mean, sd=sd, n_seeds=len(raw))
            if right:
                row.update(wins=sum(v > 0 for v in signs), ties=sum(v == 0 for v in signs), losses=sum(v < 0 for v in signs))
                for key in ('wins','ties','losses'):
                    if row[key] != expected[key]:
                        raise ValueError('Incorrect exact paired sign counts')
            recomputed.append(row)
    pairs = read_csv(directory / 'side_paired.csv')
    grouped = {}
    for row in pairs:
        key = tuple(row[k] for k in ('source','target','seed','attempt','contrast'))
        grouped.setdefault(key, {})[row['pool']] = row
        if int(row['n']):
            if int(row['wrong_to_right']) - int(row['right_to_wrong']) != int(row['net_correct']):
                raise ValueError('Side correction count mismatch')
            close(row['accuracy'], int(row['net_correct']) / int(row['n']), 'side accuracy')
    for pools in grouped.values():
        for partition in (('generators','complement'), ('initially_correct','initially_wrong')):
            if sum(int(pools[p]['n']) for p in partition) != int(pools['target']['n']):
                raise ValueError('Side partition does not cover target')
            close(sum(float(pools[p]['nll_contribution']) if int(pools[p]['n']) else 0. for p in partition),
                  pools['target']['nll'], 'NLL partition conservation')
    return recomputed, seeds, dict(final_endpoints=len(final), side_paired_blocks=len(grouped),
                                  summary_recomputed_from_endpoint_counts=True, exact_accuracy_ties=True,
                                  side_correction_and_nll_partition_checks=True)


def copy_identical(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if digest(source) != digest(destination):
            raise ValueError(f'Published evidence differs: {destination}')
    else:
        shutil.copyfile(source, destination)


def archive(base, experiment):
    # Gate the entire publication before touching results or parsing metric files.
    roots = {setting: base / setting / 'attempt-001' for setting in SETTINGS}
    for root in roots.values():
        if not (root / 'seals/R5_FINAL.json').is_file():
            raise PermissionError(f'R5 not complete: {root}')
    entries, aggregate, seed_rows, models, configs = [], [], [], [], {}
    for setting, root in roots.items():
        seals = {name: io.verify(root, f'R5_{name}') for name in ('INPUT','OBS','FINAL')}
        plan = json.loads((root / 'protocol/plan.json').read_text())
        provenance = json.loads((root / 'protocol/provenance.json').read_text())
        replay = json.loads((root / 'replay.json').read_text())
        if any(replay[k] != n for k,n in dict(sources=20,scores=60,score_endpoints=240,
                    trajectories=120,attempts=3600,predictions_replayed=3720).items()):
            raise ValueError('Incomplete replay')
        available = replay['side_statuses'].get('available',0)
        if sum(replay['side_statuses'].values()) != 120 or replay['side_predictions_replayed'] != 3*available:
            raise ValueError('Incomplete side coverage')
        if list((root/'failures').glob('*.json')) or list((root/'terminal_failures').glob('*.json')):
            raise ValueError('Unexpected failure records: inspect before publication')
        calculated, seeds, audit = audit_evaluation(root/'evaluation', plan)
        aggregate.extend(calculated)
        seed_rows.extend(seeds)
        requested=json.loads((root/'protocol/requested-plan.json').read_text())
        registered=experiment/'configs'/('proteins-plan.json' if setting=='proteins_edge' else 'mutagenicity-plan.json')
        if requested != json.loads(registered.read_text()):
            raise ValueError('Requested protocol differs from registration')
        changed={key:dict(requested=requested.get(key),resolved=value) for key,value in plan.items()
                 if key not in requested or requested[key] != value}
        configs[setting] = dict(path=str(root/'protocol/plan.json'),sha256=digest(root/'protocol/plan.json'),config=plan,
                                requested_plan_values_match=True,
                                requested_plan_bytes_match=digest(root/'protocol/requested-plan.json')==digest(registered),
                                resolution_differences=changed)
        for source in plan['domains']:
            for seed in plan['seeds']:
                selected = json.loads((p4.source_dir(root,plan,source,seed)/'complete.json').read_text())
                models.append(dict(setting=setting,kind='source',**selected))
                if selected['epochs'] != 100:
                    raise ValueError('Source epoch count mismatch')
        for task in p4.tasks(plan):
            selected = json.loads((p4.score_dir(root,plan,task)/'complete.json').read_text())
            models.append(dict(setting=setting,kind='score',**selected))
            if selected['updates'] != 7200:
                raise ValueError('Score update count mismatch')
        for path in (root/'evaluation').rglob('*'):
            if path.is_file():
                copy_identical(path,experiment/'results'/setting/path.relative_to(root/'evaluation'))
        for path in (root/'seals').glob('*.json'):
            copy_identical(path,experiment/'results/final-seals'/setting/path.name)
        copy_identical(base/'logs'/f'{setting}.log',experiment/'results/final-logs'/f'{setting}.log')
        files = {**seals['INPUT']['files'], **seals['OBS']['files'], **seals['FINAL']['files']}
        entries.append(dict(setting=setting,archive_root=str(root),execution_commit=provenance['git_commit'],
                            origin=plan['setting']['action'],replay=replay,audit=audit,
                            seal_digests={k:v['digest'] for k,v in seals.items()},
                            seal_file_sha256={k:digest(root/f'seals/R5_{k}.json') for k in seals},
                            sealed_files=len(files),sealed_bytes=sum((root/path).stat().st_size for path in files)))
    output.write_csv(experiment/'results/aggregate.csv',aggregate)
    output.write_csv(experiment/'results/aggregate-seeds.csv',seed_rows)
    output.write_json(experiment/'results/model-completion.json',models)
    output.write_json(experiment/'configs/executed-configs.json',configs)
    output.write_json(experiment/'results/archive-index.json',entries)
    events = [json.loads(line) for line in (base/'launch-001-events.jsonl').read_text().splitlines()]
    if events[-1].get('event') != 'queue_finished' or events[-1].get('failed'):
        raise ValueError('Execution queue is not successfully closed')
    output.write_json(experiment/'results/evidence.json',dict(
        completed_utc=events[-1]['time_utc'],execution_commits=sorted(set(e['execution_commit'] for e in entries)),
        publication_source_sha256={name:digest(experiment/'src'/name) for name in ('archive.py','overview.py')},
        settings=entries,new_trajectories=240,reused_trajectories=120,total_trajectories=360,
        new_attempts=7200,reused_attempts=3600,total_attempts=10800,all_predictions_replayed=11160,
        analysis='post-seal audit only; no training or configuration changes',
        plan_value_differences='registered dataset plans resolved per setting; full execution configs retained'))
    copy_identical(base/'launch-001-events.jsonl',experiment/'results/final-events.jsonl')
    print(json.dumps(dict(settings=list(roots),trajectories=360,attempts=10800,audits='passed'),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root',type=Path,required=True)
    args=parser.parse_args()
    experiment=Path(__file__).resolve().parents[1]
    archive(args.archive_root,experiment)
    from .overview import overview
    overview(experiment)
