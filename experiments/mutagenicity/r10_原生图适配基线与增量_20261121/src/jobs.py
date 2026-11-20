"""Independent checkpointed model jobs and resumable bounded supervision."""
import contextlib
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

import torch

from experiments.common.io import digest, write_json
from . import data, training


def now():
    return datetime.now(timezone.utc).isoformat()


def grid(plan):
    for seed in plan['seeds']:
        yield f'source-only-{seed}', seed, None
        for _, target in plan['directions']:
            yield f'grade-M{target}-{seed}', seed, target


def target_key(seed, target):
    return f'mutagenicity_edge-M0_to_M{target}-seed-{seed}'


@contextlib.contextmanager
def lock(path):
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def read_done(root, name, *, verify=True):
    base = root/'runs'/name
    record = json.loads((base/'DONE.json').read_bytes())
    if record['job'] != name or not record['attempt'].startswith('attempt-'):
        raise ValueError('Invalid job completion identity')
    attempt = base/record['attempt']
    if verify:
        for relative, sha in record['files'].items():
            if digest(attempt/relative) != sha:
                raise PermissionError(f'Changed completed R10 job artifact: {name}/{relative}')
    return attempt, record


def model_job(root, name):
    roles, config = data.read_inputs(root)
    plan = config['plan']
    indexed = {key:(seed, target) for key, seed, target in grid(plan)}
    if name not in indexed:
        raise ValueError(f'Unregistered R10 model: {name}')
    seed, target = indexed[name]
    base = root/'runs'/name
    base.mkdir(parents=True, exist_ok=True)
    with lock(base/'job.lock'):
        if (base/'DONE.json').exists():
            read_done(root, name)
            return
        attempt = base/f'attempt-{1+len(list(base.glob("attempt-*"))):03d}'
        attempt.mkdir()
        record = dict(job=name, seed=seed, target=target, attempt=attempt.name,
                      pid=os.getpid(), ppid=os.getppid(), hostname=socket.gethostname(), started_utc=now(),
                      command=[sys.executable, *sys.orig_argv[1:]], input_sha256=digest(root/'INPUT.json'))
        write_json(attempt/'process.json', record)
        options = plan['execution']
        torch.set_num_threads(options['torch_threads_per_worker'])
        torch.set_num_interop_threads(options['interop_threads'])
        torch.set_default_dtype(torch.float32)
        torch.use_deterministic_algorithms(True)
        try:
            files = ['data/S.pt', f'data/S-labels-{seed}.pt']
            if target is not None:
                files.append(f'data/A-M{target}.pt')
            data.read_inputs(root, requested=files)
            y = torch.load(root/f'data/S-labels-{seed}.pt', weights_only=True, map_location='cpu')
            s_ids = roles['source_fit'][str(seed)]
            source = training.dataset(torch.load(root/'data/S.pt', weights_only=True, map_location='cpu'), s_ids, y)
            A = None
            a_ids = []
            if target is not None:
                a_ids = roles['tasks'][target_key(seed, target)]['a_ids']
                A = training.dataset(torch.load(root/f'data/A-M{target}.pt', weights_only=True, map_location='cpu'), a_ids)
            with (attempt/'epochs.jsonl').open('x', buffering=1) as progress:
                def log(row):
                    progress.write(json.dumps(row, allow_nan=False)+'\n')
                    if row['event'] != 'epoch' or row['epoch'] in (1, 200) or row['epoch'] % 10 == 0:
                        print(name, json.dumps(row, allow_nan=False), flush=True)

                model, initial, history = training.fit_model(source, A, s_ids, a_ids, plan, seed,
                                                            roles['node_features'], roles['classes'], log)
            targets = [t for _, t in plan['directions']] if target is None else [target]
            max_rss = max(row['peak_process_rss_mib'] for row in history)
            elapsed = history[-1]['elapsed_seconds']
            summary = dict(**record, updates=sum(row['updates'] for row in history), epochs=len(history),
                           initial_parameter_sha256=initial, final_parameter_sha256=training.state_hash(model),
                           peak_process_rss_mib=max_rss, elapsed_training_seconds=elapsed,
                           first_epoch=history[0], last_epoch=history[-1], source_graphs=len(s_ids), adapt_graphs=len(a_ids),
                           source_role_hash=digest(root/f'data/S-labels-{seed}.pt'),
                           torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                           device=options['device'], prediction_targets=targets)
            torch.save({name:value.cpu() for name,value in model.state_dict().items()}, attempt/'model.pt')
            # H topology is only deserialized after the final optimization step.
            for t in targets:
                relative = f'data/H-M{t}.pt'
                data.read_inputs(root, requested=[relative])
                h_ids = roles['tasks'][target_key(seed, t)]['h_ids']
                result = training.infer(model, torch.load(root/relative, weights_only=True, map_location='cpu'), h_ids, plan)
                torch.save(result, attempt/f'prediction-M{t}.pt')
            write_json(attempt/'summary.json', summary)
            sealed = ('process.json','epochs.jsonl','model.pt','summary.json',
                      *(f'prediction-M{t}.pt' for t in targets))
            done = dict(job=name, seed=seed, target=target, attempt=attempt.name, completed_utc=now(),
                        files={rel: digest(attempt/rel) for rel in sealed})
            temp = base/f'.DONE-{attempt.name}.json'
            write_json(temp, done)
            temp.rename(base/'DONE.json')
            print(f'Completed {name}: {elapsed:.1f} s, {max_rss:.0f} MiB peak RSS', flush=True)
        except BaseException:
            write_json(attempt/'failure.json', dict(failed_utc=now(), trace=traceback.format_exc()))
            raise


def supervise(root, module, *, first_seed_only=False):
    _, config = data.read_inputs(root)
    plan = config['plan']
    root.joinpath('runs').mkdir(exist_ok=True)
    with lock(root/'supervisor.lock'):
        logs = root/'supervision'
        logs.mkdir(exist_ok=True)
        tag = f'launch-{len(list(logs.glob("launch-*.jsonl")))+1:03d}'
        jobs = list(grid(plan))
        first = plan['seeds'][0]
        with (logs/f'{tag}.jsonl').open('x', buffering=1) as ledger:
            def log(event, **values):
                row = dict(time_utc=now(), supervisor_pid=os.getpid(), event=event, **values)
                ledger.write(json.dumps(row)+'\n')
                print(json.dumps(row), flush=True)

            log('supervisor_started', hostname=socket.gethostname(), first_seed_only=first_seed_only)
            def execute(selection, limit):
                pending, finished = {}, {}
                names = []
                for name, _, _ in selection:
                    if (root/'runs'/name/'DONE.json').is_file():
                        read_done(root, name)
                        log('reused', job=name)
                    else:
                        names.append(name)
                env = os.environ.copy()
                threads = str(plan['execution']['torch_threads_per_worker'])
                env.update(OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads, OPENBLAS_NUM_THREADS='1')
                while names or pending:
                    while names and len(pending) < limit:
                        name = names.pop(0)
                        folder = logs/name
                        folder.mkdir(exist_ok=True)
                        path = folder/f'{tag}.log'
                        argv = [sys.executable, '-u', '-m', module, 'worker', '--root', str(root), '--job', name]
                        with path.open('x') as stream:
                            child = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, env=env,
                                                     stdin=subprocess.DEVNULL)
                        log('worker_started', job=name, pid=child.pid, log=str(path), command=argv)
                        pending[child.pid] = (name, child)
                    for pid, (name, child) in list(pending.items()):
                        status = child.poll()
                        if status is not None:
                            pending.pop(pid)
                            finished[name] = status
                            log('worker_exited', job=name, pid=pid, returncode=status)
                    if pending:
                        time.sleep(2)
                failures = [name for name, status in finished.items() if status]
                if failures:
                    raise RuntimeError(f'Incomplete R10 jobs after this launch: {failures}')

            execute([row for row in jobs if row[1] == first], limit=3)
            measured = {}
            for name, _, _ in [row for row in jobs if row[1] == first]:
                attempt, _ = read_done(root,name)
                details = json.loads((attempt/'summary.json').read_bytes())
                measured[name] = {key: details[key] for key in ('updates','elapsed_training_seconds',
                                                                 'peak_process_rss_mib','source_graphs','adapt_graphs')}
            log('first_seed_verified', measurements=measured)
            if not first_seed_only:
                execute([row for row in jobs if row[1] != first], plan['execution']['max_workers_after_first_seed'])
                log('all_models_verified', models=len(jobs))
