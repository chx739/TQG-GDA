"""Strict read-only acceptance for version-locked training and resumptions."""
import json
import re
import subprocess
from pathlib import Path

import torch

from experiments.common.io import digest
from . import data, jobs, training

MODULE = 'experiments.mutagenicity.r10_原生图适配基线与增量_20261121.scripts.run'
ATTEMPT = re.compile(r'attempt-[0-9]{3,}')


def validate_code(config,repo):
    for rel,expected in config['code_hashes'].items():
        path = repo/rel
        committed = subprocess.check_output(['git','show',f"{config['code_commit']}:{rel}"],cwd=repo)
        if data.inputs.sha(committed) != expected or digest(path) != expected:
            raise PermissionError(f'R10 training code must match the prepared commit before resuming: {rel}')


def workers(root, selected=None):
    launched, exited, witnesses = {},{},[]
    paths = sorted((root/'supervision').glob('launch-*.jsonl'))
    if selected is not None:
        paths = [path for path in paths if str(path.relative_to(root)) in selected]
        if not paths or [str(path.relative_to(root)) for path in paths] != sorted(selected):
            raise PermissionError('Missing R10 supervision ledger')
    for path in paths:
        witnesses.append(str(path.relative_to(root)))
        for line in path.read_text().splitlines():
            row = json.loads(line)
            event = row['event']
            if event not in ('worker_started','worker_exited'):
                continue
            identity = row['job'], row['pid'], row['supervisor_pid']
            records = launched if event=='worker_started' else exited
            if identity in records:
                raise ValueError(f'Duplicate R10 launch/process identity: {identity}')
            records[identity] = row
    return launched,exited,witnesses


def job_files(target,plan):
    predictions = [t for _,t in plan['directions']] if target is None else [target]
    members = {'process.json','epochs.jsonl','model.pt','summary.json',*(f'prediction-M{t}.pt' for t in predictions)}
    return predictions,members


def completion_record(root,name,seed,target,plan):
    base = root/'runs'/name
    done = json.loads((base/'DONE.json').read_bytes())
    targets, members = job_files(target,plan)
    if (done.get('job')!=name or done.get('seed')!=seed or done.get('target')!=target
            or ATTEMPT.fullmatch(done.get('attempt','')) is None
            or set(done.get('files',{})) != members):
        raise PermissionError(f'Completed model does not match the exact registered identity and member set: {name}')
    attempt = base/done['attempt']
    for relative, expected in done['files'].items():
        if digest(attempt/relative)!=expected:
            raise PermissionError(f'Completed model member changed: {name}/{relative}')
    return attempt,done,targets


def validate_jobs(root,repo,*,seeds=None,allow_missing=False,sealed_ledgers=None):
    roles,config = data.read_inputs(root)
    plan = config['plan']
    execution_root = root
    if sealed_ledgers is None:
        validate_code(config,repo)
        starts,exits,ledgers = workers(root)
    else:
        # Recheck the archived execution, even if the checkout or archive has moved.
        args = config['command']
        recorded_root = (args[args.index('--root') + 1] if '--root' in args else
                         next(arg.split('=',1)[1] for arg in args if arg.startswith('--root=')))
        execution_root = (Path(config['cwd']) / recorded_root).resolve()
        starts,exits,ledgers = workers(root,sealed_ledgers)
    expected_input = digest(root/'INPUT.json')
    result,entries,missing = [],[],[]
    for name,seed,target in jobs.grid(plan):
        if seeds is not None and seed not in seeds:
            continue
        if not (root/'runs'/name/'DONE.json').is_file():
            missing.append(name)
            continue
        attempt,done,targets = completion_record(root,name,seed,target,plan)
        process = json.loads((attempt/'process.json').read_bytes())
        summary = json.loads((attempt/'summary.json').read_bytes())
        identity = name,process['pid'],process['ppid']
        launcher = starts.get(identity,{})
        exit_code = exits.get(identity,{}).get('returncode')
        command = process['command']
        expected_command = ['-u','-m',MODULE,'worker','--root',str(execution_root),'--job',name]
        if (summary.get('pid')!=process['pid'] or summary.get('ppid')!=process['ppid']
                or process['input_sha256']!=expected_input or summary['input_sha256']!=expected_input
                or (process['job'],process['seed'],process['target'],process['attempt'])!=(name,seed,target,done['attempt'])
                or command[1:]!=expected_command or command!=launcher.get('command') or exit_code!=0):
            raise PermissionError(f'R10 worker command, registered role or supervisor exit mismatch: {name}')
        opts=plan['execution']
        if (summary['device']!=opts['device'] or summary['torch_threads']!=opts['torch_threads_per_worker']
                or summary['torch_interop_threads']!=opts['interop_threads'] or summary['prediction_targets']!=targets
                or summary['source_role_hash']!=digest(root/f'data/S-labels-{seed}.pt')):
            raise ValueError(f'Execution environment or source role differs: {name}')
        epochs=plan['model']['epochs']
        source_n=len(roles['source_fit'][str(seed)])
        adapt_n=len(roles['tasks'][jobs.target_key(seed,target)]['a_ids']) if target is not None else 0
        trace=[json.loads(line) for line in (attempt/'epochs.jsonl').read_text().splitlines()]
        epoch_rows=[row for row in trace if row['event']=='epoch']
        if (not trace or trace[0]['event']!='fit_started'
                or trace[0]['initial_parameter_sha256']!=summary['initial_parameter_sha256']
                or len(epoch_rows)!=epochs or summary['updates']!=epochs or summary['epochs']!=epochs
                or epoch_rows[0]!=summary['first_epoch'] or epoch_rows[-1]!=summary['last_epoch']):
            raise ValueError(f'Training trace or step count differs: {name}')
        for epoch,row in enumerate(epoch_rows,start=1):
            if (row['epoch']!=epoch or row['updates']!=1 or row['source_graphs']!=source_n
                    or row['target_adapt_graphs']!=adapt_n or row['alpha']!=training.alpha_at(epoch-1,epochs)
                    or (target is None)!=(row['adapt_order_sha256'] is None)
                    or (target is None)!=(row['domain_ce'] is None)):
                raise ValueError(f'Full-batch epoch identity, coverage or alpha differs: {name}/{epoch}')
        for direction in targets:
            item=torch.load(attempt/f'prediction-M{direction}.pt',map_location='cpu',weights_only=True)
            ids=roles['tasks'][jobs.target_key(seed,direction)]['h_ids']
            if (set(item)!= {'ids','logits','decisions'} or item['ids'].tolist()!=ids
                    or item['logits'].shape!=(len(ids),roles['classes']) or item['logits'].dtype!=torch.float32
                    or not bool(torch.isfinite(item['logits']).all())
                    or not torch.equal(item['decisions'],item['logits'].argmax(1))):
                raise ValueError(f'Unscored H artifact fails identity or probability checks: {name}/M{direction}')
        result.append((name,seed,target,attempt,summary))
        entries.append(dict(job=name,seed=seed,target=target,attempt=done['attempt'],pid=process['pid'],
            supervisor_pid=process['ppid'],command=command,returncode=exit_code,
            done_sha256=digest(attempt.parent/'DONE.json'), initial_parameter_sha256=summary['initial_parameter_sha256'],
            updates=epochs,elapsed_training_seconds=summary['elapsed_training_seconds'],
            peak_process_rss_mib=summary['peak_process_rss_mib']))
    if missing and not allow_missing:
        raise PermissionError(f'R10 selected model set incomplete: {missing}')
    for seed in (plan['seeds'] if seeds is None else seeds):
        matched=[summary for _,s,_,_,summary in result if s==seed]
        if (len({x['initial_parameter_sha256'] for x in matched})>1
                or len({x['source_role_hash'] for x in matched})>1):
            raise ValueError(f'Models failed shared initialization or source-role validation: {seed}')
    return result,dict(training_code_commit=config['code_commit'], input_sha256=expected_input,
                        verified_jobs=entries,missing_jobs=missing,supervision_ledgers=ledgers)
