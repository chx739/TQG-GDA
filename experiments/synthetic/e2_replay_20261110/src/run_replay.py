"""Command-line entry for this experiment."""
import argparse
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback
import torch
from experiments.common.io import manifest, provenance, write_json
from experiments.synthetic.e2_full_source_20261110.src.execution import verify_archive
from experiments.synthetic.e2_replay_20261110.src.execution import read, replay_cell
from experiments.common.paths import repository_root

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    repo=repository_root();parent=args.parent.resolve();root=args.output.resolve()
    plan_path=repo/'experiments/synthetic/e2_replay_20261110/configs/replay-plan.json';plan=read(plan_path)
    integrity=verify_archive(parent,repo/'experiments/synthetic/e2_full_source_20261110/results/run-manifest.json')
    root.mkdir(parents=True,exist_ok=False);write_json(root/'parent-integrity.json',integrity)
    write_json(root/'provenance.json',provenance(repo));shutil.copy2(plan_path,root/'replay-plan.json')
    subprocess.run(['git','archive','--format=tar.gz','--output='+str(root/'execution-source.tar.gz'),'HEAD'],cwd=repo,check=True)
    with (root/'python-packages.json').open('x') as f:subprocess.run([__import__('sys').executable,'-m','pip','list','--format=json'],stdout=f,check=True)
    failures=[];rows=[];audits=[];started=time.perf_counter()
    def timeout(signum,frame):raise TimeoutError('predeclared replay cell budget exceeded')
    signal.signal(signal.SIGALRM,timeout)
    with (root/'run.log').open('x',buffering=1) as f:
        def log(message):
            line=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())+' '+message;print(line,flush=True);f.write(line+'\n')
        for dim in plan['dimensions']:
            for seed in plan['seeds']:
                signal.alarm(plan['resources']['cell_seconds'])
                try:
                    if time.perf_counter()-started>plan['resources']['total_seconds']:raise TimeoutError('total replay budget exceeded')
                    r,a=replay_cell(parent/f'd{dim}_seed{seed}',root/f'd{dim}_seed{seed}',dim,seed,plan,log)
                    rows.extend(r);audits.append(dict(dim=dim,seed=seed,**a))
                except Exception as exc:
                    failures.append(dict(dim=dim,seed=seed,error=repr(exc),traceback=traceback.format_exc()));log(f'CELL FAILED d={dim} seed={seed}: {exc}')
                finally:signal.alarm(0)
        # Re-read and hash all original artifacts after replay; none may have changed.
        after=verify_archive(parent,repo/'experiments/synthetic/e2_full_source_20261110/results/run-manifest.json')
        if after!=integrity:raise ValueError('parent integrity changed during replay')
        write_json(root/'metrics.json',rows);write_json(root/'audits.json',audits);write_json(root/'failures.json',failures)
        write_json(root/'execution.json',dict(seconds=time.perf_counter()-started,completed_cells=len(audits),states=len(rows),failures=len(failures),parent_unchanged=True))
        log(f'COMPLETE cells={len(audits)}/10 states={len(rows)} failures={len(failures)} seconds={time.perf_counter()-started:.1f}')
    write_json(root/'manifest.json',manifest(root))
    if failures:raise SystemExit(1)
