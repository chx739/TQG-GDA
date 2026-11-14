"""Command-line entry for this experiment."""
import argparse
from pathlib import Path
import shutil
import signal
import time
import traceback
import torch
from experiments.common.synthetic.archive import manifest, provenance, write_json
from experiments.common.synthetic.run_controls import evaluate, run_cell, verify_parent
from experiments.common.paths import repository_root

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    repo=repository_root()
    parent=args.parent.resolve()
    audit=verify_parent(parent,repo)
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    write_json(root/'parent-integrity.json',audit)
    write_json(root/'provenance.json',provenance(repo))
    shutil.copy2(repo/'experiments/synthetic/e2_controls_20261109/controls-plan.json',root/'controls-plan.json')
    shutil.copy2(repo/'experiments/validation-plan.md',root/'validation-plan.md')
    failures=[];completed=[];started=time.perf_counter()
    def timeout(signum,frame):raise TimeoutError('predeclared 1200 second cell budget exceeded')
    signal.signal(signal.SIGALRM,timeout)
    with (root/'run.log').open('x',buffering=1) as logfile:
        def log(message):
            line=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())+' '+message
            print(line,flush=True);logfile.write(line+'\n')
        for dim in (2,16):
            for seed in range(5):
                signal.alarm(1200)
                try:
                    if time.perf_counter()-started>14400:raise TimeoutError('global budget exceeded')
                    run_cell(parent/f'd{dim}_seed{seed}',root/f'd{dim}_seed{seed}',dim,seed,log)
                    completed.append((dim,seed))
                except Exception as exc:
                    failures.append(dict(dim=dim,seed=seed,error=repr(exc),traceback=traceback.format_exc()))
                    log(f'CELL FAILED d={dim} seed={seed}: {exc}')
                finally:signal.alarm(0)
        write_json(root/'failures.json',failures)
        evaluate(root,completed)
        log(f'COMPLETE cells={len(completed)}/10 failures={len(failures)} seconds={time.perf_counter()-started:.1f}')
    write_json(root/'manifest.json',manifest(root))
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
