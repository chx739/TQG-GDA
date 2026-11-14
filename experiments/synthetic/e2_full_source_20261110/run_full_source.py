"""Command-line entry for this experiment."""
import argparse
from pathlib import Path
import shutil
import signal
import time
import traceback
import torch
from experiments.common.synthetic.archive import manifest, provenance, write_json
from experiments.common.synthetic.run_full_source import evaluate, run_cell, verify_archive
from experiments.common.paths import repository_root

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original',type=Path,required=True);parser.add_argument('--controls',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    repo=repository_root()
    original=args.original.resolve();controls=args.controls.resolve()
    integrity=dict(original=verify_archive(original,repo/'experiments/synthetic/e1_e2_20261109/run-manifest.json'),
                   controls=verify_archive(controls,repo/'experiments/synthetic/e2_controls_20261109/run-manifest.json'))
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    write_json(root/'parent-integrity.json',integrity);write_json(root/'provenance.json',provenance(repo))
    shutil.copy2(repo/'experiments/synthetic/e2_full_source_20261110/full-source-plan.json',root/'full-source-plan.json')
    failures=[];cells=[];start=time.perf_counter()
    def timeout(signum,frame):raise TimeoutError('predeclared 1800 second cell budget exceeded')
    signal.signal(signal.SIGALRM,timeout)
    with (root/'run.log').open('x',buffering=1) as logfile:
        def log(message):
            line=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())+' '+message
            print(line,flush=True);logfile.write(line+'\n')
        for dim in (2,16):
            for seed in range(5):
                signal.alarm(1800)
                try:
                    if time.perf_counter()-start>14400:raise TimeoutError('global budget exceeded')
                    name=f'd{dim}_seed{seed}'
                    run_cell(original/name,controls/name,root/name,dim,seed,log);cells.append((dim,seed))
                except Exception as exc:
                    failures.append(dict(dim=dim,seed=seed,error=repr(exc),traceback=traceback.format_exc()))
                    log(f'CELL FAILED d={dim} seed={seed}: {exc}')
                finally:signal.alarm(0)
        write_json(root/'failures.json',failures);evaluate(root,cells)
        log(f'COMPLETE cells={len(cells)}/10 failures={len(failures)} seconds={time.perf_counter()-start:.1f}')
    write_json(root/'manifest.json',manifest(root))
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
