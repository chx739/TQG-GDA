"""Run the registered frozen-GNN R1b follow-up in a new immutable attempt."""
from experiments.common.paths import repository_root
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import torch

from experiments.common.synthetic.archive import manifest, provenance, save_tensor, write_json
from experiments.common.real_data.runtime import RunLog, seal
from experiments.common.real_data.next_round.common import NextLabelBroker, prepare_dependencies, verify_seal
from experiments.mutagenicity.r1b_20261111.evaluate import evaluate_all, replay
from experiments.mutagenicity.r1b_20261111.scales import run_scales
from experiments.common.real_data.next_round.steps import run_steps
from experiments.mutagenicity.r1b_20261111.warm import run_warm


def execute(repo, root, plan, r0_log):
    repo, root = Path(repo), Path(root)
    if repo.resolve() != repository_root():
        raise RuntimeError('Executing checkout must match the archived repository')
    registered = json.loads((repo / 'experiments/mutagenicity/r1b_20261111/next-round-plan.json').read_text())
    if plan != registered:
        raise RuntimeError('Execution settings must match the committed registered plan')
    if subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip():
        raise RuntimeError('Commit reviewed implementation and protocol before real label access')
    if '\nOK\n' not in Path(r0_log).read_text():
        raise RuntimeError('Passing R0 unittest log required')
    if os.environ.get('OMP_NUM_THREADS') != '1' or os.environ.get('MKL_NUM_THREADS') != '1':
        raise RuntimeError('Set OMP_NUM_THREADS=1 MKL_NUM_THREADS=1')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    root.mkdir(parents=True,exist_ok=False)
    for name in ('logs','protocol','evaluation','failures','training-labels'):
        (root / name).mkdir()
    shutil.copyfile(r0_log,root / 'logs/R0-tests.txt')
    for name in ('r1b-protocol.md','next-round-plan.json','r1b-pre-run-review.json'):
        shutil.copyfile(repo / 'experiments/mutagenicity/r1b_20261111' / name,root / 'protocol' / name)
    write_json(root / 'execution-plan.json',plan)
    write_json(root / 'provenance.json',provenance(repo))
    with (root / 'protocol/source.tar').open('xb') as stream:
        subprocess.run(['git','archive','--format=tar','HEAD'],cwd=repo,stdout=stream,check=True)
    packages = sorted({(d.metadata['Name'],d.version) for d in importlib.metadata.distributions() if d.metadata['Name']})
    with (root / 'protocol/environment.txt').open('x') as stream:
        stream.write('OMP_NUM_THREADS=1\nMKL_NUM_THREADS=1\nTorch deterministic algorithms=True\n')
        stream.write('\n'.join(f'{name}=={version}' for name,version in packages)+'\n')
    log = RunLog(root / 'logs/run.log',plan['resources']['max_total_seconds'],plan['resources']['max_rss_bytes'])
    started = time.monotonic()
    try:
        log('Verify original archive and copy immutable R1 dependencies; no label parsing')
        roles = prepare_dependencies(root,plan)
        inputs = seal(root,'INPUT',['deps','dependency-manifest.json','role-verification.json','execution-plan.json',
            'protocol','provenance.json','logs/R0-tests.txt'])
        log(f"INPUT sealed {inputs['digest']}")
        broker = NextLabelBroker(root)
        a_labels = {d:broker.read(roles['targets'][str(d)]['A'],'target_A','Warm-start fit and inner validation only',domain=d)
                    for d in plan['domains']}
        save_tensor(root / 'training-labels/target-A.pt',a_labels)
        log('Start 80 warm fits and 40 old-random step-zero corrections')
        warm = run_warm(root,plan,a_labels,log)
        del a_labels
        log('Start fixed-anchor six-scale observations; no labels')
        scales = run_scales(root,plan,log)
        source_labels = {}
        for seed in plan['seeds']:
            sr = json.loads((root / f'deps/roles/seed{seed}.json').read_text())
            source_labels[seed] = broker.read(sr['fit_ids'],'source_fit','Full779 CE candidate and acceptance only',seed=seed)
        save_tensor(root / 'training-labels/source-fit.pt',source_labels)
        log('Start empirical proposals, 20 learned score fits and paired one-step views')
        steps = run_steps(root,plan,source_labels,log)
        del source_labels
        log(f'Checkpoint replay: {replay(root,plan)}')
        write_json(root / 'phase-outcomes.json',dict(warm_total=len(warm),warm_failed=sum(r['status']!='success' for r in warm),
            random_correction_failed=sum(r['status']!='success' for r in json.loads((root / 'warm/random_step0_corrected/index.json').read_text())),
            scales_summary_rows=len(scales),scale_error_rows=sum(r['dependency_failed'] or r['observation_failures']>0 for r in scales),
            step_total=len(steps['steps']),step_failed=sum(r['status']!='success' for r in steps['steps']),
            step_view_failed=sum(v.get('status','success')!='success' for r in steps['steps'] for v in r.get('views',{}).values()),
            score_total=len(steps['scores']),score_failed=sum(r['status']!='success' for r in steps['scores'])))
        verify_seal(root,'INPUT')
        observations = seal(root,'OBS',['seals/INPUT.json','deps','warm','scales','steps','scores','training-labels','prediction-replay.json',
            'phase-outcomes.json','execution-plan.json','dependency-manifest.json','role-verification.json','protocol','provenance.json'])
        log(f"OBS sealed {observations['digest']}; frozen evaluation opens")
        evaluate_all(root,plan,broker)
        outcomes = json.loads((root / 'phase-outcomes.json').read_text())
        write_json(root / 'completion.json',dict(status='completed' if not any(outcomes[k] for k in ('warm_failed','random_correction_failed','step_failed','score_failed','scale_error_rows','step_view_failed')) else 'completed_with_missing',
            elapsed_seconds=time.monotonic()-started,final_values_parsed=False,GNN_retrained=False,
            full_adaptation_run=False,**outcomes))
        log('Evaluation complete; OBS unchanged, final labels closed')
        log.close()
        seal(root,'FINAL',['evaluation','completion.json','seals/INPUT.json','seals/OBS.json','logs','phase-outcomes.json'])
        write_json(root / 'manifest.json',manifest(root))
    except BaseException as exc:
        write_json(root / 'interrupted.json',dict(error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),
            elapsed_seconds=time.monotonic()-started,note='Preserve this attempt; repairs require a new attempt'))
        log.close()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=repository_root())
    parser.add_argument('--root',type=Path)
    parser.add_argument('--r0-log',type=Path,required=True)
    args = parser.parse_args()
    plan = json.loads((args.repo / 'experiments/mutagenicity/r1b_20261111/next-round-plan.json').read_text())
    execute(args.repo,args.root or plan['output_root'],plan,args.r0_log)


if __name__ == '__main__':
    main()
