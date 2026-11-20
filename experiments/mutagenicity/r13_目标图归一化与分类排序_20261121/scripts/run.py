"""R13 source/target BN calibration, sealed observation, and matched H scoring."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import traceback

import torch

from experiments.common.io import digest, write_json
from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import evaluate, paired, r5_inputs, seals
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src import inputs as r8_inputs
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs as r11_inputs
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import prior_run, run as r11_run
from tsba import encoder, pretrain
from ..src import historical, normalization, observation, scoring


def execution_branch(plan, from_pilot):
    if from_pilot is None:
        return [plan['phase1_seed']], ('node','graph'), None
    root = from_pilot.resolve()
    obs_sha = digest(root/'OBS.json')
    expected = ['OBS.json', 'endpoints.csv', 'endpoint-summary.csv', 'paired-blocks.csv', 'paired-summary.csv',
                'label-access.json', 'cohort-audit.json', 'gate.json']
    files = prior_run.verified_members(root, 'FINAL', digest(root/'FINAL.json'), expected, {'obs_sha256': obs_sha})
    execution = json.loads((root/'executed-config.json').read_bytes())
    if (execution['plan'] != plan or execution['seeds'] != [plan['phase1_seed']]
            or execution['weights'] != ['node','graph'] or execution['pilot_final_sha256'] is not None):
        raise PermissionError('Extension requires original R13 pilot from the same registration')
    scoring._verified(root, obs_sha, plan, execution['seeds'])
    record = json.loads(files['gate.json'])
    if not record['pilot']:
        raise ValueError('Only the registered pilot can trigger reserve-seed execution')
    branch = record['decision']['branch']
    if branch not in ('node','graph'):
        raise ValueError('Pilot gate did not authorize reserve seeds')
    return plan['reserve_seeds'], ('node','graph') if branch == 'graph' else ('node',), digest(root/'FINAL.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--output', type=Path)
    action.add_argument('--preflight', action='store_true')
    parser.add_argument('--extend-from', type=Path)
    args = parser.parse_args()
    if args.preflight and args.extend_from is not None:
        parser.error('--extend-from requires --output')
    repo = Path(__file__).resolve().parents[4]
    here = Path(__file__).resolve().parents[1]
    config = args.config.resolve()
    plan = json.loads(config.read_bytes())
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    protocol = (here / plan['protocol']).relative_to(repo)
    seeds, weights, pilot_sha = execution_branch(plan, args.extend_from)

    def git(*argv):
        return subprocess.check_output(['git', *argv], cwd=repo)

    revision = git('rev-parse','HEAD').decode().strip()
    checked = [config, repo/protocol, Path(__file__)]
    modules = (encoder, pretrain, historical, normalization, observation, scoring, evaluate, paired, r5_inputs,
               seals, r8_inputs, r11_inputs, prior_run, r11_run)
    checked.extend(Path(module.__file__) for module in modules)
    hashes = {str(path.resolve().relative_to(repo)): digest(path) for path in checked}
    if r8_inputs.sha(git('show', f"{plan['registration_commit']}:{protocol}")) != hashes[str(protocol)]:
        raise PermissionError('R13 protocol differs from the registration')
    if args.preflight:
        report = observation.freeze(None,repo,plan,seeds,weights,check_only=True)
        print(json.dumps(dict(preflight=True, total=len(report), rows=report)))
        return
    for name, sha in hashes.items():
        if r8_inputs.sha(git('show', f'{revision}:{name}')) != sha:
            raise PermissionError(f'Commit and review this R13 input before execution: {name}')
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'executed-config.json', dict(plan=plan, seeds=seeds, weights=list(weights),
        code_commit=revision, code_sha256=hashes, pilot_final_sha256=pilot_sha,
        command=[sys.executable,*sys.orig_argv[1:]], cwd=str(Path.cwd()),
        versions={p:importlib.metadata.version(p) for p in ('torch','torch-geometric','numpy','scipy')},
        started_utc=datetime.now(timezone.utc).isoformat(), interop_threads=torch.get_num_interop_threads(),
        torch_threads=torch.get_num_threads(), device='cpu'))
    phase = 'observation'
    try:
        obs_sha = observation.freeze(output,repo,plan,seeds,weights)
        print('R13 OBS sealed',obs_sha,flush=True)
        phase = 'score'
        checksum, decision = scoring.score(output,repo,plan,seeds,weights,obs_sha,pilot=pilot_sha is None)
        print('R13 FINAL sealed',checksum,decision,flush=True)
    except Exception:
        write_json(output/'execution-failure.json', dict(phase=phase, traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
