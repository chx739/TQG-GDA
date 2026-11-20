"""Run the R14 node-type gradient pilot with separate FIT, OBS and FINAL seals."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import traceback

import torch

from experiments.common import io
from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import evaluate, paired, r5_inputs, seals
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src import composition, inputs
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs as r11_inputs
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import prior_run as storage, run as r11_run
from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src import historical, normalization, observation
from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src import scoring as r13_scoring
from tsba import encoder, pretrain
from ..src import activations, experiment, recovery, scoring


def reserve_if_approved(plan, cfg, source):
    if source is None:
        return [plan['phase1_seed']], None
    original_seed = [plan['phase1_seed']]
    fit_sha = io.digest(source/'FIT.json')
    obs_sha = io.digest(source/'OBS.json')
    final_sha = io.digest(source/'FINAL.json')
    fitted = experiment.frozen_fit(source, plan, original_seed, fit_sha)
    executed = json.loads(fitted['executed-config.json'])
    if (executed['r14']!=cfg or executed['historical_role_plan']!=plan or executed['seeds']!=original_seed
            or executed['pilot_final_sha256'] is not None):
        raise PermissionError('Reserve seeds require a successful pilot of this registration')
    storage.verified_members(source, 'OBS', obs_sha, experiment.obs_members(plan, original_seed), {'fit_sha256':fit_sha})
    final = storage.verified_members(source, 'FINAL', final_sha, scoring.FINAL_MEMBERS, {'obs_sha256':obs_sha})
    decision = json.loads(final['gate.json'])
    if not decision['pilot'] or not decision['decision']['extend']:
        raise PermissionError('The R14 pilot did not authorize extension')
    return plan['reserve_seeds'], final_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--output', type=Path)
    mode.add_argument('--preflight', action='store_true')
    parser.add_argument('--extend-from', type=Path)
    args = parser.parse_args()
    if args.preflight and args.extend_from:
        parser.error('--extend-from requires a new --output')

    here = Path(__file__).resolve().parents[1]
    repo = Path(__file__).resolve().parents[4]
    cfg = json.loads(args.config.read_bytes())
    control = repo/cfg['r13_config']
    plan = json.loads(control.read_bytes())
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    seeds, parent = reserve_if_approved(plan,cfg,args.extend_from)

    def git(*argv):
        return subprocess.check_output(['git',*argv],cwd=repo)

    ref = cfg['registration_commit']
    protocol = (here/cfg['protocol']).relative_to(repo)
    for path in (control.relative_to(repo),protocol):
        if inputs.sha(git('show', f'{ref}:{path}')) != io.digest(repo/path):
            raise PermissionError(f'Registered R14 input changed: {path}')
    if args.preflight:
        result = experiment.fit(None,repo,plan,cfg,seeds,inspect=True)
        print(json.dumps(dict(graph_only=True, sources=result),indent=2),flush=True)
        return

    modules = (activations, experiment, recovery, scoring, composition, inputs, io, r11_inputs, storage, r11_run,
               historical, normalization, observation, r13_scoring, encoder, pretrain, evaluate, paired, r5_inputs, seals)
    files = [args.config.resolve(), control, repo/protocol, Path(__file__),
             *[Path(m.__file__) for m in modules]]
    revision = git('rev-parse','HEAD').decode().strip()
    hashes = {str(p.resolve().relative_to(repo)): io.digest(p) for p in files}
    for file, expected in hashes.items():
        if inputs.sha(git('show', f'{revision}:{file}')) != expected:
            raise PermissionError(f'Commit and review this R14 input before execution: {file}')
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    io.write_json(output/'executed-config.json',dict(
        r14=cfg, historical_role_plan=plan, seeds=seeds, pilot_final_sha256=parent,
        code_commit=revision, code_sha256=hashes, command=[sys.executable,*sys.orig_argv[1:]],
        started_utc=datetime.now(timezone.utc).isoformat(), cwd=str(Path.cwd()), torch_threads=torch.get_num_threads(),
        torch_interop_threads=torch.get_num_interop_threads(), device='cpu',
        versions={name:importlib.metadata.version(name) for name in ('torch','torch-geometric','numpy','scipy')}))
    phase = 'FIT'
    try:
        fit = experiment.fit(output,repo,plan,cfg,seeds)
        print('R14 FIT sealed',fit,flush=True)
        phase = 'OBS'
        obs = experiment.predict(output,repo,plan,cfg,seeds,fit)
        print('R14 OBS sealed',obs,flush=True)
        phase = 'FINAL'
        final, decision = scoring.score(output,repo,plan,cfg,seeds,fit,obs,pilot=parent is None)
        print('R14 FINAL sealed',final,decision,flush=True)
    except Exception:
        io.write_json(output/'execution-failure.json',dict(phase=phase, trace=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
