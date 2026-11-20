"""Run R16 source stacks, matched residual fits, and sealed H scoring."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import traceback

from experiments.common import io
from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import evaluate, paired, r5_inputs, seals
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src import composition, pilot
from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import wl
from experiments.mutagenicity.r10_原生图适配基线与增量_20261121.src import structural, structural_observation
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs as r11
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import prior_run as storage, run as r11_run
from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src import historical, observation
from experiments.mutagenicity.r14_节点类型与分类更新_20261121.src import experiment as r14
from ..src import inputs, models, experiment, scoring


def select_seeds(cfg, repo, from_pilot):
    first = [cfg['phase1_seed']]
    if from_pilot is None:
        return first,None
    history = inputs.Historical(cfg,repo)
    fit,obs,final = [io.digest(from_pilot/f'{stage}.json') for stage in ('FIT','OBS','FINAL')]
    fitted = experiment.verified_fit(from_pilot,history.plan,cfg,first,fit)
    execution = json.loads(fitted['executed-config.json'])
    if execution['seeds']!=first or execution['plan']!=cfg or execution['pilot_final_sha256'] is not None:
        raise PermissionError('The reserve phase requires the original registered R16 pilot')
    storage.verified_members(from_pilot,'OBS',obs,experiment.obs_members(history.plan,first),{'fit_sha256':fit})
    result = storage.verified_members(from_pilot,'FINAL',final,scoring.FINAL_MEMBERS,{'obs_sha256':obs})
    if not json.loads(result['gate.json'])['decision']['extend']:
        raise PermissionError('R16 pilot did not authorize the reserve seeds')
    return cfg['reserve_seeds'],final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight',action='store_true')
    mode.add_argument('--output',type=Path)
    parser.add_argument('--extend-from',type=Path)
    args = parser.parse_args()
    if args.preflight and args.extend_from is not None:
        parser.error('--extend-from needs an independent output')
    repo = Path(__file__).resolve().parents[4]
    here = Path(__file__).resolve().parents[1]
    cfg = json.loads(args.config.read_bytes())
    protocol = (here/cfg['protocol']).relative_to(repo)
    seeds,parent = select_seeds(cfg,repo,args.extend_from)

    def git(*params):
        return subprocess.check_output(['git',*params],cwd=repo)

    revision = git('rev-parse','HEAD').decode().strip()
    if r11.sha(git('show',f"{cfg['registration_commit']}:{protocol}"))!=io.digest(repo/protocol):
        raise PermissionError('R16 protocol differs from preregistration')
    if args.preflight:
        print(json.dumps(experiment.fit(None,repo,cfg,seeds,preflight=True),indent=2),flush=True)
        return
    deps = (inputs,models,experiment,scoring,io,evaluate,paired,r5_inputs,seals,composition,pilot,wl,structural,
            structural_observation,r11,storage,r11_run,historical,observation,r14)
    paths = [args.config.resolve(),repo/protocol,Path(__file__),*[Path(m.__file__) for m in deps]]
    hashes = {str(p.resolve().relative_to(repo)):io.digest(p) for p in paths}
    if any(r11.sha(git('show',f'{revision}:{name}'))!=sha for name,sha in hashes.items()):
        raise PermissionError('Commit the R16 implementation and inherited dependencies before execution')
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    io.write_json(output/'executed-config.json',dict(plan=cfg,seeds=seeds,pilot_final_sha256=parent,
                  code_commit=revision,sha256=hashes,command=[sys.executable,*sys.orig_argv[1:]],cwd=str(Path.cwd()),
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  versions={name:importlib.metadata.version(name) for name in ('numpy','scipy','torch','torch-geometric')}))
    stage = 'FIT'
    try:
        fit = experiment.fit(output,repo,cfg,seeds)
        print('R16 FIT sealed',fit,flush=True)
        stage = 'OBS'
        obs = experiment.predict(output,repo,cfg,seeds,fit)
        print('R16 OBS sealed',obs,flush=True)
        stage = 'FINAL'
        final,decision = scoring.score(output,repo,cfg,seeds,fit,obs,pilot=parent is None)
        print('R16 FINAL sealed',final,decision,flush=True)
    except Exception:
        io.write_json(output/'execution-failure.json',dict(stage=stage,traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
