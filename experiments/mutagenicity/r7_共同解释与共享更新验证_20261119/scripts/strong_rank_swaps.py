"""Freeze, audit and score the R7 strong-base ranking swaps."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import traceback

from experiments.common import io
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src import inputs as r8_inputs, pilot
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run, roc_score, roc_transfer
from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src import historical, observation
from ..src import head_crossover, head_crossover_scoring, r5_inputs, seals, strong_rank_swaps as swaps

CODE = (io,r8_inputs,pilot,inputs,prior_run,roc_score,roc_transfer,historical,observation,
        head_crossover,head_crossover_scoring,r5_inputs,seals,swaps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--phase',required=True,choices=('freeze','verify','score'))
    parser.add_argument('--obs-sha256')
    parser.add_argument('--blind',type=Path)
    args = parser.parse_args()
    if (args.phase!='freeze')!=(args.obs_sha256 is not None) or (args.phase=='verify')!=(args.blind is not None):
        parser.error('Verify needs --blind and --obs-sha256; score needs --obs-sha256')
    here = Path(__file__).resolve().parents[1]
    repo = here.parents[2]
    config = args.config.resolve()
    plan = json.loads(config.read_bytes())
    parent = here/plan['crossover_config']
    cfg = json.loads(parent.read_bytes())
    protocol = here/plan['protocol']

    def git(*argv):
        return subprocess.check_output(['git',*argv],cwd=repo)

    if inputs.sha(git('show',f"{plan['registration_commit']}:{protocol.relative_to(repo)}"))!=io.digest(protocol):
        raise PermissionError('The registered strong-rank swaps protocol changed')
    revision = git('rev-parse','HEAD').decode().strip()
    files = [config,parent,protocol,Path(__file__),*(Path(m.__file__) for m in CODE)]
    hashes = {str(p.resolve().relative_to(repo)):io.digest(p) for p in files}
    if any(inputs.sha(git('show',f'{revision}:{p}'))!=h for p,h in hashes.items()):
        raise PermissionError('Actual strong-rank swap files differ from Git')
    out = args.output.resolve()
    execution = dict(code_commit=revision,cwd=str(Path.cwd()),command=[sys.executable,*sys.orig_argv[1:]],
                     started_utc=datetime.now(timezone.utc).isoformat())
    if args.phase!='freeze':
        record = json.loads((out/'executed-config.json').read_bytes())
        if record['code_sha256']!=hashes or record['plan']!=plan or record['crossover_plan']!=cfg:
            raise PermissionError('Current execution differs from the frozen provenance')
    try:
        if args.phase=='verify':
            record = swaps.blind_check(out,args.obs_sha256,args.blind,plan)
            print('R7 blind comparison',json.dumps(record,sort_keys=True),flush=True)
            return
        c = historical.CArchive(cfg)
        loaded = swaps.load_actions(repo,here,plan,cfg,c)
        if args.phase=='freeze':
            out.parent.mkdir(parents=True,exist_ok=True)
            if out.exists():
                raise FileExistsError(out)
            with TemporaryDirectory(prefix=f'.{out.name}-',dir=out.parent) as tmp:
                stage = Path(tmp)/'OBS'
                stage.mkdir()
                io.write_json(stage/'executed-config.json',dict(**execution,code_sha256=hashes,
                              plan=plan,crossover_plan=cfg,training_updates=0,
                              label_phases={'freeze':[],'verify':[],'score':['holdout']}))
                digest = swaps.freeze(stage,*loaded,cfg,plan)
                stage.rename(out)
            print('R7 strong-rank swaps OBS',digest,flush=True)
        else:
            if (out/'score').exists():
                raise FileExistsError('The OBS already has a published score')
            with TemporaryDirectory(prefix=f'.{out.name}-score-',dir=out.parent) as tmp:
                stage = Path(tmp)/'score'
                stage.mkdir()
                digest = swaps.score(out,stage,repo,c,loaded,cfg,plan,args.obs_sha256,
                                     dict(**execution,obs_sha256=args.obs_sha256))
                stage.rename(out/'score')
            print('R7 strong-rank swaps FINAL',digest,flush=True)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)


if __name__=='__main__':
    main()
