"""Run the preregistered R13 input-only readout scale audit."""
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
from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import r5_inputs,seals
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs as r11,prior_run
from tsba import encoder,pretrain
from ..src import historical,observation,readout_scale


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--config',required=True,type=Path)
    cli.add_argument('--output',required=True,type=Path)
    args=cli.parse_args()
    here=Path(__file__).resolve().parents[1]
    repo=Path(__file__).resolve().parents[4]
    cfg=json.loads(args.config.read_bytes())
    archive=here/cfg['cohort_config']
    protocol=here/cfg['protocol']
    plan=json.loads(archive.read_bytes())
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    def git(*argv):
        return subprocess.check_output(['git',*argv],cwd=repo)

    head=git('rev-parse','HEAD').decode().strip()
    checked=[args.config.resolve(),Path(__file__),protocol,archive,repo/cfg['r9_checks']]
    checked.extend(Path(m.__file__) for m in (historical,observation,readout_scale,io,r5_inputs,seals,
                                           r11,prior_run,encoder,pretrain))
    hashes={str(p.resolve().relative_to(repo)):io.digest(p) for p in checked}
    for path in (archive,protocol,repo/cfg['r9_checks']):
        key=str(path.relative_to(repo))
        if r11.sha(git('show',f"{cfg['registration_commit']}:{key}"))!=hashes[key]:
            raise PermissionError(f'Registered audit member differs: {key}')
    if any(r11.sha(git('show',f'{head}:{f}'))!=sha for f,sha in hashes.items()):
        raise PermissionError('Commit the audit runner, inherited code and config before execution')
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    io.write_json(out/'executed-config.json',dict(code_commit=head,code_sha256=hashes,plan=cfg,historical_plan=plan,
                  command=[sys.executable,*sys.orig_argv[1:]],cwd=str(Path.cwd()),graph_roles=['V','Q','R'],
                  seed=plan['phase1_seed'],started_utc=datetime.now(timezone.utc).isoformat(),
                  versions={n:importlib.metadata.version(n) for n in ('torch','torch-geometric','numpy','scipy')}))
    try:
        checksum,result=readout_scale.run(out,repo,plan,cfg)
        print('R13 readout FINAL',checksum,result,flush=True)
    except Exception:
        io.write_json(out/'execution-failure.json',dict(traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
