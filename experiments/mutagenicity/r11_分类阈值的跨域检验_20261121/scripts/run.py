"""Execute the registered R11 threshold-only historical diagnostic."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import traceback

from experiments.common.io import digest,write_json
from ..src import inputs, run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[4]
    here=Path(__file__).resolve().parents[1]
    output=args.output.resolve()
    plan=json.loads(args.config.read_bytes())
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    files=[args.config.resolve(), here/plan['protocol'], *here.rglob('*.py')]
    hashes={str(p.resolve().relative_to(repo)):digest(p) for p in files}
    for rel,sha in hashes.items():
        committed=subprocess.check_output(['git','show',f'{revision}:{rel}'],cwd=repo)
        if inputs.sha(committed)!=sha:
            raise ValueError(f'Commit and review R11 code before execution: {rel}')
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'executed-config.json',dict(plan=plan,code_commit=revision,code_sha256=hashes,
        command=[sys.executable,*sys.orig_argv[1:]],cwd=str(Path.cwd()),
        started_utc=datetime.now(timezone.utc).isoformat(),
        versions={p:importlib.metadata.version(p) for p in ('torch','numpy','scipy')}))
    try:
        checksum,original=run.freeze(output,repo,plan)
        print('R11 OBS sealed',checksum,flush=True)
        result=run.score(output,repo,checksum,plan,original)
        print('R11 FINAL sealed',result,flush=True)
    except Exception:
        write_json(output/'execution-failure.json',dict(trace=traceback.format_exc(),obs_exists=(output/'OBS.json').exists()))
        raise


if __name__=='__main__':
    main()
