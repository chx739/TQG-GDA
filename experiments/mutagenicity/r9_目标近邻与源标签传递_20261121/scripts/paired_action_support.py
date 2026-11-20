"""Run the R9 input-only bound with authenticated R13 Q/R and R7 A logits."""
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
from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import r5_inputs, seals
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src import composition, pilot
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run
from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src import historical, observation, readout_scale
from ..src import paired_action_support


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parents[1]
    repo = here.parents[2]
    config = args.config.resolve()
    cfg = json.loads(config.read_bytes())

    def git(*argv):
        return subprocess.check_output(['git', *argv], cwd=repo)

    protocol = here / cfg['protocol']
    rel_protocol = protocol.relative_to(repo)
    revision = git('rev-parse', 'HEAD').decode().strip()
    modules = (paired_action_support, io, r5_inputs, seals, composition, pilot, inputs,
               prior_run, historical, observation, readout_scale)
    checked = [config, protocol, Path(__file__), *(Path(module.__file__) for module in modules)]
    hashes = {str(path.resolve().relative_to(repo)): io.digest(path) for path in checked}
    if inputs.sha(git('show', f"{cfg['registration_commit']}:{rel_protocol}")) != hashes[str(rel_protocol)]:
        raise PermissionError('R9 protocol differs from registration')
    if any(inputs.sha(git('show', f'{revision}:{file}')) != sha for file, sha in hashes.items()):
        raise PermissionError('Actual config or code differs from the execution commit')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    io.write_json(out / 'executed-config.json', dict(code_commit=revision, code_sha256=hashes, plan=cfg,
                  command=[sys.executable, *sys.orig_argv[1:]], cwd=str(Path.cwd()), torch_threads=1,
                  graph_roles=['Q', 'R'], started_utc=datetime.now(timezone.utc).isoformat(),
                  versions={n: importlib.metadata.version(n) for n in ('torch', 'torch-geometric', 'numpy')}))
    try:
        final = paired_action_support.run(out, repo, cfg)
        print('R9 paired-action FINAL', final, flush=True)
    except Exception:
        io.write_json(out / 'execution-failure.json', dict(traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
