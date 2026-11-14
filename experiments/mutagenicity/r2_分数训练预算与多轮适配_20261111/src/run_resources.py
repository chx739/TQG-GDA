"""Resource-only amendment after the fixed pilot; scientific settings unchanged.

The original INPUT and training implementation stay immutable. This wrapper
records scheduling provenance in the audit log before any full-grid execution.
"""
from experiments.common.paths import repository_root
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess

from experiments.common.io import digest
from experiments.mutagenicity.r1_图表示与边界支持_20261111.src.data import canonical_sha
from experiments.mutagenicity.r2_分数训练预算与多轮适配_20261111.src.common import write_json
from experiments.mutagenicity.r2_分数训练预算与多轮适配_20261111.src.run import configure, parallel


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True,choices=['scores','adapt','diagnose'])
    p.add_argument('--root', type=Path)
    args=p.parse_args();configure()
    repo=repository_root()
    registered=json.loads((repo/'experiments/mutagenicity/r2_分数训练预算与多轮适配_20261111/configs/r2-plan.json').read_text())
    root=args.root or Path(registered['output_root'])
    assert registered==json.loads((root/'execution-plan.json').read_text())
    for rel,expected in json.loads((root/'provenance.json').read_text())['file_hashes'].items():
        if rel.endswith('.py') and digest(repo/rel)!=expected:raise RuntimeError('Archived implementation changed: '+rel)
    pilot=json.loads((root/'pilot/complete.json').read_text())
    assert pilot['resume_exact_all_31_states'] and pilot['peak_worker_rss_bytes']<1024**3
    record=dict(reason='Fixed pilot costs 69-74 seconds per trajectory and <1GiB RSS; idle 64-logical-CPU, 125GiB host supports 32 independent single-thread workers',
        original_workers=8,workers=32,threads_per_worker=1,scientific_settings_changed=False,
        pilot_sha256=digest(root/'pilot/complete.json'),INPUT_digest=json.loads((root/'seals/INPUT.json').read_text())['digest'],
        wrapper_sha256=digest(Path(__file__)),wrapper_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        expected_adaptation_seconds_ideal=pilot['conservative_adaptation_cpu_seconds']/32,
        registration='After timing-only pilot; before score continuation/full grid; no target evaluation labels read')
    path=root/'logs/resource-amendment.json'
    if path.exists():
        old=json.loads(path.read_text())
        for k in ('workers','threads_per_worker','pilot_sha256','INPUT_digest','wrapper_sha256'):
            assert record[k]==old[k]
    else:write_json(path,record)
    execution=copy.deepcopy(registered);execution['resources']['workers']=32
    assert {k:v for k,v in execution.items() if k!='resources'}=={k:v for k,v in registered.items() if k!='resources'}
    parallel(args.stage,root,execution)
