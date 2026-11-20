"""Replay the frozen R7 iteration-14 Mutagenicity P risk diagnosis."""
import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

from experiments.common.io import digest, write_json
from ..src import p_risk_replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--input', type=Path, help='Override the location of the same sealed P archive')
    parser.add_argument('--output', required=True, type=Path, help='New output directory')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[4]
    plan = json.loads(args.config.read_bytes())
    source = (args.input or Path(plan['input'])).resolve()

    def git(*argv):
        return subprocess.check_output(['git', *argv], cwd=repo, text=True).strip()

    files = [args.config.resolve(), Path(__file__).resolve(), Path(p_risk_replay.__file__).resolve()]
    write_json(args.output/'executed-config.json', dict(
        plan=plan, source_archive=str(source), output=str(args.output.resolve()),
        code_commit=git('rev-parse', 'HEAD'), code_status=git('status', '--short'),
        file_hashes={str(p.relative_to(repo) if p.is_relative_to(repo) else p): digest(p) for p in files},
        command=[sys.executable, *sys.orig_argv[1:]], cwd=str(Path.cwd()),
        started_utc=datetime.now(timezone.utc).isoformat(), python=sys.version,
        packages={p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'torch')}))
    tables = p_risk_replay.run(source, plan, repo)
    for name, rows in tables.items():
        path = args.output/f'{name}.csv'
        with path.open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(args.output/'evidence.json', dict(
        source_final_digest=plan['source_final_digest'], numerical_atol=plan['atol'],
        checked=['full seal chain on entry and exit', 'R5-authenticated target label snapshots',
                 'independent NumPy binary margins vs sealed log probabilities',
                 'sealed classification actions and FINAL endpoint/moment parity',
                 'original and independent-union risk maxima vs sealed records'],
        rows={name: len(rows) for name, rows in tables.items()},
        files={p.name: digest(p) for p in sorted(args.output.iterdir())}))
    print(f"Verified {len(tables['seeds'])} seeds, wrote {args.output}")


if __name__ == '__main__':
    main()
