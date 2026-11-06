"""Run an explicit list of task configurations; preserve failures by task and seed."""
import argparse
import json
from pathlib import Path

from tqg_gda.config import Config
from tqg_gda.pipeline import run_task
from tqg_gda.utils import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, help='JSON list of new-schema task config objects')
    parser.add_argument('--output', required=True, help='New summary JSON path (no overwriting)')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error('Summary output already exists')
    values = json.loads(Path(args.manifest).read_text())
    if not isinstance(values, list) or not values:
        parser.error('Manifest must be a nonempty list')
    configs = [Config.from_dict(value) for value in values]  # validate all before work
    rows = []
    output.parent.mkdir(parents=True, exist_ok=True)
    for config in configs:
        try:
            rows.append(dict(config=config.to_dict(), result=run_task(config)))
        except Exception as error:
            rows.append(dict(config=config.to_dict(), error=f'{type(error).__name__}: {error}'))
        write_json(output, dict(tasks=rows, target_labels_used=False,
            note='No target metrics: no positional seed averaging or success-only pooled result'))
    if any('error' in row for row in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
