"""Check the 2026-11-05 workflow migration without running data experiments."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]


def verify_archive(folder):
    manifest_path = folder/'artifact_manifest.json'
    if not manifest_path.exists():
        manifest_path = folder/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    total = 0
    for item in manifest['files']:
        path = (folder/item['path']).resolve()
        assert path.is_relative_to(folder.resolve()), path
        assert path.stat().st_size == item['bytes'], path
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                digest.update(block)
        assert digest.hexdigest() == item['sha256'], path
        total += item['bytes']
    result = dict(archive=folder.name, files=len(manifest['files']), bytes=total, sha256_passed=True)
    if (folder/'evaluation_index.csv').exists():
        result['evaluation_links'] = json.loads(subprocess.check_output(
            [sys.executable, 'scripts/verify_experiment_archive.py', str(folder)], cwd=ROOT))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    migration = json.loads((ROOT/'docs/workflow_migration.json').read_text())
    for old, new in migration['paths'].items():
        assert not (ROOT/old).exists(), old
        assert (ROOT/new).is_file(), new
    assert not list(ROOT.glob('*.py'))
    # Core scientific code, frozen protocols, configs and input indices are unchanged.
    changed = subprocess.check_output(['git', 'diff', '--name-only', migration['baseline_commit']], cwd=ROOT, text=True).splitlines()
    protected = [p for p in changed if p.startswith(('tqg_gda/', 'experiments/', 'idx/', 'data/', 'docs/research_plans/tqg_v2_20261031/'))
                 or (p.startswith('configs/') and not p.endswith('.md'))]
    assert not protected, protected
    draft = 'docs/v2/results/assumptions_brief_20261105.md'
    assert not subprocess.check_output(['git', 'ls-files', '--', draft], cwd=ROOT)
    subprocess.run(['git', 'check-ignore', '-q', draft], cwd=ROOT, check=True)

    documents = [ROOT/p for p in ['README.md', 'workflows/README.md', 'scripts/README.md',
                 'docs/code/README.md', 'docs/usage.md', 'docs/REPOSITORY_LAYOUT.md',
                 'configs/README.md', 'docs/validation/README.md',
                 'docs/validation/repository_20261105/README.md']]
    documents += list((ROOT/'docs/v2/implementation').glob('*.md'))
    links = 0
    for path in documents:
        for match in re.finditer(r'\]\(([^)\n]+)\)', path.read_text()):
            target = match[1].strip('<>')
            if target.startswith(('http:', 'https:', 'mailto:', '#', '/', 'data:')):
                continue
            target = target.split('#')[0]
            assert (path.parent/target).exists(), (path, target)
            links += 1
    archives = [p for p in (ROOT/'experiments').iterdir() if (p/'manifest.json').exists()]
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = sorted(pool.map(verify_archive, archives), key=lambda r: r['archive'])
    report = dict(date='2026-11-05', baseline_commit=migration['baseline_commit'],
                  moved_scripts=len(migration['paths']), root_python_files=0,
                  protected_scientific_files_unchanged=True, local_draft_untracked_and_ignored=True,
                  documents_checked=len(documents), relative_links_checked=links,
                  archives=results, archive_files=sum(r['files'] for r in results),
                  archive_bytes=sum(r['bytes'] for r in results),
                  new_real_data_experiments=0, target_labels_used_for_testing=False)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'archives'}, indent=2))


if __name__ == '__main__':
    main()
