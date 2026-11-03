from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,subprocess,re,os
root=Path(__file__).resolve().parents[3];os.chdir(root)
baseline='e1f041d49b2e239c20f36f552b380ce60e161058'
move=json.loads((root/'docs/navigation_migration.json').read_text())['canonical_paths']
def mapped(p):
 for old,new in sorted(move.items(),key=lambda x:len(x[0]),reverse=True):
  if p==old or p.startswith(old+'/'):return new+p[len(old):]
 return p
report={'migration_aliases':[], 'documents':{},'unchanged_scientific_files':0,'archives':[], 'date':'2026-11-03', 'scope':'Documentation/navigation reorganization only; no experiment or algorithm changes', 'markdown_content_checks':0, 'frozen_protocol_bytes_unchanged':True}
for old,new in move.items():
 assert (root/old).is_symlink(),old
 assert (root/old).resolve()==(root/new).resolve(),old
 report['migration_aliases'].append(old)
# All non-Markdown results, figures and method designs retain their exact historical bytes.
paths=subprocess.check_output(['git','ls-tree','-rz','--name-only',baseline,'docs'],cwd=root).decode().split('\0')
for old in filter(None,paths):
 new=mapped(old)
 if new==old:continue
 if old.endswith('.md'):
  original=subprocess.check_output(['git','show',baseline+':'+old],cwd=root).decode()
  normalize=lambda text:re.sub(r'(\]\()([^\n)]+)(\))',r'\1DESTINATION\3',text)
  assert normalize(original)==normalize((root/new).read_text()),new
  report['markdown_content_checks']+=1
 if not old.endswith('.md') or '/method/design.md' in new:
  original=subprocess.check_output(['git','show',baseline+':'+old],cwd=root)
  assert (root/new).read_bytes()==original,new
  report['unchanged_scientific_files']+=1
# Frozen original protocols and source-history bytes are untouched.
assert not subprocess.check_output(['git','diff',baseline,'--','docs/research_plans/tqg_v2_20261031','docs/history/initial_source_manifest.sha256'],cwd=root)
files=[Path('README.md'),Path('experiments/README.md'),Path('configs/README.md'),Path('docs/README.md'),Path('docs/usage.md'),Path('docs/REPOSITORY_LAYOUT.md'),Path('docs/validation/README.md'),Path('docs/research_plans/README.md'),Path('docs/experiments/README.md')]
for folder in ['docs/v1','docs/v2','docs/code','experiments/v1','experiments/v2','configs/v1','configs/v2','docs/validation/repository_20261103']:files+=list(Path(folder).rglob('*.md'))
links=0
for p in files:
 for m in re.finditer(r'\]\(([^)\n]+)\)',p.read_text()):
  target=m.group(1).strip('<>');links+=1
  if target.startswith(('http:','https:','mailto:','codex:','data:','#')):continue
  base=target.split('#')[0]
  if not base or base.startswith('/'):continue
  assert (p.parent/base).exists(),(str(p),target)
report['documents']={'files':len(files),'links':links,'broken_relative_links':0}
# Check every manifest-listed byte in every published experiment, not just the moved documents.
def verify_archive(p):
 manifest=json.loads((p/'manifest.json').read_text());total=0
 for row in manifest['files']:
  dest=(p/row['path']).resolve();assert dest.is_relative_to(p.resolve()),dest
  data=dest.read_bytes();assert len(data)==row['bytes'],dest
  assert hashlib.sha256(data).hexdigest()==row['sha256'],dest
  total+=len(data)
 return {'archive':p.name,'files':len(manifest['files']),'bytes':total,'sha256_passed':True}
archives=[p for p in (root/'experiments').iterdir() if (p/'manifest.json').exists()]
with ThreadPoolExecutor(max_workers=4) as pool:
 for result in pool.map(verify_archive,archives):report['archives'].append(result)
# Existing complete-training evaluator/checkpoint associations must also still pass.
for p in archives:
 if not (p/'evaluation_index.csv').exists():continue
 out=subprocess.check_output(['python','scripts/verify_experiment_archive.py',str(p)],cwd=root)
 association=json.loads(out);assert association['passed'];next(r for r in report['archives'] if r['archive']==p.name)['evaluation_links']=association['evaluations']
Path('/tmp/tqg_layout_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'aliases':len(move),'scientific_files_unchanged':report['unchanged_scientific_files'],'document_checks':report['documents'],'verified_archives':len(report['archives']),'archive_files':sum(x['files'] for x in report['archives']),'archive_bytes':sum(x['bytes'] for x in report['archives'])},ensure_ascii=False,indent=2))
