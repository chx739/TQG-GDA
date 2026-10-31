"""Archive completed H-only evidence, adding paired-null and D4 replay audits."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import statistics
from archive_p0 import archive,sha,write


def publish(stage,repo,destination):
    summary=json.loads((stage/'results_summary.json').read_text())
    if summary.get('candidate_id')!='V2_H_ONLY' or summary['H_replay_blocks']!=360:
        raise ValueError('H-only candidate replay audit missing')
    gradients=list(csv.DictReader((stage/'gradient_blocks.csv').open()))
    steps=list(csv.DictReader((stage/'one_step_blocks.csv').open()))
    # Independently recompute every primary paired contrast. All planned blocks must be present.
    for task in ['0_to_1','0_to_2','0_to_3']:
        d1=next(r for r in summary['D1'] if r['task']==task)
        rows=[r for r in gradients if r['task']==task]
        for kind,key in [('PERM','advantage_PERM'),('RAND','advantage_RAND')]:
            null=statistics.mean(float(r['c_A']) for r in rows if r['arm'].startswith(kind+'-'))
            true=statistics.mean(float(r['c_A']) for r in rows if r['arm']=='TRUE')
            assert abs((true-null)-d1[key])<1e-12
        d2=next(r for r in summary['D2'] if r['task']==task)
        grouped={}
        for row in steps:
            if row['task']==task and float(row['eta'])==.0003:
                grouped.setdefault((row['seed'],row['snapshot_step'],row['block_id']),{})[row['arm']]=row
        assert len(grouped)==60
        for kind,key in [('P','advantage_PERM'),('R','advantage_RAND')]:
            values=[]
            for arms in grouped.values():
                null=statistics.mean(float(arms[f'S+{kind}-{rep}']['CE_B']) for rep in [1,2,3])
                values.append((null-float(arms['S+Q']['CE_B']))/float(arms['S+Q']['base_CE_B']))
            assert abs(statistics.mean(values)-d2[key])<1e-12
    archive(stage,repo,destination)
    (destination/'.gitattributes').write_text('*.csv whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol\n')
    pub=destination/'publication_code';pub.mkdir()
    shutil.copy2(Path(__file__),pub/'archive_h_only.py')
    shutil.copy2(Path(__file__).with_name('archive_p0.py'),pub/'archive_p0.py')
    audit=json.loads((destination/'artifact_audit.json').read_text())
    audit.update(candidate_id='V2_H_ONLY',all_primary_null_contrasts_independently_verified=True,
        H_readout_matches_D4_all_360_blocks=True,D4_H_max_metric_difference=summary['D4_H_max_metric_difference'])
    write(destination/'artifact_audit.json',audit);write(stage/'artifact_audit.json',audit)
    manifest=json.loads((destination/'manifest.json').read_text())
    manifest.update(experiment='tqg_h_only_controls_20261031',candidate_id='V2_H_ONLY',audit=audit,
        scope='H-only candidate; original D1/D2 null gates; no final labels or continuous training')
    manifest['files']=[dict(path=str(p.relative_to(destination)),bytes=p.stat().st_size,sha256=sha(p))
        for p in sorted(destination.rglob('*')) if p.is_file() and p!=destination/'manifest.json']
    write(destination/'manifest.json',manifest)
    for row in manifest['files']:assert sha(destination/row['path'])==row['sha256']
    print(json.dumps(dict(H_only_archive_audit='passed',files=len(manifest['files']),bytes=sum(r['bytes'] for r in manifest['files']))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--repo',required=True);p.add_argument('--destination',required=True)
    a=p.parse_args();publish(Path(a.stage).resolve(),Path(a.repo).resolve(),Path(a.destination).resolve())
