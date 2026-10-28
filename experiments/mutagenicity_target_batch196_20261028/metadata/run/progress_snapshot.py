"""Read-only progress and numerical health; no graph cache or labels."""
import json,time
from pathlib import Path
r=Path(__file__).resolve().parent;s=r/'stage3_execution'
live=json.loads((r/'live_state.json').read_text())
rows=[]
for f in sorted(s.glob('worker_*.json')):
 x=json.loads(f.read_text());q=x.get('calibration') or {}
 rows.append(dict(seed=f.stem,domain=x.get('spec',{}).get('domain'),step=x.get('step'),status=x.get('status'),age=round(time.time()-x.get('unix',time.time())),converged=q.get('converged'),iterations=q.get('iterations')))
print(json.dumps(dict(status=live['status'],finished=live['finished_jobs'],minutes=round(live.get('elapsed_minutes',0),1),workers=rows),ensure_ascii=False))
