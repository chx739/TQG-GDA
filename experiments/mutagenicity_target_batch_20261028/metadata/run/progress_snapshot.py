"""Read-only progress; no model inference or label access."""
import json
from pathlib import Path
import time

RUN=Path(__file__).resolve().parent
STAGE=RUN/'stage3_execution'

def snapshot():
    live=json.loads((RUN/'live_state.json').read_text()) if (RUN/'live_state.json').exists() else {}
    workers=[]
    for path in sorted(STAGE.glob('worker_*.json')):
        row=json.loads(path.read_text())
        workers.append(dict(seed=int(path.stem.split('_')[1]),status=row.get('status'),
                            spec=row.get('spec'),step=row.get('step'),
                            heartbeat_age_seconds=round(time.time()-row.get('unix',time.time()),1)))
    completions=[json.loads(p.read_text()) for p in STAGE.glob('jobs/*/*/*/completion.json')]
    output=dict(status=live.get('status'),finished_jobs=len(completions),total_new_jobs=30,
                elapsed_minutes=round(live.get('elapsed_minutes',0),1),workers=workers,
                failures=len(list(STAGE.glob('jobs/*/*/*/failure_attempt_*.txt'))),
                done_marker=(RUN/'stage_3_done').exists())
    return output

if __name__=='__main__':print(json.dumps(snapshot(),ensure_ascii=False))
