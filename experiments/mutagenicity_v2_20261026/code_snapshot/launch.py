"""Five isolated seeded CPU workers, monitored budget, then sealed evaluation."""
import os
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

from experiment import CODE, RUN, STAGE, Context, atomic_json, event, read, stamp


def main():
    ctx=Context()
    if ctx.manifest is None:
        raise ValueError('Run experiment.py prepare first')
    start=time.time()
    ledger_path=STAGE/'compute_ledger.json'
    if ledger_path.exists():
        ledger=read(ledger_path)
        if ledger.get('status')=='completed':
            raise ValueError('Completed matrix must not be relaunched')
    else:
        ledger=dict(started_at=stamp(),started_unix=start,deadline_unix=start+8*3600,
            wall_seconds_cap=8*3600,gpu_hours_consumed_so_far=0,status='training')
        atomic_json(ledger_path,ledger)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    processes={};streams={};restarts={seed:0 for seed in ctx.plan['seeds']}
    evaluation_process=None

    def spawn(seed):
        command=[sys.executable,'-u',str(CODE/'experiment.py'),'worker','--seed',str(seed)]
        stream=(STAGE/f'worker_stdout_{seed}_launch_{restarts[seed]}.log').open('x')
        streams[seed]=stream
        process=subprocess.Popen(command,cwd=CODE,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        processes[seed]=process
        event('worker_launch',seed=seed,pid=process.pid,command=command,launch_number=restarts[seed])

    for seed in ctx.plan['seeds']:
        spawn(seed)
    finished=set();last_print=0
    try:
        while len(finished)<len(ctx.plan['seeds']):
            now=time.time();elapsed=now-ledger['started_unix'];fraction=elapsed/ledger['wall_seconds_cap']
            if fraction>=.5 and not ledger.get('warned_50'):
                event('budget_warning',fraction=fraction,wall_seconds=elapsed)
                ledger['warned_50']=True
            if fraction>=.8:
                ledger['status']='awaiting_budget_approval'
                atomic_json(ledger_path,ledger)
                raise TimeoutError('80% budget checkpoint requires user continuation approval')
            states=[]
            for seed,process in processes.items():
                if seed in finished:
                    continue
                state_path=STAGE/f'worker_{seed}.json'
                state=read(state_path) if state_path.exists() else {}
                states.append({'seed':seed,**state})
                code=process.poll()
                if code is not None:
                    streams[seed].close()
                    if code==0:
                        finished.add(seed)
                    else:
                        restarts[seed]+=1
                        if restarts[seed]>=5:
                            raise RuntimeError(f'Worker {seed} failed five times; inspect its logs')
                        event('worker_restart',seed=seed,returncode=code,restart=restarts[seed])
                        spawn(seed)
                elif state.get('status')=='running' and now-state['unix']>150:
                    event('hang_timeout',spec=state.get('spec'),seed=seed,seconds_since_heartbeat=now-state['unix'])
                    os.killpg(process.pid,signal.SIGINT)
            completed=len(list((STAGE/'jobs').glob('*/*/*/completion.json')))
            live=dict(status='TRAINING',finished_jobs=completed,total_jobs=105,finished_seeds=sorted(finished),
                elapsed_minutes=elapsed/60,workers=states,git_commit=ctx.commit,updated_at=stamp())
            atomic_json(RUN/'live_state.json',live)
            ledger.update(wall_seconds=elapsed,status='training',worker_restarts=restarts)
            atomic_json(ledger_path,ledger)
            if now-last_print>=30:
                print(json.dumps(live),flush=True);last_print=now
            time.sleep(5)
        ledger.update(status='evaluating',wall_seconds=time.time()-ledger['started_unix'])
        atomic_json(ledger_path,ledger)
        atomic_json(RUN/'live_state.json',dict(status='EVALUATING',finished_jobs=105,updated_at=stamp()))
        with (STAGE/'evaluation_stdout.log').open('x') as stream:
            process=subprocess.Popen([sys.executable,'-u',str(CODE/'evaluate.py')],cwd=CODE,env=env,
                stdout=stream,stderr=subprocess.STDOUT)
            evaluation_process=process
            while process.poll() is None:
                print('{"status":"EVALUATING","training_jobs":105}',flush=True)
                time.sleep(15)
            if process.returncode:
                raise RuntimeError('Offline evaluation failed; inspect evaluation_stdout.log')
        ledger.update(status='completed',wall_seconds=time.time()-ledger['started_unix'],finished_at=stamp())
        atomic_json(ledger_path,ledger)
        atomic_json(RUN/'live_state.json',dict(status='COMPLETE',finished_jobs=105,total_jobs=105,
            evaluations=225,elapsed_minutes=ledger['wall_seconds']/60,git_commit=ctx.commit,updated_at=stamp()))
        print('{"status":"COMPLETE","adaptations":105,"evaluations":225}',flush=True)
    except BaseException:
        if evaluation_process is not None and evaluation_process.poll() is None:
            evaluation_process.send_signal(signal.SIGINT)
        for seed,process in processes.items():
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGINT)
        raise
    finally:
        for stream in streams.values():
            stream.close()


if __name__=='__main__':
    main()
