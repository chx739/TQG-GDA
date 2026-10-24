"""Serial execution of the locked v1 scope; gates operate on entire phases."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from revision_common import STAGE, CODE, read, atomic_json


def run(name, arguments):
    state=STAGE/'run_state.json'
    current=read(state)
    current.update(status='running',current_command=name,pid=os.getpid(),last_update_unix=time.time())
    atomic_json(state,current)
    command=[sys.executable,*arguments]
    record=dict(name=name,command=command,cwd=str(CODE),started_unix=time.time())
    with (STAGE/'command_log.jsonl').open('a') as out:
        out.write(json.dumps(record)+'\n')
    print(json.dumps(dict(event='start_phase',name=name,command=command)),flush=True)
    log=STAGE/'logs'/f'{name}.log'
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1')
    with log.open('a') as output:
        process=subprocess.Popen(command,cwd=CODE,env=env,stdout=output,stderr=subprocess.STDOUT)
        atomic_json(STAGE/'live_process.json',dict(parent_pid=os.getpid(),pid=process.pid,name=name,
                                                  started_unix=record['started_unix'],log=str(log)))
        code=process.wait()
    record.update(returncode=code,finished_unix=time.time())
    with (STAGE/'command_log.jsonl').open('a') as out:
        out.write(json.dumps(record)+'\n')
    if code:
        current=read(state)
        current.update(status='failed',failed_command=name,returncode=code,last_update_unix=time.time())
        atomic_json(state,current)
        raise RuntimeError(f'{name} exited {code}; inspect {log}')
    current=read(state)
    current.setdefault('phases',{})[name]='completed'
    atomic_json(state,current)
    print(json.dumps(dict(event='phase_complete',name=name)),flush=True)


def main():
    (STAGE/'logs').mkdir(exist_ok=True)
    run('D0',['run_d0.py'])
    run('D1_profile',['diagnostics_sample_size.py','--domain','2','--seed','3405','--blocks','2'])
    summary=read(STAGE/'diagnostics/D1/source_summary.json')
    measured=summary['measured_block_seconds']/summary['completed_blocks']
    projected=200*measured*1.3
    ledger=read(STAGE/'compute_ledger.json')
    atomic_json(STAGE/'D1_profile.json',dict(seconds_per_block=measured,projected_D1_seconds_with_buffer=projected))
    if projected>ledger['cap_seconds']-ledger['consumed_seconds']:
        raise TimeoutError('Measured D1 alone would exceed remaining approved budget')
    run('D1_source',['diagnostics_sample_size.py'])
    run('D2_source',['diagnostics_local.py','--phase','source'])
    run('D1_offline',['evaluate_sample_size.py'])
    run('D2_offline',['diagnostics_local.py','--phase','target'])
    run('diagnostic_summary',['summarize_revision.py','--phase','diagnostics'])
    run('A_profile',['experiment_revision.py','--phase','A','--profile'])
    check_training_projection('A',80)
    run('A_training',['experiment_revision.py','--phase','A','--resume'])
    run('A_evaluation',['summarize_revision.py','--phase','evaluate_A'])
    decision=read(STAGE/'diagnostics/D2/decision.json')
    if decision['allow_B']:
        run('B_profile',['experiment_revision.py','--phase','B','--profile'])
        check_training_projection('B',120)
        run('B_training',['experiment_revision.py','--phase','B','--resume'])
        run('B_evaluation',['summarize_revision.py','--phase','evaluate_B'])
    else:
        atomic_json(STAGE/'phase_B_not_started.json',dict(reason='predeclared_D2_progression_gate_failed',
                                                        decision=decision,omitted_seeds_or_tasks=False))
    run('final_report',['summarize_revision.py','--phase','final_report'])
    run('figures',['plot_revision.py'])
    state=read(STAGE/'run_state.json')
    state.update(status='local_package_completed_pending_final_audit',finished_unix=time.time())
    atomic_json(STAGE/'run_state.json',state)


def check_training_projection(phase,jobs):
    arm='T5' if phase=='A' else 'T6'
    result=read(STAGE/'jobs/0_to_2/3405'/arm/'result.json')
    predicted=result['seconds']/result['completed_updates']*200*jobs*1.3
    ledger=read(STAGE/'compute_ledger.json')
    atomic_json(STAGE/f'{phase}_profile.json',dict(profile_updates=result['completed_updates'],
        profile_seconds=result['seconds'],projected_phase_seconds_with_buffer=predicted))
    if predicted>ledger['cap_seconds']-ledger['consumed_seconds']:
        raise TimeoutError('Measured complete phase no longer fits remaining cap; preserve profile and revise scope')


if __name__=='__main__':
    main()
