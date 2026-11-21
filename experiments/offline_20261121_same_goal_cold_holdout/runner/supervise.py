#!/usr/bin/env python3
"""Seal cold starts and run the primary paired matrix before the baseline matrix."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
from cold_run import ANALYSES, ARM_MAP, HERE, HOLDOUT, PROTOCOL, SCENES, SOURCE, SOURCE_RUNNER, native, read, sha

PRIOR = ANALYSES / 'same_goal_two_witness_20261121'
PHASE_ARMS = {'primary': ('fullG','fullS'), 'baselines': ('originalS','F_strict')}
EVALUATOR_RECEIPT = HERE.parent/'measurement/EVALUATOR_FREEZE.json'
EVALUATOR_RECEIPT_SHA = '040068ecb4a373d06b904df44ee922c19f2cdbf36f998b3c16b1f99ecbd46a7e'
FINAL_PROTOCOL_SHA = '1ee151a9019d5f95796a0ec4d71d00f940e7a22922e20019a9acb3373c9e7b24'


def write_once(path, value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')


def freeze():
    protocol = read(PROTOCOL)
    assert sha(PROTOCOL)==FINAL_PROTOCOL_SHA
    assert tuple(protocol['layouts'])==SCENES and tuple(protocol['modes'])==tuple(ARM_MAP)
    assert protocol['native_budget']==512
    registered = protocol['capture_paid']
    assert registered==sorted(set(registered)) and registered[0]==0 and registered[-1]==512
    captures = registered[1:]
    assert not (HERE/'runs').exists()
    assert not any((SOURCE_RUNNER/'runs/holdout').glob('*/*/attempt.json'))
    published = PRIOR/'publication/receipt.json'
    prior = read(published)
    assert prior['status']=='verified_published' and prior['head']==protocol['source_commit']
    gate_path = HERE/'regression/receipt.json'
    gate = read(gate_path)
    assert gate['status']=='passed' and gate['source_commit']==prior['head']
    assert gate['actor_code_sha256']==sha(HERE/'cold_run.py') and gate['protocol_sha256']==sha(PROTOCOL)
    assert gate['regression_code_sha256']==sha(HERE/'regression.py')
    witnesses = set(protocol['engineering_regression_before_holdout']) - {'receipt'}
    assert set(gate['witnesses'])==witnesses
    assert all(sorted(w['modes'])==['G','S'] and all(not row['holdout_opened'] for row in w['modes'].values())
               for w in gate['witnesses'].values())
    assert all(sha(Path(path))==value for path,value in gate['files_sha256'].items())
    world_freeze = HOLDOUT/'freeze.json'
    world = read(world_freeze)
    assert world['status']=='actor_inputs_only' and sha(world_freeze)==protocol['source_actor_freeze_sha256']
    evaluator_freeze = HOLDOUT/'evaluator_freeze.json'
    assert sha(evaluator_freeze)==protocol['source_evaluator_freeze_sha256']
    # Supervisor-side receipt only. It is deliberately excluded from all worker inputs.
    measure = read(EVALUATOR_RECEIPT)
    assert measure['status']=='frozen_evaluator_reference_inputs' and measure['source_commit']==prior['head']
    assert sha(EVALUATOR_RECEIPT)==EVALUATOR_RECEIPT_SHA
    transitive=read(native.B11/'freeze.json')['files_sha256']
    runtime_deps={Path(path):digest for path,digest in transitive.items()
                  if Path(path).suffix in ('.py','.so') or path.endswith('/selected_temperature.json')}
    assert len(runtime_deps)==130 and all(sha(path)==value for path,value in runtime_deps.items())
    assert all(sha(Path(path))==value for path,value in world['files_sha256'].items())
    files = [PROTOCOL,HERE/'cold_run.py',HERE/'regression.py',Path(__file__),published,gate_path,world_freeze,
             SOURCE/'PROTOCOL.json',SOURCE/'route_choice.py',SOURCE_RUNNER/'full_run.py',
             native.B11/'protocol.json',native.B11/'freeze.json',native.B11/'runtime_binding.py',
             native.YIELD/'execute.py',native.STRICT/'controller.py',native.STRICT/'strict_policy.py',native.STRICT/'freeze.json']
    files+=list(runtime_deps)
    actor_world = [Path(path) for path in world['files_sha256'] if not path.endswith('/initial_state.json')]
    prepared = HERE/'cold_inputs'
    prepared.mkdir(exist_ok=False)
    for layout in SCENES:
        archive = HOLDOUT/'prepared'/layout/'initial_state.json'
        assert sha(archive)==world['files_sha256'][str(archive)]
        facts = read(archive)
        assert facts['paid_actions']==facts['policy_select_calls']==0 and facts['stop_budget']==protocol['native_budget']
        assert facts['scene_role']=='ordinary_heldout_parent' and facts['seed']==protocol['scene_seed']
        key_fields = ('layout','scene_id','anchor','initial_packet_sha256','initial_owner_sha256','stop_budget')
        receipt = dict(status='frozen_sensor_and_seed_inputs_only', initial={key:facts[key] for key in key_fields},
                       seed=facts['seed'],source_prepared_state_sha256=sha(archive),
                       zero_paid_no_prior_map=True,source_prepared_state_path=str(archive))
        path = prepared/f'{layout}.json'
        write_once(path,receipt)
        actor_world.append(path)
    actor_world = sorted(set(actor_world))
    files = sorted(set(files + actor_world))
    assert not any('evaluator_references' in str(path) or 'evaluator_truth' in str(path) for path in files)
    schedule = [dict(layout=name,arm=arm,phase=phase) for phase,arms in PHASE_ARMS.items()
                for name in SCENES for arm in arms]
    write_once(HERE/'holdout_freeze.json',dict(status='frozen_before_any_heldout_actor',source_commit=prior['head'],
        frozen_unix_s=time.time(),schedule=[{'layout':row['layout'],'arm':row['arm']} for row in schedule],
        phases=schedule,capture_paid=captures,registered_capture_paid=registered,
        independent_evaluator_data_freeze_sha256=sha(evaluator_freeze),
        independent_evaluator_code_and_reference_sha256=sha(EVALUATOR_RECEIPT),
        inherited_B11_runtime_hash_count=len(runtime_deps),
        world_files_sha256={str(path):sha(path) for path in actor_world},
        files_sha256={str(path):sha(path) for path in files}))


def one(job,resume):
    layout,arm=job['layout'],job['arm']
    name=f'{layout}_{arm}'
    log=HERE/'logs'/f'{name}.log'
    pid_receipt=HERE/'worker_pids'/f'{name}.json'
    run=HERE/'runs'/layout/arm
    lock=HERE/'locks'/f'{name}.lock'
    with lock.open('a') as lease:
        # An orphan worker keeps this inherited descriptor. A resume blocks
        # here until it actually stops, then checks its result before launching.
        fcntl.flock(lease,fcntl.LOCK_EX)
        if resume and ((run/'manifest.json').is_file() or (run/'failure.json').is_file()):
            saved=read(pid_receipt)
            complete=(run/'manifest.json').is_file()
            if complete:
                manifest=read(run/'manifest.json')
                assert manifest['runner_freeze_sha256']==sha(HERE/'holdout_freeze.json')
                assert all(sha(run/file)==digest for file,digest in manifest['files_sha256'].items())
            return dict(**job,pid=saved['pid'],returncode=0 if complete else 1,
                        log_sha256=sha(log),reused_completed_or_failed_record=True)
        if log.exists() or pid_receipt.exists() or run.exists():
            assert resume, f'Existing launch for {name} needs fixed-freeze --resume'
            first=HERE/'interrupted_attempts'/layout/arm
            first.mkdir(parents=True,exist_ok=True)
            archive=first/f'attempt_{len(list(first.glob("attempt_*")))+1:03d}'
            archive.mkdir()
            for previous in (log,pid_receipt,run):
                if previous.exists():
                    previous.rename(archive/previous.name)
            write_once(archive/'resume_status.json',dict(status='interrupted_without_terminal_or_failure_receipt',
                source_freeze_sha256=sha(HERE/'holdout_freeze.json'),
                recovery='repeat_same_frozen_start_only_because_live_mapper_state_unavailable'))
        with log.open('x') as stream:
            action=subprocess.Popen([sys.executable,str(HERE/'cold_run.py'),layout,arm],stdout=stream,
                                    stderr=subprocess.STDOUT,pass_fds=(lease.fileno(),))
            write_once(pid_receipt,dict(**job,pid=action.pid,started_unix_s=time.time(),
                runner_freeze_sha256=sha(HERE/'holdout_freeze.json'),status='launched_under_exclusive_arm_lease'))
            rc=action.wait()
        return dict(**job,pid=action.pid,returncode=rc,completion_unix_s=time.time(),log_sha256=sha(log))


def workers_for_host(requested):
    logical = len(os.sched_getaffinity(0))
    with Path('/proc/meminfo').open() as stream:
        fields = dict((line.split(':',1)[0],int(line.split()[1])) for line in stream)
    available_kib=fields['MemAvailable']
    # Preserve 3 GiB for the controlling desktop/evaluator. Historical workers
    # used less than 1 GiB each; budget 1.5 GiB per holdout worker.
    memory_slots=max(1,(available_kib-3*1024**2)//(1536*1024))
    return min(requested,8,logical,memory_slots),dict(logical=logical,available_kib=available_kib,
                                                    memory_worker_slots=memory_slots)


def run_phase(phase,frozen,requested,resume):
    count, host = workers_for_host(requested)
    directory=HERE/'logs'
    directory.mkdir(exist_ok=True)
    (HERE/'worker_pids').mkdir(exist_ok=True)
    (HERE/'locks').mkdir(exist_ok=True)
    result_path=HERE/f'{phase}_supervisor_receipt.json'
    work=[row for row in frozen['phases'] if row['phase']==phase]
    if result_path.exists():
        assert resume
        prior=read(result_path)
        if all(any((HERE/'runs'/job['layout']/job['arm']/filename).exists()
                   for filename in ('manifest.json','failure.json')) for job in work):
            return prior
    if not resume:
        assert all(not (directory/f'{row["layout"]}_{row["arm"]}.log').exists() for row in work)
    if resume:
        history=HERE/'interrupted_attempts'/'phase_journals'
        history.mkdir(parents=True,exist_ok=True)
        for path in (result_path,HERE/f'{phase}_launch.json'):
            if path.exists():path.rename(history/f'{time.time_ns()}_{path.name}')
    started=time.monotonic()
    write_once(HERE/f'{phase}_launch.json',dict(status='armed_once',phase=phase,
        expected_jobs=work,actual_single_thread_workers=count,host_at_launch=host,runner_freeze_sha256=sha(HERE/'holdout_freeze.json')))
    jobs=[]
    with ThreadPoolExecutor(max_workers=count) as workers:
        futures=[workers.submit(one,row,resume) for row in work]
        for pending in as_completed(futures):
            result=pending.result()
            jobs.append(result)
            print(json.dumps(result),flush=True)
    jobs.sort(key=lambda job:work.index({key:job[key] for key in ('layout','arm','phase')}))
    success=all(row['returncode']==0 for row in jobs)
    receipt=dict(status='completed' if success else 'recorded_failures',phase=phase,workers=count,
        jobs=jobs,elapsed_s=time.monotonic()-started,runner_freeze_sha256=sha(HERE/'holdout_freeze.json'))
    write_once(HERE/f'{phase}_supervisor_receipt.json',receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--phase',choices=('all','primary','baselines'),default='all')
    parser.add_argument('--resume',action='store_true',help='Continue an interrupted fixed-freeze matrix without duplicate arms')
    parser.add_argument('--workers',type=int,choices=range(2,9),default=8)
    args=parser.parse_args()
    if args.freeze:
        freeze()
        return 0
    frozen=read(HERE/'holdout_freeze.json')
    assert all(sha(Path(path))==value for path,value in frozen['files_sha256'].items())
    assert sha(EVALUATOR_RECEIPT)==frozen['independent_evaluator_code_and_reference_sha256']
    phases=list(PHASE_ARMS) if args.phase=='all' else [args.phase]
    receipts=[]
    with (HERE/'schedule.lock').open('a') as master:
        fcntl.flock(master,fcntl.LOCK_EX | fcntl.LOCK_NB)
        for phase in phases:
            if phase=='baselines':
                read(HERE/'primary_supervisor_receipt.json')
            receipts.append(run_phase(phase,frozen,args.workers,args.resume))
        if args.phase=='all':
            overall=HERE/'supervisor_receipt.json'
            if overall.exists():
                assert args.resume
                history=HERE/'interrupted_attempts'/'phase_journals'
                history.mkdir(parents=True,exist_ok=True)
                overall.rename(history/f'{time.time_ns()}_{overall.name}')
            write_once(overall,dict(status='completed' if all(r['status']=='completed' for r in receipts)
                else 'recorded_failures',phases=receipts,runner_freeze_sha256=sha(HERE/'holdout_freeze.json')))
    return 0 if all(r['status']=='completed' for r in receipts) else 1


if __name__=='__main__':
    raise SystemExit(main())
