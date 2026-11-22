#!/usr/bin/env python3
"""Freeze, launch and recover exact checkpoints for the one new Chain 17 arm."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent/'route_intervention_20261122'
PROTOCOL = ROOT/'PROTOCOL.json'
MEASUREMENT = ROOT/'evaluation/MEASUREMENT_PROTOCOL.json'
FREEZE = ROOT/'RUN_FREEZE.json'
EXECUTION = ROOT/'execution'
LAYOUT = 'paired_chain_17'
RUN = ROOT/'runner/runs'/LAYOUT/'native'
PRIOR_RUNS = PREVIOUS/'runner/runs'/LAYOUT
SCHEDULE = [{'layout': LAYOUT, 'arm': 'native', 'route_id': 'native'}]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path, payload):
    with Path(path).open('x') as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def hashes(paths):
    return {str(path.resolve()): sha(path) for path in sorted(set(paths))}


def verify(ledger):
    for path, digest in ledger.items():
        if sha(path) != digest:
            raise ValueError(f'Invalid frozen bytes for {path}')


def branches():
    return {'category': PRIOR_RUNS/'category', 'geometry_equal_cost': PRIOR_RUNS/'geometry', 'geometry_top': RUN}


def seal():
    if RUN.exists() or EXECUTION.exists():
        raise ValueError('Native experiment already started')
    protocol = read(PROTOCOL)
    published = read(PREVIOUS/'publication/receipt.json')
    if published['head'] != protocol['source_commit'] or published['status'] != 'verified_published_draft':
        raise ValueError('Source release must bind the requested GitHub commit')
    earlier = read(PREVIOUS/'RUN_FREEZE.json')
    observed_set = ('fork_ready.json','attempt.json','manifest.json','intervention.json','steps.json',
                    'fork/checkpoint.json','fork/belief.npz','fork/native_mesh.ply')
    old = [PRIOR_RUNS/arm for arm in ('category','geometry')]
    manifests = [read(directory/'manifest.json') for directory in old]
    for i, directory in enumerate(old):
        manifest = manifests[i]
        assert manifest['runner_freeze_sha256'] == sha(PREVIOUS/'RUN_FREEZE.json')
        assert manifest['native_budget'] == protocol['native_budget_including_return'] == 192
        verify({str(directory/member): digest for member,digest in manifest['files_sha256'].items()})
    assert sha(old[0]/'fork_ready.json') == sha(old[1]/'fork_ready.json')
    assert read(old[0]/'fork_ready.json')['identity']['target_pose'] == protocol['target_pose']
    fork = read(old[0]/'fork_ready.json')['identity']
    assert (fork['actual_paid'],fork['selection_call_id']) == (protocol['fork_paid'],protocol['selection_call_id'])
    local_actor = [PROTOCOL, ROOT/'source/actor_anchor.json', *(ROOT/'runner').glob('*.py'),
                   PREVIOUS/'runner/run.py', PREVIOUS/'source/frozen_anchors.json',
                   *(directory/member for directory in old for member in observed_set)]
    actor = earlier['actor_files_sha256'] | hashes(local_actor)
    assert not any('evaluator_references' in path or 'evaluator_truth' in path or 'evaluator_bindings' in path for path in actor)
    offline = earlier['evaluator_files_sha256'] | hashes([Path(__file__), MEASUREMENT,
                *(ROOT/'evaluation').glob('*.py'), *(path for path in (ROOT/'source').iterdir()
                    if path.is_file() and path.name!='actor_anchor.json'),
                PREVIOUS/'RUN_FREEZE.json', PREVIOUS/'evaluation/evaluate.py', PREVIOUS/'publication/receipt.json',
                PREVIOUS/'evaluation/results/measured.json', PREVIOUS/'publication/payload/EVIDENCE_INDEX.json'])
    verify(actor)
    verify(offline)
    write(FREEZE, dict(schema=protocol['schema'], status='sealed_before_one_native_geometry_run',
                      frozen_unix_s=time.time(), protocol_sha256=sha(PROTOCOL),
                      measurement_protocol_sha256=sha(MEASUREMENT),
                      source_runtime_freeze_sha256=sha(PREVIOUS/'RUN_FREEZE.json'),
                      source_fork_ready_sha256=sha(old[0]/'fork_ready.json'),
                      reused_manifest_sha256={arm: sha(path/'manifest.json') for arm,path in branches().items()
                                             if arm!='geometry_top'},
                      schedule=SCHEDULE, actor_files_sha256=actor, evaluator_files_sha256=offline))
    print(FREEZE)


def actor_command():
    return [sys.executable,str(ROOT/'runner/run.py'),'--freeze',str(FREEZE),'--out',str(RUN)]


def env():
    return dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')


def run():
    frozen = read(FREEZE)
    assert frozen['protocol_sha256'] == sha(PROTOCOL) and frozen['schedule'] == SCHEDULE
    verify(frozen['actor_files_sha256'])
    verify(frozen['evaluator_files_sha256'])
    EXECUTION.mkdir(exist_ok=False)
    command = actor_command()
    log = EXECUTION/'native.log'
    started = time.time()
    with log.open('x') as stream:
        child = subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=env())
        launch = dict(pid=child.pid,started_unix_s=started,command=command,schedule=SCHEDULE[0],runner_freeze_sha256=sha(FREEZE))
        write(EXECUTION/'launch.json', launch)
        exit_code = child.wait()
    result = RUN/('manifest.json' if exit_code==0 else 'failure.json')
    write(EXECUTION/'exit.json', launch|dict(exit_code=exit_code,exit_unix_s=time.time(), log_sha256=sha(log),
            result_path=str(result),result_sha256=sha(result) if result.exists() else None))
    if exit_code!=0 or not result.is_file():
        raise SystemExit(f'The native actor failed with status {exit_code}. Its original log and output are retained.')
    print(result)


def capture_clocks():
    paths = branches()
    manifests = {name:read(path/'manifest.json') for name,path in paths.items()}
    receipts, done = [], {}
    for comparison in ('geometry_top','geometry_equal_cost'):
        chosen_arms = ('category',comparison)
        clock = max(manifests[arm]['route_end']['actual_paid'] for arm in chosen_arms)
        for arm in chosen_arms:
            manifest, native = manifests[arm], paths[arm]
            record = dict(comparison=comparison,arm=arm,paid=clock,trace_receipt_sha256=sha(native/'manifest.json'))
            compatible = next((name for name,checkpoint in manifest['checkpoints'].items()
                               if checkpoint['status']=='captured' and checkpoint['actual_paid']==clock),None)
            if compatible is not None:
                receipts.append(record|dict(status='native_capture',checkpoint=compatible))
                continue
            if manifest['paid_actions'] < clock:
                receipts.append(record|dict(status='native_terminal_before_clock',terminal_paid=manifest['paid_actions']))
                continue
            if (arm,clock) not in done:
                output = ROOT/'runner/replays'/arm/f'paid_{clock}'
                if arm=='geometry_top':
                    command = actor_command()
                    command[-1] = str(output)
                else:
                    original_arm = 'category' if arm=='category' else 'geometry'
                    command = [sys.executable,str(PREVIOUS/'runner/run.py'),LAYOUT,original_arm,
                               '--freeze',str(PREVIOUS/'RUN_FREEZE.json'),'--out',str(output)]
                command.extend(['--capture-at',str(clock),'--original-run',str(native)])
                log = EXECUTION/f'capture_{arm}_{clock}.log'
                with log.open('x') as stream:
                    subprocess.run(command,check=True,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=env())
                result = output/'manifest.json'
                diagnostic = read(result)
                assert diagnostic['original_full_trace_verified'] and diagnostic['extra_capture_paid']==clock
                assert diagnostic['verified_against_original_manifest_sha256']==record['trace_receipt_sha256']
                done[arm,clock] = dict(status='exact_capture_replay',output=str(output),result_sha256=sha(result),
                                       command=command, log_sha256=sha(log))
            receipts.append(record|done[arm,clock])
    write(EXECUTION/'paired_clock_receipt.json',dict(runner_freeze_sha256=sha(FREEZE),captures=receipts))
    print(EXECUTION/'paired_clock_receipt.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('seal','run','capture-clocks'))
    args=parser.parse_args()
    {'seal':seal,'run':run,'capture-clocks':capture_clocks}[args.phase]()


if __name__=='__main__':
    main()
