#!/usr/bin/env python3
"""Seal and supervise all four arms on every prospective placement."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run import ARMS, CALIBRATION, FREEZE, HERE, PROTOCOL, SCENES, WEIGHTED, read, sha, write


def seal():
    if FREEZE.exists() or (HERE/'runs').exists():
        raise FileExistsError('An earlier seal or attempted actor output already exists')
    plan=read(PROTOCOL)
    dev=read(CALIBRATION)
    assert dev['calibration_passes']==1 and not dev['actor_action_issued']
    weights=read(WEIGHTED/'fit_weights.json')['performance']
    assert all(weights[k]==v for k,v in plan['weights'].items())
    actor_paths=[PROTOCOL,CALIBRATION,HERE/'direct_inspection.py',HERE/'run.py',SCENES/'scene.py']
    evaluator_paths=[HERE/'prepare.py',SCENES/'reference_protocol.json']
    placement_receipts={}
    for name in plan['placements']:
        start=SCENES/'prepared'/name
        evaluation=SCENES/'evaluator_references'/name
        actor_paths.extend(p for p in start.iterdir() if p.is_file())
        evaluator_paths.extend(p for p in evaluation.iterdir() if p.is_file())
        initial=read(start/'initial_state.json')
        preflight=read(evaluation/'optical_preflight.json')
        assert initial['paid_actions']==initial['policy_select_calls']==0
        assert initial['seed']==plan['scene_seed'] and initial['stop_budget']==plan['native_budget']
        assert preflight['lateral_cell_reachable'] and not preflight['actor_outcome_seen']
        placement_receipts[name]=preflight
    for side in ('east','west'):
        visible=placement_receipts[f'{side}_visible']['visible_shelf_samples']
        blocked=placement_receipts[f'{side}_screened']['visible_shelf_samples']
        assert visible>blocked,'Nominal fixed side-window condition not corroborated by source-ray preflight'
    historical = read(WEIGHTED/'holdout_freeze.json')['files_sha256']
    for path,digest in historical.items():
        if sha(path)!=digest:
            raise ValueError(f'Published G_perf dependencies changed: {path}')
    actor=historical|{str(path):sha(path) for path in actor_paths}
    artifacts={str(path):sha(path) for path in evaluator_paths}
    assert not any('evaluator_references' in path for path in actor)
    assert not set(artifacts)&set(actor)
    payload=dict(status='sealed_before_any_new_rollout',frozen_unix_s=time.time(),actor_files_sha256=actor,
                 evaluator_files_sha256=artifacts,independent_evaluator_geometry_only=placement_receipts,
                 schedule=[{'layout':layout,'arm':name} for layout in plan['placements'] for name in ARMS])
    with FREEZE.open('x') as handle:
        json.dump(payload,handle,indent=2,allow_nan=False)
        handle.write('\n')
    print(FREEZE,flush=True)


def supervise(workers):
    freeze=read(FREEZE)
    for field in ('actor_files_sha256','evaluator_files_sha256'):
        for path,digest in freeze[field].items():
            if sha(path)!=digest:
                raise ValueError(f'Frozen execution dependency changed: {path}')
    # The inherited runtime independently hashes this conventional name in
    # its own manifest, as well as reading the in-process frozen binding.
    native_seal=HERE/'holdout_freeze.json'
    if not native_seal.is_symlink() and not native_seal.exists():
        native_seal.symlink_to(FREEZE.name)
    if sha(native_seal)!=sha(FREEZE):
        raise ValueError('Native runtime seal alias has different bytes')
    out=HERE/'execution'
    out.mkdir(exist_ok=False)
    schedule=list(freeze['schedule'])
    procs,completed=[],[]
    export=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
    while schedule or procs:
        while schedule and len(procs)<workers:
            item=schedule.pop(0)
            layout,arm=item['layout'],item['arm']
            name=f'{layout}_{arm}'
            cmd=[sys.executable,str(HERE/'run.py'),'--layout',layout,'--arm',arm]
            log=out/f'{name}.log'
            started=time.time()
            with log.open('x') as stream:
                proc=subprocess.Popen(cmd,cwd=HERE,stdout=stream,stderr=subprocess.STDOUT,env=export)
            launch=item|dict(pid=proc.pid,command=cmd,started_unix_s=started,freeze_sha256=sha(FREEZE))
            write(out/f'{name}.launch.json',launch)
            procs.append((name,proc,launch,log))
        for item in procs[:]:
            name,proc,launch,log=item
            status=proc.poll()
            if status is None:
                continue
            folder=HERE/'runs'/launch['layout']/launch['arm']
            record=folder/('manifest.json' if status==0 else 'failure.json')
            entry=launch|dict(exit_code=status,finished_unix_s=time.time(),log_sha256=sha(log),
                              receipt_path=str(record),receipt_sha256=sha(record) if record.exists() else None)
            write(out/f'{name}.exit.json',entry)
            print(name,status,flush=True)
            completed.append(entry)
            procs.remove(item)
        if procs:
            time.sleep(2.)
    write(out/'phase_receipt.json',dict(freeze_sha256=sha(FREEZE),execution_records=completed))
    assert len(completed)==len(freeze['schedule'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('seal','run'))
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    if args.phase=='seal':
        seal()
    else:
        if not 1<=args.workers<=4:
            parser.error('worker count must be 1–4')
        supervise(args.workers)


if __name__=='__main__':
    main()
