#!/usr/bin/env python3
"""Seal each stage, supervise isolated native runs and retain attempts."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess
import sys
import time

from regression import OUT as CONTROL, PRIOR as WEIGHTED, read, sha, write
from runner import ARM_MODES

ROOT=Path(__file__).resolve().parent
EXECUTION=ROOT/'execution'


def hashes(paths):
    return {str(p.absolute()):sha(p) for p in sorted(set(paths)) if p.is_file()}


def verify(files):
    for path,digest in files.items():
        if sha(path)!=digest:
            raise ValueError(f'Frozen file drift {path}')


def freeze(phase):
    return ROOT/f'{phase.upper()}_FREEZE.json'


def seal(phase):
    if (EXECUTION/phase).exists():
        raise ValueError('Trials exist before this seal')
    protocol=ROOT/f'{phase.upper()}_PROTOCOL.json'
    spec=read(protocol)
    verified=ROOT/'regression/verified.json'
    assert read(verified)['status']=='exact_native_regression_passed'
    if phase=='formal':
        assert (ROOT/'results/development/summary.json').is_file() and (ROOT/'evaluate.py').is_file()
    scene_root=WEIGHTED/'scenes' if phase=='development' else ROOT/'scenes'
    old=read(WEIGHTED/'holdout_freeze.json')
    old_code={path:digest for path,digest in old['files_sha256'].items() if (Path(path).suffix in ('.py','.so')
              and WEIGHTED/'scenes' not in Path(path).parents) or path.endswith(('selected_temperature.json','execution/protocol.json'))}
    actor_initial=[scene_root/'prepared'/layout/name for layout in spec['layouts']
                   for name in ('initial_state.json','initial_frame.npz','initial_scan.npz')]
    assert all(p.is_file() for p in actor_initial)
    actor=old_code|hashes([protocol, ROOT/'runner.py', ROOT/'regional_adapter.py',ROOT/'regression.py',Path(__file__),
                          ROOT/'policy/observed_regions.py',scene_root/'scene.py',*actor_initial,
                          *(scene_root/'precomputed_assets').glob('*')])
    evaluator=[p for layout in spec['layouts'] for p in (scene_root/'evaluator_references'/layout).glob('*')]
    assert evaluator
    if phase=='formal':
        actor|=hashes([scene_root/'actor_freeze_holdout.json'])
        evaluator += [scene_root/'evaluator_freeze_holdout.json',scene_root/'formal_geometry_audit.json',
                      ROOT/'results/development/summary.json']
    offline=hashes([*evaluator,verified,ROOT/'evaluate.py',ROOT/'policy/test_observed_regions.py',WEIGHTED/'measurement/evaluate.py'])
    verify(actor)
    verify(offline)
    assert not any('evaluator_references' in path or 'evaluator_truth' in path for path in actor)
    jobs=[]
    for layout in spec['layouts']:
        for arm in spec['arms']:
            prior,region=ARM_MODES[arm]
            reused=phase=='development' and arm=='original_G'
            record={'reused_run':str(CONTROL),'reused_manifest_sha256':sha(CONTROL/'manifest.json')} if reused else {}
            jobs.append(dict(layout=layout,arm=arm,prior=prior,regional_enabled=region,literal=not reused,**record))
    write(freeze(phase),dict(schema=spec['schema'],phase=phase,frozen_unix_s=time.time(),protocol_relative=protocol.name,
                            protocol_sha256=sha(protocol),scene_root=str(scene_root),budget=spec['budget'],capture_paid=spec['capture_paid'],
                            local_auxiliary_weights=spec['local_auxiliary_weights'],local_route_invariants=spec['local_route_invariants'],
                            actor_files=actor,evaluator_files=offline,actor_initial_files=hashes([scene_root/'scene.py',*actor_initial]),jobs=jobs))
    print(freeze(phase),flush=True)


def execute(root,job,path,resume):
    key=root/job['layout']/job['arm']
    key.mkdir(parents=True,exist_ok=True)
    attempts=sorted(key.glob('attempt_*'))
    for attempt in attempts:
        if (attempt/'run/manifest.json').is_file() and (attempt/'exit.json').is_file():
            receipt=read(attempt/'exit.json')
            assert receipt['result_sha256']==sha(attempt/'run/manifest.json') and receipt['freeze_sha256']==sha(path)
            return receipt
    if attempts and not resume:
        raise ValueError(f'Prior incomplete attempt {key}; inspect recorded process before --resume')
    attempt=key/f'attempt_{len(attempts)+1:03d}'
    attempt.mkdir(exist_ok=False)
    output=attempt/'run'
    command=[sys.executable,str(ROOT/'runner.py'),job['layout'],job['arm'],'--freeze',str(path),'--out',str(output)]
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
             PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    log=attempt/'worker.log'
    started=time.time()
    with log.open('x') as stream:
        worker=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=env)
        launch=job|dict(pid=worker.pid,started_unix_s=started,command=command,freeze_sha256=sha(path))
        write(attempt/'launch.json',launch)
        status=worker.wait()
    result=output/('manifest.json' if status==0 else 'failure.json')
    receipt=launch|dict(exit_code=status,observed_exit_unix_s=time.time(),run_path=str(output),result_path=str(result),
                        result_sha256=sha(result) if result.is_file() else None,worker_log_sha256=sha(log))
    write(attempt/'exit.json',receipt)
    print(job['layout'],job['arm'],'exit',status,flush=True)
    return receipt


def dispatch(phase,workers,resume):
    path=freeze(phase)
    spec=read(path)
    verify(spec['actor_files'])
    verify(spec['evaluator_files'])
    root=EXECUTION/phase
    root.mkdir(parents=True,exist_ok=resume)
    final=root/'phase_receipt.json'
    if final.is_file():
        if all(r['exit_code']==0 for r in read(final)['episodes']):
            return
        assert resume
        final.rename(root/f'phase_receipt_{time.time_ns()}.json')
    rows=[]
    for job in spec['jobs']:
        if not job['literal']:
            result=Path(job['reused_run'])/'manifest.json'
            assert sha(result)==job['reused_manifest_sha256']
            rows.append(job|dict(exit_code=0,run_path=str(result.parent),result_path=str(result),
                                 result_sha256=sha(result),freeze_sha256=sha(path)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(execute,root,job,path,resume) for job in spec['jobs'] if job['literal']]
        for item in as_completed(futures):
            rows.append(item.result())
    write(final,dict(freeze_sha256=sha(path),phase=phase,episodes=rows))
    if any(r['exit_code']!=0 for r in rows):
        raise SystemExit('Technical failures are retained; inspect before resuming the same freeze.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('seal','run'))
    parser.add_argument('phase',choices=('development','formal'))
    parser.add_argument('--workers',type=int,default=5)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.action=='seal':
        seal(args.phase)
    else:
        dispatch(args.phase,args.workers,args.resume)
