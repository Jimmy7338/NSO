#!/usr/bin/env python3
"""Seal contract tests/handwritten trace. Never acquire or replay a world."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter
import unittest
import zipfile

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tests/virtual3d'))
from scripts.preflight_observed_decisions_v26 import source_snapshot,forbid_world_calls,COUNTERS
from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
from test_persistent_observation_option_v29 import (
    PersistentObservationOptionV29,state_for,cue_for,target,moved,replace)

OUT=ROOT/'audit_results/persistent_option_contract_v29_20260917'
CAP=1024**2
COUNTS=dict(actual_mapper_constructions=0,new_worlds=0,new_sensor_packets=0,new_main_tasks=0,
            TSDF_integrations=0,mesh_extractions=0,Q_evaluations=0)


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(name,value):
    blob=(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    used=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    if used+len(blob)>CAP or shutil.disk_usage(OUT).free-len(blob)<64*1024**2:
        raise RuntimeError('evidence space bound')
    (OUT/name).write_bytes(blob)


def forbidden_mapper(*args,**kwargs):
    COUNTS['actual_mapper_constructions']+=1
    raise AssertionError('real mapper forbidden in contract verification')


def trace():
    state=state_for(budget=70)
    option=PersistentObservationOptionV29();option.initialize(state,state)
    option.open(target(),(cue_for(),))
    for i in range(7):
        action=option.next_action()
        patches=tuple(replace(p,bits=5,best_range=2.) for p in state.patches)
        state=moved(state,action,patches=patches)
        option.consume(state,state,action=action)
    return dict(source='handwritten_observed_map_fixture_not_collected_sensor_history',
                closed=option.closed,events=option.events,paid_ledger=option.paid_ledger)


def run():
    if OUT.exists():raise ValueError('sealed output exists; no overwrite')
    if shutil.disk_usage(OUT.parent).free<CAP+64*1024**2:raise RuntimeError('disk reserve')
    OUT.mkdir();started=perf_counter()
    try:
        guards=forbid_world_calls()
        ObservedRuntimeMapperV10.__init__=forbidden_mapper
        suite=unittest.defaultTestLoader.discover(str(ROOT/'tests/virtual3d'),
            pattern='test_persistent_observation_option_v29.py')
        sources=source_snapshot()
        protocol=ROOT/'docs/research/V29_PERSISTENT_OPTION_PROTOCOL_20260917.md'
        sources[str(protocol.relative_to(ROOT))]=sha(protocol)
        data=io.BytesIO()
        with zipfile.ZipFile(data,'w',zipfile.ZIP_DEFLATED) as z:
            for name in sorted(sources):z.writestr(name,(ROOT/name).read_bytes())
        if len(data.getvalue())>CAP//2:raise RuntimeError('source archive bound')
        (OUT/'sources.zip').write_bytes(data.getvalue())
        save('manifest.json',dict(source_sha256=sources,source_archive_sha256=sha(OUT/'sources.zip'),
            guarded_world_classes=guards,real_mapper_constructor_guarded=True,
            scope='unit contracts plus handwritten seven-action option; not closed-loop experiment'))
        log=io.StringIO();result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
        (OUT/'tests.log').write_text(log.getvalue())
        if not result.wasSuccessful():raise AssertionError('contract tests failed; see tests.log')
        save('handwritten_trace.json',trace())
        for name,expected in sources.items():
            if sha(ROOT/name)!=expected:raise AssertionError('source changed: '+name)
        save('result.json',dict(status='passed',tests_run=result.testsRun,failures=len(result.failures),
            errors=len(result.errors),counters=COUNTS,world_tripwire_counters=COUNTERS,
            elapsed_seconds=perf_counter()-started,main_tasks_used=12,
            whole_option_contract_verified=True,synthetic_trace_paid_steps=7,
            persistent_policy_closed_loop_efficacy_proven=False,quality_feedback_calibrated=False))
    except BaseException as error:
        save('failure.json',dict(type=type(error).__name__,error=str(error),counters=COUNTS))
        raise
    finally:
        save('artifact_hashes.json',{str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*'))
            if p.is_file() and p.name!='artifact_hashes.json'})
    print((OUT/'result.json').read_text())


if __name__=='__main__':run()
