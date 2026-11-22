#!/usr/bin/env python3
"""Exercise the native adapter with the regional switch disabled."""

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent/'weighted_geometric_control_20261122'
FREEZE = PRIOR/'holdout_freeze.json'
REGRESSION = ROOT/'regression'
RUNS = REGRESSION/'runs'
SPLIT, LAYOUT, ARM = 'holdout', 'paired_cycle_18', 'G_perf'
REFERENCE = PRIOR/'runner/runs'/SPLIT/LAYOUT/ARM
OUT = RUNS/SPLIT/LAYOUT/ARM
sys.dont_write_bytecode = True


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')


def import_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def archived_metrics():
    rows=read(PRIOR/'measurement/results/holdout/episodes.json')['rows']
    return next(r for r in rows if r['layout']==LAYOUT and r['arm']==ARM)


def run():
    REGRESSION.mkdir(exist_ok=True)
    sys.path.insert(0,str(PRIOR/'runner'))
    actor=import_file('weighted_regression_source',PRIOR/'runner/weighted_run.py')
    weights=actor.Weights(**read(PRIOR/'fit_weights.json')['performance'])
    frozen=actor.check_launch(LAYOUT,ARM,SPLIT,'G',weights,FREEZE)
    original=read(REFERENCE/'manifest.json')
    assert original['runner_freeze_sha256']==sha(FREEZE)
    for relative, expected in original['files_sha256'].items():
        assert sha(REFERENCE/relative)==expected
    metrics=archived_metrics()
    write(REGRESSION/'prelaunch.json',dict(regional_enabled=False,experiment_layout_role='seen_development',
        archive_layout=LAYOUT,protocol_sha256=sha(PRIOR/'PROTOCOL.json'),run_freeze_sha256=sha(FREEZE),
        original_manifest_sha256=sha(REFERENCE/'manifest.json'),source_commits=['4c388a5c06d51e53251a9c901bbefd511d5b2a95','8071a50fc7d04980d2eb390f633be7cae950706e'],
        weights=read(PRIOR/'fit_weights.json')['performance'],budget=192,archive_metrics={k:metrics[k] for k in ('C_free','F1_scene_05cm','J','actual_paid','elapsed_s','decision_wall_s','path_length_m')},
        environment=dict(host=os.uname().nodename,machine=os.uname().machine,interpreter=sys.executable,python=sys.version,
            available_cores=len(os.sched_getaffinity(0)),worker_threads={name:os.getenv(name) for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
            total_disk_bytes=shutil.disk_usage(ROOT).total,free_disk_bytes=shutil.disk_usage(ROOT).free,
            memtotal_kib=int(Path('/proc/meminfo').read_text().splitlines()[0].split()[1])),
        metric_tolerance_absolute=1e-12,planned_start_unix_s=time.time()))
    OUT.mkdir(parents=True,exist_ok=False)
    original_setup=actor.ORIGINAL_SETUP

    def with_disabled_hook(*args,**kwargs):
        result=original_setup(*args,**kwargs)
        _, (_, _, _, components, _, _, _), _, global_mode, _=result
        backend=components._cpu_backend
        select=backend.select_target
        original_source=Path(inspect.getfile(select.__func__)).resolve()
        assert frozen['files_sha256'][str(original_source)]==sha(original_source)
        write(REGRESSION/'active_binding.json',dict(runtime_class=type(backend).__name__,method=select.__func__.__name__,
              method_source=str(original_source),method_source_sha256=sha(original_source),global_mode=global_mode,
              switch='disabled_passthrough_before_selected_macro_reaches_local_option'))

        def disabled_selector(scene_index):
            return select(scene_index)

        backend.select_target=disabled_selector
        return result

    actor.ORIGINAL_SETUP=with_disabled_hook
    try:
        actor.source_run(LAYOUT,ARM,SPLIT,'G',weights,actor.read(PRIOR/'PROTOCOL.json'),frozen,FREEZE,OUT)
    except BaseException as failure:
        write(OUT/'failure.json',dict(error=repr(failure),traceback=traceback.format_exc()))
        raise


def verify():
    actual, old = [read(directory/'manifest.json') for directory in (OUT,REFERENCE)]
    criterion = read(REGRESSION/'prelaunch.json')
    assert sha(REFERENCE/'manifest.json')==criterion['original_manifest_sha256']
    for key in ('paid_actions','returned','collisions','initial_packet_sha256','final_packet_chain_sha256','path_length_m'):
        assert actual[key]==old[key],key
    for name in ('global_choices.json','initial_raw_rgbd_scan_sha256.json','terminal/native_mesh.ply','terminal/belief.npz'):
        assert sha(OUT/name)==sha(REFERENCE/name),name
    runtime_only={'decision_wall_s'}
    steps_a,steps_b=[read(p/'steps.json') for p in (OUT,REFERENCE)]
    for a,b in zip(steps_a,steps_b,strict=True):
        assert {k:v for k,v in a.items() if k not in runtime_only}=={k:v for k,v in b.items() if k not in runtime_only}
    ev=import_file('frozen_weighted_regression_evaluator',PRIOR/'measurement/evaluate.py')
    frozen=read(FREEZE)
    actor,_,metadata=ev.verify_freeze(SPLIT)
    initial=next(r for r in actor['layouts'] if r['layout']==LAYOUT)
    ref=ev.load_reference(LAYOUT,initial)
    metric=ev.files.load_metric(ev.read(ev.files.SCENES/'freeze.json'),ev.read(ev.files.OLD_EVAL/'freeze.json'))
    job=next(j for j in frozen['schedule'] if j['layout']==LAYOUT and j['arm']==ARM)
    row,clocks,_,_=ev.evaluate_run(SPLIT,LAYOUT,ARM,ref,metric,False,RUNS,job,sha(FREEZE),sha(PRIOR/'PROTOCOL.json'))
    archived=archived_metrics()
    for key in ('C_free','F1_scene_05cm','J'):
        assert abs(row[key]-archived[key])<=criterion['metric_tolerance_absolute'],key
    choices=read(OUT/'global_choices.json')
    write(REGRESSION/'verified.json',dict(status='exact_native_regression_passed',regional_enabled=False,
           matched_macro_choices=len(choices),matched_actual_charged_actions=len(steps_a),
           matched_map_and_observation_hashes=True,measured=row,checkpoint_metrics=clocks,
           archived_elapsed_s=archived['elapsed_s'],archive_manifest_sha256=criterion['original_manifest_sha256'],
           new_manifest_sha256=sha(OUT/'manifest.json'),active_binding=read(REGRESSION/'active_binding.json'),evaluation_metadata=metadata))
    print('Exact native baseline verified',LAYOUT,len(choices),len(steps_a),row['J'])


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('phase',choices=('run','verify'))
    {'run':run,'verify':verify}[cli.parse_args().phase]()
