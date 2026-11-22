#!/usr/bin/env python3
"""Verify and package experiment evidence without moving native measurements."""

import argparse
import hashlib
from pathlib import Path
import re
import shutil
import zipfile

from study import EXECUTION, ROOT, freeze, read, sha, verify, write


REMOTE='experiments/offline_20261122_region_structure_floor'


def zip_run(path, destination, include_mesh=False):
    files={}
    with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as bundle:
        for artifact in sorted(p for p in path.rglob('*') if p.is_file()):
            key=artifact.relative_to(path).as_posix()
            published=include_mesh or artifact.suffix!='.ply'
            files[key]={'sha256':sha(artifact),'bytes':artifact.stat().st_size,'in_compact_archive':published}
            if published:
                bundle.write(artifact,key)
    with zipfile.ZipFile(destination) as bundle:
        assert bundle.testzip() is None
        for name in bundle.namelist():
            assert files[name]['sha256']==hashlib.sha256(bundle.read(name)).hexdigest()
    return {'native_directory':str(path),'archive':destination.name,'archive_sha256':sha(destination),'files':files}


def stage(phase):
    destination=ROOT/'publication'/phase
    payload=destination/'payload'
    snapshot=read(freeze(phase))
    verified=read(ROOT/'results'/phase/'summary.json')
    completion=read(EXECUTION/phase/'phase_receipt.json')
    assert verified['status']=='evaluated' and verified['run_freeze_sha256']==completion['freeze_sha256']==sha(freeze(phase))
    assert len(completion['episodes'])==len(snapshot['jobs'])==verified['frozen_trials']
    verify(snapshot['actor_files'])
    verify(snapshot['evaluator_files'])
    listings=[]
    payload.mkdir(parents=True,exist_ok=False)
    source=[ROOT/'results'/phase,EXECUTION/phase/'phase_receipt.json']
    if phase=='development':
        source.extend(ROOT/name for name in ('README.md','DEVELOPMENT_PROTOCOL.json','DEVELOPMENT_FREEZE.json',
            'FORMAL_PROTOCOL.json','FORMAL_FREEZE.json','study.py','formal_study.py','runner.py','formal_runner.py',
            'regional_adapter.py','evaluate.py','figures.py','regression.py','archive.py', 'policy',
            'regression/prelaunch.json','regression/active_binding.json','regression/verified.json',
            'scenes/SCENES.md','scenes/scene.py','scenes/prepare_scenes.py','scenes/seal_scenes.py','scenes/geometry_audit.py',
            'scenes/formal_geometry_audit.json','scenes/actor_freeze_holdout.json','scenes/evaluator_freeze_holdout.json',
            'scenes/reference_protocol.json','docs/method.md','docs/flow.mmd','docs/evidence_index.md'))
    else:
        source.extend(ROOT/name for name in (
            'docs/evidence_index.md', 'paper.py', 'archive.py', 'figures.py',
            'local/README.md', 'local/PROTOCOL.json', 'local/EVIDENCE_INDEX.json',
            'local/FROZEN_INPUTS.json', 'local/MEASUREMENT_BINDING_EFFECTIVE.json',
            'local/calibration.json', 'local/results', 'local/prepare.py', 'local/run.py',
            'local/study.py', 'local/direct_inspection.py', 'local/evaluate.py', 'local/report.py',
            'local/recover.py', 'local/archive.py', 'local/scenes/scene.py',
            'local/execution/phase_receipt.json', 'local/evidence_checks/aborted_checkpoint_diagnostics.json',
            'local/evidence_checks/gc_lifetime_diagnosis/completed.json',
        ))
        pose_audit=ROOT/'docs/pose_hardware_audit'
        source.extend(pose_audit/name for name in (
            'README.md', 'REPLAY_PROTOCOL.json', 'audit_archive.py',
            'csm_replay_metrics.py', 'prepare_simulated_scan_trace.py', 'run_csm_replay.py',
            'hardware_chain.csv', 'hardware_chain_receipt.json', 'paired_csm_receipt.json',
            'refuse_csm_boundaries.py', 'map_refusion/map_refusion_receipt.json',
        ))
        for trace in ('replay_original_G','replay_region_G'):
            source.extend(pose_audit/trace/name for name in (
                'trace_receipt.json', 'csm_replay_metrics.json', 'csm_stamped_poses.csv',
                'csm_stamped_poses.json', 'independent_simulator_reference.npz', 'native_sensor_input.npz',
            ))
    source_files=[file for p in source for file in (p.rglob('*') if p.is_dir() else [p])
                  if file.is_file() and '__pycache__' not in file.parts and 'interim' not in file.parts]
    for file in source_files:
        target=payload/file.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        if file==ROOT/'docs/evidence_index.md':
            target.write_text(re.sub(r'\[Original thesis proposal\]\(<[^>]+>\)',
                                      'User-provided original thesis proposal (local evidence)',file.read_text()))
        elif file.suffix=='.svg' and file.stat().st_size>1024**2:
            with zipfile.ZipFile(target.with_suffix('.svg.zip'),'x',compression=zipfile.ZIP_DEFLATED) as bundle:
                bundle.write(file,file.name)
            with zipfile.ZipFile(target.with_suffix('.svg.zip')) as bundle:
                assert hashlib.sha256(bundle.read(file.name)).hexdigest()==sha(file)
        else:
            shutil.copyfile(file,target)
    bundles=payload/'run_evidence'
    bundles.mkdir()
    fixtures={}
    if phase=='formal':
        for name in ('prepared','precomputed_assets','evaluator_references'):
            fixtures[name]=zip_run(ROOT/'scenes'/name,bundles/f'formal_scenes_{name}.zip',include_mesh=True)
        fixtures['estimated_pose_map_refusion']=zip_run(
            ROOT/'docs/pose_hardware_audit/map_refusion',bundles/'formal_estimated_pose_refusion.zip')
    for job in completion['episodes']:
        result=Path(job['result_path'])
        if result.is_file():
            assert sha(result)==job['result_sha256']
        path=Path(job['run_path'])
        if job['exit_code']==0:
            receipt=read(result)
            verify({str(path/name):digest for name,digest in receipt['files_sha256'].items()})
        if path.exists():
            log_dir=path.parent if job['literal'] else path
            row=zip_run(log_dir,bundles/f"{phase}_{job['layout']}_{job['arm']}.zip")
        else:
            row={'native_directory':str(path),'status':'no_artifact'}
        listings.append({**row,'layout':job['layout'],'arm':job['arm'],'literal_of_this_phase':job['literal'],
                          'exit_code':job['exit_code'],'manifest_or_failure_sha256':job['result_sha256']})
    write(payload/f'EVIDENCE_INDEX_{phase.upper()}.json',{'phase':phase,'run_freeze_sha256':sha(freeze(phase)),
        'protocol_sha256':snapshot['protocol_sha256'],'physical_robot_run':False,
        'source_metric_summary_sha256':sha(ROOT/'results'/phase/'summary.json'),'run_bundles':listings,
        'formal_scene_fixtures':fixtures})
    plan=[]
    for name in sorted(p for p in payload.rglob('*') if p.is_file()):
        content=name.read_bytes()
        plan.append({'stage_path':str(name),'repo_path':REMOTE+'/'+name.relative_to(payload).as_posix(),
                    'size':len(content),'sha256':hashlib.sha256(content).hexdigest(),
                    'git_sha1':hashlib.sha1(f'blob {len(content)}\0'.encode()+content).hexdigest()})
    write(destination/'upload.json',plan)
    print('Staged',len(plan),'files for',phase)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('development','formal'))
    stage(parser.parse_args().phase)
