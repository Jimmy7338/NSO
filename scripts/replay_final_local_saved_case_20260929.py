#!/usr/bin/env python3
"""Reconstruct fixed case0 from saved packets only; never query a World.

This is a new controller/TSDF computation on an existing paid history, not a
new online task and not a new reference/quality measurement. Exact equality
is required for every map and recorded controller state/decision, plus both
stored raw/ROI mesh snapshots. The output is outside the original case.
"""
import argparse
from copy import deepcopy
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
BATCH=ROOT/'audit_results/final_local_layout_validation_20260929'
CASE_ID='L02_side_column_h0_b42_n350918_G'
DEFAULT_OUTPUT=ROOT/'audit_results/final_delivery_20260929/saved_replay_case0'


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def plain(x):
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    if hasattr(x,'tolist'):return x.tolist()
    return x
def payload(x):return (json.dumps(plain(x),sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def write(path,value):
    with Path(path).open('xb') as stream:stream.write(payload(value))
def require(ok,message):
    if not ok:raise ValueError(message)


def verify_batch():
    freeze=read(BATCH/'source_freeze.json')
    for group in ('source_sha256','protected_20260923_sources','input_sha256'):
        for name,pin in freeze[group].items():require(sha(ROOT/name)==pin,'source/input changed: '+name)
    require(sha(BATCH/'source_archive.zip')==freeze['source_archive_sha256'],'source archive changed')
    with zipfile.ZipFile(BATCH/'source_archive.zip') as archive:
        require(set(archive.namelist())==set(freeze['source_sha256']),'source archive inventory differs')
        for name,pin in freeze['source_sha256'].items():
            require(hashlib.sha256(archive.read(name)).hexdigest()==pin,'archive source differs: '+name)
    protocol=read(BATCH/'protocol.json')
    require(protocol['cases'][0]['id']==CASE_ID and protocol['cases'][0]['mode']=='G','fixed representative case differs')
    require(protocol['budget']==42 and protocol['noise_seed']==350918,'protocol differs')
    config=ROOT/'configs/virtual3d/final_local_layout_validation_20260929.json'
    require(sha(config)==freeze['config_sha256'] and read(config)==protocol,'protocol/config binding differs')
    return protocol,freeze


def versions():
    answer={}
    for name in ('numpy','scipy','open3d'):
        try:answer[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:answer[name]=str(importlib.import_module(name).__version__)
    return answer


def compare_mesh(mesh,path):
    import numpy as np
    comparisons={}
    with np.load(path,allow_pickle=False) as saved:
        for name in ('vertices','triangles'):
            actual=np.asarray(getattr(mesh,name)); expected=saved[name]
            equal=actual.dtype==expected.dtype and np.array_equal(actual,expected)
            require(equal,'mesh array differs exactly: '+path.name+'/'+name)
            comparisons[name]=dict(shape=list(actual.shape),dtype=str(actual.dtype),exact_equal=True)
    return comparisons


def run(output):
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        require(os.environ.get(name)=='1',name+' must equal 1')
    require(not output.exists(),'replay output exists; use a new explicit output directory')
    protocol,freeze=verify_batch();case=protocol['cases'][0];folder=BATCH/'cases'/CASE_ID
    seal=read(folder/'seal.json')
    require('result.json' in seal,'fixed representative did not complete; do not silently choose another case')
    for name,pin in seal.items():require(sha(folder/name)==pin,'case artifact changed: '+name)
    result=read(folder/'result.json');require(result['status']=='completed','fixed case incomplete')
    preparation=read(BATCH/'preparation_seal.json');template=BATCH/'public_templates'/case['scene']
    for name,pin in preparation[case['scene']].items():require(sha(template/name)==pin,'public template changed')
    inputs={str((folder/name).relative_to(ROOT)):pin for name,pin in seal.items()}
    for path in [BATCH/'source_freeze.json',BATCH/'source_archive.zip',BATCH/'protocol.json',
                 BATCH/'preparation_seal.json',folder/'seal.json',*template.iterdir()]:
        if path.is_file():inputs[str(path.relative_to(ROOT))]=sha(path)
    output.mkdir(parents=True);(output/'source.py').write_bytes(Path(__file__).read_bytes())
    write(output/'predeclaration.json',dict(schema='final.saved_packet_replay.v1',case_index=0,case_id=CASE_ID,
        case_selected_before_new_results=True,source_sha256=sha(Path(__file__)),inputs=inputs,
        exact_map_hash_required=True,exact_controller_receipts_required=True,exact_mesh_arrays_required=True,
        new_online_tasks=0,world_or_sensor_calls_allowed=0,new_quality_evaluations_allowed=0,
        saved_packet_TSDF_reintegration=True))
    started=time.monotonic();frames=[];snapshots={};counts_before=None
    try:
        import numpy as np
        from env.information_pixel_v34 import InformationConfigV34,sensor_counts_v34
        from nso.cpu_sensor_contract_v10 import GridTransform
        from nso.sensor_contract_v34 import validate_sensor_packet_v34
        from nso.decision_replay_v13 import load_packet,array_hash
        from nso.pixel_information_v34 import SavedPotentialModelV34
        from nso.observation_belief_v35 import PublicTemplatesV35
        from nso.cpu_four_modules_v35 import CPUFourModuleControllerV35,ObservationV35
        from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
        from nso.surface_measurement_v34 import extract_observed_asset_mesh
        counts_before=dict(sensor_counts_v34())
        actual_versions=versions();expected_versions=read(folder/'started.json')['runtime']['versions']
        require(actual_versions==expected_versions,'scientific library versions differ from original')
        geometry=read(template/'geometry.json');info=read(template/'metadata.json');parent=geometry['parent']
        models=tuple(SavedPotentialModelV34(t,['unused']*len(t['poses'])) for t in geometry['tables'])
        with np.load(template/'sensor_templates.npz',allow_pickle=False) as d:
            templates=PublicTemplatesV35(models[0].poses,d['depth'],d['ranges'])
        cells=np.asarray(parent['nav_cells']);size=cells.max(axis=0)-cells.min(axis=0)+1
        config=InformationConfigV34(width_m=float(size[0]),height_m=float(size[1]),max_steps=42)
        shape=(int(round(size[1]/.2)),int(round(size[0]/.2)));transform=GridTransform(shape,.2)
        controller=CPUFourModuleControllerV35(models,templates,info['prefix'],mode='G',
            total_budget=42,informative_nodes=info['informative_nodes'])
        mapper=ObservedRuntimeMapperV10(shape,config,truncation_m=.12)
        trace=read(folder/'trace.json');saved=read(folder/'controller.json');actions=[]
        require(len(trace)==len(saved['history'])==result['paid_actions']+1,'saved frame counts differ')
        for paid,(expected,history) in enumerate(zip(trace,saved['history'])):
            packet=load_packet(folder/'packets'/f'{paid:03d}.npz')
            validate_sensor_packet_v34(packet,transform,config)
            require(packet.sha256()==expected['packet_sha256'],'saved packet identity differs')
            require(packet.action_id==paid and packet.action==(actions[-1] if paid else None),'replayed action does not match next saved packet')
            xy=packet.frame.world_from_camera[:2,3]-np.asarray(info['shift'][:2])
            require(np.allclose(xy,np.rint(xy),atol=1e-9,rtol=0),'off-grid saved odometry')
            pose=(*map(int,np.rint(xy)),int(packet.heading))
            mapper.update(packet.frame,packet.scan);map_hash=array_hash(mapper.belief)
            require(map_hash==expected['measured_map_sha256']==history['state']['measured_map_sha256'],'per-frame map hash differs')
            controller.accept(ObservationV35(frame_id=f'paid-{paid}',step=paid,pose=pose,action=packet.action,
                depth=packet.frame.depth_m,rgb=packet.frame.color_rgb,ranges=packet.scan.ranges_m,
                collision=bool(packet.collision),measured_map_sha256=map_hash))
            action=controller.next_action()
            actual_history=dict(step=paid,posterior=controller.posterior_receipts[-1],state=deepcopy(controller.state),next_action=action)
            require(payload(actual_history)==payload(history),'controller state/posterior/decision differs at '+str(paid))
            require(action==expected['next_action'],'trace decision differs')
            frames.append(dict(paid=paid,action=packet.action,next_action=action,map_sha256=map_hash,
                controller_exact_equal=True,packet_sha256=packet.sha256()))
            for stage,take in [('prefix',paid==18),('final',action is None)]:
                if not take:continue
                raw=mapper.mesh();mesh,crop=extract_observed_asset_mesh(raw,info['public_bounds'])
                snapshots[stage]=dict(raw=compare_mesh(raw,folder/f'{stage}_raw_mesh.npz'),
                    roi=compare_mesh(mesh,folder/f'{stage}_mesh.npz'))
                with np.load(folder/f'{stage}_map.npz',allow_pickle=False) as d:
                    require(np.array_equal(mapper.belief,d['belief']),'snapshot map differs')
                require(payload(crop)==payload(read(folder/f'{stage}_crop.json')),'ROI crop receipt differs')
            if action is not None:actions.append(action)
        for name,actual in [('plans',controller.plans),('calls',controller.calls),('actions',actions),('summary',controller.summary())]:
            require(payload(actual)==payload(saved[name]),'complete controller '+name+' differs')
        require(set(snapshots)=={'prefix','final'},'both measured snapshots required')
        counts_after=dict(sensor_counts_v34());require(counts_after==counts_before,'new World or sensor calls detected')
        report=dict(schema='final.saved_packet_replay_result.v1',status='passed',case_id=CASE_ID,
            saved_frames_replayed=len(frames),paid_actions=len(actions),new_online_tasks=0,new_worlds=0,
            new_sensor_queries=0,new_quality_evaluations=0,TSDF_reintegration_of_saved_frames=True,
            exact_per_frame_map_and_controller=True,exact_full_controller_plans_calls_actions=True,
            exact_raw_and_roi_meshes=True,snapshots=snapshots,library_versions=actual_versions,
            sensor_counts_before=counts_before,sensor_counts_after=counts_after,elapsed_seconds=time.monotonic()-started)
    except BaseException as exc:
        report=dict(schema='final.saved_packet_replay_result.v1',status='failed',case_id=CASE_ID,
            exception=repr(exc),traceback=traceback.format_exc(),saved_frames_replayed=len(frames),
            new_online_tasks=0,new_quality_evaluations=0,elapsed_seconds=time.monotonic()-started)
    write(output/'frame_checks.json',frames);write(output/'result.json',report)
    # Original bytes must remain unchanged after the independent computation.
    for rel,pin in inputs.items():require(sha(ROOT/rel)==pin,'input modified by replay: '+rel)
    write(output/'manifest.json',dict(schema='final.saved_packet_replay_manifest.v1',
        source_sha256=sha(Path(__file__)),original_case_seal_sha256=sha(folder/'seal.json'),
        files={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
    print(json.dumps(report,ensure_ascii=False),flush=True)
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    sys.exit(run(parser.parse_args().output))
