#!/usr/bin/env python3
"""One full-history, class-blind two-instance representation check; no sensors."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter
import traceback
import zipfile

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import open3d as o3d
from scipy.ndimage import label
from env.facility_choice_v24_1 import FacilityChoiceWorldV24_1
from nso.cpu_sensor_contract_v10 import json_value, digest
from nso.decision_replay_v13 import array_hash,load_packet
from nso.observed_shape_v24 import ObservedShapeBackendV24,VERSION
from utils.facility_outline_v23 import OutlineEvaluatorV23
from scripts.probe_facility_choice_v24_prefix import MarkerTracks,marker_mask
from scripts.probe_facility_shape_v23 import versions

SOURCE=ROOT/'audit_results/facility_choice_v24_paid_prefix_20260915'
PROTOCOL=ROOT/'docs/research/V24_PAIRED_PREFIX_SHAPE_PROTOCOL_20260915.md'
DEFAULT=ROOT/'audit_results/facility_choice_v24_shape_prefix_20260915'
CONFIG=dict(version='paired-prefix-shape-check-v24-1',task_asset_count=2,
    output_limit_bytes=8*1024**2,free_reserve_bytes=32*1024**2,receipt_reserve_bytes=65536,
    outline=dict(boundary_spacing_m=.01,thresholds=[.02,.05,.10],
                 completion_tolerance_m=.05,completion_minimum_iou=.9),
    association='first measured seed uniquely inside an evaluation-only fixed window; no reassignment',
    representations=['raw','measured','inferred','completed'],
    once_per_case_at='final paid prefix endpoint',new_sensor_frames=0,new_tsdf_fusions=0)
PROGRESS=dict(case_index=None,last_saved_packet_id=None,backend_snapshot_index=None)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def total_bytes(root):
    return sum(p.stat().st_size for p in root.rglob('*') if p.is_file())


def safe_bytes(root,path,payload,receipt=False):
    allowance=CONFIG['output_limit_bytes']-(0 if receipt else CONFIG['receipt_reserve_bytes'])
    if total_bytes(root)+len(payload)>allowance:
        raise OSError('new representation output would exceed frozen 8 MiB cap')
    allocated=((len(payload)+4095)//4096)*4096+(0 if receipt else CONFIG['receipt_reserve_bytes'])
    if shutil.disk_usage(root).free-allocated<CONFIG['free_reserve_bytes']:
        raise OSError('write would cross frozen 32 MiB free-space reserve')
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('xb') as stream:
        stream.write(payload)
    os.replace(temporary,path)


def write(root,path,value,receipt=False):
    payload=(json.dumps(json_value(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    safe_bytes(root,path,payload,receipt)


def seal(root):
    write(root,root/'artifact_hashes.json',{str(p.relative_to(root)):sha(p)
        for p in sorted(root.rglob('*')) if p.is_file() and p!=root/'artifact_hashes.json'})


def mesh_hash(mesh):
    return {k:array_hash(np.asarray(getattr(mesh,k))) for k in ('vertices','triangles')}


def mesh_from_arrays(vertices,triangles):
    mesh=o3d.geometry.TriangleMesh()
    mesh.vertices=o3d.utility.Vector3dVector(vertices)
    mesh.triangles=o3d.utility.Vector3iVector(triangles)
    return mesh


def save_mesh(root,path,mesh):
    stream=io.BytesIO()
    np.savez_compressed(stream,vertices=np.asarray(mesh.vertices),triangles=np.asarray(mesh.triangles))
    safe_bytes(root,path,stream.getvalue())


def check_original():
    manifest=read(SOURCE/'manifest.json')
    result=read(SOURCE/'result.json')
    if manifest['status']!='complete' or result['status']!='complete':
        raise ValueError('four completed source prefixes required')
    inventory=read(SOURCE/'artifact_hashes.json')
    actual_files={str(p.relative_to(SOURCE)) for p in SOURCE.rglob('*')
                  if p.is_file() and p!=SOURCE/'artifact_hashes.json'}
    if actual_files!=set(inventory):
        raise ValueError('sealed source file set differs from inventory')
    for name,expected in inventory.items():
        if sha(SOURCE/name)!=expected:
            raise ValueError('source artifact changed: '+name)
    for name,expected in manifest['source_sha256'].items():
        if sha(ROOT/name)!=expected:
            raise ValueError('original frozen source changed: '+name)
    if versions()!=manifest['versions'] or sha(SOURCE/'sources.zip')!=manifest['source_archive_sha256']:
        raise ValueError('original frozen dependency versions or source archive differ')
    counts={key:result[key] for key in ('physical_trajectories','independent_process_replays',
        'physical_paid_actions','replay_paid_actions','saved_raw_packets')}
    if counts!=dict(physical_trajectories=4,independent_process_replays=4,
                   physical_paid_actions=976,replay_paid_actions=976,saved_raw_packets=980):
        raise ValueError('source 4+4/976+976/980 contract differs')
    if len(manifest['cases'])!=4 or [c['index'] for c in manifest['cases']]!=list(range(4)):
        raise ValueError('source four-case order differs')
    processes=[]
    for index in range(4):
        folder=SOURCE/f'case_{index:02d}'
        verify=read(folder/'verification.json')
        if verify['status']!='passed' or verify['physical_process_id']==verify['replay_process_id']:
            raise ValueError('four independent source verification receipts required')
        processes.extend((verify['physical_process_id'],verify['replay_process_id']))
        paid=234 if index<2 else 254
        saved=read(folder/'result.json')
        if saved['paid_actions']!=paid or len(saved['trace'])!=paid+1 or saved['raw_frames']!=paid+1:
            raise ValueError('source case paid length or trace length differs')
        if [row['action_id'] for row in saved['trace']]!=list(range(paid+1)):
            raise ValueError('source trace action IDs are not continuous')
        if {p.name for p in (folder/'packets').iterdir()}!={f'{i:04d}.npz' for i in range(paid+1)}:
            raise ValueError('source packet file sequence differs')
        if verify['raw_packets_verified']!=paid+1 or verify['paid_actions']!=paid:
            raise ValueError('independent verification frame/action count differs')
    if len(set(processes))!=8:
        raise ValueError('eight distinct recorded physical/replay processes required')
    return manifest


def check_frozen(root):
    manifest=read(root/'manifest.json')
    for name,expected in manifest['source_sha256'].items():
        if sha(ROOT/name)!=expected:
            raise ValueError('representation source changed: '+name)
    if sha(root/'sources.zip')!=manifest['source_archive_sha256'] or versions()!=manifest['versions']:
        raise ValueError('new source archive or dependency versions changed')
    for name,expected in manifest['input_sha256'].items():
        if sha(ROOT/name)!=expected:
            raise ValueError('sealed input changed: '+name)
    check_original()
    return manifest


def prepare(root):
    original=check_original()
    if shutil.disk_usage(ROOT).free<CONFIG['free_reserve_bytes']+CONFIG['output_limit_bytes']:
        raise OSError('8 MiB output plus 32 MiB reserve required before prepare')
    root.mkdir(parents=True,exist_ok=False)
    new_sources=[Path(__file__).resolve(),PROTOCOL]
    sources=dict(original['source_sha256'])
    sources.update({str(p.relative_to(ROOT)):sha(p) for p in new_sources})
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in new_sources:
            archive.writestr(str(path.relative_to(ROOT)),path.read_bytes())
    safe_bytes(root,root/'sources.zip',stream.getvalue())
    inputs=[SOURCE/name for name in ('manifest.json','result.json','artifact_hashes.json','sources.zip')]
    manifest=dict(status='prepared',config=CONFIG,backend_version=VERSION,source_sha256=sources,
        source_archive_sha256=sha(root/'sources.zip'),versions=versions(),
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},
        old_source_archive_referenced_not_duplicated=True,source_root=str(SOURCE),
        cases=[dict(index=c['index'],parent=c['parent'],assignment=c['assignment'],
            packets=c['route']['paid_actions']+1) for c in original['cases']],
        preparation_sensor_frames=0,preparation_snapshots=0,preparation_quality_evaluations=0)
    write(root,root/'manifest.json',manifest);check_frozen(root)
    print('Prepared 4 cases x 2 backends; 0 observations/snapshots/evaluations',flush=True)


def component_seed_hits(frame,minimum_pixels):
    groups,count=label(marker_mask(frame)&(frame.depth_m>.15)&np.isfinite(frame.depth_m),
                       np.ones((3,3),bool))
    hits=[]
    for number in range(1,count+1):
        rr,cc=np.nonzero(groups==number)
        if len(rr)<minimum_pixels:
            continue
        depth=frame.depth_m[rr,cc]
        local=np.column_stack(((cc-frame.intrinsic[0,2])*depth/frame.intrinsic[0,0],
                               (rr-frame.intrinsic[1,2])*depth/frame.intrinsic[1,1],depth))
        points=local@frame.world_from_camera[:3,:3].T+frame.world_from_camera[:3,3]
        median=np.median(points,axis=0)
        chosen=int(np.argmin(np.linalg.norm(points-median,axis=1)))
        hits.append(dict(observed_seed_xyz=points[chosen].tolist(),pixel=[int(rr[chosen]),int(cc[chosen])],
            component_pixels=len(rr),distance_to_component_median_m=float(np.linalg.norm(points[chosen]-median))))
    return hits


class ReferenceWorld(FacilityChoiceWorldV24_1):
    def step(self,*args,**kwargs):
        raise AssertionError('reference evaluator must never execute world.step')
    def sense(self,*args,**kwargs):
        raise AssertionError('reference evaluator must never generate RGBD')
    def scan(self,*args,**kwargs):
        raise AssertionError('reference evaluator must never generate lidar')


def fixed_seed_association(seeds,objects,observed_track_count):
    rows=[];ids=[int(item['id']) for item in objects]
    for index,seed in enumerate(seeds):
        matches=[]
        if seed is not None:
            point=np.asarray(seed['observed_seed_xyz'])
            for item in objects:
                lo,hi=np.asarray(item['evaluation_bounds'])
                if np.all(point>=lo) and np.all(point<=hi):
                    matches.append(int(item['id']))
        rows.append(dict(observed_slot=index,matches=matches,
            status='unassigned' if not matches else 'unique' if len(matches)==1 else 'ambiguous',
            reference_id=matches[0] if len(matches)==1 else None))
    counts={identifier:sum(row['reference_id']==identifier for row in rows) for identifier in ids}
    missing=[identifier for identifier,count in counts.items() if count==0]
    duplicates=[identifier for identifier,count in counts.items() if count>1]
    passed=observed_track_count==len(ids) and all(row['status']=='unique' for row in rows) and not missing and not duplicates
    return dict(rule=CONFIG['association'],rows=rows,missing_reference_ids=missing,
        duplicate_reference_ids=duplicates,observed_track_count=observed_track_count,
        expected_track_count=len(ids),seed_association_gate_passed=passed,
        gate_scope='seed-to-window association only; surface instance separation is a separate gate',
        mapping_is_evaluation_only=True,never_fed_back_or_used_to_correct_predictions=True,
        union_window_quality_does_not_certify_instance_identity=True)


def execute(root):
    manifest=check_frozen(root);original=read(SOURCE/'manifest.json')
    manifest['status']='running';write(root,root/'manifest.json',manifest)
    (root/'meshes').mkdir()
    started=perf_counter();cases=[];observe_calls=0;packet_count=0
    for case in original['cases']:
        index=case['index'];PROGRESS.update(case_index=index,last_saved_packet_id=None,backend_snapshot_index=None)
        folder=SOURCE/f'case_{index:02d}';source_result=read(folder/'result.json')
        backends=[ObservedShapeBackendV24() for _ in range(CONFIG['task_asset_count'])]
        tracks=MarkerTracks(original['config']);seeds=[None]*len(backends);extras=[]
        history=[]
        for action_id,expected in enumerate(source_result['trace']):
            packet=load_packet(folder/'packets'/f'{action_id:04d}.npz')
            if packet.sha256()!=expected['packet_sha256'] or packet.action_id!=action_id:
                raise ValueError('source packet content/order mismatch')
            rows=tracks.consume(packet)
            if rows!=expected['marker_components']:
                raise ValueError('frozen observed marker association differs')
            hits=component_seed_hits(packet.frame,original['config']['marker_minimum_valid_pixels_per_component'])
            if len(rows)!=len(hits):
                raise ValueError('binary component and observed-track order mismatch')
            current_seeds={}
            for row,hit in zip(rows,hits):
                track_index=row['observed_track_index']
                if track_index>=len(backends):
                    extras.append(dict(action_id=action_id,observed_track_index=track_index,**hit))
                elif seeds[track_index] is None:
                    receipt=dict(action_id=action_id,observation_id=action_id,
                        observed_slot=track_index,packet_sha256=packet.sha256(),**hit)
                    seeds[track_index]=receipt
                    current_seeds[track_index]=receipt['observed_seed_xyz']
            for slot,backend in enumerate(backends):
                f=packet.frame
                accepted=backend.observe(f.depth_m,f.intrinsic,f.world_from_camera,
                    observation_id=action_id,observed_seed_xyz=current_seeds.get(slot))
                if not accepted:
                    raise ValueError('unexpected duplicate observation')
                observe_calls+=1
            history.append(packet.sha256());packet_count+=1
            PROGRESS['last_saved_packet_id']=action_id
        with np.load(folder/'final_mesh.npz',allow_pickle=False) as data:
            actual={k:array_hash(data[k]) for k in ('vertices','triangles','vertex_colors')}
            if actual!=source_result['final_mesh_sha256']:
                raise ValueError('saved final TSDF array hashes differ')
            raw=mesh_from_arrays(data['vertices'].copy(),data['triangles'].copy())
        snapshots=[];instance_rows=[]
        for slot,backend in enumerate(backends):
            PROGRESS['backend_snapshot_index']=slot
            snapshot=backend.snapshot(raw_mesh=raw)
            geometry={key:dict(count=len(snapshot[key]),sha256=array_hash(snapshot[key])) for key in
                ('measured_points_xyz','ground_points_xyz','cleaned_points_xyz','unassigned_points_xyz')}
            meshes={key:mesh_hash(snapshot[key]) for key in ('observed_mesh','inferred_mesh','completed_mesh')}
            no_inference=len(snapshot['inferred_mesh'].triangles)==0
            equality=meshes['observed_mesh']==meshes['completed_mesh']
            if no_inference and not equality:
                raise ValueError('no inferred triangles but completed differs from measured')
            if not snapshot['completion']['accepted'] and not no_inference:
                raise ValueError('rejected hypothesis unexpectedly supplies triangles')
            files={}
            for key in ('observed_mesh','inferred_mesh'):
                path=root/'meshes'/f'case_{index:02d}_slot_{slot}_{key}.npz'
                save_mesh(root,path,snapshot[key]);files[key]=str(path.relative_to(root))
            instance_rows.append(dict(observed_slot=slot,seed=seeds[slot],observed_frames=snapshot['observed_frames'],
                ground=snapshot['ground_plane'],completion=snapshot['completion'],geometry_arrays=geometry,
                mesh_sha256=meshes,mesh_files=files,
                mesh_sizes={key:dict(vertices=len(snapshot[key].vertices),triangles=len(snapshot[key].triangles))
                    for key in ('observed_mesh','inferred_mesh','completed_mesh')},
                completed_composition=['observed_mesh','inferred_mesh'],
                no_new_inferred_triangles=no_inference,completed_equals_measured=equality,
                inferred_surfaces_are_measured=False))
            # Retain only bounded geometry/audit; large point arrays are not duplicated on disk.
            snapshots.append({key:snapshot[key] for key in ('observed_mesh','inferred_mesh','completed_mesh')})
            del snapshot
            print(f"case={index} slot={slot} frames={len(backend.frames)} accepted={instance_rows[-1]['completion']['accepted']} reason={instance_rows[-1]['completion']['reason']}",flush=True)
        del backends
        # The first reference-world construction occurs AFTER both backend outputs.
        reference=ReferenceWorld(case['parent'],case['assignment'],
            original['config']['sensor_model'],original['config']['noise_seed'])
        evaluator=OutlineEvaluatorV23.from_world(reference,**CONFIG['outline'])
        association=fixed_seed_association(seeds,reference.objects,len(tracks.tracks))
        association['extra_observed_track_receipts']=extras
        arguments=dict(coverage=source_result['final_coverage_2d'],
            returned=source_result['returned_to_anchor'],collisions=source_result['collisions'],
            failed=False,paid_actions=source_result['paid_actions'],budget=source_result['paid_actions'])
        submitted=dict(raw=raw)
        for name,key in (('measured','observed_mesh'),('inferred','inferred_mesh'),('completed','completed_mesh')):
            combined=o3d.geometry.TriangleMesh()
            for snapshot in snapshots:
                combined+=snapshot[key]
            submitted[name]=combined
        scores={name:evaluator.evaluate(mesh,**arguments) for name,mesh in submitted.items()}
        for row,snapshot in zip(instance_rows,snapshots):
            # Report ALL reference-window rows for this fixed observed slot.
            # No quality-maximizing remapping or GT correction is performed.
            row['completed_per_reference_window']=evaluator.evaluate(snapshot['completed_mesh'],**arguments)['instances']
            row['fixed_seed_association']=association['rows'][row['observed_slot']]
        if all(row['no_new_inferred_triangles'] for row in instance_rows) and scores['completed']!=scores['measured']:
            raise ValueError('no inference but union completed/measured evaluation differs')
        duplicate_component=instance_rows[0]['geometry_arrays']['cleaned_points_xyz']['count']>0 and (
            instance_rows[0]['geometry_arrays']['cleaned_points_xyz']==instance_rows[1]['geometry_arrays']['cleaned_points_xyz'])
        nonempty=all(row['geometry_arrays']['cleaned_points_xyz']['count']>0 and
                     row['mesh_sizes']['observed_mesh']['triangles']>0 for row in instance_rows)
        separation=nonempty and not duplicate_component
        case_result=dict(index=index,parent=case['parent'],assignment=case['assignment'],
            endpoint_action_id=source_result['paid_actions'],saved_packets_processed=len(history),
            input_packet_sequence_sha256=digest(history),observed_track_summary=tracks.summary(),
            fixed_seed_association=association,instances=instance_rows,
            raw_source=dict(path=str((folder/'final_mesh.npz').relative_to(ROOT)),sha256=sha(folder/'final_mesh.npz'),
                            array_sha256=actual),
            representation_mesh_sha256={name:mesh_hash(mesh) for name,mesh in submitted.items()},
            representations=scores,reference_signature=evaluator.reference_signature,
            prefix_coverage=arguments['coverage'],prefix_returned=arguments['returned'],
            prefix_coverage_at_least_80=arguments['coverage']>=.8,
            all_no_inference=all(row['no_new_inferred_triangles'] for row in instance_rows),
            duplicate_seeded_components=duplicate_component,all_observed_instances_nonempty=nonempty,
            minimum_instance_separation_gate_passed=separation,
            observed_instance_seed_and_separation_gate_passed=association['seed_association_gate_passed'] and separation,
            mapping_does_not_correct_backend_output=True,new_sensor_frames=0,new_tsdf_fusions=0,
            evaluation_context='scripted prefix only; metric budget equals paid prefix length',
            formal_choice_task_eligibility_not_assessed=True)
        case_result=json_value(case_result)
        write(root,root/f'case_{index:02d}.json',case_result)
        cases.append(case_result)
        print(f"case={index} seed_association={association['seed_association_gate_passed']} separation={separation} rawQ={scores['raw']['05cm']['outline_macro_quality']:.6f} measuredQ={scores['measured']['05cm']['outline_macro_quality']:.6f} completedQ={scores['completed']['05cm']['outline_macro_quality']:.6f}",flush=True)
        del snapshots,submitted,raw,reference,evaluator
    pairs=[]
    for offset in (0,2):
        a,b=cases[offset:offset+2]
        def class_blind_output(case):
            return dict(instances=[{key:row[key] for key in ('observed_slot','observed_frames','ground','completion',
                'geometry_arrays','mesh_sha256','no_new_inferred_triangles','completed_equals_measured')}
                for row in case['instances']],
                seed_geometry=[None if row['seed'] is None else {k:row['seed'][k] for k in
                    ('action_id','observed_slot','observed_seed_xyz','pixel','component_pixels','distance_to_component_median_m')}
                    for row in case['instances']],
                representation_mesh_sha256=case['representation_mesh_sha256'])
        left,right=class_blind_output(a),class_blind_output(b)
        pairs.append(dict(parent=a['parent'],geometry_and_rejection_pair_identical=left==right,
                          left_geometry_sha256=digest(left),right_geometry_sha256=digest(right)))
    result=dict(status='complete',backend_version=VERSION,cases=cases,pairs=pairs,
        saved_packets_processed=packet_count,backend_observe_calls=observe_calls,backend_snapshots=8,
        full_history_geometry_pair_passed=all(p['geometry_and_rejection_pair_identical'] for p in pairs),
        all_seed_associations_unique=all(c['fixed_seed_association']['seed_association_gate_passed'] for c in cases),
        all_observed_instance_seed_and_separation_gates_passed=all(
            c['observed_instance_seed_and_separation_gate_passed'] for c in cases),
        any_inference_accepted=any(row['completion']['accepted'] for c in cases for row in c['instances']),
        new_sensor_frames=0,new_physical_actions=0,new_tsdf_fusions=0,semantic_policy=False,
        full_architecture_efficacy_proven=False,source_prefixes_previously_inspected=True,
        scope='common class-blind terminal prefix representation, not policy return or independent test scene',
        implicit_enclosure_faces_are_prior_contribution_not_measured_accuracy=True)
    if packet_count!=980 or observe_calls!=1960:
        raise ValueError('full-history processing count differs from frozen task')
    check_frozen(root)
    write(root,root/'result.json',result)
    write(root,root/'timing.json',dict(process_id=os.getpid(),elapsed_s=perf_counter()-started,
        new_sensor_frames=0,new_tsdf_fusions=0,output_bytes_before_inventory=total_bytes(root)))
    manifest['status']='complete';write(root,root/'manifest.json',manifest);seal(root)
    if total_bytes(root)>CONFIG['output_limit_bytes'] or shutil.disk_usage(root).free<CONFIG['free_reserve_bytes']:
        raise OSError('final resource boundary violated')
    print(dict(status='complete',geometry_pair=result['full_history_geometry_pair_passed'],
        output_mib=total_bytes(root)/1024**2,free_mib=shutil.disk_usage(root).free/1024**2),flush=True)


def failure(root,phase,error):
    if not root.exists():
        return
    path=root/f'failure_{phase}.json'
    if not path.exists():
        write(root,path,dict(status='failed',phase=phase,error=repr(error),traceback=traceback.format_exc(),
            process_id=os.getpid(),progress=PROGRESS),receipt=True)
    if (root/'manifest.json').exists():
        manifest=read(root/'manifest.json');manifest.update(status='failed',error=repr(error))
        write(root,root/'manifest.json',manifest,receipt=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--run',action='store_true')
    args=parser.parse_args();root=args.output.resolve()
    # Stale/repeated entry is rejected before acquiring a writable experiment.
    if args.prepare:
        if root.exists():
            raise ValueError('new output directory required; old evidence untouched')
    else:
        if read(root/'manifest.json')['status']!='prepared':
            raise ValueError('only a prepared, unstarted representation run is writable')
        check_frozen(root)
    try:
        prepare(root) if args.prepare else execute(root)
    except Exception as error:
        failure(root,'prepare' if args.prepare else 'run',error)
        raise


if __name__=='__main__':
    main()
