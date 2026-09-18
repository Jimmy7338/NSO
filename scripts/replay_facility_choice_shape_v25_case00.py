#!/usr/bin/env python3
"""One frozen V25 two-instance snapshot of235 saved case00 packets; no sensors."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
import traceback
import uuid
import zipfile

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import open3d as o3d
import scipy
import shapely
from scipy.ndimage import label
from nso.cpu_sensor_contract_v10 import json_value,digest
from nso.decision_replay_v13 import array_hash,load_packet
from nso.observed_shape_v25 import ObservedShapeBackendV25,VERSION

PAID=ROOT/'audit_results/facility_choice_v24_paid_prefix_20260915'
OLD=ROOT/'audit_results/facility_choice_v24_shape_prefix_20260915'
PROTOCOL=ROOT/'docs/research/V25_CASE00_SAVED_HISTORY_PROTOCOL_20260915.md'
DEFAULT=ROOT/'audit_results/facility_choice_v25_case00_shape_20260915'
BACKEND=ROOT/'nso/observed_shape_v25.py'
EXPECTED_BACKEND_SHA='cb8685f30f9d5e9da5e5d2eacfdbf1d6d7ce8485a0f4d633e03908f4d5febe71'
CONFIG=dict(version='v25-case00-paid-history-shape-check-1',case_index=0,task_asset_count=2,
    saved_packets=235,paid_actions=234,maximum_worker_seconds=600,
    output_limit_bytes=4*1024**2,free_reserve_bytes=64*1024**2,receipt_reserve_bytes=65536,
    outline=dict(boundary_spacing_m=.01,thresholds=[.02,.05,.10],
                 completion_tolerance_m=.05,completion_minimum_iou=.9),
    raw_and_measured_scores='carry original scores only after exact unchanged mesh verification',
    newly_evaluated=['union_inferred','union_completed','slot0_completed','slot1_completed'],
    old_backend_reruns=0,new_world_actions=0,new_sensor_frames=0,new_tsdf_fusions=0,
    output_representations=['raw','measured','inferred','completed'],development_diagnostic=True)
PROGRESS=dict(phase='unstarted',last_packet_id=None,observe_calls=0,snapshots=0,quality_evaluations=0)


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def read(path):return json.loads(path.read_text())

def versions():
    return dict(python=sys.version,numpy=np.__version__,open3d=o3d.__version__,
        scipy=scipy.__version__,shapely=shapely.__version__,geos=shapely.geos_version_string)

def total_bytes(root):return sum(p.stat().st_size for p in root.rglob('*') if p.is_file())

def capacity(root,payload_size,receipt=False):
    reserve=0 if receipt else CONFIG['receipt_reserve_bytes']
    if total_bytes(root)+payload_size > CONFIG['output_limit_bytes']-reserve:
        raise OSError('write would exceed4MiB output cap including receipt reserve')
    allocation=((payload_size+4095)//4096)*4096+4096+reserve
    if shutil.disk_usage(root).free-allocation < CONFIG['free_reserve_bytes']:
        raise OSError('write would cross64MiB actual disk reserve')

def safe_bytes(root,path,payload,receipt=False):
    capacity(root,len(payload),receipt)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('xb') as stream:stream.write(payload)
    os.replace(temporary,path)

def write(root,path,value,receipt=False):
    payload=(json.dumps(json_value(value),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode()
    safe_bytes(root,path,payload,receipt)

def seal(root):
    write(root,root/'artifact_hashes.json',{str(p.relative_to(root)):sha(p)
        for p in sorted(root.rglob('*')) if p.is_file() and p!=root/'artifact_hashes.json'})

def mesh_hash(mesh):
    return {name:array_hash(np.asarray(getattr(mesh,name))) for name in ('vertices','triangles')}

def mesh_from_arrays(vertices,triangles):
    mesh=o3d.geometry.TriangleMesh()
    mesh.vertices=o3d.utility.Vector3dVector(vertices)
    mesh.triangles=o3d.utility.Vector3iVector(triangles)
    return mesh

def save_mesh(root,path,mesh):
    stream=io.BytesIO()
    np.savez_compressed(stream,vertices=np.asarray(mesh.vertices),triangles=np.asarray(mesh.triangles))
    safe_bytes(root,path,stream.getvalue())

def verify_inventory(root):
    inventory=read(root/'artifact_hashes.json')
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p!=root/'artifact_hashes.json'}
    if actual!=set(inventory):raise ValueError('sealed file set differs: '+str(root))
    for name,h in inventory.items():
        if sha(root/name)!=h:raise ValueError('sealed artifact changed: '+str(root/name))

def check_inputs():
    for root in (PAID,OLD):
        manifest=read(root/'manifest.json')
        if manifest['status']!='complete' or read(root/'result.json')['status']!='complete':
            raise ValueError('completed original evidence required')
        verify_inventory(root)
        if sha(root/'sources.zip')!=manifest['source_archive_sha256'] or versions()!=manifest['versions']:
            raise ValueError('original source archive or dependency versions changed')
    old=read(OLD/'manifest.json')
    if len(old['source_sha256'])!=147:raise ValueError('original147 explicit frozen source list required')
    for name,h in old['source_sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError('old frozen source changed: '+name)
    if sha(BACKEND)!=EXPECTED_BACKEND_SHA:raise ValueError('V25 prototype differs from authorised frozen source')
    source=read(PAID/'case_00/result.json')
    if source['paid_actions']!=234 or source['raw_frames']!=235 or len(source['trace'])!=235:
        raise ValueError('case00 paid/packet counts differ')
    if [row['action_id'] for row in source['trace']]!=list(range(235)):
        raise ValueError('case00 trace must contain consecutive IDs0..234')
    if {p.name for p in (PAID/'case_00/packets').iterdir()}!={f'{i:04d}.npz' for i in range(235)}:
        raise ValueError('case00 packet file sequence differs')
    verify=read(PAID/'case_00/verification.json')
    if (verify['status']!='passed' or verify['raw_packets_verified']!=235 or verify['paid_actions']!=234
            or verify['physical_process_id']==verify['replay_process_id']):
        raise ValueError('original independent case00 verification invalid')
    return old

def check_frozen(root):
    m=read(root/'manifest.json')
    if m['config']!=CONFIG:raise ValueError('configuration changed')
    for name,h in m['source_sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError('new frozen source changed: '+name)
    for name,h in m['input_sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError('new frozen input changed: '+name)
    if sha(root/'sources.zip')!=m['source_archive_sha256'] or versions()!=m['versions']:
        raise ValueError('source archive/dependency versions changed')
    check_inputs()
    old=read(OLD/'case_00.json')
    expected=[{key:row[key] for key in ('observed_slot','seed','ground','geometry_arrays','mesh_sha256','mesh_files')}
              for row in old['instances']]
    if (m['seed_and_shared_geometry_checks']!=expected
            or m['input_packet_sequence_sha256']!=old['input_packet_sequence_sha256']
            or m['case']!=dict(index=0,parent=old['parent'],assignment=old['assignment'],packets=235,paid_actions=234)):
        raise ValueError('prepared public seed/history metadata differs from original record')
    return m

def prepare(root):
    old_manifest=check_inputs()
    old=read(OLD/'case_00.json')
    source=read(PAID/'case_00/result.json')
    if old['index']!=0 or old['saved_packets_processed']!=235 or len(old['instances'])!=2:
        raise ValueError('exactly two original case00 instances required')
    seeds=[row['seed'] for row in old['instances']]
    if [row['action_id'] for row in seeds]!=[24,111] or [row['pixel'] for row in seeds]!=[[32,57],[33,3]]:
        raise ValueError('original actual observed seed receipts differ')
    if shutil.disk_usage(ROOT).free < CONFIG['free_reserve_bytes']+CONFIG['output_limit_bytes']:
        raise OSError('prepare needs4MiB output plus64MiB reserve')
    root.mkdir(parents=True,exist_ok=False)
    new=[BACKEND,Path(__file__).resolve(),PROTOCOL]
    sources=dict(old_manifest['source_sha256']);sources.update({str(p.relative_to(ROOT)):sha(p) for p in new})
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in new:archive.writestr(str(p.relative_to(ROOT)),p.read_bytes())
    safe_bytes(root,root/'sources.zip',stream.getvalue())
    inputs=[base/name for base in (PAID,OLD) for name in ('manifest.json','result.json','artifact_hashes.json','sources.zip')]
    inputs += [OLD/'case_00.json',PAID/'case_00/result.json',PAID/'case_00/verification.json',PAID/'case_00/final_mesh.npz']
    summary=[{key:row[key] for key in ('observed_slot','seed','ground','geometry_arrays','mesh_sha256','mesh_files')}
             for row in old['instances']]
    manifest=dict(status='prepared',config=CONFIG,backend_version=VERSION,source_sha256=sources,
        source_archive_sha256=sha(root/'sources.zip'),versions=versions(),
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},
        old_source_archives_referenced_not_copied=True,seed_and_shared_geometry_checks=summary,
        case=dict(index=0,parent=old['parent'],assignment=old['assignment'],packets=235,paid_actions=234),
        input_packet_sequence_sha256=old['input_packet_sequence_sha256'],
        original_raw_array_sha256=source['final_mesh_sha256'],
        preparation_observe_calls=0,preparation_snapshots=0,preparation_quality_evaluations=0,
        full_architecture_or_semantic_efficacy_proven=False)
    write(root,root/'manifest.json',manifest);check_frozen(root)
    print('PREPARED case00 only;235 packets,2 snapshots;0 new sensor/TSDF; awaiting single run',flush=True)

def verify_seed_pixel(packet,seed):
    if packet.sha256()!=seed['packet_sha256'] or packet.action_id!=seed['action_id']:
        raise ValueError('first observed seed packet differs')
    f=packet.frame;rr,cc=seed['pixel'];depth=float(f.depth_m[rr,cc])
    colors=np.array([[40,100,220],[220,60,40]],dtype=np.uint8)
    binary=np.any(np.all(f.color_rgb[:,:,None,:]==colors[None,None,:,:],axis=3),axis=2)
    valid=binary&(f.depth_m>.15)&np.isfinite(f.depth_m)
    if not valid[rr,cc]:raise ValueError('original seed is not a real binary marker depth hit')
    groups,count=label(valid,np.ones((3,3),bool))
    if int(np.sum(groups==groups[rr,cc]))!=seed['component_pixels']:
        raise ValueError('original seed connected component size changed')
    local=np.array([(cc-f.intrinsic[0,2])*depth/f.intrinsic[0,0],
        (rr-f.intrinsic[1,2])*depth/f.intrinsic[1,1],depth])
    xyz=local@f.world_from_camera[:3,:3].T+f.world_from_camera[:3,3]
    if not np.allclose(xyz,seed['observed_seed_xyz'],atol=1e-10,rtol=0):
        raise ValueError('original seed differs from actual pixel backprojection')

def progress(root,phase):
    PROGRESS['phase']=phase
    write(root,root/'progress.json',dict(process_id=os.getpid(),**PROGRESS))

def execute_worker(root,token):
    m=read(root/'manifest.json');claim=read(root/'execution_claim.json')
    if m['status']!='running' or claim['token']!=token or claim['parent_process_id']!=os.getppid():
        raise ValueError('worker must belong to the unique authorised parent claim')
    m=check_frozen(root)
    start=perf_counter();progress(root,'observe_saved_history')
    source=read(PAID/'case_00/result.json')
    expected=m['seed_and_shared_geometry_checks']
    backends=[ObservedShapeBackendV25(),ObservedShapeBackendV25()]
    history=[]
    for action_id,row in enumerate(source['trace']):
        packet=load_packet(PAID/'case_00/packets'/f'{action_id:04d}.npz')
        if packet.sha256()!=row['packet_sha256'] or packet.action_id!=action_id:
            raise ValueError('saved packet SHA/order mismatch')
        for slot,backend in enumerate(backends):
            seed=expected[slot]['seed'];current=None
            if action_id==seed['action_id']:
                verify_seed_pixel(packet,seed);current=seed['observed_seed_xyz']
            f=packet.frame
            if not backend.observe(f.depth_m,f.intrinsic,f.world_from_camera,
                observation_id=action_id,observed_seed_xyz=current):
                raise ValueError('unexpected duplicate history frame')
            PROGRESS['observe_calls']+=1
        history.append(packet.sha256());PROGRESS['last_packet_id']=action_id
        if action_id%20==0 or action_id==234:progress(root,'observe_saved_history')
    if len(history)!=235 or PROGRESS['observe_calls']!=470 or digest(history)!=m['input_packet_sequence_sha256']:
        raise ValueError('complete original history sequence not reproduced')
    with np.load(PAID/'case_00/final_mesh.npz',allow_pickle=False) as data:
        raw_hash={name:array_hash(data[name]) for name in ('vertices','triangles','vertex_colors')}
        if raw_hash!=m['original_raw_array_sha256']:raise ValueError('raw TSDF arrays changed')
        raw=mesh_from_arrays(data['vertices'].copy(),data['triangles'].copy())
    capacity(root,4096);(root/'meshes').mkdir()
    snapshots=[];rows=[]
    for slot,backend in enumerate(backends):
        progress(root,f'snapshot_slot{slot}')
        snapshot=backend.snapshot(raw_mesh=raw);PROGRESS['snapshots']+=1
        geometry={key:dict(count=len(snapshot[key]),sha256=array_hash(snapshot[key])) for key in
            ('measured_points_xyz','ground_points_xyz','cleaned_points_xyz','unassigned_points_xyz')}
        if json_value(snapshot['ground_plane'])!=expected[slot]['ground'] or geometry!=expected[slot]['geometry_arrays']:
            raise ValueError('shared ground/input point/component evidence changed')
        meshes={key:mesh_hash(snapshot[key]) for key in ('observed_mesh','inferred_mesh','completed_mesh')}
        if meshes['observed_mesh']!=expected[slot]['mesh_sha256']['observed_mesh']:
            raise ValueError('shared observed mesh changed')
        old_path=OLD/expected[slot]['mesh_files']['observed_mesh']
        with np.load(old_path,allow_pickle=False) as saved:
            saved_hash={name:array_hash(saved[name]) for name in ('vertices','triangles')}
        if saved_hash!=meshes['observed_mesh']:raise ValueError('referenced observed mesh arrays differ')
        no_inference=len(snapshot['inferred_mesh'].triangles)==0
        equality=meshes['observed_mesh']==meshes['completed_mesh']
        if no_inference and not equality:raise ValueError('zero inference must equal measured geometry')
        if not snapshot['completion']['accepted'] and not no_inference:
            raise ValueError('rejected enclosure supplied inferred geometry')
        path=root/'meshes'/f'slot_{slot}_inferred_mesh.npz';save_mesh(root,path,snapshot['inferred_mesh'])
        rows.append(dict(observed_slot=slot,seed=expected[slot]['seed'],observed_frames=snapshot['observed_frames'],
            ground=snapshot['ground_plane'],geometry_arrays=geometry,shared_input_and_measured_output_equal_old=True,
            completion=snapshot['completion'],mesh_sha256=meshes,
            mesh_sizes={key:dict(vertices=len(snapshot[key].vertices),triangles=len(snapshot[key].triangles))
                for key in ('observed_mesh','inferred_mesh','completed_mesh')},
            observed_mesh_reference=dict(path=str(old_path.relative_to(ROOT)),sha256=sha(old_path),array_sha256=saved_hash),
            inferred_mesh_file=str(path.relative_to(root)),completed_composition=['referenced_observed_mesh','inferred_mesh_file'],
            no_new_inferred_triangles=no_inference,completed_equals_measured=equality,inferred_surfaces_are_measured=False))
        snapshots.append({key:snapshot[key] for key in ('observed_mesh','inferred_mesh','completed_mesh')})
        del snapshot
        print('SNAPSHOT',slot,rows[-1]['completion']['accepted'],rows[-1]['completion']['reason'],flush=True)
    if PROGRESS['snapshots']!=2:raise ValueError('exactly two snapshots required before evaluation')
    del backends
    write(root,root/'snapshot_checkpoint.json',dict(instances=rows,all_snapshots_complete_before_reference_loading=True,
        saved_packets=235,observe_calls=470,new_tsdf_fusions=0))
    # Evaluation classes and GT world are first imported/constructed AFTER BOTH snapshots.
    progress(root,'load_evaluation_only_reference')
    from scripts.replay_facility_choice_shape_v24 import ReferenceWorld,fixed_seed_association
    from utils.facility_outline_v23 import OutlineEvaluatorV23
    old=read(OLD/'case_00.json');paid_manifest=read(PAID/'manifest.json');case=m['case']
    reference=ReferenceWorld(case['parent'],case['assignment'],
        paid_manifest['config']['sensor_model'],paid_manifest['config']['noise_seed'])
    evaluator=OutlineEvaluatorV23.from_world(reference,**CONFIG['outline'])
    if evaluator.reference_signature!=old['reference_signature']:
        raise ValueError('frozen V23 outline reference signature differs')
    association=fixed_seed_association([row['seed'] for row in rows],reference.objects,
        old['fixed_seed_association']['observed_track_count'])
    for key in ('rows','missing_reference_ids','duplicate_reference_ids','seed_association_gate_passed'):
        if association[key]!=old['fixed_seed_association'][key]:raise ValueError('fixed evaluation association changed')
    arguments=dict(coverage=source['final_coverage_2d'],returned=source['returned_to_anchor'],
        collisions=source['collisions'],failed=False,paid_actions=234,budget=234)
    submitted=dict(raw=raw)
    for name,key in (('measured','observed_mesh'),('inferred','inferred_mesh'),('completed','completed_mesh')):
        joined=o3d.geometry.TriangleMesh()
        for snapshot in snapshots:joined+=snapshot[key]
        submitted[name]=joined
    union_hashes={name:mesh_hash(mesh) for name,mesh in submitted.items()}
    for name in ('raw','measured'):
        if union_hashes[name]!=old['representation_mesh_sha256'][name]:
            raise ValueError('raw/measured union no longer identical; cannot reuse original score')
    scores={name:old['representations'][name] for name in ('raw','measured')}
    for name in ('inferred','completed'):
        progress(root,f'evaluate_{name}')
        scores[name]=evaluator.evaluate(submitted[name],**arguments);PROGRESS['quality_evaluations']+=1
    for slot,row in enumerate(rows):
        progress(root,f'evaluate_slot{slot}_completed')
        row['completed_per_reference_window']=evaluator.evaluate(snapshots[slot]['completed_mesh'],**arguments)['instances']
        PROGRESS['quality_evaluations']+=1
        row['fixed_seed_association']=association['rows'][slot]
    if all(row['no_new_inferred_triangles'] for row in rows) and scores['completed']!=scores['measured']:
        raise ValueError('zero inference but completed score differs from unchanged measured')
    duplicate=rows[0]['geometry_arrays']['cleaned_points_xyz']['count']>0 and (
        rows[0]['geometry_arrays']['cleaned_points_xyz']==rows[1]['geometry_arrays']['cleaned_points_xyz'])
    nonempty=all(row['geometry_arrays']['cleaned_points_xyz']['count']>0 and row['mesh_sizes']['observed_mesh']['triangles']>0 for row in rows)
    comparisons=[]
    for name in CONFIG['output_representations']:
        comparisons.append(dict(representation=name,old_v24=old['representations'][name]['05cm'],
            new_v25=scores[name]['05cm'],delta_quality=scores[name]['05cm']['outline_macro_quality']-old['representations'][name]['05cm']['outline_macro_quality'],
            delta_joint=scores[name]['05cm']['joint_outline']-old['representations'][name]['05cm']['joint_outline']))
    if PROGRESS['quality_evaluations']!=4:raise ValueError('exactly four new frozen evaluator calls required')
    result=dict(status='complete',backend_version=VERSION,case=case,instances=rows,representations=scores,
        representation_mesh_sha256=union_hashes,score_provenance={name:'reused exact unchanged old V24 mesh score' if name in ('raw','measured') else 'new frozen V23 evaluator call' for name in scores},
        old_v24_comparison=comparisons,old_v24_completion=[row['completion'] for row in old['instances']],
        old_v24_case_reference=dict(path=str((OLD/'case_00.json').relative_to(ROOT)),sha256=sha(OLD/'case_00.json')),
        raw_source=dict(path=str((PAID/'case_00/final_mesh.npz').relative_to(ROOT)),sha256=sha(PAID/'case_00/final_mesh.npz'),array_sha256=raw_hash),
        fixed_seed_association=association,duplicate_seeded_components=duplicate,all_observed_instances_nonempty=nonempty,
        minimum_instance_separation_gate_passed=nonempty and not duplicate,
        observed_instance_seed_and_separation_gate_passed=association['seed_association_gate_passed'] and nonempty and not duplicate,
        saved_packets_processed=235,backend_observe_calls=470,backend_snapshots=2,new_quality_evaluations=4,
        input_packet_sequence_sha256=digest(history),reference_signature=evaluator.reference_signature,
        all_snapshots_complete_before_reference_loading=True,shared_geometry_inputs_and_measured_outputs_equal_old=True,
        coverage_2d=arguments['coverage'],returned=arguments['returned'],collisions=arguments['collisions'],
        elapsed_s=perf_counter()-start,worker_process_id=os.getpid(),old_backend_reruns=0,new_physical_actions=0,
        new_sensor_frames=0,new_tsdf_fusions=0,independent_confirmation=False,semantic_efficacy_proven=False,
        full_architecture_efficacy_proven=False,prior_completion_is_not_measured_accuracy=True,
        scope='case00 previously inspected paid prefix development diagnosis; not a new trajectory, choice task or generalization test')
    progress(root,'final_integrity_checks');check_frozen(root)
    write(root,root/'result.json',result)
    progress(root,'complete')
    print('WORKER COMPLETE',dict(Q_completed=scores['completed']['05cm']['outline_macro_quality'],
        any_inference=any(row['completion']['accepted'] for row in rows),elapsed_s=result['elapsed_s']),flush=True)

def failure(root,phase,error):
    if not root.exists():return
    receipt=root/f'failure_{phase}.json'
    info=dict(status='failed',phase=phase,error=repr(error),traceback=traceback.format_exc()[-8000:],
        process_id=os.getpid(),progress=read(root/'progress.json') if (root/'progress.json').exists() else PROGRESS)
    if not receipt.exists():write(root,receipt,info,receipt=True)
    if (root/'manifest.json').exists():
        m=read(root/'manifest.json');m.update(status='failed',error=repr(error));write(root,root/'manifest.json',m,receipt=True)
    # The small failed inventory uses the separately reserved receipt allowance.
    write(root,root/'artifact_hashes.json',{str(p.relative_to(root)):sha(p)
        for p in sorted(root.rglob('*')) if p.is_file() and p!=root/'artifact_hashes.json'},receipt=True)

def supervise(root):
    m=check_frozen(root)
    if m['status']!='prepared':raise ValueError('only prepared unstarted output is writable')
    token=uuid.uuid4().hex
    claim=dict(token=token,parent_process_id=os.getpid(),maximum_worker_seconds=CONFIG['maximum_worker_seconds'])
    payload=(json.dumps(claim,separators=(',',':'))+'\n').encode();capacity(root,len(payload))
    descriptor=os.open(root/'execution_claim.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    # Successful exclusive creation is the only point where this invocation owns writes.
    try:
        with os.fdopen(descriptor,'wb') as stream:stream.write(payload)
        m['status']='running';m['parent_process_id']=os.getpid();write(root,root/'manifest.json',m)
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'--output',str(root),'--worker','--token',token]
        finished=subprocess.run(command,timeout=CONFIG['maximum_worker_seconds'],check=False)
        if finished.returncode!=0:raise RuntimeError(f'unique worker exited{finished.returncode}')
        result=read(root/'result.json')
        if result['status']!='complete':raise ValueError('worker did not produce complete result')
        m=check_frozen(root);m.update(status='complete',worker_process_id=result['worker_process_id'])
        write(root,root/'manifest.json',m);seal(root)
        if total_bytes(root)>CONFIG['output_limit_bytes'] or shutil.disk_usage(root).free<CONFIG['free_reserve_bytes']:
            raise OSError('final actual resource boundary violated')
        print('COMPLETE',dict(output_bytes=total_bytes(root),free_bytes=shutil.disk_usage(root).free),flush=True)
    except Exception as error:
        failure(root,'run',error)
        raise

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--run',action='store_true')
    mode.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--token',help=argparse.SUPPRESS)
    args=parser.parse_args();root=args.output.resolve()
    if args.prepare:
        if root.exists():raise ValueError('new output root required; existing evidence untouched')
        try:prepare(root)
        except Exception as error:
            failure(root,'prepare',error);raise
    elif args.run:
        # Entry refusal and failed ownership checks do not mutate existing roots.
        if read(root/'manifest.json')['status']!='prepared' or (root/'execution_claim.json').exists():
            raise ValueError('only prepared unclaimed output may run once')
        supervise(root)
    else:
        claim=read(root/'execution_claim.json')
        if not args.token or claim['token']!=args.token or claim['parent_process_id']!=os.getppid():
            raise ValueError('worker invocation requires its authorised parent claim')
        execute_worker(root,args.token)

if __name__=='__main__':main()
