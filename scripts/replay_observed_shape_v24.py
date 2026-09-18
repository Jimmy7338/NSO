#!/usr/bin/env python3
"""New-version development processing of saved V23 depth, without new sensing."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse
import hashlib
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
import scipy
import shapely
from nso.observed_shape_v24 import ObservedShapeBackendV24, VERSION, mesh_from_arrays
from nso.decision_replay_v13 import load_packet, array_hash
from nso.cpu_sensor_contract_v10 import json_value
from env.virtual3d_inspection_v4 import read_inspection_markers_rgb
from env.facility_shape_probe_v23 import ShapeProbeWorldV23
from utils.facility_outline_v23 import OutlineEvaluatorV23

SOURCE=ROOT/'audit_results/facility_shape_v23_probe_20260915'
DEFAULT=ROOT/'audit_results/observed_shape_v24_development_20260915'
PROTOCOL=ROOT/'docs/research/V24_SAVED_DEPTH_SHAPE_PROTOCOL_20260915.md'
STAGES={'coarse':23,'extra':73,'returned':96}


class ReferenceWorld(ShapeProbeWorldV23):
    def sense(self): raise AssertionError('no new sensor observation permitted')
    def scan(self): raise AssertionError('no new lidar observation permitted')
    def step(self,action): raise AssertionError('no new physical action permitted')


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write(path,value):
    temp=path.with_name(path.name+'.tmp')
    with temp.open('x') as f:
        f.write(json.dumps(json_value(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def versions():
    return dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
                open3d=o3d.__version__,shapely=shapely.__version__,geos=shapely.geos_version_string)


def check_original():
    inventory=read(SOURCE/'artifact_hashes.json')
    actual={str(p.relative_to(SOURCE)) for p in SOURCE.rglob('*') if p.is_file() and p!=SOURCE/'artifact_hashes.json'}
    if set(inventory)!=actual: raise ValueError('old evidence file set changed')
    for name,h in inventory.items():
        if sha(SOURCE/name)!=h: raise ValueError('old evidence hash changed: '+name)
    manifest=read(SOURCE/'manifest.json')
    for name,h in manifest['source_sha256'].items():
        if sha(ROOT/name)!=h: raise ValueError('old source changed: '+name)
    return manifest


def prepare(root):
    check_original()
    preflight=ROOT/'audit_results/observed_shape_v24_preflight_20260915'
    checks=read(preflight/'run.json')
    if checks['status']!='passed' or checks['exit_code']!=0 or not checks['sources_unchanged']:
        raise ValueError('passing independent preflight required')
    for name,h in checks['source_sha256_after'].items():
        if sha(ROOT/name)!=h: raise ValueError('source changed after preflight')
    for name,h in read(preflight/'artifact_hashes.json').items():
        if sha(preflight/name)!=h['sha256'] or (preflight/name).stat().st_size!=h['bytes']:
            raise ValueError('preflight evidence changed')
    root.mkdir(exist_ok=False,parents=True)
    sources=[Path(__file__).resolve(),ROOT/'nso/observed_shape_v24.py',
             ROOT/'tests/virtual3d/test_observed_shape_v24.py',PROTOCOL]
    with zipfile.ZipFile(root/'sources.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for p in sources: archive.write(p,str(p.relative_to(ROOT)))
    write(root/'manifest.json',dict(status='prepared',version=VERSION,
        source_root=str(SOURCE),old_inventory_sha256=sha(SOURCE/'artifact_hashes.json'),
        sources={str(p.relative_to(ROOT)):sha(p) for p in sources},
        source_archive_sha256=sha(root/'sources.zip'),versions=versions(),
        preflight_result_sha256=sha(preflight/'run.json'),stages=STAGES,
        primary_representation='observed_mesh + explicitly inferred_mesh',
        new_physical_actions=0,new_sensor_frames=0,training=False,
        seed_source='first valid observed RGB binary marker hit; class value discarded'))
    print('Prepared saved-depth development: no new sensor or action.',flush=True)


def check_frozen(root):
    m=read(root/'manifest.json')
    if versions()!=m['versions']: raise ValueError('environment changed')
    for name,h in m['sources'].items():
        if sha(ROOT/name)!=h: raise ValueError('new frozen source changed: '+name)
    if sha(root/'sources.zip')!=m['source_archive_sha256']: raise ValueError('source archive changed')
    if sha(ROOT/'audit_results/observed_shape_v24_preflight_20260915/run.json')!=m['preflight_result_sha256']:
        raise ValueError('preflight receipt changed')
    if sha(SOURCE/'artifact_hashes.json')!=m['old_inventory_sha256']: raise ValueError('old inventory changed')
    check_original()
    return m


def seed_from_rgb(frame):
    mask=(read_inspection_markers_rgb(frame.color_rgb)>0)&(frame.depth_m>0)
    rows,cols=np.nonzero(mask)
    if not len(rows): return None
    z=frame.depth_m[rows,cols]
    pixel=np.stack([cols,rows,np.ones(len(rows))],axis=1)
    xyz=(pixel@np.linalg.inv(frame.intrinsic).T*z[:,None])@frame.world_from_camera[:3,:3].T+frame.world_from_camera[:3,3]
    return xyz[np.argmin(np.linalg.norm(xyz-np.median(xyz,axis=0),axis=1))]


def mesh_digest(mesh):
    return {name:array_hash(np.asarray(getattr(mesh,name))) for name in ('vertices','triangles')}


def execute(root,verify=False):
    manifest=check_frozen(root)
    if (verify and manifest['status']!='complete') or (not verify and manifest['status']!='prepared'):
        raise ValueError('only prepared execution or complete independent verification permitted')
    if shutil.disk_usage(root).free < 10*1024**2: raise OSError('10MiB reserve required for this bounded development')
    original=read(SOURCE/'manifest.json')
    result=dict(status='complete',version=VERSION,stage_results=[],seed_receipts=[],
        saved_frames_processed=0,new_physical_actions=0,new_sensor_frames=0,
        new_tsdf_fusions=0,semantic_policy=False,full_architecture_efficacy_proven=False,
        source_data_previously_viewed=True,scope='saved-depth common-backend development, not independent scene confirmation')
    if not verify:
        manifest['status']='running';write(root/'manifest.json',manifest)
        (root/'meshes').mkdir()
    else:
        if read(root/'timing.json')['process_id']==os.getpid(): raise ValueError('fresh process required')
        if (root/'verification.json').exists(): raise ValueError('verification already complete')
        for name,h in read(root/'artifact_hashes.json').items():
            if sha(root/name)!=h: raise ValueError('development artifact changed: '+name)
    t0=perf_counter()
    for case in original['cases']:
        index,kind=case['index'],case['kind']
        folder=SOURCE/f'case_{index:02d}'
        old=read(folder/'result.json')
        world=ReferenceWorld(kind)
        evaluator=OutlineEvaluatorV23.from_world(world,**original['config']['outline'])
        if evaluator.reference_signature!=case['outline_reference_signature']: raise ValueError('reference differs')
        backend=ObservedShapeBackendV24();seeded=False
        by_action={r['action_id']:r for r in old['checkpoints']}
        for action_id in range(97):
            packet=load_packet(folder/'packets'/f'{action_id:04d}.npz')
            if packet.sha256()!=old['trace'][action_id]['packet_sha256']: raise ValueError('saved packet differs')
            seed=None if seeded else seed_from_rgb(packet.frame)
            if seed is not None:
                result['seed_receipts'].append(dict(kind=kind,action_id=action_id,observed_seed_xyz=seed.tolist()))
                seeded=True
            f=packet.frame
            backend.observe(f.depth_m,f.intrinsic,f.world_from_camera,
                observation_id=action_id,observed_seed_xyz=seed)
            result['saved_frames_processed']+=1
            if action_id not in STAGES.values(): continue
            stage=next(k for k,v in STAGES.items() if v==action_id)
            with np.load(folder/'meshes'/f'{stage}.npz',allow_pickle=False) as arrays:
                raw=mesh_from_arrays(arrays['vertices'],arrays['triangles'])
            snapshot=backend.snapshot(raw_mesh=raw)
            original_stage=by_action[action_id]
            fields=original_stage['outline']
            arguments=dict(coverage=fields['coverage_2d'],returned=fields['returned'],
                collisions=fields['collisions'],failed=fields['failed'],paid_actions=action_id,budget=96)
            row=dict(kind=kind,action_id=action_id,stage=stage,ground=snapshot['ground_plane'],
                completion=snapshot['completion'],raw_outline=fields,
                geometry_arrays={key:dict(count=len(snapshot[key]),sha256=array_hash(snapshot[key])) for key in
                  ('measured_points_xyz','ground_points_xyz','cleaned_points_xyz','unassigned_points_xyz')},
                observed_outline=evaluator.evaluate(snapshot['observed_mesh'],**arguments),
                primary_outline=evaluator.evaluate(snapshot['completed_mesh'],**arguments),
                meshes={key:mesh_digest(snapshot[key]) for key in ('observed_mesh','inferred_mesh','completed_mesh')})
            if not verify:
                for key in ('observed_mesh','inferred_mesh','completed_mesh'):
                    mesh=snapshot[key]
                    with (root/'meshes'/f'{kind}_{stage}_{key}.npz').open('xb') as stream:
                        np.savez_compressed(stream,vertices=np.asarray(mesh.vertices),triangles=np.asarray(mesh.triangles))
            else:
                for key in ('observed_mesh','inferred_mesh','completed_mesh'):
                    with np.load(root/'meshes'/f'{kind}_{stage}_{key}.npz',allow_pickle=False) as arrays:
                        stored=mesh_from_arrays(arrays['vertices'],arrays['triangles'])
                    if mesh_digest(stored)!=row['meshes'][key]: raise ValueError('saved output mesh differs')
            result['stage_results'].append(json_value(row))
            print(f'{kind} {stage} primaryQ={row["primary_outline"]["05cm"]["outline_macro_quality"]:.6f} inferred={row["completion"]["accepted"]} reason={row["completion"]["reason"]}',flush=True)
    simple=next(r for r in result['stage_results'] if r['kind']=='simple' and r['stage']=='coarse')
    complex_row=next(r for r in result['stage_results'] if r['kind']=='complex' and r['stage']=='coarse')
    result['coarse_nonsemantic_geometry_identical']=(simple['geometry_arrays']==complex_row['geometry_arrays']
        and simple['meshes']==complex_row['meshes'] and simple['completion']==complex_row['completion'])
    if not result['coarse_nonsemantic_geometry_identical']: raise ValueError('class-blind coarse output differs')
    check_frozen(root)
    if verify:
        if result!=read(root/'result.json'): raise ValueError('independent saved-data processing differs')
        write(root/'verification.json',dict(status='passed',independent_process=True,
            original_process_id=read(root/'timing.json')['process_id'],verification_process_id=os.getpid(),
            saved_frames_processed=194,stage_meshes_verified=18,stage_results_verified=6,
            new_physical_actions=0,new_sensor_frames=0,new_tsdf_fusions=0))
    else:
        write(root/'result.json',result)
        write(root/'timing.json',dict(process_id=os.getpid(),elapsed_s=perf_counter()-t0))
        manifest['status']='complete';write(root/'manifest.json',manifest)
    inventory={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p!=root/'artifact_hashes.json'}
    write(root/'artifact_hashes.json',inventory)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=DEFAULT)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true');group.add_argument('--run',action='store_true');group.add_argument('--verify',action='store_true')
    args=parser.parse_args();root=args.output.resolve()
    try:
        if args.prepare: prepare(root)
        else: execute(root,args.verify)
    except Exception as error:
        if root.exists():
            path=root/('verification_failure.json' if args.verify else 'failure.json')
            if not path.exists():write(path,dict(status='failed',error=repr(error),traceback=traceback.format_exc(),process_id=os.getpid()))
            if not args.verify and (root/'manifest.json').exists():
                failed=read(root/'manifest.json');failed['status']='failed';write(root/'manifest.json',failed)
        raise


if __name__=='__main__':main()
