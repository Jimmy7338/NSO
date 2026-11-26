#!/usr/bin/env python3
"""Freeze/generate independent RGB-D calibration records. No model or policy import."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
SOURCE=Path('/home/ouc/nso_robot/analyses/reproduction_20261104/simulation_publication_stage/experiments/unknown_exploration_20261104/source')
RENDERER=Path('/home/ouc/nso_robot/analyses/semantic_scene_redesign_20261119/assets/textured_raycast.py')
from spec import specification,objects,CONDITIONS,SENSOR_CONFIG,SCENE,opaque


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def resource(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def write(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module


def dependencies():
    sys.path.insert(0,str(SOURCE))
    import numpy as np
    from utils.rgbd_contract import RGBDFrame
    from build_cases import build_scene,private_truth
    return np,RGBDFrame,build_scene,private_truth,load('B9_sensor_renderer',RENDERER)


def frozen_paths():
    return [HERE/'spec.py',HERE/'build_cases.py',Path(__file__),HERE/'test_generator.py',HERE/'generator_spec.json',
        HERE/'protocol.json',RENDERER,SOURCE/'utils/rgbd_contract.py']


def verify():
    f=json.loads((HERE/'freeze.json').read_text())
    assert {str(p):sha(p)for p in frozen_paths()}==f['files_sha256']


def freeze():
    assert not (HERE/'freeze.json').exists()
    write(HERE/'generator_spec.json',specification())
    write(HERE/'freeze.json',dict(schema='B9_generator_freeze_v1',created_unix_s=time.time(),
        files_sha256={str(p):sha(p)for p in frozen_paths()},status='frozen_before_any_calibration_data_render',
        model_or_policy_executed=False))
    verify()


def render_case(record,condition,output_dir,counterfactual=False):
    np,RGBDFrame,build_scene,_,renderer_module=dependencies()
    scene,target_nodes=build_scene(record,condition)
    renderer=renderer_module.TexturedRayRenderer(scene)
    target_ids=[row['geometry_id']for row in renderer.stats if row['node']in target_nodes]
    cfg=SENSOR_CONFIG;K=renderer_module.intrinsic(cfg['width_px'],cfg['height_px'],cfg['fov_deg'])
    output_dir.mkdir(parents=True,exist_ok=False);frames=[]
    target=np.array(SCENE['front_anchor_world'],float);target[2]=1.2 if counterfactual else SCENE['look_at_height_m']
    for i,azimuth in enumerate(CONDITIONS[condition]):
        angle=np.deg2rad(azimuth);eye=np.array(SCENE['front_anchor_world'],float)
        eye[:2]+=SCENE['camera_radius_m']*np.array([np.sin(angle),-np.cos(angle)]);eye[2]=cfg['camera_height_m']
        pose=renderer_module.look_at(eye,target,up=(0,0,1))
        hit=renderer.render(K,pose,cfg['width_px'],cfg['height_px'],max_depth_m=cfg['max_depth_m'],min_depth_m=.15)
        valid=hit['depth_m']>0;visible=np.isin(hit['reference_geometry_id'],target_ids)&valid
        # The owner is a current first-hit instance mask. It is NOT derived from class.
        owners=visible.astype(np.uint16)
        labels=np.where(visible,2 if record['family']=='open' else 3,0).astype(np.uint8)
        frame=RGBDFrame(float(i),hit['depth_m'],hit['rgb'],K,pose,labels).validate()
        fp=output_dir/f'frame_{i:02d}.npz';op=output_dir/f'owner_{i:02d}.npz'
        frame.save(fp);np.savez_compressed(op,visible_owner=owners)
        assert not owners[~valid].any()
        frames.append(dict(frame=resource(fp),owner=resource(op),frame_id=f'frame-{i}',
            visible_instance_pixels=int(visible.sum()),valid_depth_pixels=int(valid.sum())))
    manifest=dict(schema='B9_sanitized_observation_sequence_v1',sensor_config=cfg,mapper_shape=SCENE['shape'],
        frames=frames,ideal_current_first_hit_instance_support=True,truth_fields_in_actor_input=False,
        limitations='Noiseless exact-pose procedural RGB-D; not a natural detector or physical calibration dataset.')
    write(output_dir/'manifest.json',manifest)
    return resource(output_dir/'manifest.json')


def generate():
    verify();assert not (HERE/'dataset_manifest.json').exists()
    np,_,_,private_truth,_=dependencies()
    private=ROOT/'evaluator_private';private.mkdir(exist_ok=False)
    obs=ROOT/'observations';obs.mkdir(exist_ok=False)
    rows=[];objs=[];start=time.monotonic()
    for record in objects():
        oid=record['object_id'];tp=private/f'{oid}.json';write(tp,private_truth(record));case_ids=[]
        for condition in CONDITIONS:
            cid=opaque(oid,condition);case_ids.append(cid)
            om=render_case(record,condition,obs/cid)
            rows.append(dict(case_id=cid,object_id=oid,split=record['split'],condition=condition,
                observation_manifest=om,truth=resource(tp)))
        objs.append(dict(object_id=oid,split=record['split'],cases=case_ids))
        if len(objs)%8==0:print(json.dumps(dict(objects_complete=len(objs),objects_total=48,frames_complete=len(objs)*12)),flush=True)
    assert len(rows)==192
    # A separate metamorphic fixture holds the complete visible input constant.
    # Its two hidden truths are not extra calibration objects or sensor trials.
    record=copy.deepcopy(objects()[0]);record.update(width_m=1.4,height_m=2.1,yaw_deg=0.,depth_m=.3,
        family='closed',shape_regime='regular_panels',panel_thickness_m=.025,door_recess_m=.012)
    fixture=[]
    for label,depth in [('a',.3),('b',1.1)]:
        record['depth_m']=depth;record['object_id']='counterfactual_'+label
        cp=ROOT/'counterfactual'/label;om=render_case(record,'front',cp,counterfactual=True)
        tp=private/f'counterfactual_{label}.json';write(tp,private_truth(record))
        fixture.append(dict(observation_manifest=om,truth=resource(tp)))
    parity={}
    for filename in ('frame_00.npz','owner_00.npz'):
        with np.load(ROOT/'counterfactual/a'/filename)as a,np.load(ROOT/'counterfactual/b'/filename)as b:
            assert a.files==b.files
            for key in a.files:
                np.testing.assert_array_equal(a[key],b[key])
                parity[filename+':'+key]=hashlib.sha256(np.ascontiguousarray(a[key]).tobytes()).hexdigest()
    write(HERE/'counterfactual_sensor_receipt.json',dict(status='exact_visible_array_invariance_passed',
        fixtures=fixture,visible_array_hashes_shared=parity,additional_independent_samples=0,
        additional_fixture_frames=2,model_posterior_invariance_not_yet_evaluated=True))
    write(HERE/'dataset_manifest.json',dict(schema='B9_private_calibration_dataset_manifest_v1',
        objects=objs,cases=rows,freeze=resource(HERE/'freeze.json'),elapsed_s=time.monotonic()-start,
        counts=dict(objects=48,calibration_objects=24,test_objects=24,cases=192,frames=576),
        counterfactual_receipt=resource(HERE/'counterfactual_sensor_receipt.json'),
        access='This index contains private truth links and splits. Never pass it to inference; pass only an observation_manifest path.',
        geometry_or_fit_or_candidate_score_executed=False))
    verify()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--generate',action='store_true');args=parser.parse_args()
    if args.freeze:freeze()
    if args.generate:generate()
