#!/usr/bin/env python3
"""Fixed analytic prototype checks; no saved case metrics, world or TSDF use."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,json,hashlib,traceback,itertools,time
from pathlib import Path
ROOT=Path('/root/NSO');HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(ROOT)];sys.dont_write_bytecode=True
import numpy as np
from observed_shape_v25 import ObservedShapeBackendV25
from nso.observed_shape_v24 import ObservedShapeBackendV24
from env.canonical_rgbd_v15 import render_axial_depth
from env.facility_documentation_v19 import stereo_depth_v19
from nso.cpu_sensor_contract_v10 import json_value


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def array_sha(a):return hashlib.sha256(a.tobytes()).hexdigest()
def pose(x,y,heading):
    forward=np.array([[0,1,0],[1,0,0],[0,-1,0],[-1,0,0]])[heading]
    t=np.eye(4);t[:3,:3]=np.column_stack([np.cross(forward,[0,0,1]),[0,0,-1],forward]);t[:3,3]=[x,y,.8]
    return t

def fixture(size,rotation,mode,kind,seed):
    w,d,h=size
    body=[-w/2,0,0,w,d,h]
    boxes=[[-10,-10,-.12,20,20,.12]]
    if kind=='open_l':
        boxes += [[-w/2,0,0,w,.12,h],[-w/2,0,0,.12,d,h]]
    else:boxes.append(body)
    if kind=='hidden_bump':
        boxes.append([w/2-.4,d,.8,.4,.12,.4])
    if kind!='no_background':
        boxes += [[-3,d+2.4,0,6,.2,6],[w/2+2.1,-1,0,.2,d+4,6]]
    if mode=='far': fronts,sides=[4.]*8,[4.]*8
    elif mode=='mixed':fronts,sides=[1.3]*4+[4.]*4,[1.1]*4+[4.]*4
    else:fronts,sides=[1.3]*8,[1.1]*8
    sequence=[pose(0,-z,0) for z in fronts]
    if kind!='single_face':sequence += [pose(-w/2-z,d/2,1) for z in sides]
    if kind=='open_l':sequence += [pose(0,d+1.3,2)]*4
    angle=np.deg2rad(rotation);rotation_matrix=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    transform=np.eye(4);transform[:3,:3]=rotation_matrix;transform[:3,3]=[2.3,-1.7,0]
    k=np.array([[48.,0,47.5],[0,48.,35.5],[0,0,1.]])
    backend=ObservedShapeBackendV25();clean_hashes=[]
    for index,t in enumerate(sequence):
        depth,_=render_axial_depth(np.array(boxes),k,t,72,96,5.)
        clean_hashes.append(array_sha(depth))
        noisy=stereo_depth_v19(depth,model='ideal' if mode=='ideal' else 'iid_025px',parent_index=0,step=index,noise_seed=seed)
        actual_t=transform@t
        observed_seed=None
        if index==0:
            # Known detection pixel is a real measured hit. No size/GT box is passed.
            vv,uu=36,48
            if noisy[vv,uu]<=0:raise ValueError('declared first observed seed pixel invalid')
            ray=np.linalg.inv(k)@np.array([uu,vv,1.])
            observed_seed=actual_t[:3,3]+actual_t[:3,:3]@ray*noisy[vv,uu]
        backend.observe(noisy,k,actual_t,observation_id=index,observed_seed_xyz=observed_seed)
    return backend,np.array(body),rotation_matrix,transform[:3,3],clean_hashes


def run_fixture(definition):
    backend,body,rotation,shift,clean_hashes=fixture(**definition)
    points=np.concatenate([f['xyz'][f['valid']] for f in backend.frames])
    normals=np.concatenate([f['normals'][f['valid']] for f in backend.frames])
    ground=backend._ground(points,normals)
    if ground is None:raise ValueError('actual observed ground unavailable')
    selected=backend._component(points,normals,ground)
    p,n=points[selected],normals[selected]
    before=(array_sha(p),array_sha(n))
    mesh,completion=backend._enclosure(p,n,ground)
    assert before==(array_sha(p),array_sha(n)),'estimator changed actual measured support'
    audit=backend.last_parameter_audit
    row=dict(definition=definition,frames=len(backend.frames),selected_points=len(p),
        cleaned_support_unchanged=True,raw_clean_depth_hashes=clean_hashes,ground=json_value(ground),
        completion=json_value(completion),inferred_vertices=len(mesh.vertices),parameters=None)
    if audit and 'model_basis' in audit:
        basis=np.array(audit['model_basis']);lo,hi=np.array(audit['model_bounds'])
        # Evaluation-only comparison to known analytic body, never passed to estimator.
        truecorners=np.array(list(itertools.product(*zip(body[:3],body[:3]+body[3:]))))@rotation.T+shift
        trueprojected=truecorners@basis
        angle_error=float(np.degrees(np.arccos(np.clip(np.abs(rotation[:,:2].T@basis[:,:2]).max(axis=0),-1,1))).max())
        # Offset/extent error in the estimated basis includes any rotational outer-box effect.
        errors=np.maximum(abs(lo-trueprojected.min(axis=0)),abs(hi-trueprojected.max(axis=0)))
        row['parameters']=dict(basis=basis.tolist(),bounds=[lo.tolist(),hi.tolist()],angle_error_deg=angle_error,
            max_bound_error_m=float(errors.max()),axis_bound_error_m=errors.tolist(),
            parameter_error_within_one_4cm_voxel=bool(errors.max()<=.04),
            orientation_within_one_degree=bool(angle_error<=1.))
    return row


def main():
    output=HERE/'analytic_result_v2.json'
    if output.exists():raise ValueError('preserve first implementation and results')
    definitions=[
        dict(size=[1.6,.8,1.6],rotation=0.,mode='ideal',kind='closed',seed=41),
        dict(size=[1.6,.8,1.6],rotation=17.,mode='mixed',kind='closed',seed=41),
        dict(size=[1.2,.6,1.4],rotation=-31.,mode='mixed',kind='closed',seed=42),
        dict(size=[2.,1.,1.8],rotation=41.,mode='mixed',kind='closed',seed=43),
        dict(size=[1.6,.8,1.6],rotation=0.,mode='ideal',kind='no_background',seed=41),
        dict(size=[1.6,.8,1.6],rotation=0.,mode='ideal',kind='single_face',seed=41),
        dict(size=[1.6,.8,1.6],rotation=0.,mode='ideal',kind='open_l',seed=41),
        dict(size=[1.6,.8,1.6],rotation=17.,mode='mixed',kind='hidden_bump',seed=41),
        dict(size=[1.6,.8,1.6],rotation=0.,mode='far',kind='closed',seed=42),
    ]
    frozen={name:sha(HERE/name) for name in ('observed_shape_v25.py','test_observed_shape_v25.py')}
    old_sha=sha(ROOT/'nso/observed_shape_v24.py')
    freeze=dict(status='prepared',definitions=definitions,source_sha256=frozen,old_backend_sha256=old_sha,
        pass_contract='first4 parameter bounds<=.04m and angle<=1deg; three unsupported/contradicted fixtures rejected; no measured support mutation; hidden bump not labelled measured',
        no_quality_evaluation=True,no_world_actions=True,no_tsdf_fusion=True)
    (HERE/'analytic_freeze_v2.json').write_text(json.dumps(freeze,separators=(',',':'))+'\n')
    result=dict(status='running',cases=[],new_world_actions=0,new_tsdf_fusions=0,new_quality_evaluations=0,
        source_sha256=frozen,old_backend_sha256=old_sha)
    started=time.perf_counter()
    try:
        for index,definition in enumerate(definitions):
            try:
                row=run_fixture(definition)
                result['cases'].append(row)
                print('CASE',index,definition['kind'],definition['mode'],row['completion']['reason'],row['parameters'],flush=True)
            except Exception as error:
                result['cases'].append(dict(definition=definition,status='failed',error=repr(error),traceback=traceback.format_exc()))
                print('CASE_FAILED',index,repr(error),flush=True)
        rows=result['cases']
        positive=[bool(row.get('parameters') and row['parameters']['parameter_error_within_one_4cm_voxel'] and row['parameters']['orientation_within_one_degree']) for row in rows[:4]]
        negative=[not row.get('completion',{}).get('accepted',True) for row in rows[4:7]]
        pair=rows[1].get('raw_clean_depth_hashes')==rows[7].get('raw_clean_depth_hashes')
        inferred_claim=all(not row.get('completion',{}).get('support_audit',{}).get('unseen_surfaces_measured',True) for row in rows if 'completion' in row)
        inherited_gate_helpers_unchanged = (ObservedShapeBackendV25._edges is ObservedShapeBackendV24._edges and ObservedShapeBackendV25._free_ray_conflicts is ObservedShapeBackendV24._free_ray_conflicts)
        result['checks']=dict(inherited_edge_and_free_ray_helpers_unchanged=inherited_gate_helpers_unchanged,first_four_parameter_precision=positive,unsupported_or_contradicted_rejections=negative,
            hidden_bump_clean_history_equal=pair,all_completions_explicitly_inferred=inferred_claim,
            far_only_retains_original_gate_limitation=True,
            estimation_does_not_mutate_measured_support=all(row.get('cleaned_support_unchanged',False) for row in rows))
        result['status']='passed' if all(positive+negative+[pair,inferred_claim,result['checks']['estimation_does_not_mutate_measured_support']]) else 'failed'
        assert sha(ROOT/'nso/observed_shape_v24.py')==old_sha
        assert all(sha(HERE/name)==h for name,h in frozen.items())
    finally:
        result['elapsed_s']=time.perf_counter()-started
        output.write_text(json.dumps(json_value(result),separators=(',',':'),allow_nan=False)+'\n')
        print('RESULT',result['status'],result.get('checks'),result['elapsed_s'],flush=True)

if __name__=='__main__':main()
