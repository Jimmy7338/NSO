#!/usr/bin/env python3
"""Read-only case00 fit diagnosis plus bounded analytic depth controls."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import sys,json,hashlib,itertools,traceback
from types import SimpleNamespace
from time import perf_counter
ROOT=Path('/root/NSO');sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import numpy as np
from nso.observed_shape_v24 import ObservedShapeBackendV24
from nso.decision_replay_v13 import load_packet,array_hash
from nso.cpu_sensor_contract_v10 import json_value,digest
from env.facility_choice_v24 import LAYOUTS_V24
from env.canonical_rgbd_v15 import render_axial_depth
from env.facility_documentation_v19 import stereo_depth_v19
SOURCE=ROOT/'audit_results/facility_choice_v24_paid_prefix_20260915'
SHAPE=ROOT/'audit_results/facility_choice_v24_shape_prefix_20260915'
OUTPUT=Path('/dev/shm/v24_free_ray_result.json')


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(path.read_text())


def fit_basis(normals):
    """Literal orientation-estimation stage from the frozen enclosure code."""
    up=np.array([0.,0.,1.])
    vertical=(np.abs(normals@up)<np.sin(np.deg2rad(20)))&(np.linalg.norm(normals,axis=1)>.5)
    ns=normals[vertical]
    ref=np.eye(3)[np.argmin(np.abs(up))]
    u=np.cross(up,ref);u/=np.linalg.norm(u);v=np.cross(up,u)
    angles=np.mod(np.arctan2(ns@v,ns@u),np.pi)
    bins=np.minimum((angles/(np.pi/36)).astype(int),35)
    hist=np.bincount(bins,minlength=36);centre=(np.argmax(hist)+.5)*np.pi/36
    distance=np.abs((angles-centre+np.pi/2)%np.pi-np.pi/2)
    fit=ns[distance<np.deg2rad(10)].copy()
    anchor=np.cos(centre)*u+np.sin(centre)*v
    fit[fit@anchor<0]*=-1
    axis=np.median(fit,axis=0);axis-=(axis@up)*up;axis/=np.linalg.norm(axis)
    other=np.cross(up,axis);basis=np.column_stack([axis,other,up])
    angle=np.degrees(np.arctan2(axis[1],axis[0]))
    return basis,dict(angle_deg=float(angle),nearest_axis_error_deg=float(abs((angle+45)%90-45)),
        mode_bin=int(np.argmax(hist)),mode_centre_deg=float(np.degrees(centre)),
        retained_normals=len(fit),vertical_normals=len(ns),histogram=hist.tolist())


class CaptureBackend(ObservedShapeBackendV24):
    def _free_ray_conflicts(self,basis,lo,hi):
        self.captured=(basis.copy(),lo.copy(),hi.copy())
        return super()._free_ray_conflicts(basis,lo,hi)


def ray_rows(frames,basis,lo,hi,detail=False):
    rows=[];conflicts=hits=0;coordinate_error=0.;axial_error=0.
    for f in frames:
        valid=f['valid'][::2,::2]
        direction=f['rays'][::2,::2][valid]@basis
        depth=f['depth'][::2,::2][valid]
        origin=f['t'][:3,3]@basis
        parallel=np.abs(direction)<1e-12;safe=np.where(parallel,1.,direction)
        ta,tb=(lo-origin)/safe,(hi-origin)/safe
        tlo,thi=np.minimum(ta,tb),np.maximum(ta,tb)
        tlo[parallel]=-np.inf;thi[parallel]=np.inf
        outside=np.any(parallel&((origin<lo)|(origin>hi)),axis=1)
        near=np.maximum(tlo.max(axis=1),0.);far=thi.min(axis=1)
        hit=~outside&(far>=near)&(far>0.)
        conflict=hit&(near<depth-(.08+.005*depth**2))
        hits+=int(hit.sum());conflicts+=int(conflict.sum())
        if detail:
            world_ray=f['rays'][::2,::2][valid];xyz=f['xyz'][::2,::2][valid]
            reconstructed=f['t'][:3,3]+world_ray*depth[:,None]
            if len(depth):
                coordinate_error=max(coordinate_error,float(np.max(np.abs(reconstructed-xyz))))
                recovered=(xyz-f['t'][:3,3])@f['t'][:3,2]
                axial_error=max(axial_error,float(np.max(np.abs(recovered-depth))))
            rows.append(dict(action_id=int(f['observation_id']),hits=int(hit.sum()),conflicts=int(conflict.sum()),
                bins=[dict(depth_min=low,depth_max=high,
                    hits=int((hit&(depth>=low)&(depth<high)).sum()),
                    conflicts=int((conflict&(depth>=low)&(depth<high)).sum()))
                    for low,high in ((0.,1.5),(1.5,2.5),(2.5,3.5),(3.5,5.01))]))
    result=dict(conflicts=conflicts,hits=hits,fraction=conflicts/max(hits,1))
    # Call the ORIGINAL function on an evaluation-only frame proxy. No GT
    # parameters are sent to the reconstruction backend or its fitting path.
    exact=ObservedShapeBackendV24._free_ray_conflicts(SimpleNamespace(frames=frames),basis,lo,hi)
    assert exact==(conflicts,hits)
    if detail:
        result.update(coordinate_roundtrip_max_m=coordinate_error,axial_roundtrip_max_m=axial_error,
            depth_bins=[dict(depth_min=rows[0]['bins'][i]['depth_min'],depth_max=rows[0]['bins'][i]['depth_max'],
                hits=sum(row['bins'][i]['hits'] for row in rows),
                conflicts=sum(row['bins'][i]['conflicts'] for row in rows)) for i in range(4)],
            highest_conflict_actions=sorted(rows,key=lambda row:row['conflicts'],reverse=True)[:8])
    return result


def corners(lo,hi):
    return np.array(list(itertools.product(*zip(lo,hi))))


def comparisons(frames,points,basis,lo,hi,truth_lo,truth_hi,ground_z):
    gt_corners=corners(truth_lo,truth_hi)
    gt_projected=gt_corners@basis
    axis_bounds=np.quantile(points,[.005,.995],axis=0);axis_bounds[0,2]=ground_z
    tests=dict(fitted=(basis,lo,hi),
        gt_body_exact=(np.eye(3),truth_lo,truth_hi),
        fitted_basis_gt_corner_bounds=(basis,gt_projected.min(axis=0),gt_projected.max(axis=0)),
        gt_axes_same_point_quantiles=(np.eye(3),axis_bounds[0],axis_bounds[1]))
    return {name:dict(basis=b.tolist(),lo=l.tolist(),hi=h.tolist(),extent=(h-l).tolist(),
                ray_test=ray_rows(frames,b,l,h,detail=name in ('fitted','gt_body_exact')))
            for name,(b,l,h) in tests.items()}


def case00():
    original=read(SOURCE/'case_00'/'result.json')
    expected=read(SHAPE/'case_00.json')
    backends=[CaptureBackend(),CaptureBackend()]
    seeds=[row['seed'] for row in expected['instances']]
    # Seed source is the original audited ACTUAL pixel, never a hidden centre.
    for action_id,row in enumerate(original['trace']):
        packet=load_packet(SOURCE/'case_00'/'packets'/f'{action_id:04d}.npz')
        assert packet.sha256()==row['packet_sha256']
        for slot,backend in enumerate(backends):
            seed=seeds[slot]['observed_seed_xyz'] if action_id==seeds[slot]['action_id'] else None
            f=packet.frame
            backend.observe(f.depth_m,f.intrinsic,f.world_from_camera,observation_id=action_id,observed_seed_xyz=seed)
    results=[]
    for slot,backend in enumerate(backends):
        old=expected['instances'][slot]
        points=np.concatenate([f['xyz'][f['valid']] for f in backend.frames])
        normals=np.concatenate([f['normals'][f['valid']] for f in backend.frames])
        ground=backend._ground(points,normals)
        selected=backend._component(points,normals,ground)
        clean=points[selected];ns=normals[selected]
        assert json_value(ground)==old['ground']
        assert array_hash(points)==old['geometry_arrays']['measured_points_xyz']['sha256']
        assert array_hash(clean)==old['geometry_arrays']['cleaned_points_xyz']['sha256']
        _,completion=backend._enclosure(clean,ns,ground)
        assert json_value(completion)==old['completion']
        basis,lo,hi=backend.captured
        estimated,orientation=fit_basis(ns);assert np.array_equal(estimated,basis)
        cx,front=LAYOUTS_V24['D24-P00']['fronts'][slot]
        truth_lo=np.array([cx-.8,front,0.]);truth_hi=np.array([cx+.8,front+.8,1.6])
        comparison=comparisons(backend.frames,clean,basis,lo,hi,truth_lo,truth_hi,float(lo[2]))
        selected_depth=np.concatenate([f['depth'][f['valid']] for f in backend.frames])[selected]
        selected_frame=np.concatenate([np.full(f['valid'].sum(),i) for i,f in enumerate(backend.frames)])[selected]
        errors=np.maximum(truth_lo-clean,clean-truth_hi)
        distance_stats=[]
        for low,high in ((0.,1.5),(1.5,2.5),(2.5,3.5),(3.5,5.01)):
            mask=(selected_depth>=low)&(selected_depth<high)
            if mask.any():
                pts=clean[mask]
                qb=np.quantile(pts,[.005,.5,.995],axis=0)
                distance_stats.append(dict(depth_min=low,depth_max=high,points=int(mask.sum()),
                    world_quantile_005_050_995=qb.tolist(),
                    outside_true_body_5cm=int((errors[mask].max(axis=1)>.05).sum())))
        results.append(dict(slot=slot,seed=seeds[slot],frame_count=len(backend.frames),
            reproduced_saved_ground_cleaned_and_rejection=True,orientation=orientation,
            captured_basis=basis.tolist(),captured_lo=lo.tolist(),captured_hi=hi.tolist(),
            diagnostics=comparison,selected_point_distance_statistics=distance_stats,
            ground=ground,completion=completion))
        print('REAL',slot,'angle_error',orientation['nearest_axis_error_deg'],
              {name:round(test['ray_test']['fraction'],6) for name,test in comparison.items()},flush=True)
    return results


def pose(x,y,heading):
    forward=np.array([[0,1,0],[1,0,0],[0,-1,0],[-1,0,0]])[heading]
    right=np.cross(forward,[0,0,1]);down=np.array([0.,0.,-1.])
    t=np.eye(4);t[:3,:3]=np.column_stack([right,down,forward]);t[:3,3]=[x,y,.8]
    return t


def analytic_controls():
    k=np.array([[48.,0,47.5],[0,48.,35.5],[0,0,1.]])
    lo=np.array([-.8,0.,0.]);hi=np.array([.8,.8,1.6])
    boxes=np.array([[-8,-8,-.12,16,16,.12],[-.8,0,0,1.6,.8,1.6],
                    [-3,3.2,0,6,.2,4.8],[2.9,-1,0,.2,4.2,4.8]])
    definitions=[('ideal_near','ideal',[1.3]*8,[1.1]*8,0),
        ('ideal_far','ideal',[4.]*8,[4.]*8,0)]
    for seed in (41,42,43):
        definitions.extend([('iid_near','iid_025px',[1.3]*8,[1.1]*8,seed),
            ('iid_far','iid_025px',[4.]*8,[4.]*8,seed),
            ('iid_mixed','iid_025px',[1.3]*4+[4.]*4,[1.1]*4+[4.]*4,seed)])
    rows=[]
    for name,model,fronts,sides,seed in definitions:
        backend=ObservedShapeBackendV24();selected_points=[];selected_normals=[]
        sequence=[pose(0.,-d,0) for d in fronts]+[pose(-.8-d,.4,1) for d in sides]
        for index,t in enumerate(sequence):
            depth,xyz=render_axial_depth(boxes,k,t,72,96,5.)
            # Known membership only in this analytic fixture, never real fitting.
            body=(depth>0)&np.all(xyz>=lo-1e-6,axis=2)&np.all(xyz<=hi+1e-6,axis=2)
            noisy=stereo_depth_v19(depth,model=model,parent_index=0,step=index,noise_seed=seed)
            backend.observe(noisy,k,t,observation_id=index)
            f=backend.frames[-1];mask=body&f['valid']
            selected_points.append(f['xyz'][mask]);selected_normals.append(f['normals'][mask])
        points=np.concatenate(selected_points);normals=np.concatenate(selected_normals)
        basis,orientation=fit_basis(normals)
        bounds=np.quantile(points@basis,[.005,.995],axis=0);bounds[0,2]=0.
        diag=comparisons(backend.frames,points,basis,bounds[0],bounds[1],lo,hi,0.)
        # Bound output: full frame/depth breakdown is only needed for case00.
        for test in diag.values():
            test['ray_test']={key:test['ray_test'][key] for key in ('conflicts','hits','fraction')}
        rows.append(dict(name=name,seed=seed,model=model,analytic_depth_arrays=len(sequence),
            true_front_distances_m=fronts,true_side_distances_m=sides,
            orientation=orientation,diagnostics=diag,
            synthetic_known_surface_membership_and_ground_used=True,
            enclosure_gate_not_run=True,not_a_planner_or_strategy_experiment=True))
        print('ANALYTIC',name,seed,'angle',round(orientation['nearest_axis_error_deg'],4),
              {key:round(value['ray_test']['fraction'],5) for key,value in diag.items()},flush=True)
    return rows


def main():
    if OUTPUT.exists():raise ValueError('retain previous memory result')
    started=perf_counter();manifest=read(SHAPE/'manifest.json')
    for name,h in manifest['source_sha256'].items():assert sha(ROOT/name)==h,name
    result=dict(status='running',scope='case00 two observed instances plus 11 fixed analytic controls',
        new_world_actions=0,new_world_sensor_packets=0,new_tsdf_fusions=0,new_quality_evaluations=0)
    try:
        result['case00']=case00()
        # Preserve the real-data diagnosis before analytic controls, even if a later control fails.
        Path('/dev/shm/v24_free_ray_real_checkpoint.json').write_text(json.dumps(json_value(result),separators=(',',':'))+'\n')
        result['analytic_controls']=analytic_controls()
        result.update(status='complete',elapsed_s=perf_counter()-started,
            reproduced_backend_enclosure_calls=2,analytic_depth_arrays=176,
            original_frozen_sources_verified=len(manifest['source_sha256']),
            input_sha256={str(p.relative_to(ROOT)):sha(p) for p in
                [SHAPE/'manifest.json',SHAPE/'result.json',SHAPE/'case_00.json',SOURCE/'case_00'/'result.json',
                 ROOT/'nso/observed_shape_v24.py',ROOT/'env/facility_documentation_v19.py',
                 ROOT/'env/canonical_rgbd_v15.py',ROOT/'env/facility_choice_v24.py']},
            diagnostic_script_sha256=sha(Path(__file__)))
        for name,h in manifest['source_sha256'].items():assert sha(ROOT/name)==h,name
    except Exception as error:
        result.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        OUTPUT.write_text(json.dumps(json_value(result),separators=(',',':'),allow_nan=False)+'\n')
        print('MEMORY_RESULT',OUTPUT,OUTPUT.stat().st_size,'bytes',result['status'],flush=True)


if __name__=='__main__':main()
