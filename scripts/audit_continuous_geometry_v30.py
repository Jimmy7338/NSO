#!/usr/bin/env python3
"""Exact box-union area plus finite visibility on ten frozen V29 path records."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import signal
import sys
from time import perf_counter
import traceback
import zipfile
import numpy as np

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from nso.box_union_geometry_v30 import BoxV30,union_exterior_faces_v30,union_volume_v30,surface_audit_v30

SOURCE=ROOT/'audit_results/v29_scene_information_20260917'
OUTPUT=ROOT/'audit_results/v30_continuous_geometry_20260917'
CONFIG=ROOT/'configs/virtual3d/v29_information_scene_20260917.json'
PROTOCOL=ROOT/'docs/research/V30_CONTINUOUS_GEOMETRY_PROTOCOL_20260917.md'
SOURCE_INVENTORY_SHA='b7a6a84f547f51c952d7a1497dab23584d1537d6da9b7cb5257fb1f84c70a475'
FROZEN_MODULE_SHA='a0fd9cc5e625ae4f926e6184fdbb935deaf687ae77ed095a9c057c0ade8087c0'
FROZEN_TEST_SHA='6f7ea1acd814080f8812d4977e15cd770f4447c7aca1c770192603248d167bac'
CAP=1024**2;RESERVE=64*1024**2;SPACING=.1


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def digest(data):return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)


def checked_inputs():
    require(sha(SOURCE/'artifact_hashes.json')==SOURCE_INVENTORY_SHA,'V29 authoritative inventory changed')
    inventory=read(SOURCE/'artifact_hashes.json')
    for name,expected in inventory.items():require(sha(SOURCE/name)==expected,'V29 artifact changed '+name)
    manifest=read(SOURCE/'manifest.json')
    for name,expected in manifest['source_sha256'].items():require(sha(ROOT/name)==expected,'V29 source changed '+name)
    require(read(SOURCE/'result.json')['status']=='complete','V29 incomplete')
    require(sha(ROOT/'nso/box_union_geometry_v30.py')==FROZEN_MODULE_SHA,'geometry module changed after tests')
    require(sha(ROOT/'tests/virtual3d/test_box_union_geometry_v30.py')==FROZEN_TEST_SHA,'unit test source changed')
    return inventory,manifest


def boxes_for(config,classes):
    boxes=[BoxV30(tuple(b)) for b in config['class_independent_wall_bounds']]
    for i,station in enumerate(config['stations']):
        for name in ('body_relative_bounds','complex_attachment_relative_bounds'):
            if name.startswith('complex') and classes[i]!='complex':continue
            b=list(config[name]);b[0]+=station['center_x'];b[1]+=station['center_x']
            boxes.append(BoxV30(tuple(b),i))
    return tuple(boxes)


def first_hits(origin,directions,boxes,max_distance):
    """Analytic segment/slab intersection. Outputs geometry only, never owner IDs."""
    directions=np.asarray(directions,float);origin=np.asarray(origin,float)
    best=np.full(len(directions),np.inf);normals=np.zeros_like(directions)
    for box in boxes:
        bounds=np.asarray(box.bounds).reshape(3,2)
        near=np.full_like(directions,-np.inf);far=np.full_like(directions,np.inf)
        invalid=np.zeros(len(directions),bool)
        for a in range(3):
            active=np.abs(directions[:,a])>1e-12
            invalid|=(~active)&((origin[a]<bounds[a,0]-1e-10)|(origin[a]>bounds[a,1]+1e-10))
            one=(bounds[a,0]-origin[a])/directions[active,a]
            two=(bounds[a,1]-origin[a])/directions[active,a]
            near[active,a]=np.minimum(one,two);far[active,a]=np.maximum(one,two)
        distance=near.max(axis=1);exit_distance=far.min(axis=1)
        hit=(~invalid)&(distance>1e-8)&(exit_distance>=distance-1e-10)&(distance<=max_distance+1e-10)&(distance<best-1e-10)
        ids=np.flatnonzero(hit)
        if len(ids):
            axes=np.argmax(near[ids],axis=1);best[ids]=distance[ids];normals[ids]=0.
            normals[ids,axes]=-np.sign(directions[ids,axes])
    return best,normals


def quadrature(faces):
    points=[];normals=[];weights=[];owners=[]
    for face in faces:
        if face.axis==2 or face.owner is None:continue
        axes=[a for a in range(3) if a!=face.axis]
        sizes=[face.bounds[2*a+1]-face.bounds[2*a] for a in axes]
        counts=[max(1,math.ceil(s/SPACING-1e-10)) for s in sizes]
        for i in range(counts[0]):
            for j in range(counts[1]):
                p=[0.,0.,0.];p[face.axis]=face.bounds[2*face.axis]
                for a,size,n,k in zip(axes,sizes,counts,(i,j)):
                    p[a]=face.bounds[2*a]+size*(k+.5)/n
                points.append(p);normals.append(face.normal);weights.append(face.area/(counts[0]*counts[1]));owners.append(face.owner)
    return dict(points=np.asarray(points,float),normals=np.asarray(normals,float),
        weights=np.asarray(weights,float),owners=np.asarray(owners,int))


def visible_mask(pose,boxes,quad,config):
    x,y,h=pose;obs=config['geometric_observation'];origin=np.array([x,y,obs['camera_height_m']])
    delta=quad['points']-origin;distance=np.linalg.norm(delta,axis=1)
    forward=np.asarray(config['heading_vectors'][h]);right=np.array([forward[1],-forward[0]])
    axial=delta[:,:2]@forward;lateral=delta[:,:2]@right
    visible=(axial>1e-8)&(np.abs(lateral)<=axial*math.tan(math.radians(obs['horizontal_fov_deg']/2))+1e-10)
    visible&=(np.abs(delta[:,2])<=axial*math.tan(math.radians(obs['vertical_fov_deg']/2))+1e-10)
    visible&=(distance<=obs['range_m']+1e-10)&(np.einsum('ij,ij->i',quad['normals'],-delta)>1e-8)
    ids=np.flatnonzero(visible)
    if len(ids):
        first,_=first_hits(origin,delta[ids]/distance[ids,None],boxes,obs['range_m'])
        visible[ids]=first>=distance[ids]-1e-7
    return visible


def ray_signature(pose,boxes,config):
    x,y,h=pose;obs=config['geometric_observation'];origin=np.array([x,y,obs['camera_height_m']])
    forward=np.r_[config['heading_vectors'][h],0.];right=np.array([forward[1],-forward[0],0.])
    horizontal=np.linspace(-math.radians(obs['horizontal_fov_deg']/2),math.radians(obs['horizontal_fov_deg']/2),17)
    vertical=np.linspace(-math.radians(obs['vertical_fov_deg']/2),math.radians(obs['vertical_fov_deg']/2),17)
    directions=np.array([forward+math.tan(a)*right+np.array([0.,0.,math.tan(b)]) for b in vertical for a in horizontal])
    directions/=np.linalg.norm(directions,axis=1)[:,None]
    first,normals=first_hits(origin,directions,boxes,obs['range_m'])
    return [None if not math.isfinite(d) else tuple(np.round(np.r_[origin+d*ray,n],9))
        for d,ray,n in zip(first,directions,normals)]


def trajectory(config,actions):
    pose=tuple(config['anchor']);states=[pose]
    for action in actions:
        x,y,h=pose
        if action=='forward':
            dx,dy=config['heading_vectors'][h];pose=(x+dx,y+dy,h)
        elif action in ('left','right'):pose=(x,y,(h+(-1 if action=='left' else 1))%4)
        else:raise ValueError('unknown saved action')
        states.append(pose)
    return states


def check_motion(states,boxes,config):
    grid=config['grid'];radius=grid['robot_radius_m']
    for a,b in zip(states,states[1:]):
        require(grid['x_min']<=b[0]<=grid['x_max'] and grid['y_min']<=b[1]<=grid['y_max'],'saved path outside frozen grid')
        for box in boxes:
            q=box.bounds
            dx=max(q[0]-max(a[0],b[0]),min(a[0],b[0])-q[1],0.)
            dy=max(q[2]-max(a[1],b[1]),min(a[1],b[1])-q[3],0.)
            require(dx*dx+dy*dy>=radius*radius-1e-7,'saved path violates frozen physical clearance')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if not args.run:parser.print_help();return
    require(not OUTPUT.exists(),'output exists; one audit only')
    require(shutil.disk_usage(OUTPUT.parent).free>RESERVE+CAP,'64 MiB reserve')
    OUTPUT.mkdir();started=perf_counter()
    def write(name,data):
        raw=(json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
        require(sum(p.stat().st_size for p in OUTPUT.iterdir() if p.is_file())+len(raw)+16384<CAP,'1 MiB output cap')
        require(shutil.disk_usage(OUTPUT).free-len(raw)>RESERVE,'64 MiB reserve')
        target=OUTPUT/name;temp=target.with_suffix(target.suffix+'.tmp');temp.write_bytes(raw);os.replace(temp,target)
    def timeout(*_):raise TimeoutError('fixed 300-second audit bound')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(300)
    try:
        inventory,old_manifest=checked_inputs();config=read(CONFIG)
        sources={name:sha(ROOT/name) for name in old_manifest['source_sha256']}
        for p in (Path(__file__),PROTOCOL,ROOT/'nso/box_union_geometry_v30.py',ROOT/'tests/virtual3d/test_box_union_geometry_v30.py'):
            sources[str(p.relative_to(ROOT))]=sha(p)
        with zipfile.ZipFile(OUTPUT/'sources.zip','x',zipfile.ZIP_DEFLATED) as archive:
            for name in sources:archive.writestr(name,(ROOT/name).read_bytes())
        write('manifest.json',dict(source_sha256=sources,source_archive_sha256=sha(OUTPUT/'sources.zip'),
            input_inventory_sha256=SOURCE_INVENTORY_SHA,source_root=str(SOURCE.relative_to(ROOT)),
            unit_tests=dict(count=8,passed=8,exit_code=0,seconds=.017,actual_tool_chunk='e10ca1',
                command='python -B -m unittest discover -s tests/virtual3d -p test_box_union_geometry_v30.py -v',
                provenance='Actual preceding independent unit invocation; not rerun by audit'),
            World=0,sensor_packets=0,mapper_updates=0,TSDF_updates=0,Q_evaluations=0,new_main_tasks=0))
        scenes=[boxes_for(config,classes) for classes in config['hypotheses']]
        common_planes=tuple(sorted({b.bounds[2*a+s] for boxes in scenes for b in boxes for s in (0,1)}) for a in range(3))
        faces=[union_exterior_faces_v30(boxes,split_planes=common_planes) for boxes in scenes]
        reference=[]
        for h,(boxes,rows) in enumerate(zip(scenes,faces)):
            full=surface_audit_v30(rows);volume=union_volume_v30(boxes)
            require(full['closed_oriented_two_manifold'],'scene union is not an oriented closed two-manifold')
            require(abs(full['signed_volume']-volume)<1e-9,'whole-union signed volume mismatch')
            assets=[]
            for owner,category in enumerate(config['hypotheses'][h]):
                asset_boxes=[b for b in boxes if b.owner==owner]
                asset_faces=union_exterior_faces_v30(asset_boxes)
                audit=surface_audit_v30(asset_faces);area=sum(f.area for f in asset_faces if f.axis!=2)
                expected_area,expected_volume=(17.12,10.272) if category=='complex' else (16.,9.6)
                require(abs(area-expected_area)<1e-10,'exact vertical area identity failed')
                require(audit['closed_oriented_two_manifold'],'facility union is not closed')
                require(abs(audit['signed_volume']-expected_volume)<1e-10,'facility oriented volume identity failed')
                require(abs(union_volume_v30(asset_boxes)-expected_volume)<1e-10,'facility grid volume identity failed')
                assets.append(dict(owner=owner,vertical_area_m2=area,volume_m3=expected_volume,
                    surface_audit=audit,faces=[asdict(f) for f in asset_faces]))
            reference.append(dict(hypothesis=h,whole_union_volume_m3=volume,whole_union_surface_audit=full,assets=assets))
        write('exact_reference.json',reference)
        prefix_states=trajectory(config,config['prefix_actions']);signatures=[[ray_signature(p,boxes,config) for p in prefix_states] for boxes in scenes]
        unequal=[i for i in range(len(prefix_states)) if signatures[0][i]!=signatures[1][i]]
        write('prefix_pairing.json',dict(states=len(prefix_states),rays_per_pose=289,total_queries=2*len(prefix_states)*289,
            nonsemantic_geometric_signature_only=True,unequal_pose_indices=unequal,
            signature_sha256=[[digest(s) for s in rows] for rows in signatures],passed=not unequal,
            scope='Fixed finite analytic rays, not actual depth pixels or all sensor channels'))
        require(not unequal,'frozen prefix differs under finite analytic ray geometry; retain failure')
        quads=[quadrature(rows) for rows in faces];caches=[{},{}];old=read(SOURCE/'main_policy_witnesses.json')
        paths=[dict(id='policy_%02d'%i,record=row) for i,row in enumerate(old['records'])]
        paths += [dict(id='complete_%02d'%i,record=row['witness']) for i,row in enumerate(old['complete_options']) if row.get('witness')]
        require(len(paths)==10,'expected exactly ten saved V29 path records')
        output=[]
        for item in paths:
            row=item['record'];h=row['actual_hypothesis'];actions=config['prefix_actions']+row['suffix_actions']
            states=trajectory(config,actions);saved_suffix=[tuple(p) for p in row['suffix_states']]
            require(states[len(config['prefix_actions']):]==saved_suffix,'saved actions/states disagree')
            require(len(actions)==row['total_paid_actions']<=48 and states[-1]==tuple(config['anchor']),'saved exact return/budget failed')
            check_motion(states,scenes[h],config);q=quads[h];seen=np.zeros(len(q['points']),bool);curve=[]
            denominator=[sum(f.area for f in faces[h] if f.axis!=2 and f.owner==i) for i in (0,1)]
            for pose in states:
                if pose not in caches[h]:caches[h][pose]=visible_mask(pose,scenes[h],q,config)
                seen|=caches[h][pose]
                fractions=[float(q['weights'][seen&(q['owners']==i)].sum())/denominator[i] for i in (0,1)]
                curve.append(dict(macro=sum(fractions)/2,per_station=fractions))
            initial=curve[len(config['prefix_actions'])];final=curve[-1]
            output.append(dict(path_id=item['id'],policy=row['policy'],hypothesis=h,
                saved_path_sha256=digest([actions,states]),paid_actions=len(actions),returned_exact_pose=True,
                exact_reference_vertical_areas_m2=denominator,quadrature_samples=len(q['points']),
                prefix=initial,final=final,delta_macro=final['macro']-initial['macro'],
                old_finite_V29_macro=row['potential_macro'],old_metric_not_overwritten=True,
                curve=curve,visibility_quadrature_spacing_max_m=SPACING,
                scope='Saved path rescoring only, no DP rerun; neither current optimal value nor Q'))
        write('saved_path_visibility.json',output)
        for name,expected in sources.items():require(sha(ROOT/name)==expected,'frozen source changed '+name)
        checked_inputs()
        write('result.json',dict(status='complete',exact_area_and_closed_boundary_passed=True,
            simple_vertical_area_m2=16.,complex_vertical_area_m2=17.12,
            old_complex_quadrature_area_m2=17.28,old_absolute_overestimate_m2=.16,
            simple_volume_m3=9.6,complex_volume_m3=10.272,
            prefix_finite_ray_pairing_passed=True,source_and_inputs_unchanged=True,
            saved_path_records=len(output),unique_saved_action_paths=len({r['saved_path_sha256'] for r in output}),
            actual_visibility_queries_by_hypothesis=[len(c) for c in caches],
            saved_policy_means={name:sum(r['final']['macro'] for r in output if r['path_id'].startswith('policy') and r['policy']==name)/2
                for name in ('G','correct_class_oracle','swapped_class_with_geometric_correction')},
            values_are_saved_routes_not_reoptimized=True,exact_continuous_visible_area_integral=False,
            semantic_information_value_reestimated=False,actual_sensor_validation=False,Q_evaluations=0,new_main_tasks=0,
            elapsed_seconds=perf_counter()-started))
    except BaseException as error:
        write('failure.json',dict(type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),
            elapsed_seconds=perf_counter()-started,new_main_tasks=0))
        raise
    finally:
        signal.alarm(0)
        write('artifact_hashes.json',{p.name:sha(p) for p in sorted(OUTPUT.iterdir()) if p.is_file() and p.name!='artifact_hashes.json'})
    print(json.dumps(read(OUTPUT/'result.json')),flush=True)


if __name__=='__main__':main()
