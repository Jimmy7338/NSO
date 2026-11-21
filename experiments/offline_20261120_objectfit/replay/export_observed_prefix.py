#!/usr/bin/env python3
"""Export only acquired mapper support at six exact original-B prefixes.

This is a model-development input export, not a policy or candidate experiment.
The frozen source simulator emits the already recorded sensor prefix. Hidden
source/reference geometry and future branch observations are never exported.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
V1=ROOT.parent/'dual_template_geometry_20261120'
V2=ROOT.parent/'dual_template_instance_20261120'
STATES={'shelf_left':(24,54,109),'shelf_right':(24,53,110)}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,data):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def resource(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def frozen_files():
    files=[Path(__file__),V1/'replay/run_staged_branches.py',V1/'replay/freeze.json',V2/'method/model.py',V2/'method/geometry.py']
    files += [Path(p) for p in read(V1/'replay/freeze.json')['files_sha256']]
    for layout,states in STATES.items():
        for paid in states:
            parent=V1/'replay'/f'{layout}_{paid}_attempt_001'
            files += [parent/name for name in ('result.json','prefix_execution.json','fixed_candidates.json','manifest.json')]
    return sorted(set(files))

def freeze():
    path=HERE/'observed_export_freeze.json'
    assert not path.exists(),'Use a new exporter version instead of changing a frozen export.'
    write(path,dict(schema='B5_observed_prefix_export_v1',states=STATES,created_unix_s=time.time(),
        files_sha256={str(p):sha(p) for p in frozen_files()},
        observed_support_only=True,new_candidate_actions=0,hardware=False))
    return path

def verify():
    expected=read(HERE/'observed_export_freeze.json')['files_sha256']
    assert {str(p):sha(p) for p in frozen_files()}==expected

def summarize(points,normals,labels,counts):
    import numpy as np
    if not len(points):return dict(points=0)
    low,high=np.quantile(points,[.05,.95],axis=0)
    values,freq=np.unique(labels,return_counts=True)
    return dict(points=len(points),min_xyz=points.min(0).tolist(),max_xyz=points.max(0).tolist(),
        q05_xyz=low.tolist(),q95_xyz=high.tolist(),span_q05_q95=(high-low).tolist(),
        normal_mean=normals.mean(0).tolist(),normal_abs_mean=np.abs(normals).mean(0).tolist(),
        label_counts={str(int(k)):int(v) for k,v in zip(values,freq)},
        observation_count_sum=int(counts.sum()),unlabeled_fraction=float(np.mean(labels==0)))

def export(layout,paid,out):
    import numpy as np
    verify();out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    try:
        old=load('B5_frozen_original_prefix',V1/'replay/run_staged_branches.py')
        old.verify();b,repair,_=old.dependencies()
        # Grouping helper is frozen and only reads acquired quality/track data.
        helper=load('B5_prior_observed_grouping',V2/'method/model.py')
        context=old.sealed_prefix(b,layout,paid)
        previous=V1/'replay'/f'{layout}_{paid}_attempt_001'
        assert context['prefix_receipt']==read(previous/'prefix_execution.json')
        assert context['routes']==read(previous/'fixed_candidates.json')
        mapper=context['mapper']
        native=old.save_mesh(mapper,out/'native_prefix_check.ply')
        expected=resource(previous/'prefix_native_tsdf_mesh.ply')
        assert native['sha256']==expected['sha256']
        (out/'native_prefix_check.ply').unlink()
        keys=np.asarray(list(mapper.quality),dtype=np.int64).reshape(-1,3)
        values=list(mapper.quality.values())
        full={key:np.asarray([row[key] for row in values]) for key in values[0]}
        tracks=mapper.instance_support_index.memberships_for_keys(keys)
        idx=np.flatnonzero((full['point'][:,2]>.12)&(full['point'][:,2]<1.8))
        crop_count=len(idx)
        if len(idx)>10000:idx=idx[np.linspace(0,len(idx)-1,10000,dtype=int)]
        q=mapper.quality_evidence(max_points=10000)
        assert q is not None
        for key in q:np.testing.assert_array_equal(q[key],full[key][idx])
        groups=helper.measured_groups(mapper,q)
        component=np.full(len(idx),-1,np.int64)
        group_rows=[]
        for group,ids,track in groups:
            assert np.all(component[ids]==-1)
            component[ids]=group
            assert np.all(tracks[idx[ids]]==track)
            row=dict(component_id=group,support_track=track,
                **summarize(q['point'][ids],q['normal'][ids],q['label'][ids],q['n'][ids]))
            span=np.asarray(row['span_q05_q95'])
            row['inherited_point_and_size_eligibility']=bool(len(ids)>=8 and max(span[:2])<=2.2 and span[2]>=.25)
            group_rows.append(row)
        arrays={'all_key':keys,'all_support_track':tracks,'model_indices':idx,
            'model_component_id':component,'observed_belief':mapper.belief,
            'acquired_camera_poses':np.asarray([frame.world_from_camera for frame in mapper.keyframes]),
            'acquired_camera_intrinsics':np.asarray([frame.intrinsic for frame in mapper.keyframes])}
        arrays.update({'all_'+k:v for k,v in full.items()})
        np.savez_compressed(out/'acquired_points.npz',**arrays)
        track_rows=[]
        for track in np.unique(tracks):
            selected=tracks==track
            track_rows.append(dict(support_track=int(track),
                **summarize(full['point'][selected],full['normal'][selected],full['label'][selected],full['n'][selected]),
                model_subset_points=int(np.count_nonzero(tracks[idx]==track))))
        stats=dict(layout=layout,state_action=paid,coordinate_frame='acquired mapper world in metres',
            actual_prefix_frames=mapper.frames,all_quality_points=len(keys),
            original_model_height_crop_m=[.12,1.8],points_in_height_crop=crop_count,
            model_subset_points=len(idx),model_max_points=10000,
            component_semantics='Original measured XY components within acquired support tracks; track0 means not associated with an observed instance.',
            label_dictionary={'0':'unlabeled/background','2':'ideal visible open structure','3':'ideal visible closed structure'},
            components=group_rows,observed_tracks=track_rows,
            source_geometry_or_future_frames_exported=False,native_prefix_byte_identical=True)
        write(out/'component_statistics.json',stats)
        write(out/'prefix_execution.json',context['prefix_receipt'])
        verify();old.verify();b.verify(layout)
        receipt=dict(status='completed_exact_observed_prefix_export',layout=layout,state_action=paid,
            native_prefix=expected,prefix_execution=resource(out/'prefix_execution.json'),
            acquired_points=resource(out/'acquired_points.npz'),component_statistics=resource(out/'component_statistics.json'),
            freeze=resource(HERE/'observed_export_freeze.json'),
            prefix_reconstruction_actions=paid,new_candidate_actions=0,hardware=False,
            source_reference_geometry_loaded_by_exporter=False,elapsed_s=time.monotonic()-started)
        write(out/'result.json',receipt)
        print(json.dumps(receipt),flush=True)
    except BaseException as exc:
        write(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),
            elapsed_s=time.monotonic()-started,new_candidate_actions=0,hardware=False))
        raise

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--freeze',action='store_true')
    p.add_argument('--layout',choices=tuple(STATES));p.add_argument('--state',type=int);p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.freeze:print(freeze());return
    if args.layout is None or args.state not in STATES[args.layout] or args.output is None:
        p.error('Specify one declared --layout --state and fresh --output')
    export(args.layout,args.state,args.output)

if __name__=='__main__':main()
