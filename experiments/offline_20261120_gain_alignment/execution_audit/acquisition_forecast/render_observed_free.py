"""Fitted-template future RGB-D with the retained B2 free-hit interpretation.

Only a single model hypothesis, current acquired native scene and acquired
keyframes enter. No source/actual future observations are accepted. Current
native hits are never erased; measured-surface support vetoes free evidence.
"""
from pathlib import Path
import hashlib,importlib.util,sys
import numpy as np
import open3d as o3d
REPAIR_PATH=Path('/home/ouc/nso_robot/analyses/dual_template_repair_20261119/method/occlusion_repair.py')
REPAIR_SHA='57c5cd1bb58a2e816d81184014cbfbad2a3d5ddeb872c43c7b5d9895378bb2fb'
assert hashlib.sha256(REPAIR_PATH.read_bytes()).hexdigest()==REPAIR_SHA,'Frozen B2 history source changed'
SPEC=importlib.util.spec_from_file_location('C_fitted_frozen_B2_history',REPAIR_PATH)
REPAIR=importlib.util.module_from_spec(SPEC);sys.modules[SPEC.name]=REPAIR;SPEC.loader.exec_module(REPAIR)

def corrected_depth(hypothesis_scene,measured_scene,pose,intrinsic,width,height,keyframes,*,min_depth_m=.15,max_depth_m=8.):
    """Render original axial-depth sensor convention after strict-free removal.

    First cast stays authoritative unless it is strictly contradicted. Only
    then inspect every hypothesis intersection. An empty listing cannot erase
    a first hit, mirroring B2's safeguard. Unknown and measured-supported
    intersections stop the ray. Final nearest acquired mesh hit is independent.
    """
    if hypothesis_scene is None:raise ValueError('one hypothesis ray scene required')
    if not(0<min_depth_m<max_depth_m):raise ValueError('ordered positive sensor range required')
    pose=np.asarray(pose,float);k=np.asarray(intrinsic,float)
    if pose.shape!=(4,4)or k.shape!=(3,3)or not np.isfinite(pose).all()or not np.isfinite(k).all():raise ValueError('finite pose and intrinsic required')
    if k[0,0]<=0 or k[1,1]<=0:raise ValueError('positive focal lengths required')
    v,u=np.mgrid[:height,:width]
    local=np.column_stack([(u.ravel()-k[0,2])/k[0,0],(v.ravel()-k[1,2])/k[1,1],np.ones(width*height)])
    rays=np.column_stack([np.tile(pose[:3,3],(width*height,1)),local@pose[:3,:3].T]).astype(np.float32)
    tensor=o3d.core.Tensor(rays)
    first=hypothesis_scene.cast_rays(tensor,nthreads=1)['t_hit'].numpy()
    corrected=first.copy();finite=np.flatnonzero(np.isfinite(first)&(first>=0))
    audit=dict(pixels=width*height,finite_hypothesis_first_hits=len(finite),strict_free_first_hits=0,listed_intersections=0,strict_free_intersections=0,measured_veto_intersections=0,empty_listing_fallbacks=0,corrected_first_hits=0)
    if len(finite):
        xyz=rays[finite,:3]+rays[finite,3:]*first[finite,None]
        initial=REPAIR.classify_history(xyz,keyframes)
        blocked=finite[initial['strict_free']];audit['strict_free_first_hits']=len(blocked)
        if len(blocked):
            hit=hypothesis_scene.list_intersections(o3d.core.Tensor(rays[blocked]),nthreads=1)
            ids=hit['ray_ids'].numpy().astype(np.int64);times=hit['t_hit'].numpy()
            valid=np.isfinite(times)&(times>=0);ids,times=ids[valid],times[valid]
            points=rays[blocked[ids],:3]+rays[blocked[ids],3:]*times[:,None]
            classified=REPAIR.classify_history(points,keyframes)
            counts=np.bincount(ids,minlength=len(blocked));nearest=np.full(len(blocked),np.inf,np.float32)
            keep=~classified['strict_free'];np.minimum.at(nearest,ids[keep],times[keep])
            has_listing=counts>0;corrected[blocked[has_listing]]=nearest[has_listing]
            audit.update(listed_intersections=len(times),strict_free_intersections=int(classified['strict_free'].sum()),measured_veto_intersections=int((classified['raw_free_evidence']&classified['measured_surface']).sum()),empty_listing_fallbacks=int((~has_listing).sum()),corrected_first_hits=int(np.count_nonzero(corrected[blocked]!=first[blocked])))
    measured=np.full(len(first),np.inf,np.float32)if measured_scene is None else measured_scene.cast_rays(tensor,nthreads=1)['t_hit'].numpy()
    fused=np.minimum(corrected,measured)
    depth=np.where(np.isfinite(fused)&(fused>=min_depth_m)&(fused<=max_depth_m),fused,0.).reshape(height,width).astype(np.float32)
    audit.update(valid_predicted_pixels=int(np.count_nonzero(depth)),measured_scene_wins=int(np.count_nonzero(np.isfinite(measured)&(measured<=corrected))),native_scene_never_filtered=True,measured_surface_veto=True,history_tolerance_m=REPAIR.DEPTH_TOLERANCE_M,sensor_min_depth_m=min_depth_m,sensor_max_depth_m=max_depth_m,axial_depth=True,reference_geometry_consumed=False)
    return depth,audit
