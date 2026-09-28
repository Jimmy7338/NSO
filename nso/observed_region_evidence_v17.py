"""Observed region correspondence and visibility evidence, without class/GT.

Region labels are temporary segmentation labels. Overlap can identify a unique
support component, not certify a permanent physical instance or hidden extent.
"""
import numpy as np
import open3d as o3d
from scipy.ndimage import binary_closing,label
from nso.semantic_completion_v5 import _aligned_quality_keys


def observed_regions(mapper,assets):
    q=mapper.quality_evidence(max_points=10000)
    if q is None:return []
    keys=_aligned_quality_keys(mapper,q,10000)
    points=q['point'];scale=mapper.config.resolution_m
    if not np.isclose(scale,.2):raise ValueError('existing observed segmentation requires 0.2 m cells')
    cells=np.column_stack([mapper.shape[0]-1-np.floor(points[:,1]/scale),np.floor(points[:,0]/scale)]).astype(int)
    inside=((cells>=0)&(cells<np.asarray(mapper.shape))).all(axis=1)
    occupied=np.zeros(mapper.shape,bool);occupied[tuple(cells[inside].T)]=True
    groups,_=label(binary_closing(occupied,structure=np.ones((3,3))))
    assignments=np.zeros(len(points),int);assignments[inside]=groups[tuple(cells[inside].T)]
    indices={a['group']:i for i,a in enumerate(assets)};regions=[]
    for group in sorted(set(assignments)-{0}):
        subset=assignments==group;ai=indices.get(int(group))
        if ai is not None and int(subset.sum())!=assets[ai]['support_points']:
            raise ValueError('region membership differs from measured assets')
        regions.append(dict(group=int(group),asset_index=ai,
            keys=sorted(set(tuple(map(int,k)) for k in keys[subset])),
            source='existing measured quality keys; before descriptor thinning',
            coordinate_epoch='fixed_world_pose',semantic_labels_used=False))
    return regions


def link_region(anchor,current_regions,peer_anchors=()):
    if anchor.get('coordinate_epoch')!='fixed_world_pose':raise ValueError('unhandled coordinate epoch')
    original={tuple(x) for x in anchor['keys']}
    if not original:raise ValueError('empty measured anchor')
    overlaps=[]
    for region in current_regions:
        if region.get('coordinate_epoch')!=anchor['coordinate_epoch']:raise ValueError('coordinate epoch changed')
        keys={tuple(x) for x in region['keys']};count=len(original&keys)
        if count:
            peers=[i for i,p in enumerate(peer_anchors) if {tuple(x) for x in p['keys']} & keys]
            overlaps.append(dict(group=region['group'],asset_index=region['asset_index'],
                retained_anchor_keys=count,overlapping_peer_indices=peers))
    status=('no_observed_overlap' if not overlaps else 'split_ambiguous' if len(overlaps)>1 else
            'merged_ambiguous' if overlaps[0]['overlapping_peer_indices'] else
            'region_not_candidate_asset' if overlaps[0]['asset_index'] is None else 'unique_observed_region')
    return dict(status=status,anchor_key_count=len(original),overlaps=overlaps,
        matched_asset_index=overlaps[0]['asset_index'] if status=='unique_observed_region' else None,
        stable_physical_instance_certified=False,semantic_labels_used=False)


def _projection(points,pose,intrinsic,image_shape,max_depth):
    points=np.asarray(points,float)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():raise ValueError('finite Nx3 points required')
    local=(points-pose[:3,3])@pose[:3,:3];z=local[:,2]
    u=np.rint(local[:,0]/np.maximum(z,.01)*intrinsic[0,0]+intrinsic[0,2]).astype(int)
    v=np.rint(local[:,1]/np.maximum(z,.01)*intrinsic[1,1]+intrinsic[1,2]).astype(int)
    valid=(z>.15)&(z<=max_depth)&(u>=0)&(u<image_shape[1])&(v>=0)&(v<image_shape[0])
    return z,u,v,valid


def actual_depth_evidence(points,frame,config):
    z,u,v,valid=_projection(points,frame.world_from_camera,frame.intrinsic,frame.depth_m.shape,config.max_depth_m)
    ids=np.flatnonzero(valid);depth=frame.depth_m[v[ids],u[ids]]
    good=np.isfinite(depth)&(depth>0)&(depth<=config.max_depth_m)
    delta=depth-z[ids];tolerance=config.truncation_m
    return dict(points=len(points),in_image=int(valid.sum()),outside_image=int((~valid).sum()),
        invalid_depth=int((~good).sum()),consistent=int(np.count_nonzero(good&(np.abs(delta)<=tolerance))),
        actual_depth_before_point=int(np.count_nonzero(good&(delta < -tolerance))),
        point_before_actual_depth=int(np.count_nonzero(good&(delta > tolerance))),
        tolerance_m=float(tolerance),future_frame_used=True,truth_used=False)


def predicted_mesh_evidence(points,pose,intrinsic,image_shape,config,observed_mesh,target_keys):
    """Raycast ONLY a reconstructed mesh frozen before the view is executed."""
    points=np.asarray(points,float)
    z,_,_,valid=_projection(points,pose,intrinsic,image_shape,config.max_depth_m)
    ids=np.flatnonzero(valid);target_keys={tuple(x) for x in target_keys}
    counts=dict(points=len(points),in_image=len(ids),outside_image=int((~valid).sum()),
        known_surface_before_point=0,known_surface_near_point=0,no_known_hit=0,known_surface_after_point=0,
        occluder_key_in_target_region=0,occluder_key_not_in_target_region=0,
        future_frame_used=False,truth_used=False,calibrated_visibility=False)
    if not len(ids):return counts
    ray=o3d.t.geometry.RaycastingScene(nthreads=1)
    ray.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(observed_mesh))
    directions=points[ids]-pose[:3,3]
    rays=np.column_stack([np.tile(pose[:3,3],(len(ids),1)),directions]).astype(np.float32)
    hit=ray.cast_rays(o3d.core.Tensor(rays),nthreads=1)['t_hit'].numpy()
    finite=np.isfinite(hit);delta=(hit-1.)*z[ids]
    blocked=finite&(delta < -config.truncation_m)
    counts.update(known_surface_before_point=int(blocked.sum()),
        known_surface_near_point=int(np.count_nonzero(finite&(np.abs(delta)<=config.truncation_m))),
        known_surface_after_point=int(np.count_nonzero(finite&(delta>config.truncation_m))),
        no_known_hit=int((~finite).sum()))
    intersections=pose[:3,3]+directions[blocked]*hit[blocked,None]
    same=sum(tuple(map(int,k)) in target_keys for k in np.floor(intersections/.15).astype(int))
    counts.update(occluder_key_in_target_region=same,occluder_key_not_in_target_region=int(blocked.sum())-same)
    return counts
