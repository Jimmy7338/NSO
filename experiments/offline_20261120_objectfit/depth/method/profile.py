"""Neutral structural depth-profile diagnostic using acquired measurements only.

This module does not bind a policy model, choose navigation candidates or start
robot/simulator actions. B6 AB geometry other than local depth stays fixed.
"""
from pathlib import Path
import hashlib,importlib.util
import numpy as np
import open3d as o3d

B6=Path('/home/ouc/nso_robot/analyses/dual_template_parts_20261120/method')
REPAIR=Path('/home/ouc/nso_robot/analyses/dual_template_repair_20261119/method/occlusion_repair.py')
REPAIR_SHA='57c5cd1bb58a2e816d81184014cbfbad2a3d5ddeb872c43c7b5d9895378bb2fb'
assert hashlib.sha256(REPAIR.read_bytes()).hexdigest()==REPAIR_SHA
spec=importlib.util.spec_from_file_location('B7_frozen_B2_history',REPAIR)
repair=importlib.util.module_from_spec(spec);spec.loader.exec_module(repair)

PRIOR_DEPTH_M=.65
PROFILE_SAMPLES=21
PIXEL_STRIDE=3
DISTANCE_CLIP_M=.15
GIBBS_SCALE_M=.06
PROFILE_BAND_NLL=1.

def digest(array):return hashlib.sha256(np.ascontiguousarray(array,dtype=np.float64).tobytes()).hexdigest()

def _scene(mesh):
    scene=o3d.t.geometry.RaycastingScene(nthreads=1)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene

def primitive_meshes(obj,depth,bits):
    """Exactly the B6 primitive construction, varying anchored depth only."""
    dims=np.asarray(obj['dims'],float);R=np.asarray(obj['rotation'],float)
    local_center=np.asarray(obj['center'],float)@R
    extent=obj['geometry_fit_audit']['extent_uncertainty'][1]
    low=extent['low_face_coordinate_m'];high=extent['high_face_coordinate_m']
    if (low is None)==(high is None):raise ValueError('one measured depth anchor required')
    x=local_center[0]-dims[0]/2;y=float(low) if low is not None else float(high)-depth
    width,height=float(dims[0]),float(dims[2])
    box=o3d.geometry.TriangleMesh.create_box(width,float(depth),height).translate((x,y,0.))
    shelf=o3d.geometry.TriangleMesh()
    for slab in obj['observed_layers_audit']['final_intervals']:
        shelf+=o3d.geometry.TriangleMesh.create_box(width,float(depth),slab['upper_m']-slab['lower_m']).translate((x,y,slab['lower_m']))
    for dx in (0.,width-.08):
        for dy in (0.,depth-.08):
            shelf+=o3d.geometry.TriangleMesh.create_box(.08,.08,height).translate((x+dx,y+dy,0.))
    hist=np.array([np.mean((bits&(1<<b))!=0) for b in range(8)])
    az=-np.pi+(int(np.argmax(hist))+.5)*2*np.pi/8
    direction=np.array([np.cos(az),np.sin(az),0.])@R
    axis=int(np.argmax(np.abs(direction[:2])))
    origin=np.array([x,y,0.]);board_dims=np.array([width,depth,height]);board_dims[axis]=.08
    if direction[axis]>0:origin[axis]+=[width,depth][axis]-.08
    shelf+=o3d.geometry.TriangleMesh.create_box(*board_dims).translate(origin)
    box.rotate(R,center=(0,0,0));shelf.rotate(R,center=(0,0,0))
    return box,shelf

def fixed_observed_rays(obj,upper_depth,keyframes):
    """Same valid acquired camera-z=1 rays for every depth and both shapes."""
    R=np.asarray(obj['rotation'],float);dims=np.asarray(obj['dims'],float)
    center=np.asarray(obj['center'],float)@R
    ext=obj['geometry_fit_audit']['extent_uncertainty'][1]
    low=ext['low_face_coordinate_m'];high=ext['high_face_coordinate_m']
    y=float(low) if low is not None else float(high)-upper_depth
    intervals=obj['observed_layers_audit']['final_intervals']
    zlow=min([0.]+[p['lower_m'] for p in intervals]);zhigh=max([float(dims[2])]+[p['upper_m'] for p in intervals])
    roi=o3d.geometry.TriangleMesh.create_box(float(dims[0]),float(upper_depth),zhigh-zlow).translate((center[0]-dims[0]/2,y,zlow)).rotate(R,center=(0,0,0))
    roi_scene=_scene(roi);seen=set();rays=[];depths=[];frames=[];duplicates=0;valid_count=0
    for frame in keyframes:
        pose=np.asarray(frame.world_from_camera,float);K=np.asarray(frame.intrinsic,float)
        h,w=frame.depth_m.shape;v,u=np.mgrid[0:h:PIXEL_STRIDE,0:w:PIXEL_STRIDE]
        observed=np.asarray(frame.depth_m)[v,u].ravel();u=u.ravel();v=v.ravel()
        valid=np.isfinite(observed)&(observed>0);observed=observed[valid];u=u[valid];v=v[valid]
        valid_count+=len(observed)
        camera=np.column_stack([(u-K[0,2])/K[0,0],(v-K[1,2])/K[1,1],np.ones(len(u))])
        world=camera@pose[:3,:3].T
        batch=np.column_stack([np.tile(pose[:3,3],(len(u),1)),world]).astype(np.float32)
        t=roi_scene.cast_rays(o3d.core.Tensor(batch),nthreads=1)['t_hit'].numpy()
        indices=np.flatnonzero(np.isfinite(t)&(t>=0))
        pose_key=hashlib.sha256(pose.tobytes()+K.tobytes()).hexdigest()
        frames.append(hashlib.sha256(pose.tobytes()+K.tobytes()+np.ascontiguousarray(frame.depth_m).tobytes()).hexdigest())
        for i in indices:
            key=(pose_key,int(u[i]),int(v[i]),float(observed[i]))
            if key in seen:duplicates+=1;continue
            seen.add(key);rays.append(batch[i]);depths.append(float(observed[i]))
    rays=np.asarray(rays,np.float32).reshape(-1,6);depths=np.asarray(depths,float)
    return rays,depths,dict(ray_count=len(rays),all_stride_valid_rays=valid_count,duplicate_rays_removed=duplicates,
        rays_sha256=digest(rays),depths_sha256=digest(depths),keyframe_fingerprints=frames,
        direction_parameterization='camera z=1, transformed to world; first-hit t is camera depth in metres',
        max_search_roi_depth_m=float(upper_depth),roi_z_bounds_m=[zlow,zhigh],
        denominator_fixed_across_depth_and_hypotheses=True,pixel_stride=PIXEL_STRIDE)

def measured_terms(scene,cloud,rays,observed_depth,keyframes):
    distances=scene.compute_distance(o3d.core.Tensor(np.asarray(cloud,np.float32)),nthreads=1).numpy()
    surface=float(np.mean(np.minimum(distances,DISTANCE_CLIP_M)**2))
    if not len(rays):return surface,0.,dict(finite_model_hits=0,front_hit_free_candidates=0,strict_free_hits=0,measured_surface_vetoes=0)
    t=scene.cast_rays(o3d.core.Tensor(rays),nthreads=1)['t_hit'].numpy()
    finite=np.isfinite(t)&(t>0)
    front=finite&(t<observed_depth-.10)
    indices=np.flatnonzero(front);penalties=np.zeros(len(rays));vetoes=0;strict_count=0
    if len(indices):
        points=rays[indices,:3]+t[indices,None]*rays[indices,3:]
        classified=repair.classify_history(points,keyframes)
        strict=classified['strict_free'];strict_count=int(strict.sum())
        vetoes=int(classified['measured_surface'].sum())
        accepted=indices[strict]
        penalties[accepted]=np.minimum(observed_depth[accepted]-t[accepted],DISTANCE_CLIP_M)**2
    return surface,float(penalties.mean()),dict(finite_model_hits=int(finite.sum()),
        front_hit_free_candidates=int(front.sum()),strict_free_hits=strict_count,measured_surface_vetoes=vetoes)

def select_profile(depths,nll,free_terms,prior_depth):
    """Heuristic likelihood band and explicit non-identification outcomes."""
    depths=np.asarray(depths,float);nll=np.asarray(nll,float);free_terms=np.asarray(free_terms,float)
    prior_index=int(np.argmin(np.abs(depths-prior_depth)))
    if not np.isclose(depths[prior_index],prior_depth,rtol=0,atol=1e-12):raise ValueError('prior must be present in fixed grid')
    plausible=nll<=float(nll.min())+PROFILE_BAND_NLL
    ids=np.flatnonzero(plausible)
    selected=prior_index;reason='prior_retained_in_heuristic_band';adopt=False
    upper_bracket=None
    if not plausible[prior_index]:
        if ids[-1]==len(depths)-1:reason='artificial_upper_search_boundary_unresolved'
        elif np.any(np.diff(ids)>1):reason='disconnected_plausible_depth_modes'
        else:
            target=int(ids[np.argmin(np.abs(depths[ids]-prior_depth))])
            if not np.any(free_terms[prior_index]>free_terms[target]+1e-12):
                reason='no_shared_measured_free_contradiction_reduction'
            else:selected=target;reason='finite_data_supported_band_excludes_prior';adopt=True
    if ids[-1]<len(depths)-1:upper_bracket=[float(depths[ids[-1]]),float(depths[ids[-1]+1])]
    return dict(point_estimate_m=float(depths[selected]),prior_depth_m=float(prior_depth),adopted=adopt,
        reason=reason,plausible_grid_depths_m=depths[plausible].tolist(),
        plausible_grid_interval_m=[float(depths[ids[0]]),float(depths[ids[-1]])],
        internal_upper_crossing_bracket_m=upper_bracket,
        lower_end_is_measured_span=bool(ids[0]==0),artificial_upper_boundary_in_band=bool(ids[-1]==len(depths)-1),
        heuristic_band_delta_nll=PROFILE_BAND_NLL,statistical_confidence_interval=False,
        exact_hidden_depth_identified=False)

def profile_object(obj,cloud,bits,keyframes):
    ext=obj['geometry_fit_audit']['extent_uncertainty'][1]
    result=dict(support_track=obj['support_track'],fit_input_points_sha256=digest(cloud),
        fitted_axis=1,current_depth_m=float(obj['dims'][1]),anchor_low_m=ext['low_face_coordinate_m'],
        anchor_high_m=ext['high_face_coordinate_m'],observed_span_m=ext['observed_span_lower_bound_m'])
    if (ext['low_face_coordinate_m'] is None)==(ext['high_face_coordinate_m'] is None):
        result.update(status='retained_without_single_observed_anchor',depth_point_estimate_m=float(obj['dims'][1]),
            uncertainty_interval_m=ext['extent_interval_m']);return result
    lower=float(ext['observed_span_lower_bound_m']);prior=float(obj['dims'][1]);upper=max(2*PRIOR_DEPTH_M,lower)
    if lower<=0 or lower>prior+1e-10 or upper<=lower:
        result.update(status='retained_inconsistent_or_degenerate_domain',depth_point_estimate_m=prior);return result
    depths=np.unique(np.r_[np.linspace(lower,upper,PROFILE_SAMPLES),prior])
    rays,observed,ray_audit=fixed_observed_rays(obj,upper,keyframes)
    surface=[];free=[];evidence=[]
    for depth in depths:
        meshes=primitive_meshes(obj,float(depth),bits);surface_row=[];free_row=[];evidence_row=[]
        for mesh in meshes:
            a,b,audit=measured_terms(_scene(mesh),cloud,rays,observed,keyframes)
            surface_row.append(a);free_row.append(b);evidence_row.append(audit)
        surface.append(surface_row);free.append(free_row);evidence.append(evidence_row)
    surface=np.asarray(surface);free=np.asarray(free)
    log_likelihood=-(surface+free)/(2*GIBBS_SCALE_M**2)
    nll=-np.logaddexp(log_likelihood[:,0],log_likelihood[:,1])+np.log(2.)
    selection=select_profile(depths,nll,free,prior)
    result.update(status='completed_acquired_depth_profile',depths_m=depths.tolist(),
        surface_mean_clipped_distance_squared_m2=surface.tolist(),free_mean_clipped_depth_gap_squared_m2=free.tolist(),
        neutral_mixture_nll=nll.tolist(),per_shape_evidence=evidence,selection=selection,fixed_ray_audit=ray_audit,
        width_height_orientation_and_layers_unchanged=True,shape_mixture_prior=[.5,.5])
    return result

def profile_model(mapper,model):
    for key in ('full_observed_support','geometry_fit','outer_orientation','observed_layers'):
        if not model.audit.get(key):raise ValueError('B7 profile requires frozen B6 AB full observed model')
    rows=list(mapper.quality.items());keys=np.asarray([k for k,_ in rows],np.int64).reshape(-1,3)
    tracks=mapper.instance_support_index.memberships_for_keys(keys)
    q={key:np.asarray([row[key] for _,row in rows]) for key in rows[0][1]}
    outputs=[]
    for obj in model.objects:
        selected=tracks==obj['support_track'];cloud=q['point'][selected]
        assert digest(cloud)==obj['fit_input_points_sha256']
        outputs.append(profile_object(obj,cloud,q['bits'][selected],mapper.keyframes))
    return dict(schema='B7_fixed_acquired_depth_profile_v1',objects=outputs,
        no_source_geometry_or_hidden_dimensions=True,no_navigation_or_policy_execution=True,
        constants=dict(prior_depth_m=PRIOR_DEPTH_M,uniform_profile_samples=PROFILE_SAMPLES,
            prior_explicitly_added=True,pixel_stride=PIXEL_STRIDE,distance_clip_m=DISTANCE_CLIP_M,
            gibbs_scale_m=GIBBS_SCALE_M,heuristic_nll_band=PROFILE_BAND_NLL),
        actor_model_sha256=hashlib.sha256((B6/'model.py').read_bytes()).hexdigest(),
        B2_classification_sha256=REPAIR_SHA,
        limitations='Fixed-grid neutral pseudo-likelihood, not a calibrated depth posterior; unresolved anchors and all other B6 geometric priors are retained.')
