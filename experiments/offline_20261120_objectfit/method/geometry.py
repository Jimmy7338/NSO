"""Observed outer-plane fitting with explicit unresolved hidden dimensions.

No source mesh, hidden object dimensions or evaluation labels enter this file.
Normals only reject grossly incompatible plane samples. Point geometry sets
orientation; acquired camera/depth agreement sets the visible side of a plane.
"""
import numpy as np

PLANE_TOLERANCE_M = .025
MIN_PLANE_POINTS = 4
MIN_TANGENTIAL_SPAN_M = .15
MIN_VERTICAL_SPAN_M = .25
MAX_PAIR_POINTS = 64
MAX_PLANES = 6
MANHATTAN_TOLERANCE_RAD = np.deg2rad(10.)
VISIBLE_DEPTH_TOLERANCE_M = .10


def observed_view_directions(points, keyframes):
    """Only acquired frames with measured depth agreement can orient a face."""
    points = np.asarray(points, float)
    accum = np.zeros_like(points); count = np.zeros(len(points), int)
    for frame in keyframes:
        pose = frame.world_from_camera
        camera = (points-pose[:3, 3]) @ pose[:3, :3]
        z = camera[:, 2]
        positive = z > 1e-8
        ids = np.flatnonzero(positive)
        if not len(ids): continue
        uv = camera[ids] @ frame.intrinsic.T
        uv = np.rint(uv[:, :2]/uv[:, 2, None]).astype(int)
        h, w = frame.depth_m.shape
        inside = (uv[:, 0]>=0)&(uv[:, 0]<w)&(uv[:, 1]>=0)&(uv[:, 1]<h)
        ids, uv = ids[inside], uv[inside]
        depth = frame.depth_m[uv[:, 1], uv[:, 0]]
        valid = np.isfinite(depth)&(depth>0)&(np.abs(z[ids]-depth)<=VISIBLE_DEPTH_TOLERANCE_M)
        ids = ids[valid]
        delta = pose[:3, 3]-points[ids]
        delta /= np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-12)
        accum[ids] += delta; count[ids] += 1
    good = count>0
    accum[good] /= count[good, None]
    return accum, count


def _vertical_planes(points, normals):
    """Deterministic pair proposals, then TLS using spatially supported points."""
    remaining = np.arange(len(points)); planes = []
    for _ in range(MAX_PLANES):
        if len(remaining)<MIN_PLANE_POINTS: break
        p = points[remaining]; ns = normals[remaining]
        subset = np.unique(np.linspace(0,len(p)-1,min(len(p),MAX_PAIR_POINTS),dtype=int))
        best = None
        for ai, a in enumerate(subset):
            for b in subset[ai+1:]:
                tangent = p[b,:2]-p[a,:2]; length = np.linalg.norm(tangent)
                if length<MIN_TANGENTIAL_SPAN_M: continue
                normal = np.array([-tangent[1], tangent[0]])/length
                distance = np.abs((p[:,:2]-p[a,:2])@normal)
                compatible = (np.abs(ns[:,:2]@normal)>.7)&(np.abs(ns[:,2])<.7)
                mask = (distance<=PLANE_TOLERANCE_M)&compatible
                if mask.sum()<MIN_PLANE_POINTS: continue
                selected = p[mask]
                span_t = float(np.ptp(selected[:,:2]@(tangent/length)))
                span_z = float(np.ptp(selected[:,2]))
                if span_t<MIN_TANGENTIAL_SPAN_M or span_z<MIN_VERTICAL_SPAN_M: continue
                score = float(mask.sum()*span_t*span_z)
                if best is None or score>best[0]: best=(score,mask)
        if best is None: break
        selected = p[best[1]]; invalid_refinement=False
        for _ in range(3):
            center = selected[:,:2].mean(0)
            _, vectors = np.linalg.eigh(np.cov(selected[:,:2].T))
            normal = vectors[:,0]
            mask = (np.abs((p[:,:2]-center)@normal)<=PLANE_TOLERANCE_M)&(np.abs(ns[:,:2]@normal)>.7)&(np.abs(ns[:,2])<.7)
            if mask.sum()<MIN_PLANE_POINTS:
                invalid_refinement=True
                break
            selected = p[mask]
        if invalid_refinement: break
        tangent = np.array([-normal[1],normal[0]])
        span_t=float(np.ptp(selected[:,:2]@tangent)); span_z=float(np.ptp(selected[:,2]))
        if span_t<MIN_TANGENTIAL_SPAN_M or span_z<MIN_VERTICAL_SPAN_M: break
        indices=remaining[mask]
        planes.append(dict(normal=normal,indices=indices,
            score=float(len(indices)*span_t*span_z),horizontal_span_m=span_t,
            vertical_span_m=span_z,rms_m=float(np.sqrt(np.mean(((selected[:,:2]-center)@normal)**2)))))
        remaining=remaining[~mask]
    return planes


def fit_envelope(cloud, normals, prior_dimensions, view_directions=None):
    cloud=np.asarray(cloud,float); normals=np.asarray(normals,float)
    if cloud.ndim!=2 or cloud.shape[1]!=3 or normals.shape!=cloud.shape or not len(cloud) or not np.isfinite(cloud).all():
        raise ValueError('finite paired observed points/normals required')
    normals=np.nan_to_num(normals)
    views=np.zeros_like(cloud) if view_directions is None else np.asarray(view_directions,float)
    if views.shape!=cloud.shape: raise ValueError('one acquired view direction per point required')
    planes=_vertical_planes(cloud,normals)
    yaw=0.; orientation_observed=False
    if planes:
        strongest=max(planes,key=lambda p:p['score'])
        theta=np.arctan2(strongest['normal'][1],strongest['normal'][0])
        agreeing=[]
        for plane in planes:
            angle=np.arctan2(plane['normal'][1],plane['normal'][0])
            delta=np.angle(np.exp(4j*(angle-theta)))/4
            if abs(delta)<=MANHATTAN_TOLERANCE_RAD: agreeing.append(plane)
        moment=sum(p['score']*np.exp(4j*np.arctan2(p['normal'][1],p['normal'][0])) for p in agreeing)
        yaw=float(np.angle(moment)/4); orientation_observed=True
    R=np.array([[np.cos(yaw),-np.sin(yaw),0.],[np.sin(yaw),np.cos(yaw),0.],[0.,0.,1.]])
    local=cloud@R; local_views=views@R
    low,high=np.min(local,axis=0),np.max(local,axis=0)
    dims=np.asarray(prior_dimensions,float).copy(); center=(low+high)/2
    extents=[]; plane_audits=[]
    for axis in (0,1):
        bounds={}; tangent=1-axis
        for plane in planes:
            local_n=plane['normal']@R[:2,:2]
            if abs(local_n[axis])<np.cos(MANHATTAN_TOLERANCE_RAD): continue
            ids=plane['indices']; coordinate=float(np.median(local[ids,axis]))
            # A measured internal plane cannot be promoted to the envelope:
            # every observed point must lie on the supported interior side.
            low_boundary=bool(np.all(local[:,axis]>=coordinate-PLANE_TOLERANCE_M))
            high_boundary=bool(np.all(local[:,axis]<=coordinate+PLANE_TOLERANCE_M))
            view=float(np.mean(local_views[ids,axis]))
            side=None
            if low_boundary and high_boundary:
                if view<-.1: side='low'
                elif view>.1: side='high'
            elif low_boundary and view<-.1: side='low'
            elif high_boundary and view>.1: side='high'
            audit=dict(axis=axis,coordinate_m=coordinate,points=len(ids),
                horizontal_span_m=plane['horizontal_span_m'],vertical_span_m=plane['vertical_span_m'],
                rms_m=plane['rms_m'],camera_outward_component=view,
                observed_outer_boundary=side is not None,side=side)
            plane_audits.append(audit)
            if side is not None and (side not in bounds or len(ids)>bounds[side][1]):
                bounds[side]=(coordinate,len(ids))
        span=float(high[axis]-low[axis]); both='low' in bounds and 'high' in bounds
        if both and bounds['high'][0]-bounds['low'][0]>.08:
            a,b=bounds['low'][0],bounds['high'][0]
            dims[axis]=b-a; center[axis]=(a+b)/2
        else:
            both=False; dims[axis]=max(dims[axis],span)
            if 'low' in bounds: center[axis]=bounds['low'][0]+dims[axis]/2
            elif 'high' in bounds: center[axis]=bounds['high'][0]-dims[axis]/2
        extents.append(dict(axis=axis,both_faces_observed=both,
            observed_span_lower_bound_m=span,extent_interval_m=[span,float(dims[axis]) if both else None],
            unresolved_extent_uses_prior=not both,point_estimate_m=float(dims[axis]),
            low_face_coordinate_m=bounds.get('low',(None,))[0],high_face_coordinate_m=bounds.get('high',(None,))[0]))
    # An upper horizontal plane needs a two-dimensional footprint and no point
    # above it. An internal shelf slab never establishes the physical top.
    candidate=None; top_audit=None
    for height in sorted(set(np.round(cloud[:,2],3)),reverse=True):
        mask=(np.abs(cloud[:,2]-height)<=PLANE_TOLERANCE_M)&(np.abs(normals[:,2])>.7)&(views[:,2]>.1)
        if mask.sum()<MIN_PLANE_POINTS: continue
        selected=local[mask]; spans=np.ptp(selected[:,:2],axis=0)
        value=float(np.median(selected[:,2]))
        if min(spans)<MIN_TANGENTIAL_SPAN_M or np.any(cloud[:,2]>value+PLANE_TOLERANCE_M): continue
        candidate=value;top_audit=dict(points=int(mask.sum()),xy_spans_m=spans.tolist());break
    dims[2]=candidate if candidate is not None else max(dims[2],float(high[2])+.08)
    center[2]=dims[2]/2
    return dict(center=center@R.T,dims=dims,rotation=R,audit=dict(
        yaw_rad=yaw,orientation_observed=orientation_observed,
        dimensions_source='observed spatially supported outer planes; hidden extents retain prior',
        grounded_vertical_assumption=True,extent_uncertainty=extents,outer_plane_candidates=plane_audits,
        input_points=len(cloud),view_supported_points=int(np.count_nonzero(np.linalg.norm(views,axis=1)>0)),
        vertical_extent=dict(observed_height_lower_bound_m=float(high[2]),upper_bound_m=None,
            complete_extent_observed=False,upper_horizontal_face_candidate_m=candidate,
            upper_face_support=top_audit,point_estimate_m=float(dims[2])),
        constants=dict(plane_tolerance_m=PLANE_TOLERANCE_M,min_plane_points=MIN_PLANE_POINTS,
            min_tangential_span_m=MIN_TANGENTIAL_SPAN_M,min_vertical_span_m=MIN_VERTICAL_SPAN_M,
            max_pair_points=MAX_PAIR_POINTS,max_planes=MAX_PLANES,
            manhattan_tolerance_rad=float(MANHATTAN_TOLERANCE_RAD))))


def contradiction_fraction(points, areas, keyframes, classify_history):
    """Retained utility; new object-fit variants disable this likelihood."""
    areas=np.asarray(areas,float);state=classify_history(points,keyframes)
    resolved=state['strict_free']|state['measured_surface']
    total=float(areas[resolved].sum());free=float(areas[state['strict_free']].sum())
    return (free/total if total else 0.),dict(strict_free_area_m2=free,resolved_area_m2=total,
        unresolved_area_m2=float(areas[~resolved].sum()),measured_surface_area_m2=float(areas[state['measured_surface']].sum()))
