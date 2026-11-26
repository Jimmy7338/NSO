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


def fit_envelope(cloud, normals, prior_dimensions, view_directions=None, *, outer_orientation=False):
    cloud=np.asarray(cloud,float); normals=np.asarray(normals,float)
    if cloud.ndim!=2 or cloud.shape[1]!=3 or normals.shape!=cloud.shape or not len(cloud) or not np.isfinite(cloud).all():
        raise ValueError('finite paired observed points/normals required')
    normals=np.nan_to_num(normals)
    views=np.zeros_like(cloud) if view_directions is None else np.asarray(view_directions,float)
    if views.shape!=cloud.shape: raise ValueError('one acquired view direction per point required')
    planes=_vertical_planes(cloud,normals)
    # Qualify a candidate in its own world-space coordinates before allowing
    # it to rotate the object's box/leg axes. A sparse internal diagonal can
    # have almost zero TLS residual without being an object outside face.
    orientation_planes=[];orientation_qualification=[]
    for plane in planes:
        ids=plane['indices'];n=plane['normal']
        coordinates=cloud[:,:2]@n;value=float(np.median(coordinates[ids]))
        low=bool(np.all(coordinates>=value-PLANE_TOLERANCE_M))
        high=bool(np.all(coordinates<=value+PLANE_TOLERANCE_M))
        view=float(np.mean(views[ids,:2]@n))
        qualified=(low and view<-.1) or (high and view>.1)
        orientation_qualification.append(dict(points=len(ids),normal_xy=n.tolist(),
            coordinate_m=value,all_points_on_positive_side=low,all_points_on_negative_side=high,
            camera_outward_component=view,qualified_outer_orientation=bool(qualified)))
        if qualified:orientation_planes.append(plane)
    if not outer_orientation:orientation_planes=planes
    yaw=0.; orientation_observed=False
    if orientation_planes:
        strongest=max(orientation_planes,key=lambda p:p['score'])
        theta=np.arctan2(strongest['normal'][1],strongest['normal'][0])
        agreeing=[]
        for plane in orientation_planes:
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
        outer_orientation=bool(outer_orientation),orientation_prior_yaw_rad=0.,
        orientation_qualification=orientation_qualification,
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


def fit_horizontal_layers(cloud, normals, views, prior_height, envelope_height):
    """Fit absolute visible slab faces, retain unconstrained old layer priors.

The inherited .08 m thickness anchors a single observed face. Opposite observed
faces whose proposed slabs overlap can constrain thickness directly. This
association is explicitly a heuristic, not identified hidden component truth.
"""
    cloud=np.asarray(cloud,float);normals=np.asarray(normals,float);views=np.asarray(views,float)
    thickness=.08
    prior=[dict(lower_m=float(a*prior_height),upper_m=float(a*prior_height+thickness),prior_id=i)
           for i,a in enumerate((.15,.5,.85))]
    valid=np.isfinite(normals).all(1)&np.isfinite(views).all(1)&(np.abs(normals[:,2])>.7)&(np.abs(views[:,2])>.1)
    remaining=np.flatnonzero(valid);planes=[]
    while len(remaining)>=MIN_PLANE_POINTS:
        points=cloud[remaining];best=None
        proposals=sorted({(float(z),int(np.sign(direction))) for z,direction in zip(points[:,2],views[remaining,2])})
        for height,side_sign in proposals:
            same_side=views[remaining,2]*side_sign>.1
            mask=(np.abs(points[:,2]-height)<=PLANE_TOLERANCE_M)&same_side
            if mask.sum()<MIN_PLANE_POINTS:continue
            value=float(np.median(points[mask,2]))
            mask=(np.abs(points[:,2]-value)<=PLANE_TOLERANCE_M)&same_side
            if mask.sum()<MIN_PLANE_POINTS:continue
            selected=points[mask];span=np.ptp(selected[:,:2],axis=0)
            covariance=np.cov(selected[:,:2].T)
            eigenvalues=np.linalg.eigvalsh(covariance)
            # Bounding spans alone allow a diagonal line to impersonate a
            # horizontal patch. Require 2-D spread at the same 2.5cm scale.
            if min(span)<MIN_TANGENTIAL_SPAN_M or eigenvalues[0]<PLANE_TOLERANCE_M**2:continue
            ids=remaining[mask];direction=float(np.mean(views[ids,2]))
            if abs(direction)<=.1:continue
            score=float(len(ids)*np.sqrt(max(float(np.linalg.det(covariance)),0.)))
            if best is None or score>best[0]:best=(score,mask,value,direction,span,eigenvalues)
        if best is None:break
        _,mask,value,direction,span,eigenvalues=best;ids=remaining[mask]
        plane_id=len(planes);side='top' if direction>0 else 'bottom'
        lower=value-thickness if side=='top' else value
        upper=value if side=='top' else value+thickness
        planes.append(dict(plane_id=plane_id,observed_face_z_m=value,observed_face_side=side,
            points=len(ids),xy_span_m=span.tolist(),xy_covariance_eigenvalues_m2=eigenvalues.tolist(),
            rms_m=float(np.sqrt(np.mean((cloud[ids,2]-value)**2))),
            mean_camera_vertical_direction=direction,
            lower_m=float(lower),upper_m=float(upper)))
        remaining=remaining[~mask]
    # Pair the closest opposite observed faces only if their old-thickness
    # slabs overlap. Acquired surfaces then replace the uncertain thickness.
    possible=[]
    for bottom in planes:
        if bottom['observed_face_side']!='bottom':continue
        for top in planes:
            if top['observed_face_side']!='top':continue
            distance=top['observed_face_z_m']-bottom['observed_face_z_m']
            overlap=max(bottom['lower_m'],top['lower_m'])<=min(bottom['upper_m'],top['upper_m'])
            if distance>1e-8 and overlap:possible.append((distance,bottom['plane_id'],top['plane_id']))
    used=set();measured=[];pairs=[]
    for _,bottom_id,top_id in sorted(possible):
        if bottom_id in used or top_id in used:continue
        bottom,top=planes[bottom_id],planes[top_id];used.update((bottom_id,top_id))
        measured.append(dict(lower_m=bottom['observed_face_z_m'],upper_m=top['observed_face_z_m'],
            source_plane_ids=[bottom_id,top_id],source_prior_ids=[],thickness_source='paired_observed_opposite_faces'))
        pairs.append(dict(bottom_plane_id=bottom_id,top_plane_id=top_id,
            measured_thickness_m=top['observed_face_z_m']-bottom['observed_face_z_m']))
    for plane in planes:
        if plane['plane_id'] not in used:
            measured.append(dict(lower_m=plane['lower_m'],upper_m=plane['upper_m'],
                source_plane_ids=[plane['plane_id']],source_prior_ids=[],thickness_source='unseen_thickness_0.08m_prior'))
    matched=set();associations=[]
    for measured_part in measured:
        ids=[]
        for old in prior:
            if max(measured_part['lower_m'],old['lower_m'])<=min(measured_part['upper_m'],old['upper_m']):
                matched.add(old['prior_id']);ids.append(old['prior_id'])
        associations.append(dict(source_plane_ids=measured_part['source_plane_ids'],overlapping_prior_ids=ids))
    parts=measured+[dict(lower_m=p['lower_m'],upper_m=p['upper_m'],source_plane_ids=[],
        source_prior_ids=[p['prior_id']],thickness_source='unconstrained_absolute_layer_prior') for p in prior if p['prior_id'] not in matched]
    merged=[]
    for part in sorted(parts,key=lambda p:(p['lower_m'],p['upper_m'])):
        if merged and part['lower_m']<=merged[-1]['upper_m']:
            merged[-1]['upper_m']=max(merged[-1]['upper_m'],part['upper_m'])
            merged[-1]['source_plane_ids']+=part['source_plane_ids'];merged[-1]['source_prior_ids']+=part['source_prior_ids']
            merged[-1]['thickness_source']='union_of_overlapping_component_intervals'
        else:merged.append(dict(part,source_plane_ids=list(part['source_plane_ids']),source_prior_ids=list(part['source_prior_ids'])))
    return merged,dict(prior_reference_height_m=float(prior_height),inherited_single_face_thickness_m=thickness,
        absolute_prior_intervals=prior,observed_horizontal_planes=planes,paired_opposite_faces=pairs,
        observation_to_prior_overlap_associations=associations,
        unconstrained_prior_ids=[p['prior_id'] for p in prior if p['prior_id'] not in matched],
        final_intervals=merged,component_count=len(merged),
        slab_bounds_exceed_envelope=bool(any(p['lower_m']<0 or p['upper_m']>envelope_height for p in merged)),
        box_leg_height_m=float(envelope_height),
        interpretation='heuristic observed component constraints plus retained unobserved absolute priors; no layer completeness claim')


def contradiction_fraction(points, areas, keyframes, classify_history):
    """Retained utility; new object-fit variants disable this likelihood."""
    areas=np.asarray(areas,float);state=classify_history(points,keyframes)
    resolved=state['strict_free']|state['measured_surface']
    total=float(areas[resolved].sum());free=float(areas[state['strict_free']].sum())
    return (free/total if total else 0.),dict(strict_free_area_m2=free,resolved_area_m2=total,
        unresolved_area_m2=float(areas[~resolved].sum()),measured_surface_area_m2=float(areas[state['measured_surface']].sum()))
