"""Class-independent candidates for acquiring an initial measured label plane.

Only already paid marker pixels, camera calibration/pose and the public motion
graph are inputs. Readiness is a frontal-label sampling/noise proxy, not a
probability of successful fitting, reconstruction area or semantic reward.
The caller owns attempt limits, paid-action accounting and the return latch.
"""
from collections import deque
from copy import deepcopy
import math

import numpy as np

from nso.instance_belief_v40 import PaidRGBDObservationV40,_components
from nso.observed_residual_v41 import observed_points_v41,pixel_mask_sha256_v41
from nso.primitive_navigation_v41 import PrimitiveStateV41,PublicPrimitiveGraphV41


MINIMUM_MARKER_PIXELS=4
FIT_MINIMUM_PIXELS=16
DEPTH_RELATIVE_SIGMA=.01
PLANE_RMS_LIMIT_M=.015
PLANE_SINGULAR_RATIO_LIMIT=.15
COMMON_LABEL_SIZE_M=(.32,.30)
FRONTAL_MINOR_SPREAD_M=min(COMMON_LABEL_SIZE_M)/math.sqrt(12.)
PROXY_MAXIMUM_AXIAL_DEPTH_M=min(PLANE_RMS_LIMIT_M/DEPTH_RELATIVE_SIGMA,
    PLANE_SINGULAR_RATIO_LIMIT*FRONTAL_MINOR_SPREAD_M/DEPTH_RELATIVE_SIGMA)
SAME_SIDE_MINIMUM_COSINE=.5


def _state(state):return dict(node=state.node,heading=state.heading)


def _camera_transform(graph,state):
    angle=state.heading*math.pi/6.;s,c=math.sin(angle),math.cos(angle)
    rotation=np.array([[s,0.,c],[-c,0.,s],[0.,-1.,0.]])
    center=np.array([*graph.positions[state.node],graph.camera_height_m])
    return center,rotation


def _distances(graph,current,home):
    """One public BFS and its reverse, rather than one route search per view."""
    distance={current:0};reverse={};queue=deque([current])
    while queue:
        state=queue.popleft()
        for action in ('left','right','forward'):
            try:successor=graph.successor(state,action)
            except ValueError:continue
            reverse.setdefault(successor,[]).append(state)
            if successor not in distance:
                distance[successor]=distance[state]+1;queue.append(successor)
    inbound={}
    if home in distance:
        inbound[home]=0;queue=deque([home])
        while queue:
            state=queue.popleft()
            for predecessor in reverse.get(state,[]):
                if predecessor not in inbound:
                    inbound[predecessor]=inbound[state]+1;queue.append(predecessor)
    return distance,inbound


class ObservedViewAcquisition:
    def __init__(self):
        self._records={};self._last_step=-1

    def observe(self,observation,accepted_associations):
        if type(observation) is not PaidRGBDObservationV40:
            raise TypeError('paid RGB-D observation required')
        if observation.paid_step!=self._last_step+1:
            raise ValueError('one consecutive paid observation per update required')
        if not isinstance(accepted_associations,(list,tuple)) or len(accepted_associations)>8:
            raise ValueError('at most eight measured associations required')
        packet_sha=observation.sha256();updated=[];skipped=[];seen=set()
        for row in accepted_associations:
            key=row['instance_id']
            if not isinstance(key,str) or not key or key in seen:raise ValueError('unique measured instance ID required')
            seen.add(key)
            if (row['observation_sha256']!=packet_sha or row['frame_id']!=observation.frame_id
                    or row['paid_step']!=observation.paid_step):raise ValueError('association and paid packet binding differs')
            raw=np.asarray(row.get('marker_pixel_indices',[]));support=np.asarray(row['pixel_indices'])
            if raw.size==0:
                skipped.append(dict(instance_id=key,reason='no_marker_pixels'));continue
            if (raw.ndim!=1 or raw.dtype.kind not in 'iu' or np.any(np.diff(raw)<=0)
                    or raw.min()<0 or raw.max()>=observation.depth_m.size or not np.isin(raw,support).all()):
                raise ValueError('sorted unique marker pixels inside measured association required')
            depth=observation.depth_m.ravel()[raw];pixels=raw[(depth>=.1)&(depth<=4.)]
            if len(pixels)<MINIMUM_MARKER_PIXELS:
                skipped.append(dict(instance_id=key,reason='insufficient_measured_marker_pixels'));continue
            mask=np.zeros(observation.depth_m.shape,bool);mask.ravel()[pixels]=True
            components=[component for component in _components(mask) if len(component)>=MINIMUM_MARKER_PIXELS]
            if len(components)!=1:
                skipped.append(dict(instance_id=key,reason='ambiguous_marker_components',
                    qualifying_components=len(components)));continue
            points=observed_points_v41(observation,pixels);anchor=np.median(points,axis=0)
            center=observation.world_from_camera[:3,3].copy();xy=anchor[:2]-center[:2]
            if np.linalg.norm(xy)<.1:
                skipped.append(dict(instance_id=key,reason='marker_bearing_too_short'));continue
            ys,xs=np.divmod(pixels,observation.depth_m.shape[1])
            record=dict(instance_id=key,anchor=anchor,camera_center=center,
                measured_lower=points.min(axis=0),measured_upper=points.max(axis=0),
                axial_depth=float(np.median(observation.depth_m.ravel()[pixels])),
                measured_range_xy=float(np.linalg.norm(xy)),marker_pixels=int(len(pixels)),
                touches_image_border=bool(np.any((xs==0)|(ys==0)|
                    (xs==observation.depth_m.shape[1]-1)|(ys==observation.depth_m.shape[0]-1))),
                frame_id=observation.frame_id,paid_step=observation.paid_step,observation_sha256=packet_sha,
                marker_mask_sha256=pixel_mask_sha256_v41(pixels,observation.depth_m.shape))
            self._records[key]=record;updated.append(self._receipt(record))
        if len(self._records)>8:raise ValueError('measured instance capacity exceeded')
        self._last_step=observation.paid_step
        return dict(schema='observed_view_acquisition.observation.v1',paid_step=self._last_step,
            observation_sha256=packet_sha,updated=updated,skipped=skipped,
            class_values_read=False,private_geometry_read=False,attempt_or_budget_mutation=False)

    @staticmethod
    def _receipt(record):
        return {key:deepcopy(value) for key,value in record.items() if key not in
            ('anchor','camera_center','measured_lower','measured_upper')}|dict(
                measured_anchor_world_m=record['anchor'].tolist(),
                previous_observing_camera_world_m=record['camera_center'].tolist())

    @staticmethod
    def _quality(record,graph,state,camera):
        center,rotation=_camera_transform(graph,state)
        p=(record['anchor']-center)@rotation;z=float(p[2])
        if not .1<=z<=4.:return None
        intrinsic=np.asarray(camera['intrinsic'],float)
        old=record['camera_center'][:2]-record['anchor'][:2]
        new=center[:2]-record['anchor'][:2];length=float(np.linalg.norm(new))
        if length<.1:return None
        same_side=float(old@new/(np.linalg.norm(old)*length))
        # Conservative framing of measured support, not an inferred full object
        # box. The padding is a declared 3-sigma depth-noise scale only.
        margin=3.*DEPTH_RELATIVE_SIGMA*max(z,record['axial_depth'])
        lower=record['measured_lower']-margin;upper=record['measured_upper']+margin
        corners=np.array([[x,y,a] for x in (lower[0],upper[0])
            for y in (lower[1],upper[1]) for a in (lower[2],upper[2])])
        local=(corners-center)@rotation
        framed=bool(np.all(local[:,2]>=.1))
        if framed:
            projected=local@intrinsic.T;uv=projected[:,:2]/projected[:,2,None]
            framed=bool(np.all(uv>=0) and np.all(uv[:,0]<camera['width']-1)
                and np.all(uv[:,1]<camera['height']-1))
        predicted_pixels=float(record['marker_pixels']*(record['axial_depth']/z)**2)
        sigma=DEPTH_RELATIVE_SIGMA*z;ratio=sigma/FRONTAL_MINOR_SPREAD_M
        ready=bool(framed and predicted_pixels>=FIT_MINIMUM_PIXELS
            and sigma<=PLANE_RMS_LIMIT_M and ratio<=PLANE_SINGULAR_RATIO_LIMIT)
        return dict(axial_depth_m=z,range_xy_m=length,same_side_cosine=same_side,
            anchor_optical_offset_rad=float(abs(math.atan2(p[0],p[2]))),
            measured_support_in_fov_with_margin=framed,measurement_margin_m=margin,
            predicted_marker_pixels=predicted_pixels,predicted_noise_scale_m=sigma,
            predicted_small_middle_ratio=ratio,readiness_proxy_satisfied=ready,
            proxy_maximum_axial_depth_m=PROXY_MAXIMUM_AXIAL_DEPTH_M,
            proxy_assumption='locally unchanged incidence for pixel scaling; uniform frontal 0.32x0.30m label and independent 1% axial-depth noise for SVD proxy',
            not_a_success_probability=True,not_reconstruction_area=True,not_semantic_reward=True)

    def candidates(self,graph,current,home,remaining,paid_states,camera,instances,planes,limit=32):
        if type(graph) is not PublicPrimitiveGraphV41:raise TypeError('validated public primitive graph required')
        graph.validate_state(current);graph.validate_state(home)
        if type(remaining) is not int or remaining<0 or type(limit) is not int or not 1<=limit<=32:
            raise ValueError('finite remaining budget and <=32 candidate cap required')
        k=np.asarray(camera['intrinsic'],float)
        if (k.shape!=(3,3) or not np.isfinite(k).all() or k[0,0]<=0 or k[1,1]<=0
                or type(camera['width']) is not int or type(camera['height']) is not int
                or min(camera['width'],camera['height'])<2):raise ValueError('finite camera calibration required')
        paid=set(paid_states)
        for state in paid:graph.validate_state(state)
        if len(instances)>8:raise ValueError('at most eight common observed instances required')
        distance,inbound=_distances(graph,current,home);positions=graph.original_nodes|{current.node}
        grouped={}
        for state,cost in distance.items():
            if state.node in positions and state not in paid and state in inbound:
                grouped.setdefault(state.node,[]).append((cost,state.heading,state))
        for rows in grouped.values():rows.sort(key=lambda r:(r[0],r[1]))
        ordered_nodes=sorted(grouped,key=lambda node:(grouped[node][0][0],node))
        eligible_count=sum(row['instance_id'] in self._records and not row.get('association_uncertain',True)
            and not planes.get(row['instance_id'],{}).get('conflict',False) for row in instances)
        protected_reserve=min(2*eligible_count,max(1,limit//2)) if eligible_count else 0
        seed_count=min(8,limit-protected_reserve)
        chosen=[grouped[node][0][2] for node in ordered_nodes[:seed_count]]
        proposed={};initial={};record_receipts=[]
        for instance in sorted(instances,key=lambda row:row['instance_id']):
            key=instance['instance_id'];record=self._records.get(key)
            if record is None or instance.get('association_uncertain',True):continue
            plane=planes.get(key,{})
            resolved=bool(plane.get('plane_fit') and plane['plane_fit'].get('accepted') and not plane.get('conflict',False))
            if plane.get('conflict',False):continue
            record_receipts.append(self._receipt(record));rows=[]
            current_quality=self._quality(record,graph,current,camera)
            for node in sorted(positions):
                delta=record['anchor'][:2]-np.asarray(graph.positions[node])
                if np.linalg.norm(delta)<.1:continue
                fraction=(math.atan2(delta[1],delta[0])%(2*math.pi))/(math.pi/6.)
                for heading in sorted({int(math.floor(fraction))%12,int(math.ceil(fraction))%12}):
                    state=PrimitiveStateV41(node,heading)
                    if state in paid or state not in distance or state not in inbound:continue
                    total=distance[state]+1+inbound[state]
                    if total>remaining:continue
                    quality=self._quality(record,graph,state,camera)
                    if quality is None or not quality['measured_support_in_fov_with_margin']:continue
                    if not resolved and quality['same_side_cosine']<SAME_SIDE_MINIMUM_COSINE-1e-12:continue
                    turn=(node==current.node and (current_quality is None or
                        not current_quality['measured_support_in_fov_with_margin'] or
                        current_quality['anchor_optical_offset_rad']>math.pi/12+1e-12))
                    closer=quality['range_xy_m']<=record['measured_range_xy']-.25+1e-12
                    improvement=bool(quality['readiness_proxy_satisfied'] or turn or closer)
                    if not resolved and not improvement:continue
                    rows.append(dict(instance_id=key,target=state,outbound_cost=distance[state],
                        return_cost=inbound[state],total_cost=total,quality_proxy=dict(quality,
                            centering_improvement=turn,closer_by_at_least_one_translation_step=closer,
                            source_frame_id=record['frame_id'],source_observation_sha256=record['observation_sha256']),
                        initialization_eligible=not resolved,turn=turn))
            rank=lambda row:(not row['quality_proxy']['readiness_proxy_satisfied'],row['total_cost'],
                row['quality_proxy']['range_xy_m'],row['target'].node,row['target'].heading)
            turn_rows=sorted((row for row in rows if row['turn']),key=rank)
            approach=sorted((row for row in rows if row['target'].node!=current.node),key=rank)
            selected=turn_rows[:1]
            for row in approach:
                if len(selected)>=2:break
                if all(row['target'].node!=old['target'].node for old in selected):selected.append(row)
            if not selected and resolved:selected=sorted(rows,key=rank)[:2]
            proposed[key]=selected
            for row in selected:
                if row['initialization_eligible']:initial[(key,row['target'])]=row
        protected={key:[] for key in proposed}
        for index in range(2):
            for key in sorted(proposed):
                if index>=len(proposed[key]):continue
                state=proposed[key][index]['target']
                if state not in chosen and len(chosen)<limit:chosen.append(state)
                if state in chosen:protected[key].append(state)
        # The old four-heading cap only limits generic fill; observed-facing
        # protected headings survive it and still count toward the total 32.
        quota={node:sum(s.node==node for s in chosen) for node in grouped}
        for _,node,heading,state in sorted((cost,node,heading,state) for node,rows in grouped.items() for cost,heading,state in rows):
            if len(chosen)>=limit:break
            if state not in chosen and quota.get(node,0)<4:
                chosen.append(state);quota[node]=quota.get(node,0)+1
        init=[{k:v for k,v in row.items() if k not in ('initialization_eligible','turn')}
            for (key,state),row in initial.items() if state in protected[key]]
        pool=dict(schema='observed_view_acquisition.pool.v1',current_state=_state(current),
            graph_input_sha256=graph.input_sha256,limit=limit,selected=len(chosen),
            reachable_primitive_states=len(distance),preferred_minimum_positions=seed_count,
            generic_fill_headings_per_position=4,protected_headings_exempt_from_generic_quota=True,
            maximum_protected_views_per_instance=2,maximum_forecast_views_per_instance_unchanged=8,
            selection_uses_class=False,selection_uses_structure_posterior=False,private_geometry_read=False,
            no_future_sensor_render=True,attempt_or_budget_mutation=False,
            measurement_records=record_receipts,initialization_proposal_count=len(init),
            protected_by_instance={key:[_state(s) for s in states] for key,states in protected.items()},
            candidates=[dict(**_state(state),outbound_primitive_cost=distance[state]) for state in chosen])
        return chosen,pool,protected,init
