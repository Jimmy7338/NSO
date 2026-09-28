"""Conservative paid RGB-D support tracking; no simulator or GT instance input.

Exact-color markers bootstrap identities. Subsequent depth-connected patches
must overlap one and only one instance's previously measured support. This is
not natural object segmentation; contacting/oversize regions may be rejected.
"""
from collections import deque
from copy import deepcopy
import hashlib
import math

import numpy as np

from nso.instance_belief_v40 import (
    InstanceBeliefV40, PaidRGBDObservationV40, _components, _positive_int,
)


def _array_sha(value):
    a=np.ascontiguousarray(value)
    return hashlib.sha256(f'{a.dtype.str}:{a.shape}:'.encode()+a.tobytes()).hexdigest()


def support_digest(points_world_m):
    """SHA256 of float64 C-order points, with dtype/shape header; pixel order matters."""
    points=np.asarray(points_world_m,dtype=np.float64)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():
        raise ValueError('finite world points[N,3] required')
    return _array_sha(points)


def _payload_sha(observation):
    # Deliberately exclude caller-provided ID/step: renaming a paid packet must
    # not create another observation with independent evidence.
    return hashlib.sha256(''.join(_array_sha(getattr(observation,k)) for k in
        ('rgb','depth_m','intrinsic','world_from_camera')).encode()).hexdigest()


def _distances(points,support):
    result=np.full(len(points),np.inf)
    for start in range(0,len(points),128):
        p=points[start:start+128]
        for base in range(0,len(support),512):
            d=np.linalg.norm(p[:,None,:]-support[None,base:base+512,:],axis=2).min(axis=1)
            result[start:start+len(p)]=np.minimum(result[start:start+len(p)],d)
    return result


def _depth_components(valid,world,link):
    height,width=valid.shape;seen=np.zeros_like(valid)
    horizontal=valid[:,:-1]&valid[:,1:]&(np.linalg.norm(world[:,:-1]-world[:,1:],axis=2)<=link)
    vertical=valid[:-1]&valid[1:]&(np.linalg.norm(world[:-1]-world[1:],axis=2)<=link)
    for y,x in zip(*np.nonzero(valid)):
        if seen[y,x]: continue
        queue=deque([(int(y),int(x))]);seen[y,x]=True;pixels=[]
        while queue:
            yy,xx=queue.popleft();pixels.append(yy*width+xx)
            neighbors=[]
            if yy and vertical[yy-1,xx]: neighbors.append((yy-1,xx))
            if yy+1<height and vertical[yy,xx]: neighbors.append((yy+1,xx))
            if xx and horizontal[yy,xx-1]: neighbors.append((yy,xx-1))
            if xx+1<width and horizontal[yy,xx]: neighbors.append((yy,xx+1))
            for ny,nx in neighbors:
                if not seen[ny,nx]: seen[ny,nx]=True;queue.append((ny,nx))
        yield np.asarray(sorted(pixels),dtype=np.int64)


class ObservedInstancesV41(InstanceBeliefV40):
    """Geometry association shared by S/G; semantics condition beliefs only."""
    def __init__(self,*,palette,structure_names,class_structure_prior,mode='S',
                 geometry_prior=None,near_m=.1,far_m=4.,voxel_size_m=.05,
                 association_distance_m=.12,min_overlap_points=4,min_overlap_fraction=.1,
                 max_growth_distance_m=.4,maximum_patch_extent_m=2.5,
                 depth_link_distance_m=.15,bootstrap_radius_m=.6,
                 minimum_component_pixels=16,minimum_marker_pixels=4,
                 minimum_novel_points=4,minimum_novel_fraction=.05,
                 minimum_semantic_views=2,maximum_instances=64,
                 maximum_support_points=4096,maximum_evidence_frames=32,
                 per_view_geometry_cap=6.,total_geometry_cap=24.):
        super().__init__(palette=palette,structure_names=structure_names,
            class_structure_prior=class_structure_prior,mode=mode,geometry_prior=geometry_prior,
            minimum_component_pixels=minimum_component_pixels,minimum_depth_pixels=minimum_marker_pixels,
            minimum_semantic_views=minimum_semantic_views,maximum_instances=maximum_instances,
            maximum_views=maximum_evidence_frames,per_view_geometry_cap=per_view_geometry_cap,
            total_geometry_cap=total_geometry_cap)
        for name,value in (('near_m',near_m),('far_m',far_m),('voxel_size_m',voxel_size_m),
            ('association_distance_m',association_distance_m),('max_growth_distance_m',max_growth_distance_m),
            ('maximum_patch_extent_m',maximum_patch_extent_m),('depth_link_distance_m',depth_link_distance_m),
            ('bootstrap_radius_m',bootstrap_radius_m)):
            if isinstance(value,bool) or not math.isfinite(value) or value<=0: raise ValueError(name+' must be positive finite')
            setattr(self,name,float(value))
        if near_m>=far_m: raise ValueError('near_m must be below far_m')
        for name,value in (('min_overlap_fraction',min_overlap_fraction),('minimum_novel_fraction',minimum_novel_fraction)):
            if isinstance(value,bool) or not math.isfinite(value) or not 0<value<=1: raise ValueError(name+' must be in (0,1]')
            setattr(self,name,float(value))
        for name,value in (('min_overlap_points',min_overlap_points),('minimum_marker_pixels',minimum_marker_pixels),
            ('minimum_novel_points',minimum_novel_points),('maximum_support_points',maximum_support_points),
            ('maximum_evidence_frames',maximum_evidence_frames)):
            setattr(self,name,_positive_int(value,name))
        self._payloads=set()

    def _support(self,state):
        return np.asarray(list(state['support'].values()),dtype=float).reshape(-1,3)

    def _public(self,key):
        state=self._instances[key];seen=state['seen_classes']
        label=next(iter(seen)) if len(seen)==1 else None
        qualified=label is not None and len(state['class_views'][label])>=self.minimum_semantic_views
        use=self.mode=='S' and qualified and not state['association_uncertain']
        prior=self.class_priors[label] if use else self.geometry_prior
        log=np.log(prior)+state['geometry_scores'];posterior=np.exp(log-log.max());posterior/=posterior.sum()
        return dict(instance_id=key,anchor_world_m=state['anchor'].tolist(),
            support_points_world_m=self._support(state).tolist(),support_sha256=_array_sha(self._support(state)),
            structure_names=list(self.names),structure_probabilities=posterior.tolist(),
            geometry_log_scores=state['geometry_scores'].tolist(),geometry_prior=self.geometry_prior.tolist(),
            active_structure_prior=prior.tolist(),observed_class=label,semantic_conditioning_used=bool(use),
            class_conflict=len(seen)>1,association_uncertain=state['association_uncertain'],
            distinct_class_supports={k:len(v) for k,v in state['class_views'].items()},
            geometric_feedback_frames=len(state['geometry_views']),novel_support_frames=state['evidence_frames'],
            accepted_observations=state['accepted_observations'],probability_calibrated=False,
            marker_anchor_world_m=state['anchor'].tolist(),marker_observation_sha256=state['marker_sha'],
            marker_frame_id=state['marker_frame_id'],marker_paid_step=state['marker_paid_step'])

    def snapshot(self):
        return deepcopy(dict(paid_step=self._last_step,mode=self.mode,
            instances=[self._public(key) for key in self._instances],
            detection='paid exact-color marker bootstrap plus unique depth-support overlap; not natural segmentation',
            depth_convention='optical axial z'))

    def geometry_snapshot(self):
        return deepcopy(dict(paid_step=self._last_step,instances=[dict(instance_id=key,
            anchor_world_m=state['anchor'].tolist(),support_points_world_m=self._support(state).tolist(),
            geometry_prior=self.geometry_prior.tolist(),geometry_log_scores=state['geometry_scores'].tolist(),
            geometric_feedback_frames=sorted(state['geometry_views']),novel_support_frames=state['evidence_frames'],
            accepted_observations=state['accepted_observations'],association_uncertain=state['association_uncertain'])
            for key,state in self._instances.items()],paid_frame_associations=deepcopy(self._frames)))

    def observe(self,observation):
        if type(observation) is not PaidRGBDObservationV40: raise TypeError('strict PaidRGBDObservationV40 required')
        if observation.frame_id in self._frames or observation.paid_step!=self._last_step+1:
            raise ValueError('new frame and consecutive paid_step starting at zero required')
        payload_sha=_payload_sha(observation)
        # Pixel equality cannot distinguish a copied packet from a legitimately
        # paid static/noiseless observation. Action provenance belongs to the
        # router/sensor boundary; this ledger only suppresses repeated evidence.
        duplicate_measurement=payload_sha in self._payloads
        packet_sha=observation.sha256();height,width=observation.depth_m.shape
        y,x=np.indices((height,width));rays=np.column_stack((x.ravel(),y.ravel(),np.ones(height*width)))@np.linalg.inv(observation.intrinsic).T
        camera=rays*observation.depth_m.ravel()[:,None]
        world=camera@observation.world_from_camera[:3,:3].T+observation.world_from_camera[:3,3]
        valid=(observation.depth_m>=self.near_m)&(observation.depth_m<=self.far_m)
        labels={label:np.all(observation.rgb==np.asarray(color,np.uint8),axis=-1)&valid for label,color in self.palette.items()}
        marker=np.logical_or.reduce(list(labels.values()));proposals=[];rejected=[]
        # All proposals use the previous frame's support, never support appended
        # by another component during this observation.
        for pixels in _depth_components(valid,world.reshape(height,width,3),self.depth_link_distance_m):
            if len(pixels)<self.minimum_component_pixels: continue
            points=world[pixels];item=dict(component_first_pixel=int(pixels[0]),component_pixels=len(pixels))
            if np.max(np.ptp(points,axis=0))>self.maximum_patch_extent_m:
                rejected.append(dict(**item,reason='oversize_depth_component'));continue
            distances={key:_distances(points,self._support(state)) for key,state in self._instances.items()}
            counts={key:int((dist<=self.association_distance_m).sum()) for key,dist in distances.items()}
            candidates=[key for key,count in counts.items() if count>=self.min_overlap_points]
            if len(candidates)>1:
                for key in candidates: self._instances[key]['association_uncertain']=True
                rejected.append(dict(**item,reason='ambiguous_geometric_association',candidate_instances=candidates));continue
            if candidates:
                key=candidates[0];distance=distances[key]
                if counts[key]/len(points)<self.min_overlap_fraction:
                    rejected.append(dict(**item,reason='insufficient_overlap_fraction',candidate_instances=candidates));continue
                keep=distance<=self.max_growth_distance_m;pixels=pixels[keep];points=points[keep]
                if len(pixels)<self.minimum_component_pixels:
                    rejected.append(dict(**item,reason='insufficient_bounded_support'));continue
                proposals.append(dict(item=item,key=key,pixels=pixels,points=points,bootstrap=False))
                continue
            patch_marker=np.zeros((height,width),bool);patch_marker.ravel()[pixels]=marker.ravel()[pixels]
            seeds=[p for p in _components(patch_marker) if len(p)>=self.minimum_marker_pixels]
            if len(seeds)!=1:
                rejected.append(dict(**item,reason='unregistered_without_marker' if not seeds else 'ambiguous_bootstrap_markers'));continue
            marker_pixels=seeds[0][:,0]*width+seeds[0][:,1];anchor=np.median(world[marker_pixels],axis=0)
            if any(np.min(dist)<=self.max_growth_distance_m for dist in distances.values()):
                rejected.append(dict(**item,reason='near_existing_support_without_unique_overlap'));continue
            keep=np.linalg.norm(points-anchor,axis=1)<=self.bootstrap_radius_m
            pixels=pixels[keep];points=points[keep]
            if len(pixels)<self.minimum_component_pixels:
                rejected.append(dict(**item,reason='insufficient_bootstrap_support'));continue
            proposals.append(dict(item=item,key=None,pixels=pixels,points=points,anchor=anchor,bootstrap=True))
        # Several components claiming one instance indicate a split/occlusion.
        # Reject the whole many-to-one assignment rather than pick a favorable patch.
        counts={key:sum(p['key']==key for p in proposals) for key in self._instances}
        ambiguous_bootstraps=set()
        for i,p in enumerate(proposals):
            if not p['bootstrap']: continue
            for j,other in enumerate(proposals[:i]):
                if other['bootstrap'] and _distances(p['points'],other['points']).min()<=self.association_distance_m:
                    ambiguous_bootstraps.update((i,j))
        accepted=[];frame_instances={}
        for proposal_index,p in enumerate(proposals):
            key=p['key'];pixels=p['pixels'];points=p['points']
            if proposal_index in ambiguous_bootstraps:
                rejected.append(dict(**p['item'],reason='ambiguous_same_frame_bootstrap'));continue
            if key is not None and counts[key]>1:
                self._instances[key]['association_uncertain']=True
                rejected.append(dict(**p['item'],reason='multiple_components_claim_instance',candidate_instances=[key]));continue
            if key is None:
                if duplicate_measurement:
                    rejected.append(dict(**p['item'],reason='duplicate_measurement_cannot_bootstrap'));continue
                if len(self._instances)>=self.maximum_instances:
                    rejected.append(dict(**p['item'],reason='instance_capacity'));continue
                key=super()._new_instance(p['anchor']);state=self._instances[key]
                state.update(support={},evidence_frames=0,marker_sha=packet_sha,
                    marker_frame_id=observation.frame_id,marker_paid_step=observation.paid_step)
            state=self._instances[key];previous=self._support(state)
            novel=np.ones(len(points),bool) if not len(previous) else _distances(points,previous)>self.voxel_size_m
            if duplicate_measurement: novel[:]=False
            novel_voxels=set(tuple(int(v) for v in row) for row in np.floor(points[novel]/self.voxel_size_m).astype(np.int64))
            capacity=state['evidence_frames']<self.maximum_evidence_frames
            eligible=capacity and len(novel_voxels)>=self.minimum_novel_points and float(novel.mean())>=self.minimum_novel_fraction
            # Fixed-first sample per world voxel: identical measured support
            # does not move indefinitely with tiny noise or changed pixel grids.
            added=0
            for point in (() if duplicate_measurement else points):
                cell=tuple(int(v) for v in np.floor(point/self.voxel_size_m).astype(np.int64))
                if cell not in state['support'] and len(state['support'])<self.maximum_support_points:
                    state['support'][cell]=point.copy();added+=1
            if not added: eligible=False
            if eligible: state['evidence_frames']+=1
            state['accepted_observations']+=1
            if not duplicate_measurement: state['association_uncertain']=False
            marker_pixels=pixels[marker.ravel()[pixels]]
            for label,mask in labels.items():
                if not duplicate_measurement and int(mask.ravel()[pixels].sum())>=self.minimum_marker_pixels:
                    state['seen_classes'].add(label)
                    if eligible: state['class_views'][label].add(observation.frame_id)
            support_sha=support_digest(points)
            frame_instances[key]=dict(pixel_indices=pixels.tolist(),support_sha256=support_sha,
                geometry_feedback_eligible=bool(eligible),feedback_applied=False,
                duplicate_measurement=bool(duplicate_measurement),no_new_support=not bool(added))
            accepted.append(dict(**p['item'],instance_id=key,pixel_indices=pixels.tolist(),
                points_world_m=points.tolist(),marker_pixel_indices=marker_pixels.tolist(),
                frame_id=observation.frame_id,paid_step=observation.paid_step,
                observation_sha256=packet_sha,support_sha256=support_sha,novel_support_voxels=len(novel_voxels),
                duplicate_measurement=bool(duplicate_measurement),no_new_support=not bool(added),
                geometry_feedback_eligible=bool(eligible),association='marker_bootstrap' if p['bootstrap'] else 'unique_measured_support_overlap',
                marker_anchor_world_m=state['anchor'].tolist(),marker_observation_sha256=state['marker_sha'],
                marker_frame_id=state['marker_frame_id'],marker_paid_step=state['marker_paid_step']))
        self._frames[observation.frame_id]=dict(step=observation.paid_step,observation_sha256=packet_sha,
            payload_sha256=payload_sha,duplicate_measurement=bool(duplicate_measurement),instances=frame_instances)
        self._payloads.add(payload_sha);self._last_step=observation.paid_step
        return deepcopy(dict(frame_id=observation.frame_id,paid_step=observation.paid_step,
            observation_sha256=packet_sha,payload_sha256=payload_sha,accepted=accepted,rejected=rejected,
            duplicate_measurement=bool(duplicate_measurement),
            no_new_support=not any(not row['no_new_support'] for row in accepted),
            instances=[self._public(k) for k in self._instances],association_uses_class=False,
            ground_truth_owner_used=False,future_observation_used=False,depth_convention='optical axial z'))

    def apply_geometry_feedback(self,instance_id,*,frame_id,observation_sha256,log_likelihoods):
        if instance_id not in self._instances or frame_id not in self._frames: raise ValueError('known instance and paid frame required')
        frame=self._frames[frame_id]
        if frame['step']!=self._last_step or frame['observation_sha256']!=observation_sha256:
            raise ValueError('current paid frame and exact observation SHA required')
        if instance_id not in frame['instances']: raise ValueError('current frame was not uniquely associated with this instance')
        scores=np.asarray(log_likelihoods,dtype=float)
        if scores.shape!=(len(self.names),) or not np.isfinite(scores).all(): raise ValueError('finite per-structure paid geometry scores required')
        association=frame['instances'][instance_id];state=self._instances[instance_id]
        applied=association['geometry_feedback_eligible'] and not association['feedback_applied']
        if applied:
            scores=np.maximum(scores-scores.max(),-self.per_view_geometry_cap)
            state['geometry_scores']+=scores;state['geometry_scores']-=state['geometry_scores'].max()
            state['geometry_scores']=np.maximum(state['geometry_scores'],-self.total_geometry_cap)
            state['geometry_views'].add(frame_id);association['feedback_applied']=True
        return deepcopy(dict(instance_id=instance_id,frame_id=frame_id,paid_step=frame['step'],
            observation_sha256=observation_sha256,support_sha256=association['support_sha256'],applied=bool(applied),
            reason='paid_novel_support_applied' if applied else 'repeated_support_or_feedback_or_history_cap',
            caller_must_supply_paid_geometry_scores=True,instance=self._public(instance_id)))
