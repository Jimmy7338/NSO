"""Bound oversized depth regions before the unchanged V41 evidence pipeline.

The only algorithmic change to V41 observe is proposal construction for raw
depth components exceeding the original extent cap. Their marker geometry or
previous measured support defines local candidates using the existing 0.6 m
bootstrap and 0.4 m growth limits. Overlap and multiple markers remain ambiguous.
Class values do not select regions. Normal-size components keep V41 behavior.

The explicit observe body below preserves V41 association, novelty suppression,
evidence caps and receipt fields. No World, private scene or evaluation is read.
"""
from copy import deepcopy

import numpy as np

from nso.instance_belief_v40 import PaidRGBDObservationV40, _components
from nso.observed_instances_v41 import (
    ObservedInstancesV41, _depth_components, _distances, _payload_sha, support_digest,
)


class ObservedInstancesLocal(ObservedInstancesV41):
    """Common S/G local geometry frontend; no plane-fit thresholds are changed."""

    def _proposal_components(self, valid, world, marker, rejected):
        height, width = valid.shape
        self._local_proposal_audit = []
        for pixels in _depth_components(valid, world.reshape(height,width,3), self.depth_link_distance_m):
            if len(pixels) < self.minimum_component_pixels:
                continue
            points = world[pixels]
            extent = np.ptp(points,axis=0)
            if np.max(extent) <= self.maximum_patch_extent_m:
                yield pixels
                continue
            raw = dict(component_first_pixel=int(pixels[0]), component_pixels=len(pixels))
            visible = np.zeros(valid.shape,bool)
            visible.ravel()[pixels] = marker.ravel()[pixels]
            seeds = [np.sort(p[:,0]*width+p[:,1]) for p in _components(visible)
                     if len(p) >= self.minimum_marker_pixels]
            locals_ = []
            # Candidates use PREVIOUS support, before observe accepts proposals.
            for key,state in self._instances.items():
                distances = _distances(points,self._support(state))
                patch = pixels[distances <= self.max_growth_distance_m]
                if (len(patch) >= self.minimum_component_pixels and
                        np.count_nonzero(distances <= self.association_distance_m) >= self.min_overlap_points):
                    locals_.append(dict(pixels=patch, known_instance=key, source='previous_measured_support'))
            for seed in seeds:
                # A marker already inside an existing support proposal uses
                # that proposal's stricter growth bound; it cannot enlarge it.
                if any(np.isin(seed,item['pixels']).sum() >= self.minimum_marker_pixels for item in locals_):
                    continue
                anchor = np.median(world[seed],axis=0)
                patch = pixels[np.linalg.norm(points-anchor,axis=1) <= self.bootstrap_radius_m]
                if len(patch) >= self.minimum_component_pixels:
                    locals_.append(dict(pixels=patch, known_instance=None, source='visible_marker_geometry'))
            reasons = {}
            for index,item in enumerate(locals_):
                marker_count = sum(np.isin(seed,item['pixels']).sum() >= self.minimum_marker_pixels for seed in seeds)
                if marker_count > 1:
                    reasons[index] = 'multiple_markers_within_local_support'
                for previous,other in enumerate(locals_[:index]):
                    if np.intersect1d(item['pixels'],other['pixels'],assume_unique=True).size:
                        reasons[index] = reasons[previous] = 'overlapping_local_support_proposals'
            audit = dict(**raw, measured_raw_extent_m=extent.tolist(), visible_marker_seeds=len(seeds),
                         local_candidates=len(locals_), accepted_local_candidates=0,
                         source='paid_pixels_and_previous_support_only',
                         bootstrap_radius_m=self.bootstrap_radius_m,
                         max_growth_distance_m=self.max_growth_distance_m,
                         maximum_patch_extent_m=self.maximum_patch_extent_m)
            for index,item in enumerate(locals_):
                if index in reasons:
                    key=item['known_instance']
                    if key is not None:
                        self._instances[key]['association_uncertain']=True
                    rejected.append(dict(**raw, reason=reasons[index],
                                         local_candidate_source=item['source']))
                else:
                    audit['accepted_local_candidates'] += 1
                    yield item['pixels']
            if not locals_:
                rejected.append(dict(**raw,reason='oversize_without_local_geometric_support'))
            self._local_proposal_audit.append(audit)

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
        for pixels in self._proposal_components(valid,world,marker,rejected):
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
                key=self._new_instance(p['anchor']);state=self._instances[key]
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
            local_proposal_audit=deepcopy(self._local_proposal_audit),
            proposal_construction='oversize regions bounded before instance association; original V41 remainder',
            no_new_support=not any(not row['no_new_support'] for row in accepted),
            instances=[self._public(k) for k in self._instances],association_uses_class=False,
            ground_truth_owner_used=False,future_observation_used=False,depth_convention='optical axial z'))


