"""Paid RGB-D marker instances and separate per-instance structural beliefs.

This is a controlled visible-marker interface, not a natural object detector.
No World, simulator, mapper, evaluator, owner ID or future observation is read.
Geometry feedback is a caller-computed score bound to a previously accepted
paid frame/instance; provenance of arbitrary numeric scores is not provable by
this ledger and must be enforced at that caller's observation boundary.
"""
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, fields
import hashlib
import math
from collections.abc import Mapping

import numpy as np


def _readonly(value, dtype=None):
    out=np.array(value,dtype=dtype,copy=True)
    out.flags.writeable=False
    return out


def _positive_int(value,name):
    if type(value) is not int or value<1: raise ValueError(name+' must be a positive integer')
    return value


def _weights(value,count,name):
    out=np.asarray(value,dtype=float)
    if out.shape!=(count,) or not np.isfinite(out).all() or np.any(out<=0):
        raise ValueError(name+' requires one positive finite weight per structure')
    out=out/np.max(out);out=out/out.sum()
    if np.any(out<1e-9): raise ValueError(name+' must not exclude a structure with near-zero probability')
    return out


@dataclass(frozen=True)
class PaidRGBDObservationV40:
    frame_id: str
    paid_step: int
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsic: np.ndarray
    world_from_camera: np.ndarray

    def __post_init__(self):
        if not isinstance(self.frame_id,str) or not self.frame_id or len(self.frame_id)>256:
            raise ValueError('nonempty opaque frame_id required')
        if type(self.paid_step) is not int or self.paid_step<0:
            raise ValueError('nonnegative integer paid_step required')
        rgb=np.asarray(self.rgb);depth=np.asarray(self.depth_m)
        k=np.asarray(self.intrinsic,dtype=float);t=np.asarray(self.world_from_camera,dtype=float)
        if depth.ndim!=2 or min(depth.shape)<1 or depth.dtype.kind not in 'fiu' or not np.isfinite(depth).all() or np.any(depth<0):
            raise ValueError('finite nonnegative HxW axial depth required')
        if rgb.dtype!=np.uint8 or rgb.shape!=depth.shape+(3,):
            raise ValueError('aligned uint8 RGB required')
        if k.shape!=(3,3) or not np.isfinite(k).all() or k[0,0]<=0 or k[1,1]<=0 or not np.allclose(k[2],[0,0,1],atol=1e-10,rtol=0) or abs(np.linalg.det(k))<1e-12:
            raise ValueError('finite nonsingular pinhole intrinsic required')
        if t.shape!=(4,4) or not np.isfinite(t).all() or not np.allclose(t[3],[0,0,0,1],atol=1e-10,rtol=0):
            raise ValueError('finite homogeneous world_from_camera required')
        if not np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-6,rtol=0) or not math.isclose(np.linalg.det(t[:3,:3]),1.,abs_tol=1e-6):
            raise ValueError('camera rotation must be proper orthonormal')
        for name,value in (('rgb',rgb),('depth_m',depth),('intrinsic',k),('world_from_camera',t)):
            object.__setattr__(self,name,_readonly(value))

    @classmethod
    def from_mapping(cls,value):
        if not isinstance(value,Mapping): raise TypeError('observation mapping required')
        allowed={f.name for f in fields(cls)}
        if set(value)!=allowed:
            raise ValueError('observation whitelist violation; missing='+str(sorted(allowed-set(value)))+
                '; extra='+str(sorted(str(k) for k in set(value)-allowed)))
        return cls(**value)

    def sha256(self):
        h=hashlib.sha256()
        for part in (self.frame_id.encode(),str(self.paid_step).encode()):
            h.update(len(part).to_bytes(8,'big'));h.update(part)
        for name in ('rgb','depth_m','intrinsic','world_from_camera'):
            a=np.ascontiguousarray(getattr(self,name));header=f'{name}:{a.dtype.str}:{a.shape}:'.encode()
            h.update(len(header).to_bytes(8,'big'));h.update(header);h.update(a.tobytes())
        return h.hexdigest()


def _components(mask):
    """Deterministic four-connected components, no learned/GT segmentation."""
    seen=np.zeros(mask.shape,bool);height,width=mask.shape
    for y,x in zip(*np.nonzero(mask)):
        if seen[y,x]: continue
        queue=deque([(int(y),int(x))]);seen[y,x]=True;pixels=[]
        while queue:
            yy,xx=queue.popleft();pixels.append((yy,xx))
            for ny,nx in ((yy-1,xx),(yy+1,xx),(yy,xx-1),(yy,xx+1)):
                if 0<=ny<height and 0<=nx<width and mask[ny,nx] and not seen[ny,nx]:
                    seen[ny,nx]=True;queue.append((ny,nx))
        yield np.asarray(pixels,dtype=int)


class InstanceBeliefV40:
    """Distance-associated marker instances; geometry and semantics stay separate."""
    def __init__(self,*,palette,structure_names,class_structure_prior,mode='S',
                 geometry_prior=None,minimum_component_pixels=16,minimum_depth_pixels=8,
                 association_radius_m=.75,minimum_view_translation_m=.25,
                 maximum_range_m=4.,maximum_instances=64,maximum_views=16,
                 minimum_semantic_views=2,per_view_geometry_cap=6.,total_geometry_cap=24.):
        if mode not in ('S','G'): raise ValueError('mode must be S or G')
        names=tuple(structure_names)
        if len(names)<2 or any(not isinstance(n,str) or not n for n in names) or len(set(names))!=len(names):
            raise ValueError('at least two distinct structure names required')
        if not isinstance(palette,Mapping) or not palette: raise ValueError('explicit nonempty class palette required')
        self.palette={}
        for label,color in palette.items():
            if not isinstance(label,(str,int)) or isinstance(label,bool) or not str(label):
                raise ValueError('opaque nonempty class labels required')
            key=str(label);array=np.asarray(color)
            if key in self.palette or array.shape!=(3,) or array.dtype.kind not in 'iu' or np.any(array<0) or np.any(array>255):
                raise ValueError('unique class key and integer RGB triplet required')
            self.palette[key]=tuple(int(v) for v in array)
        if len(set(self.palette.values()))!=len(self.palette): raise ValueError('class colors must be distinct')
        if not isinstance(class_structure_prior,Mapping): raise ValueError('explicit class-structure table required')
        priors={str(k):v for k,v in class_structure_prior.items()}
        if len(priors)!=len(class_structure_prior) or set(priors)!=set(self.palette):
            raise ValueError('class-structure table must match palette labels, independently of color values')
        self.names,self.mode=names,mode;count=len(names)
        self.geometry_prior=_weights(np.ones(count) if geometry_prior is None else geometry_prior,count,'geometry prior')
        self.class_priors={k:_weights(v,count,'class prior') for k,v in priors.items()}
        self.minimum_component_pixels=_positive_int(minimum_component_pixels,'minimum_component_pixels')
        self.minimum_depth_pixels=_positive_int(minimum_depth_pixels,'minimum_depth_pixels')
        self.maximum_instances=_positive_int(maximum_instances,'maximum_instances')
        self.maximum_views=_positive_int(maximum_views,'maximum_views')
        self.minimum_semantic_views=_positive_int(minimum_semantic_views,'minimum_semantic_views')
        if self.minimum_semantic_views>self.maximum_views: raise ValueError('semantic views exceed history cap')
        for name,value in (('association_radius_m',association_radius_m),('minimum_view_translation_m',minimum_view_translation_m),
                           ('maximum_range_m',maximum_range_m),('per_view_geometry_cap',per_view_geometry_cap),('total_geometry_cap',total_geometry_cap)):
            if isinstance(value,bool) or not math.isfinite(value) or value<=0: raise ValueError(name+' must be positive finite')
            setattr(self,name,float(value))
        self._instances={};self._frames={};self._last_step=-1

    def _new_instance(self,anchor):
        key=f'instance_{len(self._instances):04d}'
        self._instances[key]=dict(anchor=np.array(anchor,copy=True),views=[],class_views={k:set() for k in self.palette},
            seen_classes=set(),geometry_scores=np.zeros(len(self.names)),geometry_views=set(),
            accepted_observations=0,association_uncertain=False)
        return key

    def _view(self,state,origin):
        for i,p in enumerate(state['views']):
            if np.linalg.norm(origin-p)<self.minimum_view_translation_m: return i,False
        if len(state['views'])>=self.maximum_views: return None,False
        state['views'].append(np.array(origin,copy=True))
        return len(state['views'])-1,True

    def _public(self,key):
        state=self._instances[key];seen=state['seen_classes']
        label=next(iter(seen)) if len(seen)==1 else None
        qualified=label is not None and len(state['class_views'][label])>=self.minimum_semantic_views
        use=self.mode=='S' and qualified and not state['association_uncertain']
        prior=self.class_priors[label] if use else self.geometry_prior
        log=np.log(prior)+state['geometry_scores'];posterior=np.exp(log-log.max());posterior/=posterior.sum()
        return dict(instance_id=key,anchor_world_m=state['anchor'].tolist(),structure_names=list(self.names),
            structure_probabilities=posterior.tolist(),geometry_log_scores=state['geometry_scores'].tolist(),
            geometry_prior=self.geometry_prior.tolist(),active_structure_prior=prior.tolist(),
            observed_class=label,semantic_conditioning_used=bool(use),class_conflict=len(seen)>1,
            association_uncertain=state['association_uncertain'],
            distinct_class_views={k:len(v) for k,v in state['class_views'].items()},
            distinct_geometric_views=len(state['geometry_views']),distinct_observed_views=len(state['views']),
            accepted_observations=state['accepted_observations'],probability_calibrated=False)

    def snapshot(self):
        return deepcopy(dict(paid_step=self._last_step,mode=self.mode,
            instances=[self._public(key) for key in self._instances],
            instance_detection='paid exact-color connected components with measured depth; not natural segmentation'))

    def geometry_snapshot(self):
        """Explicit semantic-free contract for exact S/G geometry comparison."""
        return deepcopy(dict(paid_step=self._last_step,instances=[dict(instance_id=key,
            anchor_world_m=state['anchor'].tolist(),geometry_prior=self.geometry_prior.tolist(),
            geometry_log_scores=state['geometry_scores'].tolist(),
            view_origins_world_m=[p.tolist() for p in state['views']],
            geometric_feedback_views=sorted(state['geometry_views']),
            accepted_observations=state['accepted_observations'],
            association_uncertain=state['association_uncertain']) for key,state in self._instances.items()],
            paid_frame_associations={key:dict(step=value['step'],instances=dict(value['instances']))
                for key,value in self._frames.items()}))

    def observe(self,observation):
        if type(observation) is not PaidRGBDObservationV40: raise TypeError('strict PaidRGBDObservationV40 required')
        if observation.frame_id in self._frames or observation.paid_step!=self._last_step+1:
            raise ValueError('new frame and consecutive paid_step starting at zero required')
        origin=observation.world_from_camera[:3,3];rotation=observation.world_from_camera[:3,:3]
        inverse=np.linalg.inv(observation.intrinsic);components=[];rejected=[]
        for label,color in self.palette.items():
            mask=np.all(observation.rgb==np.asarray(color,np.uint8),axis=-1)
            for pixels in _components(mask):
                if len(pixels)<self.minimum_component_pixels: continue
                y,x=pixels[:,0],pixels[:,1];depth=observation.depth_m[y,x]
                rays=np.column_stack((x,y,np.ones(len(x))))@inverse.T
                points=rays*depth[:,None]
                valid=(depth>0)&(np.linalg.norm(points,axis=1)<=self.maximum_range_m)
                item=dict(component_first_pixel=[int(y[0]),int(x[0])],class_label=label,
                    pixels=len(pixels),valid_depth_pixels=int(valid.sum()))
                if valid.sum()<self.minimum_depth_pixels:
                    rejected.append(dict(**item,reason='insufficient_measured_depth'));continue
                world=points[valid]@rotation.T+origin
                item['anchor_world_m']=np.median(world,axis=0).tolist()
                item['candidate_instances']=[key for key,state in self._instances.items()
                    if np.linalg.norm(np.asarray(item['anchor_world_m'])-state['anchor'])<=self.association_radius_m]
                components.append(item)
        components.sort(key=lambda c:c['component_first_pixel'])
        proposed={};ambiguous_new=set()
        for i,item in enumerate(components):
            for key in item['candidate_instances']: proposed.setdefault(key,[]).append(i)
            if not item['candidate_instances']:
                for j,other in enumerate(components[:i]):
                    if not other['candidate_instances'] and np.linalg.norm(np.asarray(item['anchor_world_m'])-other['anchor_world_m'])<=self.association_radius_m:
                        ambiguous_new.update((i,j))
        accepted=[];frame_instances={}
        for i,item in enumerate(components):
            candidates=item['candidate_instances']
            ambiguous=len(candidates)>1 or any(len(proposed[k])>1 for k in candidates) or i in ambiguous_new
            if ambiguous:
                for key in candidates: self._instances[key]['association_uncertain']=True
                rejected.append(dict(**item,reason='ambiguous_geometric_association'));continue
            if candidates: key=candidates[0]
            elif len(self._instances)<self.maximum_instances: key=self._new_instance(item['anchor_world_m'])
            else:
                rejected.append(dict(**item,reason='instance_capacity'));continue
            state=self._instances[key];state['association_uncertain']=False
            view,new_view=self._view(state,origin)
            state['accepted_observations']+=1
            label=item['class_label'];state['seen_classes'].add(label)
            if view is not None: state['class_views'][label].add(view)
            frame_instances[key]=view
            accepted.append(dict(**item,instance_id=key,view_id=view,new_distinct_view=new_view,
                geometry_feedback_eligible=view is not None and view not in state['geometry_views']))
        self._frames[observation.frame_id]=dict(step=observation.paid_step,instances=frame_instances,
            observation_sha256=observation.sha256())
        self._last_step=observation.paid_step
        return deepcopy(dict(frame_id=observation.frame_id,paid_step=observation.paid_step,
            observation_sha256=observation.sha256(),accepted=accepted,rejected=rejected,
            instances=[self._public(k) for k in self._instances],
            association_uses_class=False,ground_truth_owner_used=False,future_observation_used=False))

    def apply_geometry_feedback(self,instance_id,*,frame_id,log_likelihoods):
        if instance_id not in self._instances or frame_id not in self._frames:
            raise ValueError('known instance and previously paid frame required')
        frame=self._frames[frame_id]
        if instance_id not in frame['instances']:
            raise ValueError('frame was not unambiguously associated with this instance')
        scores=np.asarray(log_likelihoods,dtype=float)
        if scores.shape!=(len(self.names),) or not np.isfinite(scores).all():
            raise ValueError('finite caller-computed paid-geometry log likelihoods required')
        state=self._instances[instance_id];view=frame['instances'][instance_id]
        applied=view is not None and view not in state['geometry_views']
        if applied:
            scores=np.maximum(scores-scores.max(),-self.per_view_geometry_cap)
            state['geometry_scores']+=scores;state['geometry_scores']-=state['geometry_scores'].max()
            state['geometry_scores']=np.maximum(state['geometry_scores'],-self.total_geometry_cap)
            state['geometry_views'].add(view)
        return deepcopy(dict(instance_id=instance_id,frame_id=frame_id,paid_step=frame['step'],view_id=view,
            applied=applied,reason='paid_geometry_applied' if applied else 'repeated_view_or_history_cap',
            caller_must_supply_paid_geometry_scores=True,instance=self._public(instance_id)))
