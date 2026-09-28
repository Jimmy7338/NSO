"""Static, route-independent visible-surface evaluation using NumPy only.

Candidates are declared before trajectories. Reachability is an upstream
contract, not inferred here from a planner path. Finite area quadrature and
nearest-GT-projection visibility approximate observable-domain boundaries.
All prediction area is accounted for; prediction owner labels are never used.
"""
from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np


def _ro(value,dtype=None):
    result=np.array(value,dtype=dtype,copy=True);result.flags.writeable=False
    return result


def _positive(value,name):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
        raise ValueError(name+' must be positive finite')
    return float(value)


@dataclass(frozen=True)
class CandidateViewV40:
    intrinsic: np.ndarray
    world_from_camera: np.ndarray
    width: int
    height: int
    near_m: float=.1
    far_m: float=4.
    view_id: str='candidate'

    def __post_init__(self):
        k=np.asarray(self.intrinsic,dtype=float);t=np.asarray(self.world_from_camera,dtype=float)
        if type(self.width) is not int or type(self.height) is not int or min(self.width,self.height)<1:
            raise ValueError('positive integer image dimensions required')
        if k.shape!=(3,3) or not np.isfinite(k).all() or k[0,0]<=0 or k[1,1]<=0 or not np.allclose(k[2],[0,0,1],rtol=0,atol=1e-12) or abs(np.linalg.det(k))<1e-12:
            raise ValueError('finite nonsingular pinhole K required')
        if t.shape!=(4,4) or not np.isfinite(t).all() or not np.allclose(t[3],[0,0,0,1],rtol=0,atol=1e-12):
            raise ValueError('rigid world_from_camera required')
        if not np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),rtol=0,atol=1e-6) or not math.isclose(np.linalg.det(t[:3,:3]),1.,abs_tol=1e-6):
            raise ValueError('proper camera rotation required')
        near=_positive(self.near_m,'near_m');far=_positive(self.far_m,'far_m')
        if near>=far: raise ValueError('near_m must be below far_m')
        if not isinstance(self.view_id,str) or not self.view_id: raise ValueError('nonempty view_id required')
        object.__setattr__(self,'intrinsic',_ro(k));object.__setattr__(self,'world_from_camera',_ro(t))
        object.__setattr__(self,'near_m',near);object.__setattr__(self,'far_m',far)


def _mesh(vertices,triangles,*,allow_empty=False,instance_ids=None):
    v=np.asarray(vertices,dtype=float);t=np.asarray(triangles)
    if v.ndim!=2 or v.shape[1]!=3 or not np.isfinite(v).all(): raise ValueError('finite vertices[N,3] required')
    if t.ndim!=2 or t.shape[1]!=3 or t.dtype.kind not in 'iu': raise ValueError('integer triangles[M,3] required')
    if not len(t) and not allow_empty: raise ValueError('empty reference mesh is meaningless')
    if t.size and (t.min()<0 or t.max()>=len(v)): raise ValueError('triangle index outside vertices')
    xyz=v[t];cross=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]);twice=np.linalg.norm(cross,axis=1)
    if not np.isfinite(twice).all() or np.any(twice<=1e-12): raise ValueError('nonfinite or degenerate triangle rejected')
    normals=np.divide(cross,twice[:,None],out=np.zeros_like(cross),where=twice[:,None]>0)
    ids=None
    if instance_ids is not None:
        raw=np.asarray(instance_ids)
        if raw.shape!=(len(t),) or raw.dtype.kind not in 'iu' or np.any(raw< -1):
            raise ValueError('integer triangle_instance_id[M], background=-1, required')
        ids=np.asarray(raw,dtype=np.int64)
        if not np.any(ids>=0): raise ValueError('at least one target instance required')
        seen=set()
        for triangle,owner in zip(xyz,ids):
            key=(int(owner),tuple(sorted(tuple(float(x) for x in p) for p in triangle)))
            if key in seen: raise ValueError('duplicate reference triangle for one instance')
            seen.add(key)
    return _ro(v),_ro(t,np.int64),None if ids is None else _ro(ids),twice/2,normals


def _sample(v,t,areas,spacing,seed,max_samples):
    spacing=_positive(spacing,'sample_spacing_m')
    if type(seed) is not int or seed<0: raise ValueError('nonnegative integer sampling seed required')
    if type(max_samples) is not int or max_samples<1: raise ValueError('positive max_samples required')
    with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
        desired=areas/(spacing*spacing)
    if not np.isfinite(desired).all() or np.any(desired>max_samples): raise ValueError('declared surface sample limit exceeded')
    counts=np.maximum(1,np.ceil(desired).astype(np.int64))
    if int(counts.sum())>max_samples: raise ValueError('declared surface sample limit exceeded')
    if not len(t): return np.empty((0,3)),np.empty(0),np.empty(0,dtype=np.int64)
    rng=np.random.default_rng(seed);points=[];weights=[];which=[]
    golden=(math.sqrt(5)-1)/2
    for i,(triangle,area,count) in enumerate(zip(v[t],areas,counts)):
        n=int(count);u=(np.arange(n)+.5)/n;w=(np.arange(n)*golden+rng.random())%1
        s=np.sqrt(u);bary=np.column_stack((1-s,s*(1-w),s*w))
        points.append(bary@triangle);weights.append(np.full(n,float(area)/n));which.append(np.full(n,i,dtype=np.int64))
    return np.concatenate(points),np.concatenate(weights),np.concatenate(which)


def _closest(points,v,t,chunk=128):
    """Exact Euclidean point-to-triangle distances, with closest point/index."""
    count=len(points);best=np.full(count,np.inf);closest=np.zeros((count,3));index=np.full(count,-1,dtype=np.int64)
    xyz=v[t]
    for start in range(0,count,chunk):
        p=points[start:start+chunk,None,:];distance=best[start:start+len(p)]
        for base in range(0,len(t),chunk):
            tri=xyz[base:base+chunk];a,b,c=tri[:,0],tri[:,1],tri[:,2]
            e0=b-a;e1=c-a;n=np.cross(e0,e1);n2=np.sum(n*n,axis=1)
            signed=np.sum((p-a)*n,axis=-1)/n2
            projection=p-signed[...,None]*n
            d00=np.sum(e0*e0,axis=1);d01=np.sum(e0*e1,axis=1);d11=np.sum(e1*e1,axis=1)
            r=projection-a;d20=np.sum(r*e0,axis=-1);d21=np.sum(r*e1,axis=-1)
            denominator=d00*d11-d01*d01
            beta=(d11*d20-d01*d21)/denominator;gamma=(d00*d21-d01*d20)/denominator
            inside=(beta>=-1e-12)&(gamma>=-1e-12)&(beta+gamma<=1+1e-12)
            local=np.where(inside,np.sum((p-projection)**2,axis=-1),np.inf);where=projection.copy()
            for left,right in ((a,b),(b,c),(c,a)):
                edge=right-left;parameter=np.clip(np.sum((p-left)*edge,axis=-1)/np.sum(edge*edge,axis=1),0,1)
                q=left+parameter[...,None]*edge;d=np.sum((p-q)**2,axis=-1);take=d<local
                local=np.where(take,d,local);where=np.where(take[...,None],q,where)
            ids=np.argmin(local,axis=1);ds=local[np.arange(len(p)),ids];take=ds<distance
            distance[take]=ds[take]
            closest[start:start+len(p)][take]=where[np.arange(len(p)),ids][take]
            index[start:start+len(p)][take]=base+ids[take]
    return np.sqrt(best),closest,index


def _unoccluded(points,origin,v,t,epsilon=1e-7,chunk=128):
    xyz=v[t];clear=np.ones(len(points),bool)
    for start in range(0,len(points),chunk):
        target=points[start:start+chunk];delta=target-origin;length=np.linalg.norm(delta,axis=1)
        direction=delta/length[:,None];blocked=np.zeros(len(target),bool)
        for base in range(0,len(t),chunk):
            triangle=xyz[base:base+chunk];a=triangle[:,0];e1=triangle[:,1]-a;e2=triangle[:,2]-a
            p=np.cross(direction[:,None,:],e2[None,:,:]);det=np.sum(e1[None,:,:]*p,axis=-1)
            valid=np.abs(det)>1e-12;inverse=np.divide(1.,det,out=np.zeros_like(det),where=valid)
            offset=origin-a;u=np.sum(offset[None,:,:]*p,axis=-1)*inverse
            q=np.cross(offset,e1);w=np.sum(direction[:,None,:]*q[None,:,:],axis=-1)*inverse
            hit=np.sum(e2*q,axis=-1)[None,:]*inverse
            blocked|=np.any(valid&(u>=-1e-10)&(w>=-1e-10)&(u+w<=1+1e-10)&(hit>epsilon)&(hit<length[:,None]-epsilon),axis=1)
            if blocked.all(): break
        clear[start:start+len(target)]=~blocked
    return clear


def _visible(points,normals,views,v,t):
    visible=np.zeros(len(points),bool)
    for view in views:
        origin=view.world_from_camera[:3,3];local=(points-origin)@view.world_from_camera[:3,:3]
        depth=local[:,2]
        projected=local@view.intrinsic.T
        uv=np.divide(projected[:,:2],projected[:,2,None],out=np.full((len(points),2),np.inf),where=np.abs(projected[:,2,None])>1e-12)
        possible=(~visible)&(depth>=view.near_m)&(depth<=view.far_m)&(uv[:,0]>=-.5)&(uv[:,0]<view.width-.5)&(uv[:,1]>=-.5)&(uv[:,1]<view.height-.5)
        possible&=np.sum(normals*(origin-points),axis=1)>1e-10
        ids=np.flatnonzero(possible)
        if len(ids): visible[ids]=_unoccluded(points[ids],origin,v,t)
    return visible


def _hash_array(value):
    a=np.ascontiguousarray(value)
    return hashlib.sha256(f'{a.dtype.str}:{a.shape}:'.encode()+a.tobytes()).hexdigest()


@dataclass(frozen=True)
class ReferenceSurfaceV40:
    vertices: np.ndarray
    triangles: np.ndarray
    triangle_instance_id: np.ndarray
    triangle_normals: np.ndarray
    points: np.ndarray
    point_instance_id: np.ndarray
    area_weights: np.ndarray
    candidate_views: tuple
    sample_spacing_m: float
    seed: int
    full_target_sample_count: int
    fingerprint: str

    def _content(self):
        arrays={key:_hash_array(getattr(self,key)) for key in ('vertices','triangles','triangle_instance_id','triangle_normals','points','point_instance_id','area_weights')}
        views=[dict(view_id=v.view_id,width=v.width,height=v.height,near_m=v.near_m,far_m=v.far_m,
            intrinsic=_hash_array(v.intrinsic),world_from_camera=_hash_array(v.world_from_camera)) for v in self.candidate_views]
        return dict(arrays=arrays,candidate_views=views,sample_spacing_m=self.sample_spacing_m,seed=self.seed,
            full_target_sample_count=self.full_target_sample_count,
            depth_convention='optical axial z; near_m and far_m both clip camera z, not radial distance')

    def manifest(self):
        current=hashlib.sha256(json.dumps(self._content(),sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if current!=self.fingerprint: raise ValueError('frozen reference content changed')
        return dict(contract='v40.observable-surface-static-1',fingerprint=self.fingerprint,
            **self._content(),target_instances=[int(i) for i in np.unique(self.point_instance_id)],
            observable_sample_count=len(self.points),
            observable_area_m2_estimate=float(self.area_weights.sum()),
            route_independent=True,reachability='declared by upstream candidate generator; not inferred from a route',
            visibility='outward-facing, within declared camera frustum and axial depth limits, with no intervening scene triangle',
            approximation='deterministic finite triangle-area quadrature; visibility boundary is not exact polygon clipping')


def freeze_reference_v40(vertices,triangles,triangle_instance_id,candidate_views,*,sample_spacing_m=.10,seed=4001,max_samples=200000):
    v,t,ids,areas,normals=_mesh(vertices,triangles,instance_ids=triangle_instance_id)
    views=tuple(candidate_views)
    if not views or any(type(view) is not CandidateViewV40 for view in views): raise ValueError('predeclared CandidateViewV40 sequence required')
    if len({view.view_id for view in views})!=len(views): raise ValueError('candidate view IDs must be unique')
    target=np.flatnonzero(ids>=0);points,weights,tri=_sample(v,t[target],areas[target],sample_spacing_m,seed,max_samples)
    tri=target[tri];visible=_visible(points,normals[tri],views,v,t)
    sample_ids=ids[tri]
    missing=set(int(i) for i in np.unique(ids[ids>=0]))-set(int(i) for i in np.unique(sample_ids[visible]))
    if missing: raise ValueError('target instances have no observable quadrature samples: '+str(sorted(missing)))
    content=dict(vertices=v,triangles=t,triangle_instance_id=ids,triangle_normals=_ro(normals),
        points=_ro(points[visible]),point_instance_id=_ro(sample_ids[visible]),area_weights=_ro(weights[visible]),
        candidate_views=views,sample_spacing_m=float(sample_spacing_m),seed=seed,full_target_sample_count=len(points))
    temporary=ReferenceSurfaceV40(**content,fingerprint='')
    fingerprint=hashlib.sha256(json.dumps(temporary._content(),sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return ReferenceSurfaceV40(**content,fingerprint=fingerprint)


def evaluate_surface_v40(reference,pred_vertices,pred_triangles,*,C_map,threshold_m=.05,sample_spacing_m=None,seed=4002,max_samples=200000):
    if type(reference) is not ReferenceSurfaceV40: raise TypeError('frozen ReferenceSurfaceV40 required')
    manifest=reference.manifest();threshold=_positive(threshold_m,'threshold_m')
    if isinstance(C_map,bool) or not isinstance(C_map,(int,float)) or not math.isfinite(C_map) or not 0<=C_map<=1:
        raise ValueError('measured C_map in [0,1] required')
    spacing=reference.sample_spacing_m if sample_spacing_m is None else sample_spacing_m
    pv,pt,_,pa,_=_mesh(pred_vertices,pred_triangles,allow_empty=True)
    submitted_area=float(pa.sum());submitted_triangles=len(pt)
    # Exact geometric duplicates carry no new reconstructed surface. Remove
    # them without consulting GT, preventing precision inflation by copying TP.
    seen=set();keep=[]
    for index,triangle in enumerate(pv[pt]):
        key=tuple(sorted(tuple(float(x) for x in p) for p in triangle))
        if key not in seen: seen.add(key);keep.append(index)
    retained=np.asarray(keep,dtype=np.int64);pt=pt[retained];pa=pa[retained]
    pp,pw,_=_sample(pv,pt,pa,spacing,seed,max_samples)
    gt_dist,gt_point,gt_tri=_closest(pp,reference.vertices,reference.triangles)
    correct=gt_dist<=threshold;owner=reference.triangle_instance_id[gt_tri]
    target_match=correct&(owner>=0);observable=np.zeros(len(pp),bool)
    selected=np.flatnonzero(target_match)
    if len(selected): observable[selected]=_visible(gt_point[selected],reference.triangle_normals[gt_tri[selected]],
        reference.candidate_views,reference.vertices,reference.triangles)
    true_positive=target_match&observable
    background=correct&(owner<0);out_of_reference=target_match&~observable;extra=~correct
    # Every false area is allocated to the nearest task surface, including
    # errors whose nearest full-scene surface happens to be background.
    allocated=np.array(owner,copy=True)
    extra_ids=np.flatnonzero(extra)
    if len(extra_ids):
        target_tri=np.flatnonzero(reference.triangle_instance_id>=0)
        _,_,nearest=_closest(pp[extra_ids],reference.vertices,reference.triangles[target_tri])
        allocated[extra_ids]=reference.triangle_instance_id[target_tri[nearest]]
    recall_distance,_,_=_closest(reference.points,pv,pt)
    recalled=recall_distance<=threshold;per_instance=[]
    for instance in np.unique(reference.point_instance_id):
        ref=reference.point_instance_id==instance
        reference_area=float(reference.area_weights[ref].sum())
        tp=float(pw[true_positive&(owner==instance)].sum())
        fp=float(pw[extra&(allocated==instance)].sum())
        recall=float(reference.area_weights[ref&recalled].sum()/reference_area)
        precision=tp/(tp+fp) if tp+fp else 0.
        f1=2*precision*recall/(precision+recall) if precision+recall else 0.
        per_instance.append(dict(instance_id=int(instance),precision=precision,completeness=recall,recall=recall,f1=f1,
            reference_area_m2_estimate=reference_area,predicted_true_positive_area_m2=tp,allocated_false_positive_area_m2=fp,
            predicted_scored_area_m2=tp+fp,reference_samples=int(ref.sum())))
    macro_f1=float(np.mean([x['f1'] for x in per_instance]));total=float(pw.sum());correct_area=float(pw[correct].sum())
    accounting=dict(observable_target_true_positive_area_m2=float(pw[true_positive].sum()),
        extra_false_positive_area_m2=float(pw[extra].sum()),correct_background_area_m2=float(pw[background].sum()),
        correct_target_out_of_reference_area_m2=float(pw[out_of_reference].sum()),total_prediction_area_m2=total)
    if not math.isclose(sum(v for k,v in accounting.items() if k!='total_prediction_area_m2'),total,rel_tol=1e-10,abs_tol=1e-10):
        raise AssertionError('prediction area accounting is incomplete')
    return dict(contract='v40.observable-surface-static-1',reference_fingerprint=reference.fingerprint,
        C_map=float(C_map),macro_precision=float(np.mean([x['precision'] for x in per_instance])),
        macro_completeness=float(np.mean([x['recall'] for x in per_instance])),macro_f1=macro_f1,
        Q=macro_f1,J=float(C_map)*macro_f1,threshold_m=threshold,per_instance=per_instance,
        full_mesh_global_precision=correct_area/total if total else 0.,**accounting,
        prediction_samples=len(pp),reference_samples=len(reference.points),prediction_empty=not len(pt),
        submitted_prediction_triangles=submitted_triangles,canonical_prediction_triangles=len(pt),
        duplicate_prediction_triangles_removed=submitted_triangles-len(pt),
        submitted_prediction_area_m2=submitted_area,duplicate_prediction_area_removed_m2=submitted_area-float(pa.sum()),
        prediction_sample_spacing_m=float(spacing),prediction_seed=seed,
        precision_domain='nearest GT target projection must be observable; correct background/out-of-reference area reported separately',
        extra_false_positive_assignment='nearest GT task instance, without a distance cutoff; prediction owner labels not accepted',
        recall_domain='fixed observable reference samples for every task instance; nearest complete prediction triangle',
        no_semantic_weighting=True,no_prediction_roi_cropping=True,route_used_to_select_reference=False,
        reachability_certified_by_this_evaluator=False,static_geometry_only=True,
        visibility_boundary_approximation=manifest['approximation'],depth_convention=manifest['depth_convention'])
