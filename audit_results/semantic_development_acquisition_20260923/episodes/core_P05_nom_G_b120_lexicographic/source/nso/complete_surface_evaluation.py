"""Full-surface evaluator preserving every numerically valid positive-area face.

Derived from the frozen V40 evaluator. Its absolute 1e-12 cross-norm floor
rejects valid submicrometre TSDF facets. This adapter removes that area floor,
not any mesh face; zero/nonfinite/numerically unresolved faces still fail.
Sampling order, weights, distance functions, reference, thresholds, ownership,
false-positive accounting and macro aggregation remain the original algorithm.
Old evaluator and historical evidence are unchanged.
"""
import math
import numpy as np
from nso.surface_evaluation_v40 import (ReferenceSurfaceV40, _positive, _ro,
    _sample, _closest, _visible)

LEGACY_EVALUATOR_SHA256 = '3e492e10e2065196f6d75161d7937efc98b7caea7792564ac47404ee1d73c035'

def _positive_area_mesh(vertices,triangles,*,allow_empty=False,instance_ids=None):
    v=np.asarray(vertices,dtype=float);t=np.asarray(triangles)
    if v.ndim!=2 or v.shape[1]!=3 or not np.isfinite(v).all(): raise ValueError('finite vertices[N,3] required')
    if t.ndim!=2 or t.shape[1]!=3 or t.dtype.kind not in 'iu': raise ValueError('integer triangles[M,3] required')
    if not len(t) and not allow_empty: raise ValueError('empty reference mesh is meaningless')
    if t.size and (t.min()<0 or t.max()>=len(v)): raise ValueError('triangle index outside vertices')
    xyz=v[t];cross=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]);twice=np.linalg.norm(cross,axis=1)
    if not np.isfinite(twice).all() or np.any(twice<=0): raise ValueError('nonfinite or degenerate triangle rejected')
    e0=xyz[:,1]-xyz[:,0];e1=xyz[:,2]-xyz[:,0]
    determinant=np.sum(e0*e0,axis=1)*np.sum(e1*e1,axis=1)-np.sum(e0*e1,axis=1)**2
    if np.any(determinant<=0) or not np.isfinite(determinant).all():
        raise ValueError('numerically unresolved triangle rejected; no surface silently removed')
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


def evaluate_complete_surface(reference,pred_vertices,pred_triangles,*,C_map,threshold_m=.05,sample_spacing_m=None,seed=4002,max_samples=200000):
    if type(reference) is not ReferenceSurfaceV40: raise TypeError('frozen ReferenceSurfaceV40 required')
    manifest=reference.manifest();threshold=_positive(threshold_m,'threshold_m')
    if isinstance(C_map,bool) or not isinstance(C_map,(int,float)) or not math.isfinite(C_map) or not 0<=C_map<=1:
        raise ValueError('measured C_map in [0,1] required')
    spacing=reference.sample_spacing_m if sample_spacing_m is None else sample_spacing_m
    pv,pt,_,pa,_=_positive_area_mesh(pred_vertices,pred_triangles,allow_empty=True)
    positive_subthreshold_faces=int(np.count_nonzero(pa<=5e-13))
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
        prediction_mesh_validation='strict_positive_area_without_absolute_area_floor',
        positive_subthreshold_faces_preserved=positive_subthreshold_faces,
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
