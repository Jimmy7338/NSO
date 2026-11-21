"""Independent finite-distribution metrics; no model/generator imports."""
import math
import numpy as np

SIGMA_GRID_M=(.01,.015,.025,.04,.06,.09,.14,.22)
LEVELS=(.50,.80,.95)

def normalize(values):
    w=np.asarray(values,dtype=np.float64)
    if w.ndim!=1 or not len(w) or not np.isfinite(w).all() or np.any(w<0) or w.sum()<=0:
        raise ValueError('finite nonnegative nonempty masses required')
    return w/w.sum()

def depth_pmf(depths,masses):
    x=np.asarray(depths,dtype=np.float64);p=normalize(masses)
    if x.shape!=p.shape or not np.isfinite(x).all():raise ValueError('finite aligned depths required')
    unique,indices=np.unique(x,return_inverse=True)
    mass=np.bincount(indices,weights=p,minlength=len(unique));mass/=mass.sum()
    return unique,mass

def joint_posterior(energy,prior_mass,sigma_m,class_prior=(.5,.5)):
    e=np.asarray(energy,dtype=np.float64);q=normalize(prior_mass);h=normalize(class_prior)
    if e.shape!=(len(q),len(h)) or not np.isfinite(e).all() or np.any(e<0):raise ValueError('nonnegative finite K by H energy required')
    if not np.isfinite(sigma_m) or sigma_m<=0:raise ValueError('positive sigma required')
    with np.errstate(divide='ignore'):
        logw=np.log(q)[:,None]+np.log(h)[None,:]-e/(2*float(sigma_m)**2)
    logw-=np.max(logw);w=np.exp(logw);w/=w.sum()
    return w

def quantile(depths,masses,level):
    x,p=depth_pmf(depths,masses)
    if not 0<=level<=1:raise ValueError('quantile level outside [0,1]')
    return float(x[min(int(np.searchsorted(np.cumsum(p),level,side='left')),len(x)-1)])

def metrics(depths,masses,truth):
    x,p=depth_pmf(depths,masses);y=float(truth)
    if not np.isfinite(y):raise ValueError('finite truth required')
    crps=float(p@np.abs(x-y)-.5*np.sum(p[:,None]*p[None,:]*np.abs(x[:,None]-x[None,:])))
    median=quantile(x,p,.5);mean=float(p@x)
    intervals={}
    for level in LEVELS:
        lower=quantile(x,p,(1-level)/2);upper=quantile(x,p,(1+level)/2)
        intervals[f'{level:.2f}']=dict(lower_m=lower,upper_m=upper,width_m=upper-lower,covered=bool(lower-1e-12<=y<=upper+1e-12))
    return dict(crps_m=crps,median_m=median,median_absolute_error_m=abs(median-y),mean_m=mean,
        mean_absolute_error_m=abs(mean-y),intervals=intervals,depth_support_m=[float(x[0]),float(x[-1])],
        truth_outside_particle_support=bool(y<x[0]-1e-12 or y>x[-1]+1e-12),
        depth_values_m=x.tolist(),depth_probability_mass=p.tolist())

def projected_truth(truth,axis_world):
    axis=np.asarray(axis_world,dtype=np.float64)
    if axis.shape!=(3,) or not np.isfinite(axis).all() or np.linalg.norm(axis)<1e-12:raise ValueError('valid inferred depth axis required')
    norm=float(np.linalg.norm(axis));axis=axis/norm
    vertices=np.asarray(truth['object_vertices_world'],dtype=np.float64)
    R=np.asarray(truth['world_rotation'],dtype=np.float64)
    if vertices.ndim!=2 or vertices.shape[1]!=3 or not np.isfinite(vertices).all() or R.shape!=(3,3):raise ValueError('invalid evaluator-only truth')
    dots=np.abs(R.T@axis);matching=int(np.argmax(dots[:2]))
    def degrees(dot):return math.degrees(math.acos(float(np.clip(dot,-1,1))))
    projected=vertices@axis
    return dict(projected_true_extent_m=float(projected.max()-projected.min()),
        original_axis_norm=norm,inferred_depth_axis_world=axis.tolist(),
        true_local_extent_m={str(i):float(np.ptp(vertices@R[:,i])) for i in range(3)},
        error_to_true_depth_axis_deg=degrees(dots[1]),best_horizontal_true_axis=matching,
        best_horizontal_axis_error_deg=degrees(dots[matching]),swapped_width_depth_axis=bool(matching==0),
        elevation_from_horizontal_deg=math.degrees(math.asin(float(np.clip(abs(axis[2]),0,1)))),
        axis_matching_used_to_reassign_target=False)

def equal_object_summary(rows):
    """Each object contributes its equally weighted conditions once."""
    if not rows:raise ValueError('no evaluation rows')
    by={}
    for row in rows:by.setdefault(row['object_id'],[]).append(row)
    def avg(values):return float(sum(values)/len(values))
    score_keys=('crps_m','median_absolute_error_m','mean_absolute_error_m')
    objects={oid:{key:avg([r['metrics'][key] for r in rs]) for key in score_keys} for oid,rs in by.items()}
    result={key:avg([o[key] for o in objects.values()]) for key in score_keys}
    result.update(objects=len(by),cases=len(rows),per_object=objects)
    result['intervals']={key:dict(covered_cases=sum(int(r['metrics']['intervals'][key]['covered']) for r in rows),case_denominator=len(rows),
        equal_object_coverage_fraction=avg([avg([float(r['metrics']['intervals'][key]['covered']) for r in rs]) for rs in by.values()]),
        equal_object_mean_width_m=avg([avg([r['metrics']['intervals'][key]['width_m'] for r in rs]) for rs in by.values()])) for key in ('0.50','0.80','0.95')}
    result['outside_support_cases']=sum(int(r['metrics']['truth_outside_particle_support']) for r in rows)
    return result
