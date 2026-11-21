"""B10 declared acquisition-reward correction using the acquired native TSDF.

Prediction-only: no source reference, future observation or category is read.
A template point already supported by the current measured mesh cannot earn
new-area reward. This is a decision-affecting reward revision, not a policy-
neutral engineering change. Its pointwise proxy does not guarantee that the
source-reference reconstruction metric will improve.
"""
from __future__ import annotations
import numpy as np

TOLERANCE_M=0.05

def unmeasured_mask(points,measured_scene,tolerance_m=TOLERANCE_M):
    """Keep points farther than tolerance from current native measured mesh.

    Existing history/free-space masks remain in the caller's weights. An
    empty mesh consumes nothing. Invalid coordinates or distances fail loudly
    rather than silently suppressing or inventing reward.
    """
    xyz=np.asarray(points,dtype=float)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or not np.isfinite(xyz).all():
        raise ValueError('finite N by3 template points required')
    if not np.isfinite(tolerance_m) or tolerance_m<0:
        raise ValueError('finite nonnegative tolerance required')
    if measured_scene is None or len(xyz)==0:
        return np.ones(len(xyz),dtype=bool),np.full(len(xyz),np.inf)
    import open3d as o3d
    distances=np.asarray(measured_scene.compute_distance(
        o3d.core.Tensor(np.asarray(xyz,dtype=np.float32)),nthreads=1).numpy(),float)
    if distances.shape!=(len(xyz),) or not np.isfinite(distances).all() or np.any(distances<0):
        raise ValueError('finite nonnegative measured-mesh distances required')
    # Distances are computed in float32. Compare to the same represented
    # tolerance, closing the boundary without adding a physical tolerance.
    represented=float(np.float32(tolerance_m))
    return distances>represented,distances

def consume(values,keep):
    """Apply one shared geometry mask; preserve exact weighted basis and zeros."""
    a=np.asarray(values,float);mask=np.asarray(keep)
    if a.ndim not in (1,2) or mask.dtype!=bool or mask.shape!=(a.shape[-1],):
        raise ValueError('gain vector/matrix and aligned boolean mask required')
    if not np.isfinite(a).all() or np.any(a<0):
        raise ValueError('finite nonnegative predicted gain required')
    return a*mask

def apply(model,measured_scene,tolerance_m=TOLERANCE_M):
    """Consume measured points once, after posterior reweighting and history.

    Call before any per-pose gain cache is populated. All external particle
    coordinates and base probabilities stay unchanged. A second identical
    call is idempotent; a different mesh requires a newly built model.
    """
    keep,distances=unmeasured_mask(model.points,measured_scene,tolerance_m)
    previous=np.asarray(model.weights,float)
    current=consume(previous,keep)
    model.weights=current
    receipt=dict(schema='B10_native_mesh_novelty_v1',tolerance_m=float(tolerance_m),
      decision_affecting_reward_revision=True,source_geometry_consumed=False,
      probability_geometry_history_unchanged=True,template_points=len(keep),
      positive_weight_points_consumed=int(np.count_nonzero((previous>0)&~keep)),
      expected_area_before_m2=float(previous.sum()),expected_area_after_m2=float(current.sum()),
      consumed_expected_area_m2=float((previous-current).sum()),
      measured_mesh_empty=measured_scene is None,
      pointwise_proxy_not_reference_area_guarantee=True)
    model.audit['native_mesh_novelty']=receipt
    return keep,distances,receipt
