"""Common observed-mesh occlusion correction for hypothetical asset utility.

The existing coverage and observed-quality terms remain intact. A no-hit ray
is unknown, not certified visibility. Known self/other occlusion blocks the
hypothetical far-plane term; actual surface quality has its own common term.
"""
from copy import deepcopy
import numpy as np
import open3d as o3d
from env.virtual3d import camera_pose
from nso.competition_candidates_v8_1 import aperture_support


def plane_points(asset):
    horizontal,vertical=np.meshgrid(np.linspace(-.5,.5,5)*asset['measured_width_m'],
        np.linspace(asset['observed_low'][2],asset['observed_high'][2],5))
    return np.column_stack([asset['rear_boundary_xy'][0]+horizontal.ravel()*asset['side_axis'][0],
        asset['rear_boundary_xy'][1]+horizontal.ravel()*asset['side_axis'][1],vertical.ravel()])


class ObservedMeshAperture:
    def __init__(self,mesh,config):
        self.config=config;self.ray=None
        if len(mesh.triangles):
            self.ray=o3d.t.geometry.RaycastingScene(nthreads=1)
            self.ray.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    def support(self,asset,pose):
        original=aperture_support(asset,pose,self.config);masked=original.copy()
        ids=np.flatnonzero(original>0)
        if not len(ids) or self.ray is None:return masked
        delta=plane_points(asset)[ids]-pose[:3,3]
        z=(delta@pose[:3,:3])[:,2]
        rays=np.column_stack([np.tile(pose[:3,3],(len(ids),1)),delta]).astype(np.float32)
        t=self.ray.cast_rays(o3d.core.Tensor(rays),nthreads=1)['t_hit'].numpy()
        blocked=np.isfinite(t)&((1.-t)*z>self.config.truncation_m)
        masked[ids[blocked]]=0.
        return masked


def correct_scores(mapper,assets,direction_support,routes,base_scores,base_rows,*,novelty_floor=.25):
    """V10.3.1 measured-direction score contract, with common occlusion masks."""
    if not 0<=novelty_floor<=1:raise ValueError('invalid measured novelty floor')
    if len(routes)!=len(base_rows):raise ValueError('route/score rows differ')
    visibility=ObservedMeshAperture(mapper.mesh(),mapper.config);cache={}
    generic=np.array([a['measured_width_m']*a['measured_height_m']+3*a['measured_width_m']*a['measured_depth_m'] for a in assets])
    semantic=np.array([a['measured_width_m']*a['measured_height_m']+3*(1-a['class_vote'])*a['measured_width_m']*a['measured_depth_m'] for a in assets])
    swapped=np.array([a['measured_width_m']*a['measured_height_m']+3*(1+a['class_vote'])*a['measured_width_m']*a['measured_depth_m'] for a in assets])
    marked=np.array([float(a['marked_points']>0) for a in assets])
    scores={m:[] for m in ('N','G','O','S','X','M')};rows=[]
    for route,old in zip(routes,base_rows):
        factors=np.zeros(len(assets));unmasked=np.zeros(len(assets))
        for state in route['outbound_states'][1:]:
            key=tuple(state)
            if key not in cache:
                pose=camera_pose(state[:2],state[2],mapper.config,mapper.shape[0]);raw=[];corrected=[]
                for ai,asset in enumerate(assets):
                    delta=pose[:2,3]-asset['aabb_center'][:2]
                    sector=int(np.floor((np.arctan2(delta[1],delta[0])+np.pi)/(2*np.pi)*8))%8
                    observed=direction_support[ai][sector] if ai<len(direction_support) else 0.
                    scale=novelty_floor+(1-novelty_floor)*(1.-observed)
                    raw.append(float(aperture_support(asset,pose,mapper.config).mean())*scale)
                    corrected.append(float(visibility.support(asset,pose).mean())*scale)
                cache[key]=(np.asarray(raw),np.asarray(corrected))
            raw,corrected=cache[key];unmasked=np.maximum(unmasked,raw);factors=np.maximum(factors,corrected)
        if not np.allclose(unmasked,old['incremental_aperture'],atol=1e-12,rtol=0):
            raise ValueError('base measured-direction score contract differs')
        if np.any(factors>unmasked+1e-12):raise ValueError('occlusion increased hypothetical support')
        potentials=dict(N=0.,G=float(factors@generic),O=float(factors@(generic*marked)),
            S=float(factors@(semantic*marked+generic*(1-marked))),
            X=float(factors@(swapped*marked+generic*(1-marked))),M=float(factors@generic))
        for mode,value in potentials.items():scores[mode].append(float(old['common_total_proxy']+value))
        rows.append(dict(candidate_id=route['candidate_id'],common_total_proxy=old['common_total_proxy'],
            original_aperture=unmasked.tolist(),corrected_aperture=factors.tolist(),prior_potentials=potentials,
            known_occlusion_only=True,unknown_rays_are_visibility_certificates=False,
            observed_quality_common_term_unchanged=True,truth_used=False,calibrated=False))
    if scores['N']!=base_scores['N']:raise ValueError('common coverage/quality score changed')
    return scores,rows
