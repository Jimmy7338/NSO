"""Evaluator-only asset-balanced geometry metrics, never planner input.

All mission instances and spatial attribution windows are fixed before runs.
False surfaces inside a window count against that instance; surfaces outside
windows remain accountable in the separately reported global metrics.
"""
import numpy as np
import open3d as o3d
from env.virtual3d import camera_pose
from utils.reconstruction_metrics import surface_samples, ray_scene, ReconstructionEvaluator


def observable_reference(world, points, truth, stride=4):
    visible = np.zeros(len(points), bool); c = world.config
    fx, fy = world.intrinsic[0,0], world.intrinsic[1,1]
    for r, col in np.argwhere(world.reachable):
        if r % stride or col % stride: continue
        for heading in range(4):
            pose = camera_pose((r,col), heading, c, world.shape[0])
            delta = points-pose[:3,3]; local = delta@pose[:3,:3]; z = local[:,2]
            u = local[:,0]/np.maximum(z,.001)*fx+world.intrinsic[0,2]
            v = local[:,1]/np.maximum(z,.001)*fy+world.intrinsic[1,2]
            possible = (~visible)&(z>.15)&(z<=c.max_depth_m)&(u>=-.5)&(u<c.width_px-.5)&(v>=-.5)&(v<c.height_px-.5)
            ids = np.flatnonzero(possible)
            if not len(ids): continue
            rays = np.column_stack([np.tile(pose[:3,3],(len(ids),1)), delta[ids]]).astype(np.float32)
            hits = truth.cast_rays(o3d.core.Tensor(rays),nthreads=1)['t_hit'].numpy()
            visible[ids[np.abs(hits-1.)<1e-4]] = True
    return visible


class FacilityEvaluatorV18:
    def __init__(self, world, count_per_asset=3000, seed=2026):
        if count_per_asset < 1: raise ValueError('positive reference count required')
        self.global_evaluator = ReconstructionEvaluator(world, count=12000, seed=seed)
        self.truth = self.global_evaluator.truth; self.instances = []
        for item in world.objects:
            mesh = world.instance_mesh(item['id'])
            points, _ = surface_samples(mesh, count_per_asset, seed+1009*item['id'])
            visible = observable_reference(world, points, self.truth)
            if not visible.any(): raise ValueError('mission instance has no observable reference surface')
            self.instances.append(dict(id=item['id'], bounds=np.asarray(item['evaluation_bounds']),
                truth=ray_scene(mesh), reference=points[visible].astype(np.float32),
                sampled=len(points), observable_samples=int(visible.sum()),
                observable_area_m2=float(mesh.get_surface_area())*float(visible.mean())))

    def evaluate(self, mesh, coverage, *, returned=False, collisions=0, thresholds=(.02,.05,.10)):
        if not np.isfinite(coverage) or not 0 <= coverage <= 1: raise ValueError('coverage outside [0,1]')
        predicted, _ = surface_samples(mesh, 24000, 812)
        reconstruction = ray_scene(mesh) if len(mesh.triangles) else None
        rows=[]
        for item in self.instances:
            mask = ((predicted>=item['bounds'][0])&(predicted<=item['bounds'][1])).all(axis=1)
            selected = predicted[mask].astype(np.float32)
            accuracy = item['truth'].compute_distance(o3d.core.Tensor(selected),nthreads=1).numpy() if len(selected) else np.empty(0)
            # A missing attributed prediction cannot receive partial credit
            # solely from a nearby floor/wall in the global reconstruction.
            recall_distance = reconstruction.compute_distance(o3d.core.Tensor(item['reference']),nthreads=1).numpy() if len(selected) and reconstruction is not None else np.full(len(item['reference']),np.inf)
            row=dict(id=item['id'], predicted_samples=len(selected),reference_samples=item['observable_samples'],
                observable_area_m2=item['observable_area_m2'],
                surface_error_mean_m=float(accuracy.mean()) if len(accuracy) else None,
                surface_error_p95_m=float(np.percentile(accuracy,95)) if len(accuracy) else None)
            for threshold in thresholds:
                p=float(np.mean(accuracy<=threshold)) if len(accuracy) else 0.
                r=float(np.mean(recall_distance<=threshold));f=2*p*r/(p+r) if p+r else 0.
                row[f'{round(threshold*100):02d}cm']=dict(precision=p,recall=r,f1=f)
            rows.append(row)
        result=dict(instances=rows,mission_asset_count=len(rows),coverage_2d=coverage,
            returned=bool(returned),collisions=int(collisions),eligible=coverage>=.8 and returned and collisions==0,
            reference_source='fixed reachable lattice every4cells, four actual yaw headings; truth-only evaluation',
            attribution='fixed disjoint instance windows; no predicted labels; global metrics retain outside-window errors')
        for threshold in thresholds:
            tag=f'{round(threshold*100):02d}cm';values=[r[tag]['f1'] for r in rows]
            mean=float(np.mean(values));completion=float(np.mean([r[tag]['precision']>=.95 and r[tag]['recall']>=.8 for r in rows]))
            result[tag]=dict(asset_macro_f1=mean,joint_asset=coverage*mean,completion_fraction=completion,
                worst_quartile_mean_f1=float(np.mean(sorted(values)[:max(1,int(np.ceil(len(values)/4)))])))
        result['global_legacy']=self.global_evaluator.evaluate(mesh,coverage,thresholds=thresholds)
        return result
