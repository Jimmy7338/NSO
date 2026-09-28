"""Evaluator-only, area-sampled precision/recall; no policy-selected GT subset."""
import numpy as np
import open3d as o3d


def surface_samples(mesh,count,seed):
    vertices=np.asarray(mesh.vertices);triangles=np.asarray(mesh.triangles)
    if not len(triangles):return np.empty((0,3)),np.empty(0,int)
    a,b,c=vertices[triangles[:,0]],vertices[triangles[:,1]],vertices[triangles[:,2]]
    area=np.linalg.norm(np.cross(b-a,c-a),axis=1)/2
    if area.sum()<=0:return np.empty((0,3)),np.empty(0,int)
    rng=np.random.default_rng(seed);ids=rng.choice(len(area),count,p=area/area.sum())
    u=np.sqrt(rng.random(count));v=rng.random(count)
    return ((1-u)[:,None]*a[ids]+(u*(1-v))[:,None]*b[ids]+(u*v)[:,None]*c[ids]),ids


def ray_scene(mesh):
    scene=o3d.t.geometry.RaycastingScene(nthreads=1)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


class ReconstructionEvaluator:
    def __init__(self,world,count=16000,seed=2026):
        self.truth=ray_scene(world.mesh);self.config=world.config
        points,triangles=surface_samples(world.mesh,count,seed)
        visible=np.zeros(len(points),bool)
        c=world.config
        # Fixed evaluation reference: observable surfaces from a deterministic
        # lattice of all reachable positions, with arbitrary planar yaw.
        vertical_tan=c.height_px/c.width_px*np.tan(np.deg2rad(c.fov_deg/2))
        cells=np.argwhere(world.reachable)
        for r,col in cells:
            if r%4 or col%4:continue
            origin=np.array([(col+.5)*c.resolution_m,(world.shape[0]-r-.5)*c.resolution_m,c.camera_height_m])
            delta=points-origin;horizontal=np.linalg.norm(delta[:,:2],axis=1)
            possible=(~visible)&(horizontal>.15)&(horizontal<c.max_depth_m)&(np.abs(delta[:,2])<=horizontal*vertical_tan)
            ids=np.flatnonzero(possible)
            if not len(ids):continue
            rays=np.column_stack([np.tile(origin,(len(ids),1)),delta[ids]]).astype(np.float32)
            hit=self.truth.cast_rays(o3d.core.Tensor(rays),nthreads=1)['t_hit'].numpy()
            visible[ids[np.abs(hit-1.)<1e-4]]=True
        self.reference=points[visible].astype(np.float32)
        self.classes=world.triangle_classes[triangles[visible]]
        if not len(self.reference):raise ValueError('empty observable surface reference')

    def evaluate(self,mesh,coverage,thresholds=(.05,.10)):
        predicted,_=surface_samples(mesh,12000,812)
        result=dict(surface_samples=len(predicted),reference_samples=len(self.reference))
        if len(predicted):
            accuracy=self.truth.compute_distance(o3d.core.Tensor(predicted.astype(np.float32)),nthreads=1).numpy()
            reconstruction=ray_scene(mesh)
            completeness=reconstruction.compute_distance(o3d.core.Tensor(self.reference),nthreads=1).numpy()
            result.update(surface_error_mean_m=float(accuracy.mean()),surface_error_p95_m=float(np.percentile(accuracy,95)))
        else:
            accuracy=np.empty(0);completeness=np.full(len(self.reference),np.inf)
            result.update(surface_error_mean_m=None,surface_error_p95_m=None)
        for threshold in thresholds:
            tag=f'{round(threshold*100):02d}cm'
            precision=float(np.mean(accuracy<=threshold)) if len(accuracy) else 0.
            recall=float(np.mean(completeness<=threshold))
            f1=2*precision*recall/(precision+recall) if precision+recall else 0.
            result.update({f'precision_{tag}':precision,f'recall_{tag}':recall,
                           f'f1_{tag}':f1,f'joint_{tag}':float(coverage*f1)})
            for label_id,label_name in ((2,'simple'),(3,'complex')):
                subset=self.classes==label_id
                result[f'{label_name}_recall_{tag}']=float(np.mean(completeness[subset]<=threshold)) if subset.any() else None
        return result
