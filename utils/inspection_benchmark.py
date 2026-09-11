"""Offline causal inspection bench, separate from the frozen exploration world.

Target geometry / evaluation masks are experimental fixtures, never planner input.
Each branch starts from the identical prefix; endpoint captures isolate sensing
effects and are NOT navigation episodes. The angular core below is the old
quality formula with its sensor-map visibility gate, extended to continuous yaw.
"""
from dataclasses import replace
import numpy as np
import open3d as o3d
from scipy.ndimage import label
from env.grid_exploration import GridConfig
from nso.quality_coverage3d import QualityCoveragePolicy
from utils.grid_geometry import inflated_obstacles
from utils.rgbd_contract import RGBDFrame
from utils.reconstruction_metrics import surface_samples, ray_scene


def look_at_xy(origin, target):
    forward = np.asarray(target, float) - origin
    forward[2] = 0
    forward /= np.linalg.norm(forward)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([np.cross(forward, [0., 0., 1.]), [0., 0., -1.], forward])
    pose[:3, 3] = origin
    return pose


class InspectionWorld:
    def __init__(self, config, seed, geometry):
        if geometry not in ('box', 'shelf'):
            raise ValueError('unsupported geometry')
        self.config, self.seed, self.geometry = config, seed, geometry
        self.shape = (30, 40)
        self.center = np.array([4., 3., config.camera_height_m])
        self.mesh = o3d.geometry.TriangleMesh()
        self.target = o3d.geometry.TriangleMesh()
        self.triangle_classes = []
        self.occupancy = np.zeros(self.shape, bool)
        rng = np.random.default_rng(seed)
        sx, sy, height = rng.uniform(.8, 1.2), rng.uniform(.5, .8), rng.uniform(1.1, 1.5)
        x, y = 4-sx/2, 3-sy/2
        self.roi = np.array([[x-.12, y-.12, .12], [x+sx+.12, y+sy+.12, height+.12]])
        def box(a, b, z, w, d, h, category):
            mesh = o3d.geometry.TriangleMesh.create_box(w, d, h).translate((a, b, z))
            # Materials do not depend on the label. No photometric metric is used.
            mesh.paint_uniform_color([.6, .6, .6])
            self.mesh += mesh
            self.triangle_classes.extend([category]*len(mesh.triangles))
            if category > 1:
                self.target += mesh
            if z < 1.2 and z+h > .15:
                r0, r1 = int(np.floor((6-b-d)/.2)), int(np.ceil((6-b)/.2))
                c0, c1 = int(np.floor(a/.2)), int(np.ceil((a+w)/.2))
                self.occupancy[max(0,r0):min(30,r1), max(0,c0):min(40,c1)] = True
        box(0, 0, -.1, 8, 6, .1, 1)
        for args in [(0,0,0,8,.2,2.4), (0,5.8,0,8,.2,2.4), (0,0,0,.2,6,2.4), (7.8,0,0,.2,6,2.4)]:
            box(*args, 1)
        if geometry == 'box':
            box(x, y, 0, sx, sy, height, 2)
        else:
            for z in (.15*height, .5*height, .85*height):
                box(x, y, z, sx, sy, .08, 3)
            for dx in (0, sx-.08):
                for dy in (0, sy-.08):
                    box(x+dx, y+dy, 0, .08, .08, height, 3)
            box(x+.3*sx, y+.3*sy, .15*height+.08, .3*sx, .4*sy, .23*height, 3)
        self.triangle_classes = np.asarray(self.triangle_classes)
        blocked = inflated_obstacles(self.occupancy, config.robot_radius_m/config.resolution_m)
        components, _ = label(~blocked)
        self.reachable = components == components[15, 5]
        self.scene = ray_scene(self.mesh)
        focal = config.width_px/(2*np.tan(np.deg2rad(config.fov_deg/2)))
        self.intrinsic = np.array([[focal,0,(config.width_px-1)/2], [0,focal,(config.height_px-1)/2], [0,0,1.]])

    def pose(self, radius, degrees):
        angle = np.deg2rad(degrees)
        ideal = self.center + [radius*np.cos(angle), radius*np.sin(angle), 0]
        cells = np.argwhere(self.reachable)
        xy = np.column_stack([(cells[:,1]+.5)*.2, (30-cells[:,0]-.5)*.2])
        index = np.argmin(np.linalg.norm(xy-ideal[:2], axis=1))
        if np.linalg.norm(xy[index]-ideal[:2]) > .2:
            raise ValueError('fixture viewpoint is not reachable')
        origin = np.r_[xy[index], self.config.camera_height_m]
        return look_at_xy(origin, self.center), tuple(map(int, cells[index]))

    def capture(self, pose, index, noise):
        c = self.config
        v, u = np.mgrid[:c.height_px, :c.width_px]
        local = np.stack([(u-self.intrinsic[0,2])/self.intrinsic[0,0], (v-self.intrinsic[1,2])/self.intrinsic[1,1], np.ones_like(u)], axis=-1)
        rays = np.empty((*u.shape,6), np.float32)
        rays[...,:3] = pose[:3,3]
        rays[...,3:] = local@pose[:3,:3].T
        hits = self.scene.cast_rays(o3d.core.Tensor(rays), nthreads=1)
        depth = hits['t_hit'].numpy()
        valid = np.isfinite(depth)&(depth>.15)&(depth<=c.max_depth_m)
        ids = hits['primitive_ids'].numpy()
        labels = np.zeros(depth.shape, np.uint8)
        labels[valid] = self.triangle_classes[ids[valid]]
        sigma = {'none':0., 'iid_01':.01, 'iid_03':.03, 'fixed_01':.01}[noise]
        # Common random numbers across branches. Fixed bias persists in sensor
        # pixels; fresh IID draws are independent across capture indices.
        noise_index = 0 if noise == 'fixed_01' else index
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, noise_index, 9431]))
        depth = depth + rng.normal(0, sigma, depth.shape)
        valid &= (depth>.15)&(depth<=c.max_depth_m)
        depth = np.where(valid, depth, 0).astype(np.float32)
        labels[~valid] = 0
        rgb = np.repeat(np.where(valid,153,0).astype(np.uint8)[...,None],3,axis=2)
        return RGBDFrame(float(index), depth, rgb, self.intrinsic.copy(), pose.copy(), labels).validate()


def angular_proxy(mapper, pose, cell):
    """Old formula, sensor-only; continuous yaw is a bench extension.

    Return gated and ungated target proxies plus the global gated proxy.
    Target isolation uses the existing sensor object labels (>1), equally for
    both scores, to match the target-only outcome. It uses no truth ROI or
    future depth. This is a regional formula diagnosis, not a geometry-only
    object-discovery baseline or a test of semantic segmentation efficacy.
    """
    c = mapper.config
    grid = GridConfig(resolution_m=c.resolution_m, robot_radius_m=c.robot_radius_m,
                      sensor_range_m=c.max_depth_m, sensor_fov_deg=360, max_steps=120)
    policy = QualityCoveragePolicy(grid, c, 'geometric_quality')
    policy.set_surface_evidence(mapper.evidence())
    points, bits, labels = policy.surface
    if not len(points):
        return dict(angular=0., semantic=0., ungated=0., scene_angular=0.)
    local = (points-pose[:3,3])@pose[:3,:3]
    z = local[:,2]
    tan = np.tan(np.deg2rad(c.fov_deg/2))
    visible = (z>.15)&(z<c.max_depth_m)&(np.abs(local[:,0])<z*tan)&(np.abs(local[:,1])<z*tan*c.height_px/c.width_px)
    az = np.arctan2(pose[1,3]-points[:,1], pose[0,3]-points[:,0])
    viewbin = np.floor((az+np.pi)/(2*np.pi)*8).astype(int)%8
    novel = (bits&(1<<viewbin)) == 0
    counts = np.asarray([int(b).bit_count() for b in bits])
    value = visible*novel*np.maximum(0,3-counts)/3*.15**2
    ungated = float(value[labels>1].sum())
    pr = mapper.shape[0]-1-np.floor(points[:,1]/c.resolution_m).astype(int)
    pc = np.floor(points[:,0]/c.resolution_m).astype(int)
    inside = (pr>=0)&(pr<mapper.shape[0])&(pc>=0)&(pc<mapper.shape[1])
    view = policy._visible(mapper.observation(cell,0,mapper.frames),cell,0)
    value *= inside & view[np.clip(pr,0,mapper.shape[0]-1),np.clip(pc,0,mapper.shape[1]-1)]
    scene_angular = float(value.sum())
    value *= labels>1
    return dict(angular=float(value.sum()), semantic=float((value*(1+.75*(labels==3))).sum()), ungated=ungated, scene_angular=scene_angular)


class TargetEvaluator:
    def __init__(self, world, reference_samples, prediction_samples):
        self.roi = world.roi
        self.truth = ray_scene(world.target)
        self.prediction_samples = prediction_samples
        points, _ = surface_samples(world.target, reference_samples, 513)
        eligible = (points[:,2]>.12)
        visible = np.zeros(len(points),bool)
        tan = world.config.height_px/world.config.width_px*np.tan(np.deg2rad(world.config.fov_deg/2))
        # Fixed reference, identical for every intervention / noise condition.
        for radius in (.9,1.4,2.2):
            for degrees in np.arange(0,360,15):
                pose, _ = world.pose(radius,degrees)
                delta = points-pose[:3,3]
                horizontal = np.linalg.norm(delta[:,:2],axis=1)
                ids = np.flatnonzero(eligible&~visible&(np.abs(delta[:,2])<=horizontal*tan)&(horizontal<world.config.max_depth_m))
                if not len(ids):
                    continue
                rays = np.column_stack([np.tile(pose[:3,3],(len(ids),1)),delta[ids]]).astype(np.float32)
                hit = world.scene.cast_rays(o3d.core.Tensor(rays),nthreads=1)['t_hit'].numpy()
                visible[ids[np.abs(hit-1)<1e-4]] = True
        self.reference = points[visible].astype(np.float32)

    def distances(self, mesh):
        if not len(mesh.triangles):
            return np.empty(0), np.full(len(self.reference),np.inf)
        samples, _ = surface_samples(mesh,self.prediction_samples,812)
        inside = ((samples>self.roi[0])&(samples<self.roi[1])).all(axis=1)
        accuracy = self.truth.compute_distance(o3d.core.Tensor(samples[inside].astype(np.float32)),nthreads=1).numpy() if inside.any() else np.empty(0)
        completeness = ray_scene(mesh).compute_distance(o3d.core.Tensor(self.reference),nthreads=1).numpy()
        return accuracy, completeness

    def evaluate(self, mesh, support=None):
        accuracy, completeness = self.distances(mesh)
        result = dict(predicted_roi_samples=len(accuracy), reference_samples=len(self.reference),
                      surface_error_m=float(accuracy.mean()) if len(accuracy) else None,
                      surface_p95_m=float(np.percentile(accuracy,95)) if len(accuracy) else None)
        for threshold in (.01,.02,.05,.10):
            p = float(np.mean(accuracy<=threshold)) if len(accuracy) else 0.
            r = float(np.mean(completeness<=threshold))
            tag = f'{round(threshold*100):02d}cm'
            result.update({f'p_{tag}':p, f'r_{tag}':r, f'f1_{tag}':2*p*r/(p+r) if p+r else 0.})
        if support is not None:
            result['fixed_support_distance_m'] = float(np.minimum(completeness[support],.1).mean()) if support.any() else None
            result['fixed_support_retention_05cm'] = float(np.mean(completeness[support]<=.05)) if support.any() else None
        return result
