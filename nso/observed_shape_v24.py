"""Class-blind, measured-depth instance support and explicit enclosure hypotheses.

No simulator, semantic field, reference model or evaluation window is accepted.
Completion is a rectilinear enclosure hypothesis, never a claim that unseen
surfaces were measured. Known free rays can reject it at every snapshot.
"""
import hashlib
import itertools

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


VERSION = 'observed-shape-v24-1'


def empty_mesh():
    return o3d.geometry.TriangleMesh()


def mesh_from_arrays(vertices, triangles):
    mesh = empty_mesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices, float).reshape(-1, 3))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(triangles, np.int32).reshape(-1, 3))
    mesh.remove_unreferenced_vertices()
    return mesh


class ObservedShapeBackendV24:
    """One observed instance, with optional raw TSDF supplied at snapshot time.

    A seed is an observed surface point (e.g. a class-stripped visible marker),
    not the centre or bounds of a hidden object. All methods must get the same
    seed/instance association. No seed means ground evidence only.
    """
    def __init__(self, up=(0., 0., 1.)):
        up = np.asarray(up, float)
        if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) < 1e-8:
            raise ValueError('finite nonzero gravity up required')
        self.up = up / np.linalg.norm(up)
        self.frames = []
        self.identities = {}
        self.seeds = []
        self.ground_band_m = .022
        self.component_voxel_m = .08
        self.support_radius_m = .10
        self.edge_tolerance_m = .08
        self.neighbour_limit_m = .25

    def observe(self, depth_m, intrinsic, world_from_camera, *, observation_id,
                observed_seed_xyz=None):
        depth = np.asarray(depth_m, float)
        k = np.asarray(intrinsic, float)
        t = np.asarray(world_from_camera, float)
        if (depth.ndim != 2 or min(depth.shape) < 3 or not np.isfinite(depth).all()
                or (depth < 0).any() or k.shape != (3, 3) or t.shape != (4, 4)
                or not np.isfinite(k).all() or not np.isfinite(t).all()
                or k[0, 0] <= 0 or k[1, 1] <= 0
                or not np.allclose(k[2], [0, 0, 1])
                or not np.allclose(t[3], [0, 0, 0, 1])
                or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-6)
                or np.linalg.det(t[:3, :3]) < .999999):
            raise ValueError('valid finite metric depth, pinhole calibration and rigid pose required')
        if not isinstance(observation_id, (str, int)) or isinstance(observation_id, bool):
            raise ValueError('stable string or integer observation id required')
        identity = hashlib.sha256()
        for a in (depth, k, t):
            identity.update(str(a.shape).encode()); identity.update(a.tobytes())
        seed = None if observed_seed_xyz is None else np.asarray(observed_seed_xyz, float)
        if seed is not None:
            if seed.shape != (3,) or not np.isfinite(seed).all():
                raise ValueError('finite observed seed point required')
            identity.update(seed.tobytes())
        identity = identity.hexdigest()
        if observation_id in self.identities:
            if self.identities[observation_id] != identity:
                raise ValueError('observation id reused with different data')
            return False
        v, u = np.indices(depth.shape)
        rays = np.stack([u, v, np.ones_like(u)], axis=-1) @ np.linalg.inv(k).T
        local = rays * depth[..., None]
        xyz = local @ t[:3, :3].T + t[:3, 3]
        valid = depth > 0
        if seed is not None:
            if not valid.any() or cKDTree(xyz[valid]).query(seed)[0] > .06:
                raise ValueError('seed must be a current measured depth hit, not a hidden centre')
        dx = np.roll(xyz, -1, 1) - np.roll(xyz, 1, 1)
        dy = np.roll(xyz, -1, 0) - np.roll(xyz, 1, 0)
        normals = np.cross(dx, dy)
        length = np.linalg.norm(normals, axis=-1)
        normal_valid = valid & (length > 1e-10)
        for axis in (0, 1):
            for direction in (-1, 1):
                normal_valid &= np.roll(valid, direction, axis)
                normal_valid &= np.linalg.norm(np.roll(xyz, direction, axis)-xyz, axis=-1) <= self.neighbour_limit_m
        normal_valid[[0, -1], :] = False; normal_valid[:, [0, -1]] = False
        normals /= np.maximum(length[..., None], 1e-10)
        normals[~normal_valid] = 0.
        self.frames.append(dict(depth=depth.copy(), k=k.copy(), t=t.copy(),
            rays=rays @ t[:3, :3].T, xyz=xyz, valid=valid, normals=normals,
            observation_id=observation_id))
        self.identities[observation_id] = identity
        if seed is not None:
            self.seeds.append(seed.copy())
        return True

    def _ground(self, points, normals):
        if not len(points):
            return None
        candidate = np.abs(normals @ self.up) >= np.cos(np.deg2rad(25))
        # Ground must lie below at least one actual camera, not at a fixed z.
        highest_camera = max(f['t'][:3, 3] @ self.up for f in self.frames)
        candidate &= points @ self.up < highest_camera - .12
        ids = np.flatnonzero(candidate)
        if len(ids) < 40:
            return None
        # Deterministic normal-guided plane hypotheses; no hidden floor height.
        pool = ids[np.linspace(0, len(ids)-1, min(len(ids), 18000), dtype=int)]
        p, n = points[pool], normals[pool]
        best = None
        for index in np.linspace(0, len(pool)-1, min(128, len(pool)), dtype=int):
            normal = n[index] * (1 if n[index] @ self.up >= 0 else -1)
            offset = -float(p[index] @ normal)
            inliers = np.abs(p @ normal + offset) <= self.ground_band_m
            count = int(inliers.sum())
            if count < max(40, .15*len(pool)):
                continue
            # A large table/platform is not the ground when a sufficiently
            # supported lower plane is visible. Rank measured inlier height,
            # not the extrapolated intercept or the largest point population.
            score = (-float(np.median(p[inliers] @ self.up)), count)
            if best is None or score > best[0]:
                best = (score, inliers)
        if best is None:
            return None
        fit = p[best[1]]
        if len(fit) < max(40, .15*len(pool)):
            return None
        for _ in range(2):
            centre = fit.mean(axis=0)
            _, _, vh = np.linalg.svd(fit-centre, full_matrices=False)
            normal = vh[-1]
            if normal @ self.up < 0:
                normal = -normal
            offset = -float(centre @ normal)
            fit = p[np.abs(p @ normal+offset) <= self.ground_band_m]
        if normal @ self.up < np.cos(np.deg2rad(25)):
            return None
        distances = points @ normal + offset
        support = candidate & (np.abs(distances) <= self.ground_band_m)
        return dict(normal=normal, offset=offset, support_count=int(support.sum()),
                    median_residual_m=float(np.median(np.abs(distances[support]))),
                    observation_count=len(self.frames), source='measured depth local normals and robust plane fit')

    def _component(self, points, normals, ground):
        if not self.seeds or ground is None or not len(points):
            return np.zeros(len(points), bool)
        signed = points @ ground['normal'] + ground['offset']
        # The uncertainty band is excluded from graph connectivity, not
        # declared nonexistent. It remains in raw data / unassigned evidence.
        candidates = np.flatnonzero(signed > self.ground_band_m)
        if not len(candidates):
            return np.zeros(len(points), bool)
        keys = np.floor(points[candidates]/self.component_voxel_m).astype(np.int64)
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        lookup = {tuple(key): i for i, key in enumerate(unique)}
        tree = cKDTree(points[candidates])
        dist, idx = tree.query(np.asarray(self.seeds))
        starts = {int(inverse[j]) for d, j in zip(dist, idx) if d <= self.support_radius_m}
        seen, queue = set(starts), list(starts)
        offsets = list(itertools.product((-1, 0, 1), repeat=3))
        while queue:
            index = queue.pop()
            key = unique[index]
            for shift in offsets:
                neighbour = lookup.get(tuple(key+shift))
                if neighbour is not None and neighbour not in seen:
                    seen.add(neighbour); queue.append(neighbour)
        mask = np.zeros(len(points), bool)
        mask[candidates] = np.isin(inverse, list(seen))
        return mask

    def _observed_mesh(self):
        vertices, triangles, offset = [], [], 0
        for f in self.frames:
            h, w = f['depth'].shape
            index = np.arange(h*w).reshape(h, w)
            a, b, c, d = index[:-1, :-1], index[:-1, 1:], index[1:, :-1], index[1:, 1:]
            tri = np.concatenate([np.stack([a,b,c], -1).reshape(-1,3),
                                  np.stack([b,d,c], -1).reshape(-1,3)])
            xyz = f['xyz'].reshape(-1, 3)
            good = f['valid'].ravel()[tri].all(axis=1)
            for i,j in ((0,1),(1,2),(2,0)):
                good &= np.linalg.norm(xyz[tri[:,i]]-xyz[tri[:,j]], axis=1) <= self.neighbour_limit_m
            vertices.append(xyz); triangles.append(tri[good]+offset); offset += len(xyz)
        return mesh_from_arrays(np.concatenate(vertices), np.concatenate(triangles)) if vertices else empty_mesh()

    def _clean_mesh(self, mesh, points, ground):
        if ground is None or not len(points):
            return empty_mesh()
        vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        if not len(triangles):
            return empty_mesh()
        xyz = vertices[triangles]
        centroid = xyz.mean(axis=1)
        normals = np.cross(xyz[:,1]-xyz[:,0], xyz[:,2]-xyz[:,0])
        normals /= np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-12)
        tree = cKDTree(points)
        near_support = tree.query(centroid)[0] <= self.support_radius_m
        # A supported centroid cannot certify a long triangle's remote corners.
        # Keep only locally supported measured triangles; do not mutate raw data.
        near_support &= (tree.query(xyz.reshape(-1,3))[0].reshape(-1,3) <= self.support_radius_m).all(axis=1)
        for i,j in ((0,1),(1,2),(2,0)):
            near_support &= np.linalg.norm(xyz[:,i]-xyz[:,j],axis=1) <= self.neighbour_limit_m
        is_ground = ((np.abs(normals @ ground['normal']) >= np.cos(np.deg2rad(35)))
            & (np.max(np.abs(xyz @ ground['normal']+ground['offset']),axis=1) <= self.ground_band_m))
        return mesh_from_arrays(vertices, triangles[near_support & ~is_ground])

    def _edges(self, tree):
        points, steps = [], []
        for f in self.frames:
            xyz, depth, valid = f['xyz'], f['depth'], f['valid']
            member = np.zeros(depth.shape, bool)
            member[valid] = tree.query(xyz[valid])[0] <= .06
            for axis in (0,1):
                for sign in (-1,1):
                    # np.roll(-sign) retrieves the neighbour in +sign direction.
                    neighbour = np.roll(depth, -sign, axis)
                    background = (member & np.roll(valid,-sign,axis)
                        & (neighbour-depth > .12 + .005*depth**2))
                    if axis == 0:
                        background[-1 if sign>0 else 0,:] = False
                    else:
                        background[:,-1 if sign>0 else 0] = False
                    pixel_delta = np.array([sign if axis==1 else 0, sign if axis==0 else 0, 0.])
                    step = f['t'][:3,:3] @ np.linalg.inv(f['k']) @ pixel_delta
                    points.append(xyz[background])
                    steps.append(np.repeat(step[None,:], int(background.sum()),axis=0))
        return (np.concatenate(points), np.concatenate(steps)) if points else (np.empty((0,3)),np.empty((0,3)))

    def _free_ray_conflicts(self, basis, lo, hi):
        conflicts, hits = 0, 0
        for f in self.frames:
            valid = f['valid'][::2,::2]
            direction = f['rays'][::2,::2][valid] @ basis
            depth = f['depth'][::2,::2][valid]
            origin = f['t'][:3,3] @ basis
            parallel = np.abs(direction) < 1e-12
            safe = np.where(parallel, 1., direction)
            ta, tb = (lo-origin)/safe, (hi-origin)/safe
            tlo, thi = np.minimum(ta,tb), np.maximum(ta,tb)
            tlo[parallel] = -np.inf; thi[parallel] = np.inf
            outside = np.any(parallel & ((origin<lo)|(origin>hi)),axis=1)
            near = np.maximum(tlo.max(axis=1), 0.)
            far = thi.min(axis=1)
            hit = ~outside & (far >= near) & (far > 0.)
            conflict = hit & (near < depth-(.08+.005*depth**2))
            hits += int(hit.sum()); conflicts += int(conflict.sum())
        return conflicts, hits

    def _enclosure(self, points, normals, ground):
        audit = dict(hypothesis='rectilinear closed enclosure inferred from observed face extents',
                     unseen_surfaces_measured=False, ground_contact_assumed=False)
        def reject(reason, conflicts=0):
            return empty_mesh(), dict(accepted=False, reason=reason, support_audit=audit,
                                      free_ray_conflicts=int(conflicts))
        if not len(points):
            return reject('no_seeded_instance_support')
        if len(points) < 80:
            return reject('insufficient_instance_support')
        if ground['normal'] @ self.up < np.cos(np.deg2rad(3)):
            return reject('enclosure_requires_nearly_level_support_plane')
        vertical = np.abs(normals @ self.up) < np.sin(np.deg2rad(20))
        vertical &= np.linalg.norm(normals,axis=1) > .5
        if vertical.sum() < 40:
            return reject('insufficient_vertical_face_support')
        ns = normals[vertical]
        # A horizontal reference only parametrizes angles; no world-axis prior.
        ref = np.eye(3)[np.argmin(np.abs(self.up))]
        u = np.cross(self.up,ref); u /= np.linalg.norm(u)
        v = np.cross(self.up,u)
        angles = np.mod(np.arctan2(ns@v,ns@u),np.pi)
        bins = np.minimum((angles/(np.pi/36)).astype(int),35)
        hist = np.bincount(bins,minlength=36)
        centre = (np.argmax(hist)+.5)*np.pi/36
        distance = np.abs((angles-centre+np.pi/2)%np.pi-np.pi/2)
        fit = ns[distance < np.deg2rad(10)].copy()
        anchor = np.cos(centre)*u+np.sin(centre)*v
        fit[fit@anchor<0] *= -1
        axis = np.median(fit,axis=0); axis -= (axis@self.up)*self.up
        axis /= np.linalg.norm(axis)
        other = np.cross(self.up,axis)
        basis = np.column_stack([axis,other,self.up])
        projected_normals = np.abs(normals @ basis)
        face_support = [int((projected_normals[:,i] >= np.cos(np.deg2rad(12))).sum()) for i in (0,1)]
        audit['orthogonal_face_support'] = face_support
        if min(face_support) < max(20, .025*len(points)):
            return reject('requires_two_orthogonal_observed_faces')
        projected = points @ basis
        lo,hi = np.quantile(projected,[.005,.995],axis=0)
        audit['robust_observed_bounds'] = [lo.tolist(),hi.tolist()]
        if np.any(hi-lo < .15) or np.any(hi-lo > 10.):
            return reject('degenerate_or_unbounded_instance_extent')
        floor_height = -(ground['offset']+ground['normal']@(basis[:,:2]@((lo[:2]+hi[:2])/2)))/(ground['normal']@self.up)
        if lo[2]-floor_height > .12 or lo[2] < floor_height-.04:
            return reject('ground_contact_not_observed')
        audit['ground_contact_assumed'] = True
        lo[2] = floor_height
        ep,es = self._edges(cKDTree(points))
        ep,es = ep@basis, es@basis
        edge_counts = {}
        for axis_index in range(3):
            for end in (0,1):
                key = f'{axis_index}_{"max" if end else "min"}'
                if axis_index==2 and end==0:
                    edge_counts[key] = int((projected[:,2]-floor_height <= .12).sum())
                    continue
                side = hi[axis_index] if end else lo[axis_index]
                directed = es[:,axis_index]*(1 if end else -1) > .3*np.linalg.norm(es,axis=1)
                edge_counts[key] = int((directed & (np.abs(ep[:,axis_index]-side) <= self.edge_tolerance_m)).sum())
        audit['background_bracketed_edge_counts'] = edge_counts
        if min(edge_counts.values()) < 3:
            return reject('observed_extents_not_bracketed_by_background_rays')
        conflicts,hits = self._free_ray_conflicts(basis,lo,hi)
        audit.update(free_ray_box_hits=hits, free_ray_conflict_fraction=conflicts/max(hits,1))
        if conflicts >= 3 and conflicts/max(hits,1) > .002:
            return reject('enclosure_contradicts_observed_free_rays',conflicts)
        box = o3d.geometry.TriangleMesh.create_box(*(hi-lo))
        box.vertices = o3d.utility.Vector3dVector((np.asarray(box.vertices)+lo)@basis.T)
        audit.update(model_basis=basis.tolist(),model_bounds=[lo.tolist(),hi.tolist()])
        return box, dict(accepted=True,reason='supported_rectilinear_hypothesis',
                         support_audit=audit,free_ray_conflicts=conflicts)

    def snapshot(self, raw_mesh=None):
        points = np.concatenate([f['xyz'][f['valid']] for f in self.frames]) if self.frames else np.empty((0,3))
        normals = np.concatenate([f['normals'][f['valid']] for f in self.frames]) if self.frames else np.empty((0,3))
        ground = self._ground(points,normals)
        ground_mask = np.zeros(len(points),bool) if ground is None else (
            (np.abs(points@ground['normal']+ground['offset']) <= self.ground_band_m)
            & (np.abs(normals@ground['normal']) >= np.cos(np.deg2rad(35))))
        selected = self._component(points,normals,ground)
        cleaned_points = points[selected]
        original = self._observed_mesh() if raw_mesh is None else raw_mesh
        cleaned_mesh = self._clean_mesh(original,cleaned_points,ground)
        inferred, completion = self._enclosure(cleaned_points,normals[selected],ground)
        return dict(version=VERSION, ground_plane=ground, measured_points_xyz=points,
            ground_points_xyz=points[ground_mask], cleaned_points_xyz=cleaned_points,
            unassigned_points_xyz=points[~selected & ~ground_mask],
            observed_mesh=cleaned_mesh, inferred_mesh=inferred,
            completed_mesh=cleaned_mesh+inferred, completion=completion,
            observed_frames=len(self.frames), category_input_used=False,
            reference_or_hidden_geometry_input_used=False,
            provenance='raw and cleaned observed support remain separate from any inferred enclosure')
