"""V25 bounded prototype: disparity planes and measured background edge brackets.

Only enclosure parameter estimation differs from frozen V24. No semantic,
reference, world dimensions, sensor simulator, or quality input is accepted.
"""
import hashlib
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.ndimage import label
from nso.observed_shape_v24 import ObservedShapeBackendV24, empty_mesh

VERSION = 'observed-shape-v25-prototype-2'


class ObservedShapeBackendV25(ObservedShapeBackendV24):
    def __init__(self, up=(0., 0., 1.), *, reference_fx_px=480., baseline_m=.12,
                 disparity_sigma_px=.25):
        super().__init__(up)
        values = np.asarray([reference_fx_px, baseline_m, disparity_sigma_px], float)
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError('positive public stereo uncertainty calibration required')
        self.fb = float(reference_fx_px*baseline_m)
        self.disparity_sigma_px = float(disparity_sigma_px)
        # Fixed robust estimation settings, expressed in calibrated residuals.
        self.inlier_sigma = 3.
        self.huber_sigma = 2.5
        self.hypotheses_per_patch = 64
        self.max_planes_per_frame = 3
        self.last_parameter_audit = None

    @staticmethod
    def _weighted_median(values, weights):
        values, weights = np.asarray(values), np.asarray(weights)
        order = np.argsort(values, kind='stable')
        return float(values[order][np.searchsorted(np.cumsum(weights[order]), weights.sum()/2)])

    def _plane_patches(self, points):
        """Plane disparity is affine in normalized pixel coordinates.

        Homoskedastic disparity residuals fit near/far observations in their
        actual uncertainty units. RANSAC only separates faces; IRLS refines each.
        """
        tree = cKDTree(points)
        patches, seen = [], set()
        for frame_index, f in enumerate(self.frames):
            identity = hashlib.sha256()
            for a in (f['depth'],f['k'],f['t']):
                identity.update(a.tobytes())
            identity = identity.digest()
            if identity in seen:
                continue
            seen.add(identity)
            member = np.zeros(f['valid'].shape, bool)
            member[f['valid']] = tree.query(f['xyz'][f['valid']])[0] <= 1e-7
            vv, uu = np.nonzero(member)
            if len(vv) < 40:
                continue
            optical = np.stack([(uu-f['k'][0,2])/f['k'][0,0],
                (vv-f['k'][1,2])/f['k'][1,1],np.ones(len(vv))],axis=1)
            disparity = self.fb/f['depth'][vv,uu]
            remaining = np.ones(len(vv),bool)
            rng = np.random.default_rng(2501+frame_index)
            for patch_id in range(self.max_planes_per_frame):
                ids = np.flatnonzero(remaining)
                if len(ids) < 40:
                    break
                best = None
                for _ in range(self.hypotheses_per_patch):
                    chosen = rng.choice(ids,3,replace=False)
                    matrix = optical[chosen]
                    if abs(np.linalg.det(matrix)) < 1e-5:
                        continue
                    coefficient = np.linalg.solve(matrix,disparity[chosen])
                    residual = (disparity-optical@coefficient)/self.disparity_sigma_px
                    mask = remaining & (np.abs(residual) <= self.inlier_sigma)
                    score = (int(mask.sum()),-float(np.minimum(residual[remaining]**2,9).sum()))
                    if best is None or score > best[0]:
                        best = (score,mask,coefficient)
                if best is None or best[0][0] < 40:
                    break
                mask, coefficient = best[1], best[2]
                for _ in range(6):
                    residual = (disparity-optical@coefficient)/self.disparity_sigma_px
                    mask = remaining & (np.abs(residual) <= self.inlier_sigma)
                    if mask.sum() < 40:
                        break
                    weight = np.minimum(1.,self.huber_sigma/np.maximum(np.abs(residual[mask]),1e-12))
                    a = optical[mask]*np.sqrt(weight[:,None])
                    b = disparity[mask]*np.sqrt(weight)
                    coefficient = np.linalg.lstsq(a,b,rcond=None)[0]
                if mask.sum() < 40:
                    break
                remaining[mask] = False
                length = np.linalg.norm(coefficient)
                normal = f['t'][:3,:3]@(coefficient/length)
                if abs(normal@self.up) >= np.sin(np.deg2rad(20)):
                    continue
                predicted = optical[mask]@coefficient
                if np.any(predicted <= 0):
                    continue
                corrected = (optical[mask]*(self.fb/predicted)[:,None])@f['t'][:3,:3].T+f['t'][:3,3]
                inlier_map = np.zeros(member.shape,bool)
                inlier_map[vv[mask],uu[mask]] = True
                # Normal information from calibrated disparity affine-fit covariance.
                covariance = self.disparity_sigma_px**2*np.linalg.pinv(optical[mask].T@optical[mask])
                tangent = np.eye(3)-np.outer(coefficient,coefficient)/length**2
                angular_variance = float(np.trace(tangent@covariance@tangent)/length**2)
                patches.append(dict(frame=f,frame_index=frame_index,patch_id=patch_id,
                    coefficient=coefficient,normal=normal,corrected=corrected,
                    count=int(mask.sum()),member=member,inlier_map=inlier_map,
                    angular_variance=max(angular_variance,1e-12),
                    residual_sigma_median=float(np.median(np.abs((disparity-optical@coefficient)[mask]))/self.disparity_sigma_px)))
        return patches,len(seen)

    def _estimate_parameters(self, points):
        patches,unique_frames = self._plane_patches(points)
        audit = dict(method='robust affine disparity planes plus valid background edge intervals',
            calibration=dict(fb=self.fb,disparity_sigma_px=self.disparity_sigma_px),
            unique_depth_pose_frames=unique_frames,plane_patches=len(patches),
            quantiles_of_mixed_xyz_used=False,hidden_size_or_world_axis_input_used=False,
            edge_estimation_background_disparity_sigma=3.,
            edge_estimation_min_contiguous_pixels=3)
        if len(patches) < 2:
            return None,audit,'insufficient_robust_observed_planes'
        ref = np.eye(3)[np.argmin(np.abs(self.up))]
        u = np.cross(self.up,ref);u /= np.linalg.norm(u);v = np.cross(self.up,u)
        angles = np.array([np.arctan2(p['normal']@v,p['normal']@u) for p in patches])
        weights = np.array([1/p['angular_variance'] for p in patches])
        vector = np.sum(weights*np.exp(4j*angles))
        if abs(vector) < 1e-8*weights.sum():
            return None,audit,'inconsistent_orthogonal_plane_directions'
        theta = np.angle(vector)/4
        # Joint, continuous orthogonal direction; robust angular IRLS, no angle bins.
        for _ in range(6):
            residual = (angles-theta+np.pi/4)%(np.pi/2)-np.pi/4
            scale = np.sqrt(np.array([p['angular_variance'] for p in patches]))
            robust = np.minimum(1.,self.huber_sigma*scale/np.maximum(np.abs(residual),1e-12))
            theta += np.sum(weights*robust*residual)/np.sum(weights*robust)
        axis = np.cos(theta)*u+np.sin(theta)*v
        basis = np.column_stack([axis,np.cross(self.up,axis),self.up])
        offsets = {(i,end):[] for i in range(3) for end in (0,1)}
        edges = {(i,end):[] for i in range(3) for end in (0,1)}
        patch_audit = []
        for p in patches:
            f = p['frame'];coefficient = p['coefficient']
            normal_alignment = np.abs(p['normal']@basis)
            axis_id = int(np.argmax(normal_alignment[:2]))
            if normal_alignment[axis_id] < np.cos(np.deg2rad(12)):
                continue
            projected = p['corrected']@basis
            location = float(projected[:,axis_id].mean())
            camera = f['t'][:3,3]@basis
            end = int(camera[axis_id] > location)
            if abs(camera[axis_id]-location) <= .15:
                continue
            # One face patch supplies a robust plane position; pixel count is
            # already reflected in its calibration covariance, not a new view.
            offsets[(axis_id,end)].append((location,1/p['angular_variance']))
            residual = p['residual_sigma_median']
            patch_audit.append(dict(frame=p['frame_index'],axis=axis_id,end=end,
                points=p['count'],offset_m=location,residual_sigma_median=residual))
            depth,valid = f['depth'],f['valid']
            for image_axis in (0,1):
                for sign in (-1,1):
                    neighbour = np.roll(depth,-sign,image_axis)
                    support = p['inlier_map'] & np.roll(valid,-sign,image_axis)
                    support &= neighbour-depth > .12+.005*depth**2
                    # A depth tail on the SAME fitted plane is not background.
                    # Require calibrated disagreement with its neighbour ray's
                    # predicted disparity, plus a continuous measured edge.
                    grid_v,grid_u = np.indices(depth.shape)
                    next_u = grid_u+(sign if image_axis==1 else 0)
                    next_v = grid_v+(sign if image_axis==0 else 0)
                    predicted_neighbour = (coefficient[0]*(next_u-f['k'][0,2])/f['k'][0,0]
                        +coefficient[1]*(next_v-f['k'][1,2])/f['k'][1,1]+coefficient[2])
                    observed_neighbour = np.divide(self.fb,neighbour,
                        out=np.zeros(neighbour.shape),where=neighbour>0)
                    support &= predicted_neighbour-observed_neighbour > self.inlier_sigma*self.disparity_sigma_px
                    if image_axis == 0:
                        support[-1 if sign>0 else 0,:] = False
                    else:
                        support[:,-1 if sign>0 else 0] = False
                    components,number = label(support,structure=np.ones((3,3),int))
                    sizes = np.bincount(components.ravel(),minlength=number+1)
                    continuous = sizes >= 3;continuous[0] = False
                    support &= continuous[components]
                    vv,uu = np.nonzero(support)
                    if len(vv) == 0:
                        continue
                    ray = np.stack([(uu-f['k'][0,2])/f['k'][0,0],
                        (vv-f['k'][1,2])/f['k'][1,1],np.ones(len(vv))],axis=1)
                    delta = np.array([sign/f['k'][0,0] if image_axis==1 else 0,
                        sign/f['k'][1,1] if image_axis==0 else 0,0.])
                    ray2 = ray+delta
                    d1,d2 = ray@coefficient,ray2@coefficient
                    good = (d1>0)&(d2>0)
                    if not good.any():
                        continue
                    one = ((ray[good]*(self.fb/d1[good])[:,None])@f['t'][:3,:3].T+f['t'][:3,3])@basis
                    two = ((ray2[good]*(self.fb/d2[good])[:,None])@f['t'][:3,:3].T+f['t'][:3,3])@basis
                    step = two-one;length = np.linalg.norm(step,axis=1)
                    midpoint = (one+two)/2
                    for i in range(3):
                        for edge_end in (0,1):
                            directed = step[:,i]*(1 if edge_end else -1) > .3*length
                            if directed.any():
                                # Robust within a patch, then independent-patch aggregate.
                                edges[(i,edge_end)].append((float(np.median(midpoint[directed,i])),
                                    1/p['angular_variance'],int(directed.sum()),
                                    float(np.median(np.abs(step[directed,i]))/2)))
        bounds = np.empty((2,3))
        boundary_audit = {}
        for i in range(3):
            for end in (0,1):
                if i==2 and end==0:
                    # Ground contact is still decided by the unchanged V24 gate.
                    bounds[end,i] = float(np.min(points@self.up))
                    continue
                values = offsets[(i,end)] or edges[(i,end)]
                if not values:
                    audit.update(plane_support=patch_audit,boundary_support=boundary_audit)
                    return None,audit,'observed_extents_not_bracketed_by_background_rays'
                bounds[end,i] = self._weighted_median([x[0] for x in values],[x[1] for x in values])
                boundary_audit[f'{i}_{end}'] = dict(source='observed_plane' if offsets[(i,end)] else 'background_edge_interval',
                    groups=len(values),position_m=float(bounds[end,i]),
                    edge_pixel_half_width_m=None if offsets[(i,end)] else [x[3] for x in values])
        audit.update(plane_support=patch_audit,boundary_support=boundary_audit,
            model_basis=basis.tolist(),model_bounds=bounds.tolist())
        return (basis,bounds[0],bounds[1]),audit,None

    def _enclosure(self, points, normals, ground):
        # Support and rejection gates below preserve frozen V24 constants and
        # checks. Only the parameter-estimation stage has been replaced.
        audit = dict(hypothesis='rectilinear closed enclosure inferred from observed face extents',
                     unseen_surfaces_measured=False, ground_contact_assumed=False)
        def reject(reason,conflicts=0):
            return empty_mesh(),dict(accepted=False,reason=reason,support_audit=audit,
                free_ray_conflicts=int(conflicts))
        if not len(points):
            return reject('no_seeded_instance_support')
        if len(points)<80:
            return reject('insufficient_instance_support')
        if ground['normal']@self.up < np.cos(np.deg2rad(3)):
            return reject('enclosure_requires_nearly_level_support_plane')
        vertical = np.abs(normals@self.up)<np.sin(np.deg2rad(20))
        vertical &= np.linalg.norm(normals,axis=1)>.5
        if vertical.sum()<40:
            return reject('insufficient_vertical_face_support')
        parameters,estimate,reason = self._estimate_parameters(points)
        self.last_parameter_audit = estimate
        audit['parameter_estimation'] = estimate
        if parameters is None:
            return reject(reason)
        basis,lo,hi = parameters
        projected_normals = np.abs(normals@basis)
        face_support = [int((projected_normals[:,i]>=np.cos(np.deg2rad(12))).sum()) for i in (0,1)]
        audit['orthogonal_face_support'] = face_support
        if min(face_support)<max(20,.025*len(points)):
            return reject('requires_two_orthogonal_observed_faces')
        projected = points@basis
        audit['robust_observed_bounds'] = [lo.tolist(),hi.tolist()]
        if np.any(hi-lo<.15) or np.any(hi-lo>10.):
            return reject('degenerate_or_unbounded_instance_extent')
        floor_height = -(ground['offset']+ground['normal']@(basis[:,:2]@((lo[:2]+hi[:2])/2)))/(ground['normal']@self.up)
        if lo[2]-floor_height>.12 or lo[2]<floor_height-.04:
            return reject('ground_contact_not_observed')
        audit['ground_contact_assumed'] = True
        lo[2] = floor_height
        ep,es = self._edges(cKDTree(points));ep,es=ep@basis,es@basis
        edge_counts = {}
        for axis_index in range(3):
            for end in (0,1):
                key=f'{axis_index}_{"max" if end else "min"}'
                if axis_index==2 and end==0:
                    edge_counts[key]=int((projected[:,2]-floor_height<=.12).sum());continue
                side=hi[axis_index] if end else lo[axis_index]
                directed=es[:,axis_index]*(1 if end else -1)>.3*np.linalg.norm(es,axis=1)
                edge_counts[key]=int((directed&(np.abs(ep[:,axis_index]-side)<=self.edge_tolerance_m)).sum())
        audit['background_bracketed_edge_counts']=edge_counts
        if min(edge_counts.values())<3:
            return reject('observed_extents_not_bracketed_by_background_rays')
        conflicts,hits=self._free_ray_conflicts(basis,lo,hi)
        audit.update(free_ray_box_hits=hits,free_ray_conflict_fraction=conflicts/max(hits,1))
        if conflicts>=3 and conflicts/max(hits,1)>.002:
            return reject('enclosure_contradicts_observed_free_rays',conflicts)
        box=o3d.geometry.TriangleMesh.create_box(*(hi-lo))
        box.vertices=o3d.utility.Vector3dVector((np.asarray(box.vertices)+lo)@basis.T)
        audit.update(model_basis=basis.tolist(),model_bounds=[lo.tolist(),hi.tolist()])
        return box,dict(accepted=True,reason='supported_rectilinear_hypothesis',
            support_audit=audit,free_ray_conflicts=conflicts)

    def snapshot(self, raw_mesh=None):
        result = super().snapshot(raw_mesh)
        result['version'] = VERSION
        return result
