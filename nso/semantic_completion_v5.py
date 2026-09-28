"""V5 measured-support geometry and conservative marker-domain objectness.

The V3 geometric clusters, acceptance tests, primitive library, candidate rules,
likelihood and visibility implementation remain shared. This version changes
only geometric envelope fitting and a declared marker-domain objectness prior.
Predicted points never enter the mapper or TSDF. Objectness is shared by O/S;
only the existing shelf/box prior uses fine categories. Missing labels disable
marker gating and recover the actual geometric pathway.
"""
from dataclasses import dataclass
import numpy as np
import open3d as o3d
from scipy.ndimage import binary_closing, label
from nso.semantic_completion_v3 import ObjectCompletionModel as ObjectCompletionModelV3


@dataclass(frozen=True)
class CompletionConfigV5:
    # QualityMapperV2 keys are floor(point / .15), independently of TSDF voxel.
    quality_voxel_m: float = .15
    support_padding_voxels: float = .5
    face_normal_alignment: float = .9
    min_face_support_voxels: int = 4
    minimum_xy_prior_m: tuple = (.9, .65)
    ground_z_m: float = 0.
    minimum_height_prior_m: float = .9
    maximum_height_prior_m: float = 1.5
    unmarked_object_weight: float = .25

    def __post_init__(self):
        if self.quality_voxel_m != .15:
            raise ValueError('frozen quality ledger uses 0.15m keys')
        values = [self.support_padding_voxels, self.face_normal_alignment,
                  *self.minimum_xy_prior_m, self.ground_z_m,
                  self.minimum_height_prior_m, self.maximum_height_prior_m,
                  self.unmarked_object_weight]
        if not np.isfinite(values).all():
            raise ValueError('V5 priors must be finite')
        if not 0 <= self.support_padding_voxels <= .5:
            raise ValueError('support padding is bounded by half a quality voxel')
        if not 0 < self.face_normal_alignment <= 1 or self.min_face_support_voxels < 3:
            raise ValueError('invalid visible-face support rule')
        if len(self.minimum_xy_prior_m) != 2 or min(self.minimum_xy_prior_m) <= 0:
            raise ValueError('positive x/y priors required')
        if not 0 < self.minimum_height_prior_m <= self.maximum_height_prior_m:
            raise ValueError('invalid grounded-object height prior')
        if not 0 < self.unmarked_object_weight <= 1:
            raise ValueError('unmarked objectness must remain nonzero and <=1')


def _aligned_quality_keys(mapper, quality, max_points):
    """Recover exactly the keys corresponding to frozen quality_evidence rows."""
    items = [(key, row) for key, row in mapper.quality.items()
             if .12 < row['point'][2] < 1.8]
    if len(items) > max_points:
        items = [items[i] for i in np.linspace(0, len(items)-1, max_points, dtype=int)]
    points = np.asarray([row['point'] for _, row in items])
    if not np.array_equal(points, quality['point']):
        raise ValueError('quality point order does not match voxel-key evidence')
    return np.asarray([key for key, _ in items], dtype=np.int64)


def support_geometry(points, normals, keys, direction, config):
    """Fit a common envelope using measured support and inward face normals.

    Padding accounts for sparse evidence-voxel sampling, not sensor certainty.
    Outermost supported plane bins anchor actually observed faces. A minimum
    extent is a prior only when the opposite face is not observed. The grounded
    height convention is inherited from V3 and is explicit, not inferred GT.
    """
    points, normals, keys = np.asarray(points), np.asarray(normals), np.asarray(keys)
    scale = config.quality_voxel_m
    point_low, point_high = points.min(axis=0), points.max(axis=0)
    key_low = np.minimum(keys.min(axis=0)*scale, point_low)
    key_high = np.maximum((keys.max(axis=0)+1)*scale, point_high)
    low = np.maximum(point_low-config.support_padding_voxels*scale, key_low)
    high = np.minimum(point_high+config.support_padding_voxels*scale, key_high)
    support_low, support_high = low.copy(), high.copy()
    faces = []
    for axis in range(3):
        row = {}
        for name, sign in (('low', 1), ('high', -1)):
            valid = normals[:, axis]*sign >= config.face_normal_alignment
            bins = sorted(set(keys[valid, axis]))
            if name == 'high': bins.reverse()
            row[name] = None
            for plane_bin in bins:
                selected = valid & (keys[:, axis] == plane_bin)
                if np.count_nonzero(selected) < config.min_face_support_voxels: continue
                # Require support in at least two tangential evidence cells.
                tangent = [other for other in range(3) if other != axis]
                if len(np.unique(keys[selected][:, tangent], axis=0)) < config.min_face_support_voxels: continue
                coordinate = float(np.median(points[selected, axis]))
                extreme = point_low[axis] if name == 'low' else point_high[axis]
                if abs(coordinate-extreme) > scale: continue
                row[name] = dict(coordinate_m=coordinate, support_voxels=int(selected.sum()))
                break
        faces.append(row)
    axes = []
    inward = normals.mean(axis=0)
    for axis in (0, 1):
        lower, upper = faces[axis]['low'], faces[axis]['high']
        if lower is not None: low[axis] = lower['coordinate_m']
        if upper is not None: high[axis] = upper['coordinate_m']
        two_sided = lower is not None and upper is not None and high[axis] > low[axis]
        if high[axis] <= low[axis]:
            low[axis], high[axis] = support_low[axis], support_high[axis]
            lower = upper = None
        span = high[axis]-low[axis]
        prior = config.minimum_xy_prior_m[axis]
        desired = span if two_sided else max(span, prior)
        if lower is not None and upper is None:
            high[axis] = low[axis]+desired
        elif upper is not None and lower is None:
            low[axis] = high[axis]-desired
        elif not two_sided:
            if point_high[axis]-point_low[axis] < prior*.55:
                sign = np.sign(inward[axis]) if abs(inward[axis]) > .2 else -np.sign(direction[axis])
                anchor = float(np.median(points[:, axis]))
                if sign > 0: low[axis], high[axis] = anchor, anchor+desired
                elif sign < 0: low[axis], high[axis] = anchor-desired, anchor
                else:
                    middle = (low[axis]+high[axis])/2
                    low[axis], high[axis] = middle-desired/2, middle+desired/2
            else:
                middle = (low[axis]+high[axis])/2
                low[axis], high[axis] = middle-desired/2, middle+desired/2
        axes.append(dict(axis=axis, opposing_faces_observed=bool(two_sided),
                         minimum_extent_prior_used=bool(not two_sided and desired>span+1e-12)))
    low[2] = config.ground_z_m
    top = faces[2]['high']
    if top is not None: high[2] = top['coordinate_m']
    height = np.clip(high[2]-low[2], config.minimum_height_prior_m, config.maximum_height_prior_m)
    high[2] = low[2]+height
    if np.any(high <= low): raise ValueError('measured-support envelope is degenerate')
    audit = dict(method='quality voxel support with visible-plane anchoring',
                 quality_voxel_m=scale, measured_low=point_low.tolist(), measured_high=point_high.tolist(),
                 support_low=support_low.tolist(), support_high=support_high.tolist(),
                 fitted_low=low.tolist(), fitted_high=high.tolist(), planes=faces, axes=axes,
                 grounded_height_prior=True, marker_objectness_is_calibrated=False)
    return (low+high)/2, high-low, audit


class ObjectCompletionModel(ObjectCompletionModelV3):
    def __init__(self,mapper,semantic,fine_categories=True,config=None):
        self.v5_config = config or CompletionConfigV5()
        self.marker_objectness_active = False
        if not isinstance(self.v5_config, CompletionConfigV5):
            raise TypeError("config must be CompletionConfigV5")
        self.mapper=mapper;self.semantic=semantic;self.objects=[]
        self.points=np.empty((0,3));self.weights=np.empty(0)
        self.hypothesis_ids=np.empty(0,dtype=int);self.hypothesis_rays=[]
        q=mapper.quality_evidence(max_points=10000)
        if q is None:return
        evidence_keys = _aligned_quality_keys(mapper, q, 10000)
        marker_domain = (getattr(mapper.config, 'semantic_source', None) == 'rgb_marker'
                         and getattr(mapper.config, 'appearance', None) == 'marked')
        marker_available = bool(np.any((q['label'] == 2) | (q['label'] == 3)))
        self.marker_objectness_active = bool(semantic and marker_domain and marker_available)
        pts=q['point'];r=mapper.shape[0]-1-np.floor(pts[:,1]/.2).astype(int);c=np.floor(pts[:,0]/.2).astype(int)
        inside=(r>=0)&(r<mapper.shape[0])&(c>=0)&(c<mapper.shape[1])
        occupancy=np.zeros(mapper.shape,bool);occupancy[r[inside],c[inside]]=True
        # Geometric connected components are shared; semantic labels never
        # decide which components the nonsemantic variant is allowed to see.
        groups,_=label(binary_closing(occupancy,structure=np.ones((3,3))))
        ids=np.zeros(len(pts),int);ids[inside]=groups[r[inside],c[inside]]
        samples=[];weights=[];hypothesis_ids=[]
        for group in sorted(set(ids)-{0}):
            indices=np.flatnonzero(ids==group)
            if len(indices)<8:continue
            cloud=pts[indices];low,high=np.quantile(cloud,[.05,.95],axis=0);span=high-low
            if max(span[:2])>2.2 or span[2]<.25:continue
            hist=np.array([np.mean((q['bits'][indices]&(1<<b))!=0) for b in range(8)])
            az=-np.pi+(int(np.argmax(hist))+.5)*2*np.pi/8
            direction=np.array([np.cos(az),np.sin(az)])
            center,dims,geometry_audit = support_geometry(
                cloud, q['normal'][indices], evidence_keys[indices], direction, self.v5_config)
            labels=q['label'][indices];known=labels[(labels==2)|(labels==3)]
            prob=.5
            object_probability=1.
            if semantic and fine_categories and len(known):
                vote=float(np.mean(known==3))
                prob=.1+.8*vote
            marker_supported = bool(len(known))
            if self.marker_objectness_active and not marker_supported:
                object_probability = self.v5_config.unmarked_object_weight
            obj=dict(center=center,dims=dims,shelf_probability=prob,object_probability=object_probability,
                     observed_points=len(indices),geometry_audit=geometry_audit,
                     marker_supported=marker_supported,marker_objectness_active=self.marker_objectness_active)
            self.objects.append(obj)
            box_points,box_area=self._box(center-dims/2,dims)
            box_mesh=o3d.geometry.TriangleMesh.create_box(*dims).translate(center-dims/2)
            shelf_mesh=o3d.geometry.TriangleMesh()
            def shelf_part(origin,size):
                shelf_mesh.__iadd__(o3d.geometry.TriangleMesh.create_box(*size).translate(origin))
                return self._box(origin,size)
            shelf_points=[];shelf_area=[]
            x,y,z=center-dims/2;sx,sy,h=dims
            for level in (.15,.5,.85):
                p,a=shelf_part([x,y,z+level*h],[sx,sy,.08]);shelf_points.extend(p);shelf_area.extend(a)
            for dx in (0,sx-.08):
                for dy in (0,sy-.08):
                    p,a=shelf_part([x+dx,y+dy,z],[.08,.08,h]);shelf_points.extend(p);shelf_area.extend(a)
            # A possible backboard lies on the initially observed side. Open
            # shelves reject this component with their already observed depth.
            axis=int(np.argmax(np.abs(direction)))
            board_origin=np.array([x,y,z]);board_dims=dims.copy();board_dims[axis]=.08
            if direction[axis]>0:board_origin[axis]+=dims[axis]-.08
            p,a=shelf_part(board_origin,board_dims);shelf_points.extend(p);shelf_area.extend(a)
            for mesh,number in ((box_mesh,len(box_points)),(shelf_mesh,len(shelf_points))):
                hypothesis_ids.extend([len(self.hypothesis_rays)]*number)
                ray=o3d.t.geometry.RaycastingScene(nthreads=1)
                ray.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
                self.hypothesis_rays.append(ray)
            # Both variants receive the SAME likelihood from measured surface
            # fit. A fixed 50/50 mixture would be an unnecessarily weak geometry
            # ablation once multiple visible faces reveal solid/shelf structure.
            # 6cm is a model-discrepancy scale, not a claimed calibrated noise SD.
            observed=o3d.core.Tensor(cloud.astype(np.float32))
            errors=[float(np.mean(np.minimum(ray.compute_distance(observed,nthreads=1).numpy(),.15)**2))
                    for ray in self.hypothesis_rays[-2:]]
            log_likelihood_ratio=(errors[0]-errors[1])/(2*.06**2)
            log_odds=np.clip(np.log(prob/(1-prob))+log_likelihood_ratio,-6.,6.)
            obj['shelf_prior_probability']=prob
            prob=float(1/(1+np.exp(-log_odds)))
            obj.update(shelf_probability=prob,fit_errors_m2=errors)
            samples.extend(box_points);weights.extend(np.asarray(box_area)*(1-prob)*object_probability)
            samples.extend(shelf_points);weights.extend(np.asarray(shelf_area)*prob*object_probability)
        if not samples:return
        points=np.asarray(samples);area=np.asarray(weights)
        unexplained=np.ones(len(points),bool)
        # Project into past actual RGB-D frames. A hypothesis in measured free
        # space or on an already measured surface has no future reward.
        for frame in mapper.keyframes:
            ids=np.flatnonzero(unexplained)
            if not len(ids):break
            local=(points[ids]-frame.world_from_camera[:3,3])@frame.world_from_camera[:3,:3]
            depth=local[:,2];positive=depth>.15
            u=np.rint(local[:,0]/np.maximum(depth,.01)*frame.intrinsic[0,0]+frame.intrinsic[0,2]).astype(int)
            v=np.rint(local[:,1]/np.maximum(depth,.01)*frame.intrinsic[1,1]+frame.intrinsic[1,2]).astype(int)
            visible=positive&(u>=0)&(u<frame.depth_m.shape[1])&(v>=0)&(v<frame.depth_m.shape[0])
            selected=np.flatnonzero(visible)
            measured=frame.depth_m[v[selected],u[selected]]
            explained=(measured>0)&(depth[selected]<=measured+.10)
            unexplained[ids[selected[explained]]]=False
        self.points=points[unexplained];self.weights=area[unexplained]
        self.hypothesis_ids=np.asarray(hypothesis_ids)[unexplained]
        if len(self.points)>3500:
            # Preserve estimated area when deterministically thinning hypotheses.
            old_area=self.weights.sum();ids=np.linspace(0,len(self.points)-1,3500,dtype=int)
            self.points=self.points[ids];self.weights=self.weights[ids];self.hypothesis_ids=self.hypothesis_ids[ids]
            self.weights*=old_area/max(self.weights.sum(),1e-9)
