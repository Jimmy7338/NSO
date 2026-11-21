"""Observed-object shape hypotheses for semantic next-best-view planning.

Both ablations use the same geometry clusters, primitive library, candidate
budget, history rejection, visibility queries and reconstruction backend.
Semantics changes the prior probability of the shelf/solid-box hypotheses.
Both variants update that prior using the fit to actually measured geometry.
Hypotheses contradicted or already measured by past depth images are removed.
They are planning hypotheses ONLY and are never fused into the TSDF or evaluator.
"""
import numpy as np
import open3d as o3d
from scipy.ndimage import binary_closing,label,distance_transform_edt
from nso.camera_mapping_v2 import CameraQualityMapperV2
from nso.camera_joint_planner_v2 import CameraJointPlannerV2
from env.virtual3d import camera_pose


class SemanticHistoryMapperV3(CameraQualityMapperV2):
    def __init__(self,shape,config,truncation_m=.12):
        super().__init__(shape,config,truncation_m)
        self.keyframes=[]

    def update(self,frame,scan=None):
        super().update(frame,scan)
        if not self.keyframes:
            self.keyframes.append(frame)
        else:
            last=self.keyframes[-1]
            moved=np.linalg.norm(last.world_from_camera[:3,3]-frame.world_from_camera[:3,3])>.45
            turned=last.world_from_camera[:3,2]@frame.world_from_camera[:3,2]<.8
            if moved or turned:self.keyframes.append(frame)
        # Bound storage while preserving the whole trajectory, not just recency.
        if len(self.keyframes)>512:self.keyframes=self.keyframes[::2]+[self.keyframes[-1]]


class ObjectCompletionModel:
    def __init__(self,mapper,semantic,fine_categories=True):
        self.mapper=mapper;self.semantic=semantic;self.objects=[]
        self.points=np.empty((0,3));self.weights=np.empty(0)
        self.hypothesis_ids=np.empty(0,dtype=int);self.hypothesis_rays=[]
        q=mapper.quality_evidence(max_points=10000)
        if q is None:return
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
            dims=np.array([max(.9,span[0]),max(.65,span[1]),np.clip(high[2]+.08,.9,1.5)])
            center=(low+high)/2;center[2]=dims[2]/2
            hist=np.array([np.mean((q['bits'][indices]&(1<<b))!=0) for b in range(8)])
            az=-np.pi+(int(np.argmax(hist))+.5)*2*np.pi/8
            direction=np.array([np.cos(az),np.sin(az)])
            inward=np.mean(q['normal'][indices],axis=0)
            for axis in (0,1):
                if span[axis]<dims[axis]*.55:
                    # Cross products of camera x/y depth tangents point inward.
                    # Use the measured face normal to extend behind a partial
                    # face; pooled azimuth bins can refer to other past faces.
                    sign=np.sign(inward[axis]) if abs(inward[axis])>.2 else -np.sign(direction[axis])
                    center[axis]+=sign*dims[axis]/2
            labels=q['label'][indices];known=labels[(labels==2)|(labels==3)]
            prob=.5
            object_probability=1.
            if semantic and fine_categories and len(known):
                vote=float(np.mean(known==3))
                prob=.1+.8*vote
            if semantic and np.any(labels>0):object_probability=float(np.mean(labels[labels>0]>1))
            obj=dict(center=center,dims=dims,shelf_probability=prob,object_probability=object_probability,observed_points=len(indices))
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

    @staticmethod
    def _box(origin,dims):
        origin=np.asarray(origin);dims=np.asarray(dims);points=[];weights=[]
        for normal in range(3):
            axes=[i for i in range(3) if i!=normal]
            counts=[max(1,int(np.ceil(dims[a]/.22))) for a in axes]
            area=dims[axes[0]]*dims[axes[1]]/(counts[0]*counts[1])
            for side in (0.,1.):
                for i in range(counts[0]):
                    for j in range(counts[1]):
                        p=origin.copy().astype(float);p[normal]+=side*dims[normal]
                        p[axes[0]]+=(i+.5)*dims[axes[0]]/counts[0]
                        p[axes[1]]+=(j+.5)*dims[axes[1]]/counts[1]
                        points.append(p);weights.append(area)
        return points,weights

    def candidates(self,safe,distances,limit=20):
        reachable=safe&(distances>=0)
        if not reachable.any():return []
        retreat,indices=distance_transform_edt(~reachable,return_indices=True)
        proposals=set()
        for obj in self.objects:
            center=obj['center']
            # Same radii / angular proposal budget for semantic and geometry.
            for radius in (.9,1.4,2.):
                for az in np.arange(0,2*np.pi,np.pi/4):
                    xy=center[:2]+radius*np.array([np.cos(az),np.sin(az)])
                    r=self.mapper.shape[0]-1-int(np.floor(xy[1]/.2));c=int(np.floor(xy[0]/.2))
                    if not (0<=r<safe.shape[0] and 0<=c<safe.shape[1]) or retreat[r,c]>2:continue
                    proposals.add(tuple(map(int,indices[:,r,c])))
        ordered=sorted(proposals,key=lambda c:(distances[c],c))
        if len(ordered)>limit:ordered=[ordered[i] for i in np.linspace(0,len(ordered)-1,limit,dtype=int)]
        return ordered

    def gain(self,cell,heading,ray):
        if not len(self.points):return np.empty(0)
        c=self.mapper.config;pose=camera_pose(cell,heading,c,self.mapper.shape[0])
        local=(self.points-pose[:3,3])@pose[:3,:3];z=local[:,2];tan=np.tan(np.deg2rad(c.fov_deg/2))
        visible=(z>.2)&(z<c.max_depth_m)&(np.abs(local[:,0])<z*tan)&(np.abs(local[:,1])<z*tan*c.height_px/c.width_px)
        ids=np.flatnonzero(visible)
        if ray is not None and len(ids):
            delta=self.points[ids]-pose[:3,3];distance=np.linalg.norm(delta,axis=1)
            rays=np.column_stack([np.tile(pose[:3,3],(len(ids),1)),delta]).astype(np.float32)
            hit=ray.cast_rays(o3d.core.Tensor(rays),nthreads=1)['t_hit'].numpy()
            visible[ids]&=hit>=1-.06/np.maximum(distance,.2)
        # Evaluate self-occlusion separately under each alternative shape.
        # Combining box and shelf into one mesh would let mutually exclusive
        # hypotheses occlude each other and systematically suppress shelves.
        for hypothesis_id in np.unique(self.hypothesis_ids[visible]):
            ids=np.flatnonzero(visible&(self.hypothesis_ids==hypothesis_id))
            delta=self.points[ids]-pose[:3,3]
            rays=np.column_stack([np.tile(pose[:3,3],(len(ids),1)),delta]).astype(np.float32)
            hit=self.hypothesis_rays[hypothesis_id].cast_rays(o3d.core.Tensor(rays),nthreads=1)['t_hit'].numpy()
            visible[ids]&=hit>=1-1e-4
        return visible*self.weights


class ObjectJointPlannerV3(CameraJointPlannerV2):
    def __init__(self,*args,hypothesis_weight=4.,fine_categories=True,**kwargs):
        super().__init__(*args,**kwargs);self.hypothesis_weight=hypothesis_weight;self.object_model=None;self.fine_categories=fine_categories

    def _recall_increment(self,qgain,npoints,camera_fraction,coverage_gate,q0):
        count=0 if self.quality is None else len(self.quality['point'])
        precision=super()._recall_increment(qgain[:count],npoints,camera_fraction,coverage_gate,q0)
        # Predicted NEW surfaces are completion, not improvements to the
        # conditional quality of already measured surfaces. In particular they
        # must not vanish when q0 reaches 1 or when the camera footprint is small.
        # The common physical-area denominator is encoded in _quality_gain.
        completion=self.quality_weight*float(qgain[count:].sum())/npoints
        return precision+completion

    def _inspection_cells(self,safe,distances):
        self.object_model=ObjectCompletionModel(self.mapper,self.semantic,self.fine_categories)
        uniform=super()._inspection_cells(safe,distances)
        if len(uniform)>10:uniform=[uniform[i] for i in np.linspace(0,len(uniform)-1,10,dtype=int)]
        return list(dict.fromkeys(uniform+self.object_model.candidates(safe,distances,limit=10)))

    def _quality_gain(self,cell,heading):
        gain,completion,precision=super()._quality_gain(cell,heading)
        if self.object_model is None:return gain,completion,precision
        area=self.object_model.gain(cell,heading,self.ray)
        # Convert predicted unobserved area to the shared normalized quality
        # surrogate. This is an explicit estimated surface-area normalizer,
        # not the evaluator's total surface area or class-weighted GT score.
        normalizer=2*self.mapper.shape[0]*self.mapper.shape[1]*self.config.resolution_m**2
        extra=area*self.hypothesis_weight*max(len(gain),1)/normalizer
        return np.r_[gain,extra],completion+float(area.sum()),precision
