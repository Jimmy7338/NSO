"""Audited adaptation of original V3 full object-template construction.

Only input grouping, explicit label-dictionary conversion, ablation switches and
receipts differ. Template generation/fit/history geometry and all constants are
preserved. Original _box and gain (including separate self-occlusion) are inherited.
"""
import numpy as np
import open3d as o3d
from scipy.ndimage import binary_closing,label
from nso.semantic_completion_v3 import ObjectCompletionModel

ORIGINAL_SOURCE_SHA256='abf559a87049e5fe350713e31ca4cd8746af92c4ba2ffddff9663f805a473310'

def measured_groups(mapper,q):
    """Original XY components within shared measured-instance/unknown partitions."""
    pts=q['point'];n=len(pts)
    if hasattr(mapper,'instance_support_index'):
        rows=[(key,row)for key,row in mapper.quality.items()if .12<row['point'][2]<1.8]
        if len(rows)>10000:rows=[rows[i]for i in np.linspace(0,len(rows)-1,10000,dtype=int)]
        if len(rows)!=n or not np.array_equal(np.asarray([x['point']for _,x in rows]),pts):
            raise ValueError('quality source-key alignment changed')
        keys=np.asarray([key for key,_ in rows],np.int64).reshape(-1,3)
        tracks=mapper.instance_support_index.memberships_for_keys(keys)
    else:
        if getattr(mapper,'asset_grouping_profile',None) is not None:
            raise ValueError('declared instance profile requires measured support index')
        tracks=np.zeros(n,np.int64)
    r=mapper.shape[0]-1-np.floor(pts[:,1]/.2).astype(int);c=np.floor(pts[:,0]/.2).astype(int)
    inside=(r>=0)&(r<mapper.shape[0])&(c>=0)&(c<mapper.shape[1]);stride=int(np.prod(mapper.shape))+1
    result=[]
    for track in np.unique(tracks):
        selected=(tracks==track)&inside;occupancy=np.zeros(mapper.shape,bool);occupancy[r[selected],c[selected]]=True
        groups,_=label(binary_closing(occupancy,structure=np.ones((3,3))))
        ids=np.zeros(n,int);ids[selected]=groups[r[selected],c[selected]]
        for group in sorted(set(ids)-{0}):result.append((int(track)*stride+int(group),np.flatnonzero(ids==group),int(track)))
    return result

class FullTemplateModel(ObjectCompletionModel):
    def __init__(self,mapper,semantic,fine_categories=True,*,fit_update=True,history_rejection=True):
        self.mapper=mapper;self.semantic=semantic;self.objects=[]
        self.points=np.empty((0,3));self.weights=np.empty(0)
        self.hypothesis_ids=np.empty(0,dtype=int);self.hypothesis_rays=[]
        self.audit=dict(schema='full_dual_template_mechanism_v1',fit_update=bool(fit_update),
            history_rejection=bool(history_rejection),prior_rebuilt_from_current_measurements=True,
            original_candidates_method_used=False,history_frames=len(mapper.keyframes),
            raw_template_points=0,points_after_history=0,output_points=0,
            removed_by_free_space=0,removed_as_measured_or_depth_tolerance=0,
            original_global_point_cap=3500,hidden_truth_input=False,
            current_label_dictionary={2:'open',3:'closed'},template_label_dictionary={2:'box',3:'shelf'},
            label_dictionary_conversion='2open->3shelf;3closed->2box at template prior entrance only',
            probability_interpretation='heuristic mean-distance Gibbs pseudo-posterior, not calibrated probability')
        q=mapper.quality_evidence(max_points=10000)
        if q is None:return
        # Keep the original global point cap and full scene depth history.
        group_rows=measured_groups(mapper,q)
        q=dict(q);raw_labels=q['label']
        q['label']=np.where(raw_labels==2,3,np.where(raw_labels==3,2,raw_labels))
        pts=q['point']
        samples=[];weights=[];hypothesis_ids=[]
        for group,indices,track in group_rows:
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
            obj.update(measured_group=int(group),support_track=int(track),
                measured_span_m=span.tolist(),template_dimensions_source='fixed original minimums plus measured quantiles; no GT dimensions',
                marked_points=int(len(known)))
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
            obj['raw_geometry_log_likelihood_ratio']=float(log_likelihood_ratio)
            if not fit_update:log_likelihood_ratio=0.
            obj['consumed_geometry_log_likelihood_ratio']=float(log_likelihood_ratio)
            log_odds=np.clip(np.log(prob/(1-prob))+log_likelihood_ratio,-6.,6.)
            obj['shelf_prior_probability']=prob
            prob=float(1/(1+np.exp(-log_odds)))
            obj.update(shelf_probability=prob,fit_errors_m2=errors)
            samples.extend(box_points);weights.extend(np.asarray(box_area)*(1-prob)*object_probability)
            samples.extend(shelf_points);weights.extend(np.asarray(shelf_area)*prob*object_probability)
        if not samples:return
        points=np.asarray(samples);area=np.asarray(weights)
        self.audit['raw_template_points']=len(points)
        self.audit['expected_area_before_history_m2']=float(area.sum())
        unexplained=np.ones(len(points),bool)
        # Project into past actual RGB-D frames. A hypothesis in measured free
        # space or on an already measured surface has no future reward.
        for frame in (mapper.keyframes if history_rejection else ()):
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
            free=explained&(depth[selected]<measured-.10)
            self.audit['removed_by_free_space']+=int(free.sum())
            self.audit['removed_as_measured_or_depth_tolerance']+=int((explained&~free).sum())
            unexplained[ids[selected[explained]]]=False
        self.points=points[unexplained];self.weights=area[unexplained]
        self.hypothesis_ids=np.asarray(hypothesis_ids)[unexplained]
        self.audit['points_after_history']=len(self.points)
        self.audit['expected_area_after_history_m2']=float(self.weights.sum())
        if len(self.points)>3500:
            # Preserve estimated area when deterministically thinning hypotheses.
            old_area=self.weights.sum();ids=np.linspace(0,len(self.points)-1,3500,dtype=int)
            self.points=self.points[ids];self.weights=self.weights[ids];self.hypothesis_ids=self.hypothesis_ids[ids]
            self.weights*=old_area/max(self.weights.sum(),1e-9)

        self.audit['output_points']=len(self.points)
        self.audit['expected_output_area_m2']=float(self.weights.sum())
        self.audit['objects']=self.objects
