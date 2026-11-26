"""Shared observed-object support and measured outer-plane geometry revision.

G and S share acquired object tracks. G uses a neutral class prior; it is not
an entirely semantics-free baseline. No source or evaluation geometry is read.
"""
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import open3d as o3d
from scipy.ndimage import binary_closing,label
from nso.semantic_completion_v3 import ObjectCompletionModel

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

_GEOMETRY_PATH = Path(__file__).with_name('geometry.py')
_geometry = _load('B6_geometry_helpers', _GEOMETRY_PATH)
_REPAIR_PATH = Path('/home/ouc/nso_robot/analyses/dual_template_repair_20261119/method/occlusion_repair.py')
_REPAIR_SHA = '57c5cd1bb58a2e816d81184014cbfbad2a3d5ddeb872c43c7b5d9895378bb2fb'
assert hashlib.sha256(_REPAIR_PATH.read_bytes()).hexdigest() == _REPAIR_SHA
_repair = _load('B6_unchanged_B2_gain', _REPAIR_PATH)

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

def absolute_applicability(geometry_probability,free_fractions,conditional_probability):
    """Retain the v1 clipped conditional shape distribution, add valid mass."""
    likelihood=np.exp(-np.asarray(free_fractions,float)*.15**2/(2*.06**2))
    geometry_mass=np.array([1-geometry_probability,geometry_probability])
    valid_mass=float(np.dot(geometry_mass,likelihood))
    mass=valid_mass*np.array([1-conditional_probability,conditional_probability])
    return mass,likelihood,valid_mass


def pool_observed_instances(groups,shape):
    """Union actually associated track>0 components; unknown track0 is intact."""
    stride=int(np.prod(shape))+1
    result=[(int(g),np.asarray(ids,dtype=int),int(t)) for g,ids,t in groups if t==0]
    for track in sorted({int(t) for _,_,t in groups if t>0}):
        parts=[np.asarray(ids,dtype=int) for _,ids,t in groups if t==track]
        ids=np.unique(np.concatenate(parts))
        result.append((track*stride,ids,track))
    return result


class FullTemplateModel(ObjectCompletionModel):
    OUTER_ORIENTATION = False
    OBSERVED_LAYERS = False
    OBJECT_SUPPORT = True
    FULL_OBSERVED_SUPPORT = False
    GEOMETRY_FIT = False
    FREE_SPACE_LIKELIHOOD = False
    INSTANCE_POOLING = True
    APPLICABILITY = False
    def __init__(self,mapper,semantic,fine_categories=True,*,fit_update=True,history_rejection=True,geometry_fit=None,free_space_likelihood=None,measured_surface_consumption=True,instance_pooling=None,applicability=None,object_support=None,full_observed_support=None,outer_orientation=None,observed_layers=None):
        outer_orientation=self.OUTER_ORIENTATION if outer_orientation is None else bool(outer_orientation)
        observed_layers=self.OBSERVED_LAYERS if observed_layers is None else bool(observed_layers)
        object_support=self.OBJECT_SUPPORT if object_support is None else bool(object_support)
        full_observed_support=self.FULL_OBSERVED_SUPPORT if full_observed_support is None else bool(full_observed_support)
        geometry_fit = self.GEOMETRY_FIT if geometry_fit is None else bool(geometry_fit)
        free_space_likelihood = self.FREE_SPACE_LIKELIHOOD if free_space_likelihood is None else bool(free_space_likelihood)
        instance_pooling = self.INSTANCE_POOLING if instance_pooling is None else bool(instance_pooling)
        applicability = self.APPLICABILITY if applicability is None else bool(applicability)
        if (outer_orientation or observed_layers) and not (object_support and full_observed_support and geometry_fit):
            raise ValueError('B6 revisions require shared full observed support and geometry fitting')
        if object_support and (free_space_likelihood or applicability):
            raise ValueError('declared object-fit stages do not enable free likelihood or Z')
        if applicability and not free_space_likelihood:
            raise ValueError('absolute applicability requires the corresponding free-space likelihood')
        self.mapper=mapper;self.semantic=semantic;self.objects=[]
        self.base_weights=np.empty(0);self.history_keep=np.empty(0,dtype=bool)
        self.points=np.empty((0,3));self.weights=np.empty(0)
        self.hypothesis_ids=np.empty(0,dtype=int);self.hypothesis_rays=[]
        self.audit=dict(schema='B6_observed_outer_orientation_absolute_layers_v1',outer_orientation=outer_orientation,observed_layers=observed_layers,object_support=object_support,full_observed_support=full_observed_support,excluded_untracked_template_groups=[],shared_object_support_baseline='G uses acquired object tracks with neutral class prior',geometry_fit=geometry_fit,free_space_likelihood=free_space_likelihood,instance_pooling=instance_pooling,applicability=applicability,skipped_pooled_groups=[],fit_update=bool(fit_update),
            history_rejection=bool(history_rejection),measured_surface_consumption=bool(measured_surface_consumption),prior_rebuilt_from_current_measurements=True,
            original_candidates_method_used=False,history_frames=len(mapper.keyframes),
            raw_template_points=0,points_after_history=0,output_points=0,
            removed_by_free_space=0,removed_as_measured_or_depth_tolerance=0,
            original_global_point_cap=3500,applied_global_point_cap=None,shared_full_quadrature=True,history_as_zero_weight=True,hidden_truth_input=False,
            current_label_dictionary={2:'open',3:'closed'},template_label_dictionary={2:'box',3:'shelf'},
            label_dictionary_conversion='2open->3shelf;3closed->2box at template prior entrance only',
            probability_interpretation='heuristic mean-distance Gibbs pseudo-posterior, not calibrated probability',
            applicability_interpretation='heuristic absolute free-consistency attenuation; residual is bookkeeping, not a fitted third-class probability')
        q=mapper.quality_evidence(max_points=10000)
        if q is None:return
        cropped_q=q
        raw_groups=measured_groups(mapper,cropped_q)
        if full_observed_support:
            if not object_support or not hasattr(mapper,'instance_support_index'):
                raise ValueError('full observed support requires acquired object tracks')
            rows=list(mapper.quality.items())
            q={key:np.asarray([row[key] for _,row in rows]) for key in rows[0][1]}
            keys=np.asarray([key for key,_ in rows],np.int64).reshape(-1,3)
            tracks=mapper.instance_support_index.memberships_for_keys(keys)
            stride=int(np.prod(mapper.shape))+1
            group_rows=[(int(t)*stride,np.flatnonzero(tracks==t),int(t)) for t in sorted(set(tracks)-{0})]
        else:
            group_rows=pool_observed_instances(raw_groups,mapper.shape) if instance_pooling else raw_groups
        if object_support:
            self.audit['excluded_untracked_template_groups']=[int(g) for g,_,t in raw_groups if t==0]
            group_rows=[(g,ids,t) for g,ids,t in group_rows if t>0]
        self.audit['observed_input_point_count']=len(q['point'])
        self.audit['support_source']='full_acquired_quality' if full_observed_support else 'legacy_height_crop'

        # Keep original measured grouping and full scene depth history; remove only the template point cap.
        source_components={group:[int(g) for g,_,t in raw_groups if (t==track if instance_pooling and track>0 else g==group)] for group,_,track in group_rows}
        self.audit['raw_component_count']=len(raw_groups)
        self.audit['model_group_count_before_size_eligibility']=len(group_rows)
        q=dict(q);raw_labels=q['label']
        q['label']=np.where(raw_labels==2,3,np.where(raw_labels==3,2,raw_labels))
        pts=q['point']
        samples=[];weights=[];hypothesis_ids=[]
        for group,indices,track in group_rows:
            if len(indices)<8:continue
            cloud=pts[indices];low,high=np.quantile(cloud,[.05,.95],axis=0);span=high-low
            if max(span[:2])>2.2 or span[2]<.25:
                if instance_pooling and track>0:self.audit['skipped_pooled_groups'].append(dict(track=int(track),span=span.tolist(),reason='inherited_template_size_eligibility'))
                continue
            dims=np.array([max(.9,span[0]),max(.65,span[1]),np.clip(high[2]+.08,.9,1.5)])
            prior_layer_height=float(dims[2])
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
            rotation=np.eye(3);geometry_audit=None
            if geometry_fit:
                views,view_counts=_geometry.observed_view_directions(cloud,mapper.keyframes)
                fitted=_geometry.fit_envelope(cloud,q['normal'][indices],np.array([.9,.65,dims[2]]),views,outer_orientation=outer_orientation)
                center,dims,rotation=fitted['center'],fitted['dims'],fitted['rotation']
                geometry_audit=fitted['audit']
                direction=(np.r_[direction,0.]@rotation)[:2]
            labels=q['label'][indices];known=labels[(labels==2)|(labels==3)]
            prob=.5
            object_probability=1.
            if semantic and fine_categories and len(known):
                vote=float(np.mean(known==3))
                prob=.1+.8*vote
            if not object_support and semantic and np.any(labels>0):object_probability=float(np.mean(labels[labels>0]>1))
            obj=dict(center=center,dims=dims,shelf_probability=prob,object_probability=object_probability,observed_points=len(indices))
            obj.update(measured_group=int(group),support_track=int(track),
                source_component_ids=source_components[group],source_component_count=len(source_components[group]),
                measured_span_m=span.tolist(),template_dimensions_source=('measured oriented envelope with unresolved prior extents' if geometry_fit else 'fixed original minimums plus measured quantiles; no GT dimensions'),
                marked_points=int(len(known)))
            obj['fit_input_points_sha256']=hashlib.sha256(np.ascontiguousarray(cloud,dtype=np.float64).tobytes()).hexdigest()
            obj['fit_input_normals_sha256']=hashlib.sha256(np.ascontiguousarray(q['normal'][indices],dtype=np.float64).tobytes()).hexdigest()
            obj['fit_input_labels_sha256']=hashlib.sha256(np.ascontiguousarray(raw_labels[indices],dtype=np.int64).tobytes()).hexdigest()
            obj['support_source']=self.audit['support_source']
            obj['support_hash_convention']='row order preserved; xyz/normal float64 C bytes, original labels int64 C bytes'
            obj['outer_orientation']=outer_orientation;obj['observed_layers']=observed_layers
            obj['prior_absolute_layer_reference_height_m']=prior_layer_height
            obj['dimensions_interpretation']='box and leg dimensions; unresolved absolute layer priors may extend outside this estimate'
            obj['geometry_fit_audit']=geometry_audit
            obj['rotation']=rotation
            self.objects.append(obj)
            box_points,box_area=self._box(center-dims/2,dims)
            box_mesh=o3d.geometry.TriangleMesh.create_box(*dims).translate(center-dims/2)
            shelf_mesh=o3d.geometry.TriangleMesh()
            def shelf_part(origin,size):
                shelf_mesh.__iadd__(o3d.geometry.TriangleMesh.create_box(*size).translate(origin))
                return self._box(origin,size)
            shelf_points=[];shelf_area=[]
            x,y,z=center-dims/2;sx,sy,h=dims
            if observed_layers:
                intervals,layer_audit=_geometry.fit_horizontal_layers(cloud,q['normal'][indices],views,prior_layer_height,h)
                obj['observed_layers_audit']=layer_audit
                for layer in intervals:
                    p,a=shelf_part([x,y,layer['lower_m']],[sx,sy,layer['upper_m']-layer['lower_m']]);shelf_points.extend(p);shelf_area.extend(a)
            else:
                obj['observed_layers_audit']=None
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
            if geometry_fit:
                # All construction takes place in an axis-aligned local frame
                # centered on the estimated world position, then rotates once.
                box_points=(np.asarray(box_points)-center)@rotation.T+center
                shelf_points=(np.asarray(shelf_points)-center)@rotation.T+center
                box_mesh.rotate(rotation,center=center);shelf_mesh.rotate(rotation,center=center)
            free_fractions=[0.,0.];free_audits=[]
            if free_space_likelihood:
                for points_h,area_h in ((box_points,box_area),(shelf_points,shelf_area)):
                    fraction,audit_h=_geometry.contradiction_fraction(points_h,area_h,mapper.keyframes,_repair.classify_history)
                    free_fractions[len(free_audits)]=fraction;free_audits.append(audit_h)
            obj['free_space_contradiction_fractions']=free_fractions
            obj['free_space_likelihood_audit']=free_audits
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
            # Use the inherited bounded discrepancy and Gibbs scale. This is
            # a declared pseudo-likelihood, not calibrated probability.
            free_llr=(free_fractions[0]-free_fractions[1])*.15**2/(2*.06**2)
            obj['free_space_log_likelihood_ratio']=float(free_llr)
            geometry_log_odds=np.clip(np.log(prob/(1-prob))+log_likelihood_ratio,-6.,6.)
            geometry_only_probability=float(1/(1+np.exp(-geometry_log_odds)))
            obj['geometry_only_shelf_probability']=geometry_only_probability
            log_likelihood_ratio+=free_llr
            log_odds=np.clip(np.log(prob/(1-prob))+log_likelihood_ratio,-6.,6.)
            obj['shelf_prior_probability']=prob
            prob=float(1/(1+np.exp(-log_odds)))
            obj.update(shelf_probability=prob,fit_errors_m2=errors)
            # Conditional shape confidence and absolute model applicability
            # are different quantities. Do not normalize away evidence that
            # both template alternatives conflict with measured free space.
            hypothesis_mass=np.array([1-prob,prob],float)
            valid_mass=1.
            if applicability:
                hypothesis_mass,free_likelihood,valid_mass=absolute_applicability(
                    geometry_only_probability,free_fractions,prob)
                # Relative free evidence has already set the conditional shape
                # distribution above. Only its evidence marginal now scales
                # valid-template mass; never multiply the relative term twice.
                obj['absolute_free_likelihood']=free_likelihood.tolist()
            obj['hypothesis_probability_mass']=hypothesis_mass.tolist()
            obj['template_valid_probability_mass']=valid_mass
            obj['unmodeled_probability_mass']=1-valid_mass
            samples.extend(box_points);weights.extend(np.asarray(box_area)*hypothesis_mass[0]*object_probability)
            samples.extend(shelf_points);weights.extend(np.asarray(shelf_area)*hypothesis_mass[1]*object_probability)
        if not samples:return
        points=np.asarray(samples);area=np.asarray(weights)
        self.audit['raw_template_points']=len(points)
        self.audit['expected_area_before_history_m2']=float(area.sum())
        unexplained=np.ones(len(points),bool)
        # Same historical projection/tolerance as B2, resolving each sample
        # once. Invalid depths never establish free or measured evidence.
        if history_rejection:
            classified=_repair.classify_history(points,mapper.keyframes)
            free=classified['strict_free']
            measured=classified['measured_surface']
            unexplained &= ~free
            if measured_surface_consumption:unexplained &= ~measured
            self.audit['removed_by_free_space']=int(free.sum())
            self.audit['removed_as_measured_or_depth_tolerance']=int(measured.sum()) if measured_surface_consumption else 0
            self.audit['measured_vetoed_conflict_count']=int((classified['raw_free_evidence']&measured).sum())
        self.points=points
        self.base_weights=area.copy()
        self.history_keep=unexplained
        self.weights=area*unexplained
        self.hypothesis_ids=np.asarray(hypothesis_ids)
        self.audit['points_after_history']=int(unexplained.sum())
        self.audit['expected_area_after_history_m2']=float(self.weights.sum())
        self.audit['output_points']=len(self.points)
        self.audit['output_active_points']=int(np.count_nonzero(self.weights))
        self.audit['expected_output_area_m2']=float(self.weights.sum())
        self.audit['objects']=self.objects

    def gain(self, cell, heading, ray):
        # Preserve the full shared integral basis externally. Zero-weight points
        # cannot affect the B2 ray calculation and need no expensive ray query.
        active=np.flatnonzero(self.weights>0)
        view=SimpleNamespace(mapper=self.mapper,points=self.points[active],weights=self.weights[active],
                             hypothesis_ids=self.hypothesis_ids[active],hypothesis_rays=self.hypothesis_rays,audit=self.audit)
        values,audit=_repair.trace_gain(view,cell,heading,ray,repair_enabled=True)
        if not hasattr(self,'_occlusion_repair_summary'):self._occlusion_repair_summary={}
        _repair._aggregate(self._occlusion_repair_summary,audit)
        output=np.zeros(len(self.points));output[active]=values
        return output


def configure(*,object_support=True,full_observed_support=True,geometry_fit=True,outer_orientation=False,observed_layers=False,free_space_likelihood=False,instance_pooling=True,applicability=False):
    """Fresh class with fixed staged switches; bind into unchanged B adapter."""
    class ConfiguredFullTemplateModel(FullTemplateModel):
        OUTER_ORIENTATION=bool(outer_orientation)
        OBSERVED_LAYERS=bool(observed_layers)
        OBJECT_SUPPORT=bool(object_support)
        FULL_OBSERVED_SUPPORT=bool(full_observed_support)
        GEOMETRY_FIT=bool(geometry_fit)
        FREE_SPACE_LIKELIHOOD=bool(free_space_likelihood)
        INSTANCE_POOLING=bool(instance_pooling)
        APPLICABILITY=bool(applicability)
    return ConfiguredFullTemplateModel


def diagnostics(model):
    return _repair.diagnostics(model)


def binding(*,object_support=True,full_observed_support=True,geometry_fit=True,outer_orientation=False,observed_layers=False,**unused):
    return dict(schema='B6_declared_observed_outer_and_layers_v1',outer_orientation=bool(outer_orientation),observed_layers=bool(observed_layers),object_support=bool(object_support),
                full_observed_support=bool(full_observed_support),geometry_fit=bool(geometry_fit),
                instance_pooling=True,free_space_likelihood=False,applicability=False,
                global_point_cap=None,shared_object_support='acquired track>0; G has neutral class prior',
                history_uses_zero_weights_on_common_point_basis=True,
                B2_gain_sha256=_REPAIR_SHA,
                geometry_sha256=hashlib.sha256(_GEOMETRY_PATH.read_bytes()).hexdigest(),
                model_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                policy_weights_candidates_budget_stops_changed=False,ground_truth_consumed=False)
