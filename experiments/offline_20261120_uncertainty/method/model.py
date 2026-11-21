"""B8 conditional-depth particles; acquired data only, no robot entrypoint.

B6 AB width/yaw/height/parts stay fixed. G/S share geometry and likelihood;
only class priors differ, so their marginal depth distributions may differ.
"""
from pathlib import Path
import copy, hashlib, importlib.util
from types import SimpleNamespace
import numpy as np
import open3d as o3d

BASE=Path('/home/ouc/nso_robot/analyses')
B6=BASE/'dual_template_parts_20261120/method/model.py'
B7=BASE/'dual_template_depth_20261120/method/profile.py'
def _load(name,path,expected):
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
b6=_load('B8_frozen_B6',B6,'92271b81c87988a0b6336691aa5057f3a6dbd7d6e77ba820d8db4f054c7254c9')
b7=_load('B8_frozen_B7',B7,'194fced5f4fbf79f84c9b593c2c908d8470faf55ea21a4fe433688a46170ebca')
REPAIR=b7.repair
DEPTH_GRID_M=np.round(np.arange(.2,1.40001,.1),10)
ALLOCATION_NODES=np.array([0.,.5,1.])
ALLOCATION_MASS=np.array([.25,.5,.25])
SIGMA_CANDIDATES_M=(.01,.015,.025,.04,.06,.09,.14,.22)

def digest(a):return hashlib.sha256(np.ascontiguousarray(a,dtype=np.float64).tobytes()).hexdigest()
def _json(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):_json(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [_json(v) for v in value]
    return value

def uniform_nodes(lower,upper=1.4):
    """Node-Voronoi cell masses for a uniform conditioned interval."""
    lower=max(.2,float(lower));upper=float(upper)
    if lower>=upper:
        return np.array([lower]),np.ones(1),dict(out_of_support=lower>upper,status='out_of_support_boundary_fallback' if lower>upper else 'degenerate_conditioned_boundary',distribution_calibratable=lower<=upper)
    nodes=np.unique(np.r_[lower,DEPTH_GRID_M[(DEPTH_GRID_M>lower)&(DEPTH_GRID_M<=upper)],upper])
    edges=np.r_[lower,(nodes[:-1]+nodes[1:])/2,upper]
    mass=np.diff(edges)/(upper-lower)
    assert np.all(mass>0) and np.isclose(mass.sum(),1.)
    return nodes,mass,dict(out_of_support=False,status='conditioned_uniform_depth',distribution_calibratable=True)

def geometric_particles(obj,cloud,mode='uncertainty'):
    if mode not in ('uncertainty','fixed'):raise ValueError(mode)
    R=np.asarray(obj['rotation'],float);local=np.asarray(cloud)@R
    minimum,maximum=float(local[:,1].min()),float(local[:,1].max());span=maximum-minimum
    ext=obj['geometry_fit_audit']['extent_uncertainty'][1]
    low,high=ext['low_face_coordinate_m'],ext['high_face_coordinate_m']
    invalid=low is not None and high is not None and float(high)<=float(low)
    if invalid:low=high=None
    if low is not None and high is not None:
        depths=np.array([float(high)-float(low)]);mass=np.ones(1)
        status=dict(status='two_observed_depth_faces',out_of_support=not(.2<=depths[0]<=1.4),distribution_calibratable=True)
    elif mode=='fixed':
        depths=np.array([max(.65,span)]);mass=np.ones(1)
        status=dict(status='fixed_depth_with_measured_lower_bound',out_of_support=depths[0]>1.4,distribution_calibratable=depths[0]<=1.4)
    else:depths,mass,status=uniform_nodes(span)
    rows=[]
    for depth,dm in zip(depths,mass):
        if low is not None:allocations=[(0.,1.,float(low))]
        elif high is not None:allocations=[(1.,1.,float(high)-depth)]
        else:allocations=[(float(a),float(am),minimum-a*(depth-span)) for a,am in zip(ALLOCATION_NODES,ALLOCATION_MASS)]
        for a,am,lower in allocations:rows.append(dict(depth_m=float(depth),allocation=a,lower_m=float(lower),upper_m=float(lower+depth),prior_mass=float(dm*am)))
    status.update(mode=mode,observed_local_min_m=minimum,observed_local_max_m=maximum,observed_span_m=span,
        anchor_low_m=low,anchor_high_m=high,inconsistent_anchor_fallback=invalid,
        prior_domain_m=[.2,1.4],conditioning_domain_m=[max(.2,span),1.4],
        depth_axis_world=R[:,1].tolist(),rotation=R.tolist(),width_height_layer_geometry_fixed=True,
        out_of_support_fallback_note='Boundary particle retains object for numerical/policy compatibility; it is not a calibrated certainty.' if status['out_of_support'] and status['status']!='two_observed_depth_faces' else None)
    return rows,status

def primitive_parts(obj,depth,lower,bits):
    """Exactly B6 parts, not a boolean union or new mesh family."""
    dims=np.asarray(obj['dims'],float);R=np.asarray(obj['rotation'],float)
    local_center=np.asarray(obj['center'],float)@R
    width,height=float(dims[0]),float(dims[2]);x=local_center[0]-width/2;y=float(lower)
    box=[(np.array([x,y,0.]),np.array([width,depth,height]))]
    shelf=[]
    for slab in obj['observed_layers_audit']['final_intervals']:
        shelf.append((np.array([x,y,slab['lower_m']]),np.array([width,depth,slab['upper_m']-slab['lower_m']])))
    for dx in (0.,width-.08):
        for dy in (0.,depth-.08):shelf.append((np.array([x+dx,y+dy,0.]),np.array([.08,.08,height])))
    hist=np.array([np.mean((bits&(1<<b))!=0) for b in range(8)])
    az=-np.pi+(int(np.argmax(hist))+.5)*2*np.pi/8
    direction=np.array([np.cos(az),np.sin(az),0.])@R;axis=int(np.argmax(np.abs(direction[:2])))
    origin=np.array([x,y,0.]);board=np.array([width,depth,height]);board[axis]=.08
    if direction[axis]>0:origin[axis]+=[width,depth][axis]-.08
    shelf.append((origin,board))
    return (box,shelf)

def mesh_scene(parts,R):
    mesh=o3d.geometry.TriangleMesh()
    for origin,size in parts:mesh+=o3d.geometry.TriangleMesh.create_box(*size).translate(origin)
    mesh.rotate(np.asarray(R),center=(0,0,0));return mesh,b7._scene(mesh)

def quadrature(parts,R):
    points=[];areas=[]
    for origin,size in parts:
        p,a=b6.ObjectCompletionModel._box(origin,size);points.extend(p);areas.extend(a)
    return np.asarray(points,float)@np.asarray(R).T,np.asarray(areas,float)

def fixed_ray_basis(obj,particles,keyframes):
    """One maximum ROI and valid acquired ray set for every hypothesis/mode."""
    R=np.asarray(obj['rotation'],float);dims=np.asarray(obj['dims'],float);center=np.asarray(obj['center'])@R
    ymin=min(p['lower_m'] for p in particles);ymax=max(p['upper_m'] for p in particles)
    intervals=obj['observed_layers_audit']['final_intervals']
    zmin=min([0.]+[p['lower_m'] for p in intervals]);zmax=max([dims[2]]+[p['upper_m'] for p in intervals])
    roi=o3d.geometry.TriangleMesh.create_box(float(dims[0]),float(ymax-ymin),float(zmax-zmin)).translate((center[0]-dims[0]/2,ymin,zmin)).rotate(R,center=(0,0,0))
    scene=b7._scene(roi);rays=[];depths=[];seen=set();duplicates=0;frames=[];valid_count=0
    for frame in keyframes:
        pose=np.asarray(frame.world_from_camera,float);K=np.asarray(frame.intrinsic,float)
        h,w=frame.depth_m.shape;v,u=np.mgrid[0:h:b7.PIXEL_STRIDE,0:w:b7.PIXEL_STRIDE]
        dep=np.asarray(frame.depth_m)[v,u].ravel();u=u.ravel();v=v.ravel();valid=np.isfinite(dep)&(dep>0)
        dep,u,v=dep[valid],u[valid],v[valid];valid_count+=len(dep)
        direction=np.column_stack([(u-K[0,2])/K[0,0],(v-K[1,2])/K[1,1],np.ones(len(u))])@pose[:3,:3].T
        batch=np.column_stack([np.tile(pose[:3,3],(len(u),1)),direction]).astype(np.float32)
        t=scene.cast_rays(o3d.core.Tensor(batch),nthreads=1)['t_hit'].numpy()
        posekey=hashlib.sha256(pose.tobytes()+K.tobytes()).hexdigest()
        frames.append(hashlib.sha256(pose.tobytes()+K.tobytes()+np.ascontiguousarray(frame.depth_m).tobytes()).hexdigest())
        for i in np.flatnonzero(np.isfinite(t)&(t>=0)):
            key=(posekey,int(u[i]),int(v[i]),float(dep[i]))
            if key in seen:duplicates+=1;continue
            seen.add(key);rays.append(batch[i]);depths.append(float(dep[i]))
    rays=np.asarray(rays,np.float32).reshape(-1,6);depths=np.asarray(depths,float)
    return rays,depths,dict(ray_count=len(rays),rays_sha256=digest(rays),depths_sha256=digest(depths),keyframe_fingerprints=frames,
        duplicate_rays_removed=duplicates,all_stride_valid_rays=valid_count,local_roi_bounds_m=[[center[0]-dims[0]/2,ymin,zmin],[center[0]+dims[0]/2,ymax,zmax]],
        camera_z_one_parameterization=True,pixel_stride=b7.PIXEL_STRIDE,denominator_shared_by_fixed_and_uncertainty=True)

def joint_posterior(energy,prior_mass,shape_prior,sigma_m,use_likelihood=True):
    energy=np.asarray(energy,float);prior=np.asarray(prior_mass,float);shape=np.asarray(shape_prior,float)
    if energy.shape!=(len(prior),2) or shape.shape!=(2,):raise ValueError('K by 2 energy and two shape priors required')
    if not np.all(np.isfinite(energy)):raise ValueError('finite observation energies required')
    if not (sigma_m>0 and np.isfinite(sigma_m)):raise ValueError('positive frozen sigma required')
    if np.any(prior<0) or np.any(shape<0) or not np.isclose(prior.sum(),1) or not np.isclose(shape.sum(),1):raise ValueError('normalized nonnegative priors required')
    with np.errstate(divide='ignore'):logw=np.log(prior[:,None])+np.log(shape[None,:])
    if use_likelihood:logw=logw-energy/(2*float(sigma_m)**2)
    logw-=np.max(logw);mass=np.exp(logw);mass/=mass.sum();return mass

class ParticleBank:
    def __init__(self,mapper,base_model):
        self.mapper=mapper;self.base_model=base_model;self.objects=[];self._materialized={}
        rows=list(mapper.quality.items())
        if not rows:self.unmodeled_tracks=[];return
        keys=np.asarray([k for k,_ in rows],np.int64).reshape(-1,3)
        tracks=mapper.instance_support_index.memberships_for_keys(keys)
        q={key:np.asarray([row[key] for _,row in rows]) for key in rows[0][1]}
        modeled=set()
        for source in base_model.objects:
            obj=copy.deepcopy(source);track=int(obj['support_track']);selected=tracks==track
            cloud=q['point'][selected];bits=q['bits'][selected];labels=q['label'][selected]
            assert digest(cloud)==obj['fit_input_points_sha256']
            variants={};all_particles=[]
            for mode in ('uncertainty','fixed'):
                particles,status=geometric_particles(obj,cloud,mode);variants[mode]=dict(particles=particles,status=status);all_particles+=particles
            rays,depths,ray_audit=fixed_ray_basis(obj,all_particles,mapper.keyframes)
            geometry={}
            for p in all_particles:
                key=(p['depth_m'],p['lower_m'])
                if key in geometry:continue
                parts=primitive_parts(obj,*key,bits);shapes=[]
                for part in parts:
                    mesh,scene=mesh_scene(part,obj['rotation']);es,ef,evidence=b7.measured_terms(scene,cloud,rays,depths,mapper.keyframes)
                    shapes.append(dict(parts=part,mesh=mesh,scene=scene,surface_energy_m2=es,free_energy_m2=ef,evidence=evidence))
                geometry[key]=shapes
            known=labels[(labels==2)|(labels==3)]
            shelf_prior=float(.1+.8*np.mean(known==2)) if len(known) else .5
            for mode,v in variants.items():
                v['surface_energy_m2']=np.asarray([[s['surface_energy_m2'] for s in geometry[(p['depth_m'],p['lower_m'])]] for p in v['particles']])
                v['free_energy_m2']=np.asarray([[s['free_energy_m2'] for s in geometry[(p['depth_m'],p['lower_m'])]] for p in v['particles']])
                v['energy_m2']=v['surface_energy_m2']+v['free_energy_m2']
                v['prior_mass']=np.asarray([p['prior_mass'] for p in v['particles']])
            self.objects.append(dict(base=obj,support_track=track,variants=variants,geometry=geometry,bits=bits,
                ray_audit=ray_audit,shelf_prior_probability=shelf_prior));modeled.add(track)
        self.unmodeled_tracks=[dict(support_track=int(t),observed_points=int((tracks==t).sum()),status='no_B6_conditional_template',
            fallback_depth_axis_world=[0.,1.,0.],axis_is_fallback=True) for t in sorted(set(tracks)-{0}-modeled)]

    def distribution(self,sigma_m=None,semantic=False,fine_categories=True,use_likelihood=True,extent_mode='uncertainty'):
        if extent_mode not in ('uncertainty','fixed'):raise ValueError('use uncertainty posterior to derive mean-collapse')
        outputs=[]
        for item in self.objects:
            v=item['variants'][extent_mode];sp=item['shelf_prior_probability'] if semantic and fine_categories else .5
            result=dict(support_track=item['support_track'],status=v['status'],particles=v['particles'],prior_mass=v['prior_mass'],
                energy_m2=v['energy_m2'],surface_energy_m2=v['surface_energy_m2'],free_energy_m2=v['free_energy_m2'],shape_prior=[1-sp,sp],
                fit_input_points_sha256=item['base']['fit_input_points_sha256'],fit_input_normals_sha256=item['base']['fit_input_normals_sha256'],
                fit_input_labels_sha256=item['base']['fit_input_labels_sha256'],rotation=item['base']['rotation'],center=item['base']['center'],
                depth_axis_world=np.asarray(item['base']['rotation'])[:,1],fixed_ray_audit=item['ray_audit'],conditional_geometry=item['base'])
            if sigma_m is not None:
                w=joint_posterior(v['energy_m2'],v['prior_mass'],[1-sp,sp],sigma_m,use_likelihood)
                ds=np.asarray([p['depth_m'] for p in v['particles']]);unique=np.unique(ds)
                result.update(joint_probability_mass=w,depth_grid_m=unique,probability_mass=[float(w[ds==d].sum()) for d in unique],
                    shape_probability_mass=w.sum(axis=0),sigma_m=float(sigma_m),likelihood_applied=bool(use_likelihood))
            outputs.append(result)
        return _json(dict(schema='B8_joint_depth_particle_distribution_v1',status='completed_conditional_templates' if outputs else 'no_conditional_templates',tracks=outputs,unmodeled_tracks=self.unmodeled_tracks,
            observed_input_sha256=getattr(self,'observed_input_sha256',None),
            old_surface_likelihood_applied_again=False,source_geometry_consumed=False,semantic=bool(semantic),extent_mode=extent_mode,
            probability_interpretation='conditional generalized joint posterior; calibration limited to declared synthetic geometry split'))

    def materialize(self,extent_mode='uncertainty'):
        if extent_mode in self._materialized:return self._materialized[extent_mode]
        points=[];areas=[];hypids=[];objectids=[];particleids=[];shapeids=[];scenes=[];slices=[]
        for oi,item in enumerate(self.objects):
            v=item['variants'][extent_mode]
            for ki,p in enumerate(v['particles']):
                for hi,shape in enumerate(item['geometry'][(p['depth_m'],p['lower_m'])]):
                    xyz,a=quadrature(shape['parts'],item['base']['rotation']);start=len(points);points.extend(xyz);areas.extend(a)
                    hypids.extend([len(scenes)]*len(a));objectids.extend([oi]*len(a));particleids.extend([ki]*len(a));shapeids.extend([hi]*len(a))
                    scenes.append(shape['scene']);slices.append((oi,ki,hi,start,len(points)))
        xyz=np.asarray(points,float).reshape(-1,3)
        history=REPAIR.classify_history(xyz,self.mapper.keyframes) if len(xyz) else dict(strict_free=np.zeros(0,bool),measured_surface=np.zeros(0,bool))
        data=dict(points=xyz,areas=np.asarray(areas,float),hypothesis_ids=np.asarray(hypids,int),object_ids=np.asarray(objectids,int),
            particle_ids=np.asarray(particleids,int),shape_ids=np.asarray(shapeids,int),hypothesis_rays=scenes,slices=slices,history=history)
        self._materialized[extent_mode]=data;return data

def observed_input_signature(mapper):
    h=hashlib.sha256();rows=list(mapper.quality.items());h.update(np.asarray(mapper.shape,np.int64).tobytes())
    if rows:
        keys=np.asarray([k for k,_ in rows],np.int64).reshape(-1,3);h.update(keys.tobytes())
        h.update(np.asarray(mapper.instance_support_index.memberships_for_keys(keys),np.int64).tobytes())
        for name in ('point','normal','label','bits'):
            h.update(np.ascontiguousarray([row[name] for _,row in rows]).tobytes())
    for f in mapper.keyframes:
        for a in (f.world_from_camera,f.intrinsic,f.depth_m):h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()

def build_particle_bank(mapper,neutral_model=None):
    signature=observed_input_signature(mapper)
    cached=getattr(mapper,'_b8_shared_particle_bank',None)
    if neutral_model is None and cached is not None and cached[0]==signature:return cached[1]
    if neutral_model is None:neutral_model=b6.configure(outer_orientation=True,observed_layers=True)(mapper,False)
    for key in ('geometry_fit','outer_orientation','observed_layers','full_observed_support'):
        if not neutral_model.audit.get(key):raise ValueError('B6 AB full-observation conditional geometry required')
    bank=ParticleBank(mapper,neutral_model);bank.observed_input_sha256=signature
    mapper._b8_shared_particle_bank=(signature,bank)
    return bank

class FullTemplateModel:
    SIGMA_M=None
    EXTENT_MODE='uncertainty'
    def __init__(self,mapper,semantic,fine_categories=True,*,sigma_m=None,extent_mode=None,bank=None,fit_update=True,history_rejection=True,measured_surface_consumption=True):
        sigma=self.SIGMA_M if sigma_m is None else sigma_m
        if sigma is None:raise ValueError('Frozen independently calibrated sigma_m is required before policy binding')
        mode=self.EXTENT_MODE if extent_mode is None else extent_mode
        if mode not in ('uncertainty','fixed','mean_collapse'):raise ValueError(mode)
        self.mapper=mapper;self.bank=bank or build_particle_bank(mapper);self.semantic=semantic
        posterior_mode='uncertainty' if mode=='mean_collapse' else mode
        posterior=self.bank.distribution(sigma,semantic,fine_categories,fit_update,posterior_mode)
        if mode=='mean_collapse':data,masses=self._collapse(posterior)
        else:
            data=self.bank.materialize(mode);masses=np.empty(len(data['points']))
            for oi,ki,hi,start,end in data['slices']:masses[start:end]=posterior['tracks'][oi]['joint_probability_mass'][ki][hi]
        self.points=data['points'];self.hypothesis_ids=data['hypothesis_ids'];self.hypothesis_rays=data['hypothesis_rays']
        self.object_ids=data['object_ids'];self.particle_ids=data['particle_ids'];self.base_weights=data['areas']*masses
        keep=np.ones(len(self.points),bool)
        if history_rejection:
            keep&=~data['history']['strict_free']
            if measured_surface_consumption:keep&=~data['history']['measured_surface']
        self.history_keep=keep;self.weights=self.base_weights*keep
        self.objects=posterior['tracks']
        self.audit=dict(schema='B8_policy_joint_particles_v1',extent_mode=mode,sigma_m=sigma,history_rejection=history_rejection,
            measured_surface_consumption=measured_surface_consumption,fit_update=fit_update,old_surface_likelihood_reapplied=False,
            common_G_S_particle_basis=mode!='mean_collapse',marginal_depth_may_differ_with_class_prior=True,
            raw_template_points=len(self.points),applied_global_point_cap=None,expected_area_before_history_m2=float(self.base_weights.sum()),
            expected_area_after_history_m2=float(self.weights.sum()),objects=self.objects,unmodeled_tracks=self.bank.unmodeled_tracks,
            candidate_weights_stops_or_navigation_changed=False,source_geometry_consumed=False)

    def _collapse(self,posterior):
        points=[];areas=[];ids=[];scenes=[];oi_all=[];ki_all=[];masses=[]
        for oi,(item,row) in enumerate(zip(self.bank.objects,posterior['tracks'])):
            w=np.asarray(row['joint_probability_mass']);particles=row['particles'];ds=np.asarray([p['depth_m'] for p in particles]);lo=np.asarray([p['lower_m'] for p in particles])
            for hi in (0,1):
                mass=float(w[:,hi].sum())
                # Priors are strictly positive; underflow can leave zero mass.
                if mass>0:depth=float(np.dot(w[:,hi],ds)/mass);lower=float(np.dot(w[:,hi],lo)/mass)
                else:depth=float(np.dot(row['prior_mass'],ds));lower=float(np.dot(row['prior_mass'],lo))
                parts=primitive_parts(item['base'],depth,lower,item['bits'])[hi]
                _,scene=mesh_scene(parts,item['base']['rotation']);xyz,a=quadrature(parts,item['base']['rotation'])
                points.extend(xyz);areas.extend(a);ids.extend([len(scenes)]*len(a));scenes.append(scene)
                oi_all.extend([oi]*len(a));ki_all.extend([hi]*len(a));masses.extend([mass]*len(a))
                row.setdefault('mean_collapse',[]).append(dict(shape_id=hi,depth_m=depth,lower_m=lower,retained_shape_mass=mass,likelihood_recomputed=False))
        xyz=np.asarray(points,float).reshape(-1,3)
        history=REPAIR.classify_history(xyz,self.mapper.keyframes) if len(xyz) else dict(strict_free=np.zeros(0,bool),measured_surface=np.zeros(0,bool))
        return dict(points=xyz,areas=np.asarray(areas),hypothesis_ids=np.asarray(ids),hypothesis_rays=scenes,
            object_ids=np.asarray(oi_all),particle_ids=np.asarray(ki_all),history=history),np.asarray(masses)

    def gain(self,cell,heading,ray):
        # Inherited B6 optimization: retain the complete external quadrature,
        # query only nonzero-weight samples, then scatter into the same basis.
        active=np.flatnonzero(self.weights>0)
        view=SimpleNamespace(mapper=self.mapper,points=self.points[active],weights=self.weights[active],
            hypothesis_ids=self.hypothesis_ids[active],hypothesis_rays=self.hypothesis_rays,audit=self.audit)
        values,audit=REPAIR.trace_gain(view,cell,heading,ray,repair_enabled=True)
        if not hasattr(self,'_occlusion_repair_summary'):self._occlusion_repair_summary={}
        REPAIR._aggregate(self._occlusion_repair_summary,audit)
        result=np.zeros(len(self.points));result[active]=values;return result

def configure(*,sigma_m=None,extent_mode='uncertainty'):
    class Configured(FullTemplateModel):
        SIGMA_M=sigma_m
        EXTENT_MODE=extent_mode
    return Configured
