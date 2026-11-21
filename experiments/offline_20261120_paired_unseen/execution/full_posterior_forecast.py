"""Paired B11 prediction on the unchanged B8 full posterior surface basis.

This actor-side module accepts only an acquired mapper, its full acquired RGB-D
sequence and the G/S models. It never accepts a world/source reference. C uses
independent volumes reconstructed from all past frames plus hypothesis-rendered
future frames. Caches remove exactly equal work, never positive probability.
"""
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
import hashlib, importlib.util, sys, time
import numpy as np
import open3d as o3d
BASE=Path('/home/ouc/nso_robot/analyses')
B10=BASE/'dual_template_gain_alignment_20261120'

def _load(name,path,expected):
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,str(path)
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
CORE=_load('B11_frozen_acquisition_core',B10/'geometry_audit/forecast_development/acquisition_forecast.py','0a912e44f474d6a97c62c674f822e3b9b216bddc30e4b29c91ac07a3c646d259')
# The fitted renderer has an inclusive .15 m lower gate, matching the inherited
# TexturedRayRenderer. Its own B2 dependency has an immutable hash assertion.
RENDER_PATH=B10/'execution_audit/acquisition_forecast/render_observed_free.py'
assert hashlib.sha256(RENDER_PATH.read_bytes()).hexdigest()=='385a6a16ee0b9d1e7970df02282170945bc92e0fcbabd72bd48479ae9531dc79'
spec=importlib.util.spec_from_file_location('B11_frozen_observed_free_renderer',RENDER_PATH)
RENDER=importlib.util.module_from_spec(spec);sys.modules[spec.name]=RENDER;spec.loader.exec_module(RENDER)

def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def _insert(cache,key,value,limit):
    cache[key]=value;cache.move_to_end(key)
    while len(cache)>limit:cache.popitem(last=False)
    return value

class _MemoizedNativeRayScene:
    """Exact native t_hit cache; unchanged renderer consumes the same rays."""
    def __init__(self,scene,counters,limit=2048):
        self.scene=scene;self.counters=counters;self.limit=limit;self.cache=OrderedDict()
    def cast_rays(self,rays,nthreads=1):
        key=digest(rays.numpy())
        if key in self.cache:
            self.counters['native_ray_cache_hits']+=1;self.cache.move_to_end(key)
            return {'t_hit':self.cache[key]}
        result=self.scene.cast_rays(rays,nthreads=nthreads)['t_hit']
        self.counters['native_ray_casts']+=1
        return {'t_hit':_insert(self.cache,key,result,self.limit)}

class FullPosteriorForecast:
    """One immutable observed prefix and one shared, uncollapsed particle bank."""
    def __init__(self,mapper,models,acquired_frames,*,depth_cache_limit=8192,after_cache_limit=256,backend='fork'):
        start=time.monotonic()
        if set(models)!={'G','S'}:raise ValueError('both paired priors required')
        self.mapper=mapper;self.models=models;self.bank=models['G'].bank
        if backend not in ('replay','fork'):raise ValueError('exact replay or process-snapshot backend required')
        self.backend=backend;self.snapshot=None
        if models['S'].bank is not self.bank:raise ValueError('shared bank required')
        self.data=self.bank.materialize('uncertainty')
        self.depth_cache_limit=int(depth_cache_limit);self.after_cache_limit=int(after_cache_limit)
        if min(self.depth_cache_limit,self.after_cache_limit)<1:raise ValueError('positive LRU capacity')
        for m in models.values():
            if m.audit['extent_mode']!='uncertainty':raise ValueError('no point-estimate geometry permitted')
            np.testing.assert_array_equal(m.points,self.data['points'])
            np.testing.assert_array_equal(m.hypothesis_ids,self.data['hypothesis_ids'])
        np.testing.assert_array_equal(models['G'].history_keep,models['S'].history_keep)
        self.original_model_hashes={name:self._model_hash(m) for name,m in models.items()}
        self.past=tuple(CORE.DepthFrame(f.depth_m,f.world_from_camera,f.intrinsic,'acquired') for f in acquired_frames)
        if len(self.past)!=mapper.frames:raise ValueError('all acquired frames, not sparse keyframes, are required')
        self.frame_hashes=tuple(CORE.frame_hash(f) for f in self.past)
        self.history=tuple(mapper.keyframes)
        self.before=mapper.mesh();self.before_ray=CORE.ray_scene(self.before)
        # An exact replay proves the mapper/native before geometry is the same
        # volume contract. The mesh is used only for distance/raycast; it is
        # never used to initialize or replace a TSDF volume.
        if backend=='fork':
            helper_path=Path(__file__).resolve().parents[1]/'performance/fork_forecast.py'
            helper=_load('B11_exact_fork_snapshot',helper_path,'29b67da6c3214afd70a7f660ae7f2200106f61867bfacf26c4f989d76faf1e02')
            self.snapshot=helper.ForkSnapshot(self.past,voxel_m=mapper.config.voxel_m,truncation_m=.12,max_depth_m=mapper.config.max_depth_m)
            before_replayed=self.snapshot.before
        else:before_replayed=CORE.integrate(self.past,mapper.config.voxel_m,.12,mapper.config.max_depth_m)
        for field in ('vertices','triangles'):np.testing.assert_array_equal(np.asarray(getattr(self.before,field)),np.asarray(getattr(before_replayed,field)))
        self.before_distance=CORE.distance(self.before,self.data['points'])
        self.depth_cache=OrderedDict();self.after_cache=OrderedDict();self.route_cache={};self.backend_fallbacks=[]
        self.visibility_cache={'G':{},'S':{}}
        self.counts=dict(rendered_depth_frames=0,depth_cache_hits=0,after_volume_fusions=0,after_volume_cache_hits=0,hypothesis_route_terms=0,zero_mass_terms_skipped=0,computational_fallbacks=0,native_ray_casts=0,native_ray_cache_hits=0)
        self.measured_renderer_scene=None if self.before_ray is None else _MemoizedNativeRayScene(self.before_ray,self.counts)
        self.init_seconds=time.monotonic()-start

    @staticmethod
    def _model_hash(m):return hashlib.sha256(''.join(digest(a)for a in (m.points,m.base_weights,m.weights,m.history_keep,m.hypothesis_ids)).encode()).hexdigest()

    def _geometry(self,oi,ki,hi):
        item=self.bank.objects[oi];p=item['variants']['uncertainty']['particles'][ki]
        shape=item['geometry'][(p['depth_m'],p['lower_m'])][hi]
        mesh=shape['mesh']
        key=hashlib.sha256((digest(np.asarray(mesh.vertices))+digest(np.asarray(mesh.triangles))).encode()).hexdigest()
        return key,shape

    def _future(self,geometry_key,scene,state):
        state=tuple(map(int,state));key=(geometry_key,state)
        if key in self.depth_cache:
            self.counts['depth_cache_hits']+=1;self.depth_cache.move_to_end(key);return self.depth_cache[key]
        from env.virtual3d import camera_pose
        pose=camera_pose(state[:2],state[2],self.mapper.config,self.mapper.shape[0])
        k=self.past[0].intrinsic;c=self.mapper.config
        depth,audit=RENDER.corrected_depth(scene,self.measured_renderer_scene,pose,k,c.width_px,c.height_px,self.history,min_depth_m=.15,max_depth_m=c.max_depth_m)
        f=CORE.DepthFrame(depth,pose,k,'predicted_hypothesis');self.counts['rendered_depth_frames']+=1
        return _insert(self.depth_cache,key,(f,CORE.frame_hash(f),audit),self.depth_cache_limit)

    def _after(self,future):
        key=hashlib.sha256(''.join(x[1]for x in future).encode()).hexdigest()
        if key in self.after_cache:
            self.counts['after_volume_cache_hits']+=1;self.after_cache.move_to_end(key);return self.after_cache[key],key
        # Reconstruct all prior TSDF voxel weights for every distinct future
        # observation sequence. Exact duplicated sequences can share a volume.
        if self.snapshot is not None:
            # Query the shared full point basis once so a byte-identical future
            # sequence can serve different hypotheses without another fusion.
            result=self.snapshot.forecast([x[0]for x in future],self.data['points'],np.ones(len(self.data['points'])),before_distance=self.before_distance)
            value=dict(distance=result['after_distance'],receipt=result['receipt'])
            if result['receipt']['computational_fallback_reason'] is not None:
                self.counts['computational_fallbacks']+=1
                self.backend_fallbacks.append(dict(future_sequence_sha256=key,receipt=result['receipt']))
        else:
            mesh=CORE.integrate([*self.past,*[x[0]for x in future]],self.mapper.config.voxel_m,.12,self.mapper.config.max_depth_m)
            value=CORE.ray_scene(mesh)
        self.counts['after_volume_fusions']+=1
        return _insert(self.after_cache,key,value,self.after_cache_limit),key

    def acquisition(self,states):
        states=tuple(tuple(map(int,s))for s in states)
        if states in self.route_cache:return self.route_cache[states]
        start=time.monotonic();n=len(self.data['points']);gain={a:np.zeros(n)for a in self.models};loss={a:np.zeros(n)for a in self.models};terms=[]
        before_counts=dict(self.counts);tolerance=np.float32(.05)
        for hid,(oi,ki,hi,lo,end) in enumerate(self.data['slices']):
            masses={a:float(m.objects[oi]['joint_probability_mass'][ki][hi])for a,m in self.models.items()}
            if not any(w>0 for w in masses.values()):
                self.counts['zero_mass_terms_skipped']+=1;continue
            key,shape=self._geometry(oi,ki,hi)
            future=[self._future(key,shape['scene'],s)for s in states]
            if future:
                ray,future_key=self._after(future)
                if isinstance(ray,dict):d1=ray['distance'][lo:end]
                else:d1=np.full(end-lo,np.inf) if ray is None else ray.compute_distance(o3d.core.Tensor(np.asarray(self.data['points'][lo:end],np.float32)),nthreads=1).numpy()
            else:d1=self.before_distance[lo:end];future_key='empty_future_same_as_before'
            before=self.before_distance[lo:end]<=tolerance;after=d1<=tolerance
            strict_keep=~self.data['history']['strict_free'][lo:end]
            new=(~before)&after&strict_keep;lost=before&(~after)&strict_keep
            # Preserve exactly the original point positions, areas and joint
            # posterior masses. No thresholding or conditional-mean collapse.
            for arm,m in self.models.items():
                gain[arm][lo:end]=m.base_weights[lo:end]*new
                loss[arm][lo:end]=m.base_weights[lo:end]*lost
            self.counts['hypothesis_route_terms']+=1
            terms.append(dict(hypothesis_id=hid,object_index=oi,particle_id=ki,shape_id=hi,geometry_sha256=key,posterior_mass=masses,future_sequence_sha256=future_key,points=end-lo,own_area_gross_new_m2=float(self.data['areas'][lo:end][new].sum()),own_area_lost_m2=float(self.data['areas'][lo:end][lost].sum())))
        result=dict(gross_m2={a:float(g.sum())for a,g in gain.items()},lost_m2={a:float(l.sum())for a,l in loss.items()},terms=terms,counts={k:self.counts[k]-before_counts[k]for k in self.counts},elapsed_s=time.monotonic()-start)
        result['net_m2']={a:result['gross_m2'][a]-result['lost_m2'][a]for a in self.models}
        self.route_cache[states]=result;return result

    def visibility(self,states):
        result={}
        for arm,m in self.models.items():
            union=np.zeros(len(m.points));cache=self.visibility_cache[arm]
            for raw in states:
                s=tuple(map(int,raw))
                if s not in cache:cache[s]=m.gain(s[:2],s[2],self.before_ray)
                np.maximum(union,cache[s],out=union)
            result[arm]=float(union.sum())
        return result

    def assert_unchanged(self):
        assert self.frame_hashes==tuple(CORE.frame_hash(f)for f in self.past)
        assert self.original_model_hashes=={a:self._model_hash(m)for a,m in self.models.items()}
        if self.snapshot is not None:self.snapshot.verify_parent_unchanged()
        return dict(backend=self.backend,depth_cache_limit=self.depth_cache_limit,after_cache_limit=self.after_cache_limit,**dict(acquired_frames=len(self.past),history_keyframes=len(self.history),full_particle_terms=len(self.data['slices']),original_point_count=len(self.data['points']),initialization_seconds=self.init_seconds,counts=self.counts,computational_fallback_records=list(self.backend_fallbacks),original_basis_and_posterior_unchanged=True,all_past_fused=True,no_positive_probability_pruning=True,geometry_collapse=False,source_geometry_read=False,policy_candidates_changed=False))
