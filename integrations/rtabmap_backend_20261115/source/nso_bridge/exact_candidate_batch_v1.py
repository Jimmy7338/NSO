"""Opt-in exact computation for the unchanged unbounded generate_options.

Retains every feasible state. Only columnar storage/gathers, stable ordering and
lazy camera construction replace Python work; aperture/routing/scoring stay in
the original function. No default hook and no vendor mutation.
"""
import inspect
import hashlib
from collections.abc import Sequence
import numpy as np

PROFILE='exact_candidate_batch_v1'
REVIEWED_SOURCE_SHA256='4aca18992d18808e4457026376bce4157a82d6e3deb95b9741fef7d5ef1df751'


class Row(dict):
    def __init__(self,pool,index):self.pool=pool;self.index=int(index)
    def __getitem__(self,key):
        p=self.pool;i=self.index
        if key=='state':return int(p.states[i])
        if key=='pose':return (int(p.rows[i]),int(p.cols[i]),int(p.headings[i]))
        if key=='camera':
            if i not in p.camera_cache:
                pose=p.templates[int(p.headings[i])].copy()
                pose[:3,3]=[(int(p.cols[i])+.5)*p.config.resolution_m,
                    (p.height-int(p.rows[i])-.5)*p.config.resolution_m,p.config.camera_height_m]
                p.camera_cache[i]=pose
            return p.camera_cache[i]
        if key=='cost':return int(p.costs[i])
        if key=='outbound_cost':return int(p.outward[i])
        if key=='radar_unknown_cells':return int(p.radar[i])
        if key=='camera_unknown_cells':return int(p.camera[i])
        if key=='unknown_cells':return int(p.radar[i]+p.camera[i])
        raise KeyError(key)


class Pool(Sequence):
    def __init__(self,feasible,cells,costs,outward,lookup,has_radar,has_camera,templates,config,height):
        self.states=np.asarray(feasible,dtype=np.int64)
        coords=cells[self.states//4]
        self.rows=coords[:,0];self.cols=coords[:,1];self.headings=self.states%4
        # dijkstra edges have exact integer unit weights; preserve Python round
        # ties-to-even via rint, checking finite int64 representation first.
        rawcost=costs[self.states];rawout=outward[self.states]
        if not np.isfinite(rawcost).all() or not np.isfinite(rawout).all() or np.any(np.abs(rawcost)>2**53) or np.any(np.abs(rawout)>2**53):
            raise ValueError('exact batch requires representable finite path costs')
        self.costs=np.rint(rawcost).astype(np.int64);self.outward=np.rint(rawout).astype(np.int64)
        self.radar=np.zeros(len(self),np.int64);self.camera=np.zeros(len(self),np.int64)
        for kind,enabled,destination in [('radar',has_radar,self.radar),('camera',has_camera,self.camera)]:
            if enabled:
                for h in range(4):
                    select=np.flatnonzero(self.headings==h)
                    destination[select]=lookup.tables[(kind,h)][0][self.rows[select],self.cols[select]]
        self.templates=templates;self.config=config;self.height=height;self.camera_cache={}
    def __len__(self):return len(self.states)
    def __getitem__(self,i):
        if isinstance(i,slice):return [Row(self,j) for j in range(*i.indices(len(self)))]
        if i<0:i+=len(self)
        if i>=len(self):raise IndexError(i)
        return Row(self,i)
    def coverage_ranked(self):
        order=np.lexsort((self.headings,self.cols,self.rows,self.costs,self.outward,-(self.radar+self.camera)))
        return (Row(self,i) for i in order)
    def rotations(self,current):
        indices=np.flatnonzero((self.rows==current[0])&(self.cols==current[1]))
        return (Row(self,i) for i in indices)
    def asset_rows(self,heading,asset_bounds,xy_bound_sq):
        indices=np.flatnonzero(self.headings==heading)
        lo,hi=asset_bounds
        # Same original arithmetic and strict > comparison; only proven-zero
        # aperture shortcut is evaluated in bulk, not a new eligibility test.
        x=(self.cols[indices].astype(float)+.5)*self.config.resolution_m
        y=(self.height-self.rows[indices].astype(float)-.5)*self.config.resolution_m
        dx=np.maximum(np.maximum(lo[0]-x,x-hi[0]),0.)
        dy=np.maximum(np.maximum(lo[1]-y,y-hi[1]),0.)
        indices=indices[~(dx*dx+dy*dy>xy_bound_sq)]
        return (Row(self,i) for i in indices)


def build_pool(original,namespace):
    """Clone only known source regions; all other original statements unchanged."""
    source=inspect.getsource(original)
    if hashlib.sha256(source.encode()).hexdigest()!=REVIEWED_SOURCE_SHA256:
        raise RuntimeError('generate_options reviewed source hash changed')
    start=source.index('    pool = []\n    for node in feasible:')
    end=source.index('    chosen, used = [], set()',start)
    expected=source[start:end]
    # Checked binding to the reviewed original loop structure. Any source drift
    # fails explicitly rather than applying a broad AST/name transformation.
    if "pose = camera_templates[direction].copy()" not in expected or "int(round(costs[node]))" not in expected or "count_lookup.count('camera',cell,direction)" not in expected:
        raise RuntimeError('generate_options reviewed loop changed')
    replacement='''    if unbounded and count_lookup is not None and count_lookup.native:
        pool = _BatchPool(feasible,cells,costs,outward,count_lookup,has_unknown,has_camera_unknown,
            camera_templates,config,belief.shape[0])
    else:
'''+''.join('    '+line if line.strip() else line for line in expected.splitlines(keepends=True))
    source=source[:start]+replacement+source[end:]
    changes=[("ranked = sorted(pool, key=lambda row: (-row['unknown_cells'], row['outbound_cost'],\n                                               row['cost'], row['pose']))",
              "ranked = (pool.coverage_ranked() if isinstance(pool,_BatchPool) else sorted(pool, key=lambda row: (-row['unknown_cells'], row['outbound_cost'],\n                                               row['cost'], row['pose'])))"),
             ("rotations = [row for row in pool if row['pose'][:2] == current[:2]", "rotations = [row for row in (pool.rotations(current) if isinstance(pool,_BatchPool) else pool) if row['pose'][:2] == current[:2]"),
             ("            for row in pool:\n", "            for row in (pool.asset_rows(wanted_heading,asset_xy_bounds[ai],xy_bound_sq) if isinstance(pool,_BatchPool) else pool):\n")]
    for old,new in changes:
        if source.count(old)!=1:raise RuntimeError('generate_options reviewed source region changed')
        source=source.replace(old,new,1)
    ns=dict(namespace);ns['_BatchPool']=Pool
    exec(compile(source,__file__+':reviewed_generate_options','exec'),ns)
    return ns['generate_options']


def install(profile):
    if profile!=PROFILE:raise ValueError('explicit exact batch compute profile required')
    from nso_bridge.joint_engine import VENDOR  # Establish the preserved joint vendor import path.
    from nso_bridge import exploration_joint_policy as module
    if getattr(module.generate_options,'compute_profile',None)==PROFILE:return module.generate_options
    original=module.generate_options
    candidate=build_pool(original,module.__dict__)
    candidate.compute_profile=PROFILE;candidate.original_generate_options=original
    candidate.reviewed_source_sha256=REVIEWED_SOURCE_SHA256
    module.generate_options=candidate
    return candidate
