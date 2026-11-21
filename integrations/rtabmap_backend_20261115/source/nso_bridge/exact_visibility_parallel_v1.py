"""Staged implementation-equivalent parallel construction of exact count tables.

Same eight heading-specific ray templates, FFT integer counts, native boundary
queries and complete state domain. No radar heading reuse is assumed: actual
ray-template hashes differ. Each C call has private C++ stamp/output buffers;
ctypes.CDLL releases the GIL. Original class remains unmodified/selectable.
"""
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
import numpy as np
from utils.grid_geometry import ray_templates
from nso_bridge.exact_visibility_counts import ExactVisibilityCounts as Original
from nso_bridge.exact_visibility_native import batch_visibility_counts,_library
PROFILE='exact_visibility_parallel_v1'
MAX_WORKERS=3

class ParallelExactVisibilityCounts(Original):
    def __init__(self,occupied,radar_unknown,camera_unknown,radar_range,camera_range,fov):
        self.occupied=np.asarray(occupied,bool)
        self.tables={};self.boundary_cache={};self.empty=np.zeros_like(self.occupied)
        self.native=batch_visibility_counts is not None
        _library()  # Resolve/cache the CDLL and argument types once before concurrency.
        jobs=[]
        for kind,mask,radius,angle in [('radar',radar_unknown,radar_range,360),('camera',camera_unknown,camera_range,fov)]:
            for heading in range(4):jobs.append((kind,np.asarray(mask,bool),radius,angle,heading))
        self.table_build_seconds={}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS,thread_name_prefix='nso-exact-count') as executor:
            futures=[executor.submit(self._build,*job) for job in jobs]
            # Preserve the original deterministic insertion order and exact data types.
            for job,future in zip(jobs,futures):
                table,elapsed=future.result()
                key=(job[0],job[4]);self.tables[key]=table;self.table_build_seconds[str(key)]=elapsed
    def _build(self,kind,mask,radius,angle,heading):
        started=perf_counter();offsets,valid=ray_templates(radius,angle,heading)
        unique=np.unique(offsets[valid],axis=0)
        kernel=np.zeros((2*radius+1,2*radius+1),float)
        kernel[unique[:,0]+radius,unique[:,1]+radius]=1
        counts=self.correlate(np.asarray(mask,float),kernel)
        hits=self.correlate(self.occupied.astype(float),kernel)
        if self.native:
            rows,cols=np.indices(self.occupied.shape)
            boundary=(rows<radius)|(cols<radius)|(rows>=self.occupied.shape[0]-radius)|(cols>=self.occupied.shape[1]-radius)
            positions=np.argwhere((hits>0)|boundary)
            values=batch_visibility_counts(self.occupied,np.asarray(mask,bool),positions,offsets,valid)
            counts[tuple(positions.T)]=values
        return (counts,hits,radius,angle,np.asarray(mask,bool)),perf_counter()-started


def install():
    """Test hook only; an active process must explicitly select a reviewed profile."""
    import nso_bridge.exact_visibility_counts as module
    module.ExactVisibilityCounts=ParallelExactVisibilityCounts
