"""Exact original ray-union counts: FFT only for obstacle-free ray templates.

Every occluded view uses the unchanged visible_mask. No candidate is removed.
"""
import numpy as np
from scipy.signal import fftconvolve
from utils.grid_geometry import ray_templates,visible_mask
try:
    from nso_bridge.exact_visibility_native import batch_visibility_counts
except ImportError:
    batch_visibility_counts=None

class ExactVisibilityCounts:
    def __init__(self,occupied,radar_unknown,camera_unknown,radar_range,camera_range,fov):
        self.occupied=np.asarray(occupied,bool);self.tables={};self.boundary_cache={};self.empty=np.zeros_like(self.occupied);self.native=batch_visibility_counts is not None
        for kind,mask,radius,angle in [('radar',radar_unknown,radar_range,360),('camera',camera_unknown,camera_range,fov)]:
            for heading in range(4):
                offsets,valid=ray_templates(radius,angle,heading)
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
                self.tables[(kind,heading)]=(counts,hits,radius,angle,np.asarray(mask,bool))
    @staticmethod
    def correlate(mask,kernel):
        raw=fftconvolve(mask,kernel[::-1,::-1],mode='same');rounded=np.rint(raw)
        if np.max(np.abs(raw-rounded))>1e-6:raise ArithmeticError('ray-union integer FFT roundoff bound exceeded')
        return rounded.astype(np.int32)
    def count(self,kind,cell,heading):
        counts,hits,radius,angle,mask=self.tables[(kind,heading)]
        if self.native:return int(counts[cell])
        if hits[cell]==0:
            r,c=cell;h,w=self.occupied.shape
            if radius<=r<h-radius and radius<=c<w-radius:return int(counts[cell])
            key=(kind,heading,min(r,radius),min(h-1-r,radius),min(c,radius),min(w-1-c,radius))
            if key not in self.boundary_cache:
                # Resource-boundary ray ordering is part of the reference.
                # Cache its exact visible offsets, then gather the actual mask.
                self.boundary_cache[key]=(np.argwhere(visible_mask(self.empty,cell,heading,radius,angle))-cell).astype(np.int32)
            offsets=self.boundary_cache[key]
            return int(np.count_nonzero(mask[r+offsets[:,0],c+offsets[:,1]]))
        return int(np.count_nonzero(visible_mask(self.occupied,cell,heading,radius,angle)&mask))
