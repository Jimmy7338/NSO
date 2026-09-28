"""Track actual RGB-D ray footprints separately from the 360-degree lidar map."""
import numpy as np
from nso.mapping3d_v2 import QualityMapperV2
from utils.grid_geometry import supercover_line


class CameraQualityMapperV2(QualityMapperV2):
    def __init__(self,shape,config,truncation_m=.12):
        super().__init__(shape,config,truncation_m)
        self.camera_seen=np.zeros(shape,bool)

    def update(self,frame,scan=None):
        super().update(frame,scan)
        origin=self.grid_cell(frame.world_from_camera[:3,3])
        points,_=frame.points(stride=6)
        # Use measured depth endpoints, not a simulator visibility mask. This is
        # a planar projection of RGB-D rays, not measured 3D surface completeness.
        cells=np.array([self.grid_cell(p) for p in points],int)
        if not len(cells):return
        for target in np.unique(cells,axis=0):
            offsets=np.asarray(supercover_line(int(target[0]-origin[0]),int(target[1]-origin[1])))
            r=offsets[:,0]+origin[0];c=offsets[:,1]+origin[1]
            valid=(r>=0)&(r<self.shape[0])&(c>=0)&(c<self.shape[1])
            self.camera_seen[r[valid],c[valid]]=True
