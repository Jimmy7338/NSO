"""Diagnostic only; exact native depth insertion support, not final fused belief.
Sources: joint_vendor/utils/rgbd_contract.py:67 and nso/mapping3d.py:19,52.
Input depth must already be canonical production depth (range adaptation unchanged).
No confidence threshold is applied here: native mapper has no confidence input.
"""
import numpy as np

def anchor_depth_support(depth, intrinsic, planning_from_camera, shape=(400,400), resolution=.2, anchor=(200,200), radius=.7212185868375829):
    v,u=np.mgrid[0:depth.shape[0]:2,0:depth.shape[1]:2]
    d=np.asarray(depth)[::2,::2]; valid=d>0
    camera=np.stack([(u-intrinsic[0,2])*d/intrinsic[0,0],(v-intrinsic[1,2])*d/intrinsic[1,1],d],-1)[valid]
    points=camera@planning_from_camera[:3,:3].T+planning_from_camera[:3,3]
    uv=np.stack([u,v],-1)[valid]
    obstacle=(points[:,2]>.15)&(points[:,2]<1.2)
    cells=np.c_[shape[0]-1-np.floor(points[:,1]/resolution).astype(int),np.floor(points[:,0]/resolution).astype(int)]
    inside=(cells[:,0]>=0)&(cells[:,0]<shape[0])&(cells[:,1]>=0)&(cells[:,1]<shape[1])
    footprint=np.linalg.norm(cells-np.asarray(anchor),axis=1)*resolution<=radius+resolution/np.sqrt(2)
    mask=obstacle&inside&footprint
    if mask.any():
        unique,counts=np.unique(cells[mask],axis=0,return_counts=True)
    else:
        unique,counts=np.empty((0,2),dtype=int),np.empty(0,dtype=int)
    return dict(point_count=int(mask.sum()),cells=unique.tolist(),counts=counts.tolist(),uv=uv[mask].tolist(),planning_points=points[mask].tolist(),valid_sample_count=int(valid.sum()))
