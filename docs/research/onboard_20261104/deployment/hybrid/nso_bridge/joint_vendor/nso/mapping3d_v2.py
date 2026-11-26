"""Configurable TSDF and sensor-derived quality statistics; no world / GT input."""
import numpy as np
import open3d as o3d
from nso.mapping3d import SensorMapper


class QualityMapperV2(SensorMapper):
    def __init__(self, shape, config, truncation_m=.12):
        super().__init__(shape,config)
        if truncation_m < config.voxel_m:
            raise ValueError('truncation must be at least one voxel')
        self.volume=o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=config.voxel_m,sdf_trunc=truncation_m,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        self.quality={}

    def update(self,frame,scan=None):
        super().update(frame,scan)
        d=frame.depth_m;h,w=d.shape
        v,u=np.mgrid[:h,:w]
        local=np.stack([(u-frame.intrinsic[0,2])*d/frame.intrinsic[0,0],
                        (v-frame.intrinsic[1,2])*d/frame.intrinsic[1,1],d],axis=-1)
        dx=np.roll(local,-1,axis=1)-np.roll(local,1,axis=1)
        dy=np.roll(local,-1,axis=0)-np.roll(local,1,axis=0)
        normals=np.cross(dx,dy);length=np.linalg.norm(normals,axis=-1)
        valid=(d>.15)&(length>1e-8)
        valid[[0,-1],:]=False;valid[:,[0,-1]]=False
        valid&=(np.roll(d,1,0)>0)&(np.roll(d,-1,0)>0)&(np.roll(d,1,1)>0)&(np.roll(d,-1,1)>0)
        normals/=np.maximum(length[...,None],1e-8)
        rows,cols=np.nonzero(valid&((v%3)==0)&((u%3)==0))
        pts=local[rows,cols]@frame.world_from_camera[:3,:3].T+frame.world_from_camera[:3,3]
        ns=normals[rows,cols]@frame.world_from_camera[:3,:3].T
        keys=np.floor(pts/.15).astype(int)
        _,indices=np.unique(keys,axis=0,return_index=True)
        for i in indices:
            p=pts[i];key=tuple(keys[i]);normal=ns[i]
            delta=frame.world_from_camera[:3,3]-p;distance=np.linalg.norm(delta)
            incidence=abs(float(normal@delta))/max(distance,1e-5)
            information=incidence**2/max(distance**2,.25)
            category=int(frame.semantic[rows[i],cols[i]])
            az=np.arctan2(delta[1],delta[0]);bit=1<<(int(np.floor((az+np.pi)/(2*np.pi)*8))%8)
            old=self.quality.get(key)
            if old is None:
                self.quality[key]=dict(point=p.copy(),normal=normal.copy(),n=1,bits=bit,label=category,
                    information=information,best_range=distance,residual=0.,normal_dispersion=0.)
            else:
                n=old['n']+1;alpha=1/min(n,20)
                residual=float((p-old['point'])@old['normal'])
                old['residual']=(1-alpha)*old['residual']+alpha*min(residual**2,.01)
                old['normal_dispersion']=(1-alpha)*old['normal_dispersion']+alpha*(1-abs(float(normal@old['normal'])))
                if normal@old['normal']<0:normal=-normal
                old['normal']=(1-alpha)*old['normal']+alpha*normal
                old['normal']/=max(np.linalg.norm(old['normal']),1e-6)
                old['point']=(1-alpha)*old['point']+alpha*p
                old['n']=n;old['bits']|=bit
                old['information']=max(old['information'],information)
                old['best_range']=min(old['best_range'],distance)
                if category:old['label']=category

    def quality_evidence(self,max_points=1600):
        rows=[q for q in self.quality.values() if .12<q['point'][2]<1.8]
        if len(rows)>max_points:
            rows=[rows[i] for i in np.linspace(0,len(rows)-1,max_points,dtype=int)]
        if not rows:return None
        return {key:np.asarray([r[key] for r in rows]) for key in rows[0]}
