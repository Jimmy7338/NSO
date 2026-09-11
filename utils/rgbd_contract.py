"""Hardware-neutral RGB-D contract: metres, seconds, optical camera coordinates."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PlanarScan:
    timestamp_s: float
    ranges_m: np.ndarray
    angle_min_rad: float
    angle_increment_rad: float
    range_max_m: float
    world_from_laser: np.ndarray  # laser x forward, y left, z up

    def save(self, path):
        np.savez_compressed(path,**self.__dict__)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:
            return cls(**{k:(data[k].copy() if k in ('ranges_m','world_from_laser') else float(data[k]))
                          for k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class RGBDFrame:
    timestamp_s: float
    depth_m: np.ndarray
    color_rgb: np.ndarray
    intrinsic: np.ndarray
    world_from_camera: np.ndarray
    semantic: np.ndarray  # 0 unknown; optional synthetic/real visible labels

    def validate(self):
        h, w = self.depth_m.shape
        if self.color_rgb.shape != (h,w,3) or self.semantic.shape != (h,w):
            raise ValueError('RGB/depth/semantic images must be aligned')
        if self.intrinsic.shape != (3,3) or self.world_from_camera.shape != (4,4):
            raise ValueError('invalid calibration shapes')
        if not np.isfinite(self.depth_m).all() or np.any(self.depth_m < 0):
            raise ValueError('depth must be finite metres; invalid pixels are zero')
        if not np.isfinite(self.world_from_camera).all() or not np.isfinite(self.intrinsic).all():
            raise ValueError('nonfinite calibration')
        if self.intrinsic[0,0] <= 0 or self.intrinsic[1,1] <= 0:
            raise ValueError('focal length must be positive')
        r = self.world_from_camera[:3,:3]
        if not np.allclose(r.T@r,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(r),1,atol=1e-5):
            raise ValueError('pose rotation must be in SO(3)')
        if not np.allclose(self.world_from_camera[3],[0,0,0,1]):
            raise ValueError('invalid homogeneous pose')
        if not np.isfinite(self.timestamp_s):
            raise ValueError('invalid timestamp')
        return self

    def save(self, path):
        self.validate()
        np.savez_compressed(path, timestamp_s=self.timestamp_s, depth_m=self.depth_m,
            color_rgb=self.color_rgb, intrinsic=self.intrinsic,
            world_from_camera=self.world_from_camera, semantic=self.semantic)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(**{k:(float(data[k]) if k=='timestamp_s' else data[k].copy())
                          for k in cls.__dataclass_fields__}).validate()

    def points(self, stride=1):
        v,u = np.mgrid[0:self.depth_m.shape[0]:stride,0:self.depth_m.shape[1]:stride]
        depth = self.depth_m[::stride,::stride]
        valid = depth > 0
        x = (u-self.intrinsic[0,2])*depth/self.intrinsic[0,0]
        y = (v-self.intrinsic[1,2])*depth/self.intrinsic[1,1]
        camera = np.stack([x,y,depth],axis=-1)[valid]
        world = camera@self.world_from_camera[:3,:3].T + self.world_from_camera[:3,3]
        return world, self.semantic[::stride,::stride][valid]
