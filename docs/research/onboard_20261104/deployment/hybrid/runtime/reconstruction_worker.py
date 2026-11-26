#!/usr/bin/env python3
"""Display-only measured RGB-D TSDF. No ROS, navigation, CAD or device access.

JSONL requests: {"method":"integrate","path":"/absolute/source.npz"}.
Each response acknowledges that the source file has been read and may be
replaced. The volume remains in memory; cloud/status JSON replace atomically.
Voxel size is a computation setting, not independently verified accuracy.
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np


def atomic_json(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,separators=(',',':'),allow_nan=False))
    os.replace(temporary,path)


def prepare_arrays(rgb, depth, intrinsic, world_from_camera, stride=2, self_mask=None, far_m=6.):
    """Select original pixels, scale K exactly, and preserve the caller arrays."""
    rgb=np.asarray(rgb);depth=np.asarray(depth);K=np.asarray(intrinsic,dtype=float)
    pose=np.asarray(world_from_camera,dtype=float)
    if depth.ndim!=2 or rgb.shape!=depth.shape+(3,) or rgb.dtype!=np.uint8:
        raise ValueError('aligned full RGB uint8 HxWx3 and depth_m HxW required')
    if (K.shape!=(3,3) or not np.isfinite(K).all() or K[0,0]<=0 or K[1,1]<=0 or
            not np.allclose(K[2],[0,0,1]) or abs(K[0,1])+abs(K[1,0])>1e-9):
        raise ValueError('finite rectified actual camera intrinsic required')
    if (pose.shape!=(4,4) or not np.isfinite(pose).all() or
            not np.allclose(pose[3],[0,0,0,1]) or
            not np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-4) or
            not np.isclose(np.linalg.det(pose[:3,:3]),1.,atol=1e-4)):
        raise ValueError('actual rigid world_from_camera is required')
    if type(stride) is not int or stride not in (1,2,4):
        raise ValueError('pixel stride must be 1, 2 or 4')
    filtered=np.array(depth,dtype=np.float32,copy=True)
    with np.errstate(invalid='ignore'):
        invalid=~np.isfinite(filtered)|(filtered<=0)|(filtered>far_m)
    if self_mask is not None:
        if self_mask.shape!=depth.shape or self_mask.dtype!=np.bool_:
            raise ValueError('approved self-mask and source image shapes differ')
        invalid|=self_mask
    filtered[invalid]=0.
    sampled_K=K.copy();sampled_K[:2]/=stride
    return (np.ascontiguousarray(rgb[::stride,::stride]),
            np.ascontiguousarray(filtered[::stride,::stride]),sampled_K,pose.copy())


class ReconstructionVolume:
    def __init__(self, output_directory, *, voxel_m=.02, trunc_m=.08,
                 pixel_stride=2, maximum_display_points=50000, far_m=6., self_mask=None):
        if not (.005<=voxel_m<=.1 and voxel_m<trunc_m<=.4 and 0<far_m<=25):
            raise ValueError('invalid display reconstruction geometry settings')
        if pixel_stride not in (1,2,4) or not 1<=maximum_display_points<=200000:
            raise ValueError('invalid display sampling settings')
        with contextlib.redirect_stdout(sys.stderr):
            import open3d as o3d
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
        self.o3d=o3d
        self.volume=o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=voxel_m,
            sdf_trunc=trunc_m,color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        self.output=Path(output_directory);self.output.mkdir(parents=True,exist_ok=True)
        self.voxel_m=voxel_m;self.trunc_m=trunc_m;self.pixel_stride=pixel_stride
        self.maximum_display_points=maximum_display_points;self.far_m=far_m;self.self_mask=self_mask
        self.integrated_frames=0;self.seen=set();self.stamp_ns=None;self.poisoned=False
        self.current_status={};self.publish('waiting_for_measured_rgbd')

    def integrate(self, path):
        if self.poisoned:raise RuntimeError('TSDF backend failed; restart display worker before further integration')
        source=Path(path)
        if not source.is_absolute():raise ValueError('absolute measured RGB-D source path required')
        # All arrays are detached before responding; the caller may replace this
        # one pending file only after receiving the JSON acknowledgement.
        with np.load(source,allow_pickle=False) as data:
            rgb=data['rgb_raw'] if 'rgb_raw' in data else data['rgb']
            K=data['intrinsic'] if 'intrinsic' in data else data['K']
            arrays=prepare_arrays(rgb,data['depth_m'],K,data['world_from_camera'],
                self.pixel_stride,self.self_mask,self.far_m)
            receipt=json.loads(str(data['receipt_json'])) if 'receipt_json' in data else {}
            frame=str(data['frame']) if 'frame' in data else 'odom'
            historical=receipt.get('historical_tf',{}).get('world_from_camera',{})
            if frame!='odom' or historical.get('target_frame','odom')!='odom':
                raise ValueError('display volume requires this experiment fixed odom frame')
            stamp=(int(data['stamp_ns']) if 'stamp_ns' in data else
                   receipt.get('headers',{}).get('depth',{}).get('stamp_ns'))
            frame_id=str(data['frame_id']) if 'frame_id' in data else None
            if stamp is None and frame_id and frame_id.isdecimal():stamp=int(frame_id)
            if stamp is not None and (type(stamp) is not int or stamp<=0):
                raise ValueError('source stamp_ns must be a positive acquisition timestamp')
        rgb,depth,K,pose=arrays
        fingerprint=hashlib.sha256()
        for array in arrays:fingerprint.update(array.tobytes())
        key=(str(stamp) if stamp is not None else frame_id,fingerprint.hexdigest())
        if key in self.seen:
            return dict(ok=True,duplicate=True,source_read_complete=True,**self.current_status)
        valid=int(np.count_nonzero(depth))
        if valid:
            rgbd=self.o3d.geometry.RGBDImage.create_from_color_and_depth(
                self.o3d.geometry.Image(rgb),self.o3d.geometry.Image(depth),
                depth_scale=1.,depth_trunc=self.far_m,convert_rgb_to_intensity=False)
            intrinsic=self.o3d.camera.PinholeCameraIntrinsic(depth.shape[1],depth.shape[0],
                K[0,0],K[1,1],K[0,2],K[1,2])
            try:
                self.volume.integrate(rgbd,intrinsic,np.linalg.inv(pose))
            except Exception:
                self.poisoned=True;self.publish('backend_error');raise
            self.integrated_frames+=1
        self.seen.add(key);self.stamp_ns=stamp
        self.publish('integrated' if valid else 'no_valid_depth')
        return dict(ok=True,duplicate=False,source_read_complete=True,
                    valid_depth_pixels=valid,**self.current_status)

    def publish(self, state):
        cloud=self.volume.extract_point_cloud()
        xyz=np.asarray(cloud.points);colors=np.asarray(cloud.colors)
        total=len(xyz);stride=max(1,math.ceil(total/self.maximum_display_points))
        xyz=xyz[::stride];colors=colors[::stride]
        stamp=None if self.stamp_ns is None else self.stamp_ns/1e9
        payload=dict(positions=np.round(xyz,4).ravel().tolist(),
            colors=np.round(np.clip(colors,0.,1.),3).ravel().tolist(),frame='odom',stamp=stamp,
            source_stamp_ns=self.stamp_ns,geometry_source='measured_rgbd_tsdf',
            physical_accuracy_verified=False)
        atomic_json(self.output/'cloud.json',payload)
        self.current_status=dict(point_count=len(xyz),reconstructed_point_count=total,
            integrated_frames=self.integrated_frames,voxel_m=self.voxel_m,sdf_trunc_m=self.trunc_m,
            frame='odom',stamp=stamp,source_stamp_ns=self.stamp_ns,status=state,
            physical_accuracy_verified=False,geometry_source='measured_rgbd_tsdf',
            pixel_stride=self.pixel_stride,depth_trunc_m=self.far_m,
            original_intrinsic_scaled_for_selected_pixels=True,
            self_mask_applied=self.self_mask is not None,
            display_subsample_stride=stride,maximum_display_points=self.maximum_display_points,
            volume_retains_all_integrated_frames=True,experiment_scoring=False,
            process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,
            updated_unix_s=time.time())
        atomic_json(self.output/'status.json',self.current_status)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory',required=True)
    parser.add_argument('--pixel-stride',type=int,choices=(1,2,4),default=2)
    parser.add_argument('--voxel-m',type=float,default=.02)
    parser.add_argument('--trunc-m',type=float,default=.08)
    parser.add_argument('--far-m',type=float,default=6.)
    parser.add_argument('--maximum-display-points',type=int,default=50000)
    parser.add_argument('--approved-self-mask',action='store_true',
                        help='apply the existing verified fixed IMU mask to raw full-size historical frames')
    args=parser.parse_args()
    mask=None
    if args.approved_self_mask:
        from visual_sensor_bridge import load_approved_mask
        mask,_=load_approved_mask()
    worker=ReconstructionVolume(args.output_directory,voxel_m=args.voxel_m,trunc_m=args.trunc_m,
        pixel_stride=args.pixel_stride,maximum_display_points=args.maximum_display_points,
        far_m=args.far_m,self_mask=mask)
    for line in sys.stdin:
        try:
            request=json.loads(line)
            method=request.get('method')
            if method=='integrate':result=worker.integrate(request['path'])
            elif method=='status':result=dict(ok=True,**worker.current_status)
            elif method=='shutdown':
                print(json.dumps(dict(ok=True,status='stopped')),flush=True);break
            else:raise ValueError('unknown reconstruction method')
        except Exception as exc:
            result=dict(ok=False,error=type(exc).__name__+': '+str(exc),source_read_complete=True)
        print(json.dumps(result,allow_nan=False,separators=(',',':')),flush=True)


if __name__=='__main__':main()
