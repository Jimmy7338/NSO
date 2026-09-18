"""ROS-independent conversion helpers for versioned robot recordings."""
import numpy as np
from scripts.record_ros1_rgbd import sensor_metadata


def convert_ranges(raw,minimum,maximum,positive_inf_policy='invalid'):
    raw=np.asarray(raw)
    if raw.ndim!=1 or not len(raw) or raw.dtype.kind not in 'fi':
        raise ValueError('nonempty numeric laser vector required')
    if not np.isfinite([minimum,maximum]).all() or not 0<=minimum<maximum:
        raise ValueError('invalid laser range limits')
    if positive_inf_policy not in ('invalid','clear_to_max'):
        raise ValueError('explicit positive-infinity policy required')
    valid=np.isfinite(raw)&(raw>=minimum)&(raw<=maximum)&(raw>0)
    result=np.where(valid,raw,0.).astype(np.float32)
    if positive_inf_policy=='clear_to_max':result[np.isposinf(raw)]=maximum
    counts=dict(valid_finite=int(valid.sum()),positive_infinity=int(np.isposinf(raw).sum()),
        negative_infinity=int(np.isneginf(raw).sum()),nan=int(np.isnan(raw).sum()),
        finite_below_or_nonpositive=int((np.isfinite(raw)&((raw<minimum)|(raw<=0))).sum()),
        finite_above_maximum=int((np.isfinite(raw)&(raw>maximum)).sum()))
    return result,dict(positive_inf_policy=positive_inf_policy,invalid_stored_as=0.,
        finite_above_maximum_clipped=False,counts=counts)


def convert_depth(raw,encoding,maximum):
    raw=np.asarray(raw)
    if raw.ndim!=2 or not np.isfinite(maximum) or maximum<=0:raise ValueError('invalid depth image or limit')
    if encoding=='16UC1' and raw.dtype.kind=='u' and raw.dtype.itemsize==2:
        depth=raw.astype(np.float32)*.001
    elif encoding=='32FC1' and raw.dtype.kind=='f' and raw.dtype.itemsize==4:
        depth=raw.astype(np.float32)
    else:raise ValueError('depth dtype does not match supported source encoding')
    return np.where(np.isfinite(depth)&(depth>0)&(depth<=maximum),depth,0.).astype(np.float32)


def rectified_projection(info,image_shape):
    """Registered left-camera P with explicit binning and already rectified ROI.

    R maps source optical coordinates into rectified coordinates, so R.T maps
    reconstructed optical rays back to the TF source basis. No image warping.
    """
    def integer(value,name):
        if isinstance(value,(bool,np.bool_)) or int(value)!=value or value<0:raise ValueError('invalid '+name)
        return int(value)
    width,height=integer(info.width,'width'),integer(info.height,'height')
    if not width or not height:raise ValueError('positive calibrated resolution required')
    bx,by=integer(info.binning_x,'binning_x') or 1,integer(info.binning_y,'binning_y') or 1
    roi={k:integer(getattr(info.roi,k),k) for k in ('x_offset','y_offset','width','height')}
    roi['do_rectify']=bool(info.roi.do_rectify)
    if roi['do_rectify']:raise ValueError('ROI requires image rectification; provide verified rectified ROI')
    x,y,w,h=(roi[k] for k in ('x_offset','y_offset','width','height'))
    if x==y==w==h==0:w,h=width,height
    if not w or not h or x+w>width or y+h>height:raise ValueError('ROI outside calibrated image')
    if tuple(image_shape)!=(h//by,w//bx):raise ValueError('depth dimensions differ from binned ROI')
    K=np.asarray(info.K,dtype=float).reshape(3,3)
    P=np.asarray(info.P,dtype=float).reshape(3,4)
    R=np.asarray(info.R,dtype=float).reshape(3,3)
    if not all(np.isfinite(v).all() for v in (K,P,R)):raise ValueError('nonfinite calibration')
    if K[0,0]<=0 or K[1,1]<=0 or P[0,0]<=0 or P[1,1]<=0:raise ValueError('calibrated K and rectified P required')
    if not np.allclose(P[2],[0,0,1,0],rtol=0,atol=1e-12) or np.any(P[:2,3]!=0):
        raise ValueError('registered left-camera projection required; stereo translation cannot be discarded')
    if not np.allclose(R.T@R,np.eye(3),rtol=0,atol=1e-5) or abs(np.linalg.det(R)-1)>1e-5:
        raise ValueError('invalid rectification rotation')
    intrinsic=P[:,:3].copy();intrinsic[0,2]-=x;intrinsic[1,2]-=y
    intrinsic[0,:]/=bx;intrinsic[1,:]/=by
    raw_from_rectified=np.eye(4);raw_from_rectified[:3,:3]=R.T
    return intrinsic,raw_from_rectified,dict(binning_x=int(info.binning_x),binning_y=int(info.binning_y),
        roi=roi,used_intrinsic=intrinsic.tolist(),source_optical_from_rectified=raw_from_rectified.tolist(),
        projection_contract='registered rectified left image; ROI offset before binning; TF pose composed with R transpose')


def transform_response(message,target,source):
    if message.header.frame_id!=target or message.child_frame_id!=source:
        raise ValueError('TF response frames differ from requested transform')
    t=message.transform.translation;q=message.transform.rotation
    xyz=np.asarray([t.x,t.y,t.z],float);quat=np.asarray([q.x,q.y,q.z,q.w],float)
    norm=float(np.linalg.norm(quat))
    if not np.isfinite(xyz).all() or not np.isfinite(quat).all() or abs(norm-1)>1e-3:
        raise ValueError('invalid TF translation or unit quaternion')
    x,y,z,w=quat/norm
    matrix=np.eye(4);matrix[:3,:3]=[
        [1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
    matrix[:3,3]=xyz
    stamp=message.header.stamp;secs,nsecs=int(stamp.secs),int(stamp.nsecs)
    if secs<0 or not 0<=nsecs<1_000_000_000:raise ValueError('invalid TF response timestamp')
    return matrix,dict(target_frame=target,source_frame=source,secs=secs,nsecs=nsecs,
        timestamp_ns=secs*1_000_000_000+nsecs,translation_xyz_m=xyz.tolist(),quaternion_xyzw=quat.tolist(),
        matrix=matrix.tolist(),calibration_truth_verified=False)


def extended_metadata(depth,rgb,scan,info,semantic,config,image_shape):
    result=sensor_metadata(depth,rgb,scan,info,semantic,config)
    intrinsic,rectified,projection=rectified_projection(info,image_shape)
    if info.header.frame_id!=result['tf_queries']['camera_source_frame']:
        raise ValueError('CameraInfo optical frame differs from TF source')
    result['schema_version']='ros1_rgbd_source_metadata/2'
    result['camera_info'].update(projection)
    return result,intrinsic,rectified
