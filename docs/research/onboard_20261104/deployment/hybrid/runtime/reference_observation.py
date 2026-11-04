"""One real observation; no goals, publishers, CAN sender or video.

Raw RGB-D, scan invalid values, source headers and historical TF are retained.
Only an explicit task calibration shared with both V35 templates may select
pixels/rays and encode existing invalid returns as zero. A missing scan is an
error, never an all-zero observation. V35 stopped capture and explicitly scoped
joint live capture retain separate motion contracts. Importing starts no ROS
subscriptions.
"""
from collections import deque
import hashlib
import io
import json
import math
from pathlib import Path
import threading
import time

import numpy as np

from mission import rgb_array,paid_frame_source_provenance
from nso_ros_coordinator import matrix

TOPICS=dict(rgb='/zed2i/zed_node/left/image_rect_color',
    depth='/zed2i/zed_node/depth/depth_registered',
    camera_info='/zed2i/zed_node/depth/camera_info',scan='/scan')
CALLERS=dict(rgb='/zed2i/zed_node',depth='/zed2i/zed_node',
             camera_info='/zed2i/zed_node',scan='/lslidar_driver_node')


def source_depth_metres(message):
    """Decode byte strides/units while retaining all original invalid values."""
    types={'32FC1':('f4',4,1.),'16UC1':('u2',2,.001)}
    if message.encoding not in types:raise ValueError('unsupported source depth encoding')
    dtype,size,scale=types[message.encoding]
    if message.step<message.width*size or len(message.data)<message.height*message.step:
        raise ValueError('truncated source depth image')
    dtype=np.dtype(('>' if message.is_bigendian else '<')+dtype)
    raw=np.ndarray((message.height,message.width),dtype=dtype,buffer=message.data,
                   strides=(message.step,size))
    with np.errstate(invalid='ignore'):
        return raw.astype(np.float32,copy=True)*scale


def calibration_sha256(value):
    raw=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _indices(value, size, name):
    if (not isinstance(value,list) or not value or
            any(type(i) is not int or not 0<=i<size for i in value) or len(set(value))!=len(value)):
        raise ValueError('explicit unique in-range '+name+' required')
    return np.asarray(value,dtype=np.int64)


def sample_reference_arrays(raw, source, calibration):
    """Pure declared sampling, not template inference or task identification."""
    if not isinstance(calibration,dict) or calibration.get('schema')!='nso_real_sensor_calibration/1':
        raise ValueError('same explicit nso_real_sensor_calibration/1 as both templates required')
    if 'scan_ranges_raw' not in raw or np.asarray(raw['scan_ranges_raw']).size==0:
        raise ValueError('actual LaserScan missing; zero substitution forbidden')
    camera,scan=calibration['rgbd'],calibration['scan']
    actual_camera,actual_scan=source['rgbd'],source['scan']
    for name in ('width','height','optical_frame'):
        if camera.get(name)!=actual_camera[name]:raise ValueError('RGB-D calibration differs: '+name)
    if not np.allclose(camera['intrinsic_3x3'],actual_camera['intrinsic_3x3'],rtol=0.,atol=1e-7):
        raise ValueError('actual CameraInfo intrinsic differs from template calibration')
    if (camera.get('depth_units')!='m' or
            camera.get('invalid_depth_policy')!='nonfinite_or_nonpositive_to_zero' or
            camera.get('rgb_policy')!='original_rgb8_at_selected_pixels'):
        raise ValueError('explicit V35 measured-depth/unaltered-RGB sampling rules required')
    for name in ('frame','range_count'):
        if scan.get(name)!=actual_scan[name]:raise ValueError('LaserScan calibration differs: '+name)
    for name in ('angle_min_rad','angle_increment_rad','range_min_m','range_max_m'):
        if not math.isfinite(scan[name]) or abs(scan[name]-actual_scan[name])>1e-7:
            raise ValueError('LaserScan calibration differs: '+name)
    if scan.get('invalid_range_policy')!='nonfinite_or_outside_sensor_limits_to_zero':
        raise ValueError('explicit V35 invalid measured range encoding required')
    for name in ('base_from_camera','base_from_laser'):
        declared=np.asarray(calibration['extrinsics'][name],dtype=float)
        if declared.shape!=(4,4) or not np.isfinite(declared).all() or not np.allclose(
                declared,source['extrinsics'][name],rtol=0.,atol=1e-7):
            raise ValueError('observed TF differs from task extrinsics: '+name)
    rows=_indices(camera['row_indices'],camera['height'],'RGB-D row_indices')
    cols=_indices(camera['column_indices'],camera['width'],'RGB-D column_indices')
    rays=_indices(scan['indices'],scan['range_count'],'LaserScan indices')
    depth=np.asarray(raw['depth_m'])[np.ix_(rows,cols)].copy()
    ranges=np.asarray(raw['scan_ranges_raw'])[rays].copy()
    with np.errstate(invalid='ignore'):
        depth[~np.isfinite(depth)|(depth<=0)]=0.
        invalid=(~np.isfinite(ranges))|(ranges<scan['range_min_m'])|(ranges>scan['range_max_m'])
    ranges[invalid]=0.
    rgb=np.asarray(raw['rgb_raw'])[np.ix_(rows,cols)].copy()
    return dict(depth=depth,rgb=rgb,ranges=ranges)


def nearest_scan(scans, stamp_ns, maximum_delta_s=.20):
    if not scans:raise ValueError('actual LaserScan missing; zero substitution forbidden')
    row=min(scans,key=lambda item:abs(item[1].header.stamp.to_nsec()-stamp_ns))
    if abs(row[1].header.stamp.to_nsec()-stamp_ns)>maximum_delta_s*1e9:
        raise ValueError('no real scan close enough to stopped RGB-D source time')
    return row


def _serialized(message):
    buffer=io.BytesIO();message.serialize(buffer)
    return np.frombuffer(buffer.getvalue(),dtype=np.uint8)


def _header(message):
    return dict(seq=int(message.header.seq),stamp_ns=int(message.header.stamp.to_nsec()),
                frame_id=message.header.frame_id,
                publisher=getattr(message,'_connection_header',{}).get('callerid'))


class _StoppedObservationPending(RuntimeError):
    """Feedback has not yet established the caller's stationary requirement."""


def _stopped(check):
    result=check()
    if result is True:return dict(stopped=True,evidence='caller_callback_boolean')
    if isinstance(result,dict) and result.get('stopped') is True:return dict(result)
    raise _StoppedObservationPending('fresh stopped chassis feedback is required for observation')


def _motion_snapshot(check):
    result=check()
    if not isinstance(result,dict):
        raise ValueError('live joint observation requires the actual motion snapshot dictionary')
    # Detach mutable CAN rows while preserving the original sample timestamps.
    return json.loads(json.dumps(result,allow_nan=False))


class ReferenceObservationCollector:
    def __init__(self,tfbuf):
        import rospy,message_filters
        from sensor_msgs.msg import Image,CameraInfo,LaserScan
        self.tfbuf=tfbuf;self.lock=threading.RLock()
        self.frames=deque(maxlen=4);self.scans=deque(maxlen=20);self.last_stamp_ns=-1
        self.filters=[message_filters.Subscriber(TOPICS[k],typ,queue_size=2,buff_size=8000000)
            for k,typ in (('rgb',Image),('depth',Image),('camera_info',CameraInfo))]
        self.sync=message_filters.TimeSynchronizer(self.filters,10)
        self.sync.registerCallback(self._frame)
        self.scan_sub=rospy.Subscriber(TOPICS['scan'],LaserScan,self._scan,queue_size=10)

    def _frame(self,rgb,depth,info):
        if len({m.header.stamp.to_nsec() for m in (rgb,depth,info)})!=1:return
        with self.lock:self.frames.append((time.monotonic(),rgb,depth,info))

    def _scan(self,message):
        with self.lock:self.scans.append((time.monotonic(),message))

    def close(self):
        for sub in self.filters+[self.scan_sub]:sub.unregister()

    def capture_live_joint(self,output_directory,heartbeat,motion_snapshot,calibration=None,*,timings=None):
        return self.capture(output_directory,calibration,heartbeat,None,
                            scope='live_joint_observation',motion_snapshot=motion_snapshot,timings=timings)

    def capture(self,output_directory,calibration,heartbeat,stopped_check,scope='paid_stopped_observation',
                *,motion_snapshot=None,timings=None):
        import rospy
        if scope not in ('paid_stopped_observation','diagnostic_not_paid','live_joint_observation'):
            raise ValueError('explicit observation scope required')
        if calibration is None and scope=='paid_stopped_observation':
            raise ValueError('paid observation requires the task/template sampling calibration')
        live=scope=='live_joint_observation'
        if live and not callable(motion_snapshot):
            raise ValueError('actual motion snapshot callback required for live joint observation')
        started=time.monotonic();deadline=started+10.;last_error='source observation unavailable'
        initial_stop=None
        if live:
            motion_before=_motion_snapshot(motion_snapshot)
            stationary_received_after=started
            stationary_source_after_ns=rospy.Time.now().to_nsec()
        while time.monotonic()<deadline and not rospy.is_shutdown():
            heartbeat()
            if not live:
                try:
                    stop=_stopped(stopped_check)
                except _StoppedObservationPending as exc:
                    initial_stop=None;last_error=str(exc);time.sleep(.02);continue
                if initial_stop is None:
                    initial_stop=stop
                    stationary_received_after=time.monotonic()
                    stationary_source_after_ns=rospy.Time.now().to_nsec()
            with self.lock:frames=list(self.frames);scans=list(self.scans)
            candidates=[r for r in frames if r[0]>=stationary_received_after and
                r[2].header.stamp.to_nsec()>=stationary_source_after_ns and
                r[2].header.stamp.to_nsec()>self.last_stamp_ns]
            if not candidates:
                time.sleep(.02);continue
            received,rgb,depth,info=candidates[-1];stamp_ns=depth.header.stamp.to_nsec()
            try:
                scan_pool=(scans if live else [r for r in scans if r[0]>=stationary_received_after
                    and r[1].header.stamp.to_nsec()>=stationary_source_after_ns])
                scan_received,scan=nearest_scan(scan_pool,stamp_ns)
                messages=dict(rgb=rgb,depth=depth,camera_info=info,scan=scan)
                for name,message in messages.items():
                    if getattr(message,'_connection_header',{}).get('callerid')!=CALLERS[name]:
                        raise ValueError('unexpected source publisher: '+name)
                    age=rospy.Time.now().to_sec()-message.header.stamp.to_sec()
                    if not -.02<=age<=.5:raise ValueError('source is stale or future: '+name)
                if not scan.ranges:raise ValueError('received LaserScan has no ranges')
                if not all(math.isfinite(x) for x in (scan.angle_min,scan.angle_increment,scan.range_min,scan.range_max)):
                    raise ValueError('invalid LaserScan angular/range metadata')
                if scan.angle_increment<=0 or not 0<=scan.range_min<scan.range_max:
                    raise ValueError('invalid LaserScan angular/range ordering')
                queries={
                    'world_from_camera':('odom',depth.header.frame_id,depth.header.stamp),
                    'map_from_camera':('map',depth.header.frame_id,depth.header.stamp),
                    'base_from_camera':('base_link',depth.header.frame_id,depth.header.stamp),
                    'odom_from_base':('odom','base_link',depth.header.stamp),
                    'map_from_base':('map','base_link',depth.header.stamp),
                    'odom_from_laser':('odom',scan.header.frame_id,scan.header.stamp),
                    'base_from_laser':('base_link',scan.header.frame_id,scan.header.stamp)}
                tf_started=time.monotonic()
                transforms={};tf_receipts={}
                for name,(target,source,stamp) in queries.items():
                    heartbeat()
                    if not live:_stopped(stopped_check)
                    tf=self.tfbuf.lookup_transform(target,source,stamp,rospy.Duration(.2))
                    transforms[name]=tf
                    tf_receipts[name]=dict(target_frame=target,source_frame=source,
                        query_stamp_ns=stamp.to_nsec(),response_header=_header(tf),
                        matrix=matrix(tf.transform).tolist(),
                        timestamp_semantics='actual historical TF response; static-only chains may have zero stamp')
                if timings is not None:
                    timings['reference_wait_and_validate_s']=tf_started-started
                    timings['reference_historical_tf_s']=time.monotonic()-tf_started
                raw=dict(rgb_raw=rgb_array(rgb),depth_m=source_depth_metres(depth),
                    intrinsic=np.asarray(info.K).reshape(3,3),
                    world_from_camera=matrix(transforms['world_from_camera'].transform),
                    map_from_camera=matrix(transforms['map_from_camera'].transform),
                    scan_ranges_raw=np.asarray(scan.ranges,dtype=np.float32),
                    scan_intensities_raw=np.asarray(scan.intensities,dtype=np.float32))
                provenance=paid_frame_source_provenance(rgb,depth,info,
                    transforms['world_from_camera'],transforms['map_from_camera'],
                    dict(rgb=raw['rgb_raw'],**{k:raw[k] for k in ('depth_m','intrinsic','world_from_camera','map_from_camera')}))
                source=dict(rgbd=dict(width=depth.width,height=depth.height,optical_frame=depth.header.frame_id,
                                     intrinsic_3x3=raw['intrinsic'].tolist()),
                    scan=dict(frame=scan.header.frame_id,range_count=len(scan.ranges),
                        angle_min_rad=scan.angle_min,angle_max_rad=scan.angle_max,
                        angle_increment_rad=scan.angle_increment,range_min_m=scan.range_min,range_max_m=scan.range_max,
                        time_increment_s=scan.time_increment,scan_time_s=scan.scan_time),
                    extrinsics={k:tf_receipts[k]['matrix'] for k in ('base_from_camera','base_from_laser')})
                heartbeat()
                if live:
                    motion_after=_motion_snapshot(motion_snapshot)
                else:
                    final_stop=_stopped(stopped_check)
                    if ('zero_since' in initial_stop and 'zero_since' in final_stop and
                            initial_stop['zero_since']!=final_stop['zero_since']):
                        raise _StoppedObservationPending('stationary interval changed during observation')
                break
            except _StoppedObservationPending as exc:
                initial_stop=None;last_error=str(exc);time.sleep(.02)
            except RuntimeError:
                raise
            except (ValueError,rospy.ROSException) as exc:
                last_error=str(exc);time.sleep(.02)
            except Exception as exc:
                # tf2 lookup exceptions vary by ROS build; no substitute TF.
                last_error=type(exc).__name__+': '+str(exc);time.sleep(.02)
        else:raise RuntimeError(('live joint' if live else 'stopped')+' observation capture incomplete: '+last_error)
        samples=(sample_reference_arrays(raw,source,calibration) if calibration is not None else
            dict(rgb=raw['rgb_raw'],depth=raw['depth_m'],ranges=raw['scan_ranges_raw']))
        directory=Path(output_directory);directory.mkdir(parents=True,exist_ok=True)
        path=directory/(str(stamp_ns)+'.npz')
        if path.exists():raise FileExistsError('observation source already archived: '+str(path))
        receipt=dict(schema='nso_real_live_joint_observation/1' if live else 'nso_real_stopped_observation/1',scope=scope,
            source_geometry=source,headers={k:_header(m) for k,m in messages.items()},
            rgbd_source_provenance=provenance,historical_tf=tf_receipts,
            rgbd_receipt_monotonic=received,scan_receipt_monotonic=scan_received,
            scan_rgbd_delta_s=(scan.header.stamp.to_nsec()-stamp_ns)/1e9,
            scan_selection='nearest actual received scan within 0.20 seconds',
            scan_angles='angle_min + original_index * angle_increment; never linspace to angle_max',
            raw_invalid_values_preserved=True,no_missing_modality_substitution=True,
            calibration_sha256=None if calibration is None else calibration_sha256(calibration),
            sampling_contract_status=('raw_source_for_joint_adapter' if live else 'requires_task_calibration')
                if calibration is None else 'explicit_calibration_applied',
            physical_extrinsic_calibration_verified=False)
        if live:
            receipt.update(motion_before=motion_before,motion_after=motion_after,
                stationary_required=False,motion_alignment='each sensor uses its own original source timestamp and historical TF')
        else:
            receipt.update(stopped_before=initial_stop,stopped_after=final_stop)
        serialized={name+'_received_ros_serialized':_serialized(m) for name,m in messages.items()}
        encoded_receipt=json.dumps(receipt,sort_keys=True,allow_nan=False)
        save_started=time.monotonic()
        np.savez_compressed(path,**raw,**samples,**serialized,frame_id=str(stamp_ns),
            capture_scope=scope,calibration_sha256='' if calibration is None else calibration_sha256(calibration),
            receipt_json=encoded_receipt,source_receipt_json=encoded_receipt,
            calibration_json=json.dumps(calibration,sort_keys=True,allow_nan=False))
        if timings is not None:
            timings['reference_npz_compress_write_s']=time.monotonic()-save_started
            timings['reference_capture_total_s']=time.monotonic()-started
        self.last_stamp_ns=stamp_ns
        return str(path)
