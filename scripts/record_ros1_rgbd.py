#!/usr/bin/env python3
"""ROS1 synchronized recorder only. Publishes no motion commands.

Run with the robot's ROS Python (e.g. Noetic Python 3.8), not .venv-3d.
Requires rospy, tf2_ros, cv_bridge, message_filters and NumPy on the robot.
"""
import json
import hashlib
import math
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils.rgbd_contract import RGBDFrame,PlanarScan


def sensor_metadata(depth, rgb, scan, info, semantic, config):
    """Preserve exact source timestamps; validate the synchronized message span.

    CameraInfo may be latched/static: record its stamp but do not pretend it was
    synchronized with the images. This function requires no ROS installation.
    """
    def header(message):
        stamp = message.header.stamp
        seconds, nanoseconds = int(stamp.secs), int(stamp.nsecs)
        if seconds < 0 or not 0 <= nanoseconds < 1_000_000_000:
            raise ValueError('invalid source ROS timestamp')
        return dict(secs=seconds, nsecs=nanoseconds, timestamp_ns=seconds * 1_000_000_000 + nanoseconds,
                    frame_id=str(message.header.frame_id), sequence=int(message.header.seq))
    headers = {name: header(message) for name, message in
               (('depth', depth), ('rgb', rgb), ('scan', scan), ('camera_info', info))}
    if semantic is not None: headers['semantic'] = header(semantic)
    slop = float(config['slop_s'])
    if not math.isfinite(slop) or slop < 0: raise ValueError('invalid synchronization tolerance')
    stamps = [row['timestamp_ns'] for name, row in headers.items() if name != 'camera_info']
    span = max(stamps) - min(stamps)
    if span > round(slop * 1_000_000_000):
        raise ValueError('source RGB/depth/scan/semantic timestamp span exceeds tolerance')
    timing = [float(scan.time_increment), float(scan.scan_time)]
    if not all(math.isfinite(v) and v >= 0 for v in timing):
        raise ValueError('invalid laser acquisition timing')
    result = dict(schema_version='ros1_rgbd_source_metadata/1', headers=headers,
        synchronized_message_span_ns=span,
        depth_relative_offsets_ns={name: row['timestamp_ns'] - headers['depth']['timestamp_ns']
                                  for name, row in headers.items()},
        image_encodings=dict(depth=str(depth.encoding), rgb=str(rgb.encoding),
                             semantic=None if semantic is None else str(semantic.encoding)),
        camera_info=dict(height=int(info.height), width=int(info.width),
                         K=list(info.K), P=list(info.P), R=list(info.R), D=list(info.D),
                         distortion_model=str(info.distortion_model),
                         association='latest received, not timestamp-synchronized'),
        laser=dict(time_increment_s=timing[0], scan_time_s=timing[1],
                   beam_count=len(scan.ranges), angle_min_rad=float(scan.angle_min),
                   angle_max_rad=float(scan.angle_max), angle_increment_rad=float(scan.angle_increment),
                   range_min_m=float(scan.range_min), range_max_m=float(scan.range_max),
                   motion_compensation_applied=False),
        tf_queries=dict(target_frame=config['world_frame'],
                        camera_source_frame=config['optical_frame'] or depth.header.frame_id,
                        camera_stamp_ns=headers['depth']['timestamp_ns'],
                        laser_source_frame=scan.header.frame_id,
                        laser_stamp_ns=headers['scan']['timestamp_ns']),
        calibration_correctness_verified=False, raw_rosbag_still_required=True)
    json.dumps(result, allow_nan=False)
    return result


def main():
    import rospy
    import message_filters
    import tf2_ros
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image,CameraInfo,LaserScan
    from tf.transformations import quaternion_matrix
    rospy.init_node('nso_record_rgbd',anonymous=False)
    output=Path(rospy.get_param('~output')).expanduser().resolve()
    output.mkdir(parents=True,exist_ok=False);(output/'frames').mkdir();(output/'scans').mkdir();(output/'metadata').mkdir()
    config=dict(world_frame=rospy.get_param('~world_frame','map'),
        optical_frame=rospy.get_param('~camera_optical_frame',''),
        depth_topic=rospy.get_param('~depth_topic','/zed/zed_node/depth/depth_registered'),
        rgb_topic=rospy.get_param('~rgb_topic','/zed/zed_node/left/image_rect_color'),
        info_topic=rospy.get_param('~camera_info_topic','/zed/zed_node/depth/camera_info'),
        scan_topic=rospy.get_param('~scan_topic','/scan'),
        semantic_topic=rospy.get_param('~semantic_topic',''),
        max_depth_m=float(rospy.get_param('~max_depth_m',4.)),
        slop_s=float(rospy.get_param('~sync_slop_s',.05)),
        min_interval_s=float(rospy.get_param('~min_interval_s',.2)))
    bridge=CvBridge();buffer=tf2_ros.Buffer(cache_time=rospy.Duration(30))
    listener=tf2_ros.TransformListener(buffer)
    state=dict(info=None,count=0,rejected=0,last_timestamp=-float('inf'))
    def info_callback(message):state['info']=message
    info_subscriber=rospy.Subscriber(config['info_topic'],CameraInfo,info_callback,queue_size=1)
    def pose(frame_id,stamp):
        t=buffer.lookup_transform(config['world_frame'],frame_id,stamp,rospy.Duration(.1)).transform
        matrix=quaternion_matrix([t.rotation.x,t.rotation.y,t.rotation.z,t.rotation.w])
        matrix[:3,3]=[t.translation.x,t.translation.y,t.translation.z]
        return matrix
    def callback(depth_msg,rgb_msg,scan_msg,*semantic_messages):
        stamp=depth_msg.header.stamp;timestamp=stamp.to_sec()
        if timestamp-state['last_timestamp']<config['min_interval_s']:return
        if state['info'] is None:return
        try:
            depth=np.asarray(bridge.imgmsg_to_cv2(depth_msg,desired_encoding='passthrough'))
            if depth_msg.encoding=='16UC1':depth=depth.astype(np.float32)*.001
            elif depth_msg.encoding=='32FC1':depth=depth.astype(np.float32)
            else:raise ValueError('depth encoding must be 32FC1 metres or 16UC1 millimetres')
            depth=np.where(np.isfinite(depth)&(depth>0)&(depth<=config['max_depth_m']),depth,0.).astype(np.float32)
            rgb=np.asarray(bridge.imgmsg_to_cv2(rgb_msg,desired_encoding='rgb8'),dtype=np.uint8)
            info=state['info']
            if (info.height,info.width)!=depth.shape:raise ValueError('CameraInfo resolution does not match depth')
            intrinsic=np.asarray(info.P).reshape(3,4)[:,:3].copy()
            if intrinsic[0,0]<=0:intrinsic=np.asarray(info.K).reshape(3,3)
            optical=config['optical_frame'] or depth_msg.header.frame_id
            if not config['optical_frame'] and 'optical' not in optical:
                raise ValueError('set camera_optical_frame to verified optical TF frame; depth header is not explicit')
            semantic=np.zeros(depth.shape,np.uint8)
            if semantic_messages:
                semantic=np.asarray(bridge.imgmsg_to_cv2(semantic_messages[0],desired_encoding='mono8'),np.uint8)
            metadata=sensor_metadata(depth_msg,rgb_msg,scan_msg,info,
                semantic_messages[0] if semantic_messages else None,config)
            frame=RGBDFrame(timestamp,depth,rgb,intrinsic,pose(optical,stamp),semantic).validate()
            ranges=np.asarray(scan_msg.ranges,np.float32)
            ranges=np.where(np.isposinf(ranges),scan_msg.range_max,ranges)
            ranges=np.where(np.isfinite(ranges)&(ranges>=scan_msg.range_min),
                            np.minimum(ranges,scan_msg.range_max),0.).astype(np.float32)
            scan=PlanarScan(scan_msg.header.stamp.to_sec(),ranges,scan_msg.angle_min,
                scan_msg.angle_increment,scan_msg.range_max,pose(scan_msg.header.frame_id,scan_msg.header.stamp))
            name='%05d.npz'%state['count'];frame.save(output/'frames'/name);scan.save(output/'scans'/name)
            metadata['payload_sha256']={str(Path(kind)/name):hashlib.sha256((output/kind/name).read_bytes()).hexdigest()
                                        for kind in ('frames','scans')}
            (output/'metadata'/('%05d.json'%state['count'])).write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
            state['count']+=1;state['last_timestamp']=timestamp
        except Exception as exc:
            state['rejected']+=1;rospy.logwarn_throttle(5,'NSO frame rejected: '+str(exc))
    subscribers=[message_filters.Subscriber(config['depth_topic'],Image),
                 message_filters.Subscriber(config['rgb_topic'],Image),
                 message_filters.Subscriber(config['scan_topic'],LaserScan)]
    if config['semantic_topic']:subscribers.append(message_filters.Subscriber(config['semantic_topic'],Image))
    sync=message_filters.ApproximateTimeSynchronizer(subscribers,30,config['slop_s']);sync.registerCallback(callback)
    def save_manifest():
        (output/'sequence.json').write_text(json.dumps(dict(config,frames=state['count'],rejected=state['rejected'],
            schema_version='ros1_rgbd_recording/2',source_metadata='metadata/<frame_index>.json',
            camera_info_association='latest received, not timestamp-synchronized',
            contract='depth axial metres; optical xyz right/down/forward; world_from_camera; scan at own timestamp',
            semantics='none' if not config['semantic_topic'] else 'external visible label image',
            publishes_motion=False),indent=2)+'\n')
    rospy.on_shutdown(save_manifest)
    rospy.loginfo('Recording synchronized sensor frames; no motion commands are published.')
    rospy.spin()


if __name__=='__main__':main()
