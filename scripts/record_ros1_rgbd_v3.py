#!/usr/bin/env python3
"""Version 3 ROS1 recorder: raw depth/scans, projection and TF provenance.

Publishes no commands. Run with the robot ROS Python; preserves recording/2.
"""
import json
import hashlib
import threading
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils.rgbd_contract import RGBDFrame,PlanarScan
from scripts.ros1_recording_v3_contract import convert_depth,convert_ranges,extended_metadata,transform_response


def main():
    import rospy
    import message_filters
    import tf2_ros
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image,CameraInfo,LaserScan
    rospy.init_node('nso_record_rgbd_v3',anonymous=False)
    output=Path(rospy.get_param('~output')).expanduser().resolve()
    output.mkdir(parents=True,exist_ok=False);(output/'frames').mkdir();(output/'scans').mkdir();(output/'metadata').mkdir();(output/'raw').mkdir()
    config=dict(world_frame=rospy.get_param('~world_frame','map'),
        optical_frame=rospy.get_param('~camera_optical_frame',''),
        depth_topic=rospy.get_param('~depth_topic','/zed/zed_node/depth/depth_registered'),
        rgb_topic=rospy.get_param('~rgb_topic','/zed/zed_node/left/image_rect_color'),
        info_topic=rospy.get_param('~camera_info_topic','/zed/zed_node/depth/camera_info'),
        scan_topic=rospy.get_param('~scan_topic','/scan'),
        semantic_topic=rospy.get_param('~semantic_topic',''),
        max_depth_m=float(rospy.get_param('~max_depth_m',4.)),
        slop_s=float(rospy.get_param('~sync_slop_s',.05)),
        positive_inf_policy=rospy.get_param('~positive_inf_policy','invalid'),
        min_interval_s=float(rospy.get_param('~min_interval_s',.2)))
    bridge=CvBridge();buffer=tf2_ros.Buffer(cache_time=rospy.Duration(30))
    listener=tf2_ros.TransformListener(buffer)
    state=dict(info=None,count=0,rejected=0,last_timestamp=-float('inf'))
    def info_callback(message):state['info']=message
    info_subscriber=rospy.Subscriber(config['info_topic'],CameraInfo,info_callback,queue_size=1)
    lock=threading.RLock()
    def pose(frame_id,stamp):
        if stamp.secs==0 and stamp.nsecs==0:
            raise ValueError('zero sensor time requests latest TF; historical sensor stamp required')
        response=buffer.lookup_transform(config['world_frame'],frame_id,stamp,rospy.Duration(.1))
        return transform_response(response,config['world_frame'],frame_id)
    def callback(depth_msg,rgb_msg,scan_msg,*semantic_messages):
        stamp=depth_msg.header.stamp;timestamp=stamp.to_sec()
        if timestamp-state['last_timestamp']<config['min_interval_s']:return
        if state['info'] is None:return
        try:
            raw_depth=np.asarray(bridge.imgmsg_to_cv2(depth_msg,desired_encoding='passthrough')).copy()
            depth=convert_depth(raw_depth,depth_msg.encoding,config['max_depth_m'])
            rgb=np.asarray(bridge.imgmsg_to_cv2(rgb_msg,desired_encoding='rgb8'),dtype=np.uint8)
            info=state['info']
            optical=config['optical_frame'] or depth_msg.header.frame_id
            if not config['optical_frame'] and 'optical' not in optical:
                raise ValueError('set camera_optical_frame to verified optical TF frame; depth header is not explicit')
            semantic=np.zeros(depth.shape,np.uint8)
            if semantic_messages:
                semantic=np.asarray(bridge.imgmsg_to_cv2(semantic_messages[0],desired_encoding='mono8'),np.uint8)
            metadata,intrinsic,source_from_rectified=extended_metadata(depth_msg,rgb_msg,scan_msg,info,
                semantic_messages[0] if semantic_messages else None,config,depth.shape)
            world_from_source,camera_response=pose(optical,stamp)
            world_from_laser,laser_response=pose(scan_msg.header.frame_id,scan_msg.header.stamp)
            frame=RGBDFrame(timestamp,depth,rgb,intrinsic,world_from_source@source_from_rectified,semantic).validate()
            raw_ranges=np.asarray(scan_msg.ranges,np.float32).copy()
            ranges,conversion=convert_ranges(raw_ranges,scan_msg.range_min,scan_msg.range_max,config['positive_inf_policy'])
            scan=PlanarScan(scan_msg.header.stamp.to_sec(),ranges,scan_msg.angle_min,
                scan_msg.angle_increment,scan_msg.range_max,world_from_laser)
            metadata['range_conversion']=conversion
            metadata['tf_responses']=dict(camera=camera_response,laser=laser_response)
            metadata['raw_array_scope']='cv_bridge passthrough depth and original float32 LaserScan ranges/intensities; original bag still required'
            json.dumps(metadata,allow_nan=False)
            name='%05d.npz'%state['count'];frame.save(output/'frames'/name);scan.save(output/'scans'/name)
            np.savez_compressed(output/'raw'/name,depth_source=raw_depth,ranges_source=raw_ranges,
                intensities_source=np.asarray(scan_msg.intensities,np.float32))
            metadata['payload_sha256']={str(Path(kind)/name):hashlib.sha256((output/kind/name).read_bytes()).hexdigest()
                                        for kind in ('frames','scans','raw')}
            (output/'metadata'/('%05d.json'%state['count'])).write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
            state['count']+=1;state['last_timestamp']=timestamp
            save_manifest()
        except Exception as exc:
            state['rejected']+=1;rospy.logwarn_throttle(5,'NSO frame rejected: '+str(exc))
    def synchronized_callback(*messages):
        with lock:callback(*messages)
    subscribers=[message_filters.Subscriber(config['depth_topic'],Image),
                 message_filters.Subscriber(config['rgb_topic'],Image),
                 message_filters.Subscriber(config['scan_topic'],LaserScan)]
    if config['semantic_topic']:subscribers.append(message_filters.Subscriber(config['semantic_topic'],Image))
    def save_manifest():
        temporary=output/'sequence.json.tmp'
        temporary.write_text(json.dumps(dict(config,frames=state['count'],rejected=state['rejected'],
            schema_version='ros1_rgbd_recording/3',source_metadata='metadata/<frame_index>.json',
            camera_info_association='latest received, not timestamp-synchronized',
            contract='rectified depth axial metres; pose=world_from_source_optical times R transpose; scan at own timestamp',
            semantics='none' if not config['semantic_topic'] else 'external visible label image',
            publishes_motion=False),indent=2)+'\n')
        temporary.replace(output/'sequence.json')
    def shutdown():
        with lock:save_manifest()
    save_manifest()
    rospy.on_shutdown(shutdown)
    rospy.loginfo('Recording synchronized sensor frames; no motion commands are published.')
    sync=message_filters.ApproximateTimeSynchronizer(subscribers,30,config['slop_s']);sync.registerCallback(synchronized_callback)
    rospy.spin()


if __name__=='__main__':main()
