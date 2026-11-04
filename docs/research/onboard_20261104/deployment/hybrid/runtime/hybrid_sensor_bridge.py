#!/usr/bin/env python3
"""Selected lidar odometry + measured ZED depth; no dynamic TF or actuator.

The /nso_visual names and caller are legacy interface compatibility only.
Localization uses the selected lidar profile, mapping is gmapping; ZED tracking is not used.
Producer stamps, frames, poses, twists and covariances are preserved.
rospy assigns the forwarding publisher's own header.seq; it is not a source ID.
"""
import argparse
import json
import math
import os
from pathlib import Path
import threading
import time

import numpy as np
from localization_profiles import DEFAULT_LOCALIZATION_SOURCE, LOCALIZATION_SOURCES, localization_profile
from visual_sensor_bridge import (ASSETS, CAMERA_CALLER, MASTER, NODE, OPTICAL,
    DepthProjector, depth_array, finite, fresh, load_approved_mask, pose_matrix,
    require, wrap)

ODOM_CALLER = '/rf2o_laser_odometry'
LASER_CALLER = '/lslidar_driver_node'
SCHEMA = 'nso_hybrid_sensor_bridge/1'


class LidarSource:
    """Validate original producer data without restamping, fitting, or anchoring."""
    def __init__(self, localization_source=DEFAULT_LOCALIZATION_SOURCE):
        self.profile = localization_profile(localization_source)
        self.odom = dict(valid=False, reason=self.profile['waiting_reason'], source_stamp_ns=None,
                         receipt_monotonic=None, fault=None, source=self.profile['name'],
                         source_topic=self.profile['odom_topic'],
                         source_publisher=self.profile['expected_publisher'],
                         timestamp_semantics=self.profile['timestamp_semantics'],
                         velocity_source=self.profile['velocity_source'])
        self.laser = dict(valid=False, reason='waiting_for_laser', source_stamp_ns=None,
                          receipt_monotonic=None, fault=None, source='lslidar_N301',
                          source_topic='/scan', source_publisher=LASER_CALLER,
                          caller=LASER_CALLER, frame='laser_link', frame_id='laser_link')
        self.previous_pose = None

    def invalidate(self, kind, reason, latch=False):
        row = self.odom if kind == 'odom' else self.laser
        row.update(valid=False, reason=str(reason))
        if latch: row['fault'] = str(reason)

    def state(self, kind, now, wall):
        row = dict(self.odom if kind == 'odom' else self.laser)
        if row['fault']:
            row.update(valid=False, reason=row['fault'])
        elif not fresh(row['source_stamp_ns'], row['receipt_monotonic'] or 0., now, wall, .5):
            row.update(valid=False, reason=kind+'_source_stale_or_missing')
        return row

    def _check_source(self, kind, stamp_ns, publisher, frame, receipt, now, wall):
        row = self.odom if kind == 'odom' else self.laser
        if row['fault']: return False
        expected_frame = 'odom' if kind == 'odom' else 'laser_link'
        if publisher != row['source_publisher'] or frame != expected_frame:
            self.invalidate(kind, 'unexpected_'+kind+'_publisher_or_frame', True)
            return False
        if not fresh(stamp_ns, receipt, now, wall, .5):
            self.invalidate(kind, kind+'_source_stale_or_future')
            return False
        previous = row['source_stamp_ns']
        if previous is not None and stamp_ns <= previous:
            self.invalidate(kind, kind+'_stamp_regressed' if stamp_ns < previous else kind+'_duplicate',
                            latch=stamp_ns < previous)
            return False
        return True

    def add_odom(self, stamp_ns, xyz, quaternion, twist, covariance, *, publisher,
                 frame, child, receipt, now, wall):
        if not self._check_source('odom', stamp_ns, publisher, frame, receipt, now, wall): return False
        try:
            require(child == 'base_link', self.profile['producer_label']+'_child_frame_changed')
            transform = pose_matrix(xyz, quaternion)
            require(len(twist) == 6 and finite(*twist), self.profile['producer_label']+'_twist_invalid')
            require(len(covariance) == 72 and finite(*covariance), self.profile['producer_label']+'_covariance_invalid')
            rotation = transform[:3, :3]
            roll = math.atan2(rotation[2, 1], rotation[2, 2])
            pitch = math.asin(float(np.clip(-rotation[2, 0], -1., 1.)))
            require(max(abs(roll), abs(pitch)) <= math.radians(20), 'base_tilt_outside_planar_scope')
            x, y, z = map(float, xyz)
            yaw = math.atan2(rotation[1, 0], rotation[0, 0])
            if self.previous_pose is not None:
                old_ns, ox, oy, oz, oyaw = self.previous_pose
                dt = (stamp_ns-old_ns)/1e9
                distance, dyaw, dz = math.hypot(x-ox, y-oy), abs(wrap(yaw-oyaw)), abs(z-oz)
                if distance > .04+.5*dt or dyaw > .08+dt or dz > .04+.2*dt:
                    self.odom['fault_evidence'] = dict(previous_stamp_ns=old_ns,
                        rejected_stamp_ns=stamp_ns, planar_distance_m=distance,
                        absolute_yaw_delta_rad=dyaw, absolute_height_delta_m=dz,
                        previous_pose=[ox,oy,oz,oyaw], rejected_pose=[x,y,z,yaw],
                        allowed_planar_distance_m=.04+.5*dt,
                        allowed_yaw_delta_rad=.08+dt, allowed_height_delta_m=.04+.2*dt)
                    raise ValueError('lidar_pose_jump')
            self.previous_pose = (stamp_ns,x,y,z,yaw)
            self.odom.update(valid=True, reason='ok', source_stamp_ns=int(stamp_ns),
                             receipt_monotonic=float(receipt))
            return True
        except (ValueError, TypeError) as exc:
            self.invalidate('odom', str(exc), True)
            return False

    def add_scan(self, stamp_ns, ranges, *, publisher, frame, angle_min, angle_max,
                 angle_increment, range_min, range_max, receipt, now, wall):
        if not self._check_source('laser', stamp_ns, publisher, frame, receipt, now, wall): return False
        try:
            require(finite(angle_min,angle_max,angle_increment,range_min,range_max)
                    and angle_max > angle_min and angle_increment > 0.
                    and 0. <= range_min < range_max and len(ranges) > 1,
                    'laser_geometry_invalid')
            # LaserScan implementations can use inclusive or exclusive upper bins.
            require(abs((len(ranges)-1)*angle_increment-(angle_max-angle_min))
                    <= 1.1*angle_increment, 'laser_bin_geometry_inconsistent')
            finite_returns = sum(math.isfinite(float(v)) and range_min <= v <= range_max for v in ranges)
            require(finite_returns > 0, 'laser_without_finite_returns')
            self.laser.update(valid=True, reason='ok', source_stamp_ns=int(stamp_ns),
                              receipt_monotonic=float(receipt), finite_returns=int(finite_returns),
                              range_count=len(ranges))
            return True
        except (ValueError, TypeError) as exc:
            self.invalidate('laser', str(exc), True)
            return False


def run(args):
    require(os.environ.get('ROS_MASTER_URI') == MASTER, 'use_master_11382')
    import rospy, rosgraph, message_filters
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import CameraInfo, Image, LaserScan
    from std_msgs.msg import Header, String
    owned = ['/nso_visual/'+name for name in ('odom','scan','odom_health',
             'depth_health','health','depth_filtered','laser_health')]
    publishers, _, _ = rosgraph.Master('/nso_hybrid_preflight').getSystemState()
    for topic, callers in publishers:
        require(topic not in owned or not callers, 'output_topic_already_owned:'+topic)
    mask, mask_receipt = load_approved_mask(args.mask, args.mask_provenance)
    sources = LidarSource(args.localization_source)
    # Actual partial RGB-D is an observation, including a frame with no finite
    # returns. LiDAR supplies navigation geometry; do not add a depth-count
    # planning gate. Missing depths stay missing and scans never clear space.
    projector = DepthProjector(mask, minimum_valid_points=0,
                               minimum_support_rows=0, minimum_support_columns=0)
    lock = threading.RLock()
    depth_state = dict(valid=False, reason='waiting_for_depth', source_stamp_ns=None,
                       receipt_monotonic=None, fault=None, source='ZED_RGBD')
    errors = []
    rospy.init_node(NODE.strip('/'), anonymous=False)
    require(not rospy.get_param('/use_sim_time',False), 'simulated_time_not_supported')
    for key in ('publish_tf','publish_map_tf'):
        require(rospy.get_param('/zed2i/zed_node/pos_tracking/'+key,None) is False,
                'ZED_dynamic_TF_must_be_disabled')
    odom_pub = rospy.Publisher('/nso_visual/odom',Odometry,queue_size=1)
    scan_pub = rospy.Publisher('/nso_visual/scan',LaserScan,queue_size=1)
    odom_health = rospy.Publisher('/nso_visual/odom_health',Header,queue_size=1)
    depth_health = rospy.Publisher('/nso_visual/depth_health',Header,queue_size=1)
    health_pub = rospy.Publisher('/nso_visual/health',String,queue_size=1)
    laser_health = rospy.Publisher('/nso_visual/laser_health',String,queue_size=1)
    filtered_pub = rospy.Publisher('/nso_visual/depth_filtered',Image,queue_size=1)
    def caller(message): return getattr(message,'_connection_header',{}).get('callerid')
    def odometry(message):
        p,q,t = message.pose.pose.position,message.pose.pose.orientation,message.twist.twist
        now,wall = time.monotonic(),rospy.Time.now().to_sec()
        with lock:
            good = sources.add_odom(message.header.stamp.to_nsec(),[p.x,p.y,p.z],
                [q.x,q.y,q.z,q.w],[t.linear.x,t.linear.y,t.linear.z,t.angular.x,t.angular.y,t.angular.z],
                list(message.pose.covariance)+list(message.twist.covariance),publisher=caller(message),
                frame=message.header.frame_id,child=message.child_frame_id,receipt=now,now=now,wall=wall)
            if good:
                odom_pub.publish(message)  # Source payload/stamp preserved; rospy assigns header.seq.
                odom_health.publish(message.header)
    def laser(message):
        now,wall = time.monotonic(),rospy.Time.now().to_sec()
        with lock:
            sources.add_scan(message.header.stamp.to_nsec(),message.ranges,publisher=caller(message),
                frame=message.header.frame_id,angle_min=message.angle_min,angle_max=message.angle_max,
                angle_increment=message.angle_increment,range_min=message.range_min,range_max=message.range_max,
                receipt=now,now=now,wall=wall)
    def depth(message,info):
        try:
            receipt,wall = time.monotonic(),rospy.Time.now().to_sec()
            require(caller(message)==caller(info)==CAMERA_CALLER,'unexpected_depth_publisher')
            ns = message.header.stamp.to_nsec()
            require(info.header.stamp.to_nsec()==ns,'depth_info_stamp_mismatch')
            require(message.header.frame_id==info.header.frame_id==OPTICAL,'depth_info_frame_mismatch')
            require(message.width==info.width and message.height==info.height,'depth_info_dimensions_differ')
            require(fresh(ns,receipt,receipt,wall,.5),'depth_stale_or_future')
            with lock:
                previous=depth_state['source_stamp_ns']
                require(previous is None or ns>previous,'depth_duplicate_or_regressed')
            pixels=depth_array(message.data,message.width,message.height,message.step,
                               message.encoding,message.is_bigendian)
            projected=projector.project(pixels,info.K,message.header.frame_id)
            with lock:
                depth_state.update(valid=projected['valid'],reason=projected['reason'],
                    source_stamp_ns=ns,receipt_monotonic=receipt,statistics=projected['statistics'])
                if not projected['valid']: return
                require(fresh(ns,receipt,time.monotonic(),rospy.Time.now().to_sec(),.5),
                        'depth_expired_during_projection')
                scan=LaserScan();scan.header=Header(seq=message.header.seq,stamp=message.header.stamp,frame_id='base_link')
                scan.angle_min,scan.angle_max,scan.angle_increment=projector.angle_min,projector.angle_max,projector.angle_increment
                scan.time_increment,scan.scan_time=0.,.1
                scan.range_min,scan.range_max=projector.range_min,projector.range_max
                scan.ranges=projected['ranges'].tolist()
                filtered=Image();filtered.header=message.header
                filtered.height,filtered.width=message.height,message.width
                filtered.encoding,filtered.is_bigendian,filtered.step='32FC1',False,message.width*4
                filtered.data=projected['filtered_depth'].astype('<f4').tobytes()
                scan_pub.publish(scan);filtered_pub.publish(filtered);depth_health.publish(message.header)
        except Exception as exc:
            with lock:
                depth_state.update(valid=False,reason=str(exc));errors.append(str(exc));del errors[:-20]
            rospy.logwarn_throttle(2.,str(exc))
    subscribers=[rospy.Subscriber(sources.profile['odom_topic'],Odometry,odometry,queue_size=2,tcp_nodelay=True),
                 rospy.Subscriber('/scan',LaserScan,laser,queue_size=2,tcp_nodelay=True)]
    depth_sub=message_filters.Subscriber('/zed2i/zed_node/depth/depth_registered',Image,queue_size=1,buff_size=8*1024*1024)
    info_sub=message_filters.Subscriber('/zed2i/zed_node/depth/camera_info',CameraInfo,queue_size=5)
    synchronizer=message_filters.TimeSynchronizer([depth_sub,info_sub],5);synchronizer.registerCallback(depth)
    rate=rospy.Rate(10)
    while not rospy.is_shutdown():
        with lock:
            now,wall=time.monotonic(),rospy.Time.now().to_sec()
            odom_state=sources.state('odom',now,wall);laser_state=sources.state('laser',now,wall)
            current_depth=dict(depth_state)
            if not fresh(current_depth['source_stamp_ns'],current_depth['receipt_monotonic'] or 0.,now,wall,.5):
                current_depth.update(valid=False,reason='depth_stale_or_missing')
            health=dict(schema=SCHEMA,navigation_source='lidar_rgbd',sampled_unix_s=wall,
                sampled_monotonic=now,odom=odom_state,depth=current_depth,laser_source=laser_state,
                self_mask=mask_receipt,recent_errors=list(errors),no_actuator=True,
                localization_source=sources.profile['name'],mapping_source='gmapping',
                localization_tf_owner=sources.profile['tf_owner'],
                velocity_source=sources.profile['velocity_source'],dynamic_TF_published=False,
                legacy_topic_namespace_only=True,physical_extrinsic_calibration_verified=False,
                covariance_is_measured_accuracy=False,depth_scan_clearing_allowed=False)
            health_pub.publish(String(data=json.dumps(health,allow_nan=False,separators=(',',':'))))
            laser_health.publish(String(data=json.dumps(laser_state,allow_nan=False,separators=(',',':'))))
        rate.sleep()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--localization-source',choices=LOCALIZATION_SOURCES,default=DEFAULT_LOCALIZATION_SOURCE)
    parser.add_argument('--mask',default=str(ASSETS/'fixed-imu-mask.npy'))
    parser.add_argument('--mask-provenance',default=str(ASSETS/'fixed-imu-mask-provenance.json'))
    args=parser.parse_args(argv)
    if args.run: return run(args)
    _,receipt=load_approved_mask(args.mask,args.mask_provenance)
    print(json.dumps(dict(status='validated_not_started',no_actuator=True,ROS_started=False,
        master=MASTER,navigation_source='lidar_rgbd',localization_source=args.localization_source,
        localization_tf_owner=localization_profile(args.localization_source)['tf_owner'],
        self_mask=receipt,dynamic_TF_published=False)))


if __name__=='__main__': main()
