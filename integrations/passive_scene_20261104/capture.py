#!/usr/bin/env python3
"""One receive-only diagnostic snapshot; never a paid trial observation."""
import json
import os
from pathlib import Path
import threading
import time

import cv2
from cv_bridge import CvBridge
import message_filters
import numpy as np
import rospy
from sensor_msgs.msg import Image, CameraInfo, LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import tf2_ros
from tf.transformations import quaternion_matrix

assert os.environ['ROS_MASTER_URI'] == 'http://127.0.0.1:11382'
OUT = Path(__file__).parent / ('snapshot_' + time.strftime('%H%M%S'))
OUT.mkdir(exist_ok=False)
rospy.init_node('nso_passive_scene_diagnostic', anonymous=True)
bridge = CvBridge()
buf = tf2_ros.Buffer(cache_time=rospy.Duration(15))
listener = tf2_ros.TransformListener(buf)
ready = threading.Event()
lock = threading.Lock()
current = {}
bundle = []
pixel_samples = []

def cache(key, msg):
    with lock:
        current[key] = msg

def pair(rgb, depth, info):
    # Keep one aligned keyframe only; subsequent data are a bounded scalar probe.
    with lock:
        if not bundle:
            bundle.extend((rgb, depth, info))
            ready.set()
        if len(pixel_samples) < 50:
            a = bridge.imgmsg_to_cv2(depth, desired_encoding='passthrough')
            y, x = 608, 664
            if a.shape[0] > y and a.shape[1] > x:
                z = float(a[y, x])
                window = a[y-2:y+3, x-2:x+3]
                finite = window[np.isfinite(window) & (window > 0)]
                pixel_samples.append(dict(stamp_ns=depth.header.stamp.to_nsec(),
                    depth_m=z if np.isfinite(z) else None,
                    patch_finite_count=int(finite.size),
                    patch_median_m=float(np.median(finite)) if finite.size else None))

subs = [rospy.Subscriber('/scan', LaserScan, lambda m: cache('scan', m), queue_size=1),
        rospy.Subscriber('/nso_visual/odom', Odometry, lambda m: cache('odom', m), queue_size=1),
        rospy.Subscriber('/nso_visual/health', String, lambda m: cache('health', m), queue_size=1)]
sync_subs = [message_filters.Subscriber('/zed2i/zed_node/left/image_rect_color', Image),
             message_filters.Subscriber('/zed2i/zed_node/depth/depth_registered', Image),
             message_filters.Subscriber('/zed2i/zed_node/depth/camera_info', CameraInfo)]
sync = message_filters.TimeSynchronizer(sync_subs, 10)
sync.registerCallback(pair)
result = dict(scope='receive-only diagnostic, no CAN/goal/velocity publication; not an experiment continuation',
              started_unix_s=time.time(), source_master=os.environ['ROS_MASTER_URI'])
try:
    if not ready.wait(12):
        raise RuntimeError('no exact synchronized RGB-depth-info packet in 12 seconds')
    # One bounded five-second scalar probe at the old conflict pixel, no video.
    rospy.sleep(5)
    with lock:
        rgb, depth, info = bundle
        samples = list(pixel_samples)
        status = dict(current)
    for s in sync_subs:
        s.sub.unregister()
    raw = bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8')
    a = bridge.imgmsg_to_cv2(depth, desired_encoding='passthrough')
    cv2.imwrite(str(OUT/'rgb.jpg'), raw)
    arrays = dict(depth_m=a, intrinsic=np.asarray(info.K).reshape(3, 3))
    def matrix(t):
        q = t.transform.rotation
        m = quaternion_matrix([q.x,q.y,q.z,q.w])
        v = t.transform.translation
        m[:3,3] = [v.x,v.y,v.z]
        return m
    transforms = {}
    for source, stamp in [(depth.header.frame_id, depth.header.stamp), ('base_link', rospy.Time(0))]:
        t = buf.lookup_transform('odom', source, stamp, rospy.Duration(2))
        m = matrix(t)
        transforms[source] = dict(stamp_ns=t.header.stamp.to_nsec(), matrix=m.tolist())
        arrays['odom_from_' + source.replace('/', '_')] = m
    scan = status.get('scan')
    if scan is not None:
        arrays['ranges'] = np.asarray(scan.ranges, dtype=np.float32)
        result['scan'] = dict(stamp_ns=scan.header.stamp.to_nsec(), frame=scan.header.frame_id,
            angle_min=scan.angle_min, angle_increment=scan.angle_increment,
            range_min=scan.range_min, range_max=scan.range_max)
    np.savez_compressed(OUT/'sensors.npz', **arrays)
    odom = status.get('odom')
    if odom:
        p=odom.pose.pose.position;q=odom.pose.pose.orientation
        result['odom'] = dict(stamp_ns=odom.header.stamp.to_nsec(), frame=odom.header.frame_id,
            position=[p.x,p.y,p.z], quaternion=[q.x,q.y,q.z,q.w],
            twist_linear_x=odom.twist.twist.linear.x, twist_angular_z=odom.twist.twist.angular.z)
    if status.get('health'):
        result['health']=json.loads(status['health'].data)
    result.update(status='captured', stamp_ns=depth.header.stamp.to_nsec(),
        source_frame=depth.header.frame_id, rgb_stamp_ns=rgb.header.stamp.to_nsec(),
        info_stamp_ns=info.header.stamp.to_nsec(), shape=list(a.shape),
        transforms=transforms, old_pixel_yx=[608,664], old_pixel_samples=samples,
        pixel_identity_assumed=False,
        note='The old pixel may correspond to a different world point if the pose or scene changed. No source mask, map or policy modified.')
except Exception as exc:
    result.update(status='failed', error=repr(exc))
finally:
    result['finished_unix_s']=time.time()
    (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    rospy.signal_shutdown('receive-only diagnostic complete')
    print(json.dumps(dict(output=str(OUT), status=result['status'], error=result.get('error'))))
