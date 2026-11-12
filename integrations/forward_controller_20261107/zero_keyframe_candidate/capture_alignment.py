"""Capture a same-scan fixed coordinate alignment; never publish robot state."""
import argparse
import json
import math
from pathlib import Path
import threading
import time
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
parser.add_argument('--source-topic', default='/nso_csm_zero_keyframe/pose_stamped')
parser.add_argument('--expected-caller', default='/nso_csm_zero_keyframe/matcher')
parser.add_argument('--source-frame', default='odom_zero_keyframe_20261107')
args = parser.parse_args()
rospy.init_node('nso_capture_csm_frame_alignment', anonymous=True, disable_signals=True)
lock = threading.Lock()
rows = {'source': {}, 'odom': {}}
SOURCE = args.source_topic
CALLER = args.expected_caller
FRAME = args.source_frame

def receive(name, msg):
    caller = msg._connection_header.get('callerid')
    assert caller == (CALLER if name == 'source' else '/nso_csm_odometry_bridge')
    assert msg.header.frame_id == (FRAME if name == 'source' else 'odom')
    pose = msg.pose if name == 'source' else msg.pose.pose
    p, q = pose.position, pose.orientation
    assert abs(q.x) < 1e-8 and abs(q.y) < 1e-8 and abs(p.z) < 1e-8
    yaw = math.atan2(2*q.w*q.z, 1-2*q.z*q.z)
    sample = dict(pose=[p.x, p.y, yaw], receipt_wall_ns=time.time_ns(), caller=caller)
    with lock:
        store = rows[name]
        store[msg.header.stamp.to_nsec()] = sample
        while len(store) > 100:
            del store[min(store)]

subs = [rospy.Subscriber(SOURCE, PoseStamped, lambda m: receive('source', m), queue_size=20),
        rospy.Subscriber('/nso_csm/odom', Odometry, lambda m: receive('odom', m), queue_size=20)]
try:
    time.sleep(2)
    with lock:
        common = sorted(set(rows['source']) & set(rows['odom']))
        assert len(common) >= 5, 'No sufficient same-scan observations; do not infer an alignment'
        stamp = common[-1]
        source = rows['source'][stamp]['pose']
        odom = rows['odom'][stamp]['pose']
    assert 0 <= time.time_ns()-stamp < 250_000_000, 'Alignment source is stale'
    velocity = rospy.wait_for_message('/nso_csm/control_odom', Odometry, timeout=2)
    assert abs(velocity.twist.twist.linear.x) < .001 and abs(velocity.twist.twist.angular.z) < .001, 'Handover requires a stationary robot'
    yaw = math.atan2(math.sin(odom[2]-source[2]), math.cos(odom[2]-source[2]))
    c, s = math.cos(yaw), math.sin(yaw)
    offset = [odom[0]-c*source[0]+s*source[1], odom[1]-s*source[0]-c*source[1], yaw]
    residuals = []
    for ns in common:
        a, b = rows['source'][ns]['pose'], rows['odom'][ns]['pose']
        residuals.append(math.hypot(offset[0]+c*a[0]-s*a[1]-b[0],
                                    offset[1]+s*a[0]+c*a[1]-b[1]))
    result = dict(schema='csm_fixed_se2_alignment/1', source_topic=SOURCE,
                  source_frame=FRAME, expected_caller=CALLER, target_frame='odom',
                  odom_from_source=offset,
                  anchor=dict(source_stamp_ns=stamp, odom_stamp_ns=stamp,
                              source_pose=source, odom_pose=odom),
                  captured_wall_ns=time.time_ns(), matched_scan_count=len(common),
                  prior_matched_scan_max_position_difference_m=max(residuals),
                  scope='One constant SE2 coordinate transform from simultaneous '
                        'measured source poses; preserves the current odom '
                        'coordinate epoch, does not establish physical accuracy. '
                        'No filtering, output clipping, or updated source timestamp.')
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
finally:
    for sub in subs: sub.unregister()
    rospy.signal_shutdown('alignment captured; no publishers created')
