#!/usr/bin/python3
"""Record RTAB-Map's reported loop evidence; never publish TF or commands."""
import json
import math
import threading
import time
from collections import Counter
from pathlib import Path


def transform_record(transform):
    t, q = transform.translation, transform.rotation
    values = [t.x, t.y, t.z, q.x, q.y, q.z, q.w]
    if not all(math.isfinite(x) for x in values):
        raise ValueError('nonfinite RTAB-Map correction')
    return dict(translation=values[:3], quaternion_xyzw=values[3:])


def info_record(msg):
    return dict(stamp_ns=msg.header.stamp.to_nsec(), frame=msg.header.frame_id,
                reference_id=msg.refId, loop_closure_id=msg.loopClosureId,
                proximity_detection_id=msg.proximityDetectionId,
                loop_transform=transform_record(msg.loopClosureTransform),
                working_memory_nodes=len(msg.wmState))


class Monitor:
    def __init__(self, rospy):
        from rtabmap_msgs.msg import Info, MapGraph
        from geometry_msgs.msg import PoseWithCovarianceStamped
        from std_msgs.msg import String
        from sensor_msgs.msg import CameraInfo
        import tf2_ros
        self.rospy, self.String = rospy, String
        self.lock = threading.RLock()
        self.output = Path(rospy.get_param('~output_dir'))
        self.output.mkdir(parents=True, exist_ok=True)
        self.events = (self.output / 'events.jsonl').open('a', buffering=1)
        self.latest = dict(schema='nso.rtabmap_loop_shadow.v1', started_unix_s=time.time(),
                           mode=rospy.get_param('~mode', 'mapping'),
                           database_path=rospy.get_param('~database_path'),
                           map_frame='nso_loop_map', odom_frame='odom', base_frame='base_link',
                           publish_tf=False, navigation_authority=False, info_count=0,
                           graph_count=0, loop_event_count=0, proximity_event_count=0,
                           actual_loop_reported=False, actual_proximity_reported=False,
                           readiness='waiting_for_inputs',
                           localization_pose_is_not_by_itself_proof_of_relocalization=True)
        self.seen = set()
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.last_camera_check = 0.
        self.publisher = rospy.Publisher('/nso_loop/status', String, queue_size=1, latch=True)
        self.subscribers = [rospy.Subscriber('/nso_loop/info', Info, self.info, queue_size=10),
                            rospy.Subscriber('/nso_loop/mapGraph', MapGraph, self.graph, queue_size=1),
                            rospy.Subscriber('/nso_loop/localization_pose', PoseWithCovarianceStamped,
                                             self.localization, queue_size=1),
                            rospy.Subscriber('/zed2i/zed_node/rgb/camera_info', CameraInfo,
                                             self.camera_info, queue_size=1)]
        self.timer = rospy.Timer(rospy.Duration(1), self.publish)
        rospy.on_shutdown(self.close)

    def event(self, kind, data):
        self.events.write(json.dumps(dict(event=kind, received_unix_s=time.time(), **data),
                                     allow_nan=False) + '\n')

    def info(self, msg):
        with self.lock:
            record = info_record(msg)
            self.latest.update(info=record, last_info_unix_s=time.time())
            self.latest['info_count'] += 1
            for field, kind, counter in [('loop_closure_id', 'loop_closure_reported', 'loop_event_count'),
                                          ('proximity_detection_id', 'proximity_reported', 'proximity_event_count')]:
                key = (kind, record['reference_id'], record[field])
                if record[field] > 0 and key not in self.seen:
                    self.seen.add(key)
                    self.latest[counter] += 1
                    self.latest['actual_loop_reported' if field == 'loop_closure_id'
                                else 'actual_proximity_reported'] = True
                    self.event(kind, record)

    def graph(self, msg):
        with self.lock:
            counts = Counter(str(link.type) for link in msg.links)
            self.latest['graph_count'] += 1
            self.latest['graph'] = dict(stamp_ns=msg.header.stamp.to_nsec(), frame=msg.header.frame_id,
                nodes=len(msg.posesId), links=len(msg.links), link_type_counts=dict(counts),
                map_to_odom=transform_record(msg.mapToOdom))

    def localization(self, msg):
        with self.lock:
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            self.latest['localization'] = dict(stamp_ns=msg.header.stamp.to_nsec(),
                frame=msg.header.frame_id, position=[p.x, p.y, p.z], quaternion_xyzw=[q.x,q.y,q.z,q.w],
                covariance_diagonal=[msg.pose.covariance[i] for i in (0,7,14,21,28,35)])

    def camera_info(self, msg):
        now = time.monotonic()
        if now - self.last_camera_check < 1:
            return
        self.last_camera_check = now
        record = dict(stamp_ns=msg.header.stamp.to_nsec(), frame=msg.header.frame_id,
                      width=msg.width, height=msg.height, K=list(msg.K), D=list(msg.D),
                      distortion_model=msg.distortion_model,
                      intrinsics_valid=msg.width > 0 and msg.height > 0 and msg.K[0] > 0 and msg.K[4] > 0)
        try:
            transform = self.tf_buffer.lookup_transform('base_link', msg.header.frame_id,
                                                         msg.header.stamp, self.rospy.Duration(.1))
            record['base_from_camera'] = transform_record(transform.transform)
            record['tf_query_stamp_ns'] = msg.header.stamp.to_nsec()
        except Exception as exc:
            record['tf_error'] = str(exc)[:300]
        with self.lock:
            self.latest['camera_source'] = record

    def publish(self, _event=None):
        with self.lock:
            now = time.time()
            age = now - self.latest.get('last_info_unix_s', self.latest['started_unix_s'])
            self.latest.update(updated_unix_s=now, info_age_s=age,
                readiness=('graph_processing' if self.latest['info_count'] and self.latest['graph_count']
                           else 'waiting_for_inputs') if age < 5 else 'stale_or_no_info')
            content = json.dumps(self.latest, indent=2, allow_nan=False)
            temp = self.output / 'latest.json.tmp'
            temp.write_text(content + '\n')
            temp.replace(self.output / 'latest.json')
            self.publisher.publish(self.String(data=content))

    def close(self):
        with self.lock:
            if not self.events.closed:
                self.event('monitor_stopped', dict(info_count=self.latest['info_count'],
                    loop_event_count=self.latest['loop_event_count'],
                    proximity_event_count=self.latest['proximity_event_count']))
                self.events.close()


if __name__ == '__main__':
    import rospy
    rospy.init_node('rtabmap_loop_monitor')
    monitor = Monitor(rospy)
    rospy.spin()
