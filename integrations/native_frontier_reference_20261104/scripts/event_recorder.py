#!/usr/bin/env python3
"""Passive action events only: no goal, cancel, velocity or navigation publication."""
import json
import os
import threading
import time
import uuid

import rospy
from actionlib_msgs.msg import GoalID, GoalStatusArray
from move_base_msgs.msg import MoveBaseActionGoal, MoveBaseActionResult
from std_msgs.msg import String


def stamp(value):
    return {"secs": int(value.secs), "nsecs": int(value.nsecs)}


def callerid(msg):
    return (getattr(msg, '_connection_header', None) or {}).get('callerid')


class Recorder:
    def __init__(self):
        self.lock = threading.Lock()
        self.status = {}
        self.closed = False
        self.session_id = rospy.get_param("~session_id", "").strip() or str(uuid.uuid4())
        self.action_namespace = rospy.get_param('~action_namespace').rstrip('/')
        self.protection_topic = rospy.get_param('~protection_topic', '').strip()
        path = os.path.abspath(os.path.expanduser(rospy.get_param('~event_log')))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.file = open(path, 'a', encoding='utf-8', buffering=1)
        self.write('recorder_start', action_namespace=self.action_namespace,
                   protection_topic=self.protection_topic or None,
                   provenance='passive; observed action namespace may include non-explore clients')
        self.subscribers = [
            rospy.Subscriber(self.action_namespace + '/goal', MoveBaseActionGoal, self.goal, queue_size=50),
            rospy.Subscriber(self.action_namespace + '/result', MoveBaseActionResult, self.result, queue_size=50),
            rospy.Subscriber(self.action_namespace + '/cancel', GoalID, self.cancel, queue_size=50),
            rospy.Subscriber(self.action_namespace + '/status', GoalStatusArray, self.status_update, queue_size=50),
        ]
        if self.protection_topic:
            self.subscribers.append(rospy.Subscriber(self.protection_topic, String, self.protection, queue_size=50))
        rospy.on_shutdown(self.shutdown)

    def write(self, event, **fields):
        record = dict(event=event, session_id=self.session_id, wall_time_unix_s=time.time(), ros_time=stamp(rospy.Time.now()), **fields)
        with self.lock:
            if self.closed:
                return
            self.file.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')
            self.file.flush()

    def goal(self, msg):
        pose = msg.goal.target_pose
        self.write('goal', callerid=callerid(msg), goal_id=msg.goal_id.id, goal_stamp=stamp(msg.goal_id.stamp),
                   frame_id=pose.header.frame_id, pose_stamp=stamp(pose.header.stamp),
                   position=dict(x=pose.pose.position.x, y=pose.pose.position.y, z=pose.pose.position.z),
                   orientation=dict(x=pose.pose.orientation.x, y=pose.pose.orientation.y,
                                    z=pose.pose.orientation.z, w=pose.pose.orientation.w))

    def result(self, msg):
        s = msg.status
        self.write('result', callerid=callerid(msg), goal_id=s.goal_id.id, goal_stamp=stamp(s.goal_id.stamp),
                   status=int(s.status), text=s.text, header_stamp=stamp(msg.header.stamp))

    def cancel(self, msg):
        self.write('cancel', callerid=callerid(msg), goal_id=msg.id, goal_stamp=stamp(msg.stamp))

    def status_update(self, msg):
        # Status arrays are periodic; write only changes, never infer missing IDs as transitions.
        for s in msg.status_list:
            value = (int(s.status), s.text)
            with self.lock:
                previous = self.status.get(s.goal_id.id)
                if previous == value:
                    continue
                self.status[s.goal_id.id] = value
            self.write('status_transition', callerid=callerid(msg), goal_id=s.goal_id.id, goal_stamp=stamp(s.goal_id.stamp),
                       previous_status=None if previous is None else previous[0],
                       status=value[0], text=value[1], header_stamp=stamp(msg.header.stamp))

    def protection(self, msg):
        self.write('protection_event', callerid=callerid(msg), topic=self.protection_topic, message=msg.data,
                   provenance='explicit configured String topic; content not interpreted')

    def shutdown(self):
        with self.lock:
            if self.closed:
                return
            record = dict(event='recorder_stop', session_id=self.session_id, wall_time_unix_s=time.time(), ros_time=stamp(rospy.Time.now()))
            self.file.write(json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n')
            self.file.flush()
            self.closed = True
            self.file.close()


def main():
    rospy.init_node('native_frontier_event_recorder')
    Recorder()
    rospy.spin()


if __name__ == '__main__':
    main()
