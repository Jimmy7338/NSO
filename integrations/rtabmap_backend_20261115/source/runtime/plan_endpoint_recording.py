"""Observational Path endpoints; asynchronous goal context is not an ID binding."""
import math


def header_record(header):
    return dict(frame_id=header.frame_id, stamp_ns=int(header.stamp.to_nsec()))


class PlanEndpointRecorder:
    def __init__(self):
        self.previous = {}

    def record(self, topic, message, latest_goal, receipt_monotonic, receipt_unix_s):
        poses = message.poses
        first = header_record(poses[0].header) if poses else None
        last = header_record(poses[-1].header) if poses else None
        endpoint = None
        if poses:
            p, q = poses[-1].pose.position, poses[-1].pose.orientation
            endpoint = dict(x=float(p.x), y=float(p.y),
                yaw=math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z)))
        goal = dict(latest_goal) if latest_goal is not None else None
        key = (goal.get('id') if goal else None, message.header.frame_id,
               first['frame_id'] if first else None, last['frame_id'] if last else None,
               tuple(endpoint.values()) if endpoint else None)
        if self.previous.get(topic) == key:
            return None
        self.previous[topic] = key
        return dict(event='plan_endpoint_observation', topic=topic,
            path_header=header_record(message.header), first_pose_header=first,
            last_pose_header=last, endpoint=endpoint, pose_count=len(poses),
            latest_observed_goal_id=goal.get('id') if goal else None,
            latest_observed_goal_request=goal,
            association='asynchronous latest-observed goal context; not strict goal-ID binding',
            receipt_monotonic=receipt_monotonic, receipt_unix_s=receipt_unix_s,
            scope='recording only; does not affect checks, planning or admission')
