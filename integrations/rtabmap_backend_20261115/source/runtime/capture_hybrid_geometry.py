#!/usr/bin/env python3
"""Stationary diagnostic capture of real RGB-D and existing lidar TF; no publishers."""
import argparse
import json
from pathlib import Path
import queue
import sys
import time

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    import rospy,message_filters,tf2_ros
    from sensor_msgs.msg import Image,CameraInfo
    from tf2_msgs.msg import TFMessage
    from mission import rgb_array,depth_array
    from nso_ros_coordinator import matrix
    rospy.init_node('hybrid_geometry_readonly',anonymous=True,disable_rosout=True)
    tfbuf=tf2_ros.Buffer(cache_time=rospy.Duration(20.));listener=tf2_ros.TransformListener(tfbuf)
    authorities={};jobs=queue.Queue(maxsize=20)
    def tf_message(message):
        caller=message._connection_header.get('callerid')
        for t in message.transforms:
            if t.child_frame_id in ('base_link','odom'):
                key=t.header.frame_id+'->'+t.child_frame_id
                authorities.setdefault(key,set()).add(caller)
    sub=rospy.Subscriber('/tf',TFMessage,tf_message,queue_size=100)
    filters=[message_filters.Subscriber(topic,cls,queue_size=3,buff_size=8000000)
             for topic,cls in [('/zed2i/zed_node/left/image_rect_color',Image),
                              ('/zed2i/zed_node/depth/depth_registered',Image),
                              ('/zed2i/zed_node/depth/camera_info',CameraInfo)]]
    def receive(*messages):
        try:jobs.put_nowait(messages)
        except queue.Full:pass
    sync=message_filters.TimeSynchronizer(filters,12);sync.registerCallback(receive)
    deadline=time.monotonic()+18.;frames=[];errors=[];last_ns=None
    while time.monotonic()<deadline and len(frames)<3:
        try:rgb,depth,info=jobs.get(timeout=.1)
        except queue.Empty:continue
        ns=depth.header.stamp.to_nsec()
        if last_ns is not None and ns-last_ns<400000000:continue
        try:
            if len({m.header.stamp.to_nsec() for m in (rgb,depth,info)})!=1:raise ValueError('unequal stamps')
            if any(m._connection_header.get('callerid')!='/zed2i/zed_node' for m in (rgb,depth,info)):raise ValueError('wrong publisher')
            if not -.02<=rospy.Time.now().to_sec()-ns/1e9<=.5:raise ValueError('stale RGB-D')
            transforms={target:tfbuf.lookup_transform(target,depth.header.frame_id,depth.header.stamp,rospy.Duration(.3))
                        for target in ('base_link','odom','map')}
            arrays=dict(rgb=rgb_array(rgb),depth_m=depth_array(depth),intrinsic=np.array(info.K).reshape(3,3))
            arrays.update({target+'_from_camera':matrix(t.transform) for target,t in transforms.items()})
            path=out/(str(ns)+'.npz')
            np.savez_compressed(path,**arrays,source_stamp_ns=ns,diagnostic_capture_not_paid_mission=True)
            frames.append(dict(path=str(path),source_stamp_ns=ns,
                               optical_frame=depth.header.frame_id,
                               tf_response_stamps_ns={k:t.header.stamp.to_nsec() for k,t in transforms.items()}))
            last_ns=ns
        except Exception as exc:
            if len(errors)<15:errors.append(str(exc))
    for f in filters:f.unregister()
    sub.unregister()
    result=dict(frames=frames,tf_authorities={k:sorted(v) for k,v in authorities.items()},
                errors=errors,diagnostic_capture_not_paid_mission=True,
                target='user_selected_laboratory_desk_behind_box',no_motion_command=True,
                physical_extrinsics_verified=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
    return 0 if len(frames)==3 else 1


if __name__=='__main__':raise SystemExit(main())
