#!/usr/bin/env python3
"""Local ROS master + real synthetic messages + actual V3 recorder probe.

Requires a sourced ROS1 environment. Sends sensor/TF fixtures only; no commands.
This does not test robot drivers, extrinsic truth, motion or reconstruction.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import xmlrpc.client
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def environment_check():
    names=('rospy','tf2_ros','message_filters','cv_bridge','sensor_msgs','geometry_msgs','numpy')
    available={name:importlib.util.find_spec(name) is not None for name in names}
    core=shutil.which('roscore')
    return dict(available=available,roscore=core,ready=all(available.values()) and core is not None,
                python=sys.version,executable=sys.executable)


def stop(process):
    if process is None or process.poll() is not None:return
    process.send_signal(signal.SIGINT)
    try:process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:process.wait(timeout=3)
        except subprocess.TimeoutExpired:process.kill();process.wait()


def probe(output):
    output.mkdir(parents=True,exist_ok=False)
    environment=environment_check()
    (output/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    if not environment['ready']:
        result=dict(status='not_executed_missing_runtime',environment=environment,
                    live_ros_verified=False,robot_interface_ready=False)
        (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        return 2
    import numpy as np
    import rospy
    import tf2_ros
    from cv_bridge import CvBridge
    from geometry_msgs.msg import TransformStamped
    from sensor_msgs.msg import CameraInfo,LaserScan
    from scripts.audit_ros1_recording_v3 import check_metadata_sequence
    with socket.socket() as reserve:
        reserve.bind(('127.0.0.1',0));port=reserve.getsockname()[1]
    master_uri='http://127.0.0.1:%d'%port
    os.environ.update(ROS_MASTER_URI=master_uri,ROS_IP='127.0.0.1',ROS_HOSTNAME='localhost',ROS_HOME=str(output/'ros_home'))
    environment=dict(os.environ)
    core=recorder=None;result=dict(status='running',master_uri=master_uri,live_ros_verified=False,robot_interface_ready=False)
    log_core=(output/'roscore.log').open('w');log_rec=(output/'recorder.log').open('w')
    try:
        core=subprocess.Popen([shutil.which('roscore'),'-p',str(port)],env=environment,stdout=log_core,stderr=subprocess.STDOUT)
        master=xmlrpc.client.ServerProxy(master_uri)
        deadline=time.monotonic()+20
        while True:
            if core.poll() is not None:raise RuntimeError('isolated roscore exited')
            try:
                if master.getPid('/nso_v3_probe')[0]==1:break
            except (OSError,xmlrpc.client.Error):pass
            if time.monotonic()>deadline:raise RuntimeError('isolated roscore readiness timed out')
            time.sleep(.1)
        rospy.init_node('nso_v3_fixture',anonymous=False,disable_signals=True)
        bridge=CvBridge()
        depth=bridge.cv2_to_imgmsg(np.array([[1000,0],[5000,2000]],np.uint16),encoding='16UC1')
        rgb=bridge.cv2_to_imgmsg(np.zeros((2,2,3),np.uint8),encoding='rgb8')
        scan=LaserScan();scan.angle_min=-.2;scan.angle_max=.2;scan.angle_increment=.1
        scan.range_min=.2;scan.range_max=4.;scan.time_increment=.001;scan.scan_time=.5
        scan.ranges=[1.,5.,float('inf'),float('nan'),.1]
        info=CameraInfo();info.header.frame_id='left_optical';info.width=info.height=8
        info.binning_x=info.binning_y=2;info.roi.x_offset=info.roi.y_offset=2;info.roi.width=info.roi.height=4
        info.K=[8.,0.,4.,0.,8.,4.,0.,0.,1.];info.P=[8.,0.,4.,0.,0.,8.,4.,0.,0.,0.,1.,0.]
        info.R=[0.,-1.,0.,1.,0.,0.,0.,0.,1.];info.D=[0.]*5;info.distortion_model='plumb_bob'
        base='/nso_v3_fixture'
        publishers=[rospy.Publisher(base+'/depth',type(depth),queue_size=5),
                    rospy.Publisher(base+'/rgb',type(rgb),queue_size=5),
                    rospy.Publisher(base+'/scan',LaserScan,queue_size=5)]
        calibration=rospy.Publisher(base+'/camera_info',CameraInfo,queue_size=1,latch=True)
        broadcaster=tf2_ros.StaticTransformBroadcaster();transforms=[]
        for frame in ('left_optical','laser'):
            t=TransformStamped();t.header.frame_id='map';t.child_frame_id=frame;t.header.stamp=rospy.Time.now()
            t.transform.translation.x=1.;t.transform.translation.y=2.;t.transform.translation.z=3.;t.transform.rotation.w=1.
            transforms.append(t)
        broadcaster.sendTransform(transforms);calibration.publish(info)
        recording=output/'recording'
        command=[sys.executable,str(ROOT/'scripts/record_ros1_rgbd_v3.py'),'_output:='+str(recording),
            '_camera_optical_frame:=left_optical','_depth_topic:='+base+'/depth','_rgb_topic:='+base+'/rgb',
            '_scan_topic:='+base+'/scan','_camera_info_topic:='+base+'/camera_info','_min_interval_s:=0.0']
        recorder=subprocess.Popen(command,env=environment,stdout=log_rec,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+20
        while any(p.get_num_connections()==0 for p in publishers):
            if recorder.poll() is not None:raise RuntimeError('recorder exited before subscription')
            if time.monotonic()>deadline:raise RuntimeError('sensor subscriptions timed out')
            time.sleep(.1)
        for seq in range(10):
            stamp=rospy.Time.now()
            for p,msg,frame in zip(publishers,(depth,rgb,scan),('left_optical','left_optical','laser')):
                msg.header.stamp=stamp;msg.header.seq=seq;msg.header.frame_id=frame;p.publish(msg)
            time.sleep(.5)
        stop(recorder)
        audit=check_metadata_sequence(recording)
        (output/'offline_audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
        manifest=json.loads((recording/'sequence.json').read_text())
        if audit['status']!='passed' or manifest['frames']<5:raise RuntimeError('insufficient valid recordings or offline contract failure')
        for path in sorted((recording/'scans').glob('*.npz')):
            with np.load(path,allow_pickle=False) as payload:
                np.testing.assert_array_equal(payload['ranges_m'],[1,0,0,0,0])
        system=master.getSystemState('/nso_v3_probe')
        (output/'ros_master_system_state.json').write_text(json.dumps(system,indent=2)+'\n')
        published=sorted({topic for topic,nodes in system[2][0]})
        if any('cmd_vel' in topic for topic in published):raise RuntimeError('unexpected command publisher in isolated probe')
        result.update(status='passed',live_ros_verified=True,recorded_frames=manifest['frames'],
            rejected_frames=manifest['rejected'],published_topics=published,
            sensor_data='synthetic; genuine ROS serialization/subscription/TF/cv_bridge path',
            physical_robot_verified=False)
    except Exception as error:
        result.update(status='failed',error=repr(error))
    finally:
        stop(recorder)
        if rospy.core.is_initialized():rospy.signal_shutdown('probe finished')
        stop(core);log_core.close();log_rec.close()
        (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    raise SystemExit(probe(a.output.resolve()))
