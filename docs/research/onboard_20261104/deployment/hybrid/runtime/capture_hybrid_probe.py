#!/usr/bin/env python3
"""Read-only, bounded lidar/RGB-D-source capture; never a paid or motion experiment.

Only subscribes to source data and reads ROS parameters/master state. It never
opens CAN, sends a navigation goal, or publishes a topic. Absence of actuator
publishers is not an independent measurement of a stationary physical chassis.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import time

from mission import (MASTER, paid_frame_source_provenance, rgb_array, depth_array,
                     navigation_start_footprint, save)
from nso_ros_coordinator import matrix

TOPICS = dict(rgb='/zed2i/zed_node/left/image_rect_color',
    depth='/nso_visual/depth_filtered', camera_info='/zed2i/zed_node/depth/camera_info',
    health='/nso_visual/health', map='/nso_visual/map',
    local_costmap='/nso_visual/move_base/local_costmap/costmap',
    global_costmap='/nso_visual/move_base/global_costmap/costmap')
ACTUATOR_TOPICS = ['/nso_radius/guarded_cmd', '/nso_visual/actuator_state']


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    output=Path(args.output).resolve(); output.mkdir(parents=True,exist_ok=False)
    result=dict(schema='nso.hybrid_source_probe.v1',navigation_source='lidar_rgbd',mapping_source='gmapping',diagnostic_capture_not_paid_mission=True,
        independent_CAN_stationary_measurement=False,physical_stationary_state_verified=False,
        observation_stop_is_caller_assertion_only=True,no_motion_command_sent=True,
        no_CAN_device_opened=True,no_topic_publisher_created=True,maximum_capture_s=15.,
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        started_unix_s=time.time(),topics=TOPICS,frames=[],errors=[])
    subscribers=[]
    try:
        if os.environ.get('ROS_MASTER_URI')!=MASTER:
            raise ValueError('use isolated visual master '+MASTER)
        import numpy as np
        import rospy,rosgraph,message_filters,tf2_ros
        from sensor_msgs.msg import Image,CameraInfo
        from nav_msgs.msg import OccupancyGrid
        from std_msgs.msg import String
        from dynamic_reconfigure.msg import Config
        master=rosgraph.Master('/nso_visual_readonly_probe')
        def graph():
            publishers,_,_=master.getSystemState()
            return {topic:list(callers) for topic,callers in publishers}
        def check_actuators(label):
            publishers=graph()
            row=dict(stage=label,checked_unix_s=time.time(),
                actuator_publishers={topic:publishers.get(topic,[]) for topic in ACTUATOR_TOPICS},
                source_publishers={topic:publishers.get(topic,[]) for topic in TOPICS.values()})
            if result.get('probe_node_name'):
                row['probe_published_topics']=[topic for topic,callers in publishers.items()
                    if result['probe_node_name'] in callers]
            result.setdefault('publisher_checks',[]).append(row)
            if any(row['actuator_publishers'].values()):
                raise RuntimeError('actuator_publisher_present; diagnostic capture refused')
            if row.get('probe_published_topics'):
                raise RuntimeError('read-only probe unexpectedly owns a topic publisher')
        check_actuators('before_subscriptions')
        # rospy otherwise creates a /rosout publisher automatically.
        rospy.init_node('nso_visual_readonly_probe',anonymous=True,disable_signals=True,disable_rosout=True)
        result['probe_node_name']=rospy.get_name()
        tfbuf=tf2_ros.Buffer(cache_time=rospy.Duration(20.)); listener=tf2_ros.TransformListener(tfbuf)
        latest={}; jobs=queue.Queue(maxsize=8); last=[-1]
        def synchronized(rgb,depth,info):
            ns=depth.header.stamp.to_nsec()
            if (rgb.header.stamp.to_nsec()!=ns or info.header.stamp.to_nsec()!=ns or ns<=last[0]):return
            callers=[getattr(item,'_connection_header',{}).get('callerid') for item in (rgb,depth,info)]
            if callers!=['/zed2i/zed_node','/nso_visual_sensor_bridge','/zed2i/zed_node']:return
            last[0]=ns
            try:jobs.put_nowait((ns,rgb,depth,info,callers))
            except queue.Full:pass
        for kind,msg_type in [('health',String),('map',OccupancyGrid),
                              ('local_costmap',OccupancyGrid),('global_costmap',OccupancyGrid)]:
            def receive(message,kind=kind):latest[kind]=(time.monotonic(),message)
            subscribers.append(rospy.Subscriber(TOPICS[kind],msg_type,receive,queue_size=1))
        def dynamic_configuration(message):latest['dynamic_camera_configuration']=(time.monotonic(),message)
        subscribers.append(rospy.Subscriber('/zed2i/zed_node/parameter_updates',Config,dynamic_configuration,queue_size=1))
        filters=[message_filters.Subscriber(TOPICS[kind],msg_type,queue_size=2,buff_size=8000000)
                 for kind,msg_type in [('rgb',Image),('depth',Image),('camera_info',CameraInfo)]]
        subscribers.extend(filters)
        sync=message_filters.TimeSynchronizer(filters,10);sync.registerCallback(synchronized)
        # Reserve one second for an in-flight bounded TF query. Subscription
        # capture itself remains below the requested hard 15-second limit.
        started=time.monotonic();deadline=started+14.;prepared=[];next_graph_check=started+.5
        needed={'health','map','local_costmap','global_costmap','dynamic_camera_configuration'}
        while time.monotonic()<deadline and (len(prepared)<3 or not needed.issubset(latest)) and not rospy.is_shutdown():
            if 'health' in latest:
                health=json.loads(latest['health'][1].data)
                faults={kind:health.get(kind,{}).get('fault') for kind in ('odom','depth')
                        if health.get(kind,{}).get('fault')}
                if faults:
                    result['capture_source_fault']=faults
                    if needed.issubset(latest):break
                    time.sleep(.02)
                    continue
            if time.monotonic()>=next_graph_check:
                check_actuators('during_capture');next_graph_check=time.monotonic()+.5
            if len(prepared)>=3:
                time.sleep(.02);continue
            try:ns,rgb,depth,info,callers=jobs.get(timeout=.05)
            except queue.Empty:continue
            try:
                if not -.02<=rospy.Time.now().to_sec()-ns/1e9<=.5:
                    raise ValueError('source frame stale or future')
                science_tf=tfbuf.lookup_transform('odom',depth.header.frame_id,depth.header.stamp,rospy.Duration(.25))
                nav_tf=tfbuf.lookup_transform('map',depth.header.frame_id,depth.header.stamp,rospy.Duration(.25))
                payloads=dict(rgb=rgb_array(rgb),depth_m=depth_array(depth),
                    intrinsic=np.asarray(info.K).reshape(3,3),world_from_camera=matrix(science_tf.transform),
                    map_from_camera=matrix(nav_tf.transform))
                provenance=paid_frame_source_provenance(rgb,depth,info,science_tf,nav_tf,payloads)
                provenance['diagnostic_capture_not_paid_mission']=True
                provenance['independent_CAN_stationary_measurement']=False
                prepared.append(dict(ns=ns,payloads=payloads,provenance=provenance,
                    component_times=dict(rgb_timestamp_s=rgb.header.stamp.to_nsec()/1e9,
                        depth_timestamp_s=depth.header.stamp.to_nsec()/1e9,
                        pose_timestamp_s=science_tf.header.stamp.to_nsec()/1e9),callers=callers))
            except Exception as exc:
                if len(result['errors'])<32:
                    result['errors'].append(dict(stage='frame_preparation',reason=str(exc),source_stamp_ns=ns))
        result['capture_duration_s']=time.monotonic()-started
        # Stop source capture before filesystem compression or offline analysis.
        for sub in subscribers:sub.unregister()
        subscribers=[]
        check_actuators('after_capture_before_persisting')
        if 'health' in latest:
            receipt,message=latest['health']
            result['health']=dict(receipt_age_s=time.monotonic()-receipt,
                caller=getattr(message,'_connection_header',{}).get('callerid'),data=json.loads(message.data))
        result['maps']={}
        for kind in ('map','local_costmap','global_costmap'):
            if kind not in latest:
                result['maps'][kind]=dict(missing=True);continue
            receipt,message=latest[kind]
            data=np.asarray(message.data,dtype=np.int16).reshape(message.info.height,message.info.width)
            origin=message.info.origin;p=origin.position;q=origin.orientation
            np.savez_compressed(output/(kind+'.npz'),data=data,resolution_m=message.info.resolution,
                source_stamp_ns=message.header.stamp.to_nsec(),frame_id=message.header.frame_id,
                origin_xyz=np.array([p.x,p.y,p.z]),origin_xyzw=np.array([q.x,q.y,q.z,q.w]))
            description=dict(source_stamp_ns=message.header.stamp.to_nsec(),frame=message.header.frame_id,
                receipt_age_s=time.monotonic()-receipt,shape=list(data.shape),resolution_m=message.info.resolution,
                caller=getattr(message,'_connection_header',{}).get('callerid'),
                free_cells=int((data==0).sum()),unknown_cells=int((data<0).sum()),occupied_cells=int((data>=100).sum()))
            try:
                tf=tfbuf.lookup_transform(message.header.frame_id,'base_link',rospy.Time(0),rospy.Duration(.25))
                description['full_padded_footprint']=navigation_start_footprint(message,tf)
                description['full_padded_footprint']['evaluation_grid_kind']=kind
                if kind=='map':
                    description['full_padded_footprint'].update(includes_actual_layered_local_grid=False,
                        free_cell_source='raw laser gmapping grid; no navigation robot-footprint clearing')
                description['base_transform_age_s']=rospy.Time.now().to_sec()-tf.header.stamp.to_sec()
            except Exception as exc:description['footprint_error']=str(exc)
            result['maps'][kind]=description
        parameter_names=['/zed2i/zed_node/depth_confidence','/zed2i/zed_node/depth_texture_conf',
            '/zed2i/zed_node/depth/depth_mode','/zed2i/zed_node/general/serial_number',
            '/zed2i/zed_node/general/pub_frame_rate','/zed2i/zed_node/pos_tracking/pos_tracking_mode',
            '/zed2i/zed_node/pos_tracking/imu_fusion','/zed2i/zed_node/pos_tracking/area_memory',
            '/nso_visual/rtabmap/Grid/DepthDecimation','/nso_visual/rtabmap/map_always_update']
        result['actual_parameters']={name:rospy.get_param(name,None) for name in parameter_names}
        if 'dynamic_camera_configuration' in latest:
            receipt,message=latest['dynamic_camera_configuration']
            result['actual_dynamic_camera_configuration']=dict(receipt_age_s=time.monotonic()-receipt,
                caller=getattr(message,'_connection_header',{}).get('callerid'),
                source='/zed2i/zed_node/parameter_updates',
                values={item.name:item.value for group in (message.ints,message.bools,message.doubles,message.strs) for item in group})
        else:result['actual_dynamic_camera_configuration']=dict(missing=True)
        # Evaluate health and latest TF/grid relations before compressing the
        # camera frames, so archival I/O does not age these diagnostic samples.
        frames_dir=output/'frames';frames_dir.mkdir()
        for row in prepared:
            path=frames_dir/(str(row['ns'])+'.npz')
            np.savez_compressed(path,**row['payloads'],world_frame='odom',frame_id=str(row['ns']),
                timestamp_s=row['ns']/1e9,**row['component_times'],
                source_provenance_json=json.dumps(row['provenance'],sort_keys=True,allow_nan=False),
                diagnostic_capture_not_paid_mission=np.bool_(True))
            result['frames'].append(dict(path=str(path),source_stamp_ns=row['ns'],
                source_callers=row['callers'],sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                diagnostic_capture_not_paid_mission=True))
        check_actuators('after_archiving')
        if len(prepared)!=3:
            raise RuntimeError('three exact synchronized TF-qualified source frames not captured; inspect source fault/TF evidence')
        result['status']='three_actual_source_frames_archived_no_motion'
        returncode=0
    except Exception as exc:
        result['status']='refused_or_incomplete';result['error']=str(exc);returncode=1
    finally:
        for sub in subscribers:
            try:sub.unregister()
            except Exception:pass
        result['finished_unix_s']=time.time()
        save(output/'summary.json',result)
    print(json.dumps(dict(status=result['status'],output=str(output),frames=len(result['frames']),
        error=result.get('error')),ensure_ascii=False,allow_nan=False))
    return returncode


if __name__=='__main__':raise SystemExit(main())
