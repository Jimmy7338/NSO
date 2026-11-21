"""Fresh real inputs for a declared RTAB continuation; no actuator or policy call."""
import hashlib
import json
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rospy
import tf2_ros
from nav_msgs.msg import Odometry

B = Path('/home/ouc/nso_robot')
W = B / 'workspaces/nso_hybrid_20261029'
A = B / 'analyses/resume_20261115'
sys.path[:0] = [str(W/'runtime'), str(W/'nso_bridge')]
from reference_observation import ReferenceObservationCollector
from joint_observation import JointObservationCollector, freeze_joint_grid
from nso_bridge.rtabmap_geometry import corrected_geometry

def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def main():
    episode = sys.argv[1]
    out = A / episode
    out.mkdir(exist_ok=False)
    rospy.init_node('nso_rtab_continuation_capture', anonymous=True, disable_signals=True)
    buf = tf2_ros.Buffer(cache_time=rospy.Duration(60))
    listener = tf2_ros.TransformListener(buf)
    latest = [None]
    lock = threading.Lock()
    def cb(m):
        with lock:
            latest[0] = dict(source_topic='/nso_csm/control_odom', caller=m._connection_header.get('callerid'),
                source_stamp_ns=m.header.stamp.to_nsec(), receipt_ros_s=rospy.Time.now().to_sec(),
                receipt_monotonic=time.monotonic(), linear_x=m.twist.twist.linear.x,
                angular_z=m.twist.twist.angular.z, semantics='actual CAN sourced control odometry')
    def motion_snapshot():
        with lock:
            if latest[0] is None: raise ValueError('actual control odometry missing')
            return dict(latest[0])
    sub = rospy.Subscriber('/nso_csm/control_odom', Odometry, cb, queue_size=1)
    collector = joint = None
    registry_path = A/'processes.json'
    registry = json.loads(registry_path.read_text())
    backend_args = registry['unified_mapping_backend']['argv']
    snapshot = Path(next(a.split(':=',1)[1] for a in backend_args if a.startswith('_latest_snapshot_path:=')))
    fresh_map = registry.get('current_mapping_has_prior_manual_history') is False
    try:
        time.sleep(2)
        t = buf.lookup_transform('nso_pair_map','base_link',rospy.Time(0),rospy.Duration(3))
        q,p = t.transform.rotation,t.transform.translation
        pose = [p.x,p.y,math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))]
        settings = json.loads((A/'NSO_rtabmap_integrated_01/configuration.json').read_text())
        settings.pop('shared_observed_prefix',None)
        cal,c = settings['calibration'],settings['configuration']
        floor = cal['initial_grid']['floor_odom_z_m']
        grid = freeze_joint_grid(pose,shape=tuple(c['shape']),floor_odom_z_m=floor)
        collector = ReferenceObservationCollector(buf)
        timings = {}
        raw = collector.capture_live_joint(out/'observed',lambda:None,motion_snapshot,timings=timings)
        with np.load(raw,allow_pickle=False) as z: arrays={k:z[k].copy() for k in z.files}
        corrected,receipt = corrected_geometry(arrays,json.loads(snapshot.read_text()))
        pose = corrected['base_odom_pose'].tolist()
        grid = freeze_joint_grid(pose,shape=tuple(c['shape']),floor_odom_z_m=floor)
        c.update(episode_id=episode,planning_from_odom=grid['planning_from_odom'],return_anchor=grid['return_anchor'])
        if fresh_map:
            c['scene_id']='fresh_RTAB_unknown_local_20261115'
        cal.update(initial_grid=grid,initial_actual_base_odom_pose=pose,
            initial_odom_stamp_ns=receipt['rgbd']['sensor_stamp_ns'],initial_odom_frame='nso_pair_map',
            initialization_unix_s=time.time(),prior_episode_ledger_imported=False,
            old_pose_or_planning_transform_reused=False,
            initialization_scope='New actual stopped observation, same current-session manual and autonomous mapping history; no prior research ledger',
            history_acquisition='Current RTAB navigation map retained; no shared research observed prefix imported',
            navigation_map_initialization='Existing same-session unified RTAB map; NOT from-blank unknown-space trial',
            evidence_scope='Execution-repair continuation of declared unbounded v4. Not independent matched G/S evidence or reconstruction accuracy proof.')
        cal.update(rtabmap_graph_snapshot_path=str(snapshot),restore_epoch_id=registry['epoch_id'],
            local_controller_profile='teb_onsite_062x059_v1',
            paired_map_epoch_receipt=dict(path=str(registry_path),sha256=hashlib.sha256(registry_path.read_bytes()).hexdigest(),epoch_id=registry['epoch_id']))
        for key in ('warm_history_candidate','warm_history_paid_actions','warm_history_arrival_credit','source_context'):
            cal.pop(key,None)
        if fresh_map:
            cal.update(initialization_scope='Fresh actual stopped observation and fresh own RTAB mapping DB; no prior manual map or research observations imported',
                history_acquisition='Current RTAB epoch began at this stopped physical location; prior scene history excluded',
                navigation_map_initialization='New empty RTAB DB, actual first sensor frames only',
                evidence_scope='Fresh-map initialization; not yet a completed matched G/S comparison or reconstruction accuracy proof.')
        joint = JointObservationCollector(buf,grid,topic='/nso_pair/map',publisher='/nso_rtabmap_mapping_backend',
            map_frame='nso_pair_map',apply_approved_self_mask=cal.get('apply_approved_self_mask',True),
            geometry_pose_backend=cal['geometry_pose_backend'],graph_snapshot_path=str(snapshot))
        packet = joint.capture(out/'joint_inputs',raw,lambda:None,timings=timings)
        save(out/'configuration.json',settings)
        save(out/'fresh_capture.json',dict(raw=str(raw),joint=str(packet),pose=pose,frame='nso_pair_map',
            geometry_receipt=receipt,motion=motion_snapshot(),timings=timings))
        sources=[W/'runtime/mission.py',W/'runtime/visual_velocity_adapter.py',
            W/'runtime/rtabmap_mapping_backend.py',W/'runtime/ros_viewpoint_coordinator.py',
            W/'nso_bridge/nso_bridge/ros_viewpoint_unbounded_v4.py',
            W/'config/joint_navigation_teb_rtabmap_062x059_v1.yaml']
        protocol=dict(schema='nso.rtab_continuation/1',episode=episode,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            frozen_unix_s=time.time(),research_policy=c['policy_variant'],score_mode=c['score_mode'],
            local_controller_profile='teb_onsite_062x059_v1',ordinary_reverse_alignment_profile='teb_native_v1',
            mission_and_other_workload_cpus=[0,1,4,5,6,7],adapter_cpu=2,native_sender_cpu=3,native_sender_rt_priority=20,
            retained_navigation_map=not fresh_map,independent_matched_repeat=False,prior_research_ledger_imported=False,
            policy_changed_from_integrated01=False,grouping_changed_from_integrated01=False,
            watchdog_thresholds_changed=False,total_time_limit=None,waypoint_limit=None,roundtrip_budget=None,
            automatic_return=False,map_source='/nso_pair/map',world_frame='nso_pair_map',
            current_input=str(packet),fresh_capture=str(out/'fresh_capture.json'),
            configuration=str(out/'configuration.json'),configuration_sha256=hashlib.sha256((out/'configuration.json').read_bytes()).hexdigest(),
            source_bindings={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
        save(out/'protocol.json',protocol)
        print(json.dumps(dict(configuration=protocol['configuration'],protocol=str(out/'protocol.json'),pose=pose,no_actuator=True)))
    finally:
        if joint: joint.close()
        if collector: collector.close()
        sub.unregister()
        rospy.signal_shutdown('actual capture complete')

if __name__=='__main__': main()
