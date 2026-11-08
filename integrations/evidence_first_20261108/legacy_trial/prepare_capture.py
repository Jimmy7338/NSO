"""Prepare one actual legacy-mount integration trial, with no actuator."""
import copy, hashlib, json, socket, struct, sys, threading, time
from pathlib import Path
import numpy as np
WORK=Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029')
OUT=Path(__file__).resolve().parent/'fresh'
sys.path.insert(0,str(WORK/'runtime'))
import rospy, tf2_ros
from reference_observation import ReferenceObservationCollector
from joint_observation import JointObservationCollector, freeze_joint_grid
def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


OUT.mkdir(parents=True,exist_ok=False)
rospy.init_node('nso_legacy_mount_fresh_capture', disable_signals=True, disable_rosout=True)
tfbuf = tf2_ros.Buffer(cache_time=rospy.Duration(30))
listener = tf2_ros.TransformListener(tfbuf)
stop = threading.Event()
lock = threading.Lock()
motion = {}
can = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
can.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, b''.join(
    struct.pack('=II', cid, 0xc00007ff) for cid in (0x211, 0x221)))
can.setsockopt(socket.SOL_SOCKET, 35, 1)
can.bind(('can0',))
can.settimeout(.1)


def receive():
    while not stop.is_set():
        try:
            raw, anc, flags, _ = can.recvmsg(16, 128)
        except socket.timeout:
            continue
        cid, dlc, payload = struct.unpack('=IB3x8s', raw)
        if flags & (8 | 32) or dlc != 8 or cid not in (0x211, 0x221):
            continue
        stamp = next((struct.unpack('@ll', v[:struct.calcsize('@ll')])
                      for level, kind, v in anc if level == socket.SOL_SOCKET and kind == 35), None)
        if stamp is None:
            continue
        row = dict(receipt_monotonic=time.monotonic(), kernel_unix_s=stamp[0]+stamp[1]/1e9,
                   raw_frame_hex=raw.hex())
        if cid == 0x211:
            row.update(state=payload[0], mode=payload[1], error=struct.unpack('>H', payload[4:6])[0])
        else:
            v, w = struct.unpack('>hh', payload[:4]); row.update(v=v*.001, w=w*.001)
        with lock:
            motion['status' if cid == 0x211 else 'velocity'] = row


def snapshot():
    with lock:
        return copy.deepcopy(motion)


thread=threading.Thread(target=receive,daemon=True);thread.start()
collector=ReferenceObservationCollector(tfbuf)
joint=None
try:
    raw_path=collector.capture_live_joint(OUT/'observed',lambda:None,snapshot)
    with np.load(raw_path,allow_pickle=False) as z: receipt=json.loads(str(z['receipt_json']))
    historical=receipt['historical_tf'];base=np.asarray(historical['odom_from_base']['matrix'])
    pose=[float(base[0,3]),float(base[1,3]),float(np.arctan2(base[1,0],base[0,0]))]
    old=np.array([[0.,0.,1.,.061],[-1.,0.,0.,.0591],[0.,-1.,0.,.303],[0.,0.,0.,1.]])
    actual=np.asarray(historical['base_from_camera']['matrix'])
    assert np.allclose(actual,old,rtol=0,atol=1e-9), 'actual mount differs from user-requested legacy profile'
    grid=freeze_joint_grid(pose)
    joint=JointObservationCollector(tfbuf,grid,apply_approved_self_mask=False)
    packet=joint.capture(OUT/'joint_inputs',raw_path,source_role='fresh_legacy_mount_trial_no_motion_preparation')
    template=Path('/home/ouc/nso_robot/analyses/rapid_evidence_20261108/physical/fresh_S/configuration.json')
    prior=json.loads(template.read_text())
    cfg={k:copy.deepcopy(prior['configuration'][k]) for k in ['robot_radius_m','sensor_config','planner_revision','policy_variant','sensor_range_profile','measured_extent_profile']}
    cfg.update(shape=grid['shape'],planning_from_odom=grid['planning_from_odom'],return_anchor=grid['return_anchor'],total_budget=40,score_mode='S',episode_id='legacy_mount_S40_20261108_01',scene_id='legacy_mount_unknown_exploration_20261108')
    calibration=dict(source='actual RGBD/scan and per-sensor historical TF',pixel_stride=4,initial_grid=grid,
        initial_actual_base_odom_pose=pose,initial_odom_stamp_ns=historical['odom_from_base']['query_stamp_ns'],
        initial_odom_stamp_semantics='actual historical query',initial_odom_frame='odom',initialization_unix_s=time.time(),
        initialization_scope='current legacy-mount epoch; existing stationary navigation map retained; no old paid/semantic ledger',
        source_registration='fresh current epoch with no previous SE2 alignment',localization_source='lidar_csm',
        navigation_map_source=dict(topic='/nso_visual/map',publisher='/slam_gmapping',map_frame='map'),
        joint_turn_execution_profile='measured_in_place_turn_v1',apply_approved_self_mask=False,self_mask_pixels=0,
        self_mask_profile='unmasked_after_camera_relocation_v1',camera_mount_calibrated=False,
        camera_extrinsic_policy='user_requested_legacy_mount_rollback',camera_extrinsics_verified=False,
        captured_base_from_camera=actual.tolist(),captured_base_from_laser=historical['base_from_laser']['matrix'],
        floor_reference_profile='original_fixed_minus_0_233_retained',candidate_floor_profile_deployed=False,
        nominal_view_forecast_calibrated=False,physical_reconstruction_accuracy_verified=False,
        nominal_camera_height_m=cfg['sensor_config']['camera_height_m'],
        actual_camera_height_from_base_TF_and_fixed_floor_m=float(actual[2,3]-grid['floor_odom_z_m']),
        depth_confidence=rospy.get_param('/zed2i/zed_node/depth_confidence'),depth_texture_confidence=rospy.get_param('/zed2i/zed_node/depth_texture_conf'),
        SDK_depth_mode=rospy.get_param('/zed2i/zed_node/depth/depth_mode'),
        ROS_execution_frame='odom',ROS_xy_latch=True,ROS_xy_goal_tolerance_m=.08,Navfn_default_tolerance=0,
        controller_odometry_source='/nso_csm/control_odom',controller_odometry_projection='original default-keyframe CSM pose and kernel-timestamped CAN control velocity',
        correction_reverse_profile='endpoint_only_max0.05_mps_existing_authorization',
        external_deadline_s=600,evidence_scope='one legacy mount integration episode; uncalibrated floor reference; no matched efficacy or accuracy claim',
        prior_episode_ledger_imported=False,history_acquisition='current stationary navigation epoch retained; actual fresh paid observations')
    settings=dict(configuration=cfg,execution_allowed=True,calibration=calibration,semantic_configuration=json.loads((WORK/'config/real_semantic_frontend.json').read_text()))
    save(OUT/'configuration.json',settings)
    save(OUT/'source.json',dict(raw_observation=str(raw_path),joint_input=str(packet),grid=grid,no_motion=True,actual_CAN_snapshot=snapshot(),template=str(template),template_sha256=hashlib.sha256(template.read_bytes()).hexdigest(),inherited_configuration_fields=list(cfg),old_calibration_fields_copied=False))
    print(json.dumps(dict(configuration=str(OUT/'configuration.json'),pose=pose,CAN=snapshot())))
finally:
    collector.close()
    if joint is not None:joint.close()
    stop.set();thread.join(1);can.close();rospy.signal_shutdown('fresh legacy trial capture complete')
