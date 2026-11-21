"""Prepare explicitly versioned ground-reference input diagnostic; no actuator."""
import argparse, copy, hashlib, json, socket, struct, sys, threading, time
from pathlib import Path
import numpy as np
WORK=Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029')
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--episode',required=True);parser.add_argument('--budget',type=int,required=True);parser.add_argument('--duration',type=int,required=True);parser.add_argument('--map-source-receipt',type=Path,required=True);args=parser.parse_args()
if args.budget != 40: parser.error('this controlled trial fixes budget at 40')
if not 1 <= args.duration <= 600: parser.error('duration must respect original mission 1..600s')
OUT=args.output
PROFILE=Path('/home/ouc/nso_robot/analyses/sensor_repair_20261108/execution_ground_profile.json')
profile=json.loads(PROFILE.read_text())
assert abs(float(profile['floor_relative_to_training_base_m'])-(-.12447258789516671))<1e-12, 'unexpected frozen ground profile'
CAMERA_RECEIPT=Path('/home/ouc/nso_robot/analyses/calibrated_mount_20261108/tf_verification.json')
camera_receipt=json.loads(CAMERA_RECEIPT.read_text())
expected_camera=np.asarray(camera_receipt['expected_base_from_camera_optical'],dtype=float)
MAP_SOURCE=dict(topic='/nso_pair/map',publisher='/nso_pair/gmapping',map_frame='nso_pair_map')
map_receipt=json.loads(args.map_source_receipt.read_text())
assert map_receipt['initial_map']['publisher']==MAP_SOURCE['publisher'] and map_receipt['initial_map']['frame']==MAP_SOURCE['map_frame']
assert set(map_receipt['costmap_sources'])=={'global_costmap','local_costmap'}
assert all(v==MAP_SOURCE['topic'] for v in map_receipt['costmap_sources'].values())
sys.path.insert(0,str(WORK/'runtime'))
import rospy, tf2_ros
from reference_observation import ReferenceObservationCollector
from joint_observation import JointObservationCollector, freeze_joint_grid
def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


OUT.mkdir(parents=True,exist_ok=False)
rospy.init_node('nso_depth_ground_fresh_capture', disable_signals=True, disable_rosout=True)
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
    actual=np.asarray(historical['base_from_camera']['matrix'])
    assert actual.shape==(4,4) and expected_camera.shape==(4,4) and np.allclose(actual,expected_camera,rtol=0,atol=1e-9), 'actual mount differs from confirmed Nov06 composed TF'
    camera_tf_max_error=float(np.max(np.abs(actual-expected_camera)))
    floor_odom_z=float(base[2,3])+float(profile['floor_relative_to_training_base_m'])
    grid=freeze_joint_grid(pose,floor_odom_z_m=floor_odom_z)
    joint=JointObservationCollector(tfbuf,grid,**MAP_SOURCE,apply_approved_self_mask=False)
    packet=joint.capture(OUT/'joint_inputs',raw_path,source_role='fresh_confirmed_Nov06_mount_trial_no_motion_preparation')
    template=Path('/home/ouc/nso_robot/analyses/rapid_evidence_20261108/physical/fresh_S/configuration.json')
    prior=json.loads(template.read_text())
    cfg={k:copy.deepcopy(prior['configuration'][k]) for k in ['robot_radius_m','sensor_config','planner_revision','policy_variant','sensor_range_profile','measured_extent_profile']}
    assert abs(cfg['robot_radius_m']-.7212185868375829)<1e-12
    assert cfg['planner_revision']=='v10_3_1' and cfg['policy_variant']=='exploration_joint_v1'
    assert cfg['sensor_range_profile']=='sensor_range_matched_v1' and cfg['measured_extent_profile']=='measured_extent_3m_v1'
    assert abs(cfg['sensor_config']['camera_height_m']-.536)<1e-12 and abs(cfg['sensor_config']['laser_height_m']-.684)<1e-12
    assert 'asset_grouping_profile' not in cfg
    cfg.update(shape=grid['shape'],planning_from_odom=grid['planning_from_odom'],return_anchor=grid['return_anchor'],total_budget=args.budget,score_mode='S',episode_id=args.episode,scene_id='depth_ground_unknown_exploration_20261108')
    calibration=dict(source='actual RGBD/scan and per-sensor historical TF',pixel_stride=4,initial_grid=grid,
        initial_actual_base_odom_pose=pose,initial_odom_stamp_ns=historical['odom_from_base']['query_stamp_ns'],
        initial_odom_stamp_semantics='actual historical query',initial_odom_frame='odom',initialization_unix_s=time.time(),
        initialization_scope='current confirmed Nov06 mount odometry epoch; fresh isolated navigation map; no old paid/semantic ledger',
        source_registration='fresh current epoch with no previous SE2 alignment',localization_source='lidar_csm',
        navigation_map_source=MAP_SOURCE,paired_map_epoch_receipt=map_receipt,paired_map_epoch_receipt_sha256=hashlib.sha256(args.map_source_receipt.read_bytes()).hexdigest(),
        joint_turn_execution_profile='measured_in_place_turn_v1',apply_approved_self_mask=False,self_mask_pixels=0,
        self_mask_profile='unmasked_after_camera_relocation_v1',camera_mount_calibrated=True,
        camera_mount_user_confirmed_unchanged_since_Nov06=True,
        camera_extrinsic_policy='confirmed_Nov06_composed_TF_actual_capture_match',camera_extrinsics_verified=False,
        camera_TF_receipt_path=str(CAMERA_RECEIPT),camera_TF_receipt_sha256=hashlib.sha256(CAMERA_RECEIPT.read_bytes()).hexdigest(),camera_TF_max_abs_error=camera_tf_max_error,
        captured_base_from_camera=actual.tolist(),captured_base_from_laser=historical['base_from_laser']['matrix'],
        floor_reference_profile='depth_ground_reference_v1',input_geometry_profile='depth_ground_reference_v1',candidate_floor_profile_deployed=True,ground_profile_reuse='legacy mount training scalar reused unchanged for single camera-TF input comparison; not the separately fitted Nov06 ground profile',
        ground_reference_receipt=dict(path=str(PROFILE),sha256=hashlib.sha256(PROFILE.read_bytes()).hexdigest(),training_source=profile['source_training'],floor_relative_to_training_base_m=profile['floor_relative_to_training_base_m'],actual_base_z_m=float(base[2,3]),actual_floor_odom_z_m=floor_odom_z,original_floor_odom_z_m=-.233,method_effect='vertical input projection, occupancy and quality/asset heights change; original thresholds and scorer/candidates unchanged',independent_physical_truth_verified=False),
        nominal_view_forecast_calibrated=False,physical_reconstruction_accuracy_verified=False,
        nominal_camera_height_m=cfg['sensor_config']['camera_height_m'],
        actual_camera_origin_height_over_ground_profile_m=float(actual[2,3]-profile['floor_relative_to_training_base_m']),
        actual_laser_origin_height_over_ground_profile_m=float(np.asarray(historical['base_from_laser']['matrix'])[2,3]-profile['floor_relative_to_training_base_m']),
        forecast_geometry_limit='Nominal camera .536m and laser .684m unchanged; actual origin heights use reused old-mount training scalar, not independent ground truth; original nominal camera projection does not encode measured camera pitch/offset or actual intrinsics.',
        depth_confidence=rospy.get_param('/zed2i/zed_node/depth_confidence'),depth_texture_confidence=rospy.get_param('/zed2i/zed_node/depth_texture_conf'),
        SDK_depth_mode=rospy.get_param('/zed2i/zed_node/depth/depth_mode'),
        ROS_execution_frame='odom',ROS_xy_latch=True,ROS_xy_goal_tolerance_m=.08,Navfn_default_tolerance=0,
        controller_odometry_source='/nso_csm/control_odom',controller_odometry_projection='original default-keyframe CSM pose and kernel-timestamped CAN control velocity',
        correction_reverse_profile='endpoint_only_max0.05_mps_existing_authorization',
        external_deadline_s=args.duration,evidence_scope='controlled physical-input integration: confirmed Nov06 composed camera TF with the same frozen legacy-trained depth_ground_reference_v1 scalar; no refit, no change to nominal forecasts, scorer, radius or thresholds; no matched efficacy or independently verified geometric accuracy',
        prior_episode_ledger_imported=False,history_acquisition='fresh per-trial isolated navigation map; original odometry and display retained; actual fresh paid observations')
    settings=dict(configuration=cfg,execution_allowed=True,calibration=calibration,semantic_configuration=json.loads((WORK/'config/real_semantic_frontend.json').read_text()))
    save(OUT/'configuration.json',settings)
    save(OUT/'source.json',dict(raw_observation=str(raw_path),joint_input=str(packet),grid=grid,no_motion=True,actual_CAN_snapshot=snapshot(),camera_TF_receipt=str(CAMERA_RECEIPT),camera_TF_receipt_sha256=hashlib.sha256(CAMERA_RECEIPT.read_bytes()).hexdigest(),ground_profile_sha256=hashlib.sha256(PROFILE.read_bytes()).hexdigest(),template=str(template),template_sha256=hashlib.sha256(template.read_bytes()).hexdigest(),inherited_configuration_fields=list(cfg),old_calibration_fields_copied=False))
    print(json.dumps(dict(configuration=str(OUT/'configuration.json'),pose=pose,CAN=snapshot())))
finally:
    collector.close()
    if joint is not None:joint.close()
    stop.set();thread.join(1);can.close();rospy.signal_shutdown('fresh Nov06 trial capture complete')
