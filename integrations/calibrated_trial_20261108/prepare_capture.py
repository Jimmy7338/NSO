"""Actual ROS capture and matched semantic-profile replay; no motion sender."""
import argparse
import copy
import hashlib
import math
import json
from pathlib import Path
import socket
import struct
import sys
import threading
import time

import numpy as np

WORK = Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029')
ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--budget', type=int, default=24)
parser.add_argument('--episode', default='calibrated_S24_01')
args = parser.parse_args()
if args.budget < 1: parser.error('budget must be positive')
OUT = args.output.resolve()
TEMPLATE = Path('/home/ouc/nso_robot/analyses/mount_unmasked_20261107/fresh_start/configuration.json')
CALIBRATION = Path('/home/ouc/nso_robot/analyses/calibrated_mount_20261108/tf_verification.json')
sys.path.insert(0, str(WORK / 'runtime'))
import rospy
import tf2_ros
from reference_observation import ReferenceObservationCollector
from joint_observation import JointObservationCollector, freeze_joint_grid
from dashboard_overlay_server import camera_calibration_metadata


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


OUT.mkdir(parents=True,exist_ok=False)
rospy.init_node('nso_calibrated_trial_fresh_capture', disable_signals=True, disable_rosout=True)
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


thread = threading.Thread(target=receive, daemon=True); thread.start()
collector = ReferenceObservationCollector(tfbuf)
joint = None
try:
    raw_path = collector.capture_live_joint(OUT / 'observed', lambda: None, snapshot)
    with np.load(raw_path, allow_pickle=False) as z:
        receipt = json.loads(str(z['receipt_json']))
    historical = receipt['historical_tf']
    base = np.asarray(historical['odom_from_base']['matrix'])
    pose = [float(base[0,3]), float(base[1,3]), float(np.arctan2(base[1,0],base[0,0]))]
    actual_camera = np.asarray(historical['base_from_camera']['matrix'])
    calibration_status = camera_calibration_metadata(CALIBRATION, actual_camera)
    if not calibration_status['camera_extrinsics_loaded']:
        raise ValueError('captured camera TF does not match Nov06 calibration: '+str(calibration_status))
    grid = freeze_joint_grid(pose)
    joint = JointObservationCollector(tfbuf, grid, apply_approved_self_mask=False)
    path = joint.capture(OUT/'joint_inputs',raw_path,source_role='fresh_calibrated_Nov08_trial_initial_observation_no_motion')
    template = json.loads(TEMPLATE.read_text())
    original = template['configuration']
    # Whitelist constants; do not inherit a prior calibration, transform or ledger.
    configuration = {k:copy.deepcopy(original[k]) for k in ('robot_radius_m','sensor_config','planner_revision')}
    configuration.update(policy_variant='exploration_joint_v1',sensor_range_profile='sensor_range_matched_v1',
        measured_extent_profile='measured_extent_3m_v1',shape=grid['shape'],planning_from_odom=grid['planning_from_odom'],
        return_anchor=grid['return_anchor'],total_budget=args.budget,score_mode='S',episode_id=args.episode,
        scene_id='real_unknown_calibrated_20261108')
    source_state = dict(localization_source=rospy.get_param('/nso_visual/localization_source',None),
        active_matcher='/nso_csm/matcher',matcher_parameters=rospy.get_param('/nso_csm/matcher',{}),
        control_velocity_source=rospy.get_param('/nso_csm_odometry_bridge/control_velocity_source',None),
        pose_source=rospy.get_param('/nso_csm_odometry_bridge/pose_source',{}),
        controller_odom_topic=rospy.get_param('/nso_visual/move_base/DWAPlannerROS/odom_topic',None),
        camera_extrinsics_loaded=True,camera_extrinsics_verified=False,old_camera_TF_retained=False,
        map_and_TSDF_epoch_retained_within_Nov08_session=True,prior_episode_ledger_imported=False,
        semantic_model_executed_during_preparation=False,preparation_no_motion=True)
    calibration = dict(source='actual ZED K and actual camera/laser historical TF retained per capture',
        pixel_stride=4,raw_observations_preserved=True,initial_grid=grid,initial_actual_base_odom_pose=pose,
        initial_odom_stamp_ns=historical['odom_from_base']['query_stamp_ns'],
        initial_odom_stamp_semantics='actual historical TF query stamp',initial_odom_frame='odom',
        initialization_unix_s=time.time(),initialization_scope='fresh actual Nov08 observation and planning anchor; no motion sender',
        initialization_evidence='new actual RGB-D/scan/map packet, not a reused paid/semantic observation ledger',
        initialization_source_state=source_state,localization_source='lidar_csm',
        source_state='Nov08 fresh original-default-keyframe CSM epoch; explicit original-kernel CAN control velocity',
        source_registration='Nov08 session coordinates; no prior fixed SE2 alignment or historical registration imported',
        coordinate_alignment_scope='fresh local planning transform anchored to actual current historical TF only',
        history_acquisition='no prior joint/paid/semantic ledger imported; measured navigation map retained within Nov08 session and may contain manual-driving observations; not a blank initial map',
        navigation_map_source=dict(topic='/nso_visual/map',publisher='/slam_gmapping',map_frame='map'),
        controller_odometry_source='/nso_csm/control_odom',
        controller_odometry_projection='original CSM pose/source stamps; explicit kernel-timestamped CAN vx/w at or before scan with150ms TTL, vy=0',
        joint_turn_execution_profile='measured_in_place_turn_v1',apply_approved_self_mask=False,
        self_mask_pixels=0,self_mask_profile='unmasked_after_camera_relocation_v1',
        camera_mount_calibrated=True,camera_mount_user_confirmed_unchanged_since_Nov06=True,
        camera_calibration_receipt=str(CALIBRATION),camera_calibration_receipt_sha256=hashlib.sha256(CALIBRATION.read_bytes()).hexdigest(),
        camera_calibration_status=calibration_status,camera_extrinsic_policy='Nov06 left calibration loaded and matched to actual capture historical TF',
        captured_base_from_camera=actual_camera.tolist(),
        captured_base_from_laser=historical['base_from_laser']['matrix'],
        historical_semantic_source='no old semantic history; pinned frontend infers actual trial observations',
        semantic_labels_available='model predictions, not ground truth',nominal_view_forecast_calibrated=False,
        physical_reconstruction_accuracy_verified=False,evidence_scope='calibrated-mount closed-loop integration; no semantic causal benefit or reconstruction accuracy established',
        SDK_depth_mode=rospy.get_param('/zed2i/zed_node/depth/depth_mode'),
        depth_confidence=rospy.get_param('/zed2i/zed_node/depth_confidence'),
        depth_texture_confidence=rospy.get_param('/zed2i/zed_node/depth_texture_conf'),
        ROS_execution_frame='odom',ROS_xy_latch=True,ROS_xy_goal_tolerance_m=.08,Navfn_default_tolerance=0,
        ROS_short_goal_forward_point_distance_m=rospy.get_param('/nso_visual/move_base/DWAPlannerROS/forward_point_distance'),
        execution_contract='ROS success and actual original pending grid state; metric error retained as diagnostic',
        execution_controller_profile='exact_ros_goal_v1',
        correction_reverse_profile='nso_nominal_correction_reverse_v1; same-pending endpoint retry only, maxreverse .05m/s; ordinary motion no reverse',
        sparse_actual_RGBD_observation_allowed=True,
        inherited_constant_fields=['configuration.robot_radius_m','configuration.sensor_config','configuration.planner_revision'],
        inherited_calibration_fields=[],
        nominal_camera_height_m=configuration['sensor_config']['camera_height_m'],
        actual_camera_height_from_base_TF_and_fixed_floor_m=float(actual_camera[2,3]-grid['floor_odom_z_m']),
        camera_height_comparison_scope='Nominal policy height retained; actual base-relative optical TF plus existing fixed floor offset is not an independently measured floor/accuracy calibration',
        nominal_sensor_constant_limit='Original nominal camera/laser heights and freeze_joint_grid floor -0.233 retained; these are not newly measured floor/nominal forecast calibration. Actual sensor projection uses captured K/TF.')
    settings=dict(configuration=configuration,execution_allowed=True,calibration=calibration,
        semantic_configuration=json.loads((WORK/'config/real_semantic_frontend.json').read_text()))
    save(OUT/'configuration.json',settings)
    files=[Path(raw_path),Path(path),TEMPLATE,CALIBRATION,Path(__file__),WORK/'config/real_semantic_frontend.json',OUT/'configuration.json']
    save(OUT/'source.json',dict(raw_observation=str(raw_path),joint_input=str(path),grid=grid,no_motion=True,
        actual_CAN_snapshot=snapshot(),initialization_source_state=source_state,
        reference_script='/home/ouc/nso_robot/analyses/exploration_resume_20261107/capture_fresh_S40.py',
        template_scope='whitelisted configuration constants only; no old calibration fields copied',
        planned_duration_s=600,files=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]))
    print(json.dumps(dict(configuration=str(OUT/'configuration.json'),CAN=snapshot(),pose=pose,budget=args.budget)))
finally:
    collector.close()
    if joint is not None:joint.close()
    stop.set();thread.join(1);can.close()
    rospy.signal_shutdown('actual capture complete')
