"""Fresh physical pair initialization; measured isolated map, no motion sender."""
import argparse
import copy
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
parser.add_argument('--map-source-receipt', type=Path, required=True)
parser.add_argument('--grid-from', type=Path, help='First arm configuration fixing the common planning frame')
args = parser.parse_args()
OUT = args.output
map_source = dict(topic='/nso_pair/map',publisher='/nso_pair/gmapping',map_frame='nso_pair_map')
map_receipt=json.loads(args.map_source_receipt.read_text())
sys.path.insert(0, str(WORK / 'runtime'))
import rospy
import tf2_ros
from reference_observation import ReferenceObservationCollector
from joint_observation import JointObservationCollector, freeze_joint_grid
from reference_transport import ReferenceWorker
from visual_sensor_bridge import load_approved_mask


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


OUT.mkdir(parents=True,exist_ok=False)
rospy.init_node('nso_pair_fresh_capture', disable_signals=True, disable_rosout=True)
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
    base = np.asarray(receipt['historical_tf']['odom_from_base']['matrix'])
    pose = [base[0, 3], base[1, 3], np.arctan2(base[1, 0], base[0, 0])]
    grid = (json.loads(args.grid_from.read_text())['calibration']['initial_grid']
            if args.grid_from else freeze_joint_grid(pose))
    joint = JointObservationCollector(tfbuf, grid, **map_source, apply_approved_self_mask=False)
    path = joint.capture(OUT / 'joint_inputs', raw_path, source_role='fresh_isolated_mapping_pair_initialization_no_motion')
    settings = json.loads(Path('/home/ouc/nso_robot/analyses/exploration_resume_20261107/fresh_start/configuration.json').read_text())
    _, mask_receipt = load_approved_mask()
    settings['configuration'].update(total_budget=40, sensor_range_profile='sensor_range_matched_v1', measured_extent_profile='measured_extent_3m_v1', policy_variant='exploration_joint_v1', shape=grid['shape'], planning_from_odom=grid['planning_from_odom'],
        return_anchor=grid['return_anchor'], episode_id='nso_exploration_resume_S40_20261107_01')
    settings['calibration'].update(initial_grid=grid, initialization_unix_s=time.time(),
        initialization_scope='fresh actual capture at current aligned odom pose; original matcher/odom/display-TSDF retained; independent fresh experiment map; preparation only, no actuator or robot action', joint_turn_execution_profile='measured_in_place_turn_v1',
        self_mask_asset_directory=str(Path(mask_receipt['asset_path']).parent),
        self_mask_file_sha256=mask_receipt['file_sha256'], self_mask_pixels=0, apply_approved_self_mask=False, self_mask_profile='unmasked_after_camera_relocation_v1', camera_mount_calibrated=False, camera_extrinsic_policy='user_deferred_update_reusing_old_TF_integration_only', evidence_scope='uncalibrated_camera_integration_only_no_accuracy_or_benefit_claim', depth_confidence=80, SDK_depth_mode='NEURAL', initialization_evidence='one fresh actual RGBD/scan/current-map packet; no prior paid or semantic observation ledger imported')
    settings.pop('shared_observed_prefix', None)
    settings['calibration']['initial_odom_stamp_ns'] = receipt['historical_tf']['odom_from_base']['query_stamp_ns']
    settings['calibration']['initial_odom_stamp_semantics'] = 'actual historical TF query stamp'
    settings['calibration']['source_state'] = 'original uninterrupted default-keyframe CSM, CAN control velocity and fixed SE2 odom coordinate continuity after bounded B diagnostic'
    settings['calibration']['source_registration'] = 'aligned odom/display-TSDF retained; experiment map has fresh independent epoch; planning origin frozen from first arm actual base pose'
    settings['calibration']['history_acquisition'] = 'no old joint/paid/semantic ledger imported; new independent measured ROS experiment map; old display map is excluded from research input/navigation; no prior paid or semantic ledger imported'
    settings['configuration']['scene_id'] = 'real_unknown_reachable_20261107'
    settings['execution_allowed'] = True
    settings['semantic_configuration'] = json.loads((WORK/'config/real_semantic_frontend.json').read_text())
    source_state = dict(
        localization_source='lidar_csm', active_matcher='/nso_csm/matcher',
        matcher_parameters=rospy.get_param('/nso_csm/matcher', {}),
        effective_keyframe_parameters=dict(
            kf_dist_linear=rospy.get_param('/nso_csm/matcher/kf_dist_linear', .1),
            kf_dist_angular=rospy.get_param('/nso_csm/matcher/kf_dist_angular', float(np.pi/18))),
        control_velocity_source=rospy.get_param('/nso_csm_odometry_bridge/control_velocity_source', None),
        pose_source=rospy.get_param('/nso_csm_odometry_bridge/pose_source', {}),
        controller_odom_topic=rospy.get_param('/nso_visual/move_base/DWAPlannerROS/odom_topic', None),
        camera_extrinsics_verified=False, old_camera_TF_retained=True,
        old_approved_self_mask_applied=False, semantic_model_executed_during_preparation=False,
        semantic_configuration_reused=True,
        map_and_TSDF_epoch_retained=False, display_TSDF_epoch_retained=True,
        experiment_map_source=map_source, experiment_map_receipt=map_receipt, preparation_no_motion=True)
    settings['calibration'].update(
        initialization_source_state=source_state,
        controller_odometry_projection='unfiltered original CSM registration in a fixed SE2-aligned odom frame; source stamps preserved; control vx/w are measured original-kernel CAN at or before scan with150ms TTL; control vy=0; raw derived CSM twist retained',
        historical_semantic_source='no semantic predictions or history imported in preparation; original pinned frontend will infer from actual runtime observation',
        initial_odom_frame='odom',
        coordinate_alignment_scope='fixed same-stamp SE2 coordinate continuity, not independent accuracy correction')
    settings['calibration']['navigation_map_source']=map_source
    settings['calibration']['initial_base_odom_pose_this_arm']=list(map(float,pose))
    settings['calibration']['paired_map_epoch_receipt']=map_receipt
    settings['calibration']['common_grid_source']=str(args.grid_from.resolve()) if args.grid_from else None
    save(OUT/'source.json', dict(raw_observation=raw_path, joint_input=path, grid=grid,
        no_motion=True, actual_CAN_snapshot=snapshot(), initialization_source_state=source_state,
        original_reference_script='/home/ouc/nso_robot/analyses/exploration_resume_20261107/capture_fresh_S40.py',
        experiment_map_source=map_source, experiment_map_receipt_path=str(args.map_source_receipt.resolve())))
    save(OUT/'configuration.json', settings)
    print(json.dumps(dict(configuration=str(OUT/'configuration.json'),CAN=snapshot(),pose=pose)))
finally:
    collector.close()
    if joint is not None: joint.close()
    stop.set(); thread.join(1); can.close()
    rospy.signal_shutdown('actual capture complete')
