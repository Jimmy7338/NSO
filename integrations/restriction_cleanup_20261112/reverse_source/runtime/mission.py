#!/usr/bin/env python3
"""Standard ROS navigation supervisor; the NSO reference entry awaits real task inputs."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from stop_feedback_wait import stop_feedback_cleanup_sequence
from standby_native_handshake import NativeStartOnce
from lifecycle_receipt import LifecycleReceipt
from duration_profile import STANDARD, PROFILES, validate_duration

WORK = Path(__file__).resolve().parents[1]
ROOT = Path('/home/ouc/nso_robot')
MASTER = 'http://127.0.0.1:11382'
NATIVE_STAGE = ROOT/'staging/ans-live-20261110-snapshot-v1'
NATIVE = NATIVE_STAGE/'bin/nso_scout_attended_node'
SDK = ROOT/'staging/scout-owned-sdk-build-20261009-r0/workspace/devel/lib'

def paid_frame_source_provenance(rgb, depth, info, science_tf, navigation_tf, payloads):
    """Record actual synchronized headers/TF responses, without inventing times.

    The hashes describe the exact arrays that will be persisted, including the
    unfilled measured depth with its original invalid values. Physical camera
    calibration remains a separate claim from this source-alignment receipt.
    """
    import numpy as np
    package_path=str(WORK/'nso_bridge')
    if package_path not in sys.path:sys.path.insert(0,package_path)
    from nso_bridge.source_provenance import SCHEMA, array_receipt, tf_timestamp_rounding_bound_ns
    messages = dict(rgb=rgb, depth=depth, camera_info=info)
    stamps = {kind: int(message.header.stamp.to_nsec()) for kind, message in messages.items()}
    frames = {kind: message.header.frame_id for kind, message in messages.items()}
    query = stamps['depth']
    if query <= 0 or len(set(stamps.values())) != 1:
        raise ValueError('paid source headers are not exactly synchronized')
    optical = frames['depth']
    if not isinstance(optical, str) or not optical or len(set(frames.values())) != 1:
        raise ValueError('paid source optical frames differ')
    height, width = np.asarray(payloads['depth_m']).shape
    if np.asarray(payloads['rgb']).shape != (height, width, 3):
        raise ValueError('paid source RGB and depth array shapes differ')
    sources = {}
    for kind, message in messages.items():
        if (message.height, message.width) != (height, width):
            raise ValueError('paid source '+kind+' dimensions differ')
        row = dict(stamp_ns=stamps[kind], frame_id=frames[kind],
                   width=int(message.width), height=int(message.height))
        if kind != 'camera_info':
            row.update(encoding=message.encoding, step=int(message.step),
                       is_bigendian=int(message.is_bigendian),
                       raw_message_data_sha256=hashlib.sha256(bytes(message.data)).hexdigest())
        else:
            row.update(K=[float(x) for x in message.K], P=[float(x) for x in message.P],
                       R=[float(x) for x in message.R], D=[float(x) for x in message.D],
                       distortion_model=message.distortion_model,
                       binning_x=int(message.binning_x), binning_y=int(message.binning_y),
                       roi=dict(x_offset=int(message.roi.x_offset), y_offset=int(message.roi.y_offset),
                                width=int(message.roi.width), height=int(message.roi.height),
                                do_rectify=bool(message.roi.do_rectify)))
        sources[kind] = row
    transforms = {}
    for name, target, transform in [('world_from_camera', 'odom', science_tf),
                                    ('map_from_camera', 'map', navigation_tf)]:
        if transform.header.frame_id != target or transform.child_frame_id != optical:
            raise ValueError('paid historical TF response frames differ')
        actual_stamp = int(transform.header.stamp.to_nsec())
        # These chains contain dynamic odometry, so zero is invalid. tf2's
        # Python/C++ time boundary may round an epoch double by submicroseconds;
        # retain both raw integers and accept only its bounded numeric precision.
        rounding_bound=tf_timestamp_rounding_bound_ns(query)
        difference=actual_stamp-query
        if actual_stamp<=0 or abs(difference)>rounding_bound:
            raise ValueError('paid dynamic historical TF response exceeds serialization rounding bound: '
                             +str(dict(query_ns=query,response_ns=actual_stamp,delta_ns=difference,bound_ns=rounding_bound)))
        value = np.asarray(payloads[name])
        if value.shape != (4, 4) or not np.isfinite(value).all():
            raise ValueError('paid historical TF matrix invalid')
        transforms[name] = dict(target_frame=target, source_frame=optical,
            query_stamp_ns=query, response_stamp_ns=actual_stamp,
            response_query_delta_ns=difference,maximum_serialization_rounding_ns=rounding_bound,
            response_target_frame=transform.header.frame_id,
            response_source_frame=transform.child_frame_id,
            timestamp_semantics='historical_lookup_response_header', matrix=value.tolist())
    if not np.array_equal(np.asarray(info.K).reshape(3, 3), payloads['intrinsic']):
        raise ValueError('paid intrinsic differs from synchronized CameraInfo')
    return dict(schema=SCHEMA, frame_id=str(query), world_frame='odom', navigation_frame='map',
        navigation_source='lidar_rgbd', pose_source='lidar_navigation_odometry',
        sources=sources, synchronization=dict(policy='exact_header_stamp', query_stamp_ns=query,
            all_header_stamps_equal=True, all_optical_frames_equal=True),
        transforms=transforms, payload_hashes={name:array_receipt(value) for name,value in payloads.items()},
        physical_extrinsic_calibration_verified=False)


class HealthWindow:
    """One loss episode spans reason changes; preparation needs stable health."""
    def __init__(self, grace_s=1., stable_s=1.):
        self.grace_s,self.stable_s=grace_s,stable_s
        self.bad_since=self.good_since=None

    def update(self, reason, now, preparing=False):
        if reason:
            self.good_since=None
            if self.bad_since is None:self.bad_since=now
            return 'expired' if not preparing and now-self.bad_since>=self.grace_s else 'waiting'
        self.bad_since=None
        if self.good_since is None:self.good_since=now
        return 'waiting' if preparing and now-self.good_since<self.stable_s else 'ready'


class ActuatorTelemetry:
    """Read-only mirror of the separate adapter's original CAN accounting.

    Heavy image processing in this supervisor cannot stall CAN reception or
    alter the adapter/native command watchdogs.
    """
    def __init__(self,session_id):
        self.session_id=session_id;self.receipt=None;self.fault=None;self.ever_armed=False
        self.can_distance=0.;self.can_absolute_yaw=0.;self.can_zero_since=None
        self.can_accounting_complete=False;self.last_hardware_reason='adapter_telemetry_missing'
        self.can_status=self.can_velocity=None
        self.driver_connected=False

    def update(self,row,publisher,receipt):
        if publisher!='/nso_radius_guard' or row.get('session_id')!=self.session_id:
            raise ValueError('actuator telemetry session/publisher differs')
        if not 0<=receipt-row['sampled_monotonic']<=.5:
            raise ValueError('actuator telemetry snapshot is stale or future')
        self.receipt=receipt;self.fault=row.get('fault');self.ever_armed=row['ever_armed']
        self.can_distance=row['can_distance_m'];self.can_absolute_yaw=row['can_absolute_yaw_rad']
        self.can_zero_since=row['can_zero_since_monotonic']
        self.can_accounting_complete=row['feedback_accounting_complete']
        self.can_status=row.get('can_status');self.can_velocity=row.get('can_velocity')
        self.driver_connected=row.get('driver_connected') is True
        self.last_hardware_reason=row.get('hardware_reason')

    def hardware_reason(self,now):
        if self.receipt is None or not 0<=now-self.receipt<=.5:return 'adapter_telemetry_stale'
        for name,row in [('status',self.can_status),('velocity',self.can_velocity)]:
            if row is None or not 0<=now-row['source']<=.5:return 'original_CAN_'+name+'_stale'
        return self.last_hardware_reason


def visual_health_reason(health, odom, grid, now, wall):
    if not health or not 0<=now-health[0]<=.5:return 'bridge_health_receipt_stale'
    for kind in ['odom','depth']:
        row=health[1].get(kind,{})
        if row.get('fault'):raise RuntimeError('visual_'+kind+'_fault:'+str(row['fault']))
        if not row.get('valid'):return kind+':'+str(row.get('reason','invalid'))
        stamp=row.get('source_stamp_ns')
        if not isinstance(stamp,(int,float)) or not math.isfinite(stamp) or not -.02<=wall-stamp/1e9<=.5:
            return kind+'_source_stamp_stale_or_future'
    if not odom or not 0<=now-odom[0]<=.5:return 'visual_odometry_receipt_stale'
    if not -.02<=wall-odom[1].header.stamp.to_sec()<=.5:return 'visual_odometry_source_stale_or_future'
    # OccupancyGrid is latched and may stay unchanged while the robot waits.
    # Runtime checks fresh RTAB Info separately; old map receipt is not loss.
    if not grid:return 'visual_map_missing'
    return None


def hybrid_health_reason(health, odom, grid, now, wall):
    """Require the actual laser source in addition to aligned RGB-D and odom.

    A fresh bridge receipt cannot refresh the original laser timestamp. The
    map may be latched; navigation freshness comes from source health and TF,
    not from a fictitious RTAB processing heartbeat in this lidar pipeline.
    """
    reason = visual_health_reason(health, odom, grid, now, wall)
    if reason:
        return reason
    if health[1].get('navigation_source') != 'lidar_rgbd':
        return 'navigation_source_not_lidar_rgbd'
    row = health[1].get('laser_source', {})
    if row.get('fault'):
        raise RuntimeError('laser_source_fault:' + str(row['fault']))
    if row.get('valid') is not True:
        return 'laser_source:' + str(row.get('reason', 'invalid'))
    stamp = row.get('source_stamp_ns')
    if (isinstance(stamp, bool) or not isinstance(stamp, (int, float))
            or not math.isfinite(stamp) or not -.02 <= wall-stamp/1e9 <= .5):
        return 'laser_source_stamp_stale_or_future'
    return None


def recoverable_sensor_health_reason(reason, health, odom, now, wall):
    """Exact stale-only admission; never turn future/invalid/fault into recovery."""
    if reason not in {'bridge_health_receipt_stale', 'visual_odometry_receipt_stale',
            'odom_source_stamp_stale_or_future', 'depth_source_stamp_stale_or_future',
            'laser_source_stamp_stale_or_future', 'visual_odometry_source_stale_or_future',
            'odom:odom_source_stale_or_missing', 'depth:depth_stale_or_missing',
            'laser_source:laser_source_stale_or_missing'}:
        return False
    if health is None or odom is None:
        return False
    if not (math.isfinite(now) and math.isfinite(wall) and health[0] <= now and odom[0] <= now):
        return False
    for kind in ('odom', 'depth', 'laser_source'):
        row=health[1].get(kind,{})
        stamp=row.get('source_stamp_ns')
        stale_reason={'odom':'odom_source_stale_or_missing', 'depth':'depth_stale_or_missing',
                      'laser_source':'laser_source_stale_or_missing'}[kind]
        if (row.get('fault') or (row.get('valid') is not True and row.get('reason') != stale_reason)
                or type(stamp) not in (int,float)
                or not math.isfinite(stamp) or stamp<=0 or stamp/1e9>wall+.02):
            return False
    stamp=odom[1].header.stamp.to_sec()
    return math.isfinite(stamp) and 0<stamp<=wall+.02


def load_experiment_config(path):
    """Read exact experiment bytes once; the same bytes are frozen per attempt."""
    if path is None:
        return None, None, None
    raw = Path(path).read_bytes()
    def reject_nonfinite(value):
        raise ValueError('nonfinite experiment JSON constant: ' + value)
    config = json.loads(raw, parse_constant=reject_nonfinite)
    if not isinstance(config, dict):
        raise ValueError('experiment config must be an object')
    if config.get('schema') != 'nso_physical_experiment/1':
        raise ValueError('unsupported experiment schema')
    if not isinstance(config.get('experiment_id'), str) or not config['experiment_id'].strip():
        raise ValueError('nonempty experiment_id required')
    if config.get('navigation_source') != 'lidar_rgbd':
        raise ValueError('experiment navigation_source must be lidar_rgbd')
    from nso_ros_coordinator import validate_experiment_config
    config = validate_experiment_config(config)
    return config, raw, hashlib.sha256(raw).hexdigest()


def pose_evidence(transform, wall, receipt, target_frame='map'):
    h=transform.header;t=transform.transform.translation;q=transform.transform.rotation
    stamp=h.stamp.to_sec()
    values=[t.x,t.y,t.z,q.x,q.y,q.z,q.w,stamp,wall,receipt]
    if not all(math.isfinite(v) for v in values):raise ValueError('nonfinite_map_pose')
    if h.frame_id!=target_frame or transform.child_frame_id!='base_link':raise ValueError('map_pose_frames_differ')
    if stamp<=0 or not -.02<=wall-stamp<=.5:raise ValueError('map_pose_TF_stale_or_future')
    if abs(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w-1.)>.01:raise ValueError('map_pose_quaternion_invalid')
    return dict(pose=[t.x,t.y,math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))],
                source_stamp_s=stamp,source_age_s=wall-stamp,receipt_monotonic=receipt,
                source_stamp_ns=h.stamp.to_nsec() if hasattr(h.stamp,'to_nsec') else None,
                target_frame=target_frame,source_frame='base_link',
                translation=[t.x,t.y,t.z],quaternion=[q.x,q.y,q.z,q.w])


def arrival_recording_frame(mode, execution_frame, goal_frame):
    """Native exploration follows its costmap frame; NSO keeps its declared frame.

    This selects the TF used for arrival records only, not an execution target
    or a motion-admission condition. Unsupported frames remain unverified.
    """
    return goal_frame if mode == 'visual' else execution_frame


def arrival_evidence(goal, terminal, before, after, can_start, can_end, expected_frame='map'):
    """Verify the ROS result against fresh pose and chassis feedback, including turns."""
    evidence=dict(goal=goal,terminal=terminal,before=before,after=after,can_start=can_start,can_end=can_end,
                  verified=False,xy_tolerance_m=.15,yaw_tolerance_rad=.35)
    reason=None
    if goal is None or goal.get('id')!=terminal.get('id'):reason='goal_id_missing_or_different'
    elif terminal.get('status')!=3:reason='action_not_succeeded'
    elif expected_frame not in ('map','odom'):reason='unsupported_execution_frame'
    elif goal.get('frame')!=expected_frame:reason='target_frame_not_'+expected_frame
    elif not before or not after:reason='fresh_pose_evidence_missing'
    elif any(row.get('target_frame')!=expected_frame for row in [before,after]):reason='pose_frame_different'
    elif any(not -.02<=row['source_age_s']<=.5 for row in [before,after]):reason='pose_source_not_fresh'
    elif after['source_stamp_s']<=before['source_stamp_s']:reason='pose_source_time_not_advanced'
    elif after['receipt_monotonic']<=before['receipt_monotonic']:reason='pose_receipt_time_not_advanced'
    elif not can_start or not can_end:reason='CAN_evidence_missing'
    else:
        xy=math.hypot(after['pose'][0]-goal['x'],after['pose'][1]-goal['y'])
        yaw=abs(math.atan2(math.sin(after['pose'][2]-goal['yaw']),math.cos(after['pose'][2]-goal['yaw'])))
        displacement=math.hypot(after['pose'][0]-before['pose'][0],after['pose'][1]-before['pose'][1])
        distance=can_end['distance_m']-can_start['distance_m']
        angle=can_end['absolute_yaw_rad']-can_start['absolute_yaw_rad']
        evidence.update(target_error_m=xy,target_yaw_error_rad=yaw,displacement_m=displacement,
                        CAN_distance_delta_m=distance,CAN_absolute_yaw_delta_rad=angle)
        if not all(math.isfinite(v) for v in [xy,yaw,displacement,distance,angle]):reason='nonfinite_arrival_evidence'
        elif not can_start['accounting_complete'] or not can_end['accounting_complete']:reason='CAN_accounting_incomplete'
        elif not can_start['feedback_fresh'] or not can_end['feedback_fresh']:reason='CAN_feedback_not_fresh'
        elif distance<0 or angle<0:reason='CAN_counters_regressed'
        elif xy>.15 or yaw>.35:reason='target_pose_error'
    evidence.update(verified=reason is None,reason=reason or 'verified_ROS_goal_pose_and_fresh_chassis_feedback')
    return evidence


def unexpected_child_exit(name, returncode, explorer_stopped):
    return returncode is not None and name!='worker' and not (name=='explore' and explorer_stopped)

def navigation_start_footprint(grid, transform):
    """Read-only geometry for manual probes; not used for motion admission."""
    import numpy as np,cv2
    from nso_ros_coordinator import matrix
    if transform.header.frame_id!=grid.header.frame_id or transform.child_frame_id!='base_link':
        raise ValueError('local costmap and base transform frames differ')
    T=matrix(transform.transform)
    corners=np.array([[-.432,-.3325,0.,1.],[.64,-.3325,0.,1.],
                      [.64,.3325,0.,1.],[-.432,.3325,0.,1.]])@T.T
    origin=np.array([grid.info.origin.position.x,grid.info.origin.position.y])
    cells=np.floor((corners[:,:2]-origin)/grid.info.resolution).astype(np.int32)
    a=np.asarray(grid.data,dtype=np.int16).reshape(grid.info.height,grid.info.width)
    inside=bool(np.all(cells>=0) and np.all(cells[:,0]<a.shape[1]) and np.all(cells[:,1]<a.shape[0]))
    mask=np.zeros(a.shape,np.uint8);cv2.fillPoly(mask,[cells],1);values=a[mask.astype(bool)]
    unknown=int((values<0).sum());lethal=int((values>=100).sum())
    return dict(clear=bool(inside and len(values) and not unknown and not lethal),
        frame=grid.header.frame_id,costmap_source_stamp_s=grid.header.stamp.to_sec(),
        base_transform_source_stamp_s=transform.header.stamp.to_sec(),
        grid_origin_xy=origin.tolist(),resolution_m=grid.info.resolution,
        corners_world=corners[:,:2].tolist(),all_vertices_inside=inside,
        unknown_cells=unknown,lethal_cells=lethal,
        includes_actual_layered_local_grid=True,unknown_assumed_free=False,
        free_cell_source='layered navigation map, including standard robot footprint clearing; not all cells certified sensor-measured free')

def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)

def module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def rgb_array(message):
    """Decode original ROS byte strides without the legacy cv_bridge ABI."""
    import numpy as np
    channels={'rgb8':3,'bgr8':3,'rgba8':4,'bgra8':4}.get(message.encoding)
    if channels is None or message.step<message.width*channels or len(message.data)<message.height*message.step:
        raise ValueError('unsupported or truncated RGB image')
    a=np.ndarray((message.height,message.width,channels),dtype=np.uint8,
                 buffer=message.data,strides=(message.step,channels,1))[:,:,:3]
    return (a[:,:,::-1] if message.encoding.startswith('bgr') else a).copy()

def depth_array(message):
    import numpy as np
    types={'32FC1':('f4',4,1.),'16UC1':('u2',2,.001)}
    if message.encoding not in types:raise ValueError('unsupported depth encoding')
    dtype,size,scale=types[message.encoding]
    if message.step<message.width*size or len(message.data)<message.height*message.step:
        raise ValueError('truncated depth image')
    dtype=np.dtype(('>' if message.is_bigendian else '<')+dtype)
    a=np.ndarray((message.height,message.width),dtype=dtype,buffer=message.data,
                 strides=(message.step,size)).astype(np.float32)*scale
    a[~np.isfinite(a)|(a<=0)]=np.nan
    return a

def other_senders():
    names = {'scout_base_node','nso_scout_attended_node','nso_scout_radius_node_v1',
             'attended_velocity_adapter.py','visual_velocity_adapter.py'}
    found = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():
            continue
        try:
            args=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x]
            if names.intersection(Path(a).name for a in args if len(a)<512):
                found.append(dict(pid=int(p.name),args=args))
        except OSError:
            pass
    return found

def exploration_arguments():
    """Run the pinned explorer directly: ROS1 roslaunch rejects absolute node types."""
    tree=ET.parse(WORK/'config/visual_exploration.launch').getroot()
    nodes=tree.findall('.//node')
    if len(nodes)!=1 or nodes[0].get('pkg')!='explore_lite':
        raise ValueError('exactly one explore_lite node required')
    node=nodes[0];binary=Path(node.get('type',''))
    if not binary.is_absolute() or not binary.is_file() or not os.access(binary,os.X_OK):
        raise ValueError('pinned explore binary unavailable')
    command=[str(binary),'__ns:=/nso_visual','__name:=explore']
    for item in node:
        if item.tag!='param' or set(item.attrib)!={'name','value'} or '$(' in item.get('value',''):
            raise ValueError('literal explorer parameters required')
        command.append('_'+item.get('name')+':='+item.get('value'))
    return command

def write_source_manifest(output):
    """Identify the active implementation without copying its tree or model."""
    paths=[WORK/'run_hybrid.sh']
    for name in ('runtime','config','nso_bridge/nso_bridge'):
        paths.extend(p for p in (WORK/name).glob('*') if p.is_file()
                     and p.suffix in ('.py','.yaml','.json','.launch'))
    vendor=WORK/'nso_bridge/vendor_nso'
    paths.extend(p for p in vendor.rglob('*.py') if '__pycache__' not in p.parts)
    paths.append(vendor/'source_manifest.json')
    joint_vendor=WORK/'nso_bridge/joint_vendor'
    paths.extend(p for p in joint_vendor.rglob('*.py') if '__pycache__' not in p.parts)
    paths.append(joint_vendor/'SOURCE_MANIFEST.json')
    paths.extend(p for p in (WORK/'config/assets').rglob('*') if p.is_file()
                 and p.suffix in ('.npy','.json'))
    rows=[dict(path=str(p.relative_to(WORK)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
          for p in sorted(set(paths)) if p.is_file()]
    native_paths=dict(node_source=NATIVE_STAGE/'native/radius_driver_node.cpp',
        gate_header=NATIVE_STAGE/'native/scout_radius_command_timeout_v1.hpp',
        binary=NATIVE, sdk=SDK/'libugv_sdk.so', profile_manifest=NATIVE_STAGE/'manifest.json')
    native_identity={k:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None)
                     for k,p in native_paths.items()}
    native_profile=json.loads((NATIVE_STAGE/'manifest.json').read_text())['profile']
    save(output/'source_manifest.json',dict(files=rows,source_root=str(WORK),
        native_sender=dict(profile=native_profile,files=native_identity,
            sdk_resolution='SDK directory prepended to LD_LIBRARY_PATH at native spawn'),
        source_copies_created=False,recording='navigation_results_and_sampled_chassis_feedback'))


def argument_parser():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=['visual','nso','joint','reference','navigation'],default='nso')
    ap.add_argument('--duration',type=float,default=600.)
    ap.add_argument('--duration-profile',choices=PROFILES,default=STANDARD)
    ap.add_argument('--execute',action='store_true')
    ap.add_argument('--report')
    ap.add_argument('--experiment-config',help='experiment identity metadata; cannot enable an unavailable NSO task')
    ap.add_argument('--task',help='Frozen real V35 task-bundle JSON for NSO execution')
    ap.add_argument('--policy-mode',choices=['G','S'],default='G')
    ap.add_argument('--resume-from',help='Previous V35 reference attempt with unchanged task')
    ap.add_argument('--joint-configuration',help='Frozen real sensor/frame settings for unknown-coverage NSO')
    ap.add_argument('--joint-xy-goal-tolerance',type=float,choices=(.03,.08),default=.08,
        help='ROS execution only: retain .08m default or test .03m grid-center stopping; original NSO cell/heading acceptance unchanged')
    ap.add_argument('--ros-allow-reverse',action='store_true',help='Explicit ordinary ROS reverse in nso/joint/navigation; linear command bounded to +/-.05m/s')
    ap.add_argument('--navigation-allow-reverse',action='store_true',help='Navigation setup only: temporarily allow DWA and adapter reverse down to -0.05m/s; restore DWA afterwards')
    ap.add_argument('--sensor-loss-profile',choices=['legacy_terminal_v1','recoverable_sensor_pause_v1'],default='legacy_terminal_v1',help='Explicit stale sensor pause profile; zero while stale, recover only with fresh sources/proposal')
    ap.add_argument('--feedback-speed-profile',choices=['original_015_v1','measured_020_v1'],default='original_015_v1',help='Explicit stop-behavior change: measured_020_v1 raises linear CAN feedback stop threshold to .20 m/s; requires user approval')
    ap.add_argument('--nso-correction-reverse',action='store_true',help='Explicit nso/joint nominal endpoint retry only: reverse down to -.05m/s and center-point correction; ordinary execution remains forward-only')
    ap.add_argument('--navigation-goal-json',help='Standard ROS setup reposition only; JSON target_odom, never an NSO episode')
    return ap


def result_summary(result, output):
    """Small user-facing record; detailed decisions and arrivals stay in their records."""
    return dict(report=str(output),status=result.get('status'),reason=result.get('reason'),
        goals=len(result.get('goals',[])),verified_arrivals=len(result.get('successful_arrivals',[])),
        distance_m=result.get('actual_CAN_distance_m'),elapsed_s=result.get('elapsed_run_s'),
        stopped=result.get('feedback_zero_after_close',False),
        execution_interrupted=result.get('execution_interrupted',False))


def execution_interruption(result, reason):
    if result.get('status')!='execution_interrupted':
        result['decision_status_before_interruption']=result.get('status')
    result.update(status='execution_interrupted',execution_interrupted=True,reason=str(reason))


def main(argv=None):
    args=argument_parser().parse_args(argv)
    duration_profile = validate_duration(args.duration, args.duration_profile, args.mode)
    if args.ros_allow_reverse and args.mode not in ('nso','joint','navigation'):
        raise ValueError('ros-allow-reverse is restricted to nso/joint/navigation')
    if args.navigation_allow_reverse and args.mode!='navigation':
        raise ValueError('navigation-allow-reverse is restricted to navigation setup mode')
    if args.nso_correction_reverse and args.mode not in ('nso','joint'):
        raise ValueError('nso-correction-reverse is restricted to nso/joint mode')
    if args.joint_xy_goal_tolerance != .08 and args.mode not in ('nso','joint'):
        raise ValueError('joint-xy-goal-tolerance is restricted to nso/joint mode')
    navigation_goal=None
    if args.mode=='navigation':
        if not args.navigation_goal_json:
            raise ValueError('standard navigation setup requires an explicit goal JSON')
        from standard_navigation_goal import validate_goal_odom
        navigation_goal=validate_goal_odom(json.loads(Path(args.navigation_goal_json).read_text())['target_odom'])
    elif args.navigation_goal_json:
        raise ValueError('navigation setup goal is separate from NSO method execution')
    reference_worker=reference_task=reference_output=None
    if args.resume_from and args.mode!='reference':raise ValueError('resume-from is only for the reference NSO task')
    if args.mode=='reference' and not args.task:
        from nso_ros_coordinator import ensure_nso_runtime_ready
        try:
            ensure_nso_runtime_ready()
        except RuntimeError as exc:
            if args.execute:
                raise SystemExit(str(exc))
            print(json.dumps(dict(status='nso_task_input_missing',reason=str(exc),no_motion=True)))
            return
    if args.mode=='reference' and args.task:
        import atexit,tempfile
        from reference_transport import ReferenceWorker
        from nso_ros_coordinator import ensure_nso_runtime_ready
        if args.execute:
            reference_output=Path(args.report) if args.report else ROOT/'reports/nso-hybrid-20261029'/('reference_'+str(time.time_ns()))
            reference_output.mkdir(parents=True,exist_ok=False)
            reference_worker=ReferenceWorker(WORK,reference_output/'nso')
            atexit.register(reference_worker.close)
            try:
                settings=dict(task_path=str(Path(args.task).resolve()),mode=args.policy_mode)
                if args.resume_from:settings['resume_log']=str(Path(args.resume_from).resolve()/'nso/decisions.jsonl')
                reference_task=reference_worker.call('init',settings,lambda:None)
                ensure_nso_runtime_ready(reference_task)
            except BaseException:
                reference_worker.close()
                raise
        else:
            with tempfile.TemporaryDirectory(prefix='nso-reference-preview-') as temporary:
                worker=ReferenceWorker(WORK,Path(temporary)/'nso')
                try:
                    settings=dict(task_path=str(Path(args.task).resolve()),mode=args.policy_mode)
                    if args.resume_from:settings['resume_log']=str(Path(args.resume_from).resolve()/'nso/decisions.jsonl')
                    task=worker.call('init',settings,lambda:None)
                    print(json.dumps(dict(status='reference_task_loaded_no_motion',task=task['task'],summary=task['summary'],no_motion=True)))
                finally:worker.close()
            return
    if args.mode in ('nso','joint'):
        if not args.joint_configuration:
            if args.execute:raise SystemExit('Prepare a real joint NSO sensor/frame configuration before execution.')
            print(json.dumps(dict(status='joint_configuration_required',method='unknown_coverage_and_observed_surface',no_motion=True)))
            return
        import atexit,tempfile
        from reference_transport import ReferenceWorker
        from joint_ros_coordinator import validate_joint_identity
        settings=json.loads(Path(args.joint_configuration).read_text())
        # Natural semantics is the declared default for the deployed NSO path.
        # An explicit null in a comparison configuration disables it; a missing
        # model is an error, never an unreported geometry-only fallback.
        if 'semantic_configuration' not in settings:
            settings['semantic_configuration']=json.loads((WORK/'config/real_semantic_frontend.json').read_text())
        if args.execute:
            reference_output=Path(args.report) if args.report else ROOT/'reports/nso-hybrid-20261029'/('joint_'+str(time.time_ns()))
            reference_output.mkdir(parents=True,exist_ok=False)
            save(reference_output/'joint_configuration.json',settings)
            reference_worker=ReferenceWorker(WORK,reference_output/'joint',module='nso_bridge.joint_engine')
            atexit.register(reference_worker.close)
            try:
                reference_task=reference_worker.call('init',settings,lambda:None)
                validate_joint_identity(reference_task)
            except BaseException:
                reference_worker.close();raise
        else:
            with tempfile.TemporaryDirectory(prefix='nso-joint-preview-') as temporary:
                worker=ReferenceWorker(WORK,Path(temporary)/'joint',module='nso_bridge.joint_engine')
                try:
                    task=worker.call('init',settings,lambda:None)
                    print(json.dumps(dict(status='joint_configuration_loaded_no_motion',receipt=task['receipt'],summary=task['summary'],no_motion=True)))
                finally:worker.close()
            return
    experiment_config,experiment_raw,experiment_sha=load_experiment_config(args.experiment_config)
    adapter_path=WORK/'runtime/visual_velocity_adapter.py'
    if not args.execute:
        print(json.dumps(dict(status='validated_no_motion',mode=args.mode,
            duration_s=args.duration,duration_profile=duration_profile,master=MASTER,native_exists=NATIVE.is_file(),
            adapter_exists=adapter_path.is_file(),navigation_source='lidar_rgbd',
            experiment_config_sha256=experiment_sha,recording='navigation_only')))
        return
    if os.environ.get('ROS_MASTER_URI')!=MASTER:
        raise ValueError('source ROS setup and use isolated hybrid master 11382')
    found=other_senders()
    if found:
        raise ValueError('another actuator sender/adapter exists: '+repr(found))
    explore_command=exploration_arguments() if args.mode=='visual' else None
    try:
        subprocess.run(['/usr/bin/python3','-s','-B',str(WORK/'runtime/start_hybrid_stack.py'),'--start'],check=True)
    except BaseException:
        if reference_worker:reference_worker.close()
        raise
    import rospy, actionlib, tf2_ros
    from nav_msgs.msg import Odometry,OccupancyGrid,Path as NavigationPath
    from move_base_msgs.msg import MoveBaseAction,MoveBaseActionGoal,MoveBaseActionResult
    from std_msgs.msg import Bool,Int32,String
    from std_srvs.srv import Trigger
    from rosgraph_msgs.msg import Log
    rospy.init_node('nso_onboard_supervisor',disable_signals=True)
    from joint_observation import navigation_map_source
    joint_map_source=navigation_map_source(reference_task['calibration']
        if args.mode in ('nso','joint') else None)
    localization_source=rospy.get_param('/nso_visual/localization_source','lidar_rf2o')
    interrupted=threading.Event()
    signal.signal(signal.SIGINT,lambda *_:interrupted.set())
    signal.signal(signal.SIGTERM,lambda *_:interrupted.set())
    session='hybrid_'+args.mode+'_'+str(time.time_ns())
    output=reference_output or (Path(args.report) if args.report else ROOT/'reports/nso-hybrid-20261029'/session)
    if reference_output is None:output.mkdir(parents=True,exist_ok=False)
    lifecycle=LifecycleReceipt(output/'cleanup_milestones')
    lifecycle.record('supervisor_started')
    signal.signal(signal.SIGINT,lifecycle.signal_handler(interrupted))
    signal.signal(signal.SIGTERM,lifecycle.signal_handler(interrupted))
    if reference_task:save(output/'reference-task-receipt.json',reference_task)
    if experiment_raw is not None:
        (output/'experiment_config.json').write_bytes(experiment_raw)
    save(output/'experiment_receipt.json',dict(schema='nso_physical_attempt/1',
        session_id=session,experiment_config_sha256=experiment_sha,
        source_config_path=None if args.experiment_config is None else str(Path(args.experiment_config).resolve()),
        navigation_source='lidar_rgbd',mode=args.mode,
        duration_s=args.duration,duration_profile=duration_profile,created_unix_s=time.time(),status='attempt_registered_before_prearm'))
    write_source_manifest(output)
    save(output/'runtime_parameters.json',dict(captured_unix_s=time.time(),
        navigation_source='lidar_rgbd',experiment_config_sha256=experiment_sha,
        localization_source=localization_source,
        control_velocity_source=(rospy.get_param('/nso_csm_odometry_bridge/control_velocity_source','csm_se2')
                                 if localization_source=='lidar_csm' else 'RF2O_original_twist'),
        control_velocity_parameters=(rospy.get_param('/nso_csm_odometry_bridge',{})
                                     if localization_source=='lidar_csm' else {}),
        localization_parameters=rospy.get_param('/nso_csm',{}) if localization_source=='lidar_csm' else rospy.get_param('/rf2o_laser_odometry',{}),
        shadow_matcher_parameters=rospy.get_param('/nso_csm_zero_keyframe',{}) if localization_source=='lidar_csm' else {},
        camera_depth=rospy.get_param('/zed2i/zed_node/depth',{}),
        camera_depth_confidence=rospy.get_param('/zed2i/zed_node/depth_confidence',None),
        camera_texture_confidence=rospy.get_param('/zed2i/zed_node/depth_texture_conf',None),
        camera_tracking=rospy.get_param('/zed2i/zed_node/pos_tracking',{}),
        lidar_mapper=rospy.get_param(joint_map_source['publisher'],{}),
        navigation_map_source=joint_map_source,
        DWA=rospy.get_param('/nso_visual/move_base/DWAPlannerROS',{}),
        Navfn=rospy.get_param('/nso_visual/move_base/NavfnROS',{}),
        global_obstacles=rospy.get_param('/nso_visual/move_base/global_costmap/obstacle_layer',{}),
        local_obstacles=rospy.get_param('/nso_visual/move_base/local_costmap/obstacle_layer',{})))
    maximum_feedback_linear = .20 if args.feedback_speed_profile == 'measured_020_v1' else .15
    execution_max_linear = .05 if args.ros_allow_reverse or args.mode in ('nso', 'joint') else .1
    execution_profile = ('ordinary_ros_reverse_005_v1' if args.ros_allow_reverse else
                         'nso_dwa_linear_consistency_005_v1' if args.mode in ('nso', 'joint')
                         else 'legacy_linear_010_v1')
    adapter_cfg=dict(ordinary_reverse=args.ros_allow_reverse,sensor_loss_profile=args.sensor_loss_profile,maximum_feedback_linear_m_s=maximum_feedback_linear,feedback_speed_profile=args.feedback_speed_profile,maximum_linear_m_s=execution_max_linear,execution_clipping_profile=execution_profile,session_id=session,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        duration_s=args.duration,duration_profile=args.duration_profile,mission_mode=args.mode,output_directory=str(output/'adapter'),master_uri=MASTER,
        navigation_source='lidar_rgbd')
    if args.nso_correction_reverse:
        adapter_cfg.update(nso_correction_reverse=True,mission_mode=args.mode,
                           correction_supervisor=rospy.get_name())
    if args.navigation_allow_reverse:
        adapter_cfg.update(navigation_allow_reverse=True,mission_mode='navigation')
    save(output/'adapter-config.json',adapter_cfg)
    adapter_hash=hashlib.sha256((output/'adapter-config.json').read_bytes()).hexdigest()
    save(output/'authorization.json',dict(source='direct user instruction to continue the recommended experiment; onsite readiness confirmed',
        maximum_linear_m_s=execution_max_linear,execution_clipping_profile=execution_profile,
        sensor_loss_profile=args.sensor_loss_profile,
        maximum_feedback_linear_m_s=maximum_feedback_linear,maximum_feedback_angular_rad_s=.45,
        feedback_speed_profile=args.feedback_speed_profile,
        feedback_stop_behavior_changed=args.feedback_speed_profile != "original_015_v1",
        execution_clipping_change="positive adapter command cap aligned to DWA .05; feedback stop threshold recorded separately by feedback_speed_profile",
        maximum_angular_rad_s=.3,duration_s=args.duration,duration_profile=duration_profile,
        ordinary_reverse=args.ros_allow_reverse,
        ordinary_reverse_profile="ordinary_ros_reverse_005_v1" if args.ros_allow_reverse else None,
        automatic_reverse=args.navigation_allow_reverse or args.ros_allow_reverse,
        minimum_linear_m_s=-.05 if args.navigation_allow_reverse or args.ros_allow_reverse else 0.,
        reverse_scope=('ordinary_nso_joint_navigation' if args.ros_allow_reverse else
                       'navigation_setup_only' if args.navigation_allow_reverse else None),
        prior_short_range_caps=False,physical_stop_independently_observed=False,
        nso_correction_reverse=args.nso_correction_reverse,
        correction_reverse_profile='nso_nominal_correction_reverse_v1' if args.nso_correction_reverse else None,
        correction_reverse_scope=('existing same-pending nominal retry; ordinary minimum follows ordinary_reverse authorization; retry minvx-.05/fp0; restore after; not original-method equivalence') if args.nso_correction_reverse else None))
    mod=module(adapter_path,'visual_adapter')
    journal=mod.Journal(output/'events.jsonl')
    lock=threading.RLock()
    hardware=ActuatorTelemetry(session)
    state=dict(odom=None,map=None,costmap=None,local_costmap=None,health=None,driver=None,
               goals=[],results=[],explorer_stopped=False,map_count=0,
               goal_starts={},terminal_goals=set(),arrival_records=[])
    children=[]; streams=[]; run_start=None
    result=dict(session_id=session,mode=args.mode,status='preparing',navigation_source='lidar_rgbd',
        experiment_config_sha256=experiment_sha,duration_profile=duration_profile,
        successful_arrivals=[],physical_stop_independently_observed=False,cleanup=[])
    stop=rospy.Publisher('/nso_radius/stop',Bool,queue_size=1,latch=True)
    tfbuf=tf2_ros.Buffer(cache_time=rospy.Duration(30.))
    tflistener=tf2_ros.TransformListener(tfbuf)
    client=actionlib.SimpleActionClient('/nso_visual/move_base',MoveBaseAction)
    def capture_pose():
        transform=tfbuf.lookup_transform('map','base_link',rospy.Time(0),rospy.Duration(.05))
        return pose_evidence(transform,rospy.Time.now().to_sec(),time.monotonic())
    def capture_odom_pose():
        transform=tfbuf.lookup_transform('odom','base_link',rospy.Time(0),rospy.Duration(.05))
        return pose_evidence(transform,rospy.Time.now().to_sec(),time.monotonic(),target_frame='odom')
    navigation_execution_frame=('odom' if args.mode=='navigation' else
        reference_task['calibration'].get('ROS_execution_frame','map')
        if args.mode in ('nso','joint') else 'map')
    def capture_navigation_pose(frame):
        if frame == 'odom':return capture_odom_pose()
        if frame == 'map':return capture_pose()
        raise ValueError('unsupported_arrival_recording_frame:'+str(frame))
    def can_snapshot():
        with lock:
            return dict(distance_m=hardware.can_distance,absolute_yaw_rad=hardware.can_absolute_yaw,
                        accounting_complete=hardware.can_accounting_complete,
                        feedback_fresh=hardware.hardware_reason(time.monotonic()) is None,
                        receipt_monotonic=time.monotonic())
    def odom(m):
        with lock:state['odom']=(time.monotonic(),m)
    def grid(m):
        with lock:state.update(map=(time.monotonic(),m),map_count=state['map_count']+1)
    def costmap(m):
        with lock:state['costmap']=(time.monotonic(),m)
    def local_costmap(m):
        with lock:state['local_costmap']=(time.monotonic(),m)
    def health(m):
        try:
            d=json.loads(m.data)
            with lock:state['health']=(time.monotonic(),d)
        except ValueError:pass
    def driver(m):
        with lock:state['driver']=(time.monotonic(),m.data)
    def actuator_state(m):
        try:
            row=json.loads(m.data)
            with lock:
                hardware.update(row,getattr(m,'_connection_header',{}).get('callerid'),time.monotonic())
        except (ValueError,KeyError,TypeError) as exc:
            with lock:hardware.fault='actuator_telemetry_invalid:'+str(exc)
    from plan_endpoint_recording import PlanEndpointRecorder
    plan_endpoints=PlanEndpointRecorder()
    def plan_endpoint(m,topic):
        receipt_monotonic,receipt_unix_s=time.monotonic(),time.time()
        with lock:
            latest=state['goals'][-1] if state['goals'] else None
            row=plan_endpoints.record(topic,m,latest,receipt_monotonic,receipt_unix_s)
        if row is not None:journal.add(row)
    def goal(m):
        p=m.goal.target_pose.pose.position;q=m.goal.target_pose.pose.orientation
        r=dict(event='goal',id=m.goal_id.id,x=p.x,y=p.y,
               yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z)),
               frame=m.goal.target_pose.header.frame_id,stamp=m.header.stamp.to_sec(),
               receipt_monotonic=time.monotonic())
        recording_frame=arrival_recording_frame(args.mode,navigation_execution_frame,r['frame'])
        try:before=capture_navigation_pose(recording_frame);error=None
        except Exception as exc:before=None;error=str(exc)
        initial=dict(goal=r,before=before,can_start=can_snapshot(),pose_error=error,
                     recording_frame=recording_frame)
        with lock:
            state['goals'].append(r)
            if r['id'] not in state['goal_starts']:state['goal_starts'][r['id']]=initial
        journal.add(r)
    def goal_result(m):
        r=dict(event='goal_result',id=m.status.goal_id.id,status=m.status.status,text=m.status.text,
               receipt_monotonic=time.monotonic())
        with lock:
            initial=state['goal_starts'].get(r['id'],{})
        recording_frame=initial.get('recording_frame',navigation_execution_frame)
        try:after=capture_navigation_pose(recording_frame);error=None
        except Exception as exc:after=None;error=str(exc)
        final_can=can_snapshot()
        with lock:
            state['results'].append(r)
            if r['id'] in state['terminal_goals']:return
            state['terminal_goals'].add(r['id'])
            evidence=arrival_evidence(initial.get('goal'),r,initial.get('before'),after,
                                      initial.get('can_start'),final_can,expected_frame=recording_frame)
            evidence.update(before_pose_error=initial.get('pose_error'),after_pose_error=error,
                            recording_frame=recording_frame)
            state['arrival_records'].append(evidence)
            if evidence['verified']:result['successful_arrivals'].append(evidence)
        journal.add(r)
        journal.add(dict(event='arrival_evidence',**evidence))
    def log(m):
        if m.name=='/nso_visual/explore' and m.msg=='Exploration stopped.':
            with lock:state['explorer_stopped']=True
    subscribers=[rospy.Subscriber('/nso_visual/odom',Odometry,odom,queue_size=20),
        rospy.Subscriber(joint_map_source['topic'],OccupancyGrid,grid,queue_size=2),
        rospy.Subscriber('/nso_visual/move_base/global_costmap/costmap',OccupancyGrid,costmap,queue_size=1),
        rospy.Subscriber('/nso_visual/move_base/local_costmap/costmap',OccupancyGrid,local_costmap,queue_size=1),
        rospy.Subscriber('/nso_visual/health',String,health,queue_size=2),
        rospy.Subscriber('/nso_radius/driver_fault',Int32,driver,queue_size=2),
        rospy.Subscriber('/nso_visual/actuator_state',String,actuator_state,queue_size=2),
        rospy.Subscriber('/nso_visual/move_base/goal',MoveBaseActionGoal,goal,queue_size=10),
        rospy.Subscriber('/nso_visual/move_base/NavfnROS/plan',NavigationPath,
            lambda m:plan_endpoint(m,'/nso_visual/move_base/NavfnROS/plan'),queue_size=1),
        rospy.Subscriber('/nso_visual/move_base/DWAPlannerROS/global_plan',NavigationPath,
            lambda m:plan_endpoint(m,'/nso_visual/move_base/DWAPlannerROS/global_plan'),queue_size=1),
        rospy.Subscriber('/nso_visual/move_base/result',MoveBaseActionResult,goal_result,queue_size=10),
        rospy.Subscriber('/rosout',Log,log,queue_size=20)]
    def spawn(argv,name):
        stream=(output/(name+'.log')).open('xb');streams.append(stream)
        env=dict(os.environ,LD_LIBRARY_PATH=str(SDK)+':'+os.environ.get('LD_LIBRARY_PATH',''))
        p=subprocess.Popen([str(a) for a in argv],env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
        children.append((name,p))
        save(output/(name+'-start.json'),dict(pid=p.pid,argv=[str(a) for a in argv]))
        return p
    prepare_window,active_window=HealthWindow(),HealthWindow()
    last_health_reason=None
    def check(active=False):
        nonlocal last_health_reason
        deadline=time.monotonic()+30.
        while True:
            now=time.monotonic()
            if active and run_start is not None and now-run_start>=args.duration:
                raise RuntimeError('execution_duration_complete')
            if interrupted.is_set() or rospy.is_shutdown() or (output/'STOP').exists():
                raise RuntimeError('operator_stop')
            with lock:
                # A callback may have updated its receipt while we waited for
                # this lock. Read the comparison clock after taking the lock.
                now=time.monotonic()
                fault=hardware.fault;chassis_reason=hardware.hardware_reason(now)
                h,o,g,d=state['health'],state['odom'],state['map'],state['driver']
                explicitly_stopped=state['explorer_stopped']
                zero=hardware.can_zero_since is not None and now-hardware.can_zero_since>=1.
            if fault:raise RuntimeError('chassis:'+str(fault))
            if active and chassis_reason:raise RuntimeError('chassis:'+chassis_reason)
            if d and d[1]!=0:raise RuntimeError('native_driver_fault:'+str(d[1]))
            if active and (not d or not 0<=now-d[0]<=.25):raise RuntimeError('native_driver_status_stale')
            for name,p in children:
                if unexpected_child_exit(name,p.poll(),explicitly_stopped):
                    raise RuntimeError(name+'_unexpected_exit:'+str(p.returncode))
            if journal.error:raise RuntimeError(journal.error)
            reason=hybrid_health_reason(h,o,g,now,rospy.Time.now().to_sec())
            if not active:reason=reason or chassis_reason or (None if zero else 'need_one_second_actual_CAN_zero')
            if reason!=last_health_reason:
                journal.add(dict(event='supervisor_health_transition',reason=reason,active=active,
                                 at_monotonic=now,health=None if h is None else h[1],
                                 motion_zero_policy='existing adapter original-source TTL; no stop latch during grace'))
                last_health_reason=reason
            disposition=(active_window if active else prepare_window).update(reason,now,preparing=not active)
            if disposition=='ready':return
            if disposition=='expired' and not (args.sensor_loss_profile=='recoverable_sensor_pause_v1'
                    and recoverable_sensor_health_reason(reason,h,o,now,rospy.Time.now().to_sec())):
                raise RuntimeError('hybrid_health_loss_at_least_one_second:'+str(reason))
            if not active and now>=deadline:raise RuntimeError('prearm_health_not_stable:'+str(reason))
            # Adapter alone admits commands and zeroes stale sources. Wait here
            # without a stop latch, so no new decision runs during recovery.
            time.sleep(.02)
    last_progress=None
    navigation_restore=None
    def progress():
        nonlocal last_progress
        with lock:
            row=dict(phase=result['status'],goals=len(state['goals']),
                results=len(state['results']),verified_arrivals=len(result['successful_arrivals']))
        signature=tuple(row.values())
        if signature!=last_progress:
            print(json.dumps(row),flush=True)
            last_progress=signature
    try:
        if args.mode in ('nso','joint'):
            expected_localization=reference_task['calibration'].get('localization_source','lidar_rf2o')
            if expected_localization != localization_source:
                raise ValueError('joint calibration belongs to a different localization source; create a fresh frame/anchor before execution')
            from dynamic_reconfigure.client import Client as ConfigurationClient
            navigation_configuration=ConfigurationClient('/nso_visual/move_base/DWAPlannerROS',timeout=4.)
            old_navigation=navigation_configuration.get_configuration(timeout=3.)
            navigation_restore=dict(xy_goal_tolerance=old_navigation['xy_goal_tolerance'])
            if args.nso_correction_reverse:
                navigation_restore.update(min_vel_x=old_navigation['min_vel_x'],
                                          forward_point_distance=old_navigation['forward_point_distance'])
            applied_navigation=navigation_configuration.update_configuration(
                dict(xy_goal_tolerance=args.joint_xy_goal_tolerance))
            save(output/'joint_navigation_contract.json',dict(original_xy_goal_tolerance=navigation_restore['xy_goal_tolerance'],
                actual_Navfn_default_tolerance=rospy.get_param('/nso_visual/move_base/NavfnROS/default_tolerance',None),
                execution_xy_goal_tolerance=args.joint_xy_goal_tolerance,
                execution_controller_profile='grid_center_3cm_v1' if args.joint_xy_goal_tolerance == .03 else 'default_center_8cm_v1',
                reason='ROS stopping accuracy within original 0.2m cell; actual exposure cell and heading independently validated without changing worker acceptance',
                applied_xy_goal_tolerance=applied_navigation.get('xy_goal_tolerance'),
                applied_yaw_goal_tolerance=applied_navigation.get('yaw_goal_tolerance'),
                execution_frame=navigation_execution_frame,
                root_latch_xy_goal_tolerance=rospy.get_param('/nso_visual/move_base/latch_xy_goal_tolerance',False),
                execution_yaw_goal_tolerance=rospy.get_param('/nso_visual/move_base/DWAPlannerROS/yaw_goal_tolerance'),
                actual_global_costmap_frame=rospy.get_param('/nso_visual/move_base/global_costmap/global_frame'),
                actual_global_costmap_rolling_window=rospy.get_param('/nso_visual/move_base/global_costmap/rolling_window'),
                changes_to_policy=False,shared_navigation_profile_required_for_comparisons=True))
        if args.navigation_allow_reverse or args.ros_allow_reverse:
            from dynamic_reconfigure.client import Client as ConfigurationClient
            navigation_configuration=ConfigurationClient('/nso_visual/move_base/DWAPlannerROS',timeout=4.)
            old_navigation=navigation_configuration.get_configuration(timeout=3.)
            if navigation_restore is None:navigation_restore={}
            navigation_restore['min_vel_x']=old_navigation['min_vel_x']
            reverse_request=dict(min_vel_x=-.05)
            if args.ros_allow_reverse:
                for key in ('max_vel_x','max_vel_trans'):
                    navigation_restore.setdefault(key,old_navigation[key])
                    reverse_request[key]=.05
            applied_reverse=navigation_configuration.update_configuration(reverse_request)
            result['navigation_reverse_contract']=dict(profile='ordinary_ros_reverse_005_v1' if args.ros_allow_reverse else 'navigation_setup_reverse_005_v1',
                setup_only=not args.ros_allow_reverse,ordinary_reverse=args.ros_allow_reverse,NSO_action=False,changes_to_NSO_policy=False,
                original_min_vel_x=old_navigation['min_vel_x'],requested_min_vel_x=-.05,
                applied_response=applied_reverse,restoration_required=True)
            save(output/'navigation_reverse_contract.json',result['navigation_reverse_contract'])
            if args.ros_allow_reverse and any(abs(float(applied_reverse[k])-.05)>1e-9 for k in ('max_vel_x','max_vel_trans')):
                raise RuntimeError('DWA did not admit declared ordinary positive .05 limit')
            if abs(float(applied_reverse['min_vel_x'])+.05)>1e-9:
                raise RuntimeError('DWA did not admit the declared setup reverse limit')
        if not client.wait_for_server(rospy.Duration(6.)):raise RuntimeError('move_base_unavailable')
        spawn(['/usr/bin/python3','-s','-B',adapter_path,'--config',output/'adapter-config.json',
               '--config-sha256',adapter_hash,'--run'],'adapter')
        rospy.wait_for_service('/nso_visual/arm',timeout=6.)
        def launch_native():
            spawn([NATIVE,'_execute:=true','_session_id:='+session,'_expected_boot_id:='+adapter_cfg['boot_id'],
                   '_can_interface:=can0','_guard_node:=/nso_radius_guard',
                   '_timing_output:='+str(output/'native_timing.json'),'__name:=nso_scout_radius_driver_v1'],'native')
        native_start=NativeStartOnce(launch_native)
        end=time.monotonic()+8.
        while time.monotonic()<end:
            if interrupted.is_set() or rospy.is_shutdown() or (output/'STOP').exists():
                raise RuntimeError('operator_stop')
            with lock:
                now=time.monotonic()
                standby_snapshot=dict(receipt=hardware.receipt,fault=hardware.fault,
                    ever_armed=hardware.ever_armed,can_zero_since=hardware.can_zero_since,
                    can_status=None if hardware.can_status is None else dict(hardware.can_status),
                    can_velocity=None if hardware.can_velocity is None else dict(hardware.can_velocity))
                # Native only enables SDK CAN mode and emits zero before arm.
                # Eligibility does not relax either original check() or arm().
                if native_start.consider_standby(standby_snapshot,now,execute=args.execute):
                    result['native_standby_handshake']=dict(reason=native_start.context,
                        at_monotonic=now,snapshot=standby_snapshot,motion_authorized=False)
                    journal.add(dict(event='native_standby_handshake',**result['native_standby_handshake']))
                ready=(hardware.hardware_reason(now) is None and hardware.can_zero_since is not None
                    and now-hardware.can_zero_since>=1. and state['odom'] and state['map'] and state['health'])
            if ready:break
            time.sleep(.05)
        check()
        with lock:local_grid=state['local_costmap']
        if not local_grid or not 0<=time.monotonic()-local_grid[0]<=2.:
            raise RuntimeError('fresh_local_costmap_required_before_arm')
        # Standard move_base/DWA owns collision checking. Retain fresh
        # navigation data and TF without duplicating its geometry in Python.
        capture_pose()
        client.cancel_all_goals()
        rospy.wait_for_service('/nso_visual/arm',timeout=6.)
        native_start.start('original_postcheck_native_start')
        end=time.monotonic()+2.
        while time.monotonic()<end:
            with lock:driver_ready=(state['driver'] and 0<=time.monotonic()-state['driver'][0]<=.25
                                   and hardware.driver_connected)
            if driver_ready:break
            time.sleep(.02)
        check(True)
        run_start=time.monotonic()
        armed=rospy.ServiceProxy('/nso_visual/arm',Trigger)()
        if not armed.success:raise RuntimeError('arm_rejected:'+armed.message)
        with lock:hardware.ever_armed=True
        result['status']='running'
        save(output/'running.json',dict(session_id=session,started_unix_s=time.time(),duration_s=args.duration))
        if args.mode=='reference':
            from nso_ros_coordinator import run_nso
            from reference_observation import ReferenceObservationCollector
            collector=ReferenceObservationCollector(tfbuf)
            def heartbeat():
                check(True)
                if time.monotonic()-run_start>=args.duration:raise RuntimeError('execution_duration_complete')
            def stopped_check():
                with lock:
                    now=time.monotonic()
                    return dict(stopped=(hardware.hardware_reason(now) is None and hardware.can_zero_since is not None
                        and now-hardware.can_zero_since>=1.),feedback=can_snapshot(),zero_since=hardware.can_zero_since)
            def capture_observation(calibration,step,pose,action):
                return collector.capture(output/'paid',calibration,heartbeat,stopped_check,scope='paid_stopped_observation')
            try:
                run_nso(dict(worker=reference_worker,task_receipt=reference_task,task_path=args.task,
                    resume_attempt=args.resume_from,
                    mode=args.policy_mode,output=output,result=result,journal=journal,client=client,
                    tfbuf=tfbuf,capture_pose=capture_pose,capture_odom_pose=capture_odom_pose,
                    can_snapshot=can_snapshot,check=check,progress=progress,
                    capture_observation=capture_observation,run_start=run_start,duration=args.duration,
                    xy_tolerance_m=rospy.get_param('/nso_visual/move_base/DWAPlannerROS/xy_goal_tolerance'),
                    yaw_tolerance_rad=rospy.get_param('/nso_visual/move_base/DWAPlannerROS/yaw_goal_tolerance')))
            finally:collector.close()
        elif args.mode in ('nso','joint'):
            from joint_ros_coordinator import run_joint
            from joint_observation import JointObservationCollector
            from reference_observation import ReferenceObservationCollector
            collector=ReferenceObservationCollector(tfbuf)
            joint_collector=JointObservationCollector(tfbuf,reference_task['grid'],
                **joint_map_source,
                apply_approved_self_mask=reference_task['calibration'].get('apply_approved_self_mask',True))
            def heartbeat():
                check(True)
                if time.monotonic()-run_start>=args.duration:raise RuntimeError('execution_duration_complete')
            def motion_snapshot():
                with lock:
                    return dict(feedback=can_snapshot(),can_status=hardware.can_status,
                        can_velocity=hardware.can_velocity,zero_since=hardware.can_zero_since)
            joint_capture_timings={}
            def capture_joint_observation(calibration,action_id,pose,action):
                joint_capture_timings.clear()
                path=collector.capture_live_joint(output/'observed',heartbeat,motion_snapshot,timings=joint_capture_timings)
                return joint_collector.capture(output/'joint_inputs',path,heartbeat,timings=joint_capture_timings)
            correction_scope=None
            if args.nso_correction_reverse:
                from correction_reverse_scope import correction_reverse_scope
                from std_srvs.srv import SetBool
                phase_service='/nso_visual/correction_reverse/'+session
                phase_client=rospy.ServiceProxy(phase_service,SetBool)
                def phase(enabled):
                    rospy.wait_for_service(phase_service,timeout=3.)
                    response=phase_client(enabled)
                    if not response.success:
                        raise RuntimeError('correction_reverse_phase_rejected:'+response.message)
                def correction_scope(endpoint,record):
                    def record_phase(event,**fields):
                        result.setdefault('correction_reverse_phases',[]).append(dict(event=event,**fields))
                        record(event,**fields)
                    return correction_reverse_scope(navigation_configuration,phase,record_phase,endpoint)
                result['correction_reverse_profile']=dict(name='nso_nominal_correction_reverse_v1',
                    authorized=True,session_id=session,ordinary_min_vel_x=-.05 if args.ros_allow_reverse else 0.,retry_min_vel_x=-.05,
                    retry_forward_point_distance=0.,original_method_equivalence=False)
            try:
                run_joint(dict(worker=reference_worker,task_receipt=reference_task,output=output,result=result,journal=journal,
                    client=client,tfbuf=tfbuf,capture_pose=capture_pose,capture_odom_pose=capture_odom_pose,
                    execution_frame=navigation_execution_frame,ordinary_reverse_authorized=args.ros_allow_reverse,
                    can_snapshot=can_snapshot,check=check,progress=progress,capture_observation=capture_joint_observation,
                    capture_timings=joint_capture_timings,
                    run_start=run_start,duration=args.duration,
                    correction_reverse_scope=correction_scope,
                    xy_tolerance_m=rospy.get_param('/nso_visual/move_base/DWAPlannerROS/xy_goal_tolerance'),
                    yaw_tolerance_rad=rospy.get_param('/nso_visual/move_base/DWAPlannerROS/yaw_goal_tolerance')))
            finally:
                collector.close();joint_collector.close()
        elif args.mode=='navigation':
            from standard_navigation_goal import run_standard_navigation_goal
            def navigation_heartbeat():
                check(True)
                progress()
                if time.monotonic()-run_start>=args.duration:
                    raise RuntimeError('setup_navigation_duration_complete')
            run_standard_navigation_goal(dict(goal_odom=navigation_goal,client=client,
                capture_odom_pose=capture_odom_pose,can_snapshot=can_snapshot,
                heartbeat=navigation_heartbeat,result=result,journal=journal))
        else:
            spawn(explore_command,'explore')
            while time.monotonic()-run_start<args.duration:
                check(True);progress()
                with lock:stopped=state['explorer_stopped']
                if stopped:
                    result['status']='explorer_stopped';break
                time.sleep(.05)
            else:result['status']='duration_complete'
    except BaseException as exc:
        execution_interruption(result,type(exc).__name__+': '+str(exc))
    finally:
        with lock:
            if hardware.fault:
                execution_interruption(result,'chassis_before_stop:'+str(hardware.fault))
        lifecycle.record('stop_request_begin')
        stop.publish(Bool(data=True))
        lifecycle.record('stop_requested')
        lifecycle.record('worker_close_begin',present=reference_worker is not None)
        if reference_worker:reference_worker.close()
        lifecycle.record('worker_close_end')
        # Only termination verification gets its own receive-only socket, when
        # this process no longer performs image conversion, inference or NPZ I/O.
        stop_gate=mod.Gate(args.duration, duration_profile=args.duration_profile, mission_mode=args.mode)
        stop_feedback=None
        try:stop_feedback=mod.FeedbackReader(stop_gate,lock,journal)
        except Exception as exc:result['stop_feedback_error']=str(exc)
        try:client.cancel_all_goals()
        except Exception:pass
        if navigation_restore is not None:
            try:
                restored_navigation=navigation_configuration.update_configuration(navigation_restore)
                if args.navigation_allow_reverse or args.ros_allow_reverse:
                    result['navigation_reverse_restore']=dict(requested=navigation_restore,
                        applied_response=restored_navigation,
                        restored=abs(float(restored_navigation['min_vel_x'])-float(navigation_restore['min_vel_x']))<1e-9)
                    save(output/'navigation_reverse_restore.json',result['navigation_reverse_restore'])
            except Exception as exc:result['navigation_restore_error']=str(exc)
        def stop_feedback_sample(_poll_now):
            # Sample clocks after taking the CAN-reader lock. A concurrent
            # fresh frame must never appear newer than the sampled clock.
            with lock:
                now=time.monotonic()
                reason=stop_gate.hardware_reason(now)
                confirmed=(reason is None and stop_gate.can_zero_since is not None
                           and now-stop_gate.can_zero_since>=1.)
                snapshot=dict(sampled_monotonic=now,
                    zero_since_monotonic=stop_gate.can_zero_since,
                    zero_interval_s=None if stop_gate.can_zero_since is None else now-stop_gate.can_zero_since,
                    can_status=None if stop_gate.can_status is None else dict(stop_gate.can_status),
                    can_velocity=None if stop_gate.can_velocity is None else dict(stop_gate.can_velocity),
                    status_source_age_s=None if stop_gate.can_status is None else now-stop_gate.can_status['source'],
                    velocity_source_age_s=None if stop_gate.can_velocity is None else now-stop_gate.can_velocity['source'],
                    hardware_reason=reason,reader_fault=stop_gate.fault)
                return dict(confirmed=confirmed,
                    reason=reason if reason is not None else None if confirmed else
                        'zero_feedback_missing' if stop_gate.can_zero_since is None else 'zero_interval_less_than_1s',
                    snapshot=snapshot)
        def close_actuator_children():
            lifecycle.record('actuator_close_begin')
            for name,p in reversed(children):
                row=dict(name=name)
                try:
                    if p.poll() is None:
                        p.send_signal(signal.SIGINT)
                        try:p.wait(timeout=5.)
                        except subprocess.TimeoutExpired:
                            p.terminate()
                            try:p.wait(timeout=3.)
                            except subprocess.TimeoutExpired:p.kill();p.wait(timeout=2.)
                    row.update(exited=p.poll() is not None,returncode=p.poll())
                except Exception as exc:
                    row.update(exited=False,returncode=None,
                               error=type(exc).__name__+': '+str(exc))
                result['cleanup'].append(row)
            lifecycle.record('actuator_close_end',cleanup=result['cleanup'])
        stop_verification=stop_feedback_cleanup_sequence(stop_feedback_sample,close_actuator_children,
            milestone=lambda phase, value:lifecycle.record(phase,verification=value))
        result['stop_feedback_verification']=stop_verification
        result['feedback_zero_before_close']=stop_verification['before_close']['confirmed']
        for phase in ('before_close','after_close'):
            journal.add(dict(event='stop_feedback_verification',phase=phase,
                             **stop_verification[phase]))
        with lock:
            result.update(actual_CAN_distance_m=hardware.can_distance,actual_CAN_absolute_yaw_rad=hardware.can_absolute_yaw,
                feedback_accounting_complete=hardware.can_accounting_complete,
                feedback_zero_after_close=stop_verification['after_close']['confirmed'],
                goals=state['goals'],goal_results=state['results'],map_messages=state['map_count'],
                arrival_evidence=state['arrival_records'],
                elapsed_run_s=None if run_start is None else time.monotonic()-run_start)
        if stop_feedback is not None:stop_feedback.close()
        journal.close()
        for stream in streams:stream.close()
        result['journal_error']=journal.error
        if (output/'adapter/result.json').is_file():
            final=json.loads((output/'adapter/result.json').read_text())
            result['adapter_result_path']='adapter/result.json'
            result['adapter_shutdown_reason']=final.get('reason')
            result['actual_CAN_distance_m']=final['can_distance_m']
            result['actual_CAN_absolute_yaw_rad']=final['can_absolute_yaw_rad']
            result['feedback_accounting_complete']=final['feedback_accounting_complete']
        if not result['feedback_zero_after_close']:
            execution_interruption(result,'chassis_stop_not_confirmed')
        elif stop_verification['cleanup_error'] or any(row.get('error') for row in result['cleanup']):
            execution_interruption(result,'actuator_cleanup_failed')
        elif journal.error:
            execution_interruption(result,'event_recording_error:'+str(journal.error))
        if args.mode in ('nso','joint'):
            # Audit-only after worker close and chassis stop verification. Keep
            # original supervisor joint_summary and all execution status intact.
            try:
                from final_worker_observation import final_worker_observation_audit
                audit=final_worker_observation_audit(output,result.get('joint_summary'),
                    worker_closed=reference_worker is not None and reference_worker.process.poll() is not None,
                    stop_confirmed=result['feedback_zero_after_close'])
                result.update(audit)
            except Exception as exc:
                result['final_worker_observation_receipt']=dict(status='audit_error',
                    error=type(exc).__name__+': '+str(exc),read_only=True,execution_authority=False)
        result['lifecycle_recording_errors']=list(lifecycle.errors)
        save(output/'result.json',result)
        lifecycle.record('root_saved',path=str(output/'result.json'))
        save(WORK/'STATUS.json',dict(phase='mission_ended',goal_complete=False,last_report=str(output),
                                    mode=args.mode,result=result['status'],updated_unix_s=time.time(),
                                    navigation_source='lidar_rgbd',localization_source=localization_source,
                                    experiment_config_sha256=experiment_sha))
        summary=result_summary(result,output)
        save(output/'summary.json',summary)
        print(json.dumps(summary),flush=True)
        rospy.signal_shutdown('mission ended')

if __name__=='__main__':
    main()

