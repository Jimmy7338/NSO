#!/usr/bin/env python3
"""Visual ROS velocity adapter; no CAN transmission or planning.

The native driver remains the sole actuator sender. A receive-only CAN socket
checks original chassis feedback. This independent attended version runs at
most 600 seconds after arm by default (explicit nso/joint long profile:1200 seconds), with no displacement/path/angular travel limit.
Visual odometry, its velocity-fit health, and depth-derived scan/health must
be fresh and published by the configured bridge. Importing opens no devices.
"""
from duration_profile import STANDARD, validate_duration
import argparse
from collections import deque
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import socket
import struct
import threading
import time



class ControlTimingRing:
    """Best-effort primitive-only diagnostic ring; never a control clock."""
    def __init__(self, capacity=256, clock=time.perf_counter_ns):
        self.rows=deque(maxlen=capacity)
        self.clock=clock
        self.current=None
        self.previous=None
        self.errors=deque(maxlen=8)
        self.saved=False

    def begin(self):
        try:
            tick=self.clock()
            self.current={'cycle_enter_ns':tick}
            self.previous=tick
        except Exception as exc:self.errors.append(type(exc).__name__+': '+str(exc))

    def mark(self, phase, **fields):
        try:
            if self.current is None:return
            tick=self.clock()
            self.current[phase+'_ns']=tick-self.previous
            self.current.update(fields)
            self.previous=tick
        except Exception as exc:self.errors.append(type(exc).__name__+': '+str(exc))

    def finish(self, outcome):
        try:
            if self.current is None:return
            self.current['outcome']=outcome
            self.rows.append(self.current)
            self.current=None
        except Exception as exc:self.errors.append(type(exc).__name__+': '+str(exc))

    def save_fault_once(self, path, reason, evidence):
        # Invoked only after native-stop request, feedback cleanup and original
        # result write. Diagnostic failures cannot prevent these operations.
        if self.saved or not reason:return False
        self.saved=True
        try:
            with Path(path).open('x',encoding='utf8') as stream:
                json.dump(dict(schema='nso.adapter_control_timing.v1',fault=reason,
                    fault_evidence=evidence,capacity=self.rows.maxlen,rows=list(self.rows),
                    errors=list(self.errors),clock='perf_counter_ns; diagnostic only',
                    limits='Wall durations cannot uniquely distinguish OS scheduling, GIL, lock contention or publisher work.'),stream,allow_nan=False,indent=2)
                stream.write('\n')
            return True
        except Exception as exc:
            self.errors.append(type(exc).__name__+': '+str(exc))
            return False


def finite(*values):
    return all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in values)


def visual_scan_geometry_valid(angle_min, angle_max, angle_increment, range_min, range_max, ranges):
    """NaN is unknown, never a clearing ray; a valid depth view may have no obstacles."""
    if not finite(angle_min, angle_max, angle_increment, range_min, range_max):
        return False
    if not (angle_increment > 0 and angle_max > angle_min and 0 <= range_min < range_max):
        return False
    if not ranges or abs((len(ranges)-1)*angle_increment-(angle_max-angle_min)) > angle_increment*.1:
        return False
    return all(math.isnan(value) or (math.isfinite(value) and range_min <= value <= range_max)
               for value in ranges)


def validate_maximum_linear(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= .1:
        raise ValueError('maximum_linear_m_s must be finite numeric in (0, .1]')
    return value


def configuration(cfg):
    """Validate the contract without importing ROS or opening any socket."""
    result = dict(master_uri='http://127.0.0.1:11382', can_interface='can0',
                  navigation_source='visual', laser_health_topic='/nso_visual/laser_health',
                  odom_topic='/nso_visual/odom', scan_topic='/nso_visual/scan',
                  depth_health_topic='/nso_visual/depth_health', odom_health_topic='/nso_visual/odom_health',
                  proposal_topic='/nso_visual/cmd_vel_proposed',
                  bridge_publisher='/nso_visual_sensor_bridge', proposal_publisher='/nso_visual/move_base',
                  driver_publisher='/nso_scout_radius_driver_v1',
                  odom_frame='odom', base_frame='base_link', depth_frame='zed2i_left_camera_optical_frame')
    result.update(cfg)
    result.setdefault('maximum_linear_m_s', .1)
    validate_maximum_linear(result['maximum_linear_m_s'])
    reverse = result.setdefault('navigation_allow_reverse', False)
    mission_mode = result.setdefault('mission_mode', None)
    profile = result.setdefault('duration_profile', STANDARD)
    validate_duration(result.get('duration_s', 600.), profile, mission_mode)
    correction = result.setdefault('nso_correction_reverse', False)
    result.setdefault('correction_supervisor', '/nso_onboard_supervisor')
    if type(correction) is not bool or (correction and mission_mode not in ('nso','joint')):
        raise ValueError('correction reverse requires explicit nso/joint authorization')
    if type(reverse) is not bool or (reverse and mission_mode != 'navigation'):
        raise ValueError('reverse is an explicit navigation setup-only option')
    if result['navigation_source'] not in ('visual', 'lidar_rgbd'):
        raise ValueError('unsupported navigation source')
    if result['master_uri'] != 'http://127.0.0.1:11382':
        raise ValueError('visual runtime requires isolated master 11382')
    for key in ['odom_topic','scan_topic','depth_health_topic','odom_health_topic','proposal_topic',
                'bridge_publisher','proposal_publisher','driver_publisher','laser_health_topic','correction_supervisor']:
        if not isinstance(result[key],str) or not re.fullmatch(r'/[A-Za-z][A-Za-z0-9_/]*',result[key]):
            raise ValueError('invalid absolute ROS name: '+key)
    if len({result[key] for key in ['odom_topic','scan_topic','depth_health_topic','odom_health_topic','proposal_topic']}) != 5:
        raise ValueError('source/proposal topics must be distinct')
    for key in ['odom_frame','base_frame','depth_frame']:
        if not isinstance(result[key],str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_/]*',result[key]):
            raise ValueError('invalid frame: '+key)
    if not re.fullmatch(r'can[0-9]+',result['can_interface']):
        raise ValueError('invalid CAN interface')
    return result


def velocity_proposal_record(gate, values, receipt, caller):
    """Keep raw proposals audit-safe even when rejecting nonfinite numbers."""
    accepted=(gate.fault is None and gate.proposal is not None and
              gate.proposal['number']==gate.proposal_count)
    return dict(event='velocity_proposal', receipt_monotonic=receipt,
                axes=['linear.x', 'linear.y', 'linear.z', 'angular.x', 'angular.y', 'angular.z'],
                raw_values=[x if finite(x) else repr(x) for x in values],
                caller=caller, number=gate.proposal_count,
                accepted=accepted,
                bounded_planar_command=None if not accepted else
                    dict(v=gate.proposal['v'],w=gate.proposal['w']),
                saturated=accepted and (values[0]!=gate.proposal['v'] or values[5]!=gate.proposal['w']),
                fault=gate.fault)


def snapshot(gate, now, wall=None):
    """Pure state copy; caller holds the gate lock and supplies sample clocks.

    CAN source/receipt and zero_since stay in the original monotonic clock
    domain. A newly sampled status never renews a chassis measurement.
    """
    if not finite(now) or (wall is not None and not finite(wall)):
        raise ValueError('finite actuator snapshot clocks required')
    reason=gate.hardware_reason(now)
    return dict(schema='nso_visual_actuator_state/1',session_id=getattr(gate,'session_id',None),
        navigation_source=gate.navigation_source, laser_source=deepcopy(gate.laser_health),
        sampled_monotonic=now,sampled_unix_s=wall,
        CAN_timestamp_basis='original kernel timestamps converted to host monotonic by receive-only reader',
        fault=gate.fault,armed=gate.armed,ever_armed=gate.ever_armed,
        started_monotonic=gate.started,duration_s=gate.duration,duration_profile=gate.duration_profile,
        can_status=deepcopy(gate.can_status),can_velocity=deepcopy(gate.can_velocity),
        can_distance_m=gate.can_distance,can_absolute_yaw_rad=gate.can_absolute_yaw,
        can_zero_since_monotonic=gate.can_zero_since,
        feedback_accounting_complete=gate.can_accounting_complete,hardware_reason=reason,
        zero_feedback_confirmed=(reason is None and gate.can_zero_since is not None and now-gate.can_zero_since>=1.),
        arm_stationarity_metrics=deepcopy(gate.arm_stationarity_metrics),
        correction_reverse_authorized=gate.nso_correction_reverse,
        correction_reverse_active=gate.correction_reverse_active,
        minimum_linear_m_s=gate.minimum_linear_m_s,
        maximum_linear_m_s=gate.maximum_linear_m_s,
        fault_evidence=deepcopy(gate.fault_evidence),
        driver_connected=gate.driver_connected,
        driver_status=None if gate.driver is None else dict(code=gate.driver[0],receipt_monotonic=gate.driver[1]),
        last_control_tick_monotonic=gate.last_tick,
        visual_sources={kind:None if getattr(gate,kind) is None else
                        dict(source_stamp_unix_s=getattr(gate,kind)['stamp'],receipt_monotonic=getattr(gate,kind)['receipt'])
                        for kind in ['odom','scan','odom_health','depth_health']})


class Gate:
    """Pure decision policy. Caller serializes access; clocks are injectable."""
    def __init__(self, duration_s=600., *, proposal_ttl_s=.20, observation_ttl_s=.50,
                 bridge_publisher='/nso_visual_sensor_bridge',
                 proposal_publisher='/nso_visual/move_base',
                 odom_frame='odom', base_frame='base_link',
                 depth_frame='zed2i_left_camera_optical_frame', navigation_source='visual',
                 navigation_allow_reverse=False, mission_mode=None, nso_correction_reverse=False,
                 correction_supervisor='/nso_onboard_supervisor', duration_profile=STANDARD,
                 maximum_linear_m_s=.1):
        self.maximum_linear_m_s = validate_maximum_linear(maximum_linear_m_s)
        self.duration_profile = validate_duration(duration_s, duration_profile, mission_mode)
        if not finite(duration_s, proposal_ttl_s, observation_ttl_s):
            raise ValueError('finite durations required')
        if not .05 <= proposal_ttl_s <= .25 or not .15 <= observation_ttl_s <= .5:
            raise ValueError('duration or freshness bound outside supported limits')
        if type(navigation_allow_reverse) is not bool or (navigation_allow_reverse and mission_mode != 'navigation'):
            raise ValueError('reverse is an explicit navigation setup-only option')
        if type(nso_correction_reverse) is not bool or (nso_correction_reverse and mission_mode not in ('nso','joint')):
            raise ValueError('correction reverse requires explicit nso/joint authorization')
        self.nso_correction_reverse = nso_correction_reverse
        self.correction_supervisor = correction_supervisor
        self.correction_reverse_active = False
        self.navigation_allow_reverse = navigation_allow_reverse
        self.minimum_linear_m_s = -.05 if navigation_allow_reverse else 0.
        self.can_status = None
        self.can_velocity = None
        self.can_distance = 0.
        self.can_absolute_yaw = 0.
        self.can_zero_since = None
        self.can_accounting_complete = True
        self.duration = duration_s
        self.proposal_ttl = proposal_ttl_s
        self.observation_ttl = observation_ttl_s
        self.fault = None
        self.fault_evidence = None
        self.armed = False
        self.ever_armed = False
        self.started = None
        self.origin = None
        self.odom = None
        self.scan = None
        self.depth_health = None
        self.odom_health = None
        if navigation_source not in ('visual', 'lidar_rgbd'):
            raise ValueError('unsupported navigation source')
        self.navigation_source = navigation_source
        self.laser_health = None
        self.pending_sources = {}
        self.bridge_publisher = bridge_publisher
        self.proposal_publisher = proposal_publisher
        self.odom_frame, self.base_frame, self.depth_frame = odom_frame, base_frame, depth_frame
        self.proposal = None
        self.driver = None
        self.history = deque(maxlen=100)
        self.last_tick = None
        self.wait_started = None
        self.await_proposal_after = None
        self.proposal_count = 0
        self.trajectory_distance = 0.
        self.trajectory_absolute_yaw = 0.
        self.zero_published = 0
        self.driver_connected = False
        self.arm_stationarity_metrics = None

    def set_correction_reverse(self, enabled, caller, session):
        if (type(enabled) is not bool or not self.nso_correction_reverse
                or caller != self.correction_supervisor
                or not session or session != getattr(self, 'session_id', None)):
            return False, 'correction_reverse_authorization_rejected'
        if enabled and (not self.armed or self.fault is not None):
            return False, 'correction_reverse_requires_armed_fault_free_session'
        self.correction_reverse_active = enabled
        self.minimum_linear_m_s = -.05 if enabled or self.navigation_allow_reverse else 0.
        if not enabled and self.proposal is not None:
            # Never renew proposal receipt, number or freshness on phase exit.
            self.proposal['v'] = max(self.minimum_linear_m_s, self.proposal['v'])
        return True, 'correction_reverse_enabled' if enabled else 'correction_reverse_disabled'

    def fail(self, reason, evidence=None):
        self.correction_reverse_active = False
        self.minimum_linear_m_s = -.05 if self.navigation_allow_reverse else 0.
        if self.proposal is not None:
            self.proposal['v'] = max(self.minimum_linear_m_s, self.proposal['v'])
        if self.fault is None:
            self.fault = reason
            self.fault_evidence = deepcopy(evidence)
        self.armed = False

    def driver_status(self, code, receipt):
        if not finite(receipt) or type(code) is not int or code != 0:
            self.fail('native_driver_fault:'+str(code))
        self.driver = (code, receipt)

    def hardware(self, cid, payload, source, receipt):
        if cid == 0x211:
            row=dict(source=source,receipt=receipt,state=payload[0],mode=payload[1],
                     error=struct.unpack('>H',payload[4:6])[0])
            old=self.can_status
            self.can_status=row
            if old and source<=old['source']:
                self.fail('chassis_status_source_regressed')
            if self.ever_armed and (row['state'],row['mode'],row['error'])!=(0,1,0):
                self.fail('chassis_state_mode_or_fault')
        elif cid == 0x221:
            values=tuple(x*.001 for x in struct.unpack('>hh',payload[:4]))
            old=self.can_velocity
            row=dict(source=source,receipt=receipt,v=values[0],w=values[1],uninterpreted_tail_hex=payload[4:].hex())
            self.can_velocity=row
            if old:
                dt=source-old['source']
                if dt<=0 or (self.ever_armed and dt>.15):
                    if self.ever_armed:self.can_accounting_complete=False
                    self.fail('chassis_velocity_source_gap')
                elif self.ever_armed:
                    self.can_distance+=(abs(old['v'])+abs(row['v']))*.5*dt
                    self.can_absolute_yaw+=(abs(old['w'])+abs(row['w']))*.5*dt
            if max(map(abs,values)) <= .005:
                if self.can_zero_since is None:self.can_zero_since=source
            else:self.can_zero_since=None
            if self.ever_armed and (abs(row['v'])>.15 or abs(row['w'])>.45):
                self.fail('chassis_speed_feedback_limit', dict(
                    can_id='0x221', payload_hex=payload.hex(),
                    timestamp_semantics='original kernel timestamp converted to host monotonic by receive-only reader',
                    limits=dict(maximum_abs_v_m_s=.15, maximum_abs_w_rad_s=.45),
                    source_monotonic=source, receipt_monotonic=receipt,
                    can_velocity=deepcopy(row), proposal=deepcopy(self.proposal),
                    can_status=deepcopy(self.can_status)))

    def hardware_reason(self, now):
        for name,row,ttl in [('status',self.can_status,.5),('velocity',self.can_velocity,.15)]:
            if row is None or now-row['source']>ttl or now<row['source']:
                return 'chassis_'+name+'_stale'
        if (self.can_status['state'],self.can_status['mode'],self.can_status['error'])!=(0,1,0):
            return 'chassis_state_mode_or_fault'
        return None

    def observe(self, kind, stamp, receipt, wall, *, x=None, y=None, yaw=None, v=None, w=None,
                caller=None, frame=None, child_frame=None):
        if kind not in ('odom', 'scan', 'depth_health', 'odom_health'):
            raise ValueError('unknown source')
        if caller != self.bridge_publisher:
            self.fail('unexpected_'+kind+'_publisher:'+str(caller)); return
        expected_frame = dict(odom=self.odom_frame, scan=self.base_frame,
                              depth_health=self.depth_frame, odom_health=self.odom_frame)[kind]
        if frame != expected_frame or (kind == 'odom' and child_frame != self.base_frame):
            self.fail(kind+'_frames_differ'); return
        if not finite(stamp, receipt, wall) or stamp <= 0 or stamp-wall > .02:
            self.fail(kind+'_invalid_or_future_stamp')
            return
        previous = self.pending_sources.get(kind, getattr(self, kind))
        if previous is not None and stamp < previous['stamp']:
            self.fail(kind+'_stamp_regressed')
            return
        if previous is not None and stamp == previous['stamp']:
            return  # duplicate source messages never renew receipt age
        if wall-stamp > self.observation_ttl:
            return  # old unadmitted packet; it does not renew a live source
        row = dict(stamp=stamp, receipt=receipt)
        if kind == 'odom':
            if not finite(x, y, yaw, v, w):
                self.fail('nonfinite_odometry')
                return
            row.update(x=x, y=y, yaw=yaw, v=v, w=w)
            if previous is not None:
                dt = stamp-previous['stamp']
                distance = math.hypot(x-previous['x'], y-previous['y'])
                angle = abs(math.atan2(math.sin(yaw-previous['yaw']), math.cos(yaw-previous['yaw'])))
                if distance > .02+.20*dt or angle > .04+.60*dt:
                    self.fail('odometry_jump', evidence=dict(
                        previous=deepcopy(previous), new=deepcopy(row),
                        source_dt_s=dt, planar_distance_m=distance,
                        absolute_yaw_delta_rad=angle,
                        existing_distance_bound_m=.02+.20*dt,
                        existing_angle_bound_rad=.04+.60*dt,
                        previous_source='pending-or-admitted source used by unchanged guard',
                        source_stamp_semantics='original incoming source seconds; not renewed'))
                    return
        self.pending_sources[kind] = row
        observation, health = ('odom', 'odom_health') if kind in ('odom', 'odom_health') else ('scan', 'depth_health')
        observation_row = self.pending_sources.get(observation)
        health_row = self.pending_sources.get(health)
        # Commit one source/health pair atomically. Separate ROS subscriber
        # callbacks cannot momentarily replace healthy data with an unmatched
        # newest sample, or renew odometry with an old health heartbeat.
        if observation_row and health_row and abs(observation_row['stamp']-health_row['stamp']) <= 1e-6:
            old_admitted = getattr(self, observation)
            if old_admitted is None or observation_row['stamp'] > old_admitted['stamp']:
                if observation == 'odom':
                    if old_admitted is not None and self.ever_armed:
                        self.trajectory_distance += math.hypot(observation_row['x']-old_admitted['x'],observation_row['y']-old_admitted['y'])
                        self.trajectory_absolute_yaw += abs(math.atan2(math.sin(observation_row['yaw']-old_admitted['yaw']),math.cos(observation_row['yaw']-old_admitted['yaw'])))
                    self.history.append(observation_row)
                setattr(self, observation, observation_row)
                setattr(self, health, health_row)

    def observe_laser_health(self, row, caller, receipt, wall):
        """Independent raw-laser freshness in the lidar/RGB-D profile.

        A repeated bridge heartbeat cannot renew the original scan stamp.
        Invalid data stops admission immediately; source faults remain latched.
        """
        if caller != self.bridge_publisher:
            self.fail('unexpected_laser_health_publisher:'+str(caller)); return
        if (not isinstance(row, dict) or row.get('source') != 'lslidar_N301'
                or row.get('caller') != '/lslidar_driver_node'
                or row.get('frame') != 'laser_link'):
            self.fail('laser_health_provenance_invalid'); return
        if row.get('fault'):
            self.fail('laser_source_fault:'+str(row['fault'])); return
        if row.get('valid') is not True:
            self.laser_health = None
            return
        stamp_ns = row.get('source_stamp_ns')
        if (type(stamp_ns) is not int or stamp_ns <= 0
                or not finite(receipt, wall) or stamp_ns/1e9 > wall+.02):
            self.fail('laser_source_stamp_invalid'); return
        stamp = stamp_ns/1e9
        previous = self.laser_health
        if previous is not None:
            if stamp_ns < previous['stamp_ns']:
                self.fail('laser_source_stamp_regressed'); return
            if stamp_ns == previous['stamp_ns']:
                return
        if wall-stamp > self.observation_ttl:
            return
        self.laser_health = dict(stamp=stamp, stamp_ns=stamp_ns, receipt=receipt,
                                 source=row['source'], caller=row['caller'], frame=row['frame'])

    def propose(self, values, receipt, caller):
        self.proposal_count += 1
        if caller != self.proposal_publisher:
            self.fail('unexpected_proposal_publisher:'+str(caller))
            return
        if len(values) != 6 or not finite(*values, receipt):
            self.fail('nonfinite_velocity_proposal')
            return
        v, vy, vz, wx, wy, w = values
        if vy != 0 or vz != 0 or wx != 0 or wy != 0:
            self.fail('velocity_proposal_outside_bounds')
            return
        # Saturate trusted planar navigation proposals at the existing physical
        # command limits; reverse is available only through the setup-only opt-in.
        # No extra numerical-excess threshold changes admission.
        # Raw proposals and bounded values remain available in the sampled log.
        # A new receipt replaces the proposal. No output operation changes it.
        self.proposal = dict(v=max(self.minimum_linear_m_s,min(v,self.maximum_linear_m_s)), w=max(-.3,min(w,.3)), receipt=receipt,
                             number=self.proposal_count)

    def sources_reason(self, now, wall):
        if not self.driver_connected:
            return 'native_driver_not_connected'
        if self.driver is None or not 0 <= now-self.driver[1] <= .20:
            return 'native_driver_status_stale'
        if self.navigation_source == 'lidar_rgbd':
            laser = self.laser_health
            if laser is None:
                return 'laser_source_missing_or_invalid'
            if laser['receipt'] > now or laser['stamp'] > wall+.02:
                self.fail('laser_source_clock_regressed')
                return self.fault
            if now-laser['receipt'] > self.observation_ttl or wall-laser['stamp'] > self.observation_ttl:
                return 'laser_source_stale'
        for kind in ('odom', 'scan', 'depth_health', 'odom_health'):
            row = getattr(self, kind)
            if row is None:
                return kind+'_missing'
            if row['receipt'] > now or row['stamp'] > wall+.02:
                self.fail(kind+'_clock_regressed')
                return self.fault
            if now-row['receipt'] > self.observation_ttl or wall-row['stamp'] > self.observation_ttl:
                return kind+'_stale'
        for observation, health in [('odom', 'odom_health'), ('scan', 'depth_health')]:
            if abs(getattr(self, observation)['stamp']-getattr(self, health)['stamp']) > 1e-6:
                return observation+'_health_stamp_mismatch'
        return None

    def arm(self, now, wall):
        if self.ever_armed or self.fault:
            return False, 'session_already_used_or_faulted'
        reason = self.sources_reason(now, wall)
        if reason:
            return False, reason
        reason = self.hardware_reason(now)
        if reason:
            return False, reason
        if self.can_zero_since is None or now-self.can_zero_since<1.:
            return False, 'need_one_second_actual_zero_feedback'
        recent = [r for r in self.history if now-r['receipt'] <= 1.5]
        if self.zero_published < 10 or len(recent) < 4 or recent[-1]['receipt']-recent[0]['receipt'] < 1.:
            return False, 'need_zero_outputs_and_one_second_stationary_odometry'
        self.arm_stationarity_metrics = dict(
            window_s=recent[-1]['receipt']-recent[0]['receipt'],count=len(recent),
            maximum_position_displacement_m=max(math.hypot(r['x']-recent[0]['x'],r['y']-recent[0]['y']) for r in recent),
            maximum_yaw_displacement_rad=max(abs(math.atan2(math.sin(r['yaw']-recent[0]['yaw']),
                math.cos(r['yaw']-recent[0]['yaw']))) for r in recent),
            maximum_odom_derivative_v_m_s=max(abs(r['v']) for r in recent),
            maximum_odom_derivative_w_rad_s=max(abs(r['w']) for r in recent),
            actual_zero_feedback_duration_s=now-self.can_zero_since,
            derivative_velocities_used_for_stationary_decision=False,
            stationary_basis='continuous raw chassis 0x221 zero feedback >=1s; odometry displacement is diagnostic',
            navigation_source=self.navigation_source, source_quality_required=True)
        # A stationary camera estimate can drift beyond 1 cm. Mechanical
        # stationarity comes from fresh raw chassis velocity for >=1 second,
        # while visual continuity/jump checks and fit-health remain mandatory.
        # Reference for evidence only, never a motion boundary.
        self.origin = (self.odom['x'], self.odom['y'], self.odom['yaw'])
        self.started = now
        self.armed = self.ever_armed = True
        self.await_proposal_after = now
        self.proposal = None  # proposals before arming cannot move the robot
        self.history.clear()
        return True, 'armed_once'

    def decision(self, now, wall):
        if not finite(now, wall):
            self.fail('invalid_clock')
        previous_tick=self.last_tick
        if self.last_tick is not None and (now < self.last_tick or now-self.last_tick > .08):
            self.fail('adapter_control_tick_invalid',dict(
                now_monotonic=now,previous_tick_monotonic=previous_tick,
                delta_s=now-previous_tick,threshold_s=.08,
                classification='clock_regressed' if now<previous_tick else 'tick_overrun'))
        self.last_tick = now
        if self.fault or not self.armed:
            return 0., 0., self.fault or 'not_armed'
        if now-self.started >= self.duration:
            self.fail('duration_complete')
            return 0., 0., self.fault
        hardware_reason = self.hardware_reason(now)
        if hardware_reason:
            if hardware_reason=='chassis_velocity_stale':self.can_accounting_complete=False
            self.fail(hardware_reason)
            return 0.,0.,self.fault
        reason = self.sources_reason(now, wall)
        if reason:
            self.proposal = None
            self.await_proposal_after = now
            if self.wait_started is None:
                self.wait_started = now
            if now-self.wait_started > 1.:
                self.fail('source_loss:'+reason)
            return 0., 0., self.fault or reason
        self.wait_started = None
        row = self.proposal
        if row is None or row['receipt'] <= self.await_proposal_after:
            return 0., 0., 'await_new_proposal'
        if now < row['receipt']:
            self.fail('proposal_receipt_clock_regressed')
            return 0., 0., self.fault
        if now-row['receipt'] > self.proposal_ttl:
            return 0., 0., 'proposal_expired'
        return row['v'], row['w'], 'current_proposal'


class EventSampler:
    """Sample logs only; command admission and CAN accounting never use this."""
    periodic={'command','velocity_proposal','odom','scan','depth_health','odom_health',
              'laser_source_health','CAN_feedback'}

    def __init__(self, interval_s=1.):
        self.interval_s=interval_s
        self.last={}
        self.pending={}

    @staticmethod
    def timestamp(value):
        return value.get('at_monotonic',value.get('receipt_monotonic'))

    def keep(self, value):
        event=value.get('event')
        if event not in self.periodic:
            return True
        now=self.timestamp(value)
        if not finite(now):
            return True
        key=(event,value.get('id'))
        if event=='command':
            signature=(value.get('v')!=0 or value.get('w')!=0,value.get('reason'))
        elif event=='CAN_feedback':
            signature=(value.get('fault'),value.get('ever_armed'),value.get('moving'),
                       value.get('state'),value.get('mode'),value.get('error'))
        elif event=='velocity_proposal':
            values=value.get('raw_values',[])
            moving=bool(len(values)==6 and (values[0]!=0 or values[5]!=0))
            signature=(value.get('accepted'),value.get('fault'),moving,value.get('caller'))
        elif event=='laser_source_health':
            source=value.get('source') or {}
            signature=(source.get('valid'),source.get('fault'),source.get('reason'),value.get('caller'))
        else:
            signature=(value.get('caller'),value.get('frame'),value.get('child_frame'))
        previous=self.last.get(key)
        if (previous is None or signature!=previous[1] or now<previous[0]
                or now-previous[0]>=self.interval_s):
            self.last[key]=(now,signature)
            self.pending.pop(key,None)
            return True
        self.pending[key]=value
        return False

    def finish(self):
        # Preserve the actual last sample, including a repeated final zero.
        rows=sorted(self.pending.values(),key=self.timestamp)
        self.pending.clear()
        return rows


class Journal:
    """Disk writes stay off the control thread; overflow ends motion."""
    def __init__(self, path):
        self.queue = queue.Queue(maxsize=4096)
        self.error = None
        self.sampler=EventSampler()
        self.sample_lock=threading.Lock()
        self.closed=False
        self.stream = path.open('x', encoding='utf8')
        self.thread = threading.Thread(target=self.write, daemon=True)
        self.thread.start()

    def add(self, value):
        with self.sample_lock:
            if self.closed or not self.sampler.keep(value):
                return
            try:
                self.queue.put_nowait(value)
            except queue.Full:
                self.error = 'journal_queue_full'

    def write(self):
        try:
            while True:
                item = self.queue.get()
                if item is None:
                    break
                self.stream.write(json.dumps(item, allow_nan=False, separators=(',',':'))+'\n')
            self.stream.flush()
        except Exception as exc:
            self.error = 'journal_error:'+type(exc).__name__
        finally:
            self.stream.close()

    def close(self):
        with self.sample_lock:
            self.closed=True
            for row in self.sampler.finish():
                try:self.queue.put_nowait(row)
                except queue.Full:self.error='journal_queue_full'
        try:
            self.queue.put(None, timeout=1.)
        except queue.Full:
            self.error = 'journal_close_queue_full'
        self.thread.join(timeout=3.)
        if self.thread.is_alive():
            self.error = 'journal_close_timeout'


class FeedbackReader:
    """Receive-only extraction from scout_tether_attended_control_v4 RawCan.

    Same SocketCAN standard-frame/error filters, kernel SO_TIMESTAMPNS and
    overflow checks, big-endian V2 0x211/0x221 fields. No send method exists.
    """
    def __init__(self, gate, lock, journal, can_interface='can0'):
        self.gate,self.lock,self.journal=gate,lock,journal
        self.offset=time.time()-time.monotonic()
        self.done=threading.Event()
        self.socket=socket.socket(socket.AF_CAN,socket.SOCK_RAW,socket.CAN_RAW)
        try:
            self.socket.setsockopt(socket.SOL_CAN_RAW,socket.CAN_RAW_FILTER,b''.join(
                struct.pack('=II',cid,0xc00007ff) for cid in [0x211,0x221]))
            self.socket.setsockopt(socket.SOL_CAN_RAW,socket.CAN_RAW_ERR_FILTER,struct.pack('=I',0x1fffffff))
            self.socket.setsockopt(socket.SOL_SOCKET,35,1)
            self.socket.setsockopt(socket.SOL_SOCKET,40,1)
            self.socket.bind((can_interface,));self.socket.settimeout(.05)
        except BaseException:
            self.socket.close();raise
        self.thread=threading.Thread(target=self.receive,daemon=True)
        self.thread.start()

    def receive(self):
        try:
            while not self.done.is_set():
                try:raw,anc,flags,_=self.socket.recvmsg(16,128)
                except socket.timeout:continue
                now,wall=time.monotonic(),time.time()
                if len(raw)!=16 or flags&(8|32) or abs(wall-now-self.offset)>.02:
                    raise ValueError('CAN raw/clock/truncation')
                stamps=[]
                for level,kind,value in anc:
                    if level==socket.SOL_SOCKET and kind==35:
                        sec,ns=struct.unpack('@ll',value[:struct.calcsize('@ll')])
                        if not 0<=ns<1000000000:raise ValueError('CAN timestamp')
                        stamps.append(sec+ns/1e9)
                    if level==socket.SOL_SOCKET and kind==40 and struct.unpack('=I',value[:4])[0]!=0:
                        raise ValueError('CAN receive overflow')
                if len(stamps)!=1:raise ValueError('CAN kernel timestamp missing')
                cid,dlc,payload=struct.unpack('=IB3x8s',raw)
                source=stamps[0]-self.offset
                if cid&0xe0000000 or dlc!=8 or cid not in (0x211,0x221) or not 0<=now-source<=.15:
                    raise ValueError('CAN id/length/source-age')
                with self.lock:
                    self.gate.hardware(cid,payload,source,now)
                    # Counters still integrate every original CAN frame. Only
                    # disk records are sampled; motion edges keep source times.
                    row=dict(event='CAN_feedback',id=hex(cid),raw_frame_hex=raw.hex(),
                        receipt_monotonic=now,source_monotonic=source,kernel_unix_s=stamps[0],
                        fault=self.gate.fault,ever_armed=self.gate.ever_armed,
                        can_distance_m=self.gate.can_distance,
                        can_absolute_yaw_rad=self.gate.can_absolute_yaw,
                        feedback_accounting_complete=self.gate.can_accounting_complete)
                    if cid==0x221:
                        velocity=self.gate.can_velocity
                        row.update(v=velocity['v'],w=velocity['w'],
                            moving=max(abs(velocity['v']),abs(velocity['w']))>.005,
                            motion_threshold=.005)
                    else:
                        row.update({key:self.gate.can_status[key] for key in ('state','mode','error')})
                    self.journal.add(row)
        except Exception as exc:
            with self.lock:
                if self.gate.ever_armed:self.gate.can_accounting_complete=False
                self.gate.fail('CAN_receive_fault:'+type(exc).__name__+':'+str(exc))

    def close(self):
        self.done.set();self.thread.join(timeout=1.);self.socket.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--run', action='store_true', help='Explicit zero-only ROS startup; arm service remains required')
    args = parser.parse_args()
    raw = Path(args.config).read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.config_sha256:
        raise ValueError('configuration hash differs')
    cfg = configuration(json.loads(raw))
    session = cfg['session_id']
    if not re.fullmatch('[A-Za-z0-9_-]{1,128}', session):
        raise ValueError('invalid session')
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != cfg['boot_id']:
        raise ValueError('boot changed')
    gate = Gate(cfg.get('duration_s',600.), **{key:cfg[key] for key in
        ['bridge_publisher','proposal_publisher','odom_frame','base_frame','depth_frame','navigation_source','navigation_allow_reverse','mission_mode','nso_correction_reverse','correction_supervisor','duration_profile','maximum_linear_m_s']})
    gate.session_id=session
    if not args.run:
        print(json.dumps(dict(status='configuration_validated_only',no_ros_started=True)))
        return
    if os.environ.get('ROS_MASTER_URI') != cfg['master_uri']:
        raise ValueError('set ROS_MASTER_URI to isolated visual master 11382')
    import rospy
    from geometry_msgs.msg import Twist, TwistStamped
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import Bool, Header, Int32, String
    from std_srvs.srv import Trigger, TriggerResponse, SetBool, SetBoolResponse
    rospy.init_node('nso_radius_guard', anonymous=False, disable_signals=False)
    if rospy.get_name()!='/nso_radius_guard':
        raise ValueError('adapter must run at root namespace as /nso_radius_guard')
    if rospy.get_param('/use_sim_time',False):
        raise ValueError('wall-clock ROS time required')
    output = Path(cfg['output_directory'])
    output.mkdir(parents=True, exist_ok=False)
    journal = Journal(output/'events.jsonl')
    lock = threading.RLock()
    feedback = FeedbackReader(gate,lock,journal,cfg['can_interface'])
    publisher = rospy.Publisher('/nso_radius/guarded_cmd', TwistStamped, queue_size=1, tcp_nodelay=True)
    stopper = rospy.Publisher('/nso_radius/stop', Bool, queue_size=1, latch=True)
    state_publisher = rospy.Publisher('/nso_visual/actuator_state',String,queue_size=1,tcp_nodelay=True,latch=True)
    started = time.monotonic()
    pending_deadline = started+3600.
    last_stamp = 0
    sequence = 0
    next_state_monotonic=0.
    control_timing=ControlTimingRing()

    def publish_state(state):
        try:
            state_publisher.publish(String(data=json.dumps(state,allow_nan=False,separators=(',',':'))))
        except (rospy.ROSException,TypeError,ValueError) as exc:
            # Monitoring transport never changes actuator admission or prevents
            # the existing native stop/feedback cleanup from running.
            rospy.logwarn_throttle(1.,'actuator state publication failed: '+str(exc))

    def odom(msg):
        p,q,t = msg.pose.pose.position,msg.pose.pose.orientation,msg.twist.twist
        now, wall = time.monotonic(), rospy.Time.now().to_sec()
        with lock:
            if not finite(q.x,q.y,q.z,q.w) or abs(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w-1.)>.01:
                gate.fail('odometry_quaternion_invalid'); return
            yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
            caller=getattr(msg,'_connection_header',{}).get('callerid')
            previous_fault=gate.fault
            gate.observe('odom',msg.header.stamp.to_sec(),now,wall,x=p.x,y=p.y,yaw=yaw,v=t.linear.x,w=t.angular.z,
                         caller=caller,frame=msg.header.frame_id,child_frame=msg.child_frame_id)
            if previous_fault is None and gate.fault_evidence is not None:
                journal.add(dict(event='first_fault_evidence',reason=gate.fault,
                    evidence=deepcopy(gate.fault_evidence),receipt_monotonic=now,
                    receipt_unix_s=wall,new_source_stamp_ns=msg.header.stamp.to_nsec(),
                    caller=caller,frame=msg.header.frame_id,child_frame=msg.child_frame_id))
            journal.add(dict(event='odom',receipt_monotonic=now,source_stamp=msg.header.stamp.to_sec(),
                             x=p.x,y=p.y,yaw=yaw,v=t.linear.x,w=t.angular.z,caller=caller,
                             frame=msg.header.frame_id,child_frame=msg.child_frame_id,
                             pose_covariance=list(msg.pose.covariance),twist_covariance=list(msg.twist.covariance)))

    def scan(msg):
        with lock:
            if not visual_scan_geometry_valid(msg.angle_min,msg.angle_max,msg.angle_increment,
                                               msg.range_min,msg.range_max,msg.ranges):
                gate.fail('visual_scan_geometry_invalid'); return
            caller=getattr(msg,'_connection_header',{}).get('callerid')
            now=time.monotonic()
            gate.observe('scan',msg.header.stamp.to_sec(),now,rospy.Time.now().to_sec(),
                         caller=caller,frame=msg.header.frame_id)
            journal.add(dict(event='scan',source_stamp=msg.header.stamp.to_sec(),receipt_monotonic=now,
                             frame=msg.header.frame_id,caller=caller,
                             finite_returns=sum(math.isfinite(value) for value in msg.ranges)))

    def health(kind,msg):
        with lock:
            caller=getattr(msg,'_connection_header',{}).get('callerid')
            now=time.monotonic()
            gate.observe(kind,msg.stamp.to_sec(),now,rospy.Time.now().to_sec(),
                         caller=caller,frame=msg.frame_id)
            journal.add(dict(event=kind,source_stamp=msg.stamp.to_sec(),receipt_monotonic=now,
                             frame=msg.frame_id,caller=caller))

    def proposal(msg):
        with lock:
            values=[msg.linear.x,msg.linear.y,msg.linear.z,msg.angular.x,msg.angular.y,msg.angular.z]
            receipt=time.monotonic()
            caller=getattr(msg,'_connection_header',{}).get('callerid')
            gate.propose(values,receipt,caller)
            journal.add(velocity_proposal_record(gate,values,receipt,caller))

    def laser_health(msg):
        with lock:
            now, wall = time.monotonic(), rospy.Time.now().to_sec()
            caller = getattr(msg, '_connection_header', {}).get('callerid')
            try:
                row = json.loads(msg.data)
                gate.observe_laser_health(row, caller, now, wall)
                journal.add(dict(event='laser_source_health',receipt_monotonic=now,
                                 caller=caller,source=row))
            except (ValueError, TypeError) as exc:
                gate.fail('laser_health_invalid:'+str(exc))

    def driver(msg):
        with lock:
            if getattr(msg,'_connection_header',{}).get('callerid') != cfg['driver_publisher']:
                gate.fail('unexpected_native_driver_publisher'); return
            gate.driver_status(msg.data,time.monotonic())

    def arm(_):
        with lock:
            gate.driver_connected = publisher.get_num_connections() == 1
            ok,reason = gate.arm(time.monotonic(),rospy.Time.now().to_sec())
            journal.add(dict(event='arm',ok=ok,reason=reason,origin=gate.origin,
                stationarity_metrics=gate.arm_stationarity_metrics,at_monotonic=time.monotonic()))
            return TriggerResponse(success=ok,message=reason)

    def stop(_):
        with lock:
            gate.fail('explicit_stop')
        return TriggerResponse(success=True,message='stop_latched')

    subscribers = [rospy.Subscriber(cfg['odom_topic'],Odometry,odom,queue_size=1,tcp_nodelay=True),
        rospy.Subscriber(cfg['scan_topic'],LaserScan,scan,queue_size=1,tcp_nodelay=True),
        rospy.Subscriber(cfg['depth_health_topic'],Header,lambda msg:health('depth_health',msg),queue_size=1,tcp_nodelay=True),
        rospy.Subscriber(cfg['odom_health_topic'],Header,lambda msg:health('odom_health',msg),queue_size=1,tcp_nodelay=True),
        rospy.Subscriber(cfg['proposal_topic'],Twist,proposal,queue_size=1,tcp_nodelay=True),
        rospy.Subscriber('/nso_radius/driver_fault',Int32,driver,queue_size=1,tcp_nodelay=True)]
    if cfg['navigation_source'] == 'lidar_rgbd':
        subscribers.append(rospy.Subscriber(cfg['laser_health_topic'],String,laser_health,
                                            queue_size=1,tcp_nodelay=True))
    services = [rospy.Service('/nso_visual/arm',Trigger,arm),rospy.Service('/nso_visual/stop',Trigger,stop)]
    if cfg['nso_correction_reverse']:
        def correction_phase(request):
            caller=getattr(request,'_connection_header',{}).get('callerid')
            with lock:
                ok,reason=gate.set_correction_reverse(request.data,caller,session)
                journal.add(dict(event='correction_reverse_phase',enabled=request.data,ok=ok,reason=reason,
                    caller=caller,session_id=session,at_monotonic=time.monotonic(),
                    active=gate.correction_reverse_active,minimum_linear_m_s=gate.minimum_linear_m_s))
            return SetBoolResponse(success=ok,message=reason)
        services.append(rospy.Service('/nso_visual/correction_reverse/'+session,SetBool,correction_phase))
    (output/'started.json').write_text(json.dumps(dict(session_id=session,boot_id=cfg['boot_id'],
        started_monotonic=started,config_sha256=args.config_sha256,zero_only_until_arm=True,
        correction_reverse_authorized=gate.nso_correction_reverse,
        correction_reverse_service=None if not gate.nso_correction_reverse else '/nso_visual/correction_reverse/'+session,
        passive_chassis_supervision=True,duration_s=gate.duration,duration_profile=gate.duration_profile,navigation_source=cfg['navigation_source'],
        actuator_state_topic='/nso_visual/actuator_state',actuator_state_frequency_hz=10.,
        disk_recording=dict(periodic_sample_hz=1.,motion_and_fault_transitions=True,
                            CAN_counters_use_all_original_feedback=True),
        source_contract={key:cfg[key] for key in ['odom_topic','scan_topic','depth_health_topic','odom_health_topic',
                        'bridge_publisher','proposal_publisher','driver_publisher','odom_frame','base_frame','depth_frame',
                        'laser_health_topic','navigation_source']},
        motion_deadline_starts_at_arm=True,pending_deadline_monotonic=pending_deadline,
        motion_envelope=dict(max_linear_m_s=gate.maximum_linear_m_s,max_angular_rad_s=.3,
            min_linear_m_s=gate.minimum_linear_m_s,no_reverse=not gate.navigation_allow_reverse,
            reverse_profile='navigation_setup_reverse_005_v1' if gate.navigation_allow_reverse else None,
            spatial_limit=None,path_limit=None,angular_travel_limit=None)),indent=2),encoding='utf8')
    try:
        while not rospy.is_shutdown():
            control_timing.begin()
            state=None
            with lock:
                control_timing.mark('lock_wait')
                # Capture clocks after acquiring the same lock as callbacks.
                # A callback while waiting must not look newer than this check.
                now=time.monotonic(); wall=rospy.Time.now()
                gate.driver_connected = publisher.get_num_connections() == 1
                if publisher.get_num_connections() > 1:
                    gate.fail('multiple_command_subscribers')
                if journal.error:
                    gate.fail(journal.error)
                if not gate.ever_armed and now>=pending_deadline:
                    gate.fail('not_armed_timeout')
                control_timing.mark('pre_decision',gate_now_monotonic=now)
                v,w,reason=gate.decision(now,wall.to_sec())
                control_timing.mark('decision')
                stamp=wall.to_nsec()
                if stamp<=last_stamp:
                    gate.fail('ROS_output_clock_regressed')
                    control_timing.mark('output_clock_check')
                    control_timing.finish('ROS_output_clock_regressed')
                    break
                last_stamp=stamp; sequence+=1
                message=TwistStamped();message.header.seq=sequence;message.header.stamp=wall
                message.header.frame_id=session;message.twist.linear.x=v;message.twist.angular.z=w
                control_timing.mark('guard_message_build')
                publisher.publish(message)
                control_timing.mark('guarded_publish')
                if v==0 and w==0:
                    gate.zero_published+=1
                control_timing.mark('pre_journal')
                journal.add(dict(event='command',at_monotonic=now,guard_stamp_ns=stamp,v=v,w=w,reason=reason,
                    proposal=None if gate.proposal is None else dict(gate.proposal),
                    odom_source_stamp=None if gate.odom is None else gate.odom['stamp'],
                    scan_source_stamp=None if gate.scan is None else gate.scan['stamp'],
                    depth_health_source_stamp=None if gate.depth_health is None else gate.depth_health['stamp'],
                    odom_health_source_stamp=None if gate.odom_health is None else gate.odom_health['stamp']))
                control_timing.mark('journal_add')
                if now>=next_state_monotonic or gate.fault:
                    state=snapshot(gate,now,wall.to_sec())
                    state['last_command']=dict(v=v,w=w,reason=reason,at_monotonic=now)
                    next_state_monotonic=now+.1
                fault=gate.fault
                control_timing.mark('snapshot')
            control_timing.mark('lock_exit')
            if state is not None:publish_state(state)
            control_timing.mark('state_publish',state_published=state is not None)
            if fault:
                control_timing.finish(fault)
                break
            # Original sleep expression/predicate stays untouched. Requested
            # duration is derived from the same existing monotonic sample.
            sleep_requested=max(0.,.02-(time.monotonic()-now))
            control_timing.mark('pre_sleep',sleep_requested_s=sleep_requested)
            time.sleep(sleep_requested)
            control_timing.mark('sleep')
            control_timing.finish('continue')
    finally:
        with lock:
            gate.fail('adapter_shutdown')
            shutdown_state=snapshot(gate,time.monotonic(),time.time())
        stopper.publish(Bool(data=True))
        publish_state(shutdown_state)
        # Native stop latch provides its own zero attempts; no rearming here.
        time.sleep(1.25)
        feedback.close()
        with lock:
            final_state=snapshot(gate,time.monotonic(),time.time())
            final_state['finished']=True
        publish_state(final_state)
        journal.add(dict(event='finished',reason=gate.fault,origin=gate.origin,
            distance_from_odometry_m=gate.trajectory_distance,
            absolute_yaw_from_odometry_rad=gate.trajectory_absolute_yaw,
            actual_chassis_feedback_covered=True,can_distance_m=gate.can_distance,
            can_absolute_yaw_rad=gate.can_absolute_yaw,closed_monotonic=time.monotonic()))
        journal.close()
        result=dict(session_id=session,reason=gate.fault,ever_armed=gate.ever_armed,origin=gate.origin,
            fault_evidence=deepcopy(gate.fault_evidence),
            odometry_distance_m=gate.trajectory_distance,odometry_absolute_yaw_rad=gate.trajectory_absolute_yaw,
            journal_error=journal.error,native_stop_requested=True,
            can_distance_m=gate.can_distance,can_absolute_yaw_rad=gate.can_absolute_yaw,
            feedback_accounting_complete=gate.can_accounting_complete,
            spatial_limit=None,path_limit=None,angular_travel_limit=None,
            zero_feedback_confirmed=(gate.hardware_reason(time.monotonic()) is None and
                gate.can_zero_since is not None and time.monotonic()-gate.can_zero_since>=1.),
            physical_stop_independently_observed=False)
        (output/'result.json').write_text(json.dumps(result,indent=2),encoding='utf8')
        control_timing.finish('exception_or_shutdown')
        control_timing.save_fault_once(output/'control_timing_fault.json',gate.fault,gate.fault_evidence)


if __name__ == '__main__':
    main()
