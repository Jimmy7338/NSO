"""Bounded acquisition/controller/map driver; simulator ownership stays here.

The generic loop is exercised with finite, predeclared packet fixtures. Only
the separate gated development entry point may construct a study sensor.
"""
from copy import deepcopy
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import time

import numpy as np

from env.development_sensor_v41 import SensorStepV41
from nso.instance_belief_v40 import PaidRGBDObservationV40
from utils.rgbd_contract import PlanarScan


ACTION_TO_SENSOR_V43 = {'forward': 'forward', 'left': 'turn_left',
                        'right': 'turn_right', 'observe': 'observe'}


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       ensure_ascii=False, allow_nan=False)+'\n').encode()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class OutputLimitExceededV43(RuntimeError):
    pass


class BoundedRunWriterV43:
    """New files only, with a reserved terminal-record allowance and fsync.

    Files already in the output directory (such as archived sources) count
    toward the cap. The reserve permits an error record after ordinary output
    reaches its allowance; it cannot increase the declared total cap.
    """
    def __init__(self, root, *, maximum_bytes=64*1024**2,
                 maximum_file_bytes=32*1024**2, terminal_reserve_bytes=65536):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError('existing episode output directory required')
        for value in (maximum_bytes, maximum_file_bytes, terminal_reserve_bytes):
            if type(value) is not int or value <= 0:
                raise ValueError('positive integer byte limits required')
        if terminal_reserve_bytes >= maximum_bytes or maximum_file_bytes > maximum_bytes:
            raise ValueError('inconsistent output caps')
        self.maximum_bytes, self.maximum_file_bytes = maximum_bytes, maximum_file_bytes
        self.terminal_reserve_bytes = terminal_reserve_bytes
        self.bytes_written = sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())
        if self.bytes_written > maximum_bytes-terminal_reserve_bytes:
            raise OutputLimitExceededV43('existing files exceed ordinary output allowance')
        self.files = {}

    def _write(self, name, content, *, terminal=False):
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or not relative.parts:
            raise ValueError('relative artifact name required')
        path = self.root/relative
        if not path.resolve().is_relative_to(self.root):
            raise ValueError('artifact path escapes output directory')
        ceiling = self.maximum_bytes if terminal else self.maximum_bytes-self.terminal_reserve_bytes
        if len(content) > self.maximum_file_bytes or self.bytes_written+len(content) > ceiling:
            raise OutputLimitExceededV43('declared artifact byte allowance exceeded')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        self.bytes_written += len(content)
        self.files[str(relative)] = dict(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
        return deepcopy(self.files[str(relative)])

    def json(self, name, value, *, terminal=False):
        return self._write(name, canonical_bytes(value), terminal=terminal)

    def arrays(self, name, **arrays):
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        return self._write(name, buffer.getvalue())


class DevelopmentStartLedgerV43:
    """Durable bounded reservations; failed attempts are retained, never reset."""
    def __init__(self, path, *, maximum_slots=5):
        if type(maximum_slots) is not int or not 1 <= maximum_slots <= 5:
            raise ValueError('V43 development batch reserves at most five start slots')
        self.path = Path(path).resolve()
        self.maximum_slots = maximum_slots

    def _mutate(self, callback):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(self.path.suffix+'.lock').open('a+b') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            data = json.loads(self.path.read_text()) if self.path.exists() else dict(
                schema='v43.development_start_ledger.v1', maximum_slots=self.maximum_slots, entries=[])
            if data.get('schema') != 'v43.development_start_ledger.v1' or data.get('maximum_slots') != self.maximum_slots:
                raise ValueError('start ledger contract mismatch')
            returned = callback(data)
            temporary = self.path.with_suffix(self.path.suffix+'.new')
            with temporary.open('xb') as stream:
                stream.write(canonical_bytes(data)); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            return returned

    def reserve(self, run_id, metadata):
        if not isinstance(run_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', run_id):
            raise ValueError('bounded opaque run identifier required')
        canonical_bytes(metadata)
        def reserve(data):
            if any(row['run_id'] == run_id for row in data['entries']):
                raise ValueError('run already reserved; no silent restart')
            if len(data['entries']) >= self.maximum_slots:
                raise ValueError('development start slot cap exhausted')
            row = dict(run_id=run_id, status='reserved_before_factory', metadata=deepcopy(metadata),
                       world_created=False, reservation_time_unix_s=time.time())
            data['entries'].append(row)
            return deepcopy(row)
        return self._mutate(reserve)

    def finish(self, run_id, *, status, world_created, result_sha256):
        if type(world_created) is not bool or not isinstance(status, str) or not status:
            raise ValueError('explicit status and world creation flag required')
        if not isinstance(result_sha256, str) or not re.fullmatch(r'[0-9a-f]{64}', result_sha256):
            raise ValueError('saved terminal-result SHA256 required')
        def finish(data):
            rows = [row for row in data['entries'] if row['run_id'] == run_id]
            if len(rows) != 1 or rows[0]['status'] != 'reserved_before_factory':
                raise ValueError('only an unfinished reservation can be finalized')
            rows[0].update(status=status, world_created=world_created, result_sha256=result_sha256,
                           finished_time_unix_s=time.time())
            return deepcopy(rows[0])
        return self._mutate(finish)


def validate_step_v43(step, *, expected_step, expected_action, expected_previous_pose=None):
    """Bind execution accounting to the exact packet before fusion or planning."""
    if type(step) is not SensorStepV41 or type(step.rgbd) is not PaidRGBDObservationV40 or type(step.scan) is not PlanarScan:
        raise TypeError('strict sensor step with RGB-D and planar scan required')
    receipt = step.receipt
    if not isinstance(receipt, dict):
        raise ValueError('sensor execution receipt required')
    allowed = {'action', 'paid_step', 'action_cost', 'initial_frames', 'collision',
               'pose_before_xyyaw_rad', 'pose_xyyaw_rad', 'returned_xy_and_yaw'}
    if set(receipt)-allowed:
        raise ValueError('execution receipt contains undeclared fields')
    if (type(receipt.get('paid_step')) is not int or type(receipt.get('action_cost')) is not int
            or step.rgbd.paid_step != expected_step or receipt.get('paid_step') != expected_step
            or receipt.get('action') != expected_action
            or receipt.get('action_cost') != (0 if expected_step == 0 else 1)
            or type(receipt.get('collision')) is not bool):
        raise ValueError('one actual primitive must match one consecutive paid packet')
    if expected_step == 0 and (receipt.get('initial_frames') != 1 or receipt['collision']):
        raise ValueError('exactly one collision-free initial sensor grant required')
    if receipt['collision'] and expected_action != 'forward':
        raise ValueError('only forward attempts can report a collision')
    pose = np.asarray(receipt.get('pose_xyyaw_rad'), dtype=float)
    if pose.shape != (3,) or not np.isfinite(pose).all():
        raise ValueError('finite execution XY/yaw required')
    from env.development_sensor_v41 import camera_transform_xyyaw, propose_motion
    expected_transform = camera_transform_xyyaw(pose, .9)
    if not np.allclose(expected_transform, step.rgbd.world_from_camera, atol=1e-7, rtol=0):
        raise ValueError('execution pose and measured camera pose disagree')
    if expected_step:
        previous = np.asarray(expected_previous_pose, dtype=float)
        before = np.asarray(receipt.get('pose_before_xyyaw_rad'), dtype=float)
        if (previous.shape != (3,) or before.shape != (3,) or not np.isfinite([previous,before]).all()
                or not np.allclose(previous, before, atol=1e-7, rtol=0)):
            raise ValueError('motion receipt must begin at the last verified pose')
        proposed = previous if receipt['collision'] else propose_motion(previous, expected_action)
        if (not np.allclose(proposed[:2], pose[:2], atol=1e-7, rtol=0)
                or abs(math.atan2(math.sin(proposed[2]-pose[2]), math.cos(proposed[2]-pose[2]))) > 1e-7):
            raise ValueError('motion does not match the paid .25 m / 30 degree primitive')
    # V41 simulation explicitly timestamps scan by paid step. This adapter
    # validates that source convention; the general V42 mapper does not assume it.
    if float(step.scan.timestamp_s) != float(expected_step):
        raise ValueError('V41 simulation scan timestamp must identify this paid step')
    laser = np.asarray(step.scan.world_from_laser, dtype=float)
    if (laser.shape != (4, 4) or not np.allclose(laser[:2, 3], pose[:2], atol=1e-7, rtol=0)
            or abs(laser[2,3]-.3) > 1e-7):
        raise ValueError('scan origin and execution position disagree')
    yaw = math.atan2(laser[1, 0], laser[0, 0])
    if abs(math.atan2(math.sin(yaw-pose[2]), math.cos(yaw-pose[2]))) > 1e-7:
        raise ValueError('scan orientation and execution pose disagree')
    return dict(frame_id=step.rgbd.frame_id, paid_step=expected_step,
                observation_sha256=step.rgbd.sha256(), action=expected_action,
                action_cost=receipt['action_cost'], collision=receipt['collision'])


def save_sensor_step_v43(writer, step):
    prefix = f'packets/{step.rgbd.paid_step:03d}'
    packet = writer.arrays(prefix+'_rgbd.npz', **{
        key: getattr(step.rgbd, key) for key in PaidRGBDObservationV40.__dataclass_fields__})
    scan = writer.arrays(prefix+'_scan.npz', **{
        key: getattr(step.scan, key) for key in PlanarScan.__dataclass_fields__})
    writer.json(prefix+'_receipt.json', dict(execution=step.receipt,
        observation_sha256=step.rgbd.sha256(), rgbd_artifact=packet, scan_artifact=scan))


def execute_episode_v43(sensor, controller, mapper, writer, *, budget, maximum_elapsed_s=600.):
    """Acquire once, save, fuse once, update all evidence, then select one action.

    Resource/start authorization is the development factory's responsibility.
    This function creates no World. Tests pass finite saved-packet doubles, not
    an alternate synthetic environment bypassing the physical resource gate.
    """
    if type(budget) is not int or not 1 <= budget <= 160 or not math.isfinite(maximum_elapsed_s) or maximum_elapsed_s <= 0:
        raise ValueError('task action budget 1..160 and positive wall-time limit required')
    began = time.monotonic()
    records, actions, collisions, received = 0, [], 0, 0
    submitted_actions = []
    stage, status, error = 'initial_acquisition', None, None
    last_decision = None
    timings = dict(acquisition_s=0., persistence_s=0., mapping_s=0., evidence_s=0., planning_s=0., final_mesh_s=0.)
    artifacts = dict(mesh_saved=False, occupancy_saved=False)
    pending = None
    previous_pose = None
    try:
        for index in range(budget+1):
            if time.monotonic()-began > maximum_elapsed_s:
                status = 'wall_time_limit'; break
            stage = 'acquisition'
            before = time.monotonic()
            sensor_action = 'initial_observation' if index == 0 else ACTION_TO_SENSOR_V43[pending]
            if index:
                submitted_actions.append(dict(expected_paid_step=index, controller_action=pending,
                                               sensor_action=sensor_action))
            step = sensor.initial_observation() if index == 0 else sensor.step(sensor_action)
            received += 1
            timings['acquisition_s'] += time.monotonic()-before
            stage = 'packet_accounting'
            before = time.monotonic()
            save_sensor_step_v43(writer, step)
            timings['persistence_s'] += time.monotonic()-before
            records += 1
            accounting = validate_step_v43(step, expected_step=index, expected_action=sensor_action,
                                          expected_previous_pose=previous_pose)
            previous_pose = list(step.receipt['pose_xyyaw_rad'])
            collisions += int(accounting['collision'])
            if index:
                actions.append(dict(paid_step=index, controller_action=pending, sensor_action=sensor_action,
                                    observation_sha256=step.rgbd.sha256(), collision=accounting['collision']))
            stage = 'mapping'
            before = time.monotonic(); map_receipt = mapper.update(step.rgbd, step.scan)
            timings['mapping_s'] += time.monotonic()-before
            stage = 'current_evidence'
            before = time.monotonic()
            evidence = controller.accept(step.rgbd, mapper,
                execution_outcome='collision' if accounting['collision'] else 'success')
            timings['evidence_s'] += time.monotonic()-before
            stage = 'planning'
            before = time.monotonic(); decision = controller.choose()
            timings['planning_s'] += time.monotonic()-before
            action = decision.get('action')
            if action not in (*ACTION_TO_SENSOR_V43, 'stop', 'blocked'):
                raise ValueError('controller emitted an undeclared action')
            last_decision = decision
            stage = 'step_record'
            writer.json(f'steps/{index:03d}.json', dict(accounting=accounting,
                mapper=map_receipt, controller_evidence=evidence, decision=decision))
            if action in ('stop', 'blocked'):
                status = 'controller_stop' if action == 'stop' else 'controller_blocked'; break
            if index >= budget:
                status = 'budget_exhausted'; break
            pending = action
        if status is None:
            status = 'budget_exhausted'
    except Exception as exc:
        status = 'artifact_limit' if isinstance(exc, OutputLimitExceededV43) else 'episode_error'
        error = dict(stage=stage, type=type(exc).__name__, message=str(exc))
    # Saving predictions never integrates a frame again and never opens GT.
    finalization_errors = []
    for name, callback in (
        ('mesh', lambda: writer.arrays('prediction/mesh.npz', **mapper.mesh_arrays())),
        ('occupancy', lambda: writer.arrays('prediction/occupancy.npz',
                                          belief=mapper.occupancy_arrays()[0], observed=mapper.occupancy_arrays()[1])),
        ('mapper_state', lambda: writer.json('prediction/mapper.json', mapper.snapshot())),
    ):
        before = time.monotonic()
        try:
            callback()
            if name in ('mesh', 'occupancy'):
                artifacts[name+'_saved'] = True
        except Exception as exc:
            finalization_errors.append(dict(artifact=name, type=type(exc).__name__, message=str(exc)))
        timings['final_mesh_s'] += time.monotonic()-before
    try:
        sensor_status = sensor.close()
    except Exception as exc:
        sensor_status = dict(close_error=type(exc).__name__+': '+str(exc))
    if finalization_errors and error is None:
        status = 'prediction_save_failed'
    if status == 'controller_stop' and sensor_status.get('returned_xy_and_yaw') is not True:
        status = 'stopped_without_confirmed_return'
    try:
        final_mapper = mapper.snapshot()
    except Exception as exc:
        final_mapper = dict(snapshot_error=type(exc).__name__+': '+str(exc))
    result = dict(schema='v43.episode_result.v1', status=status, error=error,
        finalization_errors=finalization_errors, acquired_and_saved_packets=records,
        received_sensor_packets=received, submitted_paid_actions=len(submitted_actions),
        submitted_action_records=submitted_actions,
        executed_paid_actions=len(actions), collisions=collisions, actions=actions,
        last_decision=last_decision, sensor_status=sensor_status, artifacts=artifacts,
        elapsed_s=time.monotonic()-began, timings=timings,
        mapper_frames=final_mapper.get('frames'),
        mapper_tsdf_integrations=final_mapper.get('tsdf_integration_count'),
        automatic_retry=False, actual_performance_evaluated=False,
        quality_and_coverage_scores=None, prediction_requires_separate_offline_evaluation=True,
        no_future_sensor_query_from_controller=True, source='live supplied sensor or explicit finite test double')
    # Terminal record excludes potentially large per-step controller details.
    terminal = dict(result)
    if last_decision is not None:
        terminal['last_decision'] = {key: last_decision[key] for key in
            ('action', 'reason', 'target', 'paid_step') if key in last_decision}
    writer.json('result.json', terminal, terminal=True)
    return terminal
