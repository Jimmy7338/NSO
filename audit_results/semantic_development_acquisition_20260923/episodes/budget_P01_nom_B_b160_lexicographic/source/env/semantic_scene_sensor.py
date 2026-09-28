"""Pinned new-layout assets with the existing paid RGB-D/laser physics.

Only construction differs from the frozen V41 adapter. The inherited capture,
action accounting, collision checks and output packet types remain identical.
Private geometry stays in this sensor; controllers receive paid packets only.
"""
from copy import deepcopy
import math
import re

from env import development_sensor_v41 as physical
from nso.scene_contract_v40 import validate_public_spec


def validate_sensor_contract(public_spec):
    validate_public_spec(public_spec)
    sensor, motion = public_spec['sensor'], public_spec['motion']
    if (motion != dict(translation_step_m=.25, rotation_step_deg=30., camera_height_m=.9)
            or sensor['width'] != 96 or sensor['height'] != 72
            or sensor['intrinsic'] != [[48., 0., 47.5], [0., 48., 35.5], [0., 0., 1.]]
            or sensor['depth_min_m'] != .1 or sensor['depth_max_m'] != 4.
            or sensor['lidar_max_range_m'] != 8.
            or sensor['depth_noise_relative_std'] != .01
            or public_spec['pose_model']['kind'] != 'exact'):
        raise ValueError('fixed shared .25m/30deg, 96x72, 1% depth, exact-pose contract required')


def marker_patches_from_asset(asset):
    """The pinned loader already unwraps the marker JSON into a list."""
    patches = asset['markers']
    if not isinstance(patches, list) or not all(isinstance(patch, dict) for patch in patches):
        raise ValueError('private asset loader must return a marker-patch list')
    return deepcopy(patches)


class SemanticSceneSensor(physical.DevelopmentSensorV41):
    """New scene IDs are explicit; none are aliases of old DEV assets."""

    def __init__(self, asset_dir, public_spec, *, expected_manifest_sha256,
                 episode_id, noise_seed, persistent_output_root,
                 expected_batch_peak_bytes, robot_radius_m=.2,
                 laser_height_m=.3, scan_rays=360):
        report = physical.storage_report_v41(persistent_output_root, expected_batch_peak_bytes)
        if not report['passed']:
            physical._COUNTERS['blocked_before_world_creation'] += 1
            raise physical.ResourceGateBlocked(report)
        if not isinstance(expected_manifest_sha256, str) or not re.fullmatch('[0-9a-f]{64}', expected_manifest_sha256):
            raise ValueError('external asset manifest SHA256 is required')
        validate_sensor_contract(public_spec)
        if not isinstance(episode_id, str) or not episode_id or len(episode_id) > 100:
            raise ValueError('nonempty bounded opaque episode ID required')
        physical._integer(noise_seed, 'depth noise seed', zero=True)
        if robot_radius_m != .2 or laser_height_m != .3 or type(scan_rays) is not int or scan_rays != 360:
            raise ValueError('shared 0.2m radius, 0.3m laser height and 360 scan rays required')

        # Lazy loading permits resource/pin checks before any private asset I/O.
        from nso.semantic_scene_assets import load_semantic_scene_asset
        asset = load_semantic_scene_asset(asset_dir, expected_manifest_sha256=expected_manifest_sha256)
        stored = asset['public_spec']
        for key in ('schema_version', 'sensor', 'motion', 'pose_model', 'structure_prior'):
            if public_spec[key] != stored[key]:
                raise ValueError('runtime public specification differs from pinned asset: ' + key)
        # Navigation materialization and a smaller declared motion budget are
        # runtime public changes; appearance/physics/prior must remain pinned.
        declared_task, stored_task = deepcopy(public_spec['task']), deepcopy(stored['task'])
        if declared_task.pop('max_actions') > stored_task.pop('max_actions'):
            raise ValueError('runtime budget exceeds pinned task budget')
        declared_task.pop('budget_tier'); stored_task.pop('budget_tier')
        if declared_task != stored_task:
            raise ValueError('runtime task changes paid-action or return contract')
        metadata, arrays = asset['metadata'], asset['arrays']
        workspace = asset['public_workspace']
        if metadata['public_workspace'] != workspace:
            raise ValueError('private and public starting workspace disagree')
        self._markers = marker_patches_from_asset(asset)
        self._vertices, self._triangles = arrays['vertices'], arrays['triangles']
        self._owners, self._rgb = arrays['triangle_instance_id'], arrays['triangle_rgb']
        self._footprints = [x['conservative_footprint_xy_m'] for x in metadata['private_instances']]
        for box in metadata['background_boxes']:
            if box[5] > 0:
                self._footprints.append([[box[0], box[2]], [box[1], box[2]],
                                         [box[1], box[3]], [box[0], box[3]]])
        start = workspace['start_position_world_m']
        self._pose = physical._pose([start[0], start[1], math.radians(workspace['start_yaw_deg'])])
        self._start = self._pose.copy()
        self._radius, self._laser_height, self._scan_rays = .2, .3, 360
        if physical.swept_circle_collision(self._pose[:2], self._pose[:2], self._footprints, self._radius):
            raise ValueError('declared start collides with a conservative footprint')
        self._spec = deepcopy(public_spec)
        self._episode, self._noise_seed = episode_id, noise_seed
        self._steps, self._initial_frames, self._collisions = 0, 0, 0
        self._closed = False
        self.resource_report = report
        self.asset_manifest_sha256 = expected_manifest_sha256
        physical._COUNTERS['worlds_created'] += 1


def create_semantic_scene_sensor(asset_dir, public_spec, **kwargs):
    return SemanticSceneSensor(asset_dir, public_spec, **kwargs)
