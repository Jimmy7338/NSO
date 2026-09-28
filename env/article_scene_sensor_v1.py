"""New article scene adapter with its own explicit bounded storage profile.

Frozen V41/SEM constructors and their 10 GiB gate are unchanged. This adapter
reuses their paid capture/motion physics for a separately declared small batch.
"""
from copy import deepcopy
import math
from pathlib import Path
import shutil

from env import development_sensor_v41 as physical
from env.semantic_scene_sensor import validate_sensor_contract


GIB = 1024 ** 3


def article_storage_report(output_root, *, expected_episode_bytes=64*1024**2,
                           reserve_bytes=GIB):
    if (type(expected_episode_bytes) is not int or expected_episode_bytes <= 0
            or type(reserve_bytes) is not int or reserve_bytes < GIB):
        raise ValueError('positive episode cap and at least 1 GiB reserve required')
    parent = Path(output_root).resolve()
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    return dict(schema='article.storage_profile.v1', free_bytes=free,
                episode_allowance_bytes=expected_episode_bytes,
                reserve_bytes=reserve_bytes,
                passed=free >= expected_episode_bytes+reserve_bytes,
                old_v41_storage_gate_changed=False)


class ArticleSceneSensorV1(physical.DevelopmentSensorV41):
    def __init__(self, asset_dir, public_spec, *, expected_manifest_sha256,
                 episode_id, noise_seed, persistent_output_root,
                 expected_episode_bytes=64*1024**2, reserve_bytes=GIB):
        report = article_storage_report(persistent_output_root,
            expected_episode_bytes=expected_episode_bytes, reserve_bytes=reserve_bytes)
        if not report['passed']:
            physical._COUNTERS['blocked_before_world_creation'] += 1
            raise physical.ResourceGateBlocked(report)
        validate_sensor_contract(public_spec)
        if not isinstance(episode_id, str) or not episode_id or len(episode_id) > 100:
            raise ValueError('bounded episode identifier required')
        physical._integer(noise_seed, 'depth noise seed', zero=True)
        from nso.article_scene_assets_v1 import load_private_article_scene
        asset = load_private_article_scene(asset_dir,
            expected_manifest_sha256=expected_manifest_sha256)
        stored = asset['public_spec']
        for key in ('schema_version', 'sensor', 'motion', 'pose_model', 'structure_prior'):
            if public_spec[key] != stored[key]:
                raise ValueError('runtime physical specification changed: '+key)
        task, frozen_task = deepcopy(public_spec['task']), deepcopy(stored['task'])
        if task.pop('max_actions') > frozen_task.pop('max_actions'):
            raise ValueError('runtime budget exceeds declared maximum')
        task.pop('budget_tier'); frozen_task.pop('budget_tier')
        if task != frozen_task:
            raise ValueError('paid action/return task contract changed')
        metadata, arrays, workspace = asset['metadata'], asset['arrays'], asset['public_workspace']
        if metadata['public_workspace'] != workspace:
            raise ValueError('public/private workspace mismatch')
        self._markers = deepcopy(asset['markers'])
        self._vertices, self._triangles = arrays['vertices'], arrays['triangles']
        self._owners, self._rgb = arrays['triangle_instance_id'], arrays['triangle_rgb']
        self._footprints = [x['conservative_footprint_xy_m'] for x in metadata['private_instances']]
        self._footprints += [[[b[0],b[2]],[b[1],b[2]],[b[1],b[3]],[b[0],b[3]]]
                             for b in metadata['background_boxes'] if b[5] > 0]
        start = workspace['start_position_world_m']
        self._pose = physical._pose([start[0], start[1], math.radians(workspace['start_yaw_deg'])])
        self._start = self._pose.copy()
        self._radius, self._laser_height, self._scan_rays = .2, .3, 360
        if physical.swept_circle_collision(self._pose[:2], self._pose[:2], self._footprints, self._radius):
            raise ValueError('declared start collides with conservative footprint')
        self._spec = deepcopy(public_spec)
        self._episode, self._noise_seed = episode_id, noise_seed
        self._steps = self._initial_frames = self._collisions = 0
        self._closed = False
        self.resource_report = report
        self.asset_manifest_sha256 = expected_manifest_sha256
        physical._COUNTERS['worlds_created'] += 1
