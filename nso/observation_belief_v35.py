"""Observation-only two-template belief for the V35 controlled experiment.

Both templates are public task priors, not a forecast of the actual world.
Only paid depth, scan and artificial RGB marker observations update belief.
This is a bounded, robust pseudo-likelihood, not calibrated uncertainty.
No renderer, world, reward table, episode identity or evaluation API is used.
"""
from pathlib import Path
import hashlib
import math

import numpy as np


MODES_V35 = ('G', 'S', 'swapped', 'swapped_no_feedback')
CONFIG_V35 = {
    'marker_colors': {'2': [40, 100, 220], '3': [220, 60, 40]},
    'marker_quorum_pixels': 16,
    'marker_distinct_xy': 2,
    'semantic_reliability': .9,
    'stereo_fb_px_m': 480. * .12,
    'stereo_sigma_disparity_px': .25,
    'scan_sigma_m': .02,
    'template_difference_m': 1e-5,
    'minimum_depth_samples': 8,
    'minimum_scan_samples': 3,
    'loss_cap': 12.5,
    'pose_log_odds_cap': 6.,
    'total_geometry_log_odds_cap': 24.,
    'forecast_reliability': .99,
    'calibrated_probability': False,
    'geometry_repetition_policy': 'first observation at each exact pose only',
    'semantic_repetition_policy': 'one global prior after two distinct XY; conflicting supported classes clear it',
    'zero_depth_policy': 'ignore absent measured depth; do not infer a surface from a missing return',
}


def _pose(value):
    values = tuple(value)
    if len(values) != 3 or any(isinstance(x, (bool, np.bool_)) or
                             not isinstance(x, (int, np.integer)) for x in values):
        raise ValueError('integer public (x,y,heading) pose required')
    if not 0 <= values[2] < 4:
        raise ValueError('heading must be in range(4)')
    return tuple(int(x) for x in values)


def _numeric(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in 'fiu' or not np.isfinite(array).all():
        raise ValueError(f'finite {name} with shape {shape} required')
    if (array < 0).any():
        raise ValueError(f'nonnegative {name} required')
    return array.astype(np.float64, copy=False)


def _array_digest(array):
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    for part in (value.dtype.str.encode(), str(value.shape).encode(), value.tobytes()):
        digest.update(len(part).to_bytes(8, 'big')); digest.update(part)
    return digest.hexdigest()


def _geometry_score(depth, ranges, template_depth, template_ranges):
    """Positive log odds favour template 0; symmetric scales avoid GT input."""
    cfg = CONFIG_V35
    depth = _numeric(depth, template_depth.shape[1:], 'paid depth')
    ranges = _numeric(ranges, template_ranges.shape[1:], 'paid ranges')
    separated = np.abs(template_depth[0] - template_depth[1]) > cfg['template_difference_m']
    usable = separated & (depth > 0)
    count = int(usable.sum())
    modalities = []
    diagnostics = {'depth_samples': count, 'scan_samples': 0,
                   'depth_mean_losses': None, 'scan_mean_losses': None}
    if count >= cfg['minimum_depth_samples']:
        observed_disparity = cfg['stereo_fb_px_m'] / depth[usable]
        losses = []
        for hypothesis in (0, 1):
            predicted = template_depth[hypothesis][usable]
            loss = np.full(count, cfg['loss_cap'], np.float64)
            valid = predicted > 0
            residual = (observed_disparity[valid] - cfg['stereo_fb_px_m'] / predicted[valid]) / cfg['stereo_sigma_disparity_px']
            loss[valid] = np.minimum(.5 * residual ** 2, cfg['loss_cap'])
            losses.append(float(loss.mean()))
        modalities.append(losses[1] - losses[0])
        diagnostics['depth_mean_losses'] = losses
    separated = np.abs(template_ranges[0] - template_ranges[1]) > cfg['template_difference_m']
    # Zero ranges are invalid/missing in the declared scan contract.
    usable = separated & (ranges > 0)
    count = int(usable.sum()); diagnostics['scan_samples'] = count
    if count >= cfg['minimum_scan_samples']:
        losses = []
        for hypothesis in (0, 1):
            residual = (ranges[usable] - template_ranges[hypothesis][usable]) / cfg['scan_sigma_m']
            losses.append(float(np.minimum(.5 * residual ** 2, cfg['loss_cap']).mean()))
        modalities.append(losses[1] - losses[0])
        diagnostics['scan_mean_losses'] = losses
    score = float(np.clip(np.mean(modalities), -cfg['pose_log_odds_cap'], cfg['pose_log_odds_cap'])) if modalities else 0.
    diagnostics['log_likelihood_ratio'] = score
    diagnostics['usable_modalities'] = len(modalities)
    return score, diagnostics


class PublicTemplatesV35:
    """The same paired public sensor predictions are available to every mode."""
    def __init__(self, poses, depth, ranges, *, provenance=None):
        self.poses = tuple(_pose(value) for value in poses)
        if not self.poses or len(set(self.poses)) != len(self.poses):
            raise ValueError('nonempty unique public pose list required')
        self.node_for_pose = {value: index for index, value in enumerate(self.poses)}
        depth = np.asarray(depth); ranges = np.asarray(ranges)
        if depth.ndim != 4 or ranges.ndim != 3 or depth.shape[:2] != (2, len(self.poses)) or ranges.shape[:2] != (2, len(self.poses)):
            raise ValueError('two complete public sensor template arrays required')
        self.depth = np.array(_numeric(depth, depth.shape, 'template depth'), copy=True)
        self.ranges = np.array(_numeric(ranges, ranges.shape, 'template ranges'), copy=True)
        self.depth.flags.writeable = False; self.ranges.flags.writeable = False
        self.provenance = dict(provenance or {})
        self._forecast = {}

    @classmethod
    def from_saved(cls, directory, parent_id, poses):
        """Read only paired raw NPZ arrays; caller provides public graph poses."""
        if parent_id not in ('P00', 'P01'):
            raise ValueError('declared public parent required')
        paths = [Path(directory) / f'{parent_id}_h{h}_pixels.npz' for h in (0, 1)]
        fields = ('depth', 'ranges', 'cell', 'heading', 'intrinsic', 'camera_pose', 'laser_pose', 'scan_calibration')
        arrays = []
        for path in paths:
            with np.load(path, allow_pickle=False) as payload:
                arrays.append({name: np.array(payload[name], copy=True) for name in fields})
        for field in fields[2:]:
            if not np.array_equal(arrays[0][field], arrays[1][field]):
                raise ValueError(f'public template calibration/pose differs: {field}')
        poses = tuple(_pose(value) for value in poses)
        if len(poses) != len(arrays[0]['heading']) or not np.array_equal(np.asarray(poses)[:, 2], arrays[0]['heading']):
            raise ValueError('public poses do not match template node order')
        # All V34 legal positions are exact original metre-grid locations;
        # the camera origin differs from them by one constant public XY shift.
        shifts = arrays[0]['camera_pose'][:, :2, 3] - np.asarray(poses)[:, :2]
        if not np.allclose(shifts, shifts[0], rtol=0., atol=1e-9):
            raise ValueError('public pose XY order does not match template camera positions')
        provenance = {'parent_id': parent_id, 'files': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
                      'loaded_fields': list(fields), 'template_rgb_used': False,
                      'task_prior': 'both possible public templates; actual hypothesis unavailable'}
        return cls(poses, np.stack([a['depth'] for a in arrays]),
                   np.stack([a['ranges'] for a in arrays]), provenance=provenance)

    def geometry_evidence(self, pose, depth, ranges):
        node = self.node_for_pose[_pose(pose)]
        return _geometry_score(depth, ranges, self.depth[:, node], self.ranges[:, node])

    def template_information(self, node):
        """Conservative public forecast channel, never an actual observation."""
        if isinstance(node, (bool, np.bool_)) or not isinstance(node, (int, np.integer)) or not 0 <= node < len(self.poses):
            raise ValueError('public node index required')
        node = int(node)
        if node not in self._forecast:
            scores = [_geometry_score(self.depth[h, node], self.ranges[h, node], self.depth[:, node], self.ranges[:, node])[0] for h in (0, 1)]
            threshold = math.log(CONFIG_V35['forecast_reliability'] / (1 - CONFIG_V35['forecast_reliability']))
            reliable = scores[0] >= threshold and scores[1] <= -threshold
            self._forecast[node] = {'node': node, 'pose': list(self.poses[node]), 'informative': reliable,
                                    'correct_probability': CONFIG_V35['forecast_reliability'] if reliable else .5,
                                    'public_clean_template_log_odds': scores,
                                    'minimum_signed_log_odds': threshold,
                                    'scope': 'fixed assumed soft forecast channel, not calibrated real-sensor probability'}
        return dict(self._forecast[node])


def _rgb_class(rgb):
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError('paid RGB must be uint8 HxWx3')
    counts = {int(code): int(np.count_nonzero(np.all(rgb == np.asarray(color, np.uint8), axis=-1)))
              for code, color in CONFIG_V35['marker_colors'].items()}
    present = [code for code, count in counts.items() if count > 0]
    supported = [code for code, count in counts.items() if count >= CONFIG_V35['marker_quorum_pixels']]
    chosen = supported[0] if len(supported) == 1 and len(present) == 1 else None
    return chosen, counts


class ObservationBeliefV35:
    """Mutable belief with no route, geometry truth, labels or episode ID input."""
    def __init__(self, templates, mode='G'):
        if not isinstance(templates, PublicTemplatesV35) or mode not in MODES_V35:
            raise ValueError('public templates and declared intervention mode required')
        self.templates = templates; self.mode = mode
        self.geometry_log_odds = 0.; self.semantic_log_odds = 0.
        self._geometry_poses = set(); self._class_positions = {2: set(), 3: set()}
        self._last_step = -1; self.updates = 0

    @property
    def probabilities(self):
        log_odds = self.geometry_log_odds + self.semantic_log_odds
        p0 = 1. / (1. + math.exp(-log_odds))
        return (p0, 1. - p0)

    def update(self, *, pose, depth, rgb, ranges, step):
        pose = _pose(pose)
        if isinstance(step, (bool, np.bool_)) or not isinstance(step, (int, np.integer)) or step < self._last_step:
            raise ValueError('nondecreasing nonnegative paid step required')
        if step < 0:
            raise ValueError('nonnegative paid step required')
        score, residuals = self.templates.geometry_evidence(pose, depth, ranges)
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8 or rgb.shape != np.asarray(depth).shape + (3,):
            raise ValueError('aligned paid RGB required')
        new_pose = pose not in self._geometry_poses
        feedback = self.mode != 'swapped_no_feedback'
        if new_pose:
            self._geometry_poses.add(pose)
            if feedback:
                cap = CONFIG_V35['total_geometry_log_odds_cap']
                self.geometry_log_odds = float(np.clip(self.geometry_log_odds + score, -cap, cap))
        chosen = None; counts = None
        if self.mode != 'G':
            chosen, counts = _rgb_class(rgb)
            simultaneous_conflict = all(count >= CONFIG_V35['marker_quorum_pixels'] for count in counts.values())
            if simultaneous_conflict:
                # Two independently supported labels in one frame are a
                # contradiction, not an unknown reading that preserves a
                # previously strong prior. Swapping both labels is identical.
                for positions in self._class_positions.values():
                    positions.add(pose[:2])
            elif chosen is not None:
                if self.mode in ('swapped', 'swapped_no_feedback'):
                    chosen = 5 - chosen
                self._class_positions[chosen].add(pose[:2])
            present = [code for code, positions in self._class_positions.items() if positions]
            qualified = [code for code in present if len(self._class_positions[code]) >= CONFIG_V35['marker_distinct_xy']]
            self.semantic_log_odds = 0.
            if len(present) == 1 and len(qualified) == 1:
                reliability = CONFIG_V35['semantic_reliability']
                self.semantic_log_odds = (1. if qualified[0] == 2 else -1.) * math.log(reliability / (1. - reliability))
        self._last_step = int(step); self.updates += 1
        return {'step': int(step), 'pose': list(pose), 'mode': self.mode,
                'probabilities': list(self.probabilities), 'geometry_log_odds': self.geometry_log_odds,
                'semantic_log_odds': self.semantic_log_odds,
                'new_geometry_pose': new_pose, 'geometry_feedback_enabled': feedback,
                'geometry_applied_log_odds': score if new_pose and feedback else 0.,
                'geometry_unique_poses': len(self._geometry_poses), 'residuals': residuals,
                'observed_class_after_intervention': chosen, 'rgb_pixel_counts': counts,
                'class_distinct_xy': {str(code): len(positions) for code, positions in self._class_positions.items()},
                'class_conflict': all(self._class_positions.values()),
                'observation_sha256': {'depth': _array_digest(depth), 'rgb': _array_digest(rgb), 'ranges': _array_digest(ranges)},
                'scope': 'paid observations and public two-template task prior only'}
