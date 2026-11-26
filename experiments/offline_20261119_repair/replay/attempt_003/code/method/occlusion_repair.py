"""Isolated B2 repair: measured-free template intersections do not occlude.

Nothing is installed at import. This module does not change measurements,
template construction, weights, candidates, fit, reward history or budget.
"""
from pathlib import Path
import hashlib
import inspect
import numpy as np
import open3d as o3d

SOURCE = Path('/home/ouc/nso_robot/analyses/reproduction_20261104/simulation_publication_stage/experiments/unknown_exploration_20261104/source')
B_MODEL = Path('/home/ouc/nso_robot/analyses/dual_template_exploration_20261119/method/model.py')
SOURCE_SHA = 'abf559a87049e5fe350713e31ca4cd8746af92c4ba2ffddff9663f805a473310'
B_MODEL_SHA = 'ec98759a9f6d45d57bcaa3dc08ee0aef1e954938ab99eea8926740d4f526ee99'
DEPTH_TOLERANCE_M = .10  # The unchanged source history tolerance.
_installed = {}


def _aggregate(summary, audit):
    summary['gain_calls'] = summary.get('gain_calls', 0) + 1
    summary['repair_enabled_calls'] = summary.get('repair_enabled_calls', 0) + int(audit['repair_enabled'])
    for name in ('original_self_occluded_rays', 'original_self_occluded_free_first_hits',
                 'candidate_intersections', 'strict_free_intersections', 'measured_surface_intersections',
                 'unresolved_intersections', 'history_conflicts_vetoed', 'recovered_points',
                 'recovered_expected_area_m2'):
        summary[name] = summary.get(name, 0) + audit[name]
    stages = summary.setdefault('stage_sums_across_gain_calls', {})
    for name, row in audit['stages'].items():
        old = stages.setdefault(name, dict(points=0, expected_area_m2=0.))
        for key, value in row.items(): old[key] += value
    summary['units_note'] = 'sums over gain function calls, not unique surfaces or independent observations'


def classify_history(points, keyframes):
    """Classify intersections from actual depth only; measured support vetoes free.

    Exact source nearest-pixel projection, positive-depth and .10m tolerance.
    No category, mesh reference, full object identity or GT is consumed.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    free = np.zeros(len(points), bool)
    measured_surface = np.zeros(len(points), bool)
    supported = np.zeros(len(points), bool)
    for frame in keyframes:
        local = (points - frame.world_from_camera[:3, 3]) @ frame.world_from_camera[:3, :3]
        z = local[:, 2]
        u = np.rint(local[:, 0] / np.maximum(z, .01) * frame.intrinsic[0, 0] + frame.intrinsic[0, 2]).astype(int)
        v = np.rint(local[:, 1] / np.maximum(z, .01) * frame.intrinsic[1, 1] + frame.intrinsic[1, 2]).astype(int)
        valid = (z > .15) & (u >= 0) & (u < frame.depth_m.shape[1]) & (v >= 0) & (v < frame.depth_m.shape[0])
        ids = np.flatnonzero(valid)
        measured = frame.depth_m[v[ids], u[ids]]
        valid_depth = np.isfinite(measured) & (measured > 0)
        supported[ids] |= valid_depth
        free[ids] |= valid_depth & (z[ids] < measured - DEPTH_TOLERANCE_M)
        # Use the same lower bound as strict_free to avoid a floating-point
        # gap at exactly measured - tolerance; near-surface support is closed.
        measured_surface[ids] |= valid_depth & (z[ids] >= measured - DEPTH_TOLERANCE_M) & (z[ids] <= measured + DEPTH_TOLERANCE_M)
    return dict(strict_free=free & ~measured_surface, raw_free_evidence=free,
                measured_surface=measured_surface, projected_valid_depth=supported)


def _measure(mask, model):
    return dict(points=int(mask.sum()), expected_area_m2=float(model.weights[mask].sum()))


def trace_gain(model, cell, heading, measured_ray, repair_enabled=False):
    """Return gain and gate counts without mutating model, ray, mapper or cache."""
    from env.virtual3d import camera_pose
    count = len(model.points)
    receipt = dict(schema='template_occlusion_repair_trace_v1', cell=list(map(int, cell)), heading=int(heading),
                   repair_enabled=bool(repair_enabled), history_keyframes=len(model.mapper.keyframes),
                   history_reward_filter_enabled=getattr(model, 'audit', {}).get('history_rejection'),
                   tolerance_m=DEPTH_TOLERANCE_M, measured_surface_veto=True,
                   stages={}, original_self_occluded_rays=0, original_self_occluded_free_first_hits=0,
                   candidate_intersections=0, strict_free_intersections=0, measured_surface_intersections=0,
                   unresolved_intersections=0, history_conflicts_vetoed=0, recovered_points=0,
                   recovered_expected_area_m2=0., hypotheses={})
    if not count:
        for name in ('all', 'range', 'fov', 'measured_TSDF', 'original_self_occlusion', 'final'):
            receipt['stages'][name] = dict(points=0, expected_area_m2=0.)
        return np.empty(0), receipt
    config = model.mapper.config
    pose = camera_pose(cell, heading, config, model.mapper.shape[0])
    local = (model.points - pose[:3, 3]) @ pose[:3, :3]
    z = local[:, 2]
    tan = np.tan(np.deg2rad(config.fov_deg / 2))
    ranged = (z > .2) & (z < config.max_depth_m)
    visible = ranged & (np.abs(local[:, 0]) < z * tan) & (np.abs(local[:, 1]) < z * tan * config.height_px / config.width_px)
    receipt['stages']['all'] = _measure(np.ones(count, bool), model)
    receipt['stages']['range'] = _measure(ranged, model)
    receipt['stages']['fov'] = _measure(visible, model)
    ids = np.flatnonzero(visible)
    if measured_ray is not None and len(ids):
        delta = model.points[ids] - pose[:3, 3]
        distance = np.linalg.norm(delta, axis=1)
        rays = np.column_stack([np.tile(pose[:3, 3], (len(ids), 1)), delta]).astype(np.float32)
        hit = measured_ray.cast_rays(o3d.core.Tensor(rays), nthreads=1)['t_hit'].numpy()
        visible[ids] &= hit >= 1 - .06 / np.maximum(distance, .2)
    receipt['stages']['measured_TSDF'] = _measure(visible, model)
    original = visible.copy()
    for hid in np.unique(model.hypothesis_ids[visible]):
        ids = np.flatnonzero(visible & (model.hypothesis_ids == hid))
        delta = model.points[ids] - pose[:3, 3]
        rays = np.column_stack([np.tile(pose[:3, 3], (len(ids), 1)), delta]).astype(np.float32)
        scene = model.hypothesis_rays[int(hid)]
        hit = scene.cast_rays(o3d.core.Tensor(rays), nthreads=1)['t_hit'].numpy()
        passing = hit >= 1 - 1e-4
        original[ids] &= passing
        blocked = np.flatnonzero(~passing)
        receipt['original_self_occluded_rays'] += int(len(blocked))
        per = dict(input_points=len(ids), original_visible_points=int(passing.sum()), recovered_points=0)
        if repair_enabled and len(blocked):
            blocked_rays = rays[blocked]
            first_points = blocked_rays[:, :3] + blocked_rays[:, 3:] * hit[blocked, None]
            first = classify_history(first_points, model.mapper.keyframes)
            receipt['original_self_occluded_free_first_hits'] += int(first['strict_free'].sum())
            intersections = scene.list_intersections(o3d.core.Tensor(blocked_rays), nthreads=1)
            ray_ids = intersections['ray_ids'].numpy().astype(np.int64)
            times = intersections['t_hit'].numpy()
            before = np.isfinite(times) & (times >= 0) & (times < 1 - 1e-4)
            ray_ids, times = ray_ids[before], times[before]
            xyz = blocked_rays[ray_ids, :3] + blocked_rays[ray_ids, 3:] * times[:, None]
            observed = classify_history(xyz, model.mapper.keyframes)
            total = np.bincount(ray_ids, minlength=len(blocked))
            unresolved = np.bincount(ray_ids[~observed['strict_free']], minlength=len(blocked))
            # Never reveal a point solely because a numerical listing is empty.
            # Its original first hit and every listed preceding intersection
            # must be contradicted, with no measured-surface evidence.
            recovered = first['strict_free'] & (total > 0) & (unresolved == 0)
            passing[blocked[recovered]] = True
            per['recovered_points'] = int(recovered.sum())
            receipt['candidate_intersections'] += int(len(times))
            receipt['strict_free_intersections'] += int(observed['strict_free'].sum())
            receipt['measured_surface_intersections'] += int(observed['measured_surface'].sum())
            receipt['unresolved_intersections'] += int((~observed['strict_free']).sum())
            receipt['history_conflicts_vetoed'] += int((observed['raw_free_evidence'] & observed['measured_surface']).sum())
        visible[ids] &= passing
        receipt['hypotheses'][str(int(hid))] = per
    receipt['stages']['original_self_occlusion'] = _measure(original, model)
    receipt['stages']['final'] = _measure(visible, model)
    recovered = visible & ~original
    receipt['recovered_points'] = int(recovered.sum())
    receipt['recovered_expected_area_m2'] = float(model.weights[recovered].sum())
    return visible * model.weights, receipt


def install(model_class, enabled=True, record_calls=False):
    """Explicit process-local binding on B's declared class, never file edits.

    Apply equally to G and S. S_no_history still skips reward history filtering
    but retains this repaired physical-visibility consistency when enabled.
    """
    path = Path(inspect.getfile(model_class)).resolve()
    assert path == B_MODEL and hashlib.sha256(path.read_bytes()).hexdigest() == B_MODEL_SHA
    src = SOURCE / 'nso/semantic_completion_v3.py'
    assert hashlib.sha256(src.read_bytes()).hexdigest() == SOURCE_SHA
    from nso.semantic_completion_v3 import ObjectCompletionModel
    key = id(model_class)
    if key not in _installed:
        assert model_class.gain is ObjectCompletionModel.gain
        state = dict(enabled=bool(enabled), original=model_class.gain, record_calls=bool(record_calls),
                     summary={}, by_variant={})
        def gain(self, cell, heading, ray):
            values, audit = trace_gain(self, cell, heading, ray, repair_enabled=state['enabled'])
            if not hasattr(self, '_occlusion_repair_summary'): self._occlusion_repair_summary = {}
            _aggregate(self._occlusion_repair_summary, audit)
            _aggregate(state['summary'], audit)
            variant = f"semantic={int(self.semantic)},fit={int(self.audit['fit_update'])},reward_history={int(self.audit['history_rejection'])}"
            _aggregate(state['by_variant'].setdefault(variant, {}), audit)
            if state['record_calls']:
                if not hasattr(self, '_occlusion_repair_audit'): self._occlusion_repair_audit = []
                self._occlusion_repair_audit.append(audit)
            return values
        state['wrapper'] = gain
        _installed[key] = state
        model_class.gain = gain
    else:
        state = _installed[key]
        assert model_class.gain is state['wrapper']
        if state['enabled'] != bool(enabled) and state['summary'].get('gain_calls', 0):
            raise RuntimeError('Use a fresh process when changing repair on/off after gain caches have been evaluated')
        state['enabled'] = bool(enabled)
        state['record_calls'] = bool(record_calls)
    return dict(schema='declared_B2_occlusion_repair_binding_v1', enabled=bool(enabled),
                original_source_sha256=SOURCE_SHA, original_B_model_sha256=B_MODEL_SHA,
                repair_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                effect='ignore only measured-free template intersections with measured-surface veto',
                score_weights_candidates_dimensions_reward_history_or_budget_modified=False,
                no_history_reward_ablation_keeps_visibility_repair=bool(enabled))


def audit_log(model):
    return list(getattr(model, '_occlusion_repair_audit', ()))


def diagnostics(model=None):
    """JSON-compatible counters; no per-point arrays and no actor mutation."""
    import copy
    if model is not None:
        return copy.deepcopy(getattr(model, '_occlusion_repair_summary', {}))
    return copy.deepcopy(dict(schema='B2_gain_diagnostics_v1',
        bindings=[dict(enabled=s['enabled'], summary=s['summary'], by_variant=s['by_variant']) for s in _installed.values()]))
