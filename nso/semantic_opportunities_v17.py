"""V17 observed-mesh corrected route/instance features; no fitted model.

Approach is a distance proxy, not a certificate of hidden surface visibility.
Second-view witnesses use only other endpoints in the shared candidate pool
and known-safe paths, with a reserved return to the fixed task anchor.
"""
from numbers import Integral
from copy import deepcopy

import numpy as np
from scipy.sparse.csgraph import dijkstra

from env.virtual3d import camera_pose
from nso.visibility_corrected_scores_v17 import ObservedMeshAperture
from nso.response_candidates_v7 import _actions
from nso.route_coverage_v2 import orientation_graph
from nso.semantic_gain_v12 import _stable_mean
from nso.semantic_gain_v11 import _asset_confidence
from utils.grid_geometry import inflated_obstacles


KERNEL_NAMES = (
    'endpoint_proximity', 'approach_progress', 'direction_gap_proximity',
    'outbound_aperture', 'affordable_second_view',
    'support_uncertainty_proximity', 'mean_direction_gap_proximity',
    'aperture_history_debt_proximity',
)
COMMON_NAMES = (
    'remaining_budget_fraction', 'outbound_fraction', 'return_fraction',
    'reserved_slack_fraction', 'unknown_allocated_grid_fraction',
    'expected_camera_fraction', 'expected_radar_fraction',
    'camera_yield_posterior', 'radar_yield_posterior', 'log_observed_instances',
) + tuple('mean_' + name for name in KERNEL_NAMES)


def candidate_capacity(count, coverage_slots=4):
    for value, name, minimum in ((count, 'count', 0), (coverage_slots, 'coverage_slots', 1)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(coverage_slots) + 1 + 2 * int(count)


def _unit(value, name):
    value = float(value)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f'{name} outside [0,1]')
    return value


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'invalid {name}')
    return int(value)


def _vector(value, shape, name):
    value = np.asarray(value, dtype=float)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f'invalid {name}')
    return value


def bounded_residual(geometry_score, raw_residual, confidence, max_correction):
    """Zero confidence exactly returns the geometry score, including head bias.

    The correction bound is explicit input to be frozen before training. This
    utility supplies no learned weights and performs no uncertainty calibration.
    """
    if not np.isfinite([geometry_score, raw_residual, max_correction]).all() or max_correction < 0:
        raise ValueError('nonfinite score or invalid correction bound')
    confidence = _unit(confidence, 'confidence')
    return float(geometry_score + confidence * max_correction * np.tanh(raw_residual))


def observed_descriptors(runtime, candidates):
    """Project current module state; planned return observations are excluded."""
    state = runtime.components._cpu_backend.scenes[0]
    mapper, ledger, assets = state['mapper'], state['ledger'], state['assets']
    visibility = ObservedMeshAperture(mapper.mesh(), mapper.config)
    history = ledger.planning_camera_poses()
    attempted = state['gain'].attempted_camera_mask(mapper, history)
    prefix = np.zeros((len(assets), 25))
    for pose in history:
        for ai, asset in enumerate(assets):
            prefix[ai] = np.maximum(prefix[ai], visibility.support(asset, pose))
    records = [dict(asset_index=i, class_vote=a['class_vote'],
                    semantic_confidence=_asset_confidence(a),
                    inverse_sqrt_support_uncertainty_proxy=1. / np.sqrt(max(1, a['support_points'])),
                    direction_unobserved_fractions=[1. - float(np.mean(
                        (np.asarray(a['bits'], dtype=np.int64) & (1 << sector)) != 0))
                        for sector in range(8)]) for i, a in enumerate(assets)]
    result = []
    for c in candidates:
        states = c['states'][1:c['outbound_cost'] + 1]
        radar, camera = state['gain'].route_masks(mapper, states, attempted)
        support = np.zeros_like(prefix)
        for s in states:
            pose = camera_pose(tuple(s[:2]), s[2], mapper.config, mapper.shape[0])
            for ai, asset in enumerate(assets):
                support[ai] = np.maximum(support[ai], visibility.support(asset, pose))
        result.append(dict(candidate_id=c['candidate_id'], remaining_budget=ledger.remaining_budget,
            total_budget=ledger.total_budget, roundtrip_cost=c['cost'], outbound_cost=c['outbound_cost'],
            return_cost=c['return_cost'], observation_horizon='outbound_only_v17_observed_mesh',
            unknown_allocated_grid_fraction=float(np.mean(mapper.belief == -1)),
            expected_new_camera_cells=int(camera.sum()), expected_new_radar_cells=int(radar.sum()),
            camera_yield_posterior=state['gain'].posterior_mean('camera'),
            radar_yield_posterior=state['gain'].posterior_mean('radar'), observed_assets=deepcopy(records),
            observed_aperture_audit=dict(aperture_factors=np.maximum(0., support - prefix).mean(axis=1).tolist(),
                                         prefix_aperture_support=prefix.mean(axis=1).tolist())))
    return assets, result


def route_instance_features(mapper, candidates, observed_assets, descriptors, *, confidence_scale=1.):
    """Compute equal-dimensional geometry/semantic inputs from a restored history.

    Descriptors are V17 observed-mesh masked outbound-only records, not continuation result records.
    Only explicitly named fields are projected; identities remain audit metadata.
    No reference mesh, world, evaluator or future packets are accepted.
    """
    confidence_scale = _unit(confidence_scale, 'confidence_scale')
    if len(candidates) != len(descriptors):
        raise ValueError('candidate/descriptor count mismatch')
    if not candidates:
        return []
    visibility = ObservedMeshAperture(mapper.mesh(), mapper.config)
    belief = np.asarray(mapper.belief)
    if belief.ndim != 2 or not np.isin(belief, [-1, 0, 1]).all():
        raise ValueError('invalid observed occupancy')
    config = mapper.config
    resolution = float(config.resolution_m)
    radius = float(config.robot_radius_m)
    scale = float(config.max_depth_m)
    if not np.isfinite([resolution, radius, scale]).all() or resolution <= 0 or radius < 0 or scale <= 0:
        raise ValueError('invalid sensor/map scale')
    safe = ~inflated_obstacles(belief != 0, radius / resolution)
    graph, cells, ids = orientation_graph(safe)
    n = len(observed_assets)
    geometry_fields = ('observed_low', 'observed_high', 'aabb_center', 'front_axis',
                       'back_axis', 'side_axis', 'rear_boundary_xy', 'measured_width_m')
    # Explicit projection excludes class values and any accidental truth fields.
    assets = [{key: np.asarray(a[key], dtype=float) for key in geometry_fields}
              for a in observed_assets]
    for a in assets:
        for key in ('observed_low', 'observed_high', 'aabb_center'):
            _vector(a[key], (3,), key)
        for key in ('front_axis', 'back_axis', 'side_axis', 'rear_boundary_xy'):
            _vector(a[key], (2,), key)
        if not np.isfinite(a['measured_width_m']) or a['measured_width_m'] < 0:
            raise ValueError('invalid measured width')

    def node(pose):
        if len(pose) != 3:
            raise ValueError('invalid route pose')
        row, col, heading = [_integer(x, 'route pose') for x in pose]
        if row >= safe.shape[0] or col >= safe.shape[1] or heading > 3 or not safe[row, col]:
            raise ValueError('route leaves known-safe map')
        return int(ids[row, col]) * 4 + heading

    endpoints, start, anchor, budget, total = [], None, None, None, None
    candidate_ids = set()
    for c, d in zip(candidates, descriptors):
        if d.get('observation_horizon') != 'outbound_only_v17_observed_mesh':
            raise ValueError('outbound-only descriptor contract required')
        if c['candidate_id'] != d['candidate_id'] or c['candidate_id'] in candidate_ids:
            raise ValueError('candidate identity mismatch or duplicate')
        candidate_ids.add(c['candidate_id'])
        states = [tuple(p) for p in c['states']]
        out = _integer(c['outbound_cost'], 'outbound cost', 1)
        back = _integer(c['return_cost'], 'return cost')
        b = _integer(d['remaining_budget'], 'budget', 1)
        t = _integer(d['total_budget'], 'total budget', b)
        if len(states) != out + back + 1 or c['cost'] != out + back or out + back > b:
            raise ValueError('route budget/count mismatch')
        if (d['outbound_cost'], d['return_cost'], d['roundtrip_cost']) != (out, back, out + back):
            raise ValueError('descriptor cost mismatch')
        for pose in states:
            node(pose)
        if (_actions(states[:out + 1]) != c['outbound_actions'] or
                _actions(states[out:]) != c['return_actions'] or
                _actions(states) != c['actions']):
            raise ValueError('route action/state mismatch')
        # _actions is a converter, so also verify each edge exists in the graph.
        for before, after in zip(states, states[1:]):
            if graph[node(before), node(after)] != 1:
                raise ValueError('invalid observed-map route transition')
        identity = (tuple(states[0]), tuple(states[-1]), b, t)
        if start is None:
            start, anchor, budget, total = identity
        elif identity != (start, anchor, budget, total):
            raise ValueError('pool history/anchor/budget mismatch')
        endpoints.append(states[out])
    distances = dijkstra(graph, directed=True, indices=[node(p) for p in endpoints])
    end_nodes = [node(p) for p in endpoints]
    poses = [camera_pose(tuple(p[:2]), p[2], config, belief.shape[0]) for p in endpoints]
    start_xy = camera_pose(start[:2], start[2], config, belief.shape[0])[:2, 3]
    support = np.asarray([[float(visibility.support(a, p).mean()) for a in assets]
                          for p in poses]).reshape(len(candidates), n)
    rows = []
    for ci, (c, d) in enumerate(zip(candidates, descriptors)):
        audit_assets = d['observed_assets']
        pa = d['observed_aperture_audit']
        if len(audit_assets) != n:
            raise ValueError('observed instance count mismatch')
        apertures = _vector(pa['aperture_factors'], (n,), 'apertures')
        prefix = _vector(pa['prefix_aperture_support'], (n,), 'prefix support')
        if np.any((apertures < 0) | (apertures > 1) | (prefix < 0) | (prefix > 1)):
            raise ValueError('aperture/support outside [0,1]')
        kernels, signed, confidences, witnesses = [], [], [], []
        for ai, (a, record) in enumerate(zip(assets, audit_assets)):
            if record['asset_index'] != ai:
                raise ValueError('descriptor instance order mismatch')
            conf = _unit(record['semantic_confidence'], 'semantic confidence') * confidence_scale
            vote = float(record['class_vote'])
            if not np.isfinite(vote) or abs(vote) > 1:
                raise ValueError('invalid class vote')
            gap = _vector(record['direction_unobserved_fractions'], (8,), 'direction gaps')
            if np.any((gap < 0) | (gap > 1)):
                raise ValueError('invalid direction gap')
            uncertainty = float(record['inverse_sqrt_support_uncertainty_proxy'])
            if not np.isfinite(uncertainty) or not 0 <= uncertainty <= 1:
                raise ValueError('invalid support uncertainty')
            offset = poses[ci][:2, 3] - a['aabb_center'][:2]
            endpoint_distance = float(np.linalg.norm(offset))
            proximity = 1 / (1 + endpoint_distance / scale)
            progress = (np.linalg.norm(start_xy - a['aabb_center'][:2]) - endpoint_distance) / scale
            # Same world-XY azimuth convention as quality evidence (8 sectors).
            sector = int(np.floor((np.arctan2(offset[1], offset[0]) + np.pi) / (2 * np.pi) * 8)) % 8
            choices = []
            for wi, witness in enumerate(candidates):
                extra = float(distances[ci, end_nodes[wi]])
                home = float(distances[wi, node(anchor)])
                paid = c['outbound_cost'] + extra + home
                if extra > 0 and np.isfinite(paid) and paid <= budget and support[wi, ai] > 0:
                    value = support[wi, ai] * (1 - prefix[ai]) / (1 + extra / total)
                    choices.append((value, -paid, tuple(endpoints[wi]), wi, extra, home))
            best = max(choices) if choices else None
            witnesses.append(None if best is None else dict(
                candidate_id=candidates[best[3]]['candidate_id'],
                known_safe_extra_cost=int(best[4]), reserved_return_cost=int(best[5]),
                total_cost=int(c['outbound_cost'] + best[4] + best[5]),
                hypothetical_aperture_support=float(support[best[3], ai])))
            kernels.append([proximity, float(progress), proximity * gap[sector], apertures[ai],
                            0. if best is None else best[0], uncertainty * proximity,
                            _stable_mean(gap) * proximity, (1 - prefix[ai]) * proximity])
            signed.append(vote * conf)
            confidences.append(conf)
        kernel = np.asarray(kernels, dtype=float).reshape(n, len(KERNEL_NAMES))
        means = np.asarray([_stable_mean(kernel[:, i]) for i in range(len(KERNEL_NAMES))])
        counts = [float(d[key]) for key in ('expected_new_camera_cells', 'expected_new_radar_cells')]
        if not np.isfinite(counts).all() or any(x < 0 or x > belief.size for x in counts):
            raise ValueError('invalid observed coverage proxy')
        common = np.r_[budget / total, c['outbound_cost'] / total, c['return_cost'] / total,
                       (budget - c['cost']) / total,
                       _unit(d['unknown_allocated_grid_fraction'], 'unknown fraction'),
                       np.asarray(counts) / belief.size,
                       _unit(d['camera_yield_posterior'], 'camera yield'),
                       _unit(d['radar_yield_posterior'], 'radar yield'), np.log1p(n), means]
        sem_tail = [_stable_mean(np.asarray(signed) * kernel[:, i]) for i in range(len(KERNEL_NAMES))]
        geo_tail = [_stable_mean(kernel[:, i] ** 2) for i in range(len(KERNEL_NAMES))]
        rows.append(dict(candidate_id=c['candidate_id'], geometry=np.r_[common, geo_tail].tolist(),
                         semantic=np.r_[common, sem_tail].tolist(),
                         residual_confidence=max(confidences, default=0.), instance_kernels=kernel.tolist(),
                         second_view_witnesses=witnesses, observed_instance_count=n))
    return rows
