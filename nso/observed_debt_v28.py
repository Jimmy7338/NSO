"""V28 observation-only descriptor and shortlist development prototype.

Cue neighborhoods are NOT instance segmentation. Direction deficits are NOT Q
or calibrated uncertainty. G never receives cue positions or classes. O/S share
the same geometry/cue-position proposal pool; class changes ranking only.
No runtime execution or feedback write-back is enabled by this module.
"""
from copy import deepcopy
from dataclasses import dataclass, replace
import math
import numpy as np

from nso.observed_state_v26 import GeometryStateV26, SemanticCueV26
from nso.observed_planner_v26 import ObservedPlannerV26, _cell, _xy, _heading, _sector


VERSION = 'v28-observed-debt-shortlist-preflight-1'
CONFIDENCE_MIN = .6
NEIGHBOR_RADIUS_M = 1.5
PROPOSAL_RADIUS_M = 1.6
GAIN_SCALE = .05


def describe_neighborhood(state, cue):
    """Use full whitelisted geometry, including background; never infer a mesh.

    Residual is the mapper's mean squared point-to-plane residual, in m^2.
    Observation counts are frame counts, not distinct viewpoints.
    """
    if not isinstance(state, GeometryStateV26) or not isinstance(cue, SemanticCueV26):
        raise TypeError('typed observed geometry and cue required')
    if cue.action_id > state.action_id:
        raise ValueError('future cue forbidden')
    patches = tuple(p for p in state.patches if math.hypot(
        p.point[0]-cue.center[0], p.point[1]-cue.center[1]) <= NEIGHBOR_RADIUS_M)
    support = tuple(float(np.mean([bool(p.bits & (1 << k)) for p in patches]))
                    if patches else 0. for k in range(8))
    descriptor = dict(cue_id=cue.cue_id, action_id=state.action_id,
        patch_count=len(patches), directional_support=list(support),
        directional_deficit=[1.-v for v in support],
        median_best_range_m=float(np.median([p.best_range for p in patches])) if patches else None,
        median_point_plane_residual_m2=float(np.median([p.residual for p in patches])) if patches else None,
        median_observation_frames=float(np.median([p.n for p in patches])) if patches else None,
        distinct_viewpoints=None, effective_baseline_m=None, boundary_fragmentation=None,
        normal_stability=None, calibrated_quality_uncertainty=None,
        support_scope='all measured patches within 1.5m of observed cue; may include background',
        unavailable_fields_are_not_zero=True, instance_segmentation=False,
        predicts_Q=False, evaluation_truth_used=False)
    return descriptor, patches


def directional_weights(cue, class_conditioned=True):
    """Normalize the V26 four direction prior across eight absolute sectors.

    The reference is first observed line of sight, NOT an estimated object axis.
    Equal total mass avoids assigning a bigger arbitrary reward to class 3.
    """
    if not class_conditioned:
        return np.full(8, 1./8)
    outward = np.asarray(cue.outward[:2], float)
    outward /= np.linalg.norm(outward)
    tangent = np.array([-outward[1], outward[0]])
    directions = np.asarray([outward, -outward, tangent, -tangent])
    masses = np.array((.15, .35, .25, .25) if cue.class_id == 3 else (.6, .1, .15, .15))
    angle = -np.pi + (np.arange(8)+.5)*np.pi/4
    sector_vectors = np.column_stack((np.cos(angle), np.sin(angle)))
    kernels = np.maximum(0., sector_vectors @ directions.T)**2
    kernels /= kernels.sum(axis=0, keepdims=True)
    weights = kernels @ masses
    return weights / weights.sum()


@dataclass(frozen=True)
class DebtContextV28:
    state: GeometryStateV26
    cues: tuple
    descriptors: dict
    geometry_plan: dict
    pool: tuple
    cue_geometry: dict
    semantic_route_queries: int


def build_context(state, full_state, cues, geometry_plan=None):
    """Generate a class-independent cue-position pool on the current safe map.

    Geometry proposal baseline remains V26 G; only that baseline's final pool
    is reused here. This is a bounded proposal/ranking prototype, not full ANS.
    """
    if not isinstance(state, GeometryStateV26) or not isinstance(full_state, GeometryStateV26):
        raise TypeError('typed observed geometry required')
    for name in state.__dataclass_fields__:
        if name in ('patches', 'geometry_sha256'):
            continue
        same = np.array_equal(state.belief, full_state.belief) if name == 'belief' else (
            getattr(state, name) == getattr(full_state, name))
        if not same:
            raise ValueError('full support belongs to a different observation: '+name)
    if len(full_state.patches) > 256:
        sampled = tuple(full_state.patches[i] for i in np.linspace(
            0, len(full_state.patches)-1, 256, dtype=int))
    else:
        sampled = full_state.patches
    if state.patches != sampled:
        raise ValueError('full support does not match original planning sample')
    cues = tuple(cues)
    if any(not isinstance(c, SemanticCueV26) or c.action_id > state.action_id for c in cues):
        raise ValueError('only observed semantic cues allowed')
    if len({c.cue_id for c in cues}) != len(cues):
        raise ValueError('duplicate cue identity')
    planner = ObservedPlannerV26('G')
    base = planner.plan(state) if geometry_plan is None else deepcopy(geometry_plan)
    if base['audit']['mode'] != 'G' or base['audit']['geometry_sha256'] != state.geometry_sha256:
        raise ValueError('geometry baseline must belong to this observation')
    space = planner._space(state)
    pool = {tuple(r['pose']): deepcopy(r) for r in base['candidates']}
    descriptors, patches_by_cue = {}, {}
    queries = 0
    for cue in sorted(cues, key=lambda c: c.cue_id):
        descriptors[cue.cue_id], patches_by_cue[cue.cue_id] = describe_neighborhood(full_state, cue)
        if cue.confidence < CONFIDENCE_MIN:
            continue
        for k in range(8):
            az = -math.pi+(k+.5)*math.pi/4
            xy = np.asarray(cue.center[:2])+PROPOSAL_RADIUS_M*np.array([math.cos(az), math.sin(az)])
            cell = _cell(state, xy)
            if any(v < 0 or v >= n for v, n in zip(cell, state.belief.shape)):
                continue
            pose = (*cell, _heading(np.asarray(cue.center[:2])-_xy(state, (*cell, 0))))
            queries += 1
            route = space.route(pose, group='observed_cue_direction_v28')
            if route is not None and tuple(pose) not in pool:
                pool[tuple(pose)] = dict(**route, sources=[dict(kind='observed_cue_position_ring',
                    cue_id=cue.cue_id, cue_action_id=cue.action_id)], semantic_evidence=[], semantic_gain=0.)
    arrays, masks = planner._patch_arrays(state), {}
    known = float(np.count_nonzero(state.belief != -1)/state.belief.size)
    cue_geometry = {}
    for pose, row in pool.items():
        gain, support = planner._geometry_gain(state, pose, arrays, masks)
        unknown = int(space.visibility(pose[:2]).sum())
        # Existing geometry score/cost convention retained for this mechanism
        # comparison; a new budget controller is deliberately not conflated.
        row.update(geometry_gain=gain, geometry_score=(unknown/state.belief.size+known*gain)
            /max(1, row['outbound_cost']), coverage_unknown_cells=unknown, observed_support=support)
        info = {}
        for cue in cues:
            delta = _xy(state, pose)-np.asarray(cue.center[:2])
            if cue.confidence < CONFIDENCE_MIN or not .5 <= np.linalg.norm(delta) <= 2.5:
                continue
            if _heading(-delta) != pose[2]:
                continue
            local = replace(state, patches=patches_by_cue[cue.cue_id])
            _, visible = planner._geometry_gain(state, pose, planner._patch_arrays(local), masks)
            count = visible['visible_patches']
            if count:
                info[cue.cue_id] = dict(sector=_sector(delta), visible_patches=count,
                    local_patches=len(local.patches), visible_fraction=count/len(local.patches))
        cue_geometry[pose] = info
    return DebtContextV28(state, cues, descriptors, base, tuple(pool.values()), cue_geometry, queries)


def rank_context(context, mode='S', reserve_instances=True):
    """S=class prior; O=same cue locations with uniform prior; G=geometry only.

    Reservation is symmetric across cues, never a forced class-3 selection.
    At most four existing coverage routes and twelve total routes are kept.
    """
    if mode not in ('G', 'O', 'S', 'X', 'L'):
        raise ValueError('unknown V28 mode')
    if mode == 'G' or mode == 'L' or not any(c.confidence >= CONFIDENCE_MIN for c in context.cues):
        result = deepcopy(context.geometry_plan)
        result['v28_audit'] = dict(mode=mode, version=VERSION, exact_geometry_fallback=True,
            eligible_cue_ids=[], retained_cue_ids=[], selected_cue_ids=[], predictions_calibrated=False,
            evaluation_truth_used=False, forced_class_selection=False)
        return result
    cues = {c.cue_id: replace(c, class_id=5-c.class_id) if mode == 'X' else c for c in context.cues}
    known = float(np.count_nonzero(context.state.belief != -1)/context.state.belief.size)
    rows = []
    for source in context.pool:
        row = deepcopy(source)
        evidence = []
        for cue_id, geometry in context.cue_geometry[tuple(row['pose'])].items():
            cue = cues[cue_id]
            d = context.descriptors[cue_id]
            weights = directional_weights(cue, class_conditioned=mode != 'O')
            k = geometry['sector']
            debt = float(weights[k]*d['directional_deficit'][k])
            gain = GAIN_SCALE*cue.confidence*8.*debt*geometry['visible_fraction']
            if gain > 0:
                evidence.append(dict(cue_id=cue_id, class_id=cue.class_id if mode != 'O' else None,
                    sector=k, hypothesis_gain=gain, normalized_prior_weight=float(weights[k]),
                    observed_direction_deficit=d['directional_deficit'][k], **{key: value for key, value
                        in geometry.items() if key != 'sector'}, calibrated=False))
        total = sum(e['hypothesis_gain'] for e in evidence)
        row.update(semantic_evidence=evidence, semantic_gain=total,
            score=row['geometry_score']+known*total/max(1, row['outbound_cost']))
        rows.append(row)
    order = lambda r: (-r['score'], r['outbound_cost'], r['cost'], tuple(r['pose']))
    coverage = sorted((r for r in rows if r['group'].startswith('coverage_')), key=order)[:4]
    chosen = {tuple(r['pose']): r for r in coverage}
    eligible = sorted({e['cue_id'] for r in rows for e in r['semantic_evidence']})
    reserved = []
    if reserve_instances:
        for cue_id in eligible:
            if any(any(e['cue_id'] == cue_id for e in r['semantic_evidence']) for r in chosen.values()):
                continue
            if len(chosen) == 12:
                break
            candidates = [r for r in rows if any(e['cue_id'] == cue_id for e in r['semantic_evidence'])]
            row = min(candidates, key=order)
            chosen[tuple(row['pose'])] = row
            reserved.append(cue_id)
    for row in sorted(rows, key=order):
        if len(chosen) >= 12:
            break
        chosen.setdefault(tuple(row['pose']), row)
    shortlist = sorted(chosen.values(), key=order)
    for i, row in enumerate(shortlist):
        row['candidate_id'] = i
    selected = shortlist[0] if shortlist and shortlist[0]['score'] > 0 else None
    retained = sorted({e['cue_id'] for r in shortlist for e in r['semantic_evidence']})
    audit = dict(version=VERSION, mode=mode, exact_geometry_fallback=False,
        raw_pool_size=len(rows), candidate_count=len(shortlist), candidate_limit=12,
        eligible_cue_ids=eligible, retained_cue_ids=retained, reserved_cue_ids=reserved,
        selected_cue_ids=[] if selected is None else [e['cue_id'] for e in selected['semantic_evidence']],
        dropped_eligible_cue_ids=sorted(set(eligible)-set(retained)),
        class_independent_proposal_pool=True, instance_reservation_enabled=reserve_instances,
        forced_class_selection=False, evaluation_truth_used=False, predictions_calibrated=False,
        mu_delta_Q=None, sigma_delta_Q=None, actual_C80_guaranteed=False,
        coverage_routes_retained=len(coverage), semantic_route_queries=context.semantic_route_queries,
        observed_geometry_sha256=context.state.geometry_sha256)
    return dict(candidates=shortlist, selected=selected, v28_audit=audit)
