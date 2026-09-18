"""V26 observation-bound candidate proposal/ranking prototype.

No simulator, object list, service catalogue, evaluator, or future reward input.
Semantic proposals use an EXPLICIT UNTRAINED directional hypothesis, not a
qualified quality predictor. This file establishes the experimental interface;
it does not establish semantic efficacy, novelty, or trained RPN uncertainty.
"""
from time import perf_counter
from types import SimpleNamespace
import math
import numpy as np

from nso.execution_guard_v8 import ObservedExecutionGuard
from nso.facility_candidates_v20 import ObservedRouteSpaceV20
from nso.observed_state_v26 import GeometryStateV26, SemanticCueV26
from utils.grid_geometry import visible_mask


MODES = ('G', 'GP', 'GR', 'S', 'S_no_feedback')


def _xy(state, pose):
    return np.array([(pose[1] + .5) * state.resolution_m,
                     (state.belief.shape[0] - pose[0] - .5) * state.resolution_m])


def _cell(state, xy):
    return (state.belief.shape[0] - 1 - int(math.floor(xy[1] / state.resolution_m)),
            int(math.floor(xy[0] / state.resolution_m)))


def _heading(delta):
    return int(round(math.atan2(float(delta[0]), float(delta[1])) / (math.pi / 2))) % 4


def _sector(delta):
    return int(math.floor((math.atan2(float(delta[1]), float(delta[0])) + math.pi)
                          / (2 * math.pi) * 8)) % 8


class ObservedPlannerV26:
    """ANS-style global proposals with a shared observed local execution gate.

    G: geometry proposals/ranking; GP: add semantic proposals only;
    GR: geometry proposals + semantic ranking; S: both + observation feedback;
    S_no_feedback: both without the extra per-cue feedback correction.
    All modes still update their measured geometric state after every action.
    """
    def __init__(self, mode='G', max_candidates=12):
        if mode not in MODES:
            raise ValueError('unsupported V26 mechanism mode')
        if type(max_candidates) is not int or max_candidates < 4:
            raise ValueError('at least four candidate slots required')
        self.mode, self.max_candidates = mode, max_candidates
        self.feedback = {}
        self.feedback_action = None
        self.calls = []

    def _record(self, module, operation, payload):
        self.calls.append(dict(call_id=len(self.calls) + 1, module=module,
                               operation=operation, **payload))

    @staticmethod
    def _space(state):
        # A whitelist adapter: old route code receives NO mapper quality,
        # semantic images, object metadata, world configuration, or GT mask.
        config = SimpleNamespace(resolution_m=state.resolution_m,
            robot_radius_m=state.robot_radius_m, max_depth_m=state.max_depth_m)
        mapper = SimpleNamespace(belief=state.belief, shape=state.belief.shape, config=config)
        return ObservedRouteSpaceV20(mapper, state.position, state.heading,
                                     state.anchor, state.remaining_budget)

    @staticmethod
    def _patch_arrays(state):
        patches = state.patches
        if not patches:
            return None
        return dict(points=np.asarray([p.point for p in patches]),
            normals=np.asarray([p.normal for p in patches]),
            n=np.asarray([p.n for p in patches]), bits=np.asarray([p.bits for p in patches]),
            best_range=np.asarray([p.best_range for p in patches]),
            residual=np.asarray([p.residual for p in patches]))

    @staticmethod
    def _geometry_gain(state, pose, arrays, camera_masks):
        if arrays is None:
            return 0., dict(visible_patches=0, novel_direction_patches=0)
        xy = _xy(state, pose)
        origin = np.r_[xy, state.camera_height_m]
        delta = arrays['points'] - origin
        heading = pose[2]
        forward = np.array([(0, 1), (1, 0), (0, -1), (-1, 0)][heading])
        right = np.array([forward[1], -forward[0]])
        axial = delta[:, :2] @ forward
        lateral = delta[:, :2] @ right
        tangent = math.tan(math.radians(state.fov_deg) / 2)
        visible = ((axial > .15) & (axial < state.max_depth_m)
            & (abs(lateral) < axial * tangent)
            & (abs(delta[:, 2]) < axial * tangent * state.height_px / state.width_px))
        key = tuple(pose)
        if key not in camera_masks:
            camera_masks[key] = visible_mask(state.belief == 1, tuple(pose[:2]), heading,
                int(state.max_depth_m / state.resolution_m), state.fov_deg)
        cells = np.asarray([_cell(state, p[:2]) for p in arrays['points']])
        inside = ((cells[:, 0] >= 0) & (cells[:, 0] < state.belief.shape[0])
                  & (cells[:, 1] >= 0) & (cells[:, 1] < state.belief.shape[1]))
        ids = np.flatnonzero(inside)
        ray_visible = np.zeros(len(visible), bool)
        ray_visible[ids] = camera_masks[key][cells[ids, 0], cells[ids, 1]]
        visible &= ray_visible
        distance = np.linalg.norm(delta, axis=1)
        sector = np.asarray([_sector(-d[:2]) for d in delta])
        novel = (arrays['bits'] & (1 << sector)) == 0
        # Uncalibrated measurable proxies: additional direction and precision
        # support. Every surface uses the SAME rule, with no objectness mask.
        incidence = np.abs(np.sum(arrays['normals'] * (-delta), axis=1)) / np.maximum(distance, .01)
        precision = np.maximum(0., 1. - distance / np.maximum(arrays['best_range'], .01))
        repeat = 1. / (arrays['n'] + 1.)
        gain = visible * incidence * (.5 * novel + .3 * precision + .2 * repeat)
        return float(np.sum(gain) / max(1, len(visible))), dict(
            visible_patches=int(visible.sum()), novel_direction_patches=int((visible & novel).sum()),
            geometric_proxy='mean_direction_precision_and_repeat_support; not evaluator Q')

    @staticmethod
    def _targets(state):
        # Spatial representatives from ALL measured geometry. No class filter,
        # binary marker mask, catalogue role A/B, or known facility count.
        groups = {}
        for patch in state.patches:
            key = tuple(np.floor(np.asarray(patch.point[:2]) / .8).astype(int))
            deficit = (1. / (patch.n + 1.) + (8 - int(patch.bits).bit_count()) / 8.
                       + min(patch.residual / .01, 1.))
            row = (-deficit, patch.key, patch)
            if key not in groups or row[:2] < groups[key][:2]:
                groups[key] = row
        return [row[2] for row in sorted(groups.values(), key=lambda x: x[:2])[:12]]

    def _semantic_gain(self, state, pose, cues, camera_masks):
        total, evidence = 0., []
        xy = _xy(state, pose)
        for cue in cues:
            delta = xy - np.asarray(cue.center[:2])
            distance = float(np.linalg.norm(delta))
            if not .5 <= distance <= 2.5:
                continue
            if _heading(-delta) != pose[2]:
                continue
            sector = _sector(delta)
            outward = np.asarray(cue.outward[:2])
            tangent = np.array([-outward[1], outward[0]])
            center = np.asarray(cue.center[:2])
            hypotheses = (center, center - outward,
                          center - .5 * outward + .8 * tangent,
                          center - .5 * outward - .8 * tangent)
            weights = ((.15, .35, .25, .25) if cue.class_id == 3 else (.6, .1, .15, .15))
            # Both classes share this public, untrained template. Evaluate
            # predicted visibility against observed obstacles, never GT rays.
            key = tuple(pose)
            if key not in camera_masks:
                camera_masks[key] = visible_mask(state.belief == 1, tuple(pose[:2]), pose[2],
                    int(state.max_depth_m / state.resolution_m), state.fov_deg)
            visible_hypotheses = []
            forward = np.array([(0, 1), (1, 0), (0, -1), (-1, 0)][pose[2]])
            right = np.array([forward[1], -forward[0]])
            fov_tangent = math.tan(math.radians(state.fov_deg) / 2)
            for hypothesis in hypotheses:
                cell = _cell(state, hypothesis)
                inside = all(0 <= v < n for v, n in zip(cell, state.belief.shape))
                vector = hypothesis - xy
                axial, lateral = float(vector @ forward), float(vector @ right)
                in_view = (.15 < axial < state.max_depth_m and abs(lateral) < axial * fov_tangent
                    and abs(cue.center[2] - state.camera_height_m)
                    < axial * fov_tangent * state.height_px / state.width_px)
                visible_hypotheses.append(bool(inside and in_view and camera_masks[key][cell]))
            directional_prior = sum(w for w, visible in zip(weights, visible_hypotheses) if visible)
            if directional_prior <= 0:
                continue
            count, observed_yield = self.feedback.get((cue.cue_id, sector), (0, 0.))
            correction = ((1. + observed_yield) / (1. + count)
                          if self.mode == 'S' else 1.)
            # Geometry already acquired around the seed suppresses repeated
            # direction hypotheses for all modes, including no-IGCR ablation.
            nearby = [p for p in state.patches
                      if np.linalg.norm(np.asarray(p.point[:2]) - cue.center[:2]) <= 1.5]
            support = (sum(bool(p.bits & (1 << sector)) for p in nearby) / len(nearby)
                       if nearby else 0.)
            gain = .05 * cue.confidence * directional_prior * (1. - support) * correction
            total += gain
            evidence.append(dict(cue_id=cue.cue_id, cue_action_id=cue.action_id,
                cue_source=cue.source, class_id=cue.class_id, confidence=cue.confidence,
                sector=sector, measured_direction_support=support,
                visible_template_points=visible_hypotheses, hypothesis_gain=gain,
                feedback_attempts=count, feedback_factor=correction,
                visibility_model='observed_2d_occlusion_and_3d_frustum; unknown_is_hypothesis'))
        return float(total), evidence

    def plan(self, geometry, cues=()):
        started = perf_counter()
        if not isinstance(geometry, GeometryStateV26):
            raise TypeError('planner accepts only the V26 observation whitelist')
        state = geometry
        cues = tuple(cues)
        if self.mode == 'G' and cues:
            raise ValueError('G must not receive class cues or marker locations')
        if any(not isinstance(c, SemanticCueV26) or c.action_id > state.action_id for c in cues):
            raise ValueError('only past/current measured semantic cues allowed')
        self._record('OV-SDF', 'consume_observation_whitelist', dict(
            action_id=state.action_id, geometry_sha256=state.geometry_sha256,
            measured_patch_count=len(state.patches), visible_semantic_cue_count=len(cues)))
        space = self._space(state)
        arrays, camera_masks = self._patch_arrays(state), {}
        coverage, coverage_audit = space.coverage_candidates(slots=min(4, self.max_candidates))
        pool = {}
        def offer(route, source):
            if route is None:
                return
            key = tuple(route['pose'])
            if key not in pool:
                pool[key] = dict(route=route, sources=[])
            pool[key]['sources'].append(source)
        for route in coverage:
            offer(route, dict(kind='observed_frontier', action_id=state.action_id,
                             geometry_sha256=state.geometry_sha256))
        geometric_proposals = semantic_proposals = 0
        if space.available:
            for patch in self._targets(state):
                center = np.asarray(patch.point[:2])
                for radius in (1., 1.6):
                    for sector in range(8):
                        az = -math.pi + (sector + .5) * math.pi / 4
                        xy = center + radius * np.array([math.cos(az), math.sin(az)])
                        cell = _cell(state, xy)
                        if any(v < 0 or v >= n for v, n in zip(cell, state.belief.shape)):
                            continue
                        pose = (*cell, _heading(center - _xy(state, (*cell, 0))))
                        route = space.route(pose, group='observed_geometry_revisit')
                        geometric_proposals += 1
                        if route is not None:
                            gain, _ = self._geometry_gain(state, pose, arrays, camera_masks)
                            if gain > 0:
                                offer(route, dict(kind='observed_surface_deficit',
                                    patch_key=list(patch.key), action_id=state.action_id,
                                    geometry_sha256=state.geometry_sha256))
            if self.mode in ('GP', 'S', 'S_no_feedback'):
                for cue in cues:
                    center = np.asarray(cue.center[:2])
                    cue_routes = []
                    for sector in range(8):
                        az = -math.pi + (sector + .5) * math.pi / 4
                        xy = center + 1.6 * np.array([math.cos(az), math.sin(az)])
                        cell = _cell(state, xy)
                        if any(v < 0 or v >= n for v, n in zip(cell, state.belief.shape)):
                            continue
                        pose = (*cell, _heading(center - _xy(state, (*cell, 0))))
                        route = space.route(pose, group='semantic_direction_hypothesis')
                        semantic_proposals += 1
                        if route is not None:
                            prior_gain, _ = self._semantic_gain(state, pose, (cue,), camera_masks)
                            if prior_gain > 0:
                                cue_routes.append((prior_gain, route))
                    # A bounded class-CONDITIONAL proposal choice, not merely
                    # a class-blind ring around a supplied marker centroid.
                    for prior_gain, route in sorted(cue_routes,
                            key=lambda item: (-item[0], item[1]['cost'], item[1]['pose']))[:4]:
                        offer(route, dict(kind='visible_class_direction_hypothesis',
                            cue_id=cue.cue_id, cue_action_id=cue.action_id, cue_source=cue.source,
                            class_id=cue.class_id, confidence=cue.confidence,
                            proposal_prior_gain=prior_gain,
                            prior='untrained_generic_directional_v26; no hidden attachment position'))
        rows = []
        known = float(np.count_nonzero(state.belief != -1) / state.belief.size)
        for item in pool.values():
            route = item['route']
            geometry_gain, support = self._geometry_gain(state, route['pose'], arrays, camera_masks)
            semantic_gain, semantic_evidence = self._semantic_gain(state, route['pose'], cues, camera_masks)
            # Endpoint coverage proxy; unlike GT coverage this includes the
            # public rectangular ROI. Return cost is reserved by route().
            unknown = int(space.visibility(route['pose'][:2]).sum())
            coverage_gain = unknown / state.belief.size
            base = (coverage_gain + known * geometry_gain) / max(1, route['outbound_cost'])
            score = base + (known * semantic_gain / max(1, route['outbound_cost'])
                if self.mode in ('GR', 'S', 'S_no_feedback') else 0.)
            rows.append(dict(**route, sources=item['sources'], geometry_gain=geometry_gain,
                coverage_unknown_cells=unknown, geometry_score=base, semantic_gain=semantic_gain,
                score=score, observed_support=support, semantic_evidence=semantic_evidence))
        # All modes have identical final capacity. Proposal modes may spend
        # extra computation; record this instead of calling it equal search.
        coverage_rows = [r for r in rows if r['group'].startswith('coverage_')]
        quality_rows = [r for r in rows if not r['group'].startswith('coverage_')]
        order = lambda r: (-r['score'], r['outbound_cost'], r['cost'], r['pose'])
        # G/GR share the FINAL geometry shortlist, not only raw proposals.
        shortlist_order = (lambda r: (-r['geometry_score'], r['outbound_cost'], r['cost'], r['pose'])) \
            if self.mode in ('G', 'GR') else order
        chosen = sorted(coverage_rows, key=shortlist_order)[:4]
        chosen += sorted(quality_rows, key=shortlist_order)[:max(0, self.max_candidates - 4)]
        chosen.sort(key=order)
        for index, row in enumerate(chosen):
            row['candidate_id'] = index
        selected = chosen[0] if chosen and chosen[0]['score'] > 0 else None
        audit = dict(mode=self.mode, action_id=state.action_id,
            geometry_sha256=state.geometry_sha256, candidate_count=len(chosen),
            candidate_limit=self.max_candidates, geometric_proposal_queries=geometric_proposals,
            semantic_proposal_queries=semantic_proposals, scored_unique_poses=len(rows),
            semantic_cue_count=len(cues), observed_public_roi_known_fraction=known,
            minimum_return_cost=(int(space.backward[space.start]) if space.available else None),
            coverage=coverage_audit, no_fixed_prefix=True, no_mandatory_revisit=True,
            class_conditional_prior_trained=False, predicted_quality_calibrated=False,
            route_catalogue_used=False, evaluation_truth_used=False,
            planning_seconds=perf_counter() - started,
            action_if_no_selected='observed_safe_return_or_stop')
        self._record('STGHP', 'propose_and_rank_observed_targets', audit)
        return dict(candidates=chosen, selected=selected, audit=audit)

    def local_action(self, geometry, selected=None):
        """Check first outbound/return action on current observed map each step."""
        if not isinstance(geometry, GeometryStateV26):
            raise TypeError('local controller requires observed state')
        guard = ObservedExecutionGuard(geometry.resolution_m, geometry.robot_radius_m)
        route = None
        if selected is not None:
            # Recompute directed path; never execute a stale saved action list.
            route = self._space(geometry).route(selected['pose'], group=selected['group'])
        if route is not None and route['outbound_actions']:
            action = route['outbound_actions'][0]
            reason = 'refresh_selected_target_on_current_observation'
        else:
            home = guard.return_plan(geometry.belief, geometry.position, geometry.heading,
                                     geometry.anchor, geometry.remaining_budget)
            action = home.actions[0] if home.available and home.actions else None
            reason = home.reason
        assessment = (guard.assess(geometry.belief, geometry.position, geometry.heading,
                                   action, geometry.remaining_budget) if action is not None else None)
        allowed = bool(assessment and assessment.allowed)
        self._record('RPN-UQ', 'observed_action_and_return_guard', dict(
            action_id=geometry.action_id, proposed_action=action, allowed=allowed,
            reason=assessment.reason if assessment and not allowed else reason,
            uncertainty_calibrated=False, truth_used=False))
        return action if allowed else None

    def observe_transition(self, before, after, selected=None, cues=()):
        """IGCR gets actual consecutive sensor states, never offline Q labels.

        The measured local surface yield only calibrates directional hypotheses
        when the selected endpoint was actually reached, not during approach.
        """
        if after.action_id != before.action_id + 1:
            raise ValueError('feedback requires exactly one actually observed paid transition')
        if self.feedback_action is not None and after.action_id <= self.feedback_action:
            raise ValueError('feedback cannot count a paid observation twice')
        old = {p.key: p for p in before.patches}
        # The bounded sample can change membership. A new sample key is NOT
        # proof of a newly reconstructed surface. Only compare common keys.
        improved = [p for p in after.patches if p.key in old
            and (bool(p.bits & ~old[p.key].bits)
                 or p.best_range < old[p.key].best_range - .01)]
        reached = selected is not None and tuple(selected['pose']) == (*after.position, after.heading)
        updates = []
        if reached:
            predicted = {row['cue_id']: row for row in selected.get('semantic_evidence', ())
                         if row['hypothesis_gain'] > 0}
            _, visible_now = self._semantic_gain(after, selected['pose'], cues, {})
            visible_ids = {row['cue_id'] for row in visible_now}
            for cue in cues:
                if (cue.action_id > before.action_id or cue.cue_id not in predicted
                        or cue.cue_id not in visible_ids):
                    continue
                center = np.asarray(cue.center[:2])
                delta = _xy(after, (*after.position, after.heading)) - center
                if not .5 <= np.linalg.norm(delta) <= 2.5:
                    continue
                key = cue.cue_id, _sector(delta)
                if key[1] != predicted[cue.cue_id]['sector']:
                    continue
                comparable = sum(p.key in old and
                    np.linalg.norm(np.asarray(p.point[:2]) - center) <= 1.5 for p in after.patches)
                if comparable < 4:
                    updates.append(dict(cue_id=cue.cue_id, sector=key[1],
                        status='unavailable_insufficient_common_measured_support',
                        comparable_patches=comparable))
                    continue
                nearby = sum(np.linalg.norm(np.asarray(p.point[:2]) - center) <= 1.5 for p in improved)
                observed = min(1., nearby / 16.)
                count, total = self.feedback.get(key, (0, 0.))
                self.feedback[key] = (count + 1, total + observed)
                updates.append(dict(cue_id=cue.cue_id, sector=key[1], improved_common_patches=nearby,
                                    comparable_patches=comparable, status='updated', observed_yield_proxy=observed))
        self.feedback_action = after.action_id
        event = dict(action_id=after.action_id, known_cell_delta=int(
            np.count_nonzero(after.belief != -1) - np.count_nonzero(before.belief != -1)),
            sampled_key_entries=sum(p.key not in old for p in after.patches),
            improved_common_patches=len(improved), selected_endpoint_reached=reached,
            directional_updates=updates, evaluation_Q_used=False,
            proxy_caveat='bounded sampled patches; not calibrated outline improvement')
        self._record('IGCR', 'measured_transition_feedback', event)
        return event
