"""Observed-surface SWAP/VISTA-inspired mechanisms on the common CPU graph.

These are disclosed adaptations, not executions of the authors' full systems.
Paid RGB marks target membership; its type never weights a structural template.
Sparse observed-point forecasts are separate from unweighted public CAD support.
No World, renderer, evaluation mesh, hidden identity or TSDF is accepted here.
"""
from collections import deque
from copy import deepcopy
from functools import lru_cache
from math import isfinite
from time import monotonic

import numpy as np

from nso.cpu_four_modules_v35 import CPUFourModuleControllerV35, ObservationV35
from nso.observation_belief_v35 import CONFIG_V35, _rgb_class
from nso.online_planner_v35 import PlannerLimitV35


CONFIG_V39 = dict(
    inspection_angle_deg=45., inspection_range_m=3., vista_range_m=4.,
    inspection_reference_resolution_px_per_cm2=2.56,
    inspection_reference_focal_px=480., gamma=.8, geometric_decay=.9,
    geometric_decay_after=5, semantic_relevance=1.,
    coverage_minimum=.8, tie_tolerance=1e-12,
    maximum_coverage_states=1500000, maximum_seconds=20.,
    public_support_definition='mean over the two templates of newly exposed bits / total public support bits',
    vista_geometry_definition='max(sparse measured angular novelty, unweighted public new support fraction)',
    semantic_query='all discovered facility categories equally relevant',
    labels={'SWAP': 'SWAP-inspection-inspired/CPU', 'VISTA': 'VISTA-view-inspired/CPU'},
)


class ExternalPlannerV39:
    def __init__(self, models, ledger, *, method, intrinsic, translation,
                 camera_height_m=.9):
        if method not in CONFIG_V39['labels'] or len(models) != 2:
            raise ValueError('SWAP or VISTA and two common public models required')
        a, b = models
        if (a.poses, a.edges, a.anchor) != (b.poses, b.edges, b.anchor):
            raise ValueError('both templates must share the graph and anchor')
        self.models, self.ledger, self.method = tuple(models), ledger, method
        self.poses, self.edges, self.anchor = a.poses, a.edges, a.anchor
        self.intrinsic = np.array(intrinsic, dtype=float, copy=True)
        self.translation = np.array(translation, dtype=float, copy=True)
        if self.intrinsic.shape != (3, 3) or self.translation.shape != (2,) or not np.isfinite(self.intrinsic).all() or not np.isfinite(self.translation).all() or not isfinite(camera_height_m):
            raise ValueError('finite public K, XY calibration and camera height required')
        self.camera_height_m = float(camera_height_m)
        self.intrinsic.flags.writeable = self.translation.flags.writeable = False
        self._total_support = tuple(m.target_bits+len(m.floor_cells) for m in models)
        if min(self._total_support) < 1 or any(not m.floor_cells for m in models):
            raise ValueError('nonempty public support and floor cells required')
        self._floor_at = tuple(tuple(int(mask) >> m.target_bits for mask in m.observed_masks) for m in models)
        self.paths = tuple(self._bfs(n) for n in range(len(self.poses)))
        self.return_distance = {n: len(paths[self.anchor]) for n, paths in enumerate(self.paths) if self.anchor in paths}
        self._node_for_pose = {tuple(p): n for n, p in enumerate(self.poses)}
        self._class_positions = {2: set(), 3: set()}
        self._frames, self._pose_counts = set(), {}
        self._last_step = -1
        self._decision_index = 0
        self._cached_selection = None
        self.last_observation_receipt = None
        # Nondeterministic performance telemetry must not enter replay receipts.
        self.planning_timings = []

    def _bfs(self, start):
        paths = {start: ()}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for action, following in self.edges[node]:
                if following not in paths:
                    paths[following] = paths[node] + ((action, following),)
                    queue.append(following)
        return paths

    def camera_transform(self, node):
        x, y, heading = self.poses[node]
        forward = np.asarray(((0., 1., 0.), (1., 0., 0.), (0., -1., 0.), (-1., 0., 0.))[heading])
        out = np.eye(4)
        out[:3, :3] = np.column_stack((np.cross(forward, [0., 0., 1.]), [0., 0., -1.], forward))
        out[:3, 3] = (x+self.translation[0], y+self.translation[1], self.camera_height_m)
        return out

    @property
    def registered_class(self):
        present = [code for code, xy in self._class_positions.items() if xy]
        if len(present) == 1 and len(self._class_positions[present[0]]) >= CONFIG_V35['marker_distinct_xy']:
            return present[0]
        return None

    def observe(self, observation):
        """Record paid cue/repetition only; the driver updates the depth ledger."""
        if type(observation) is not ObservationV35:
            raise TypeError('sanitized paid ObservationV35 required')
        if observation.frame_id in self._frames or observation.step <= self._last_step or observation.pose not in self._node_for_pose:
            raise ValueError('new paid frame in increasing step order on the public graph required')
        code, counts = _rgb_class(observation.rgb)
        if all(n >= CONFIG_V35['marker_quorum_pixels'] for n in counts.values()):
            for xy in self._class_positions.values():
                xy.add(observation.pose[:2])
        elif code is not None:
            self._class_positions[code].add(observation.pose[:2])
        self._frames.add(observation.frame_id)
        self._last_step = observation.step
        self._pose_counts[observation.pose] = self._pose_counts.get(observation.pose, 0)+1
        self._cached_selection = None
        self.last_observation_receipt = dict(step=observation.step,
            registered_class=self.registered_class,
            class_distinct_xy={str(k): len(v) for k, v in self._class_positions.items()},
            class_conflict=all(self._class_positions.values()),
            repeated_pose=self._pose_counts[observation.pose] > 1,
            semantic_query=CONFIG_V39['semantic_query'],
            posterior_used_to_choose_action=False, actual_hypothesis_input=False)
        return deepcopy(self.last_observation_receipt)

    def _support_gain(self, masks, following):
        updated = tuple(old | int(m.observed_masks[following]) for old, m in zip(masks, self.models))
        gain = sum((new ^ old).bit_count()/total for old, new, total in zip(masks, updated, self._total_support))/2
        return gain, updated

    def select(self, node, remaining, masks, probability0, geometry_feedback=True,
               excluded_information_nodes=()):
        if type(node) is not int or node not in range(len(self.poses)) or type(remaining) is not int or remaining < 0 or len(masks) != 2 or not isfinite(probability0) or not 0 <= probability0 <= 1:
            raise ValueError('valid graph state, budget, masks and received posterior required')
        if self._last_step < 0:
            raise ValueError('paid observation required before external planning')
        masks = tuple(int(m) for m in masks)
        if any(mask < 0 or mask.bit_length() > total for mask, total in zip(masks, self._total_support)):
            raise ValueError('mask outside public support')
        cache_key = (self._last_step, node, remaining, masks)
        if self._cached_selection is not None and self._cached_selection[0] == cache_key:
            result = deepcopy(self._cached_selection[1])
            result['received_probability0_ignored'] = float(probability0)
            return result
        start = monotonic()
        coverage_states = 0
        floor_masks = tuple(mask >> m.target_bits for mask, m in zip(masks, self.models))

        def qualified(f0, f1):
            return all(f.bit_count()/len(m.floor_cells) >= CONFIG_V39['coverage_minimum']-1e-12 for f, m in zip((f0, f1), self.models))

        @lru_cache(None)
        def continuation(n, left, f0, f1):
            nonlocal coverage_states
            coverage_states += 1
            if coverage_states > CONFIG_V39['maximum_coverage_states'] or (coverage_states % 1024 == 0 and monotonic()-start > CONFIG_V39['maximum_seconds']):
                raise PlannerLimitV35('external coverage-feasibility resource limit exceeded')
            if self.return_distance.get(n, 10**9) > left:
                return False
            if qualified(f0, f1):
                return True  # The exact return exists and coverage is monotone.
            if not left:
                return False
            links = sorted(enumerate(self.edges[n]), key=lambda item: (
                -((self._floor_at[0][item[1][1]] & ~f0).bit_count()+(self._floor_at[1][item[1][1]] & ~f1).bit_count()), item[0]))
            return any(continuation(nxt, left-1, f0 | self._floor_at[0][nxt], f1 | self._floor_at[1][nxt]) for _, (_, nxt) in links)

        if not continuation(node, remaining, *floor_masks):
            raise ValueError('no both-template coverage-qualified exact-return continuation')
        registered = self.registered_class is not None
        relevance = CONFIG_V39['semantic_relevance'] if registered else 0.
        decision = self._decision_index
        c_i = CONFIG_V39['geometric_decay'] ** max(0, decision-CONFIG_V39['geometric_decay_after'])
        snapshot = self.ledger.snapshot()
        surfels = len(snapshot['points'])
        view_cache = {}

        def view(n):
            if n not in view_cache:
                view_cache[n] = self.ledger.score_view(self.intrinsic, self.camera_transform(n),
                    mode='inspection' if self.method == 'SWAP' else 'vista',
                    semantic_relevance=relevance,
                    max_range_m=CONFIG_V39['inspection_range_m'] if self.method == 'SWAP' else CONFIG_V39['vista_range_m'],
                    inspection_angle_deg=CONFIG_V39['inspection_angle_deg'])
            return view_cache[n]

        candidates = []
        rejected = {'return_budget': 0, 'coverage_continuation': 0}
        for destination in range(len(self.poses)):
            path = self.paths[node].get(destination)
            if not path:
                continue
            cost = len(path)
            back = self.return_distance.get(destination, 10**9)
            if cost+back > remaining:
                rejected['return_budget'] += 1
                continue
            planned_masks = masks
            support, geometric, semantic = [], [], []
            inspection_ids = set()
            for _, following in path:
                gain, planned_masks = self._support_gain(planned_masks, following)
                score = view(following)
                support.append(gain)
                geometric.append(max(float(score['geometry_gain']), gain))
                semantic.append(float(score['semantic_gain']))
                if registered:
                    inspection_ids.update(int(i) for i in score['new_inspection_indices'])
            floors = tuple(mask >> m.target_bits for mask, m in zip(planned_masks, self.models))
            if not continuation(destination, remaining-cost, *floors):
                rejected['coverage_continuation'] += 1
                continue
            weights = [CONFIG_V39['gamma'] ** (cost-1-k) for k in range(cost)]
            geometry_score = sum(w*g for w, g in zip(weights, geometric))
            semantic_score = sum(w*s for w, s in zip(weights, semantic))
            inspection_gain = len(inspection_ids)/max(1, surfels)
            candidates.append(dict(destination=destination, action=path[0][0],
                following=path[0][1], path_actions=[a for a, _ in path], path_nodes=[n for _, n in path],
                outbound_cost=cost, reserved_return_cost=back,
                both_template_coverage_continuation=True,
                measured_angular_gain_sum=sum(float(view(n)['geometry_gain']) for _, n in path),
                public_support_gain_sum=sum(support), public_support_is_forecast=True,
                discounted_geometry=geometry_score, discounted_semantics=semantic_score,
                inspection_new_surfel_count=len(inspection_ids), inspection_gain=inspection_gain,
                inspection_score=inspection_gain/cost,
                geometric_exploration_score=sum(geometric)/cost,
                vista_score=c_i*geometry_score+semantic_score,
                vista_without_semantic_score=c_i*geometry_score))

        tol = CONFIG_V39['tie_tolerance']

        def winner(rows, field, extra=None):
            if not rows:
                return None
            best = max(r[field] for r in rows)
            ties = [r for r in rows if r[field] >= best-tol]
            if extra is not None:
                best_extra = max(r[extra] for r in ties)
                ties = [r for r in ties if r[extra] >= best_extra-tol]
            return min(ties, key=lambda r: (r['outbound_cost'], r['destination']))

        if self.method == 'SWAP':
            inspecting = any(r['inspection_score'] > tol for r in candidates)
            phase = 'inspection' if inspecting else 'geometry_exploration'
            selected = winner(candidates, 'inspection_score' if inspecting else 'geometric_exploration_score', 'inspection_gain' if inspecting else None)
            without_semantic = winner(candidates, 'geometric_exploration_score')
            utility_field = 'inspection_score' if inspecting else 'geometric_exploration_score'
            no_semantic_utility_field = 'geometric_exploration_score'
        else:
            phase = 'semantic_direction_exploration'
            selected = winner(candidates, 'vista_score')
            without_semantic = winner(candidates, 'vista_without_semantic_score')
            utility_field = 'vista_score'
            no_semantic_utility_field = 'vista_without_semantic_score'
        # Zero marginal information permits stopping only after exact return.
        if selected is not None and selected[utility_field] <= tol and qualified(*floor_masks):
            selected = next((r for r in candidates if r['destination'] == self.anchor), None)
            phase = 'return_no_positive_view_gain'
        if without_semantic is not None and without_semantic[no_semantic_utility_field] <= tol and qualified(*floor_masks):
            without_semantic = next((r for r in candidates if r['destination'] == self.anchor), None)
        if selected is None and node != self.anchor:
            raise ValueError('external planner could not select a legal exact-return action')
        selected_action = None if selected is None else selected['action']
        no_sem_action = None if without_semantic is None else without_semantic['action']
        semantic_nonzero = registered and any(r['discounted_semantics'] > tol for r in candidates)
        elapsed = monotonic()-start
        if elapsed > CONFIG_V39['maximum_seconds']:
            raise PlannerLimitV35('external declared planning-time limit exceeded')
        result = dict(action=selected_action, method=CONFIG_V39['labels'][self.method],
            mechanism_phase=phase, selected_candidate=deepcopy(selected), candidates=candidates,
            candidate_rejections=rejected, candidate_count=len(candidates),
            received_probability0_ignored=float(probability0), posterior_used=False,
            posterior_used_to_choose_action=False, class_conditioned_structure_forecast=False,
            registered_class=self.registered_class, semantic_query=CONFIG_V39['semantic_query'],
            semantic_term_nonzero=semantic_nonzero,
            semantic_changed_first_action=bool(registered and selected_action != no_sem_action),
            without_semantic_action=no_sem_action,
            semantic_candidate_range=[min((r['discounted_semantics'] for r in candidates), default=0.), max((r['discounted_semantics'] for r in candidates), default=0.)],
            measured_direction_candidate_range=[min((r['measured_angular_gain_sum'] for r in candidates), default=0.), max((r['measured_angular_gain_sum'] for r in candidates), default=0.)],
            positive_semantic_views=sum(float(v['semantic_gain']) > tol for v in view_cache.values()),
            inspection_candidate_count=sum(r['inspection_new_surfel_count'] > 0 for r in candidates),
            observed_surfel_count=surfels, valid_normal_count=int(np.count_nonzero(snapshot['normal_valid'])),
            decision_index=decision, geometric_weight=c_i, gamma=CONFIG_V39['gamma'],
            coverage_states=coverage_states, timing_stored_outside_replay_receipt=True,
            both_template_terminal_qualification=True, exact_return_reserved=True,
            geometry_feedback_parameter_ignored=bool(geometry_feedback),
            excluded_information_nodes_not_forbidden=len(tuple(excluded_information_nodes)),
            forecast_scope='common public support; uniform over both templates, no posterior or future actual sensors',
            objective_scope='external mechanism score, not measured reconstruction quality',
            geometry_scope=CONFIG_V39['vista_geometry_definition'], full_original_system=False)
        self._decision_index += 1
        self.planning_timings.append(dict(step=self._last_step, decision_index=decision, seconds=elapsed))
        self._cached_selection = (cache_key, deepcopy(result))
        continuation.cache_clear()
        return result


class ExternalControllerV39(CPUFourModuleControllerV35):
    """Same paid-action lifecycle; receipts do not claim structural posterior use."""
    def select_target(self):
        if self.state is None:
            raise ValueError('consume action-zero observation first')
        if self.pending is not None or self.terminal_reason is not None:
            raise ValueError('consume pending observation and require an active episode')
        step, node = self.state['step'], self.state['node']
        if self.selected is not None and self.selected['step'] == step:
            return deepcopy(self.selected)
        if step < len(self.prefix_actions):
            result = dict(action=self.prefix_actions[step], phase='common_forced_prefix', posterior_used_to_choose_action=False)
        else:
            result = deepcopy(self.planner.select(node, self.budget-step, self.masks,
                self.belief.probabilities[0], geometry_feedback=True,
                excluded_information_nodes=tuple(sorted(self.visited_nodes))))
            result['phase'] = 'external_cpu_mechanism_planning'
            result['posterior_used_to_choose_action'] = False
        action = result['action']
        destination = self.links[node].get(action) if action is not None else None
        selected = dict(step=step, from_node=node, action=action, node=destination,
            pose=None if destination is None else list(self.poses[destination]),
            probabilities=list(self.belief.probabilities), planning=result)
        self.selected = deepcopy(selected)
        self.plans.append(deepcopy(selected))
        self._record('STGHP', 'select_external_mechanism_next_pose', **selected)
        return deepcopy(selected)

    def summary(self):
        out = super().summary()
        out.update(online_global_plans=sum(r['planning']['phase'] == 'external_cpu_mechanism_planning' for r in self.plans),
            external_method=CONFIG_V39['labels'][self.planner.method],
            structural_posterior_used=False,
            scope='controlled external-inspired CPU planner; shared public graph, paid RGB-D and two public templates; not original full system')
        return out
