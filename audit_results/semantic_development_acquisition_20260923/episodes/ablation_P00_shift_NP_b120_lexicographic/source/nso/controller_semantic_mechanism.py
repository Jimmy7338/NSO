"""Observed-only hierarchical controller with one paid diagnostic lookahead.

Uses the existing four CPU interfaces. All methods share finite candidates,
prototype sensing channels, macro costs and return safety. The shared-class
belief is standard hierarchical Bayes; this module makes no novelty or measured
performance claim. Predicted areas and channels remain uncalibrated proxies.
"""
from copy import deepcopy
from functools import lru_cache
import numpy as np

from nso.controller_v43 import ANSControllerV43, _digest, _state_receipt
from nso.public_navigation_v43 import select_candidate_states_v43
from nso.semantic_reliability import SemanticReliabilityBelief, discrete_observation_evi


METHODS = ('G', 'fixed_semantic', 'bayes_semantic', 'shared_semantic',
           'shared_no_future', 'shared_no_cross_future', 'shared_no_feedback')


def select_unique_target(options, *, tie_rule, rng):
    """One vote per executable target, independent of diagnostic row count."""
    representatives = {}
    for score, target, receipt in options:
        previous = representatives.get(target)
        key = (receipt['kind'], receipt.get('instance_id', ''))
        previous_key = None if previous is None else (
            previous[2]['kind'], previous[2].get('instance_id', ''))
        if (previous is None or score > previous[0] or
                (score == previous[0] and key < previous_key)):
            representatives[target] = (score, target, receipt)
    if not representatives:
        return None
    best = max(item[0] for item in representatives.values())
    if best <= 0:
        return None
    ties = sorted((item for item in representatives.values() if item[0] >= best - 1e-12),
                  key=lambda item: (item[1].node, item[1].heading))
    if tie_rule == 'seeded_random':
        return ties[int(rng.integers(len(ties)))]
    return ties[-1] if tie_rule == 'reverse' else ties[0]


def paid_followup_options(graph, *, current, diagnostic, home, remaining, candidates, route_query=None):
    """Exact primitive costs for move+observe, move+observe, full-pose return.

    Every +1 is an explicit observe action in the shared macro executor. The
    movement frames are also integrated, but are not predicted by this finite
    one-event model. Direct return is represented separately by the caller.
    """
    route = graph.route if route_query is None else route_query
    outbound = route(current, diagnostic)
    direct_return = route(diagnostic, home)
    if outbound is None or direct_return is None:
        return []
    if outbound.cost + 1 + direct_return.cost > remaining:
        return []
    rows = []
    for target in candidates:
        if target == diagnostic:
            continue
        middle, inbound = route(diagnostic, target), route(target, home)
        if middle is None or inbound is None:
            continue
        cost = outbound.cost + 1 + middle.cost + 1 + inbound.cost
        if cost <= remaining:
            rows.append(dict(target=target, total_cost=cost,
                             first_macro_cost=outbound.cost + 1,
                             second_macro_cost=middle.cost + 1,
                             return_cost=inbound.cost))
    return rows


def paid_diagnostic_value(belief, *, instance_id, channel, followups,
                          structure_gains, discovery_gains,
                          allow_future=True, allow_cross_instance_future=True):
    """Score only the final view gain, divided by the entire paid route cost.

    The first view's surface gain is deliberately not added: the existing
    predictor supplies aggregate areas, not surface-set overlaps. Its area
    cannot safely be added to the final view's area. Return without a second
    view is a zero-gain option; direct-view plans are compared by the caller.
    """
    if not followups or not structure_gains:
        return dict(score=0., evi=0., future_information_used=False, branches=[])
    keys = sorted(structure_gains)
    matrices = {}
    for key in keys:
        matrix = np.zeros((len(followups) + 1, belief.structure_count))
        for index, row in enumerate(followups, 1):
            matrix[index] = np.asarray(structure_gains[key][row['target']]) / row['total_cost']
        if not allow_cross_instance_future and key != instance_id:
            expected = matrix @ np.asarray(belief.posterior(key)['structure_probabilities'])
            matrix = np.repeat(expected[:, None], belief.structure_count, axis=1)
        matrices[key] = matrix
    # Add a belief-independent discovery term exactly once per global action.
    for index, row in enumerate(followups, 1):
        matrices[keys[0]][index] += discovery_gains[row['target']] / row['total_cost']
    value = discrete_observation_evi(belief, instance_id, channel, matrices)
    before = value['best_utility_before_observation']
    after = value['expected_best_utility_after_observation']
    return dict(score=after if allow_future else before, evi=value['evi'],
                before_observation_score=before, after_observation_score=after,
                future_information_used=bool(allow_future),
                cross_instance_future_used=bool(allow_future and allow_cross_instance_future),
                cost_includes_both_observations_and_full_return=True,
                first_view_surface_gain_counted=False,
                branches=[dict(outcome=row['outcome'], probability=row['probability'],
                               best_action=row['best_action']) for row in value['branches']])


class SemanticMechanismController(ANSControllerV43):
    """Global view selection, committed local macro, paid evidence correction.

    The controller does not accept true instance structures or reference
    surfaces. A known-structure reference needs a separately declared runner
    and is not implemented by silently passing truth through this interface.
    """
    def __init__(self, graph, *, method='G', tie_rule='lexicographic', tie_seed=0,
                 observation_frontend='bounded_local', measurement_acquisition=False, **kwargs):
        if method not in METHODS or 'mode' in kwargs:
            raise ValueError('explicit mechanism method required; no legacy mode argument')
        if tie_rule not in ('lexicographic', 'reverse', 'seeded_random'):
            raise ValueError('unknown declared tie rule')
        if type(tie_seed) is not int or tie_seed < 0:
            raise ValueError('nonnegative fixed tie seed required')
        if observation_frontend not in ('bounded_local', 'legacy'):
            raise ValueError('explicit common observation frontend required')
        if type(measurement_acquisition) is not bool:
            raise ValueError('measurement acquisition must be an explicit boolean')
        if kwargs.get('uncertainty_penalty', 0.) != 0.:
            raise ValueError('shared correlated beliefs do not support the legacy independent variance penalty')
        super().__init__(graph, mode='G' if method == 'G' else 'S', **kwargs)
        if observation_frontend == 'bounded_local':
            from nso.observed_instances_local import ObservedInstancesLocal
            self._ledger = ObservedInstancesLocal(
                palette=self._ledger.palette, structure_names=self._ledger.names,
                class_structure_prior=self._ledger.class_priors, mode=self._ledger.mode,
                geometry_prior=self._ledger.geometry_prior,
                maximum_instances=self._configuration['maximum_instances'])
        self.method, self.tie_rule = method, tie_rule
        self._tie_rng = np.random.default_rng(tie_seed)
        self._belief = SemanticReliabilityBelief(
            geometry_prior=self._ledger.geometry_prior,
            class_structure_priors=self._ledger.class_priors,
            share_across_instances=method.startswith('shared_'))
        self._actual_feedback = method not in ('fixed_semantic', 'shared_no_feedback')
        self._future_information = method not in ('fixed_semantic', 'shared_no_future')
        self._cross_future = method != 'shared_no_cross_future'
        self._committed_target = None
        self._macro_observe_pending = False
        self._macro_id = 0
        self._last_global_selection = None
        self._acquisition = None
        if measurement_acquisition:
            from nso.observed_view_acquisition import ObservedViewAcquisition
            self._acquisition = ObservedViewAcquisition()
        self._initialization_attempts = {}
        self._initialization_targets = {}
        self._initialization_spent = 0
        self._initialization_active = None
        self._initialization_primitive_pending = False
        self._qualified_plane_instances = set()
        self._configuration.update(method=method, tie_rule=tie_rule, tie_seed=tie_seed,
            observation_frontend=observation_frontend,
            observation_frontend_common_to_all_methods=True,
            tie_unit='unique executable target pose; retain maximum score per target',
            route_cache_scope='one synchronous global selection; cleared before new observed safety updates',
            actual_structure_feedback=self._actual_feedback,
            future_information=self._future_information,
            cross_instance_future=self._cross_future,
            macro='move to selected pose, execute one extra paid observe, then replan',
            planning_horizon='one prototype diagnosis followed by one budget-feasible view',
            lookahead_reward='final view only; divided by both macros and full-pose return',
            initial_semantic_prior='equal mixture of geometry prior and nominal class prior',
            instance_covariance_assumption='class reliability induces dependence; no summed marginal variance penalty',
            inference_status='standard hierarchical generalized Bayes; not calibrated sensor likelihood')
        if self._acquisition is not None:
            self._configuration.update(measurement_acquisition=True,
                measurement_acquisition_scope='shared paid geometry initialization; not semantic utility',
                initialization_action_cap=self._router.budget // 5,
                initialization_attempts_per_instance=2,
                initialization_priority='least attempted eligible instance, then full macro plus return cost',
                initialization_progress='first actual reliable plane per observed instance only',
                initialization_changes_plane_thresholds=False)
        self._configuration_sha256 = _digest(self._configuration)

    def accept(self, observation, mapper, *, execution_outcome='success'):
        evidence = super().accept(observation, mapper, execution_outcome=execution_outcome)
        try:
            completed = self._macro_observe_pending
            if self._acquisition is not None:
                # Only an accepted paid primitive consumes initialization cost.
                if self._initialization_primitive_pending:
                    self._initialization_spent += 1
                self._initialization_primitive_pending = False
                if self._initialization_spent > self._configuration['initialization_action_cap']:
                    raise RuntimeError('shared initialization action cap exceeded')
                acquired = self._acquisition.observe(observation, evidence['association']['accepted'])
                first_planes = []
                for key, row in self._predictor.observed_plane_receipts().items():
                    if (row['plane_fit'] is not None and not row['conflict']
                            and key not in self._qualified_plane_instances):
                        self._qualified_plane_instances.add(key)
                        first_planes.append(key)
                if first_planes:
                    self._no_progress = 0
                evidence.update(measurement_acquisition=acquired,
                    first_actual_reliable_planes=sorted(first_planes),
                    no_progress_paid_actions=self._no_progress,
                    progress_rule='new occupancy/support or first actual reliable plane per instance',
                    initialization_actions_spent=self._initialization_spent)
            if completed:
                self._committed_target = None
                self._macro_observe_pending = False
                self._initialization_active = None
            # All current-frame geometry updates complete before any shared marginal
            # is queried. Cumulative messages replace, rather than multiply, history.
            for instance in self._ledger.snapshot()['instances']:
                key = instance['instance_id']
                label = instance['observed_class'] if (
                    self.method != 'G' and instance['semantic_conditioning_used']
                    and not instance['association_uncertain']) else None
                if key not in self._belief.instance_ids:
                    self._belief.register(key, label)
                else:
                    self._belief.set_class(key, label)
                scores = instance['geometry_log_scores'] if self._actual_feedback else np.zeros(4)
                self._belief.replace_log_evidence(key, scores)
            evidence.update(structure_belief=self._belief.snapshot(),
                            completed_macro_id=self._macro_id if completed else None)
            self._last_accept = deepcopy(evidence)
        except Exception:
            # The base controller has already accepted this paid frame. Neither
            # a partial shared-belief update nor the paid history is retryable.
            self._poisoned = True
            raise
        return evidence

    def _planning_instances(self):
        result = self._ledger.snapshot()['instances']
        for instance in result:
            posterior = self._belief.posterior(instance['instance_id'])
            instance['active_structure_prior'] = posterior['active_structure_prior']
            instance['structure_probabilities'] = posterior['structure_probabilities']
            instance['semantic_conditioning_used'] = posterior['semantic_conditioning_used']
            if not self._actual_feedback:
                instance['geometry_log_scores'] = [0.] * 4
        return result

    def _select_global(self):
        from nso.prototype_observation_channel import build_channel
        # Observed safety changes between accept/choose calls, never during
        # this synchronous selection. Cache exact routes only for its lifetime.
        route = lru_cache(maxsize=None)(self._graph.route)
        current, home = self._router.state, self._router.home
        remaining = self._router.budget - self._router.step
        instances = self._planning_instances()
        protected = {}
        if self._acquisition is None:
            states, pool = select_candidate_states_v43(self._graph, current,
                paid_camera_states=tuple(self._paid_states), limit=self._configuration['maximum_candidates'])
        else:
            states, pool, protected, initialization_rows = self._acquisition.candidates(
                self._graph, current, home, remaining, tuple(self._paid_states), self._camera,
                instances, self._predictor.observed_plane_receipts(),
                limit=self._configuration['maximum_candidates'])
            target = self._select_initialization(initialization_rows, instances, pool, remaining)
            if target is not None:
                return target
        views = [self._candidate_view(state) for state in states]
        discovery_raw = self._safety.discovery_utilities(states,
            radius_m=self._configuration['discovery_radius_m'])
        discovery = {s: 0. if s.node in self._visited_nodes else
                     self._configuration['discovery_weight'] * float(discovery_raw[s]) for s in states}
        gains, forecasts, allocations = {}, [], []
        for instance in instances:
            key = instance['instance_id']
            gains[key] = {s: np.zeros(4) for s in states}
            ordered = self._instance_candidate_order(instance, states, views)
            reserved = protected.get(key, [])
            selected = ([row for state in reserved for row in ordered if row[0] == state]
                        + [row for row in ordered if row[0] not in reserved])[
                            :self._configuration['maximum_views_per_instance']]
            allocations.append(dict(instance_id=key, candidates=[_state_receipt(s) for s, _ in selected]))
            if selected:
                forecast = self._predictor.forecast(instance, [v for _, v in selected])
                forecasts.append(forecast)
                for (state, _), row in zip(selected, forecast['candidates']):
                    gains[key][state] = (self._configuration['inspection_weight'] *
                                        np.asarray(row['structure_new_surface_area_m2']))
        direct_rows, diagnostic_rows, options = [], [], []
        for state in states:
            out, back = route(current, state), route(state, home)
            if out is None or back is None or out.cost + 1 + back.cost > remaining:
                continue
            cost = out.cost + 1 + back.cost
            gain = discovery[state] + sum(float(matrix[state] @ np.asarray(
                self._belief.posterior(key)['structure_probabilities'])) for key, matrix in gains.items())
            row = dict(kind='direct', target=_state_receipt(state), expected_gain=gain,
                       total_cost=cost, score=gain / cost)
            direct_rows.append(row)
            options.append((row['score'], state, row))
        planes = self._predictor.observed_plane_receipts()
        for forecast in forecasts:
            key = forecast['instance_id']
            for item in forecast['candidates']:
                state = next(s for s in states if f'{s.node}:{s.heading}' == item['view_id'])
                followups = paid_followup_options(self._graph, current=current, diagnostic=state,
                    home=home, remaining=remaining, candidates=states, route_query=route)
                if not followups or item['fallback'] or item['repeated_view_excluded']:
                    continue
                channel = build_channel(planes.get(key), self._candidate_view(state),
                                        repeated_view=item['repeated_view_excluded'])
                if channel['fallback']:
                    continue
                value = paid_diagnostic_value(self._belief, instance_id=key,
                    channel=channel['channel'], followups=followups, structure_gains=gains,
                    discovery_gains=discovery, allow_future=self._future_information,
                    allow_cross_instance_future=self._cross_future)
                row = dict(kind='diagnose_then_observe', target=_state_receipt(state),
                    instance_id=key, **value,
                    followups=[dict(target=_state_receipt(r['target']), total_cost=r['total_cost'])
                               for r in followups],
                    channel=channel)
                diagnostic_rows.append(row)
                options.append((row['score'], state, row))
        selected = select_unique_target(options, tie_rule=self.tie_rule, rng=self._tie_rng)
        self._last_global_selection = dict(paid_step=self._router.step, remaining=remaining,
            candidate_pool=pool, instance_candidate_allocations=allocations,
            direct_options=direct_rows, diagnostic_options=diagnostic_rows,
            selected=None if selected is None else deepcopy(selected[2]), forecasts=forecasts,
            predicted_channel_only=True, future_sensor_rendered=False, score_is_calibrated=False)
        return None if selected is None else selected[1]

    def _select_initialization(self, rows, instances, pool, remaining):
        """Shared prerequisite scheduling, separate from m² surface utility.

        An attempt is consumed on commitment. Executed primitives are charged
        on accept(), including a macro later cancelled after an observed change.
        """
        eligible = {x['instance_id'] for x in instances if not x['association_uncertain']}
        planes = self._predictor.observed_plane_receipts()
        available = []
        cap = self._configuration['initialization_action_cap']
        for row in rows:
            key = row['instance_id']
            plane = planes.get(key, {})
            attempts = self._initialization_attempts.get(key, 0)
            if (key not in eligible or attempts >= 2 or plane.get('conflict', False)
                    or plane.get('plane_fit') is not None
                    or row['target'] in self._initialization_targets.get(key, set())
                    or row['outbound_cost'] + 1 > cap-self._initialization_spent
                    or row['total_cost'] > remaining):
                continue
            available.append((attempts, row))
        if not available:
            return None
        least = min(x[0] for x in available)
        options = []
        for attempts, row in available:
            if attempts != least:
                continue
            receipt = dict(kind='measurement_initialization', instance_id=row['instance_id'],
                target=_state_receipt(row['target']), total_cost=row['total_cost'],
                outbound_cost=row['outbound_cost'], return_cost=row['return_cost'],
                quality_proxy=deepcopy(row['quality_proxy']), expected_gain=None, score=None, evi=None,
                scheduling_priority=1./row['total_cost'], semantic_information_used=False,
                counted_as_surface_gain=False, attempt_number=attempts+1)
            options.append((receipt['scheduling_priority'], row['target'], receipt))
        selected = select_unique_target(options, tie_rule=self.tie_rule, rng=self._tie_rng)
        if selected is None:
            return None
        receipt = selected[2]
        key = receipt['instance_id']
        self._initialization_attempts[key] = self._initialization_attempts.get(key, 0)+1
        self._initialization_targets.setdefault(key, set()).add(selected[1])
        self._initialization_active = dict(instance_id=key, target=selected[1])
        self._last_global_selection = dict(paid_step=self._router.step, remaining=remaining,
            candidate_pool=pool, instance_candidate_allocations=[], direct_options=[], diagnostic_options=[],
            initialization_options=[deepcopy(x[2]) for x in options], selected=deepcopy(receipt), forecasts=[],
            initialization_actions_spent=self._initialization_spent,
            initialization_action_cap=cap, predicted_channel_only=True,
            future_sensor_rendered=False, score_is_calibrated=False)
        return selected[1]

    def _check_initialization_commitment(self):
        """Recheck observed association, changed routes and finite paid allowance."""
        active = self._initialization_active
        if active is None:
            return None
        instance = next((x for x in self._ledger.snapshot()['instances']
                         if x['instance_id'] == active['instance_id']), None)
        out = self._graph.route(self._router.state, active['target'])
        back = self._graph.route(active['target'], self._router.home)
        plane = self._predictor.observed_plane_receipts().get(active['instance_id'], {})
        reason = ('return_already_latched' if self._return_latched else
            'association_uncertain' if instance is None or instance['association_uncertain'] else
            'observed_plane_conflict' if plane.get('conflict', False) else
            'plane_acquired' if plane.get('plane_fit') is not None else
            'changed_route_unavailable' if out is None or back is None else
            'initialization_allowance' if out.cost+1 >
                self._configuration['initialization_action_cap']-self._initialization_spent else
            'full_return_budget' if out.cost+1+back.cost > self._router.budget-self._router.step else None)
        if reason is not None:
            receipt = dict(instance_id=active['instance_id'], target=_state_receipt(active['target']),
                reason=reason, attempts_refunded=False, actions_refunded=False)
            self._initialization_active = None
            self._committed_target = None
            self._macro_observe_pending = False
            return receipt
        return None

    def choose(self):
        self._ready()
        if self._camera is None or self._router.pending is not None:
            raise ValueError('accept current paid packet before choosing another primitive')
        initialization_cancelled = self._check_initialization_commitment() if self._acquisition is not None else None
        guard = self._safety.guard_action('observe')
        if not guard['allowed']:
            self._terminal = True
            self._router.pending = None
            self._committed_target = None
            self._macro_observe_pending = False
            result = dict(schema='mechanism.controller_decision.v1', action='blocked',
                reason='observed_current_footprint_conflict', paid_step=self._router.step,
                routing=None, safety_guard=guard, macro_id=self._macro_id, macro_target=None,
                macro_observe_submitted=False, global_replanned=False, global_selection=None,
                configuration_sha256=self._configuration_sha256, method=self.method,
                source_observation_sha256=self._last_observation_sha256,
                actual_future_sensor_rendered=False, ground_truth_scene_input=False)
            self._last_choose = deepcopy(result)
            return result
        if (self._committed_target is None and
                self._no_progress >= self._configuration['no_progress_patience']):
            self._return_latched = True
            self._committed_target = None
        replanned = False
        if not self._return_latched and self._committed_target is None:
            self._committed_target = self._select_global()
            replanned = True
            if self._committed_target is not None:
                self._macro_id += 1
        utilities = {} if self._committed_target is None else {self._committed_target: 1.}
        routing = self._router.choose(utilities)
        # The router ranks a singleton only; upper-level route-cost scores are
        # not divided by outbound cost for a second time.
        guard = self._safety.guard_action(routing['action'])
        action = guard['action']
        if routing['reason'] == 'return':
            self._committed_target = None
            self._initialization_active = None
        if not guard['allowed'] or action in ('stop', 'blocked'):
            self._terminal = True
            self._router.pending = None
        self._macro_observe_pending = bool(action == 'observe' and self._committed_target is not None)
        self._initialization_primitive_pending = bool(self._initialization_active is not None
            and action in ('left', 'right', 'forward', 'observe') and guard['allowed'])
        result = dict(schema='mechanism.controller_decision.v1', action=action,
            reason=guard['reason'] if not guard['allowed'] else routing['reason'],
            paid_step=self._router.step, routing=routing, safety_guard=guard,
            macro_id=self._macro_id, macro_target=None if self._committed_target is None else
            _state_receipt(self._committed_target), macro_observe_submitted=self._macro_observe_pending,
            global_replanned=replanned,
            global_selection=deepcopy(self._last_global_selection) if replanned else None,
            configuration_sha256=self._configuration_sha256, method=self.method,
            source_observation_sha256=self._last_observation_sha256,
            actual_future_sensor_rendered=False, ground_truth_scene_input=False)
        if self._acquisition is not None:
            result.update(initialization_cancelled=initialization_cancelled,
                initialization_actions_spent=self._initialization_spent,
                initialization_action_cap=self._configuration['initialization_action_cap'],
                initialization_primitive_pending=self._initialization_primitive_pending)
        self._last_choose = deepcopy(result)
        return result

    def snapshot(self):
        snapshot = super().snapshot()
        snapshot.update(method=self.method, structure_belief=self._belief.snapshot(),
            macro_id=self._macro_id, macro_target=None if self._committed_target is None else
            _state_receipt(self._committed_target), macro_observe_pending=self._macro_observe_pending)
        if self._acquisition is not None:
            snapshot.update(initialization_actions_spent=self._initialization_spent,
                initialization_attempts=deepcopy(self._initialization_attempts),
                initialization_attempted_targets={key:[_state_receipt(state) for state in
                    sorted(values, key=lambda s:(s.node,s.heading))]
                    for key, values in self._initialization_targets.items()},
                first_reliable_plane_instances=sorted(self._qualified_plane_instances),
                initialization_active=None if self._initialization_active is None else dict(
                    instance_id=self._initialization_active['instance_id'],
                    target=_state_receipt(self._initialization_active['target'])))
        return snapshot
