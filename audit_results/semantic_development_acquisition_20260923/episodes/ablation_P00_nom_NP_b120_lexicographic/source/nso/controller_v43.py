"""Bounded observed-only CPU composition of the four ANS module interfaces.

This is a functional greedy controller, not a trained policy or evidence of
performance. Geometry and public candidate generation are identical for G/S;
only the observed instance structure prior can affect inspection utility.
"""
from copy import deepcopy
import hashlib
import json
import math

import numpy as np

from nso.instance_belief_v40 import PaidRGBDObservationV40
from nso.observed_instances_v41 import ObservedInstancesV41
from nso.observed_mapper_v42 import ObservedMapperV42
from nso.observed_residual_v41 import ObservedResidualV41
from nso.observed_safety_v43 import ObservedSafetyV43
from nso.primitive_navigation_v41 import (
    PrimitiveStateV41, PublicPrimitiveGraphV41, ReturnAwarePrimitiveRouterV41,
)
from nso.public_navigation_v43 import select_candidate_states_v43
from nso.surface_evaluation_v40 import CandidateViewV40
from nso.view_quality_v42 import ViewQualityPredictorV42


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _array_digest(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(f'{a.dtype.str}:{a.shape}:'.encode()+a.tobytes()).hexdigest()


def _number(value, name, lo=0., hi=100.):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not lo <= value <= hi:
        raise ValueError(name+' outside finite bounded interval')
    return float(value)


def _integer(value, name, lo, hi):
    if type(value) is not int or not lo <= value <= hi:
        raise ValueError(name+' outside bounded integer interval')
    return value


def _state_receipt(state):
    return dict(node=state.node, heading=state.heading)


class ANSControllerV43:
    """Accept one fused paid packet, then choose one primitive or terminal action.

    Caller updates its mapper exactly once before ``accept``. On any exception
    after acceptance starts, this controller is poisoned and must be discarded;
    it never permits retries against partially updated module histories.
    """
    def __init__(self, graph, *, home, budget, palette, structure_names,
                 class_structure_prior, mode='S', geometry_prior=None,
                 maximum_candidates=32, maximum_views_per_instance=8,
                 maximum_instances=8, discovery_weight=1., inspection_weight=1.,
                 uncertainty_penalty=0., discovery_radius_m=2.,
                 no_progress_patience=12, robot_radius_m=.2):
        if type(graph) is not PublicPrimitiveGraphV41:
            raise TypeError('strict validated public navigation graph required')
        graph.validate_state(home)
        budget = _integer(budget, 'budget', 1, 511)
        maximum_candidates = _integer(maximum_candidates, 'maximum_candidates', 1, 32)
        maximum_views_per_instance = _integer(maximum_views_per_instance, 'maximum_views_per_instance', 1, 8)
        maximum_instances = _integer(maximum_instances, 'maximum_instances', 1, 8)
        no_progress_patience = _integer(no_progress_patience, 'no_progress_patience', 1, 511)
        discovery_weight = _number(discovery_weight, 'discovery_weight')
        inspection_weight = _number(inspection_weight, 'inspection_weight')
        uncertainty_penalty = _number(uncertainty_penalty, 'uncertainty_penalty', hi=10.)
        discovery_radius_m = _number(discovery_radius_m, 'discovery_radius_m', .25, 4.)
        if discovery_weight+inspection_weight <= 0:
            raise ValueError('at least one positive task utility weight required')
        self._graph = deepcopy(graph)
        self._router = ReturnAwarePrimitiveRouterV41(self._graph, home=deepcopy(home), budget=budget)
        self._ledger = ObservedInstancesV41(palette=deepcopy(palette), structure_names=tuple(structure_names),
            class_structure_prior=deepcopy(class_structure_prior), mode=mode,
            geometry_prior=deepcopy(geometry_prior), maximum_instances=maximum_instances)
        self._residual = ObservedResidualV41()
        self._predictor = ViewQualityPredictorV42(maximum_instances=maximum_instances)
        self._safety = ObservedSafetyV43(self._graph, robot_radius_m=robot_radius_m,
                                       unknown_space_policy='provided_navigation_prior')
        self._configuration = dict(mode=mode, budget=budget, maximum_candidates=maximum_candidates,
            palette={name: list(color) for name, color in self._ledger.palette.items()},
            structure_names=list(self._ledger.names),
            class_structure_prior={name: weights.tolist() for name, weights in self._ledger.class_priors.items()},
            geometry_prior=self._ledger.geometry_prior.tolist(),
            maximum_views_per_instance=maximum_views_per_instance, maximum_instances=maximum_instances,
            discovery_weight=discovery_weight, inspection_weight=inspection_weight,
            uncertainty_penalty=uncertainty_penalty, discovery_radius_m=discovery_radius_m,
            no_progress_patience=no_progress_patience, robot_radius_m=float(robot_radius_m),
            public_graph_sha256=graph.input_sha256, forced_prefix_actions=0,
            utility_units='weighted 2D unknown-cell m2 plus 3D predicted-new-surface m2',
            weights_status='declared design constants; no fitted calibration or performance claim',
            uncertainty_status='posterior nominal-area dispersion; no learned or calibrated uncertainty',
            instance_covariance_assumption='independent latent structures for diagnostic variance sum',
            discovery_revisit_rule='zero at any previously paid node; full-circle planar lidar viewpoint already paid')
        self._configuration_sha256 = _digest(self._configuration)
        self._paid_states = set()
        self._visited_nodes = set()
        self._camera = None
        self._last_observation_sha256 = None
        self._accepted_packet_hashes = []
        self._last_accept = None
        self._last_choose = None
        self._no_progress = 0
        self._return_latched = False
        self._poisoned = False
        self._terminal = False

    def _ready(self):
        if self._poisoned:
            raise RuntimeError('controller transaction failed; discard controller and episode')
        if self._terminal:
            raise RuntimeError('controller is terminal; start a new controller for another episode')

    def accept(self, observation, mapper, *, execution_outcome='success'):
        self._ready()
        if type(observation) is not PaidRGBDObservationV40 or type(mapper) is not ObservedMapperV42:
            raise TypeError('strict paid packet and ObservedMapperV42 required')
        # Revalidate immutable copies before any controller mutation. The
        # external mapper has already integrated this packet; caller owns its
        # recovery if it supplied an invalid execution outcome or wrong pose.
        obs = PaidRGBDObservationV40.from_mapping({key: getattr(observation, key)
            for key in PaidRGBDObservationV40.__dataclass_fields__})
        mapping = mapper.snapshot()
        belief, observed = mapper.occupancy_arrays()
        if mapping['backend_poisoned'] or mapping['frames'] != obs.paid_step+1 or not mapping['receipts']:
            raise ValueError('mapper must have integrated exactly the current paid history')
        if [row['observation_sha256'] for row in mapping['receipts'][:-1]] != self._accepted_packet_hashes:
            raise ValueError('mapper history must equal previously accepted paid packets')
        map_receipt = mapping['receipts'][-1]
        if (map_receipt['observation_sha256'] != obs.sha256() or map_receipt['frame_id'] != obs.frame_id
                or map_receipt['paid_step'] != obs.paid_step
                or mapping['occupancy_sha256'] != _array_digest(belief)):
            raise ValueError('current mapper frame or occupancy hash mismatch')
        if mapping['near_m'] != .1 or mapping['far_m'] != 4.:
            raise ValueError('controller requires the frozen axial clipping contract [0.1,4.0]')
        camera = dict(intrinsic=obs.intrinsic.tolist(), width=obs.depth_m.shape[1], height=obs.depth_m.shape[0])
        if self._camera is not None and camera != self._camera:
            raise ValueError('camera calibration/image dimensions must remain fixed within an episode')
        try:
            routing = self._router.accept(obs, execution_outcome=execution_outcome)
            association = self._ledger.observe(obs)
            residual = self._residual.observe(obs, association['accepted'])
            feedback = [self._ledger.apply_geometry_feedback(row['instance_id'], frame_id=obs.frame_id,
                observation_sha256=obs.sha256(), log_likelihoods=row['log_likelihoods'])
                for row in residual['results'] if row['accepted']]
            view_evidence = self._predictor.observe(obs, association['accepted'])
            safety = self._safety.update(mapping, (belief, observed), self._router.state)
            self._paid_states.add(self._router.state)
            self._visited_nodes.add(self._router.state.node)
            self._camera = deepcopy(camera)
            progress = (map_receipt['newly_known_cells'] > 0 or any(
                row['novel_support_voxels'] > 0 and not row['duplicate_measurement']
                for row in association['accepted']))
            self._no_progress = 0 if progress else self._no_progress+int(obs.paid_step > 0)
            self._last_observation_sha256 = obs.sha256()
            self._accepted_packet_hashes.append(obs.sha256())
            self._last_accept = dict(schema='v43.controller_accept.v1', paid_step=obs.paid_step,
                frame_id=obs.frame_id, observation_sha256=obs.sha256(), execution_outcome=execution_outcome,
                configuration_sha256=self._configuration_sha256, mapper_receipt=map_receipt,
                routing=routing, association=association, observed_residual=residual,
                geometry_feedback=feedback, view_evidence=view_evidence, safety=safety,
                no_progress_paid_actions=self._no_progress,
                progress_rule='new measured occupancy cell or newly associated support voxel',
                module_order=['RPN-UQ paid-action validation', 'OV-SDF observed-instance belief',
                              'IGCR measured-depth correction', 'STGHP observed-view cache', 'RPN-UQ observed safety'])
        except Exception:
            self._poisoned = True
            raise
        return deepcopy(self._last_accept)

    def _candidate_view(self, state):
        yaw = state.heading*math.pi/6.
        transform = np.eye(4)
        transform[:3, :3] = [[math.sin(yaw), 0., math.cos(yaw)],
                            [-math.cos(yaw), 0., math.sin(yaw)], [0., -1., 0.]]
        transform[:3, 3] = [*self._graph.positions[state.node], self._graph.camera_height_m]
        return CandidateViewV40(np.asarray(self._camera['intrinsic']), transform,
            self._camera['width'], self._camera['height'], view_id=f'{state.node}:{state.heading}')

    @staticmethod
    def _instance_candidate_order(instance, states, views):
        """Observed-anchor alignment only, with no class or posterior access."""
        anchor = np.asarray(instance['anchor_world_m'], dtype=float)
        order = []
        for state, view in zip(states, views):
            point = (anchor-view.world_from_camera[:3, 3])@view.world_from_camera[:3, :3]
            distance = float(np.linalg.norm(point))
            alignment = float(np.arctan2(np.linalg.norm(point[:2]), point[2]))
            order.append(((alignment, distance, state.node, state.heading), state, view))
        return [(state, view) for _, state, view in sorted(order, key=lambda row: row[0])]

    def choose(self):
        self._ready()
        if self._camera is None or self._router.pending is not None:
            raise ValueError('accept current paid packet before choosing exactly one action')
        # Check the occupied current footprint before forecasting any goal.
        stationary_guard = self._safety.guard_action('observe')
        if not stationary_guard['allowed']:
            self._terminal = True
            self._last_choose = dict(schema='v43.controller_decision.v1', action='blocked',
                reason='observed_current_footprint_conflict', paid_step=self._router.step,
                safety_guard=stationary_guard, candidate_utilities=[], forecasts=[],
                source_observation_sha256=self._last_observation_sha256,
                configuration_sha256=self._configuration_sha256)
            return deepcopy(self._last_choose)
        if self._no_progress >= self._configuration['no_progress_patience']:
            self._return_latched = True
        if self._return_latched:
            states, pool_receipt = [], dict(reason='no_observed_progress_watchdog_return')
        else:
            states, pool_receipt = select_candidate_states_v43(self._graph, self._router.state,
                paid_camera_states=tuple(self._paid_states), limit=self._configuration['maximum_candidates'])
        views = [self._candidate_view(state) for state in states]
        discovery_raw = self._safety.discovery_utilities(states, radius_m=self._configuration['discovery_radius_m'])
        means, variances = {state: 0. for state in states}, {state: 0. for state in states}
        forecasts, allocations = [], []
        for instance in self._ledger.snapshot()['instances']:
            selected = self._instance_candidate_order(instance, states, views)[:self._configuration['maximum_views_per_instance']]
            allocations.append(dict(instance_id=instance['instance_id'], candidates=[_state_receipt(s) for s, _ in selected],
                                    rule='ascending observed-anchor angular offset, distance, node, heading; no semantic input'))
            if not selected:
                continue
            forecast = self._predictor.forecast(instance, [view for _, view in selected])
            forecasts.append(forecast)
            probabilities = np.asarray(forecast['structure_probabilities'])
            for (state, _), row in zip(selected, forecast['candidates']):
                area = np.asarray(row['structure_new_surface_area_m2'])
                mean = float(row['expected_new_surface_area_m2'])
                means[state] += mean
                variances[state] += float(probabilities@((area-mean)**2))
        utility_rows, utilities = [], {}
        for state in states:
            repeated_xy = state.node in self._visited_nodes
            discovery = 0. if repeated_xy else float(discovery_raw[state])
            spread = math.sqrt(max(0., variances[state]))
            inspection = max(0., means[state]-self._configuration['uncertainty_penalty']*spread)
            utility = self._configuration['discovery_weight']*discovery+self._configuration['inspection_weight']*inspection
            utilities[state] = float(utility)
            utility_rows.append(dict(**_state_receipt(state), discovery_unknown_area_m2=discovery,
                unmasked_discovery_unknown_area_m2=float(discovery_raw[state]), discovery_xy_already_paid=repeated_xy,
                inspection_expected_new_area_m2=means[state], inspection_area_std_m2=spread,
                inspection_after_uncertainty_penalty_m2=inspection, combined_design_utility=utility))
        try:
            routing = self._router.choose(utilities)
            guard = self._safety.guard_action(routing['action'])
            action = guard['action']
            if not guard['allowed']:
                # Do not leave an apparently retryable pending action after a
                # safety veto. Only a new episode may resume this controller.
                self._terminal = True
                self._router.pending = None
            if action in ('stop', 'blocked'):
                self._terminal = True
            self._last_choose = dict(schema='v43.controller_decision.v1', action=action,
                reason=guard['reason'] if not guard['allowed'] else routing['reason'],
                paid_step=self._router.step, routing=routing, safety_guard=guard,
                configuration_sha256=self._configuration_sha256, configuration=deepcopy(self._configuration),
                source_observation_sha256=self._last_observation_sha256,
                candidate_pool=pool_receipt, instance_candidate_allocations=allocations,
                candidate_utilities=utility_rows, forecasts=forecasts,
                discovery=self._safety.discovery_receipt(), watchdog_return_latched=self._return_latched,
                modules=dict(OV_SDF='observed-instance posterior', STGHP='shared nominal visible-area expectation',
                    RPN_UQ='posterior-area dispersion, observed safety and primitive return reserve; CPU surrogate',
                    IGCR='already-paid depth residual feedback'),
                calibrated_quality=False, learned_uncertainty=False, forced_prefix_actions=0,
                actual_future_sensor_rendered=False, ground_truth_scene_input=False)
        except Exception:
            self._poisoned = True
            raise
        return deepcopy(self._last_choose)

    def snapshot(self):
        return deepcopy(dict(schema='v43.controller_state.v1', configuration=self._configuration,
            configuration_sha256=self._configuration_sha256, paid_step=self._router.step,
            current_state=None if self._router.state is None else _state_receipt(self._router.state),
            pending_action=None if self._router.pending is None else self._router.pending['action'],
            poisoned=self._poisoned, terminal=self._terminal, return_latched=self._return_latched,
            no_progress_paid_actions=self._no_progress, observed_instances=self._ledger.snapshot(),
            geometry=self._ledger.geometry_snapshot(), paid_camera_states=[_state_receipt(s)
                for s in sorted(self._paid_states, key=lambda s: (s.node, s.heading))],
            last_observation_sha256=self._last_observation_sha256))
