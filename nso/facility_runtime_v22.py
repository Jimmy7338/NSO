"""Early observed coverage-route interventions with common V21-N continuation.

N/A/B are experiment arms, not a deployed semantic policy. All three choices
are constructed from the same initial observations and positive-coverage pool.
Class votes are logged but never rank candidates. The old motion/return guard
and five-action target refresh remain authoritative, including cancellation.
"""
from copy import deepcopy
from time import perf_counter
import numpy as np

from nso.cpu_four_modules_v16 import axis_components_v16
from nso.cpu_sensor_contract_v10 import digest, json_value
from nso.facility_runtime_v21 import FacilityBackendV21, FacilityRuntimeV21
from nso.observed_precision_response_v19 import ObservedPrecisionResponseV19


def initial_option_map(routes, n_scores, assets, precision_rows):
    """Select observed A/B routes without classes, hidden geometry or rewards.

    A/B are the left/right marked observed clusters. Their route association
    maximizes the existing per-asset precision proxy; no new weights are fit.
    N uses precisely the V21-N ranking and tie breaks. Coverage, paid outbound
    and return paths are the original shared candidates, never relabelled NBVs.
    """
    if len(routes) != len(n_scores) or len(routes) != len(precision_rows):
        raise ValueError('One N score and precision response per common candidate required')
    if not np.isfinite(n_scores).all():
        raise ValueError('Finite common N scores required')
    ids = [i for i, r in enumerate(routes)
           if r['group'].startswith('coverage_') and n_scores[i] > 0]
    if not ids:
        raise ValueError('No positive shared coverage route at initial boundary')
    targets = sorted([i for i, a in enumerate(assets) if a['marked_points'] > 0],
                     key=lambda i: tuple(assets[i]['aabb_center'][:2]))
    if len(targets) != 2:
        raise ValueError('Exactly two marked observed instances required')
    for i in ids:
        route = routes[i]
        if (route['candidate_id'] != i or route['outbound_cost'] <= 0
                or len(route['outbound_actions']) != route['outbound_cost']
                or len(route['return_actions']) != route['return_cost']
                or route['cost'] != route['outbound_cost'] + route['return_cost']):
            raise ValueError('Invalid paid shared coverage route')
        values = precision_rows[i]['per_asset_expected_precision_change']
        if len(values) != len(assets) or not np.isfinite(values).all():
            raise ValueError('Finite precision response per observed instance required')
    tie = lambda i: (routes[i]['outbound_cost'], routes[i]['cost'], routes[i]['candidate_id'])
    choices = {'N': min(ids, key=lambda i: (-n_scores[i], *tie(i)))}
    for name, ai in zip(('A', 'B'), targets):
        best = min(ids, key=lambda i: (
            -precision_rows[i]['per_asset_expected_precision_change'][ai], *tie(i)))
        if precision_rows[best]['per_asset_expected_precision_change'][ai] <= 0:
            raise ValueError('No positive observed precision opportunity for ' + name)
        choices[name] = best
    return choices, targets


class FacilityBackendV22(FacilityBackendV21):
    def __init__(self, args, num_scenes, shape):
        if getattr(args, 'cpu_score_mode', None) != 'N':
            raise ValueError('V22 information probe requires common N continuation')
        if getattr(args, 'cpu_v22_first_option', None) not in ('N', 'A', 'B'):
            raise ValueError('Declare the initial N/A/B intervention')
        super().__init__(args, num_scenes, shape)
        self.capabilities['v22'] = 'observed early coverage-route information probe; no trained semantic policy'

    def select_target(self, scene_idx):
        state = self.scenes[scene_idx]
        initial = state['packet'].action_id == 0 and state['plans'] == 0
        nominal_target = super().select_target(scene_idx)
        if not initial:
            return nominal_target
        started = perf_counter()
        proposal = deepcopy(state['last_selection'])
        if proposal['selected'] is None:
            raise ValueError('Initial common N proposal must exist')
        routes = proposal['candidates']
        response = ObservedPrecisionResponseV19(state['mapper'], state['assets'],
            intrinsic=state['packet'].frame.intrinsic, max_points_per_asset=128,
            occlusion_tolerance_m=.12)
        precision = [response.score_route(r) if r['group'].startswith('coverage_') else {}
                     for r in routes]
        choices, targets = initial_option_map(routes, proposal['scores']['N'], state['assets'], precision)
        nominal = proposal['selected']
        if choices['N'] != nominal['candidate_id']:
            raise RuntimeError('Declared N no longer matches the common N proposal')
        requested = self.args.cpu_v22_first_option
        chosen = choices[requested]
        selected = deepcopy(routes[chosen])
        for key in ('option_id', 'selected_map_version', 'selected_feedback_version',
                    'parent_region', 'v21_coverage_intent', 'route_revision', 'route_map_version'):
            selected[key] = nominal[key]
        selected.update(selection_call_id=len(self.calls) + 1,
            v22_first_option=requested,
            v22_observed_target_index=None if requested == 'N' else targets[requested == 'B'])
        if not selected['v21_coverage_intent']:
            raise RuntimeError('Initial arms must share the original coverage task role')
        receipt = dict(requested_option=requested, option_map=choices,
            chosen_candidate_id=chosen, nominal_N_candidate_id=choices['N'],
            common_pool_sha256=digest(routes), action_id=0,
            initial_observation_only=True, common_continuation='V21_N',
            class_used_for_ranking=False, evaluation_truth_used=False,
            observed_targets=[dict(role=role, asset_index=ai,
                center=state['assets'][ai]['aabb_center'],
                class_vote=state['assets'][ai]['class_vote'],
                marked_points=state['assets'][ai]['marked_points'])
                for role, ai in zip(('A', 'B'), targets)],
            association_scores=[dict(candidate_id=i, **precision[i])
                for i, route in enumerate(routes) if route['group'].startswith('coverage_')],
            original_coverage_and_motion_rules=True,
            intermediate_proposal_executed=False, trained=False,
            semantic_policy_deployed=False)
        actual = {**proposal, 'selected': selected, 'v22_first_choice': receipt,
                  'effective_objective': 'declared_initial_coverage_option',
                  'planner_revision': 'v22_initial_coverage_then_common_N'}
        state['selected'] = deepcopy(selected)
        state['v22_first_choice'] = json_value(receipt)
        state['last_selection'] = json_value(actual)
        self._record(scene_idx, 'STGHP', 'select_initial_coverage_v22',
            dict(nominal_selection_call_id=nominal['selection_call_id'],
                 action_id=0, requested_option=requested), actual, started)
        return np.asarray(selected['pose'][:2], int)

    def summary(self, scene_idx):
        result = super().summary(scene_idx)
        state = self.scenes[scene_idx]
        result['v22_first_choice'] = None if state is None else state.get('v22_first_choice')
        return result


class FacilityRuntimeV22(FacilityRuntimeV21):
    """Inherit paid action guards and target refresh without overriding them."""


def facility_components_v22(args, shape):
    components = axis_components_v16(args, shape)
    components._cpu_backend = FacilityBackendV22(args, 1, shape)
    components.capabilities = dict(components._cpu_backend.capabilities)
    return components
