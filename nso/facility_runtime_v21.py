"""Joint planning with fixed-ROI mapping progress; observed motion guards unchanged.

V20 candidates, scoring, and commitment rules are preserved. Only the shared
coverage-budget ledger changes to V21 known-cell units. The ROI fraction is an
uncalibrated planning proxy, never evaluator coverage or motion permission.
"""
from copy import deepcopy
from time import perf_counter
import numpy as np
from nso.cpu_four_modules_v16 import CPUFourModulesV16, axis_components_v16
from nso.facility_runtime_v19 import FacilityBackendV19, FacilityRuntimeV19
from nso.recovery_runtime_v14 import RecoveryRuntimeV14
from nso.axis_history_view_v16 import AxisHistoryViewV16
from nso.hierarchical_options_v16_3 import generate_options
from nso.cpu_sensor_contract_v10 import digest, json_value
from nso.facility_candidates_v20 import ObservedRouteSpaceV20
from nso.coverage_budget_v21 import ObservedCoverageLedgerV21, scan_hit_mask
from utils.grid_geometry import visible_mask


class FacilityBackendV21(FacilityBackendV19):
    def __init__(self, args, num_scenes, shape):
        if getattr(args, 'cpu_score_mode', None) not in ('N', 'G'):
            raise ValueError('V21 common capacity probe supports N and G only')
        CPUFourModulesV16.__init__(self, args, num_scenes, shape)
        self.capabilities['v21'] = 'fixed public ROI known-cell budget; G joint planning and semantics unproven'

    def start_scene(self, scene_idx, **kwargs):
        super().start_scene(scene_idx, **kwargs)
        s = self.scenes[scene_idx]; m = s['mapper']; p = s['packet']
        ledger = ObservedCoverageLedgerV21(m.shape, s['return_anchor'][:2],
            m.config.robot_radius_m / m.config.resolution_m,
            prefix_actions=int(getattr(self.args, 'cpu_v20_replan_interval', 5)))
        ledger.start(m.belief, action_id=p.action_id,
            radar_hits=scan_hit_mask(p.scan, m.shape, m.config.resolution_m))
        s['coverage_v21'] = ledger
        s['v21_pending_prediction'] = None
        self._record(scene_idx, 'IGCR', 'coverage_bootstrap_v21',
            dict(task_shape=m.shape, target=.8, coverage_proxy='fixed_task_roi_known_fraction'), ledger.snapshot())

    def assess_action(self, scene_idx, action):
        result = super().assess_action(scene_idx, action)
        if result['allowed']:
            s = self.scenes[scene_idx]; m = s['mapper']; c = m.config
            # Only a prediction: the original guard already checked this step.
            s['v21_pending_prediction'] = visible_mask(m.belief == 1,
                tuple(result['next_pose'][:2]), 0, int(c.max_depth_m / c.resolution_m), 360.) & (m.belief == -1)
        return result

    def compute_reward(self, scene_idx):
        value, parts = super().compute_reward(scene_idx)
        s = self.scenes[scene_idx]; p = s['packet']; m = s['mapper']
        option = s.get('execution_option')
        intent = bool(option and option.get('v21_coverage_intent', option['group'].startswith('coverage_')))
        event = s['coverage_v21'].observe(m.belief, action_id=p.action_id,
            coverage_intent=intent, predicted_mask=s.pop('v21_pending_prediction', None),
            radar_hits=scan_hit_mask(p.scan, m.shape, m.config.resolution_m))
        self._record(scene_idx, 'IGCR', 'observe_unique_coverage_v21',
            dict(action_id=p.action_id, coverage_intent=intent), event)
        return value, {**parts, 'coverage_v21': s['coverage_v21'].snapshot()}

    def route_space(self, scene_idx):
        s = self.scenes[scene_idx]; p = s['packet']
        return ObservedRouteSpaceV20(s['mapper'], p.position, p.heading,
            s['return_anchor'], s['ledger'].remaining_budget)

    def select_target(self, scene_idx):
        started = perf_counter(); s = self.scenes[scene_idx]; p = s['packet']; m = s['mapper']
        if s['graph_version'] != m.frames or s['semantic_version'] != m.frames:
            raise RuntimeError('V21 planning requires current four-module observations')
        if s.get('region_continuation') is not None:
            raise RuntimeError('V21 capacity probe does not install two-stage semantic interventions')
        space = self.route_space(scene_idx)
        coverage, caudit = space.coverage_candidates(slots=int(getattr(self.args, 'cpu_v20_coverage_slots', 8)))
        attempted = s['gain'].attempted_camera_mask(m, s['ledger'].planning_camera_poses())
        original, oaudit = generate_options(AxisHistoryViewV16(m, s['axis_frames']),
            p.position, p.heading, s['ledger'].remaining_budget, s['return_anchor'],
            max_candidates=5 + 5 * len(s['assets']), coverage_strategy='total_diverse',
            coverage_slots=4, attempted_camera_mask=attempted)
        # Preserve all measured quality roles in both N/G candidate pools.
        routes = coverage + [r for r in original if not r['group'].startswith('coverage_')]
        region = s['graph']['current_region']; anchor_region = s['graph']['anchor_region']
        routes = [r for r in routes if region > 0 and region == anchor_region
            == int(s['graph']['regions'][tuple(r['pose'][:2])])]
        for i, route in enumerate(routes): route['candidate_id'] = i
        mode = self.args.cpu_score_mode
        if mode == 'G' and routes:
            _, rows = self._scores_v10_1(s, routes, attempted)
        else:
            rows = [dict(candidate_id=r['candidate_id'], quality_score_evaluated=False) for r in routes]
        scores = {'N': [], 'G': []}
        for route, row in zip(routes, rows):
            mask = space.route_mask(route)
            count = int(mask.sum()); cost = route['outbound_cost']
            budget = s['coverage_v21'].assess_route(route, mask, s['ledger'].remaining_budget)
            coverage_rate = count / max(1, m.belief.size * cost)
            terms = row.get('v19_task_proxy', {})
            quality = terms.get('observed_direction_term', 0.) + terms.get('observed_precision_term', 0.)
            scores['N'].append(float(coverage_rate))
            scores['G'].append(float(coverage_rate + quality / max(1, cost)))
            row['v21_coverage'] = dict(unique_predicted_unknown_cells=count,
                predicted_mask_sha256=digest(mask), rate_per_task_cell_per_action=coverage_rate,
                coverage_budget=budget, quality_score_divided_by_paid_outbound=True,
                class_used=False, evaluation_truth_used=False)
        coverage_ids = [i for i, r in enumerate(routes) if r['group'].startswith('coverage_') and scores['N'][i] > 0]
        admitted = [i for i in range(len(routes)) if rows[i]['v21_coverage']['coverage_budget']['allowed']
                    and scores['G'][i] > 0]
        pressure = mode == 'N' or not admitted
        allowed = coverage_ids if pressure else admitted
        score_mode = 'N' if pressure else 'G'
        index = min(allowed, key=lambda i: (-scores[score_mode][i], routes[i]['outbound_cost'],
                    routes[i]['cost'], routes[i]['candidate_id'])) if allowed else None
        s['selected'] = None if index is None else deepcopy(routes[index])
        s['plans'] += 1
        if s['selected'] is not None:
            s['selected'].update(option_id=f'{s["scene_id"]}/{s["episode_id"]}/plan-{s["plans"]}',
                selection_call_id=len(self.calls) + 1, selected_map_version=m.frames,
                selected_feedback_version=s['feedback']['feedback_version'], parent_region=region,
                v21_coverage_intent=pressure or routes[index]['group'].startswith('coverage_'),
                route_revision=0, route_map_version=m.frames)
        result = dict(selected=s['selected'], candidates=routes, scores=scores, score_audit=rows,
            candidate_audit=dict(coverage=caudit, observed_quality_roles=oaudit), mode=mode,
            effective_objective=score_mode, coverage_pressure=pressure,
            quality_budget_admitted_candidates=admitted, coverage_state=s['coverage_v21'].snapshot(),
            plan_number=s['plans'], planner_revision='v21_fixed_roi_known_budget')
        s['last_selection'] = json_value(result)
        self._record(scene_idx, 'STGHP', 'select_topo_target',
            dict(remaining_budget=s['ledger'].remaining_budget, semantic_version=s['semantic_version'],
                graph_version=s['graph_version']), result, started)
        return None if s['selected'] is None else np.asarray(s['selected']['pose'][:2], int)

    def summary(self, scene_idx):
        result = super().summary(scene_idx)
        s = self.scenes[scene_idx]
        result['v21_coverage'] = None if s is None else s['coverage_v21'].snapshot()
        result['region_continuation'] = 'not installed by common N/G capacity probe'
        return result


class FacilityRuntimeV21(FacilityRuntimeV19):
    def choose_goal(self, scene_idx, proposed_local, bounds):
        result = super().choose_goal(scene_idx, proposed_local, bounds)
        s = self.states[scene_idx]
        if s is not None: s['v21_last_route_update'] = s['packet'].action_id
        return result

    def next_local_action(self, scene_idx):
        s = self.states[scene_idx]
        if s is None or s['closed']: return None
        if s['pending'] is not None: raise RuntimeError('consume paid observation before V21 route update')
        interval = int(getattr(self.args, 'cpu_v20_replan_interval', 5))
        if interval < 1: raise ValueError('positive route update interval required')
        p = s['packet']; backend = self.components._cpu_backend; b = backend.scenes[scene_idx]
        if s['phase'] == 'outbound' and s['active_actions'] and s['option'] is not None and (
                p.action_id - s.get('v21_last_route_update', p.action_id) >= interval):
            started = perf_counter(); old = s['option']; space = backend.route_space(scene_idx)
            route = space.refresh(old)
            assessment = None; reason = 'target_no_longer_affordable_on_observed_map'
            if route is not None:
                mask = space.route_mask(route)
                assessment = b['coverage_v21'].assess_route(route, mask, b['ledger'].remaining_budget)
                if old.get('v21_coverage_intent'):
                    retain = bool(mask.any())
                    reason = 'same_coverage_target_updated' if retain else 'no_remaining_unique_coverage_gain'
                else:
                    retain = assessment['allowed']
                    reason = 'same_quality_target_updated' if retain else 'reserve_remaining_coverage_budget'
                if not retain: route = None
            event = dict(action_id=p.action_id, option_id=old['option_id'],
                selection_call_id=old['selection_call_id'], old_target=old['pose'],
                old_remaining_outbound=len(s['active_actions']), reason=reason,
                coverage_budget=assessment, target_retained=route is not None,
                new_remaining_outbound=None if route is None else route['outbound_cost'])
            if route is None:
                s.update(option=None, active_actions=[], phase='planning')
                b['selected'] = None; b['region_continuation'] = None
            else:
                route.update(route_revision=old.get('route_revision', 0) + 1,
                    route_map_version=s['mapper'].frames)
                s['option'] = route; s['active_actions'] = list(route['outbound_actions'])
                b['selected'] = deepcopy(route)
            s['v21_last_route_update'] = p.action_id
            backend._record(scene_idx, 'STGHP', 'refresh_observed_route_v21',
                dict(map_version=s['mapper'].frames, current_pose=[*p.position, p.heading]), event, started)
            self.audit.append(dict(event='v21_route_update', **event))
        # The original guard alone registers the next paid action prediction.
        return RecoveryRuntimeV14.next_local_action(self, scene_idx)

    def install_pilot_option(self, option_name):
        raise RuntimeError('V21 first validates common coverage; A/B requires a later explicit protocol')


def facility_components_v21(args, shape):
    comp = axis_components_v16(args, shape)
    comp._cpu_backend = FacilityBackendV21(args, 1, shape)
    comp.capabilities = dict(comp._cpu_backend.capabilities)
    return comp
