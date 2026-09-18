"""V17 shared visibility scoring and explicit two-stage observed-region options."""
from copy import deepcopy
from time import perf_counter
from nso.cpu_four_modules_v10 import _OBSERVED_GAIN_REVISIONS
from nso.axis_history_view_v16 import AxisHistoryViewV16
from nso.cpu_sensor_contract_v10 import digest, json_value
from nso.hierarchical_options_v16_3 import generate_options
import numpy as np
from nso.cpu_four_modules_v16_3 import CPUFourModulesV16_3
from nso.cpu_four_modules_v16 import axis_components_v16
from nso.visibility_corrected_scores_v17 import correct_scores
from nso.observed_region_evidence_v17 import observed_regions
from nso.observed_continuation_v17 import select_continuation


class CPUFourModulesV17(CPUFourModulesV16_3):
    def _scores_v10_1(self, state, routes, attempted):
        scores, rows = super()._scores_v10_1(state, routes, attempted)
        if state['packet'].action_id < getattr(self.args, 'cpu_visibility_start_action', 0):
            return scores, rows
        support = state['planning_direction_support'] or [[0.] * 8 for _ in state['assets']]
        corrected, audit = correct_scores(state['mapper'], state['assets'], support,
            routes, scores, rows, novelty_floor=getattr(self.args, 'cpu_measured_novelty_floor', .25))
        combined = []
        for old, new in zip(rows, audit):
            combined.append({**old, **new, 'incremental_aperture': new['corrected_aperture'],
                'visibility_revision': 'observed_mesh_v17', 'legacy_unmasked_audit': old})
        return corrected, combined

    def arm_region_continuation(self, scene_idx, route):
        state = self.scenes[scene_idx]
        if state.get('region_continuation') is not None:
            raise RuntimeError('a committed first stage is already pending')
        if state['semantic_version'] != state['mapper'].frames:
            raise RuntimeError('region anchor requires current semantic state')
        regions = observed_regions(state['mapper'], state['assets'])
        anchors = [r for r in regions if r['asset_index'] is not None]
        anchor = next((r for r in anchors if r['asset_index'] == route.get('asset_index')), None)
        if anchor is None:
            raise ValueError('two-stage first route needs an observed asset anchor')
        state['region_continuation'] = deepcopy(dict(anchor=anchor,
            peers=[r for r in anchors if r is not anchor], expected_pose=route['pose'],
            first_candidate_id=route['candidate_id'], armed_action_id=state['packet'].action_id))

    def select_target(self, scene_idx):
        if self.scenes[scene_idx]['packet'].action_id < getattr(self.args, 'cpu_viewpoint_start_action', 0):
            return super().select_target(scene_idx)
        self.args.cpu_max_candidates = 5 + 5 * len(self.scenes[scene_idx]['assets'])
        started = perf_counter(); s = self.scenes[scene_idx]; p = s["packet"]
        if s["graph_version"] != s["mapper"].frames or s["semantic_version"] != s["mapper"].frames:
            raise RuntimeError("global planning requires current OV-SDF and STGHP state")
        history = s["ledger"].planning_camera_poses()
        attempted = s["gain"].attempted_camera_mask(s["mapper"], history)
        routes, audit = generate_options(AxisHistoryViewV16(s["mapper"], s['axis_frames']), p.position, p.heading,
            s["ledger"].remaining_budget, s["return_anchor"],
            max_candidates=int(getattr(self.args, "cpu_max_candidates", 12)),
            coverage_strategy=("total_diverse" if self.planner_revision in _OBSERVED_GAIN_REVISIONS
                               and self.planner_revision != "v11_1"
                               else "rate_single"),
            coverage_slots=int(getattr(self.args, "cpu_coverage_slots", 4)),
            attempted_camera_mask=(attempted if self.planner_revision in _OBSERVED_GAIN_REVISIONS
                                   else None))
        graph = s["graph"]
        routes = [r for r in routes if graph["current_region"] > 0
                  and graph["current_region"] == graph["anchor_region"]
                  == int(graph["regions"][tuple(r["pose"][:2])])]
        mode = getattr(self.args, "cpu_score_mode", "S")
        if mode not in ("N", "G", "O", "S", "X", "M", "L", "K", "Y", "Q"):
            raise ValueError("unknown CPU semantic ablation")
        modern = self.planner_revision in _OBSERVED_GAIN_REVISIONS
        if routes and self.planner_revision in ("v11", "v11_1"):
            scores, rows = self._scores_v11(s, routes)
        else:
            score_method = self._scores_v10_1 if modern else self._scores
            scores, rows = (score_method(s, routes, attempted) if routes and modern
                            else score_method(s, routes) if routes
                            else ({m: [] for m in ("N","G","O","S","X","M")}, []))
        index = min(range(len(routes)), key=lambda i: (-scores[mode][i], routes[i]["cost"],
                                                      routes[i]["candidate_id"])) if routes else None
        continuation = None
        pending = s.get('region_continuation')
        if pending is not None:
            if p.action_id <= pending['armed_action_id']:
                raise RuntimeError('cannot continue before first-stage observation')
            if list((*p.position, p.heading)) == pending['expected_pose']:
                continuation = select_continuation(pending['anchor'], observed_regions(s['mapper'], s['assets']),
                    pending['peers'], routes, scores['G'], rows)
                index = continuation['candidate_index']
            else:
                valid = [i for i, value in enumerate(scores['G']) if value > 0]
                index = min(valid, key=lambda i: (-scores['G'][i], routes[i]['cost'], routes[i]['candidate_id'])) if valid else None
                continuation = dict(status='common_geometry_fallback' if valid else 'request_guarded_return',
                    reason='first_stage_not_reached', candidate_index=index,
                    candidate_id=None if index is None else routes[index]['candidate_id'],
                    target_specific_credit_allowed=False, motion_authorized=False, semantic_labels_used=False)
            continuation['first_stage'] = deepcopy(pending)
            s['region_continuation'] = None
        relative_mode = mode in ("L", "K", "Y", "Q")
        s["selected"] = None if index is None or (continuation is None and not relative_mode and scores[mode][index] <= 0) else deepcopy(routes[index])
        s["plans"] += 1
        if s["selected"] is not None:
            s["selected"].update(option_id=f'{s["scene_id"]}/{s["episode_id"]}/plan-{s["plans"]}',
                selection_call_id=len(self.calls) + 1, selected_map_version=s["mapper"].frames,
                selected_feedback_version=s["feedback"]["feedback_version"],
                parent_region=graph["current_region"])
            if self.planner_revision in ("v11", "v11_1"):
                s["selected"]["v11_feedback_key"] = rows[index]["v11_feedback_key"]
                s["selected"]["v11_asset_index"] = s["selected"].get("asset_index")
        result = dict(region_continuation=continuation, selected=s["selected"], candidates=routes, scores=scores, score_audit=rows,
                      candidate_audit=audit, mode=mode, plan_number=s["plans"],
                      planner_revision=self.planner_revision,
                      gain_calibration=s["gain"].snapshot())
        s["last_selection"] = json_value(result)
        self._record(scene_idx, "STGHP", "select_topo_target",
            dict(semantic_version=s["semantic_version"], graph_version=s["graph_version"],
                 planning_feedback_version=s["feedback"]["planning_feedback_version"],
                 actual_feedback_version=s["feedback"]["feedback_version"],
                 planning_camera_poses_sha256=digest(s["ledger"].planning_camera_poses()),
                 remaining_budget=s["ledger"].remaining_budget), result, started)
        return None if s["selected"] is None else np.asarray(s["selected"]["pose"][:2], int)


    def summary(self, scene_idx):
        result = super().summary(scene_idx)
        result['visibility_revision'] = 'observed_mesh_v17'
        result['region_continuation'] = 'explicit committed option; common G second-stage/fallback'
        result['learned_feature_revision'] = 'semantic_opportunities_v17; unfitted'
        return result


def visibility_components_v17(args, shape):
    comp = axis_components_v16(args, shape)
    comp._cpu_backend = CPUFourModulesV17(args, 1, shape)
    comp.capabilities = dict(comp._cpu_backend.capabilities)
    comp.capabilities['visibility'] = 'observed TSDF occlusion in common fixed scores'
    return comp
