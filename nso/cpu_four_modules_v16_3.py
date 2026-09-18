"""V16.3 common feasible-viewpoint backend; no learned semantic response is qualified."""
from copy import deepcopy
from time import perf_counter
import numpy as np
from nso.cpu_four_modules_v10 import _OBSERVED_GAIN_REVISIONS
from nso.cpu_four_modules_v16 import axis_components_v16
from nso.cpu_four_modules_v16_2 import CPUFourModulesV16_2
from nso.axis_history_view_v16 import AxisHistoryViewV16
from nso.cpu_sensor_contract_v10 import digest, json_value
from nso.hierarchical_options_v16_3 import generate_options


class CPUFourModulesV16_3(CPUFourModulesV16_2):
    def summary(self, scene_idx):
        result = super().summary(scene_idx)
        result['candidate_revision'] = 'hierarchical_options_v16_3/1'
        return result

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
        relative_mode = mode in ("L", "K", "Y", "Q")
        s["selected"] = None if index is None or (not relative_mode and scores[mode][index] <= 0) else deepcopy(routes[index])
        s["plans"] += 1
        if s["selected"] is not None:
            s["selected"].update(option_id=f'{s["scene_id"]}/{s["episode_id"]}/plan-{s["plans"]}',
                selection_call_id=len(self.calls) + 1, selected_map_version=s["mapper"].frames,
                selected_feedback_version=s["feedback"]["feedback_version"],
                parent_region=graph["current_region"])
            if self.planner_revision in ("v11", "v11_1"):
                s["selected"]["v11_feedback_key"] = rows[index]["v11_feedback_key"]
                s["selected"]["v11_asset_index"] = s["selected"].get("asset_index")
        result = dict(selected=s["selected"], candidates=routes, scores=scores, score_audit=rows,
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


def viewpoint_components_v16_3(args, shape):
    comp = axis_components_v16(args, shape)
    comp._cpu_backend = CPUFourModulesV16_3(args, 1, shape)
    comp.capabilities = dict(comp._cpu_backend.capabilities)
    comp.capabilities['staged_views'] = 'shared feasible aperture viewpoints; actual visibility unverified'
    return comp
