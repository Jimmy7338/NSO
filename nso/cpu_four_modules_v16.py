"""Explicit V16 shared-axis backend; old V10/V15 source files remain frozen.

Only OV-SDF asset representation and STGHP option generation are overridden.
The inherited geometric score consumes corrected scene assets; execution and
observation guards retain their original implementation. No trained head here.
"""
from copy import deepcopy
from time import perf_counter
import numpy as np
from nso.cpu_four_modules_v10 import CPUFourModules, _OBSERVED_GAIN_REVISIONS
from nso.cpu_sensor_contract_v10 import digest, json_value
from nso.hierarchical_options_v16 import generate_options
from nso.observed_asset_axes_v16 import measured_assets_v16
from nso.semantic_taxonomy_v15 import canonical_observed_records, canonical_vote
from nso.axis_history_view_v16 import AxisHistoryViewV16


class CPUFourModulesV16(CPUFourModules):
    def __init__(self, args, num_scenes, shape):
        if getattr(args, 'cpu_planner_revision', None) != 'v10_3_1':
            raise ValueError('V16 axis integration requires the common v10_3_1 execution/scoring core')
        if getattr(args, 'cpu_score_mode', None) not in ('N', 'G', 'O', 'S', 'X', 'M'):
            raise ValueError('V16 has no qualified learned semantic model')
        self.semantic_source_schema = getattr(args, 'cpu_semantic_source_schema', None)
        canonical_vote(0., self.semantic_source_schema)
        super().__init__(args, num_scenes, shape)
        self.capabilities['asset_axes'] = 'v16 observed surface normal with observed-view fallback'
        self.capabilities['semantic_encoding'] = 'open-negative/closed-positive; explicit source conversion'

    def start_scene(self, scene_idx, **kwargs):
        super().start_scene(scene_idx, **kwargs)
        state = self.scenes[scene_idx]
        packets = kwargs['prefix_packets']
        state['axis_frames'] = [p.frame for p in packets]
        state['axis_frame_ids'] = [p.frame_id for p in packets]
        state['axis_action_id'] = packets[-1].action_id

    def summary(self, scene_idx):
        result = super().summary(scene_idx)
        result['shared_axis_revision'] = 'observed_asset_axes_v16/1'
        result['semantic_source_schema'] = self.semantic_source_schema
        result['semantic_model_efficacy_proven'] = False
        state = self.scenes[scene_idx]
        result['axis_observation_history'] = None if state is None else dict(
            frame_count=len(state['axis_frames']), frame_ids=list(state['axis_frame_ids']),
            last_action_id=state['axis_action_id'], full_actual_history=True)
        return result

    def update_semantic(self, scene_idx):
        started = perf_counter(); s = self.scenes[scene_idx]; mapper = s["mapper"]
        packet = s['packet']
        if packet.action_id != s['axis_action_id']:
            if (packet.action_id != s['axis_action_id'] + 1 or packet.frame_id in s['axis_frame_ids']
                    or packet.frame.timestamp_s <= s['axis_frames'][-1].timestamp_s):
                raise ValueError('axis evidence requires consecutive actual observations')
            s['axis_frames'].append(packet.frame); s['axis_frame_ids'].append(packet.frame_id)
            s['axis_action_id'] = packet.action_id
        elif packet.frame_id != s['axis_frame_ids'][-1]:
            raise ValueError('axis observation identity mismatch')
        view = AxisHistoryViewV16(mapper, s['axis_frames'])
        assets = canonical_observed_records(measured_assets_v16(view), self.semantic_source_schema) if mapper.quality else []
        # Geometry groups are refreshed from measured support. No permanent
        # identity or unobserved surface is inferred from the current group id.
        s["assets"] = assets; s["density"][:] = 0.
        support = []
        for asset in assets:
            bits = np.asarray(asset["bits"], dtype=np.int64)
            support.append([float(np.mean((bits & (1 << sector)) != 0))
                            for sector in range(8)])
        s["current_direction_support"] = support
        if s["planning_direction_support"] is None:
            s["planning_direction_support"] = deepcopy(support)
        for row in mapper.quality.values():
            cell = mapper.grid_cell(row["point"])
            if (all(0 <= v < lim for v, lim in zip(cell, self.shape))
                    and mapper.belief[cell] != -1 and row["label"] in (2, 3)):
                s["density"][cell] = 1.
        s["semantic_version"] = mapper.frames
        self._record(scene_idx, "OV-SDF", "update_semantic", dict(packet=s["packet"].sha256()),
            dict(measured_assets=[{k: v for k, v in a.items() if k != "points"} for a in assets],
                 density_sha256=digest(s["density"]), semantic_version=s["semantic_version"]), started)

    def select_target(self, scene_idx):
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


def axis_components_v16(args, shape):
    """Initialize the existing four-interface facade before any scene begins."""
    from nso.components import NSO_Components
    comp = NSO_Components(args)
    comp.initialize('cpu', 1, *shape, *shape)
    if comp._cpu_backend is None or comp._cpu_backend.calls or any(x is not None for x in comp._cpu_backend.scenes):
        raise ValueError('axis backend must be installed before scene bootstrap')
    comp._cpu_backend = CPUFourModulesV16(args, 1, shape)
    comp.capabilities = dict(comp._cpu_backend.capabilities)
    return comp
