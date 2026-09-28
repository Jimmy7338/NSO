"""CPU implementations behind NSO_Components, using observed state only.

The category/structure prior is the existing artificial-marker positive
control. It is neither open-vocabulary inference nor a trained response model.
The global graph contains observed connected regions and pose options; its
local executor reserves a known-safe route to a fixed anchor including heading.
"""
from copy import deepcopy
from time import perf_counter
import numpy as np
from scipy.ndimage import label
from env.virtual3d import camera_pose
from nso.competition_candidates_v8 import measured_assets, aperture_support
from nso.response_features_v7 import response_features
from nso.execution_guard_v8 import ObservedExecutionGuard
from nso.hierarchical_options_v10 import generate_options
from nso.observed_feedback_v10 import FeedbackLedger
from nso.cpu_sensor_contract_v10 import digest, json_value


class CPUFourModules:
    def __init__(self, args, num_scenes, shape):
        self.args, self.shape = args, tuple(shape)
        self.scenes = [None] * num_scenes
        self.calls = []
        self.capabilities = {
            "semantic": "measured RGB-D marker support; declared category prior; open vocabulary unavailable",
            "topology": "observed connected-region graph and full pose options; room segmentation unvalidated",
            "reachability_uq": "deterministic observed-map guard; trained/calibrated uncertainty unavailable",
            "igcr": "actual sensor feedback consumed in next option score; quality is an observed proxy",
        }

    def _record(self, scene_idx, module, method, inputs, outputs, started=None):
        s = self.scenes[scene_idx]
        event = dict(call_id=len(self.calls) + 1, parent_call_id=s.get("last_call_id"),
                     run_id=str(getattr(self.args, "run_id", "cpu_v10")),
                     scene_id=s["scene_id"], episode_id=s["episode_id"],
                     module=module, method=method, frame_id=s["packet"].frame_id,
                     action_id=s["packet"].action_id, map_version=s["mapper"].frames,
                     feedback_version=s.get("feedback", {}).get("feedback_version", 0),
                     input_sha256=digest(inputs), output_sha256=digest(outputs),
                     inputs=json_value(inputs), outputs=json_value(outputs),
                     trained=False, calibrated=False, truth_used=False,
                     elapsed_s=0. if started is None else perf_counter() - started)
        execution = s.get("execution_option")
        event["executed_option_id"] = None if execution is None else execution["option_id"]
        event["selection_call_id"] = None if execution is None else execution["selection_call_id"]
        self.calls.append(event); s["last_call_id"] = event["call_id"]
        return event

    def start_scene(self, scene_idx, *, mapper, packet, prefix_packets,
                    total_budget, paid_prefix_actions, return_anchor):
        if self.scenes[scene_idx] is not None:
            raise RuntimeError("reset scene before starting a new episode")
        ledger = FeedbackLedger(packet.scene_id, packet.episode_id, total_budget,
            feedback_enabled=not getattr(self.args, "cpu_disable_feedback", False))
        self.scenes[scene_idx] = dict(scene_id=packet.scene_id, episode_id=packet.episode_id,
            mapper=mapper, packet=packet, return_anchor=tuple(return_anchor), ledger=ledger,
            guard=ObservedExecutionGuard(mapper.config.resolution_m, mapper.config.robot_radius_m),
            assets=[], density=np.zeros(self.shape, np.float32), semantic_version=0,
            graph_version=0, selected=None, plans=0, feedback={}, graph=None)
        event = ledger.bootstrap(mapper, frames=[p.frame for p in prefix_packets],
            frame_ids=[p.frame_id for p in prefix_packets], map_version=mapper.frames,
            last_action_id=packet.action_id, paid_actions=paid_prefix_actions)
        self.scenes[scene_idx]["feedback"] = event
        self._record(scene_idx, "IGCR", "bootstrap", dict(prefix_frames=len(prefix_packets),
            paid_prefix_actions=paid_prefix_actions, total_budget=total_budget), event)

    def bind_packet(self, scene_idx, packet, *, arrived=False):
        s = self.scenes[scene_idx]
        s["packet"], s["arrived"] = packet, bool(arrived)

    def update_semantic(self, scene_idx):
        started = perf_counter(); s = self.scenes[scene_idx]; mapper = s["mapper"]
        assets = measured_assets(mapper) if mapper.quality else []
        # Geometry groups are refreshed from measured support. No permanent
        # identity or unobserved surface is inferred from the current group id.
        s["assets"] = assets; s["density"][:] = 0.
        for row in mapper.quality.values():
            cell = mapper.grid_cell(row["point"])
            if (all(0 <= v < lim for v, lim in zip(cell, self.shape))
                    and mapper.belief[cell] != -1 and row["label"] in (2, 3)):
                s["density"][cell] = 1.
        s["semantic_version"] = mapper.frames
        self._record(scene_idx, "OV-SDF", "update_semantic", dict(packet=s["packet"].sha256()),
            dict(measured_assets=[{k: v for k, v in a.items() if k != "points"} for a in assets],
                 density_sha256=digest(s["density"]), semantic_version=s["semantic_version"]), started)

    def compute_reward(self, scene_idx):
        started = perf_counter(); s = self.scenes[scene_idx]; p = s["packet"]
        failed = p.collision or (p.done and ((*p.position, p.heading) != s["return_anchor"]
            or not s["guard"].safe_grid(s["mapper"].belief)[p.position]))
        event = s["ledger"].observe(s["mapper"], frame=p.frame, frame_id=p.frame_id,
            action_id=p.action_id, map_version=s["mapper"].frames, collision=p.collision,
            arrived=s.get("arrived", False), done=p.done, failed=failed,
            reason="collision" if p.collision else "terminal_without_safe_return" if failed else None,
            scene_id=p.scene_id, episode_id=p.episode_id)
        if not event["accepted"]:
            raise RuntimeError("runtime must reject duplicate packets before fusion and feedback")
        s["feedback"] = event
        self._record(scene_idx, "IGCR", "compute_reward", dict(packet=p.sha256()), event, started)
        return event["reward"], event["parts"]

    def update_topo(self, scene_idx):
        started = perf_counter(); s = self.scenes[scene_idx]
        safe = s["guard"].safe_grid(s["mapper"].belief)
        regions, n = label(safe, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
        s["graph"] = dict(regions=regions, nodes=list(range(1, n + 1)),
                          current_region=int(regions[s["packet"].position]),
                          anchor_region=int(regions[s["return_anchor"][:2]]))
        s["graph_version"] = s["mapper"].frames
        self._record(scene_idx, "STGHP", "update_topo",
            dict(semantic_version=s["semantic_version"], feedback_version=s["feedback"]["feedback_version"],
                 belief_sha256=digest(s["mapper"].belief)),
            dict(region_count=n, current_region=s["graph"]["current_region"],
                 anchor_region=s["graph"]["anchor_region"], graph_version=s["graph_version"]), started)

    def _scores(self, s, routes):
        # Only outbound actions are committed. Score their total expected gain;
        # return actions constrain feasibility but do not inflate expected gain
        # or enter the history of already obtained observations.
        outbound = [{**r, "states": r["outbound_states"], "cost": r["outbound_cost"]} for r in routes]
        features, _ = response_features(s["mapper"], outbound)
        q = s["mapper"].quality_evidence()
        npoints = 0 if q is None else len(q["point"])
        costs = np.asarray([r["outbound_cost"] for r in routes])
        common = ((features["N"][:, 0] + features["N"][:, 1]) * s["mapper"].config.max_depth_m**2
                  + features["N"][:, 2] * max(1, npoints) * .15**2) * costs
        assets = s["assets"]; config = s["mapper"].config
        history = s["ledger"].planning_camera_poses()
        old = np.zeros((len(assets), 25))
        for pose in history:
            for ai, asset in enumerate(assets):
                old[ai] = np.maximum(old[ai], aperture_support(asset, pose, config))
        generic, semantic, swapped, marked = [], [], [], []
        for a in assets:
            w, d, h, vote = a["measured_width_m"], a["measured_depth_m"], a["measured_height_m"], a["class_vote"]
            generic.append(w*h + 3*w*d); semantic.append(w*h + 3*(1-vote)*w*d)
            swapped.append(w*h + 3*(1+vote)*w*d); marked.append(float(a["marked_points"] > 0))
        generic, semantic, swapped, marked = map(np.asarray, (generic, semantic, swapped, marked))
        scores = {m: [] for m in ("N", "G", "O", "S", "X", "M")}; audit = []
        for i, route in enumerate(routes):
            planned = np.zeros_like(old)
            for state in route["outbound_states"][1:]:
                pose = camera_pose(state[:2], state[2], config, s["mapper"].shape[0])
                for ai, asset in enumerate(assets):
                    planned[ai] = np.maximum(planned[ai], aperture_support(asset, pose, config))
            factors = np.maximum(0., planned - old).mean(axis=1)
            potentials = dict(N=0., G=factors@generic, O=factors@(generic*marked),
                S=factors@(semantic*marked + generic*(1-marked)),
                X=factors@(swapped*marked + generic*(1-marked)), M=factors@generic)
            for mode, potential in potentials.items():
                scores[mode].append(float(common[i] + potential))
            audit.append(dict(candidate_id=route["candidate_id"], common_total_proxy=float(common[i]),
                incremental_aperture=factors.tolist(), prior_potentials=potentials))
        return scores, audit

    def select_target(self, scene_idx):
        started = perf_counter(); s = self.scenes[scene_idx]; p = s["packet"]
        if s["graph_version"] != s["mapper"].frames or s["semantic_version"] != s["mapper"].frames:
            raise RuntimeError("global planning requires current OV-SDF and STGHP state")
        routes, audit = generate_options(s["mapper"], p.position, p.heading,
            s["ledger"].remaining_budget, s["return_anchor"],
            max_candidates=int(getattr(self.args, "cpu_max_candidates", 12)))
        graph = s["graph"]
        routes = [r for r in routes if graph["current_region"] > 0
                  and graph["current_region"] == graph["anchor_region"]
                  == int(graph["regions"][tuple(r["pose"][:2])])]
        mode = getattr(self.args, "cpu_score_mode", "S")
        if mode not in ("N", "G", "O", "S", "X", "M"):
            raise ValueError("unknown CPU semantic ablation")
        scores, rows = self._scores(s, routes) if routes else ({m: [] for m in ("N","G","O","S","X","M")}, [])
        index = min(range(len(routes)), key=lambda i: (-scores[mode][i], routes[i]["cost"],
                                                      routes[i]["candidate_id"])) if routes else None
        s["selected"] = None if index is None or scores[mode][index] <= 0 else deepcopy(routes[index])
        s["plans"] += 1
        if s["selected"] is not None:
            s["selected"].update(option_id=f'{s["scene_id"]}/{s["episode_id"]}/plan-{s["plans"]}',
                selection_call_id=len(self.calls) + 1, selected_map_version=s["mapper"].frames,
                selected_feedback_version=s["feedback"]["feedback_version"],
                parent_region=graph["current_region"])
        result = dict(selected=s["selected"], candidates=routes, scores=scores, score_audit=rows,
                      candidate_audit=audit, mode=mode, plan_number=s["plans"])
        s["last_selection"] = json_value(result)
        self._record(scene_idx, "STGHP", "select_topo_target",
            dict(semantic_version=s["semantic_version"], graph_version=s["graph_version"],
                 planning_feedback_version=s["feedback"]["planning_feedback_version"],
                 actual_feedback_version=s["feedback"]["feedback_version"],
                 planning_camera_poses_sha256=digest(s["ledger"].planning_camera_poses()),
                 remaining_budget=s["ledger"].remaining_budget), result, started)
        return None if s["selected"] is None else np.asarray(s["selected"]["pose"][:2], int)

    def assess_action(self, scene_idx, action):
        started = perf_counter(); s = self.scenes[scene_idx]; p = s["packet"]
        budget = s["ledger"].remaining_budget
        check = s["guard"].assess(s["mapper"].belief, p.position, p.heading, action, budget)
        heading = (p.heading + (1 if action == "right" else -1 if action == "left" else 0)) % 4
        returning = None
        if check.allowed:
            returning = s["guard"].return_plan(s["mapper"].belief, check.target, heading,
                                               s["return_anchor"], budget - 1)
        allowed = check.allowed and returning.available
        result = dict(allowed=bool(allowed), reason=check.reason if not check.allowed else returning.reason,
            action=action, next_pose=[*check.target, heading], remaining_budget=budget,
            reserved_return_cost=None if returning is None else returning.paid_cost,
            learned_probability=None, calibrated_uncertainty=None)
        self._record(scene_idx, "RPN-UQ", "assess_local_action",
            dict(belief_sha256=digest(s["mapper"].belief), action=action,
                 anchor=s["return_anchor"], remaining_budget=budget), result, started)
        return result

    def plan_return(self, scene_idx):
        s = self.scenes[scene_idx]; p = s["packet"]
        result = s["guard"].return_plan(s["mapper"].belief, p.position, p.heading,
                                       s["return_anchor"], s["ledger"].remaining_budget)
        self._record(scene_idx, "RPN-UQ", "plan_return", dict(anchor=s["return_anchor"]), result.__dict__)
        return result

    def reset_scene(self, scene_idx):
        self.scenes[scene_idx] = None

    def summary(self, scene_idx):
        s = self.scenes[scene_idx]
        return {} if s is None else dict(plans=s["plans"], graph_version=s["graph_version"],
            semantic_version=s["semantic_version"], feedback=s["ledger"].snapshot(),
            return_anchor=s["return_anchor"], retained_hierarchy="region -> pose option -> paid action")
