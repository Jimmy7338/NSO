"""Explicit four-module adapter for the ANS hierarchy.

No Habitat import or model downloads. The caller supplies measured map state.
ANS still proposes a long-term goal and its existing FMM/local policy executes it.
Postprocessing is disabled while PPO trains its original action distribution.
"""
from collections import deque
import numpy as np


def infer_uq_channels(args):
    """Inspect only a compatible UQ tensor key; unavailable weights stay disabled."""
    from pathlib import Path
    import pickle
    import torch
    default = 4 if getattr(args, "use_semantic", False) else 2
    path = getattr(args, "rpn_uq_model_path", None) or getattr(args, "goal_reachability_model_path", None)
    if not path or not Path(path).is_file():
        return default
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
        weight = state.get("encoder.0.weight")
        return int(weight.shape[1]) if weight is not None and weight.shape[1] in (2,4) else default
    except (OSError, RuntimeError, ValueError, KeyError, EOFError, pickle.UnpicklingError):
        return default


class NSORuntimeIntegration:
    def __init__(self, components, num_scenes, full_shape):
        self.components = components
        self.args = components.args
        self.enabled = any((components.use_ov_sem, components.use_topo,
                            components.use_rpn_uq, components.use_igcr))
        self.allow_goal_postprocessing = bool(getattr(self.args, "eval", False)
                                               or not getattr(self.args, "train_global", False))
        self.full_shape = tuple(full_shape)
        self.states = [None for _ in range(num_scenes)]
        self.audit = []
        self.completed_episodes = []
        self._sensor_episode_ids = set()

    def reset_scene(self, scene_idx):
        if not self.enabled:
            return
        old = self.states[scene_idx]
        if old is not None and old.get("cpu_packet_backend"):
            if not old.get("closed"):
                raise RuntimeError("close the sensor episode after its last observation before reset")
            self.completed_episodes.append(self.sensor_episode_summary(scene_idx))
        self.states[scene_idx] = None
        self.components.reset_scene(scene_idx)

    @staticmethod
    def _numpy(value):
        return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)

    def observe(self, scene_idx, step, local_map, bounds, global_pose, rgb_frame=None,
                update_semantic=True, *, sensor_packet=None):
        """Consume one nonterminal sensor map, producing a diagnostic reward.

        bounds=(row0,row1,col0,col1); pose=(x metres,y metres,yaw degrees).
        Initial observations prime reward history. Every scene owns its snapshot.
        The terminal-observation adapter must call this BEFORE resetting a scene.
        """
        if not self.enabled:
            return None
        if self.components._cpu_backend is not None:
            if sensor_packet is None:
                raise ValueError("CPU closed loop requires a measured SensorPacket")
            return self._observe_sensor_packet(scene_idx, sensor_packet)
        local = self._numpy(local_map)
        r0, r1, c0, c1 = map(int, bounds)
        if local.shape != (4, r1-r0, c1-c0):
            raise ValueError("local map and row/column window disagree")
        if not (0 <= r0 < r1 <= self.full_shape[0] and 0 <= c0 < c1 <= self.full_shape[1]):
            raise ValueError("map window outside global bounds")
        old = self.states[scene_idx]
        full = np.zeros((4, *self.full_shape), np.float32) if old is None else old["map"].copy()
        previous = full[1].copy()
        full[:, r0:r1, c0:c1] = local
        pose = np.asarray(global_pose, dtype=float)
        resolution = getattr(self.args, "map_resolution", 5) / 100.
        cell = (int(np.floor(pose[1]/resolution)), int(np.floor(pose[0]/resolution)))
        component = self.components
        if update_semantic and rgb_frame is not None and component.use_ov_sem:
            rgb=np.clip(self._numpy(rgb_frame),0,255).astype(np.uint8)
            component.update_semantic(scene_idx, rgb,
                pose[0]*100., pose[1]*100., pose[2], 0., 0.,
                local_map_origin_x=r0, local_map_origin_y=c0)
        sem = component.get_sem_density(scene_idx)
        if component.use_topo:
            component.update_topo(step, scene_idx, (full[0] > .5).astype(np.uint8),
                (full[1] > .5).astype(np.uint8), agent_cell=cell, sem_density=sem, force=old is None)
        reward = None
        if old is not None and component.use_igcr:
            # Without calibrated occupancy prior/posterior this is deliberately
            # the documented area fallback, not a claimed MI calculation.
            value, parts = component.compute_reward(scene_idx, (previous > .5).astype(np.uint8),
                (full[1] > .5).astype(np.uint8), (full[0] > .5).astype(np.uint8),
                full[3], obstacle_prob=None)
            reward = dict(value=float(value), parts=parts, role="diagnostic",
                          reason="terminal sensor map unavailable in current main adapter")
        self.states[scene_idx] = dict(map=full, cell=cell, pose=pose.copy(), step=int(step))
        return reward

    def semantic_window(self, scene_idx, bounds):
        if not self.enabled:
            return None
        sem = self.components.get_sem_density(scene_idx)
        if sem is None:
            return None
        r0,r1,c0,c1 = map(int,bounds)
        return sem[r0:r1,c0:c1].copy()

    @staticmethod
    def project_path_to_window(safe, start, target, bounds):
        """Follow a connected known-free path and choose its last in-window cell.

        This does not clip a far target through walls or certify unknown space.
        An occupied or disconnected topology target yields no override.
        """
        start, target = tuple(start), tuple(map(int, target))
        h,w = safe.shape
        if any(not (0 <= r < h and 0 <= c < w) for r,c in (start,target)):
            return None
        if not safe[start] or not safe[target]:
            return None
        previous = {start: None}
        queue = deque([start])
        while queue and target not in previous:
            r,c = queue.popleft()
            for dr,dc in ((-1,0),(0,1),(1,0),(0,-1)):
                nxt=(r+dr,c+dc)
                if 0 <= nxt[0] < h and 0 <= nxt[1] < w and safe[nxt] and nxt not in previous:
                    previous[nxt]=(r,c); queue.append(nxt)
        if target not in previous:
            return None
        path=[]; cell=target
        while cell is not None:
            path.append(cell); cell=previous[cell]
        r0,r1,c0,c1=map(int,bounds)
        selected=None
        for r,c in reversed(path):
            if not (r0 <= r < r1 and c0 <= c < c1):
                break
            selected=[r-r0,c-c0]
        return selected

    def choose_goal(self, scene_idx, proposed_local, bounds):
        """Safely postprocess an ANS goal; never replace PPO's training action."""
        original=list(proposed_local)
        if not self.enabled or not self.allow_goal_postprocessing:
            return original
        if self.components._cpu_backend is not None:
            state = self.states[scene_idx]
            if state is None or state.get("closed"):
                return original
            if state["pending"] is not None:
                raise RuntimeError("consume the pending action's sensor observation before replanning")
            r0, r1, c0, c1 = map(int, bounds)
            if (r0, r1, c0, c1) != (0, self.full_shape[0], 0, self.full_shape[1]):
                raise ValueError("CPU option executor currently requires the full map window")
            target = self.components.select_topo_target(scene_idx, state["packet"].action_id)
            option = self.components.get_selected_option(scene_idx)
            state["option"] = option
            state["active_actions"] = [] if option is None else list(option["outbound_actions"])
            state["phase"] = "outbound" if option is not None else "return"
            self.audit.append(dict(event="global_choice", scene_idx=scene_idx,
                frame_id=state["packet"].frame_id, map_version=state["mapper"].frames,
                option=option, heading_preserved=True))
            return original if target is None else target.tolist()
        state=self.states[scene_idx]
        if state is None:
            return original
        comp=self.components; full=state["map"]
        mu,var=None,None
        if comp.use_rpn_uq and comp._rpn_ready:
            import torch
            count=comp._rpn_uq.encoder[0].in_channels
            sem=comp.get_sem_density(scene_idx)
            channels=[full[0],full[1]]
            if count >= 4:
                channels += [np.zeros(self.full_shape) if sem is None else sem, full[3]]
            tensor=torch.as_tensor(np.stack(channels),dtype=torch.float32,
                                   device=next(comp._rpn_uq.parameters()).device)
            pred,uncertainty=comp.predict_reachability_uq(tensor)
            if pred is not None:
                mu=self._numpy(pred)[0]; var=self._numpy(uncertainty)[0]
        target = comp.select_topo_target(scene_idx, state["step"], mu, var) if comp.use_topo else None
        # In this revision UQ gates topology proposals. No grid-logit mask is
        # applied to the two-dimensional Gaussian action produced by ANS.
        chosen=original
        if target is not None:
            from utils.grid_geometry import inflated_obstacles
            radius=getattr(self.args,"robot_radius_m",.2)/(getattr(self.args,"map_resolution",5)/100.)
            # Every cell touched by the robot footprint must be known free;
            # an observed center alone cannot certify an unknown neighbor.
            blocked=(full[0]>.5)|(full[1]<=.5)
            safe=~inflated_obstacles(blocked,radius)
            candidate=self.project_path_to_window(safe,state["cell"],target,bounds)
            if candidate is not None:
                chosen=candidate
        self.audit.append(dict(scene_idx=scene_idx,step=state["step"],proposed=original,chosen=chosen,
            topology_proposed=target is not None,uq_active=mu is not None,
            uq_status=comp.capabilities["reachability_uq"],scope="ANS goal adapter; no performance claim"))
        return chosen

    def summary(self):
        if self.components._cpu_backend is not None:
            return dict(enabled=True, goal_postprocessing=self.allow_goal_postprocessing,
                capabilities=dict(self.components.capabilities),
                igcr_role="sensor ledger feeds next global decision; no PPO reward claim",
                semantic_projection="measured aligned depth backprojection; artificial marker categories",
                uq_role="latest observed footprint and successor return budget; no learned probability",
                pose_scope="centred discrete simulator odometry; ROS extrinsic/non-oracle adapter pending")
        return dict(enabled=self.enabled, goal_postprocessing=self.allow_goal_postprocessing,
            capabilities=dict(self.components.capabilities),
            igcr_role="diagnostic only; training reward unchanged until terminal sensor adapter",
            semantic_projection="legacy approximate projection; depth backprojection pending",
            uq_role="loaded checkpoint gates topology; calibration unverified")

    def start_sensor_episode(self, scene_idx, *, config, transform, packets,
                             total_budget, return_anchor, paid_prefix_actions=0):
        """Prime one authoritative map from actual, externally supplied history.

        A fresh task supplies one frame at action 0. A separately scripted
        prefix must supply ALL frames from 0 and charge every prefix action.
        That prefix is recorded as external history, not an ANS policy result.
        """
        from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
        if self.components._cpu_backend is None:
            raise RuntimeError("initialize the cpu_v10 component backend first")
        if self.states[scene_idx] is not None:
            raise RuntimeError("reset the preceding sensor episode first")
        transform.validate_cpu_mapper(config)
        if tuple(transform.shape) != self.full_shape:
            raise ValueError("runtime and sensor map shape disagree")
        packets = list(packets)
        if (not packets or type(paid_prefix_actions) is not int
                or paid_prefix_actions != len(packets) - 1):
            raise ValueError("all actual prefix frames and paid prefix actions are required")
        if type(total_budget) is not int or total_budget < 1 or not 0 <= paid_prefix_actions <= total_budget:
            raise ValueError("invalid total task budget")
        first = packets[0]
        identity = (first.scene_id, first.episode_id)
        if identity in self._sensor_episode_ids:
            raise ValueError("a fresh sensor episode ID is required after reset")
        ids = set(); last_time = -float("inf")
        for i, p in enumerate(packets):
            p.validate(transform, config)
            if (p.scene_id != first.scene_id or p.episode_id != first.episode_id
                    or p.frame_id in ids or p.action_id != i or p.done or p.collision
                    or p.frame.timestamp_s <= last_time
                    or (i == 0) != (p.action is None)):
                raise ValueError("invalid, terminal or incomplete prefix history")
            if i:
                before = packets[i - 1]
                from utils.grid_geometry import DIRECTIONS
                dr, dc = DIRECTIONS[before.heading] if p.action == "forward" else (0, 0)
                expected = (before.position[0] + int(dr), before.position[1] + int(dc),
                            (before.heading + (1 if p.action == "right" else -1 if p.action == "left" else 0)) % 4)
                if (*p.position, p.heading) != expected:
                    raise ValueError("prefix odometry does not follow its recorded action")
            ids.add(p.frame_id); last_time = p.frame.timestamp_s
        anchor = tuple(return_anchor)
        if (len(anchor) != 3 or any(type(x) is not int for x in anchor)
                or anchor[2] not in range(4)
                or any(not 0 <= v < n for v, n in zip(anchor[:2], self.full_shape))):
            raise ValueError("return anchor requires in-map integer cell and heading")
        mapper = ObservedRuntimeMapperV10(self.full_shape, config)
        for p in packets:
            mapper.update(p.frame, p.scan)
        packet = packets[-1]
        self.states[scene_idx] = dict(cpu_packet_backend=True, mapper=mapper, transform=transform,
            packet=packet, packet_hashes={p.frame_id: p.sha256() for p in packets},
            action_frames={p.action_id: p.frame_id for p in packets},
            phase="planning", active_actions=[], option=None, pending=None, closed=False,
            arrived_count=0, externally_scripted_prefix_actions=paid_prefix_actions)
        try:
            self.components._cpu_backend.start_scene(scene_idx, mapper=mapper, packet=packet,
                prefix_packets=packets, total_budget=total_budget, paid_prefix_actions=paid_prefix_actions,
                return_anchor=anchor)
            self._update_sensor_modules(scene_idx, initial=True)
        except Exception:
            self.states[scene_idx] = None
            self.components._cpu_backend.reset_scene(scene_idx)
            raise
        self._sensor_episode_ids.add(identity)
        return self.sensor_episode_summary(scene_idx)

    def _update_sensor_modules(self, scene_idx, *, initial=False):
        s = self.states[scene_idx]; p = s["packet"]; m = s["mapper"]
        x, y = s["transform"].cell_to_world(p.position)
        self.components.update_semantic(scene_idx, p.frame.color_rgb, x*100, y*100,
            90 - p.heading*90, 0., 0.)
        reward = None
        if not initial:
            value, parts = self.components.compute_reward(scene_idx,
                None, m.belief != -1, m.belief == 1, None)
            reward = dict(value=value, parts=parts, role="planning_feedback")
        self.components.update_topo(p.action_id, scene_idx, m.belief == 1,
            m.belief != -1, agent_cell=p.position,
            sem_density=self.components.get_sem_density(scene_idx), force=True)
        return reward

    def _observe_sensor_packet(self, scene_idx, packet):
        s = self.states[scene_idx]
        if s is None:
            raise RuntimeError("start the sensor episode before observe")
        packet.validate(s["transform"], s["mapper"].config)
        previous = s["packet"]
        if packet.scene_id != previous.scene_id or packet.episode_id != previous.episode_id:
            raise ValueError("sensor packet belongs to another scene/episode")
        sha = packet.sha256()
        if packet.frame_id in s["packet_hashes"]:
            if s["packet_hashes"][packet.frame_id] != sha:
                raise ValueError("conflicting duplicate sensor packet")
            return dict(accepted=False, duplicate=True, value=0., map_version=s["mapper"].frames)
        if s["closed"]:
            raise RuntimeError("episode already closed")
        ledger = self.components._cpu_backend.scenes[scene_idx]["ledger"]
        if ledger.remaining_budget < 1 or s["mapper"].frames != len(s["packet_hashes"]):
            raise RuntimeError("budget or map version changed outside the observation owner")
        pending = s["pending"]
        if (pending is None or packet.action_id != previous.action_id + 1
                or packet.action_id in s["action_frames"] or packet.action != pending["action"]
                or packet.frame.timestamp_s <= previous.frame.timestamp_s):
            raise ValueError("sensor packet must follow the pending paid action exactly once")
        actual = [*packet.position, packet.heading]
        expected = [*previous.position, previous.heading] if packet.collision else pending["next_pose"]
        if actual != expected:
            raise ValueError("measured pose does not match pending discrete action outcome")
        # Only this line fuses a fresh frame. Deduplication and sensor/action
        # validation precede TSDF mutation and IGCR accounting.
        try:
            s["mapper"].update(packet.frame, packet.scan)
        except Exception:
            # TSDF integration is not transactional. A failed fusion must never
            # be retried in this episode as if the map were still pristine.
            s["closed"] = True
            s["termination"] = dict(reason="map_fusion_exception", failed=True,
                returned_to_anchor=False, final_frame_id=previous.frame_id,
                attempted_frame_id=packet.frame_id, final_action_id=previous.action_id)
            raise
        s["packet"] = packet
        s["packet_hashes"][packet.frame_id] = sha
        s["action_frames"][packet.action_id] = packet.frame_id
        if s["active_actions"]:
            s["active_actions"].pop(0)
        arrived = bool(s["option"] is not None and actual == s["option"]["pose"]
                       and not s["active_actions"] and not packet.collision)
        if arrived:
            s["arrived_count"] += 1
        self.components._cpu_backend.bind_packet(scene_idx, packet, arrived=arrived)
        try:
            reward = self._update_sensor_modules(scene_idx)
        except Exception:
            # The raw action and map fusion have happened. Preserve that fact,
            # poison this episode and expose any incomplete feedback commit;
            # neither a resend nor a reset may pretend the step was free.
            s["closed"] = True; s["pending"] = None
            s["termination"] = dict(reason="post_fusion_module_exception", failed=True,
                returned_to_anchor=False, final_frame_id=packet.frame_id,
                final_action_id=packet.action_id,
                actual_paid_actions=packet.action_id,
                feedback_paid_actions=ledger.snapshot()["paid_actions"])
            self.audit.append(dict(event="failed_after_fusion", **s["termination"]))
            raise
        s["pending"] = None
        self.audit.append(dict(event="observation", frame_id=packet.frame_id,
            action_id=packet.action_id, packet_sha256=sha, map_version=s["mapper"].frames,
            arrived=arrived, collision=packet.collision, done=packet.done))
        if packet.done or packet.collision:
            self.close_sensor_episode(scene_idx, reason="collision" if packet.collision else "sensor_terminal")
        return dict(accepted=True, duplicate=False, **reward)

    def next_local_action(self, scene_idx):
        """Return one authorized paid action; require observe before the next."""
        s = self.states[scene_idx]
        if s is None or s["closed"]:
            return None
        if s["pending"] is not None:
            raise RuntimeError("pending action has no sensor observation yet")
        comp = self.components; p = s["packet"]
        ledger = comp._cpu_backend.scenes[scene_idx]["ledger"]
        if ledger.remaining_budget == 0:
            self.close_sensor_episode(scene_idx, reason="budget_exhausted")
            return None
        if not s["active_actions"] and s["phase"] != "return":
            self.choose_goal(scene_idx, list(p.position), (0, self.full_shape[0], 0, self.full_shape[1]))
        if not s["active_actions"]:
            returning = comp.plan_observed_return(scene_idx)
            if not returning.available or not returning.actions:
                self.close_sensor_episode(scene_idx, reason=returning.reason if not returning.available else "no_positive_option_at_anchor")
                return None
            s["phase"] = "return"; s["option"] = None
            s["active_actions"] = list(returning.actions)
        action = s["active_actions"][0]
        comp.bind_execution_option(scene_idx, s["option"])
        check = comp.assess_local_action(scene_idx, action)
        if not check["allowed"]:
            # A fresh obstacle may invalidate the chosen option. Drop it and
            # reserve a new return on the latest map; never clear the obstacle.
            s["active_actions"] = []; s["option"] = None; s["phase"] = "return"
            comp.bind_execution_option(scene_idx, None)
            returning = comp.plan_observed_return(scene_idx)
            if returning.available and returning.actions:
                s["active_actions"] = list(returning.actions)
                action = s["active_actions"][0]; check = comp.assess_local_action(scene_idx, action)
            if not check["allowed"] or not s["active_actions"]:
                self.close_sensor_episode(scene_idx, reason=check["reason"])
                return None
        s["pending"] = check
        self.audit.append(dict(event="paid_action_authorized", frame_id=p.frame_id,
            next_action_id=p.action_id + 1, phase=s["phase"], assessment=check,
            option_id=None if s["option"] is None else s["option"]["option_id"],
            selection_call_id=None if s["option"] is None else s["option"]["selection_call_id"]))
        return action

    def close_sensor_episode(self, scene_idx, *, reason):
        s = self.states[scene_idx]
        if s["pending"] is not None:
            raise RuntimeError("consume the actual terminal observation before closing")
        if s["closed"]:
            return self.sensor_episode_summary(scene_idx)
        b = self.components._cpu_backend.scenes[scene_idx]; p = s["packet"]
        returned = tuple((*p.position, p.heading)) == b["return_anchor"]
        safe_current = bool(b["guard"].safe_grid(s["mapper"].belief)[p.position])
        failed = not returned or p.collision or not safe_current
        s["closed"] = True; s["termination"] = dict(reason=reason, returned_to_anchor=returned,
            failed=failed, current_footprint_known_safe=safe_current,
            final_frame_id=p.frame_id, final_action_id=p.action_id)
        terminal = b["ledger"].finish(reason=reason, failed=failed)
        b["feedback"] = terminal
        self.components._cpu_backend._record(scene_idx, "IGCR", "finish",
            dict(reason=reason, failed=failed), terminal)
        self.audit.append(dict(event="close_after_final_observation", **s["termination"]))
        return self.sensor_episode_summary(scene_idx)

    def sensor_episode_summary(self, scene_idx):
        s = self.states[scene_idx]
        if s is None:
            return None
        return dict(scene_id=s["packet"].scene_id, episode_id=s["packet"].episode_id,
            closed=s["closed"], mapper_frames=s["mapper"].frames,
            unique_packet_count=len(s["packet_hashes"]), arrived_count=s["arrived_count"],
            externally_scripted_prefix_actions=s["externally_scripted_prefix_actions"],
            termination=s.get("termination"), modules=self.components._cpu_backend.summary(scene_idx))
