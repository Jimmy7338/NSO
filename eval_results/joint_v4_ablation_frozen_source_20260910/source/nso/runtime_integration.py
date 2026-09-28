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

    def reset_scene(self, scene_idx):
        if not self.enabled:
            return
        self.states[scene_idx] = None
        self.components.reset_scene(scene_idx)

    @staticmethod
    def _numpy(value):
        return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)

    def observe(self, scene_idx, step, local_map, bounds, global_pose, rgb_frame=None,
                update_semantic=True):
        """Consume one nonterminal sensor map, producing a diagnostic reward.

        bounds=(row0,row1,col0,col1); pose=(x metres,y metres,yaw degrees).
        Initial observations prime reward history. Every scene owns its snapshot.
        The terminal-observation adapter must call this BEFORE resetting a scene.
        """
        if not self.enabled:
            return None
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
        return dict(enabled=self.enabled, goal_postprocessing=self.allow_goal_postprocessing,
            capabilities=dict(self.components.capabilities),
            igcr_role="diagnostic only; training reward unchanged until terminal sensor adapter",
            semantic_projection="legacy approximate projection; depth backprojection pending",
            uq_role="loaded checkpoint gates topology; calibration unverified")
