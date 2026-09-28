"""V13.2 outbound-first execution. Frozen V13 restoration and metrics are shared."""
from time import perf_counter
import numpy as np
from nso.continuation_v13_2 import force_outbound_candidate
from nso.decision_replay_v13 import save_packet
from scripts.collect_semantic_gain_v13_history import packet, write
from scripts.collect_semantic_gain_v13_continuations import restore, evaluate
from utils.counterfactual_surface_visibility import reference_visible


def execute(config, protocol, stored, checkpoint, candidate, evaluator, weight, folder):
    if protocol["first_option_execution"] != "complete_outbound_then_common_geometry_continuation":
        raise ValueError("V13.2 outbound contract required")
    folder.mkdir(); (folder / "packets").mkdir()
    world, transform, runtime = restore(config, stored, checkpoint)
    before = evaluate(runtime, world, evaluator, protocol["thresholds_m"])
    before_visible = np.zeros(len(evaluator.reference), bool)
    for p in stored[:checkpoint["action_id"] + 1]:
        before_visible |= reference_visible(evaluator.reference, p.frame, evaluator.truth,
                                            p.frame.world_from_camera, world.config.max_depth_m)
    visible = before_visible.copy()
    option = force_outbound_candidate(runtime, candidate["candidate_id"], checkpoint["candidate_sha256"])
    actions, forced_observed = [], []
    first_end_pose = None
    started = perf_counter()
    while not runtime.states[0]["closed"]:
        action = runtime.next_local_action(0)
        if action is None:
            break
        active = runtime.states[0]["option"]
        forced = active is not None and active["option_id"] == option["option_id"]
        frame, collision, done = world.step(action)
        observed = packet(world, config, action, frame, collision, done).validate(transform, world.config)
        save_packet(folder / f"packets/{world.step_count:04d}.npz", observed)
        runtime.observe(0, world.step_count, None, None, None, sensor_packet=observed)
        visible |= reference_visible(evaluator.reference, frame, evaluator.truth,
                                      frame.world_from_camera, world.config.max_depth_m)
        actions.append(dict(action_id=world.step_count, action=action, forced_option=forced,
            pose=[*world.position, world.heading], collision=bool(collision),
            packet_sha256=observed.sha256()))
        if forced:
            forced_observed.append(action)
            first_end_pose = [*world.position, world.heading]
        write(folder / "actions.json", actions)
    elapsed = perf_counter() - started
    summary = runtime.sensor_episode_summary(0)
    after = evaluate(runtime, world, evaluator, protocol["thresholds_m"])
    mesh = runtime.states[0]["mapper"].mesh()
    np.savez_compressed(folder / "final_mesh.npz", vertices=np.asarray(mesh.vertices),
        triangles=np.asarray(mesh.triangles), vertex_colors=np.asarray(mesh.vertex_colors))
    np.savez_compressed(folder / "visibility.npz", before=before_visible, after=visible)
    result = dict(parent=config["parent"], arrangement=config["arrangement"],
        checkpoint_action=checkpoint["action_id"], candidate_id=candidate["candidate_id"],
        group=candidate["group"], state_sha256=checkpoint["state"]["sha256"],
        candidate_pool_sha256=checkpoint["candidate_sha256"], total_budget=config["total_budget"],
        available_continuation_budget=config["total_budget"] - checkpoint["action_id"],
        paid_actions=len(actions), total_paid_actions=world.step_count,
        first_outbound_option_completed=forced_observed == candidate["outbound_actions"],
        first_outbound_target_reached=first_end_pose == candidate["states"][len(candidate["outbound_actions"])],
        first_outbound_end_pose=first_end_pose,
        intervention_contract="first_outbound_then_geometry_replan_with_live_return_guard",
        forced_paid_actions=len(forced_observed), collisions=world.collisions,
        termination=summary["termination"], before=before, after=after,
        joint_gain=after["joint_05cm"] - before["joint_05cm"],
        area_joint_gain=after["area_times_f1_05cm"] - before["area_times_f1_05cm"],
        new_visible_surface_m2=float(np.count_nonzero(visible & ~before_visible)) * weight,
        path_distance_m=sum(row["action"] == "forward" for row in actions) * world.config.resolution_m,
        action_time_s=len(actions) * world.config.action_duration_s,
        continuation_wall_time_s=elapsed, physical_prefix_exact=True,
        modules_called=sorted({c["module"] for c in runtime.components._cpu_backend.calls}))
    write(folder / "result.json", result)
    write(folder / "runtime_audit.json", runtime.audit)
    write(folder / "modules.json", runtime.components._cpu_backend.calls)
    return result

