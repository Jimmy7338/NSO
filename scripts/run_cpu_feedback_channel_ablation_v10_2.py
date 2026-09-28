#!/usr/bin/env python3
"""Isolate planning-history and camera-yield feedback channels in V10.2."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse, json, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from env.virtual3d_competition_v9 import create_competition_world
from nso.components import NSO_Components
from nso.runtime_integration import NSORuntimeIntegration
from nso.cpu_sensor_contract_v10 import GridTransform, SensorPacket, json_value
from scripts.eval_competition_v9_1 import restore_evaluator, snapshot_metrics
from scripts.run_cpu_semantic_inspection_v10_1 import prefix_packets
from utils.counterfactual_surface_visibility import reference_visible, surface_increment
from utils.cpu_protocol import file_hash


def write(path, value):
    Path(path).write_text(json.dumps(json_value(value), indent=2, sort_keys=True,
                                     allow_nan=False) + "\n")


def run_one(prepared, context, arrangement, condition):
    history_only = condition == "planning_history_only"
    gain_only = condition == "gain_posterior_only"
    world = create_competition_world(context, arrangement)
    transform = GridTransform(tuple(world.shape), world.config.resolution_m)
    packets = prefix_packets(prepared, context, arrangement, world, transform)
    ref = dict(np.load(prepared / context / arrangement / "reference.npz"))
    evaluator = restore_evaluator(world, ref)
    args = SimpleNamespace(nso_backend="cpu_v10", eval=True, train_global=False,
        use_open_vocab_semantic=True, use_topo_graph=True, use_rpn_uq=True, use_igcr=True,
        cpu_score_mode="S", cpu_disable_feedback=gain_only,
        cpu_freeze_gain_posterior=history_only, cpu_force_gain_feedback=gain_only,
        cpu_max_candidates=12, cpu_coverage_slots=4, cpu_planner_revision="v10_2",
        run_id=f"feedback-channel-{condition}-{context}-{arrangement}")
    comp = NSO_Components(args); comp.initialize("cpu", 1, *world.shape, *world.shape)
    runtime = NSORuntimeIntegration(comp, 1, world.shape); anchor = (*packets[-1].position, packets[-1].heading)
    runtime.start_sensor_episode(0, config=world.config, transform=transform,
        packets=packets, total_budget=198, return_anchor=anchor, paid_prefix_actions=150)
    _, before = snapshot_metrics(runtime.states[0]["mapper"], world, evaluator, ref, (.05, .10))
    seen = np.zeros_like(ref["prefix_seen"]); actions = []
    while not runtime.states[0]["closed"]:
        action = runtime.next_local_action(0)
        if action is None: break
        frame, collision, done = world.step(action); scan = world.scan()
        packet = SensorPacket(f"{context}_{arrangement}", f"v10_1-{context}-{arrangement}",
            f"frame-{world.step_count}", world.step_count, frame, scan, tuple(map(int, world.position)),
            int(world.heading), "deterministic_simulator", "simulator_exact_discrete_odometry",
            action, bool(collision), bool(done)).validate(transform, world.config)
        seen |= reference_visible(evaluator.reference, frame, evaluator.truth,
                                  frame.world_from_camera, world.config.max_depth_m)
        runtime.observe(0, world.step_count, None, None, None, sensor_packet=packet); actions.append(action)
    summary = runtime.sensor_episode_summary(0)
    _, after = snapshot_metrics(runtime.states[0]["mapper"], world, evaluator, ref, (.05, .10))
    area = surface_increment(ref["prefix_seen"], seen, ref["weights"])
    return dict(context=context, arrangement=arrangement, condition=condition,
        joint=area * after["f1_05cm"], area=area, f1_gain=after["f1_05cm"]-before["f1_05cm"],
        actions=actions, returned=summary["termination"]["returned_to_anchor"],
        failed=summary["termination"]["failed"], collisions=world.collisions,
        ledger_feedback=summary["modules"]["feedback"]["feedback_enabled"],
        gain_feedback=summary["modules"]["observed_gain_calibration"]["feedback_enabled"],
        posterior=summary["modules"]["observed_gain_calibration"]["posterior_mean"])


def main(prepared, output):
    output.mkdir(parents=True, exist_ok=False)
    source_hash = file_hash(Path(__file__).resolve()); rows=[]
    for condition in ("planning_history_only", "gain_posterior_only"):
        for context in ("Q0", "Q1", "Q2", "Q3"):
            for arrangement in ("shelf_west", "shelf_east"):
                row=run_one(prepared,context,arrangement,condition); rows.append(row)
                write(output/"partial.json",rows); print(condition,context,arrangement,row["joint"],flush=True)
    write(output/"results.json",rows)
    write(output/"summary.json",{c:{"mean_joint":float(np.mean([r["joint"] for r in rows if r["condition"]==c])),
        "mean_area":float(np.mean([r["area"] for r in rows if r["condition"]==c])),
        "mean_f1_gain":float(np.mean([r["f1_gain"] for r in rows if r["condition"]==c]))}
        for c in ("planning_history_only","gain_posterior_only")})
    write(output/"manifest.json",{"post_hoc_channel_isolation":True,"source_sha256":source_hash,
        "status":"complete","branches":len(rows)})


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--prepared",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); main(a.prepared.resolve(),a.output.resolve())
