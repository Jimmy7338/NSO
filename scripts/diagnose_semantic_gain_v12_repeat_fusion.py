#!/usr/bin/env python3
"""Measure deterministic same-pose sweep effects on the joint metric."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from env.virtual3d_inspection_v4 import InspectionConfigV4, InspectionWorldV4
from nso.components import NSO_Components
from nso.cpu_sensor_contract_v10 import GridTransform, json_value
from nso.runtime_integration import NSORuntimeIntegration
from scripts.collect_semantic_gain_v12_branches import metrics
from scripts.smoke_semantic_gain_v12_candidates import external_prefix_states, packet
from nso.response_candidates_v7 import _actions
from utils.reconstruction_metrics import ReconstructionEvaluator


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(json_value(value), indent=2, sort_keys=True,
                                     allow_nan=False) + "\n")


def main(output, sweeps):
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    scene_path = ROOT / "configs/virtual3d/semantic_gain_v12_scale_development.json"
    protocol = json.loads(scene_path.read_text())
    declared = protocol["contexts"][0]
    settings = {**protocol["shared_conditions"],
                **{key: value for key, value in declared.items() if key not in ("id", "seed")}}
    config = InspectionConfigV4(**settings)
    world = InspectionWorldV4(config, seed=declared["seed"], semantic_condition="aligned")
    transform = GridTransform(tuple(world.shape), config.resolution_m)
    states, _, anchor = external_prefix_states(world, anchor_sweep=False)
    prefix_actions = _actions(states)
    packets = [packet(world, transform, None, 0)]
    for index, action in enumerate(prefix_actions, 1):
        _, collision, done = world.step(action)
        if collision or done:
            raise RuntimeError("invalid diagnostic prefix")
        packets.append(packet(world, transform, action, index))
    args = SimpleNamespace(nso_backend="cpu_v10", eval=True, train_global=False,
        use_open_vocab_semantic=True, use_topo_graph=True, use_rpn_uq=True, use_igcr=True,
        cpu_score_mode="S", cpu_disable_feedback=False, cpu_max_candidates=53,
        cpu_coverage_slots=4, cpu_planner_revision="v10_3",
        cpu_measured_novelty_floor=.25, run_id="v12-repeat-fusion-diagnostic")
    components = NSO_Components(args)
    components.initialize("cpu", 1, *world.shape, *world.shape)
    runtime = NSORuntimeIntegration(components, 1, world.shape)
    runtime.start_sensor_episode(0, config=config, transform=transform, packets=packets,
        total_budget=config.max_steps, return_anchor=anchor,
        paid_prefix_actions=len(prefix_actions))
    mapper = runtime.states[0]["mapper"]
    evaluator = ReconstructionEvaluator(world, count=8000, seed=12026)
    rows = [{"completed_sweeps": 0, "integrated_repeat_frames": 0,
             **metrics(mapper, world, evaluator)}]
    for sweep in range(1, sweeps + 1):
        for _ in range(4):
            _, collision, done = world.step("left")
            if collision or done:
                raise RuntimeError("in-place diagnostic sweep unexpectedly failed")
            mapper.update(world.sense(), world.scan())
        row = {"completed_sweeps": sweep, "integrated_repeat_frames": 4 * sweep,
               **metrics(mapper, world, evaluator)}
        row["joint_gain_from_no_sweep"] = (row["area_times_f1_05cm"]
                                            - rows[0]["area_times_f1_05cm"])
        row["marginal_joint_gain"] = (row["area_times_f1_05cm"]
                                      - rows[-1]["area_times_f1_05cm"])
        rows.append(row)
    result = {"schema_version": "semantic_gain_v12_repeat_fusion_diagnostic/1",
        "status": "complete", "context": "D12-00",
        "physical_noise": {"depth_sigma_m": config.depth_sigma_m, "dropout": config.dropout,
                           "pose_noise_m": config.pose_noise_m},
        "prefix_paid_actions": len(prefix_actions), "anchor": anchor,
        "action_per_sweep": ["left", "left", "left", "left"],
        "same_four_headings_repeated": True,
        "rows": rows,
        "interpretation_boundary": "diagnoses metric/backend response only; no planner advantage"}
    write(output / "result.json", result)
    write(output / "artifact_hashes.json", {"result.json": sha(output / "result.json")})
    print(json.dumps([{"sweeps": row["completed_sweeps"], "f1": row["f1_05cm"],
        "joint": row["area_times_f1_05cm"], "marginal": row.get("marginal_joint_gain")}
        for row in rows], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sweeps", type=int, default=8)
    args = parser.parse_args()
    main(args.output.resolve(), args.sweeps)
