"""Seal a sensor-only route-admission rerun with the unchanged V2 simulator."""

import argparse
import importlib.util
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "region_independent_floors_20261123"
PREVIOUS = ROOT.parent / "external_support_continuity_20261126"
V1 = ROOT.parent / "external_support_repair_20261126"
SOURCE = ROOT.parent / "region_global_revision_20261122/next/global_attribution"
FREEZE = ROOT / "FORMAL_FREEZE.json"
SCHEMA = "nso_external_online_sweep_comparison_v3"
RUNS = ROOT / "execution/formal"
CONTROL = ROOT / "validation/ONLINE_SWEEP_CONTROLS.json"
FIXED_REPLAY = ROOT / "replay/FIXED_REPLAY.json"
ELIGIBLE_LAYOUT = "b26_hm3d_00770_s1"
ADMITTED = {"admitted", "existing_candidate"}


def supervisor():
    spec = importlib.util.spec_from_file_location("online_sweep_supervisor", BASE / "study.py")
    shared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)
    shared.ROOT, shared.FREEZE, shared.RUNS = ROOT, FREEZE, RUNS
    return shared


def qualify(shared, protocol):
    controls = shared.read(CONTROL)
    if (not controls.get("all_required_pass") or controls.get("total", 0) < 14
            or controls.get("passed") != controls.get("total")
            or len(controls.get("tests", [])) != controls["total"]
            or not all(row["expected_admitted"] == row["actual_admitted"] for row in controls["tests"])):
        raise ValueError("All blind sensor-admission controls must pass before sealing")
    adapter = Path(controls["adapter"].rsplit(":", 1)[0]).resolve()
    if not adapter.is_relative_to(ROOT) or shared.sha(adapter) != controls["adapter_sha256"]:
        raise ValueError("Blind control adapter is outside the study or differs from its recorded digest")

    replay = shared.read(FIXED_REPLAY)
    if (replay["layout"] != ELIGIBLE_LAYOUT or replay["verified_archived_sensor_packets_and_beliefs"] != 39
            or replay["formal_episodes_run"] or any(replay[key] is not False for key in (
                "simulator_motion_executed", "new_sensor_pose_queried", "evaluation_truth_used"))):
        raise ValueError("Replay must bind the 39 archived sensor states without formal motion or evaluator truth")
    prior = shared.read(PREVIOUS / "execution/formal/phase_receipt.json")
    native = next(row for row in prior["episodes"] if row["layout"] == ELIGIBLE_LAYOUT and row["arm"] == "region_S")
    archived = Path(native["run_path"])
    archived_names = {"steps.json", "observed_belief_trace.npz", "regional_choices.json", "global_choices.json"}
    if set(replay["inputs_sha256"]) != archived_names:
        raise ValueError("Replay lacks independent source packet, route or belief hashes")
    archived_files = {str(archived / name): digest for name, digest in replay["inputs_sha256"].items()}
    shared.verify(archived_files)

    checks = {row["paid"]: row for row in replay["checks"]}
    if len(checks) != len(replay["checks"]) or set(checks) != {17, 38} or (
            checks[38]["old_native_routes_and_regional_admission_match_archive"] is not True):
        raise ValueError("p38 must independently reproduce the old native and regional route admission")
    anchor = next(start["anchor"] for start in protocol["starts"] if start["layout"] == ELIGIBLE_LAYOUT)
    gains = {}
    for paid, pose, field, cost in ((17, (50, 25, 2), "certified_regional_routes", (4, 23)),
                                    (38, (52, 26, 2), "native_routes", (14, 28))):
        row = checks[paid]
        baseline, adapted = row["baseline_inflated_grid"], row["adapted_observed_sweep"]
        earlier, later = ({tuple(item["pose"]): item for item in result["fixed_pool_native_admission"]}
                          for result in (baseline, adapted))
        if earlier.keys() != later.keys() or any(earlier[key].get("proposal") != later[key].get("proposal") for key in earlier):
            raise ValueError(f"p{paid} candidate proposal identities changed during fixed replay")
        routes = [item for item in adapted[field] if tuple(item["pose"]) == pose]
        if len(routes) != 1 or any(tuple(item["pose"]) == pose for item in baseline[field]):
            raise ValueError(f"p{paid} has no genuinely new selected candidate in its frozen provider")
        route, = routes
        budget = protocol["budget"] - paid
        if ((route["outbound_cost"], route["return_cost"]) != cost or route["cost"] != sum(cost)
                or route["cost"] > budget or len(route["outbound_states"]) != cost[0] + 1
                or len(route["return_states"]) != cost[1] + 1 or route["outbound_states"][0] != row["pose"]
                or route["outbound_states"][-1] != route["pose"] or route["return_states"][0] != route["pose"]
                or route["return_states"][-1] != anchor):
            raise ValueError(f"p{paid} target lacks its certified outbound and return trajectories")
        if paid == 17 and (earlier[pose]["status"] in ADMITTED or later[pose]["status"] not in ADMITTED):
            raise ValueError("p17 candidate was not independently recovered from the fixed rejected pool")
        if paid == 38 and (not route["group"].startswith("coverage_") or len(later) != 9
                           or any(item["status"] != "native_observed_return_component_mismatch" for item in later.values())):
            raise ValueError("p38 must recover an actual coverage proposal while retaining all nine residual rejections")
        gains[f"paid_{paid}"] = {"pose": pose, "outbound": cost[0], "return": cost[1], "provider": field,
                                  "residual_fixed_rejections": sum(item["status"] not in ADMITTED for item in later.values())}
    inputs = shared.fingerprints([CONTROL, FIXED_REPLAY, adapter, ROOT / "runtime_sweep.py", ROOT / "observed_sweep.py",
                                 ROOT / "sensor_probe/evidence.py",
                                 ROOT / "validation/validate_online_sweep.py", ROOT / "replay/validate_replay.py",
                                 ROOT / "replay/reconstruct_history.py"])
    return inputs | archived_files, gains


def seal(shared):
    if RUNS.exists() or FREEZE.exists() or (ROOT / "FORMAL_PROTOCOL.json").exists():
        raise ValueError("Use the prospective seal before creating any new formal runs or results")
    previous_path = PREVIOUS / "FORMAL_FREEZE.json"
    previous = shared.read(previous_path)
    old_protocol_path = PREVIOUS / "FORMAL_PROTOCOL.json"
    protocol = shared.read(old_protocol_path)
    if (previous["protocol_sha256"] != shared.sha(old_protocol_path) or previous["budget"] != 192
            or protocol["budget"] != 192 or len(protocol["starts"]) != 4 or len(previous["jobs"]) != 12):
        raise ValueError("Expected the intact V2 scope of four starts and twelve 192-action jobs")
    for topic in ("actor_files", "actor_initial_files", "simulator_only_files", "evaluator_files"):
        shared.verify(previous[topic])
    if Path(previous["scene_root"]).resolve() != (PREVIOUS / "scenes").resolve():
        raise ValueError("Rerun must reuse the V2 renderer, collision caches, and scoring references untouched")
    qualification, witness = qualify(shared, protocol)

    prospect = protocol | {"schema": SCHEMA, "archive_before_repair": "9b7ebccdfdafb183cdee85b7d83d5bfa3b999588",
                          "changed_component": "Observation-side swept-footprint route admission",
                          "reference_scope": "V2 scene root, simulator execution, evaluator references and renderer unchanged",
                          "paired_sensitivity": "V2 comparator; all matched paid clocks also recounted on the common V1 reachable mask",
                          "fixed_replay_qualification": {"layout": ELIGIBLE_LAYOUT, "recovered_round_trips": witness}}
    path = ROOT / "FORMAL_PROTOCOL.json"
    shared.write(path, prospect)
    actors = shared.fingerprints([ROOT / name for name in (
        "runner.py", "study.py", "runtime_sweep.py", "observed_sweep.py", "sensor_probe/evidence.py",
        "FORMAL_PROTOCOL.json")])
    frozen = previous | {"schema": SCHEMA, "frozen_unix_s": time.time(), "before_online_sweep_freeze": shared.sha(previous_path),
                         "protocol_relative": path.name, "protocol_sha256": shared.sha(path),
                         "actor_files": previous["actor_files"] | actors,
                         "qualification_files": qualification}
    shared.write(FREEZE, frozen)
    print(f"Frozen {len(frozen['jobs'])} jobs with independently replayed p17/p38 routes: {shared.sha(FREEZE)}")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("command", choices=("seal", "run"))
    cli.add_argument("--workers", type=int, default=6)
    args = cli.parse_args()
    shared = supervisor()
    if args.command == "seal":
        seal(shared)
    else:
        frozen = shared.read(FREEZE)
        if frozen.get("schema") != SCHEMA:
            raise ValueError("The online-sweep schedule has not been sealed")
        shared.verify(frozen["qualification_files"])
        shared.run(args.workers)


if __name__ == "__main__":
    main()
