#!/usr/bin/env python3
"""Freeze observed same-goal route choices, or execute one sealed route witness."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
import sys

from supported_hidden import compare_routes


ROOT = Path(__file__).resolve().parent
ANALYSES = ROOT.parents[2]
SCENES = ROOT / "scenes"
PREVIOUS = ANALYSES / "region_structure_floor_20261122"
WEIGHTED = ANALYSES / "weighted_geometric_control_20261122"
COLD = ANALYSES / "same_goal_cold_holdout_20261121"
sys.path[:0] = [str(PREVIOUS/"local"), str(WEIGHTED/"runner"), str(COLD/"runner")]
from direct_inspection import rank_direct
from cold_run import full_local_route_adapter, native, raw_sensor_hashes
from execute import array_sha, capture, chain_hash
from weighted_score import Weights, choose_route, rank_admitted


class ForkCaptured(Exception):
    pass


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def verify(files, visible=False):
    for path, digest in files.items():
        if visible:
            assert "evaluator_references" not in Path(path).parts
        assert sha(path) == digest, path


def run(layout, arm, out, actor_seal, route_seal):
    spec = read(ROOT/"PROTOCOL.json")
    sealed = None if route_seal is None else route_seal["layouts"][layout]
    assert (sealed is None) == (arm == "probe")
    weights = Weights(**spec["weights"])
    protocol = read(ANALYSES/"same_goal_route_choice_20261121/PROTOCOL.json")
    protocol["heldout"].update(layouts=spec["layouts"], paid_action_budget=spec["native_budget"])
    config = protocol["local_route"]
    assert all(config[k]==v for k,v in spec["frozen_online_fields"].items())
    native.HOLDOUT = SCENES
    native.ARM_MODES = {"increment":("G","G")}
    old, values, initial, global_prior, _ = native.setup_runtime("holdout",layout,"increment",protocol)
    world, transform, packet, components, runtime, _, owner = values
    from runtime_binding import model as particle_module, revision, write

    assert global_prior == "G" and world.seed == initial["seed"] and world.step_count == 0
    mapper = runtime.states[0]["mapper"]
    initial_raw = raw_sensor_hashes(packet,array_sha)
    raw_chain = chain_hash("0"*64,packet.sha256())
    initial_belief = array_sha(mapper.belief)
    trace = []
    fork = None
    route_end = None
    active_option = None
    common_capture = None
    discover = arm == "probe"
    original_load = native.load_source

    def load(name, path):
        module = original_load(name,path)
        if Path(path) != native.METHOD:
            return module
        frozen_choice = module.choose_route

        def local_choice(state,native_option,mode,route_union_area,config):
            nonlocal fork, active_option
            paid = state["packet"].action_id
            try_fork = fork is None and paid <= spec["max_prefix_actions"]
            prior = "S" if try_fork else "G"
            route,audit = choose_route(module,frozen_choice,state,native_option,prior,weights,
                                       route_union_area,config,revision.reweight)
            if try_fork and audit["admitted_candidate_count"] >= 2:
                comparison = compare_routes(
                    state,audit["candidates"],spec,weights,config["score_equality_tolerance"],
                    rank_direct,rank_admitted,particle_module.REPAIR.classify_history)
                if comparison["direct"]["observed_support_voxels"] > 0:
                    fork_capture = capture(out,"fork",mapper,packet,raw_chain,paid,"measured_common_prefix")
                    payload = {
                        "status":"eligible_fork", "layout":layout, "paid_before":paid,
                        "packet_sha256":packet.sha256(), "packet_chain_sha256":raw_chain,
                        "initial_raw":initial_raw, "initial_belief_sha256":initial_belief,
                        "fork_belief_sha256":array_sha(mapper.belief), "prefix_trace":trace.copy(),
                        "fork_mesh_fields":fork_capture["native_mesh_fields"], "fork_receipt":fork_capture,
                        "target_pose":native_option["pose"], "option_id":native_option["option_id"],
                        "selection_call_id":native_option["selection_call_id"],
                        "return_states":native_option["return_states"],
                        "return_actions":native_option["return_actions"],
                        "candidate_pool_sha256":audit["candidate_pool_sha256"],
                        "assessed_candidates":audit["candidates"],
                        "common_clock":paid + min(row["outbound_cost"] for row in comparison["branch_routes"]),
                    } | comparison
                    if discover:
                        fork = payload
                        raise ForkCaptured
                    serialized = json.loads(json.dumps(payload,allow_nan=False))
                    for key,value in serialized.items():
                        if key != "fork_receipt":
                            assert value == sealed[key], f"Observed fork field changed: {key}"
                    picked = next(row for row in comparison["branch_routes"] if row["route_id"] == arm)
                    fork = payload
                    active_option = native_option["option_id"]
                    return module._same_goal_option(native_option,picked), audit | {
                        "selected_mode":"sealed_witness", "selected_route_id":arm,
                        "chosen_outbound_sha256":picked["outbound_sha256"],
                    }
            if prior == "G":
                return route,audit
            rows = audit["candidates"]
            baseline = next(row for row in rows if row["route_id"] == module.NATIVE_ID)
            winner,_ = rank_admitted(rows,baseline,"G",weights,mapper.config.resolution_m**2,
                                     config["score_equality_tolerance"])
            return module._same_goal_option(native_option,winner), audit | {
                "selected_mode":"G", "selected_route_id":winner["route_id"],
                "chosen_outbound_sha256":winner["outbound_sha256"],
            }

        module.choose_route = local_choice
        return module

    native.load_source = load
    full_local_route_adapter(components,old,"G",config,native.local_route_adapter)
    while not runtime.states[0]["closed"]:
        if fork is None and packet.action_id > spec["max_prefix_actions"]:
            break
        try:
            action = runtime.next_local_action(0)
        except ForkCaptured:
            assert discover and active_option is None
            break
        observed = runtime.states[0]
        pending, option = observed["pending"], observed["option"]
        same_option = option is not None and option["option_id"] == active_option
        if active_option and (not same_option or action is None):
            reason = next((event.get("reason") for event in reversed(runtime.audit)
                       if event.get("rejected_option_id") == active_option), None)
            route_end = {"status":"native_cancelled", "actual_paid":packet.action_id,
                         "reason":reason or "bound_option_no_longer_active"}
            break
        if action is None:
            break
        if fork is None and packet.action_id >= spec["max_prefix_actions"]:
            break
        assert pending["allowed"]
        raw,collision,done = world.step(action)
        packet = old.packet(world,transform,raw,layout,action,collision,done)
        owner = old.front.associate(mapper,raw,packet.frame_id)
        old.method.observe(mapper,packet.frame)
        runtime.observe(0,world.step_count,None,None,None,sensor_packet=packet)
        raw_chain = chain_hash(raw_chain,packet.sha256())
        pose = [*packet.position,packet.heading]
        row = raw_sensor_hashes(packet,array_sha) | {
            "action":action, "pose":pose, "collision":collision, "guard_reason":pending["reason"],
            "reserved_return_cost":pending["reserved_return_cost"], "owner_sha256":owner["owner_sha256"],
            "map_sha256":array_sha(mapper.belief), "path_m":world.moves*world.config.resolution_m,
            "option_id":None if option is None else option["option_id"],
        }
        trace.append(row)
        if not discover and fork is None:
            assert sealed["prefix_trace"][len(trace)-1] == row
        if fork is not None and packet.action_id == fork["common_clock"]:
            common_capture = capture(out,"common",mapper,packet,raw_chain,packet.action_id,
                                     "equal_paid_clock_on_the_bound_route")
        if fork is not None and (pose == fork["target_pose"] or collision or done):
            route_end = {"status":"collision" if collision else "arrived" if pose == fork["target_pose"] else "native_terminal",
                         "actual_paid":packet.action_id, "reason":pending["reason"]}
            break

    if discover:
        receipt = fork or {"status":"no_eligible_fork","layout":layout,"actual_paid":packet.action_id}
        write(out/"observed_route_bank.json", receipt | {"actor_seal_sha256":actor_seal,
              "no_candidate_witness_executed":True})
        print(layout,receipt["status"],receipt.get("paid_before"),
              len(receipt.get("branch_routes",[])),receipt.get("selected"),flush=True)
        return
    assert fork is not None and route_end is not None
    terminal = capture(out,"route_end",mapper,packet,raw_chain,packet.action_id,"actual_route_end")
    write(out/"steps.json",trace)
    write(out/"native_guard_audit.json",runtime.audit)
    files = {p.relative_to(out).as_posix():sha(p) for p in out.rglob("*") if p.is_file()}
    write(out/"manifest.json",{
        "status":"controlled_route_completed","layout":layout,"route_id":arm,
        "actor_seal_sha256":actor_seal,"route_seal_sha256":sha(ROOT/"ROUTE_FREEZE.json"),
        "candidate_pool_sha256":fork["candidate_pool_sha256"],"target_pose":fork["target_pose"],
        "first_packet_sha256":initial_raw["packet_sha256"],"fork_packet_sha256":fork["packet_sha256"],
        "same_observed_prefix_verified":True,"evaluator_inputs_read_by_actor":False,
        "frozen_common_clock":fork["common_clock"], "fork":fork["fork_receipt"],
        "route_end":route_end,"terminal":terminal,"common":common_capture,"actual_paid":packet.action_id,
        "collisions":world.collisions,"path_m":world.moves*world.config.resolution_m,"files_sha256":files,
    })
    print(layout,arm,route_end,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("layout")
    parser.add_argument("arm")
    parser.add_argument("out",type=Path)
    args = parser.parse_args()
    seal_path = ROOT/"INPUT_FREEZE.json"
    seal = read(seal_path)
    verify(seal["actor_files_sha256"],visible=True)
    verify(seal["simulator_cache_sha256"])
    route = None if args.arm=="probe" else read(ROOT/"ROUTE_FREEZE.json")
    if route is not None:
        assert route["inputs_sha256"]==sha(seal_path)
        assert any(row["route_id"]==args.arm for row in route["layouts"][args.layout]["branch_routes"])
    args.out.mkdir(parents=True,exist_ok=False)
    run(args.layout,args.arm,args.out,sha(seal_path),route)


if __name__ == "__main__":
    main()
