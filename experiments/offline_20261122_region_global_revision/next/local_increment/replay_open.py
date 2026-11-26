#!/usr/bin/env python3
"""Re-materialize an exploratory map at the shortest observed open-branch clock."""

import argparse
from pathlib import Path

import run as actor
from measure import evaluator, measure_snapshot
from seal import write


ROOT = actor.ROOT
LAYOUT = "increment_open"


def open_archive():
    plan = actor.read(ROOT/"ROUTE_FREEZE.json")
    rows = plan["layouts"][LAYOUT]["branch_routes"]
    folders = {row["route_id"]:ROOT/"runs"/LAYOUT/row["route_id"]/"run" for row in rows}
    runs = {name:actor.read(path/"manifest.json") for name,path in folders.items()}
    for name, run in runs.items():
        assert run["route_seal_sha256"] == actor.sha(ROOT/"ROUTE_FREEZE.json")
        actor.verify({str(folders[name]/key):digest for key,digest in run["files_sha256"].items()})
    clock = min(run["actual_paid"] for run in runs.values())
    assert clock < plan["layouts"][LAYOUT]["common_clock"]
    return plan,folders,runs,clock


def replay(route):
    plan,folders,archive,clock = open_archive()
    input_path = ROOT/"INPUT_FREEZE.json"
    seal = actor.read(input_path)
    actor.verify(seal["actor_files_sha256"],visible=True)
    actor.verify(seal["simulator_cache_sha256"])
    destination = ROOT/"exploratory_a19"/route
    destination.mkdir(parents=True,exist_ok=False)
    original_trace = actor.read(folders[route]/"steps.json")
    native_setup = actor.native.setup_runtime
    chain = actor.chain_hash("0"*64,archive[route]["first_packet_sha256"])

    def setup(*args):
        values = native_setup(*args)
        _, (_,_,_,_,runtime,_,_), _,_,_ = values
        mapper = runtime.states[0]["mapper"]
        native_observe = runtime.observe

        def observe(*args, **kwargs):
            nonlocal chain
            packet = kwargs["sensor_packet"]
            old = original_trace[packet.action_id-1]
            actual = actor.raw_sensor_hashes(packet,actor.array_sha)
            assert all(actual[key]==old[key] for key in actual)
            receipt = native_observe(*args,**kwargs)
            assert actor.array_sha(mapper.belief) == old["map_sha256"]
            chain = actor.chain_hash(chain,packet.sha256())
            if packet.action_id == clock:
                actor.capture(destination,"exploratory_clock",mapper,packet,chain,clock,
                              "post_outcome_common_clock_reproduced_from_archived_execution")
            return receipt

        runtime.observe = observe
        return values

    actor.native.setup_runtime = setup
    actor.run(LAYOUT,route,destination,actor.sha(input_path),plan)
    rematerialized = actor.read(destination/"manifest.json")
    assert actor.read(destination/"steps.json") == original_trace
    assert rematerialized["route_end"] == archive[route]["route_end"]
    assert rematerialized["terminal"]["native_mesh_fields"] == archive[route]["terminal"]["native_mesh_fields"]
    assert rematerialized["terminal"]["packet_chain_sha256"] == archive[route]["terminal"]["packet_chain_sha256"]
    write(destination/"exploratory_receipt.json",{
        "purpose":"Exploratory common elapsed clock after seeing which routes censored the planned a47.",
        "clock":clock,"route":route,"archive_manifest_sha256":actor.sha(folders[route]/"manifest.json"),
        "replay_manifest_sha256":actor.sha(destination/"manifest.json"),
        "snapshot_receipt_sha256":actor.sha(destination/"exploratory_clock/checkpoint.json"),
        "all_actual_executed_sensors_and_map_hashes_verified":True,"original_route_end_preserved":True,
    })


def measure():
    plan,folders,archive,clock = open_archive()
    weight,metric,contract = evaluator()
    source = weight.load_reference(
        LAYOUT,actor.read(actor.SCENES/"prepared"/LAYOUT/"initial_state.json"))
    results = {}
    for route, manifest in archive.items():
        before = manifest["fork"]
        _,base = measure_snapshot(folders[route],"fork",before,before["actual_paid"],source,weight,metric)
        out = ROOT/"exploratory_a19"/route
        proof = actor.read(out/"exploratory_receipt.json")
        assert proof["clock"]==clock and proof["archive_manifest_sha256"]==actor.sha(folders[route]/"manifest.json")
        at = actor.read(out/"exploratory_clock/checkpoint.json")
        assert actor.sha(out/"exploratory_clock/checkpoint.json")==proof["snapshot_receipt_sha256"]
        _,measured = measure_snapshot(out,"exploratory_clock",at,clock,source,weight,metric)
        results[route] = contract.facility_metrics(base["shelf"],measured["shelf"],source["weights"]["shelf"])
    maximum = max(row["net_m2"] for row in results.values())
    selected = plan["layouts"][LAYOUT]["selected"]
    report = {
        "status":"exploratory_post_outcome_clock_distinct_from_frozen_measurement",
        "paid":clock,"frozen_primary_clock":plan["layouts"][LAYOUT]["common_clock"],
        "results":results,"selected":selected,
        "regret_m2":{arm:maximum-results[route]["net_m2"] for arm,route in selected.items()},
    }
    write(ROOT/"exploratory_a19/summary.json",report)
    for route,row in results.items():
        print(route,row["net_m2"],flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route")
    request = parser.parse_args().route
    measure() if request=="measure" else replay(request)
