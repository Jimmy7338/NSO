#!/usr/bin/env python3
"""Seal, execute and measure separate short route witnesses at two archived forks."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from fractions import Fraction
from hashlib import file_digest
from itertools import combinations
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
STUDY = ROOT.parents[1]
ANALYSES = STUDY.parent
REVISION = ANALYSES / "region_global_revision_20261122"
ARCHIVE = REVISION / "next/local_increment"
REQUEST = ROOT / "REQUEST.json"
FREEZE = ROOT / "WITNESS_FREEZE.json"
ORIGINAL_BANK = ARCHIVE / "ROUTE_FREEZE.json"
ORIGINAL_INPUTS = ARCHIVE / "INPUT_FREEZE.json"
POLICY_ARMS = ("D", "D+", "S")
METRIC_CODE = ANALYSES / "route_intervention_20261122/evaluation/evaluate.py"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def write(path, data):
    assert not path.exists(), path
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def verify_bound(path_digests, root=None):
    for path, wanted in path_digests.items():
        full = Path(path) if root is None else root / path
        if sha(full) != wanted:
            raise ValueError(f"Archived evidence changed: {full}")


def geometric_bank(eligible, settings):
    """Select by admission, route geometry, exact action cost and hash alone."""
    native = next(r for r in eligible if r["route_id"] == settings["geometry_reference_route"])
    costs = sorted({r["outbound_cost"] for r in eligible if r["outbound_cost"] > native["outbound_cost"]})
    costs = costs[:settings["cost_tiers_after_cheapest"]]
    assert len(costs) == settings["cost_tiers_after_cheapest"] and settings["routes_per_tier"] == 2
    chosen = [native["route_id"]]
    tiers = []
    for cost in costs:
        rows = sorted((r for r in eligible if r["outbound_cost"] == cost), key=lambda r:r["outbound_sha256"])
        horizon = settings["geometry_horizon"]
        geometry = {r["route_id"]:set(map(tuple, r["outbound_states"][1:horizon + 1])) for r in rows}
        assert len(rows) >= 2 and all(geometry.values())

        def distance(pair):
            left, right = (geometry[r["route_id"]] for r in pair)
            return Fraction(len(left ^ right), len(left | right))

        # combinations respects the sorted order and max keeps the first tied pair.
        pair = max(combinations(rows, 2), key=distance)
        gap = distance(pair)
        route_ids = [r["route_id"] for r in pair]
        chosen.extend(route_ids)
        tiers.append({"outbound_cost": cost, "eligible_in_tier": len(rows), "route_ids": route_ids,
                      "exact_distance": [gap.numerator, gap.denominator]})
    return chosen, tiers


def seal():
    spec = read(REQUEST)
    pub_path = REVISION / "publication/followup/receipt.json"
    assert read(pub_path)["head"] == spec["published_source_commit"]
    original = read(ORIGINAL_BANK)
    assert original["inputs_sha256"] == sha(ORIGINAL_INPUTS)
    requested = spec["placements"]
    assert list(original["layouts"]) == requested
    attempts = read(ARCHIVE / "execution_receipt.json")["episodes"]
    assert all(r["status"] == 0 and r["route_freeze_sha256"] == sha(ORIGINAL_BANK) for r in attempts)
    by_assignment = {(r["layout"], r["route_id"]):r for r in attempts}
    result = {}
    for layout in requested:
        fork = original["layouts"][layout]
        assert fork["status"] == "eligible_fork"
        eligible = fork["eligible_full_pool"]
        assert all(r["admitted"] and r["goal_pose"] == fork["target_pose"]
                   and r["outbound_states"][-1] == fork["target_pose"]
                   and len(r["outbound_actions"]) == r["outbound_cost"]
                   and fork["paid_before"] + r["outbound_cost"] + r["reserved_return_cost"] <= spec["native_budget"]
                   for r in eligible)
        assert len({r["outbound_sha256"] for r in eligible}) == len(eligible)
        assert len({r["route_id"] for r in eligible}) == len(eligible)
        geometric, choices = geometric_bank(eligible, spec["witness_bank"])
        nominations = {arm:fork["selected"][arm] for arm in POLICY_ARMS}
        union = set(geometric) | set(nominations.values())
        routes = [r for r in eligible if r["route_id"] in union]
        assert len(routes) == len(union)
        min_clock = fork["paid_before"] + min(r["outbound_cost"] for r in routes)
        assert min_clock == fork["common_clock"]
        checkpoint_paid = {f"paid_{offset}": fork["paid_before"] + offset
                           for offset in spec["checkpoints_after_fork"]}
        assert all(paid <= min_clock for paid in checkpoint_paid.values())
        selections = []
        for row in routes:
            route_id = row["route_id"]
            attempt = by_assignment.get((layout, route_id))
            reuse = None
            if attempt:
                previous = ARCHIVE / "runs" / layout / route_id
                manifest = previous / "run/manifest.json"
                assert sha(manifest) == attempt["manifest_sha256"]
                verify_bound(read(manifest)["files_sha256"], manifest.parent)
                receipt = previous / "exit.json"
                assert read(receipt) == attempt
                reuse = {"run": str(manifest.parent), "manifest_sha256": sha(manifest),
                         "exit_sha256": sha(receipt), "wall_s": attempt["finished_unix_s"] - attempt["started_unix_s"]}
            selections.append({"route_id": route_id, "outbound_sha256": row["outbound_sha256"],
                               "nominal_outbound": row["outbound_cost"], "geometric_witness": route_id in geometric,
                               "policies": [a for a in POLICY_ARMS if nominations[a] == route_id],
                               "archived_run": reuse})
        result[layout] = {"fork_paid":fork["paid_before"], "fork_packet_sha256":fork["packet_sha256"],
                          "fork_candidate_pool_sha256":fork["candidate_pool_sha256"],
                          "target_pose":fork["target_pose"], "n_legal_native":len(eligible),
                          "nominal_min_clock":min_clock, "fixed_paid_clocks":checkpoint_paid,
                          "geometry_tiers":choices, "policies":nominations, "routes":selections}
    actor_sources = (Path(__file__), REQUEST, ORIGINAL_BANK, ORIGINAL_INPUTS, ARCHIVE / "run.py")
    check_sources = (ARCHIVE / "measure.py", METRIC_CODE, pub_path, ARCHIVE / "execution_receipt.json")
    actors = {str(path):sha(path) for path in actor_sources}
    evaluator = {str(path):sha(path) for path in check_sources}
    write(FREEZE, {"phase":spec["phase"], "request_sha256":sha(REQUEST), "actor_files_sha256":actors,
                   "evaluator_files_sha256":evaluator, "published_source_commit":spec["published_source_commit"],
                   "placements":result})
    print("Sealed", sha(FREEZE), [(name,len(case["routes"])) for name,case in result.items()], flush=True)


def sealed():
    locked = read(FREEZE)
    assert locked["phase"] == "development_only" and locked["request_sha256"] == sha(REQUEST)
    verify_bound(locked["actor_files_sha256"])
    return locked


def registered_formal_completion():
    """The parent must finish the registered formal stages before supplemental runs."""
    formal_path = STUDY / "FORMAL_FREEZE.json"
    formal = read(formal_path)
    path = STUDY / "execution/formal/formal_rest_receipt.json"
    receipt = read(path)
    assert receipt["stage"] == "formal_rest" and receipt["phase_freeze_sha256"] == sha(formal_path)
    assert receipt["placements"] == formal["stage_2"]
    verify_bound({row["path"]: row["sha256"] for row in receipt["placement_receipts"]})
    result = STUDY / "results/formal/summary.json"
    measured = read(result)
    assert measured["phase"] == "formal" and measured["phase_freeze_sha256"] == sha(formal_path)
    assert measured["parents"] == formal["parents"]
    assert measured["registered_policy_assignments"] == len(formal["placements"]) * len(formal["arm_order"])
    return {"formal_receipt_path":str(path), "formal_receipt_sha256":sha(path),
            "formal_results_path":str(result), "formal_results_sha256":sha(result)}


def spawn_archived_module(name):
    sys.path.insert(0, str(ARCHIVE))
    return __import__(name)


def verify_archived_replay(out, previous, current):
    old = Path(previous["run"])
    assert previous["manifest_sha256"] == sha(old / "manifest.json")
    assert previous["exit_sha256"] == sha(old.parent / "exit.json")
    archived = read(old / "manifest.json")
    verify_bound(archived["files_sha256"], old)
    for key in ("layout", "route_id", "first_packet_sha256", "fork_packet_sha256", "route_end",
                "actual_paid", "path_m", "collisions", "candidate_pool_sha256"):
        assert archived[key] == current[key], key
    assert sha(old / "steps.json") == sha(out / "steps.json")
    assert sha(old / "native_guard_audit.json") == sha(out / "native_guard_audit.json")
    for name in ("fork", "common", "route_end"):
        for filename in ("checkpoint.json", "belief.npz", "native_mesh.ply"):
            existing = old / name / filename
            assert (out / name / filename).exists() == existing.exists()
            if existing.exists():
                assert sha(existing) == sha(out / name / filename), existing


def execute_one(layout, route_id, out):
    assert registered_formal_completion()
    locked = sealed()
    case = locked["placements"][layout]
    row = next(r for r in case["routes"] if r["route_id"] == route_id)
    actor = spawn_archived_module("run")
    source_seal = read(ORIGINAL_INPUTS)
    actor.verify(source_seal["actor_files_sha256"],visible=True)
    actor.verify(source_seal["simulator_cache_sha256"])
    source_bank = read(ORIGINAL_BANK)
    fork = source_bank["layouts"][layout]
    assert fork["packet_sha256"] == case["fork_packet_sha256"]
    assert fork["candidate_pool_sha256"] == case["fork_candidate_pool_sha256"]
    expected = {r["route_id"]:r for r in case["routes"]}
    fork["branch_routes"] = [r for r in fork["eligible_full_pool"] if r["route_id"] in expected]
    original_scores = actor.compare_routes
    original_setup = actor.native.setup_runtime
    chain = actor.chain_hash("0" * 64, fork["initial_raw"]["packet_sha256"])
    extra = case["fixed_paid_clocks"]
    captures = {paid: name for name, paid in extra.items()}
    out.mkdir(parents=True, exist_ok=False)

    def select_envelope(*args):
        comparison = original_scores(*args)
        branch = [r for r in comparison["eligible_full_pool"] if r["route_id"] in expected]
        for candidate in branch:
            assert candidate["outbound_sha256"] == expected[candidate["route_id"]]["outbound_sha256"]
        assert set(expected) == {r["route_id"] for r in branch}
        return comparison | {"branch_routes":branch}

    def setup(*args):
        values = original_setup(*args)
        _, (_, _, packet, _, runtime, _, _), _, _, _ = values
        assert packet.sha256() == fork["initial_raw"]["packet_sha256"]
        observe_original = runtime.observe
        mapper = runtime.states[0]["mapper"]

        def capture_paid(*args, **kwargs):
            nonlocal chain
            receipt = observe_original(*args, **kwargs)
            new_packet = kwargs["sensor_packet"]
            paid = new_packet.action_id
            chain = actor.chain_hash(chain, new_packet.sha256())
            if paid in captures:
                actor.capture(out, captures[paid], mapper, new_packet, chain, paid, "registered_witness_equal_clock")
            return receipt

        runtime.observe = capture_paid
        return values

    actor.compare_routes, actor.native.setup_runtime = select_envelope, setup
    try:
        actor.run(layout, route_id, out, sha(ORIGINAL_INPUTS), source_bank)
        native = read(out / "manifest.json")
        if row["archived_run"]:
            verify_archived_replay(out, row["archived_run"], native)
        dispositions = {
            name: {"status": "measured" if native["actual_paid"] >= paid else "stopped_before_clock",
                   "requested_paid": paid, "checkpoint_sha256": sha(out/name/"checkpoint.json")
                   if native["actual_paid"] >= paid else None}
            for name, paid in extra.items()
        }
        write(out / "witness_receipt.json", {"layout":layout,"route_id":route_id,"freeze_sha256":sha(FREEZE),
              "native_manifest_sha256":sha(out/"manifest.json"), "archived_replay_verified":bool(row["archived_run"]),
              "paid_checkpoints":dispositions,"original_route_outcome":native["route_end"]})
    except BaseException as error:
        write(out/"technical_failure.json", {"error":repr(error),"traceback":traceback.format_exc()})
        raise


def launch_attempt(layout, route_id, attempt, authorization):
    attempt.mkdir()
    command = [sys.executable, str(Path(__file__)), "worker", layout, route_id, str(attempt/"run")]
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               PYTHONFAULTHANDLER="1")
    log = attempt / "worker.log"
    with log.open("x") as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        launch = {"layout":layout,"route_id":route_id,"pid":child.pid,
                  "start_unix_s":time.time(),"freeze_sha256":sha(FREEZE),"command":command} | authorization
        write(attempt / "launch.json", launch)
        code = child.wait()
    evidence = attempt / "run/witness_receipt.json"
    status = "verified" if code == 0 and evidence.is_file() else "technical_failure"
    record = launch | {"status":status, "exit_code":code, "finished_unix_s":time.time(),
                      "worker_log_sha256":sha(log),"witness_receipt_sha256":sha(evidence) if evidence.is_file() else None,
                      "directory":str(attempt/"run")}
    write(attempt / "exit.json", record)
    print(layout, route_id, status, flush=True)
    return record


def execute_attempt(layout, row, authorization):
    route_id = row["route_id"]
    base = ROOT / "runs" / layout / route_id
    base.mkdir(parents=True, exist_ok=True)
    attempts = sorted(base.glob("attempt_*"))
    previous = [read(attempt / "exit.json") for attempt in attempts]
    for record in previous:
        assert record["freeze_sha256"] == sha(FREEZE)
        assert all(record[name] == value for name,value in authorization.items())
    records = list(previous)
    for index in range(len(attempts), 2):
        if records and records[-1]["status"] == "verified":
            break
        attempt = base / f"attempt_{index+1:03d}"
        records.append(launch_attempt(layout, route_id, attempt, authorization))
    current = records[-1]
    if current["status"] == "verified":
        proof = Path(current["directory"]) / "witness_receipt.json"
        assert sha(proof) == current["witness_receipt_sha256"]
    return current | {"attempt_exit_sha256": {
        str(Path(record["directory"]).parent / "exit.json"):
        sha(Path(record["directory"]).parent / "exit.json") for record in records
    }}


def run(workers):
    assert 1 <= workers <= 3
    authorization = registered_formal_completion()
    locked = sealed()
    receipt = ROOT / "execution_receipt.json"
    assert not receipt.exists()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        tasks = [pool.submit(execute_attempt, layout, row, authorization)
                 for layout, case in locked["placements"].items() for row in case["routes"]]
        outcomes = [f.result() for f in as_completed(tasks)]
    outcomes.sort(key=lambda r:(r["layout"],r["route_id"]))
    write(receipt, {"freeze_sha256":sha(FREEZE), "outcomes":outcomes} | authorization)


def measure():
    locked = sealed()
    verify_bound(locked["evaluator_files_sha256"])
    verify_bound(read(ORIGINAL_INPUTS)["evaluator_files_sha256"])
    old = spawn_archived_module("measure")
    weighted, metric, checks = old.evaluator()
    executed = read(ROOT/"execution_receipt.json")
    assert executed["freeze_sha256"] == sha(FREEZE)
    verify_bound({executed["formal_receipt_path"]:executed["formal_receipt_sha256"],
                  executed["formal_results_path"]:executed["formal_results_sha256"]})
    observed = {(row["layout"],row["route_id"]):row for row in executed["outcomes"]}
    original = read(ORIGINAL_BANK)
    output = ROOT / "results"
    output.mkdir(exist_ok=False)
    rows, sets = [], {}
    for layout, case in locked["placements"].items():
        fork = original["layouts"][layout]
        candidates = {r["route_id"]:r for r in fork["eligible_full_pool"]}
        initial = read(old.SCENES / "prepared" / layout / "initial_state.json")
        reference = weighted.load_reference(layout, initial)
        clocks = case["fixed_paid_clocks"] | {"nominal_min":case["nominal_min_clock"], "outbound_event":None}
        measurements = {clock:{} for clock in clocks}
        for selected in case["routes"]:
            route_id = selected["route_id"]
            record = observed[layout, route_id]
            directory = Path(record["directory"])
            verify_bound(record["attempt_exit_sha256"])
            assert read(directory.parent/"exit.json") == {
                key:value for key,value in record.items() if key != "attempt_exit_sha256"
            }
            verify_bound({str(directory.parent/"worker.log"):record["worker_log_sha256"]})
            row = {"placement":layout, "route_id":route_id, "nominal_outbound":selected["nominal_outbound"],
                   "geometric_witness":selected["geometric_witness"], "policies":"/".join(selected["policies"]),
                   "archived_replay":bool(selected["archived_run"]), "process_status":record["status"]}
            if record["status"] == "verified":
                audit = read(directory / "witness_receipt.json")
                native_path = directory / "manifest.json"
                assert sha(directory/"witness_receipt.json") == record["witness_receipt_sha256"]
                assert audit["freeze_sha256"] == sha(FREEZE)
                assert audit["native_manifest_sha256"] == sha(native_path)
                native = read(native_path)
                verified = old.measure_route(directory,layout,fork,candidates[route_id],weighted,metric,checks,reference)
                _, baseline = old.measure_snapshot(directory,"fork",native["fork"],case["fork_paid"],
                                                    reference,weighted,metric)
                row |= {"actual_paid":native["actual_paid"], "route_outcome":native["route_end"]["status"],
                        "guard_reason":native["route_end"]["reason"], "path_m":native["path_m"],
                        "collisions":native["collisions"], "supported_hidden_m2":verified["supported_hidden_m2"]}
                values = {"outbound_event":verified["metrics"]["route_end"],
                          "nominal_min":verified["metrics"]["common"]}
                for name, paid in case["fixed_paid_clocks"].items():
                    stamp = audit["paid_checkpoints"][name]
                    assert stamp["requested_paid"] == paid
                    if stamp["status"] == "measured":
                        receipt = directory / name / "checkpoint.json"
                        assert stamp["checkpoint_sha256"] == sha(receipt)
                        geometry, match = old.measure_snapshot(directory,name,read(receipt),paid,reference,weighted,metric)
                        values[name] = {"paid":paid,"C_free":geometry["C_free"],"F1_05cm":geometry["F1_scene_05cm"]}
                        values[name] |= {item:checks.facility_metrics(baseline[item],match[item],reference["weights"][item])
                                         for item in ("shelf","cabinet")}
                    else:
                        assert native["actual_paid"] < paid
                        values[name] = None
            else:
                values = dict.fromkeys(clocks)
            for clock, paid in clocks.items():
                value = values[clock]
                if value is not None:
                    status = "measured"
                    measurements[clock][route_id] = value["shelf"]["net_m2"]
                elif record["status"] != "verified":
                    status = "technical_failure"
                elif row["route_outcome"] == "native_cancelled":
                    status = "guard_cancelled_before_clock"
                else:
                    status = "other_native_stop_before_clock"
                rows.append(row | {"clock":clock, "measurement_status":status,"requested_paid":paid,
                    "shelf_net_m2":value["shelf"]["net_m2"] if value else None,
                    "shelf_gross_m2":value["shelf"]["gross_new_m2"] if value else None,
                    "shelf_lost_m2":value["shelf"]["lost_m2"] if value else None,
                    "cabinet_net_m2":value["cabinet"]["net_m2"] if value else None,
                    "C_free":value["C_free"] if value else None,"F1_05cm":value["F1_05cm"] if value else None})
        detail = {}
        for clock, paid in clocks.items():
            measured = measurements[clock]
            cohort = [r for r in rows if r["placement"] == layout and r["clock"] == clock]
            assert len(cohort) == len(case["routes"])
            geometric = [r["route_id"] for r in cohort if r["geometric_witness"]]
            # Individual outbound events generally differ in paid clock. They are never ranked here.
            best = max(measured.values(),default=None) if paid is not None else None
            geometry_values = [measured[r] for r in geometric if r in measured]
            best_geometry = max(geometry_values,default=None) if paid is not None else None
            detail[clock] = {
                "requested_paid":paid,"registered":len(cohort),"measured":len(measured),
                "geometry_registered":len(geometric),"geometry_measured":len(geometry_values),
                "guard_cancelled_before_clock":sum(r["measurement_status"]=="guard_cancelled_before_clock" for r in cohort),
                "other_native_stop_before_clock":sum(r["measurement_status"]=="other_native_stop_before_clock" for r in cohort),
                "technical_failures":sum(r["measurement_status"]=="technical_failure" for r in cohort),
                "measured_best_union_m2":best,"measured_best_geometric_m2":best_geometry,
                "complete_union":len(measured)==len(cohort),"complete_geometric":len(geometry_values)==len(geometric),
                "policies":{arm:{"route_id":route,"net_shelf_m2":measured.get(route),
                    "regret_to_measured_union_best_m2":best-measured[route] if best is not None and route in measured else None,
                    "net_minus_measured_geometry_best_m2":measured[route]-best_geometry
                        if best_geometry is not None and route in measured else None}
                    for arm,route in case["policies"].items()},
            }
        sets[layout] = {"n_legal_native":case["n_legal_native"],"fork_paid":case["fork_paid"],"checkpoints":detail}
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (output/"routes.csv").open("x",newline="") as stream:
        table = csv.DictWriter(stream, fieldnames=fields)
        table.writeheader()
        table.writerows(rows)
    write(output/"summary.json", {"phase":"development_only","freeze_sha256":sha(FREEZE),
          "execution_receipt_sha256":sha(ROOT/"execution_receipt.json"),"routes_csv_sha256":sha(output/"routes.csv"),
          "placements":sets})


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    commands.add_parser("seal")
    commands.add_parser("measure")
    supervisor = commands.add_parser("run")
    supervisor.add_argument("--workers", type=int, default=3)
    worker = commands.add_parser("worker")
    worker.add_argument("layout")
    worker.add_argument("route")
    worker.add_argument("output", type=Path)
    args = cli.parse_args()
    if args.command == "seal":
        seal()
    elif args.command == "measure":
        measure()
    elif args.command == "run":
        run(args.workers)
    else:
        execute_one(args.layout, args.route, args.output)


if __name__ == "__main__":
    main()
