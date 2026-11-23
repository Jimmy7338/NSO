#!/usr/bin/env python3
"""Freeze new floors with a separate ledger for each information boundary."""

import argparse
import json
from pathlib import Path
import time

from geometry_audit import read, require, sha, verify_registration
from scene import ATTRIBUTION, BUDGET, HERE, LAYOUTS, REGISTRATION, SPECS


def fingerprint(paths):
    return {str(path): sha(path) for path in sorted(set(paths))}


def verify(files):
    for name, expected in files.items():
        require(sha(Path(name)) == expected, f"Frozen scene file changed: {name}")


def seal():
    registration = verify_registration()
    output = HERE.parent / "execution"
    require(not output.is_dir() or not any(path.is_dir() and path.name in LAYOUTS for path in output.rglob("*")),
            "A registered actor directory exists before scene freeze.")
    audit_file = HERE / "formal_geometry_audit.json"
    audit = read(audit_file)
    require(audit["status"] == "all_geometry_admitted" and audit["policy_calls"] == audit["actor_executions"] == 0
            and [row["layout"] for row in audit["layouts"]] == list(LAYOUTS) and not audit["excluded"],
            "Frozen cohort must consist only of the four registered and admitted geometry checks.")
    inherited = read(ATTRIBUTION / "actor_freeze_holdout.json")
    renderer_files = [Path(path) for path in inherited["files_sha256"]
                      if not Path(path).is_relative_to(ATTRIBUTION)]
    verify({str(path): inherited["files_sha256"][str(path)] for path in renderer_files})
    actor_files = [HERE / "scene.py", REGISTRATION, *renderer_files]
    evaluator_files = [
        audit_file, HERE / "geometry_audit.py", HERE / "prepare_scenes.py", HERE / "seal_scenes.py",
        HERE / "reference_protocol.json", HERE / "REGISTRATION_RECEIPT.json", ATTRIBUTION / "geometry_audit.py",
        ATTRIBUTION / "prepare_scenes.py", ATTRIBUTION / "reference_protocol.json",
    ]
    descriptions = []
    for layout in LAYOUTS:
        here, ref = (HERE / folder / layout for folder in ("prepared", "evaluator_references"))
        initial = read(here / "initial_state.json")
        receipt = read(ref / "RECEIPT.json")
        require(initial["seed"] == SPECS[layout]["seed"] and initial["stop_budget"] == BUDGET
                and initial["paid_actions"] == initial["policy_select_calls"] == 0,
                f"Cold native start does not reproduce the seal for {layout}.")
        require(not receipt["actor_executed"] and not receipt["actor_trajectory_read"],
                f"Source reference read actor outcomes for {layout}.")
        actor_files.extend(here / member for member in ("initial_state.json", "initial_frame.npz", "initial_scan.npz"))
        evaluator_files.extend(ref / member for member in (
            "RECEIPT.json", "cold_start_preflight.json", "evaluator_truth.npz", "fixed_reference.npz", "source_mesh.ply"))
        verify(receipt["source_inputs_sha256"])
        evaluator_files.extend(Path(p) for p in receipt["source_inputs_sha256"])
        descriptions.append({key: initial[key] for key in (
            "layout", "seed", "anchor", "shape", "stop_budget", "initial_packet_sha256",
            "initial_owner_sha256", "actual_config",
        )} | {"scene_role": "fresh_holdout"})
    caches = fingerprint((HERE / "precomputed_assets").glob("*"))
    require(len(caches) == 2 * len(LAYOUTS), "Unexpected simulator-only cache cardinality.")
    actor_hashes = fingerprint(actor_files)
    require(not any("evaluator_references" in path or "precomputed_assets" in path for path in actor_hashes),
            "Evaluator truth or the simulator-only cache entered the visible actor-input ledger.")
    now = time.time()
    require(now >= registration["frozen_unix_s"], "Scene seal predates registration.")
    common = {
        "split": "holdout", "layouts": descriptions, "actors_run": 0, "policy_calls": 0,
        "budget": BUDGET, "frozen_unix_s": now,
        "registry_sha256": sha(REGISTRATION), "geometry_audit_sha256": sha(audit_file),
        "seed_policy": "Each policy receives the same sensor bootstrap and fixed world seed within a floor.",
    }
    ledgers = {
        "actor": common | {
            "status": "actor_inputs_only", "files_sha256": actor_hashes,
            "simulator_only_collision_cache_sha256": caches,
            "runtime_interface": "Policy consumes native SensorPacket, acquired maps and action feedback.",
        },
        "evaluator": common | {
            "status": "reference_hidden_from_actor", "files_sha256": fingerprint(evaluator_files),
        },
    }
    for role, record in ledgers.items():
        with (HERE / f"{role}_freeze_holdout.json").open("x") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print("Frozen four new floors and separate actor, simulator-cache and evaluator inputs", flush=True)


def verify_seal():
    verify_registration()
    actor, evaluator = (read(HERE / f"{role}_freeze_holdout.json") for role in ("actor", "evaluator"))
    require(actor["layouts"] == evaluator["layouts"] and actor["actors_run"] == evaluator["actors_run"] == 0,
            "Actor and reference ledgers disagree on preregistration.")
    for ledger in (actor, evaluator):
        verify(ledger["files_sha256"])
    verify(actor["simulator_only_collision_cache_sha256"])
    print("Verified actor, simulator-only cache and offline evaluator SHA-256", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    verify_seal() if parser.parse_args().verify else seal()
