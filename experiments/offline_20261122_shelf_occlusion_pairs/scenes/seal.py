#!/usr/bin/env python3
"""Seal visible packets, simulator-internal caches and evaluator inputs separately."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from scene import ANALYSES, BUDGET, DEV_PARENTS, HERE, SOURCE, SPECS, TEST_PARENTS


NATIVE_ROOT = ANALYSES / "semantic_scene_redesign_20261119"
INITIAL_FILES = ("initial_state.json", "initial_frame.npz", "initial_scan.npz")
METRIC_FILES = ("RECEIPT.json", "cold_start_preflight.json", "evaluator_truth.npz",
                "fixed_reference.npz", "source_mesh.ply")
FILENAMES = {"development": "DEVELOPMENT_SCENE_FREEZE.json", "formal": "FORMAL_SCENE_FREEZE.json"}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def hashes(paths):
    return {str(path): digest(path) for path in sorted(set(paths))}


def read_pairs(name):
    parents = DEV_PARENTS if name == "development" else TEST_PARENTS
    layouts = [f"{parent}_{condition}" for parent in parents for condition in ("open", "screen")]
    audits = [HERE / "pair_audits" / f"{parent}.json" for parent in parents]
    return parents, layouts, audits


def verify(ledger):
    for role in ("actor_files_sha256", "simulator_only_cache_sha256", "evaluator_files_sha256"):
        for path, expected in ledger[role].items():
            assert digest(path) == expected, (role, path)


def seal(name):
    parents, layouts, audits = read_pairs(name)
    for directory in ("runs", "execution", "probes"):
        stage = HERE.parent / directory
        if stage.is_dir() and any(p.is_dir() and p.name in layouts for p in stage.rglob("*")):
            raise ValueError(f"Actors or probes for {name} exist before its scene freeze")
    source_audits = [read(path) for path in audits]
    assets = defaultdict(list)
    for parent, audit in zip(parents, source_audits, strict=True):
        assert audit["parent"] == parent and audit["policy_calls"] == 0
        assert audit["front_visible_screen_fraction"] < audit["front_visible_open_fraction"]
        assert audit["hidden_points_seen_by_side_pose"] >= 30
        assert audit["shortest_return_including_heading_actions"] <= BUDGET
        assert audit["paired_path_clearance_min_m"] >= audit["robot_radius_m"]
        assets[audit["shelf_asset"]].append(parent)
    assert len({audit["common_seed"] for audit in source_audits}) == len(parents)
    assert len({audit["occupancy_sha256"] for audit in source_audits}) == len(parents)
    optical_file = HERE / ("dev_geometry.json" if name == "development" else "test_geometry.json")
    write(optical_file, dict(role=name, parent_order=parents, placement_order=layouts, policy_calls=0,
                            independent_parents=len(parents), asset_reuse=dict(assets),
                            pairs=source_audits))
    actor = [
        HERE / "scene.py", SOURCE, ANALYSES / "dual_template_exploration_20261119/scene/controlled_world.py",
        NATIVE_ROOT / "scene_adapter/replicacad_world.py", NATIVE_ROOT / "assets/textured_raycast.py",
        NATIVE_ROOT / "assets/replicacad/assemble_apt0.py", NATIVE_ROOT / "assets/replicacad/download_receipt.json",
    ]
    offline = [HERE / "prepare.py", HERE / "reference_protocol.json", Path(__file__), optical_file, *audits]
    cache = {}
    for path in sorted((HERE / "precomputed_assets").glob("*.json")):
        spec = read(path)["spec"]
        if spec["layout_id"] in layouts and spec.get("fixture_source_sha256") == digest(HERE / "scene.py"):
            assert spec["layout_id"] not in cache
            cache[spec["layout_id"]] = [path, path.with_suffix(".npz")]
    assert set(cache) == set(layouts)
    descriptions = []
    for layout in layouts:
        folder = HERE / "prepared" / layout
        reference = HERE / "evaluator_references" / layout
        initial, receipt = read(folder / "initial_state.json"), read(reference / "RECEIPT.json")
        assert initial["seed"] == SPECS[layout]["seed"] and initial["stop_budget"] == BUDGET
        assert initial["paid_actions"] == initial["policy_select_calls"] == 0
        assert not receipt["actor_executed"] and not receipt["actor_trajectory_read"]
        for file, field in (("fixed_reference.npz", "reference_npz_sha256"),
                            ("source_mesh.ply", "source_mesh_sha256")):
            assert digest(reference / file) == receipt[field]
        for path, expected in receipt["source_inputs_sha256"].items():
            assert digest(path) == expected
            offline.append(Path(path))
        actor.extend(folder / file for file in INITIAL_FILES)
        offline.extend(reference / file for file in METRIC_FILES)
        descriptions.append({
            key: initial[key] for key in ("layout", "seed", "anchor", "shape", "stop_budget",
                                          "initial_packet_sha256", "initial_owner_sha256", "actual_config")
        } | {"parent": SPECS[layout]["parent_id"], "board": SPECS[layout]["condition"] == "screen",
             "shelf_asset": SPECS[layout]["asset"]})

    for parent in parents:
        a, b = (HERE / "evaluator_references" / f"{parent}_{condition}" for condition in ("open", "screen"))
        with np.load(a / "fixed_reference.npz", allow_pickle=False) as x, np.load(
            b / "fixed_reference.npz", allow_pickle=False
        ) as y:
            assert np.array_equal(x["open31"], y["open31"])
        assert read(a / "RECEIPT.json")["metadata"]["open31"]["area_per_sample_m2"] == read(
            b / "RECEIPT.json")["metadata"]["open31"]["area_per_sample_m2"]
    actor_files = hashes(actor)
    assert not any("precomputed_assets" in path or "evaluator_references" in path or "pair_audits" in path
                   or path.endswith("_geometry.json") for path in actor_files)
    ledger = dict(status="sealed_before_policy_execution", phase=name, parents=parents, placements=layouts,
                  scene_descriptions=descriptions, native_budget=BUDGET, policy_calls=0,
                  actor_files_sha256=actor_files,
                  simulator_only_cache_sha256=hashes(path for values in cache.values() for path in values),
                  evaluator_files_sha256=hashes(offline), frozen_unix_s=time.time())
    verify(ledger)
    write(HERE / FILENAMES[name], ledger)
    print(f"Sealed {len(parents)} {name} parent units and {len(layouts)} placements", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=tuple(FILENAMES))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify(read(HERE / FILENAMES[args.name]))
        print(f"Verified {args.name} scene seal")
    else:
        seal(args.name)
