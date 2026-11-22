#!/usr/bin/env python3
"""Reproduce two archived local interventions without opening any new layout."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from weighted_run import ANALYSES, HERE, PRIOR, SOURCE_PROTOCOL, native, raw_sensor_hashes, read
from weighted_score import Weights, choose_route, rank_admitted

ARCHIVE = ANALYSES / "same_goal_two_witness_20261121/runner/runs"
WITNESSES = ("gallery_left:56:230", "warehouse_right:142:578")
RAW_FIELDS = ("depth_values_sha256", "rgb_values_sha256", "scan_ranges_sha256")


def write(path, values):
    with path.open("x") as stream:
        json.dump(values, stream, indent=2, allow_nan=False)
        stream.write("\n")


def synthetic_check():
    rows = [
        {"route_id": "native", "outbound_cost": 8, "outbound_sha256": "a", "admitted": True,
         "scores": {"G": 1.0}, "unknown_union_cells": 100},
        {"route_id": "via_extra", "outbound_cost": 10, "outbound_sha256": "b", "admitted": True,
         "scores": {"G": 0.95}, "unknown_union_cells": 110},
        {"route_id": "rejected", "outbound_cost": 10, "outbound_sha256": "c", "admitted": False,
         "scores": {"G": 500.0}, "unknown_union_cells": 5},
    ]
    rank = lambda weights: rank_admitted(rows, rows[0], "G", weights, .2 ** 2, 1e-9)[0]["route_id"]
    assert rank(Weights()) == "native"
    assert rank(Weights(.25, 0)) == "via_extra"
    assert rank(Weights(.25, .20)) == "native"
    return dict(zero="native", positive_unknown_gain="via_extra",
                additional_cost_restores_native=True, rejected_high_area_never_selected=True)


def compare_table(audit, archived, expected_priors):
    assert audit["candidate_pool_sha256"] == archived["candidate_pool_sha256"]
    assert len(audit["candidates"]) == len(archived["candidates"])
    for actual, expected in zip(audit["candidates"], archived["candidates"], strict=True):
        assert {k: v for k, v in actual.items() if k != "scores"} == {
            k: v for k, v in expected.items() if k != "scores"
        }
        assert {k: v for k, v in actual["scores"].items() if k in expected_priors} == {
            k: v for k, v in expected["scores"].items() if k in expected_priors
        }


def one(witness, prior):
    layout, paid_text, call_text = witness.split(":")
    paid, call_id = int(paid_text), int(call_text)
    expected = read(PRIOR / "PROTOCOL.json")["engineering_regression_before_holdout"][witness]
    archive = ARCHIVE / layout / prior
    manifest = read(archive / "manifest.json")
    for name in ("steps.json", "local_choices.json", "intervention.json"):
        assert manifest["files_sha256"][name] == native.sha(archive / name)
    archived_steps = read(archive / "steps.json")
    archived_selection = next(e for e in read(archive / "local_choices.json")
                              if e["selection_call_id"] == call_id)
    source = read(SOURCE_PROTOCOL)
    old, values, _, global_mode, _ = native.setup_runtime("development", layout, "newG", source)
    world, transform, packet, components, runtime, _, owner = values
    mapper = runtime.states[0]["mapper"]
    backend = components._cpu_backend
    assert global_mode == "G" and packet.sha256() == manifest["initial_packet_sha256"]
    raw_get = components.get_selected_option
    native.local_route_adapter(components, old, "G", source["local_route"])
    ordinary = components.get_selected_option
    legacy = native.load_source("nso_weighted_parity_source", native.METHOD)
    from runtime_binding import revision
    checked = {}

    def at_witness(index):
        event = backend.calls[-1]
        if event["action_id"] != paid:
            return ordinary(index)
        assert event["call_id"] == call_id and event["method"] == "select_topo_target"
        unweighted_native = raw_get(index)
        state = backend.scenes[index]
        config = dict(source["local_route"], max_alternates=4 * mapper.belief.size + 1)
        selected, audit = choose_route(legacy, legacy.choose_route, state, unweighted_native,
                                       prior, Weights(), old.method.route_union_area, config, revision.reweight)
        archived = archived_selection["choice_audit"]
        assert audit["selected_route_id"] == archived["selected_route_id"] == expected["winners"][prior]
        assert audit["chosen_outbound_sha256"] == archived["chosen_outbound_sha256"]
        assert selected["outbound_actions"] == archived_selection["selected_outbound_actions"]
        assert selected["outbound_states"] == archived_selection["selected_outbound_states"]
        assert "remaining_candidates_after_admitted_pool_cap" not in audit["proposal_diagnostics"]["exclusions"]
        compare_table(audit, archived, {"G"} if prior == "G" else {"G", "S"})
        ranking = audit["weighted_ranking"]
        checked.update(paid_before=paid, selected_route_id=audit["selected_route_id"],
                       exact_outbound_trace_sha256=audit["chosen_outbound_sha256"],
                       exact_legacy_candidate_pool_sha256=audit["candidate_pool_sha256"],
                       exact_legacy_unweighted_area_columns=sorted({"G"} if prior == "G" else {"G", "S"}),
                       local_area_provider_calls=ranking["local_area_provider_calls"])
        if prior == "G":
            assert not ranking["local_area_provider_calls"]["S"] and not ranking["local_area_provider_calls"]["X"]
            rows = archived["candidates"]
            native_row = next(r for r in rows if r["route_id"] == "native")
            altered = [dict(row, scores=dict(row["scores"], S=600.0-i)) for i, row in enumerate(rows)]
            for weights in (Weights(), Weights(.25, .05)):
                left, left_scores = rank_admitted(rows, native_row, "G", weights, .2 ** 2, 1e-9)
                right, right_scores = rank_admitted(altered, native_row, "G", weights, .2 ** 2, 1e-9)
                assert left["route_id"] == right["route_id"] and left_scores == right_scores
            checked["neutral_replay_ignores_changed_category_column"] = True
        if prior == "S" and layout == "gallery_left":
            _, swapped = choose_route(legacy, legacy.choose_route, state, unweighted_native, "X",
                                      Weights(), old.method.route_union_area, config, revision.reweight)
            compare_table(swapped, archived, {"G"})
            x_ranking = swapped["weighted_ranking"]
            assert not x_ranking["local_area_provider_calls"]["S"]
            for entry in x_ranking["observed_instance_shape_prior_exchange"]:
                assert entry["S"] == entry["X"][::-1]
            checked["exchanged_prior_validation"] = dict(
                identical_frozen_pool=True, category_column_reversed=True,
                selected_route_id=swapped["selected_route_id"], swaps=x_ranking["observed_instance_shape_prior_exchange"])
        return selected

    components.get_selected_option = at_witness
    stop = expected["paid_first_divergence"]
    for row in archived_steps[:stop]:
        action = runtime.next_local_action(0)
        assert action == row["action"] and runtime.states[0]["pending"]["allowed"]
        frame, collision, done = world.step(action)
        assert not collision
        packet = old.packet(world, transform, frame, layout, action, collision, done)
        owner = old.front.associate(mapper, frame, packet.frame_id)
        old.method.observe(mapper, packet.frame)
        runtime.observe(0, world.step_count, None, None, None, sensor_packet=packet)
        sensor = raw_sensor_hashes(packet, native.array_sha)
        assert packet.sha256() == row["packet_sha256"] and all(sensor[key] == row[key] for key in RAW_FIELDS)
        assert [*packet.position, packet.heading] == [*row["position"], row["heading"]]
        assert owner["owner_sha256"] == row["owner_sha256"]
    assert checked and world.step_count == stop
    return dict(witness=witness, prior=prior, archived_actions_replayed=stop,
                final_actual_action=archived_steps[stop-1]["action"],
                acquired_action_pose_and_device_records_exact=True,
                archived_manifest_sha256=native.sha(archive / "manifest.json"), **checked)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--witness", choices=WITNESSES)
    parser.add_argument("--prior", choices=("G", "S"))
    parser.add_argument("--output", type=Path, default=HERE/"local_parity")
    args = parser.parse_args()
    work = args.output
    if args.witness is not None:
        assert args.prior
        write(work / f'{args.witness.split(":")[0]}_{args.prior}.json', one(args.witness, args.prior))
        return
    work.mkdir(exist_ok=False)
    children = []
    for witness in WITNESSES:
        for prior in ("G", "S"):
            with (work/f'{witness.split(":")[0]}_{prior}.log').open("x") as log:
                process = subprocess.Popen([sys.executable, str(Path(__file__)), "--witness", witness,
                                            "--prior", prior, "--output", str(work)], stdout=log,
                                           stderr=subprocess.STDOUT)
                children.append(process)
    result_codes = [proc.wait() for proc in children]
    if any(result_codes):
        raise RuntimeError(f"Archived parity workers failed: {result_codes}")
    paths = sorted(work.glob("*.json"))
    write(work / "receipt.json", dict(status="passed", legacy_layout_witnesses=[read(p) for p in paths],
          weighted_family_order_check=synthetic_check(), holdout_policy_actions=0,
          sources_sha256={str(p): native.sha(p) for p in [Path(__file__), HERE/"weighted_score.py",
                                                        HERE/"weighted_run.py", SOURCE_PROTOCOL, *paths]}))


if __name__ == "__main__":
    main()
