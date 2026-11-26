"""One separately sealed west-screened G simulator integration check."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
import sys
import time
import traceback


HERE = Path(__file__).resolve().parent
SHADOW = HERE.parent / "west_projection_shadow"
TECHNICAL = HERE.parents[2] / "region_global_revision_20261122/next/local_technical"
VERSION = "west_directional_endpoint_G_sim_v1"
sys.path[:0] = [str(SHADOW), str(TECHNICAL)]
from inputs import FREEZE as SHADOW_FREEZE, INITIAL, LOCAL, ORIGINAL, RGB_ONLY, read, verify
from projection import bind


def sha(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def seal():
    previous = verify()
    files = [Path(__file__), SHADOW_FREEZE, LOCAL / "FROZEN_INPUTS.json", LOCAL/"PROTOCOL.json",
             LOCAL/"calibration.json", LOCAL/"run.py", INITIAL, ORIGINAL/"manifest.json", ORIGINAL/"steps.json"]
    bindings = {}
    for mode, original in (("archived", ORIGINAL), ("rgb_only", RGB_ONLY)):
        path = SHADOW / "results" / f"{mode}.json"
        receipt = read(path)
        if (not receipt["shadow_accepted"] or not all(receipt["pass_checks"].values())
                or receipt["frozen_candidate_sha256"] != sha(SHADOW_FREEZE)
                or receipt["original_manifest_sha256"] != sha(original/"manifest.json")
                or receipt["fixed_frames"] != {"archived":62, "rgb_only":52}[mode]):
            raise ValueError(f"Failed independent fixed-stream preflight for {mode}")
        bindings[mode] = receipt["candidate_binding"]
        files.append(path)
    if bindings["archived"] != bindings["rgb_only"]:
        raise ValueError("The two fixed tests used different endpoint adapters")
    return {"version":VERSION, "layout":"west_screened", "arm":"G", "native_budget_including_return":192,
            "candidate_binding":bindings["archived"], "sealed_files_sha256":{str(p):sha(p) for p in files},
            "previous_fixed_shadow_input_count":len(previous["files"]),
            "scope":"one isolated cold simulation trajectory, original G scoring, guard and action budget"}


def execute_trial(out, frozen, frozen_path):
    from precision_patch import camera_cell as prior_rgb_only
    from west_replay import LAYOUT, native
    from run import execute

    sys.path.insert(0, str(native.B11))
    import runtime_binding

    original_install = runtime_binding.install
    installed = []

    def install():
        context = original_install()
        candidate = bind(prior_rgb_only)
        if candidate != frozen["candidate_binding"] or installed:
            raise ValueError("Unexpected endpoint binding or repeated native installation")
        installed.append(candidate)
        from nso.mapping3d import SensorMapper

        original_mapper = SensorMapper.__init__

        def initialize_mapper(mapper, *args, **kwargs):
            original_mapper(mapper, *args, **kwargs)
            mapper._endpoint_shadow_mode = "directional"

        SensorMapper.__init__ = initialize_mapper
        return context

    run = out / "native"
    run.mkdir()
    runtime_binding.install = install
    print(f"{VERSION} started with frozen sources {sha(frozen_path)}", flush=True)
    started = time.monotonic()
    try:
        execute(LAYOUT, "G", run)
    finally:
        runtime_binding.install = original_install
    if len(installed) != 1:
        raise ValueError("Cold actor never installed the sealed candidate")
    new_manifest = read(run / "manifest.json")
    for name in ("steps.json", "observed_belief_trace.npz", "local_choices.json", "runtime_summary.json"):
        if sha(run/name) != new_manifest["files_sha256"][name]:
            raise ValueError(f"Closed-loop artifact differed from native manifest: {name}")
    old_steps, new_steps = (read(path/"steps.json") for path in (ORIGINAL, run))
    a, b = (read(path/"initial_raw_rgbd_scan_sha256.json") for path in (ORIGINAL, run))
    if a != b or len(new_steps) != new_manifest["paid_actions"] or new_manifest["native_budget"] != 192:
        raise ValueError("Cold start packet, paid action clock or return budget differed from the protocol")
    if new_manifest["experimental_local_arm"] != "G" or new_manifest["continuation_policy"] != "G_perf":
        raise ValueError("Native run used an unexpected arm or continuation")
    poses = ("action", "position", "heading")
    difference = next((x["step"] for x,y in zip(old_steps,new_steps) if any(x[k]!=y[k] for k in poses)),None)
    terminal = read(run / "runtime_summary.json")["termination"]
    source = read(ORIGINAL / "manifest.json")
    if (terminal["returned_to_anchor"] != new_manifest["returned"]
            or terminal["returned_to_anchor"] != (new_manifest["final_pose"]==new_manifest["initial_pose"])
            or terminal["final_action_id"] != len(new_steps)
            or new_manifest["collisions"] != sum(bool(step["collision"]) for step in new_steps)):
        raise ValueError("Termination outcome does not match the actual paid motion")
    outcome_keys = ("paid_actions", "returned", "collisions", "path_length_m", "route_event", "final_pose")
    result = {"status":"completed_native_evidence", "version":VERSION, "source_freeze_sha256":sha(frozen_path),
              "new_manifest_sha256":sha(run/"manifest.json"), "old_manifest_sha256":sha(ORIGINAL/"manifest.json"),
              "same_initial_raw_observation":True, "first_actual_action_or_pose_difference_paid":difference,
              "old":{key:source[key] for key in outcome_keys} | {"terminal":source["observed_native_termination"]},
              "new":{key:new_manifest[key] for key in outcome_keys} | {"terminal":terminal},
              "elapsed_s":time.monotonic()-started, "simulation_only":True, "subsequent_trials_attempted":0}
    write(out/"receipt.json",result)
    print(json.dumps(result["new"],indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(HERE):
        raise ValueError("Use an isolated output path within this candidate version")
    frozen = seal()
    out.mkdir(parents=True)
    frozen_path = out / "SOURCE_FREEZE.json"
    write(frozen_path,frozen)
    frozen_path.chmod(0o444)
    try:
        execute_trial(out, frozen, frozen_path)
    except Exception as error:
        write(out/"failure.json",{"status":"technical_failure","version":VERSION,"error":repr(error),
                                  "traceback":traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
