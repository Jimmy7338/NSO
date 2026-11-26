"""Verify repaired geometry by replaying only the recorded old action prefixes."""

import hashlib
from pathlib import Path
import sys

import numpy as np

from study import PREVIOUS, ROOT, SCENES, supervisor


sys.path.insert(0, str(SCENES))
from scene import make_world


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def replay(path, layout, before, target):
    io = supervisor()
    world = make_world(layout)
    steps = io.read(path)
    for expected in steps[:-1]:
        frame, collision, finished = world.step(expected["action"])
        if finished or collision or [*world.position, world.heading] != [
            *expected["position"], expected["heading"],
        ]:
            raise ValueError(f"Recorded motion changed before the failure: {path} step {expected['step']}")
        for key, observed in (("depth_values_sha256", frame.depth_m),
                              ("rgb_values_sha256", frame.color_rgb),
                              ("scan_ranges_sha256", world.scan().ranges_m)):
            if array_sha(observed) != expected[key]:
                raise ValueError(f"Changed observed sensor data: {path} step {expected['step']} {key}")
    last = steps[-1]
    assert last["collision"] and last["action"] == "forward"
    assert [*world.position, world.heading] == [*before, last["heading"]]
    _, collision, _ = world.step(last["action"])
    assert world.step_count == last["step"]
    if collision:
        assert list(world.position) == before
    else:
        assert list(world.position) == target
    return {"layout": layout, "steps_sha256": io.sha(path), "failure_clock": world.step_count,
            "identical_motion_and_sensor_packets_before_failure": len(steps)-1,
            "old_failed_transition": [before, target], "repaired_transition_allowed": not collision}


def main():
    io = supervisor()
    paths = ROOT / "validation"
    controls = paths / "repaired_controls.json"
    controls_result = io.read(controls)
    adapter = ROOT / "repair_geometry.py"
    assert controls_result["all_required_pass"] and controls_result["adapter_sha256"] == io.sha(adapter)
    audit_path = PREVIOUS / "FORMAL_COLLISION_AUDIT.json"
    audit = io.read(audit_path)
    receipts = [paths / "geometry" / f"{name}_cache.json" for name in
                ("hm3d_00770", "replica_apartment1", "hm3d_00337")]
    verified = {}
    for receipt in receipts:
        item = io.read(receipt)
        for case in item["case_checks"]:
            verified[item["floor"], tuple(case["cell"]), case["heading"]] = case["verified"]
    assert len(verified) == audit["distinct_source_swept_edges"] == 6 and all(verified.values())
    checks = []
    for case in audit["failures"]:
        assert case["disposition"] == "floor_support_only"
        for old in case["runs"]:
            trace = Path(old["steps_path"])
            assert io.sha(trace) == old["steps_sha256"]
            rec = replay(trace, old["layout"], case["cell"], case["target"])
            assert rec["repaired_transition_allowed"] and rec["failure_clock"] == old["paid_collision"]
            checks.append(rec | {"arm": old["policy"]})
    assert len(checks) == audit["total_collision_outcomes"] == 11
    development = replay(PREVIOUS / "development_smoke/run/steps.json",
                         "b26_hm3d_00337_s1", [31, 47], [32, 47])
    assert not (ROOT / "execution/formal").exists()
    eligible = [adapter, Path(__file__), ROOT / "prepare_study.py", controls, audit_path, *receipts,
                paths / "validate_support_controls.py", paths / "HM3D_NAV_REFERENCE.json",
                ROOT / "boundary_inventory.json", ROOT / "replica_diagnosis/replica_support_profiles.json",
                ROOT / "hm3d_diagnosis/raw_elevation.json"]
    result = {"status": "verified", "formal_planner_invocations_before_freeze": 0,
              "evidence_sha256": io.fingerprints(eligible), "old_failed_sweeps_bidirectionally_restored": 6,
              "old_failed_policy_prefixes": checks, "untrained_development_prefix": development,
              "scope": "Measured scan-level terrain under an unchanged collision footprint, with no policy invocations"}
    io.write(ROOT / "VALIDATION.json", result)
    print(f"Six repaired swept edges, eleven identical pre-failure action/sensor prefixes, "
          f"development transition allowed={development['repaired_transition_allowed']}", flush=True)


if __name__ == "__main__":
    main()
