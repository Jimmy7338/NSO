"""Replay the old terminals and verify supported edges and unobserved exterior stops."""

import hashlib
from pathlib import Path
import sys

import numpy as np

from study import PREVIOUS, ROOT, SCENES, supervisor


sys.path.insert(0, str(SCENES))
from scene import make_world


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def replay(world, steps, target, expected):
    world.position, world.heading = world.start, world.spec["heading"]
    world.step_count = world.collisions = world.moves = 0
    io = supervisor()
    records = io.read(steps)
    for row in records[:-1]:
        frame, collision, finished = world.step(row["action"])
        observed = (frame.depth_m, frame.color_rgb, world.scan().ranges_m)
        fields = ("depth_values_sha256", "rgb_values_sha256", "scan_ranges_sha256")
        if (finished or collision or list(world.position) != row["position"] or world.heading != row["heading"]
                or any(array_sha(value) != row[key] for value, key in zip(observed, fields, strict=True))):
            raise ValueError(f"Prior step, pose or observation diverged: {steps}, step {row['step']}")
    terminal = records[-1]
    assert terminal["collision"] and terminal["action"] == "forward"
    assert list(world.position) == terminal["position"] and world.heading == terminal["heading"]
    _, collision, _ = world.step(terminal["action"])
    if collision == expected or list(world.position) != (target if expected else terminal["position"]):
        raise ValueError(f"Old terminal sweep disposition changed unexpectedly: {steps}")
    return {"steps_sha256": io.sha(steps), "identical_motion_and_sensor_packets_before_failure": len(records)-1,
            "prior_failure_paid": world.step_count, "repaired_transition_allowed": expected}


def main():
    io = supervisor()
    controls = ROOT / "validation/v2_controls.json"
    previous_controls = ROOT / "validation/v1_controls.json"
    controls_receipt = io.read(controls)
    adapter = ROOT / "repair_geometry.py"
    assert controls_receipt["adapter_sha256"] == io.sha(adapter)
    assert controls_receipt["all_required_pass"]
    prior_audit_path = PREVIOUS / "RERUN_COLLISION_AUDIT.json"
    prior_audit = io.read(prior_audit_path)
    receipts = sorted((ROOT / "validation/geometry").glob("*_cache.json"))
    statuses = {}
    for receipt in receipts:
        data = io.read(receipt)
        for edge in data["case_checks"]:
            assert edge["verified"]
            statuses[data["floor"], tuple(edge["cell"]), edge["heading"]] = edge["expected_allowed"]
    assert len(statuses) == prior_audit["distinct_edges"] == 7 and sum(statuses.values()) == 5
    prior = io.read(PREVIOUS / "execution/formal/phase_receipt.json")["episodes"]
    paths = {(r["layout"], r["arm"]): Path(r["result_path"]).with_name("steps.json") for r in prior}
    checks = []
    worlds = {}
    for case in prior_audit["failures"]:
        expected = statuses[case["source_floor"], tuple(case["cell"]), case["heading"]]
        for trial in case["runs"]:
            layout, arm = trial["layout"], trial["arm"]
            if layout not in worlds:
                worlds[layout] = make_world(layout)
            path = paths[layout, arm]
            assert trial["steps_sha256"] == io.sha(path)
            item = replay(worlds[layout], path, case["target"], expected)
            assert item["prior_failure_paid"] == trial["paid"]
            checks.append({"layout": layout, "arm": arm, **item})
    assert len(checks) == 9 and sum(x["repaired_transition_allowed"] for x in checks) == 7
    # The diagnosis is based on frozen previous decisions and must predate the rerun.
    diagnoses = [*sorted((ROOT / "plateau").glob("*.json")), *sorted((ROOT / "topology").glob("*.json"))]
    assert diagnoses and not (ROOT / "execution/formal").exists()
    eligible = [adapter, ROOT / "prepare_study.py", Path(__file__), controls, previous_controls,
                ROOT / "validation/validate_continuity_controls.py",
                PREVIOUS / "repair_geometry.py", prior_audit_path, *receipts, *diagnoses]
    result = {"status": "verified", "formal_planner_invocations_before_freeze": 0,
              "evidence_sha256": io.fingerprints(eligible), "prior_distinct_edges": 7,
              "restored_support_edges_bidirectionally": 5, "retained_unobserved_exterior_edges": 2,
              "passed_and_still_blocked_prior_prefixes": checks,
              "scope": "Measured terrain and small interior seam certification with no policy invocations"}
    io.write(ROOT / "VALIDATION.json", result)
    print("Five repaired edges and two still blocked; nine prior action and sensor prefixes reproduced.",
          flush=True)


if __name__ == "__main__":
    main()
