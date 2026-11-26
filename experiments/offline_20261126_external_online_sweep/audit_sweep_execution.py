"""Match measured sweep approvals to the actual frozen simulator action records."""

from pathlib import Path

from study import ROOT, supervisor


def main():
    io = supervisor()
    phase = ROOT / "execution/formal/phase_receipt.json"
    receipt = io.read(phase)
    results = []
    assert len(receipt["episodes"]) == 12
    for episode in receipt["episodes"]:
        directory = Path(episode["run_path"])
        assert episode["exit_code"] == 0 and io.sha(episode["result_path"]) == episode["result_sha256"]
        steps = {r["step"]: r for r in io.read(directory / "steps.json")}
        audit = io.read(directory / "audit.json")
        allowed = {r["next_action_id"]: r for r in audit if r["event"] == "paid_action_authorized"}
        initial = io.read(directory / "attempt.json")
        proof_actions = {}
        examined_not_executed = 0
        for row in audit:
            if row["event"] != "observed_swept_edge_certified":
                continue
            paid = row["paid_before"]+1
            actual, approval = steps.get(paid), allowed.get(paid)
            assert row["certificate"]["allowed"] and row["certificate"]["reason"] == "measured_full_capsule"
            assert row["certificate"]["probes"] > 0
            if actual is None or approval is None or actual["action"] != "forward":
                examined_not_executed += 1
                continue
            prev = steps.get(paid-1)
            if prev and (row["source"] != prev["position"] or row["heading"] != prev["heading"]):
                raise ValueError("Sweep receipt differs from recorded source pose")
            if approval["assessment"]["next_pose"] != [*row["target"], row["heading"]]:
                examined_not_executed += 1
                continue
            assert approval["assessment"]["allowed"] and actual["guard_allowed"]
            if not actual["collision"] and actual["position"] != row["target"]:
                raise ValueError("Observed approval did not match the recorded physical action")
            proof_actions[paid] = {"paid": paid, "source": row["source"], "target": row["target"],
                                  "heading": row["heading"], "collision": actual["collision"],
                                  "receipt": row["certificate"]}
        results.append({"layout": episode["layout"], "arm": episode["arm"], "paid": len(steps),
                        "initial_packet_sha256": initial["initial_packet_sha256"],
                        "steps_sha256": io.sha(directory / "steps.json"),
                        "audit_sha256": io.sha(directory / "audit.json"),
                        "certified_executed_count": len(proof_actions),
                        "certified_executed_collisions": sum(row["collision"] for row in proof_actions.values()),
                        "certified_but_unexecuted_checks": examined_not_executed,
                        "certified_executed_steps": list(proof_actions.values()),
                        "all_native_collisions": sum(row["collision"] for row in steps.values())})
    io.write(ROOT / "SWEEP_EXECUTION_AUDIT.json",
             {"schema": "nso_online_sweep_executed_edges_v1", "phase_sha256": io.sha(phase),
              "runs": results, "executed_with_fine_certificate": sum(r["certified_executed_count"] for r in results),
              "certified_execution_collisions": sum(r["certified_executed_collisions"] for r in results)})
    print(sum(r["certified_executed_count"] for r in results), "actual actions used fine swept certificates", flush=True)


if __name__ == "__main__":
    main()
