"""Recover gallery meshes from sealed recorded actions without invoking a planner."""

import argparse
import os
import sys
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

from evaluate import ARCHIVE_HISTORY, PRIOR, SHORT_ROOT, STUDY, prior, verify_exact_prefix

sys.path.insert(0, str(PRIOR / "runner"))
from execute import array_sha, capture, chain_hash, np, source_module, source_records

ARCHIVAL_LAYOUTS = ("gallery_left", "gallery_right")


def verify_checkpoint_files(output, current, source, fields, label):
    if any(current[key] != source[key] for key in fields):
        raise ValueError(f"Replayed checkpoint differs from short archive at {label}: {output}")


def recover(layout, output):
    protocol_path = STUDY / "PROTOCOL.json"
    frozen_runner = prior.read_json(STUDY / "runner/freeze.json")
    protocol_sha = prior.verify_bound(protocol_path, frozen_runner["files_sha256"])
    protocol = prior.read_json(protocol_path)
    if layout not in ARCHIVAL_LAYOUTS or protocol["archived_full_routes"][layout] != [1]:
        raise ValueError(f"No frozen gallery archive recovery authorized for {layout}")
    source_protocol = prior.read_json(PRIOR / "PROTOCOL.json")
    mechanism = source_module(source_protocol)
    init = mechanism.verify(layout)
    _, steps, gate_receipt, record_hashes = source_records(mechanism, layout)
    if gate_receipt["chosen_candidate_id"] != 1:
        raise ValueError("Gallery comparator differs from archived selection")
    archive_path = ARCHIVE_HISTORY / "layouts" / layout / "runs/G/attempt_001"
    archive_manifest = prior.read_json(archive_path / "artifact_manifest.json")
    for filename in ("result.json", "final_tsdf_mesh.ply", "final_belief.npz"):
        prior.verify(archive_path / filename, archive_manifest["files_sha256"][filename])
    archive_result = prior.read_json(archive_path / "result.json")
    if (archive_result["paid_actions"] != len(steps) or len(steps) > protocol["native_total_paid_cap"]
            or archive_result["effective_args"]["cpu_score_mode"] != "G"):
        raise ValueError("Unexpected archived gallery action budget")
    short_run = SHORT_ROOT / layout / "candidate_1"
    short_manifest = prior.read_json(short_run / "manifest.json")
    prior.verify(short_run / "steps.json", short_manifest["files_sha256"]["steps.json"])
    short_steps = prior.read_json(short_run / "steps.json")
    verify_exact_prefix(steps, short_steps, layout)
    sources = {}
    for name in ("start", "arrival", "paid_58"):
        cp_path = short_run / name / "checkpoint.json"
        prior.verify(cp_path, short_manifest["files_sha256"][f"{name}/checkpoint.json"])
        cp = prior.read_json(cp_path)
        for filename, digest in cp["files_sha256"].items():
            prior.verify(short_run / name / filename, digest)
        sources[cp["actual_paid"]] = cp
    old, _ = mechanism.install()
    from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
    from runtime_binding import write

    world, transform, packet = old.world_and_packet(layout)
    if packet.sha256() != init["initial_packet_sha256"]:
        raise ValueError("Recovered initial packet is not the original frozen RGB-D and scan")
    # Use the identical observation-fusion line from the archived runtime. Neither an
    # action selector nor its scoring backend is instantiated for this recovery.
    mapper = ObservedRuntimeMapperV10(world.shape, world.config)
    mapper.update(packet.frame, packet.scan)
    mapper.instance_support_index = old.front.isp.InstanceSupportIndex()
    mapper.asset_grouping_profile = "measured_instance_support_v1"
    owner = old.front.associate(mapper, packet.frame, packet.frame_id)
    if owner["owner_sha256"] != init["initial_owner_sha256"]:
        raise ValueError("Recovered initial instance ownership disagrees with archived sensor input")
    recovery_pays = {
        *sources, *protocol["optional_diagnostic_checkpoints"],
        *(v for v in protocol["required_checkpoints"]["gallery"] if isinstance(v, int)),
        len(steps),
    }
    output.mkdir(parents=True, exist_ok=False)
    chain = chain_hash("0" * 64, packet.sha256())
    snapshots = {}
    beliefs = [mapper.belief.copy()]
    matches = {}

    def capture_at_paid(paid):
        label = "start" if paid == 0 else f"paid_{paid}"
        current = capture(output, label, mapper, packet, chain, paid, "recovered_archived_observations")
        snapshots[label] = current
        if paid in sources:
            fields = ("actual_paid", "pose", "packet_sha256", "packet_chain_sha256",
                      "belief_sha256", "native_mesh_fields", "files_sha256")
            verify_checkpoint_files(output, current, sources[paid], fields, label)
            matches[label] = {key: current[key] for key in fields}

    capture_at_paid(0)
    for step in steps:
        paid = step["step"]
        raw, collision, done = world.step(step["action"])
        packet = old.packet(world, transform, raw, layout, step["action"], collision, done)
        p_sha = packet.sha256()
        chain = chain_hash(chain, p_sha)
        owner = old.front.associate(mapper, raw, packet.frame_id)
        if (paid != packet.action_id or [*packet.position, packet.heading] != [*step["position"], step["heading"]]
                or bool(collision) != step["collision"] or bool(done) != step["done"]
                or p_sha != step["packet_sha256"] or owner["owner_sha256"] != step["owner_sha256"]
                or old.base.nonsemantic_packet_sha(packet) != step["nonsemantic_packet_sha256"]):
            raise ValueError(f"Archived observations diverged at paid action {paid}")
        mapper.update(packet.frame, packet.scan)
        beliefs.append(mapper.belief.copy())
        if paid in recovery_pays:
            capture_at_paid(paid)

    terminal_checkpoint = snapshots[f"paid_{len(steps)}"]
    if terminal_checkpoint["files_sha256"]["native_mesh.ply"] != archive_manifest["files_sha256"]["final_tsdf_mesh.ply"]:
        raise ValueError("All observations matched but archived terminal native TSDF mesh differs")
    with np.load(archive_path / "final_belief.npz", allow_pickle=False) as belief:
        if (array_sha(belief["belief"]) != terminal_checkpoint["belief_sha256"]
                or [*map(int, belief["position"]), int(belief["heading"])] != terminal_checkpoint["pose"]
                or not np.array_equal(mapper.camera_seen, belief["camera_seen"])):
            raise ValueError("Recovered terminal map and archived terminal map differ")
    np.savez_compressed(output / "observed_belief_trace.npz", paid=np.arange(len(beliefs)), belief=np.asarray(beliefs))
    receipt = {
        "status": "completed_archived_observation_recovery", "layout": layout, "candidate_id": 1,
        "source_protocol_sha256": protocol_sha, "recovery_sha256": prior.sha256(__file__),
        "archived_actions_replayed": len(steps), "new_policy_actions": 0, "planning_calls": 0, "scoring_calls": 0,
        "source_manifest_sha256": prior.sha256(archive_path / "artifact_manifest.json"),
        "short_manifest_sha256": prior.sha256(short_run / "manifest.json"),
        "source_record_sha256": record_hashes, "matched_old_short_checkpoints": matches,
        "terminal_mesh_sha256": terminal_checkpoint["files_sha256"]["native_mesh.ply"],
        "terminal_belief_sha256": terminal_checkpoint["belief_sha256"],
        "terminal_packet_chain_sha256": chain, "captured": list(snapshots),
        "unavailable_paid_checkpoints": sorted(recovery_pays - set(
            s["actual_paid"] for s in snapshots.values()
        )), "files_sha256": {
            str(path.relative_to(output)): prior.sha256(path) for path in sorted(output.rglob("*")) if path.is_file()
        },
    }
    write(output / "manifest.json", receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("layout", choices=ARCHIVAL_LAYOUTS)
    args = parser.parse_args()
    root = STUDY / "measurement/archive_recovery" / args.layout / "candidate_1"
    recover(args.layout, root)
