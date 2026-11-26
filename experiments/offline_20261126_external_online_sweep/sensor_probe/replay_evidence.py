"""Check the reusable certificate against fixed, hash-verified archived packets."""

import json

from inspect_corridor import CUTOFFS, EDGES, LAYOUT, OUT, make_world, read_poses
from evidence import AccumulatedEvidence


def main():
    world = make_world(LAYOUT)
    poses = {paid: (scan, frame) for paid, _, _, scan, frame in read_poses(world)}
    evidence = AccumulatedEvidence()
    report = {}
    for paid in range(max(CUTOFFS)+1):
        if paid in poses:
            scan, frame = poses[paid]
            evidence.observe(frame, scan)
        if paid in CUTOFFS:
            report[str(paid)] = [
                {"start": start, "end": end, "result": evidence.certify(world.xy(start), world.xy(end), .20),
                 "reverse_result": evidence.certify(world.xy(end), world.xy(start), .20)} for start, end in EDGES
            ]
    (OUT/"sensor_certificate_prefix_replay.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
