"""Audit the online variants' actual recorded HM3D exit approach, without running a policy."""

from collections import Counter
import json
import sys

import numpy as np
from scipy.spatial import cKDTree

import inspect_corridor as source
from evidence import AccumulatedEvidence, _swept_xy

sys.path.insert(0,str(source.OUT.parent))
from observed_sweep import DIRECTIONS, SweepEvidence

STAGE = source.OUT.parent
CASE = "b26_hm3d_00770_s1"
FORMAL = STAGE / f"execution/formal/{CASE}"
ARM = "combined_S"
REPLAY = FORMAL / f"{ARM}/attempt_001/run"
STEPS = tuple(range(24,33)) + (48,96,144,192)
EXIT=((52,26),(53,26))
NEXT=((53,26),(54,26))
WITNESS=np.array([5.49,9.91])


def fingerprint_steps(run):
    rows=json.loads((run/"steps.json").read_text())
    return [(s["step"],s["action"],s["position"],s["heading"],s["scan_ranges_sha256"],s["depth_values_sha256"]) for s in rows]


def only_pose_recompute(world, audit, capture_ticks):
    source.ROOT=REPLAY
    prefix = list(source.read_poses(world))
    poses = {paid:(c,h,scan,frame) for paid,c,h,scan,frame in prefix}
    p,q=(np.asarray(world.xy(c)) for c in EXIT)
    probes=_swept_xy(p,q,.20)
    status = Counter()
    witness_source = None
    evidence = AccumulatedEvidence()
    builder = SweepEvidence(audit.shape[1:])
    builder.raw=evidence
    reports = {}
    best_so_far = None
    other = [(p2,(p2[0]+dr,p2[1]+dc)) for p2 in (EXIT[0],EXIT[1],NEXT[1])
             for dr,dc in DIRECTIONS]
    for paid in range(len(audit)):
        if paid in poses:
            c,h,scan,frame=poses[paid]
            builder.observe(frame,scan)
            points,_=frame.points(stride=1)
            ground=points[np.abs(points[:,2]) <= .04]
            if len(ground):
                nearest=float(np.linalg.norm(ground[:,:2]-WITNESS,axis=1).min())
                if best_so_far is None or nearest<best_so_far:
                    best_so_far=nearest
                    witness_source={"paid":paid,"cell":c,"heading":h,"distance_m":nearest}
        proof=evidence.certify(p,q,.20)
        if paid>=26:
            status[proof["reason"]]+=1
        if paid not in capture_ticks:
            continue
        tree=cKDTree(np.vstack(evidence._floor_returns))
        distance, index=tree.query(probes)
        missing=[{"xy_m":probes[i].tolist(),"nearest_floor_m":float(distance[i]),"nearest_raw_floor_xy_m":tree.data[index[i]].tolist()}
                 for i in np.flatnonzero(distance>.025)]
        transition_receipts=[]
        for start,end in other:
            short=evidence.certify(world.xy(start),world.xy(end),.20)
            transition_receipts.append({"start":start,"end":end,"map_states":[int(audit[paid,start[0],start[1]]),int(audit[paid,end[0],end[1]])],
                                   "allowed_by_raw_certificate":short["allowed"],"raw_certificate_reason":short["reason"],
                                   "unsatisfied_probe_count":short.get("unsupported_xy_probes")})
        reports[str(paid)]={"recorded_pose":motion[paid-1][2:4],"next_recorded_move":motion[paid][1:4] if paid<len(motion) else None,
                            "exit_edge_belief":[int(audit[paid,rr,cc]) for rr,cc in EXIT],
                            "exit_edge_sensor_cert":proof,"missing_ground_probes":missing,"closest_witness_ground":witness_source,
                            "adjacent_evidence":transition_receipts}
        if paid in (26,len(audit)-1):
            graph=builder.build(audit[paid])
            labels,_=graph.labels
            current=tuple(motion[paid-1][2])
            desired=(EXIT[0],EXIT[1],NEXT[1],(44,25),(61,28))
            reports[str(paid)]["regional_online_graph_at_recorded_time"]={
                "current_component_size":int((labels==labels[current]).sum()),
                "cells":[{"cell":cell,"connected_to_current":bool(labels[current] and labels[cell]==labels[current])} for cell in desired]}
    return reports,status,len(prefix)


motion = fingerprint_steps(REPLAY)


def main():
    world = source.make_world(CASE)
    region=FORMAL/"region_S/attempt_001/run"
    assert motion==fingerprint_steps(region)
    with np.load(REPLAY/"observed_belief_trace.npz",allow_pickle=False) as trace, np.load(
        region/"observed_belief_trace.npz",allow_pickle=False,
    ) as twin:
        belief=trace["belief"]
        assert np.array_equal(belief,twin["belief"])
    records, counts, unique = only_pose_recompute(world,belief,STEPS)
    regional=json.loads((REPLAY/"regional_choices.json").read_text())
    hypothetical_view=(44,25,2)
    result={"scope":"Actual formal trajectory mounted sensor pose/hash replay only. No novel pose rendered and no policy invoked.",
            "arm_trajectory_and_belief_identical":["combined_S","region_S"],"audited_unique_poses":unique,
            "transition":EXIT,"certificate_reasons_at_all_ticks_from_arrival_to_terminal":counts,
            "previous_counterfactual_supporting_view_was_actually_acquired":any((tuple(s[2]) + (s[3],))==hypothetical_view for s in motion),
            "previous_counterfactual_supporting_view_was_in_regional_candidate_log":any(
                tuple(c["pose"])==hypothetical_view for d in regional for c in d["certification"]),
            "at_checkpoint":records}
    (source.OUT/"online_terminal_exit.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
