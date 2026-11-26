"""Posthoc render-only check for an additional positive measurement of the exit floor."""

import copy
import json
from math import hypot
import sys

import numpy as np

from exit_witness import TARGET
from inspect_corridor import LAYOUT, OUT, ROOT, make_world, read_poses
sys.path.insert(0,str(OUT.parent))
from observed_sweep import SweepEvidence


PAID=38
POSE_AT_PAID=(45,25)
HOLE=np.array([5.49,9.91])
HEADING=2


def fork(evidence):
    new=copy.copy(evidence)
    for name in ("_floor_returns","_lower_returns","_depth_hits","_scan_hits","_observations"):
        setattr(new,name,getattr(evidence,name).copy())
    new._seen=evidence._seen.copy()
    new._certificates={}
    return new


def main():
    world=make_world(LAYOUT)
    with np.load(ROOT/"observed_belief_trace.npz",allow_pickle=False) as f:
        belief=f["belief"][PAID]
    builder=SweepEvidence(belief.shape)
    viewed={}
    for t,p,h,scan,frame in read_poses(world):
        if t>PAID:
            break
        builder.observe(frame,scan)
        viewed[p,h]=t
    graph=builder.build(belief)
    paths,_=graph.shortest_paths(graph.safe,POSE_AT_PAID)
    look=[]
    for r,c in np.argwhere(paths>=0):
        cell=int(r),int(c)
        x,y=world.xy(cell)
        forward=y-HOLE[1]
        lateral=abs(x-HOLE[0])
        if 1.63<=forward<=3.2 and lateral<=min(.6,.5*forward) and (cell,HEADING) not in viewed:
            look.append((int(paths[cell]),hypot(lateral,forward),cell))
    look=sorted(look)[:30]
    result = []
    for path_length,offset,cell in look:
        # A diagnostic new simulated observation from an already sensor-reachable cell.
        # An actual actor may choose a different position/heading. No movement is performed.
        world.position,world.heading,world.step_count=cell,HEADING,PAID
        raw,frame=world.scan(),world.sense()
        points,_=frame.points(stride=1)
        good=points[np.abs(points[:,2])<=.04]
        nearest=float(np.linalg.norm(good[:,:2]-HOLE,axis=1).min()) if len(good) else None
        if nearest is None or nearest>.025:
            continue
        candidate=fork(builder.raw)
        candidate.observe(frame,raw)
        proof=candidate.certify(*(world.xy(c) for c in TARGET),.20)
        result.append({"cell":cell,"heading":HEADING,"known_component_grid_moves_from_actual_p38":path_length,
                       "direct_distance_camera_center_to_missing_probe_m":offset,
                       "nearest_positive_ground_sample_to_missing_probe_m":nearest,
                       "full_certificate_after_counterfactual_frame":proof})
    (OUT/"counterfactual_registered_viewpoints.json").write_text(json.dumps(
        {"role":"posthoc synthetic sensor observation from an already online-connected center; not an achieved observation or rerun",
         "p38_start":POSE_AT_PAID,"missing_probe_xy_m":HOLE.tolist(),"candidate_count":len(look),
         "positive_floor_candidates":result},indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
