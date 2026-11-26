"""Explain the archived p38 graph boundary without changing the production certificate."""

import copy
import json
import sys

import numpy as np

from inspect_corridor import LAYOUT, OUT, ROOT, make_world, read_poses
sys.path.insert(0, str(OUT.parent))
from observed_sweep import CertifiedGraph, DIRECTIONS, SweepEvidence


PAID=38
EXIT=((52,26),(53,26))


def describe_labels(graph, current, targets):
    labs, n = graph.labels
    return {"nonzero_current_component_size":int((labs == labs[current]).sum()) if labs[current] else 0,
            "targets":[{"cell": target,"target_component":int(labs[target]),"can_reach_from_current":bool(labs[current] and labs[current] == labs[target])} for target in targets]}


def main():
    world=make_world(LAYOUT)
    summary=json.loads((OUT.parent/"replay/FIXED_REPLAY.json").read_text())
    replay=next(item for item in summary["checks"] if item["paid"]==PAID)
    view_pose=tuple(replay["pose"][:2])
    wanted=[tuple(item["pose"][:2]) for item in replay["adapted_observed_sweep"]["fixed_pool_native_admission"]]
    with np.load(ROOT/"observed_belief_trace.npz",allow_pickle=False) as f:
        belief=f["belief"][PAID]
    builder = SweepEvidence(belief.shape)
    for paid,c,h,scan,frame in read_poses(world):
        if paid > PAID:
            break
        builder.observe(frame,scan)
    graph=builder.build(belief)
    p,q=EXIT
    h=DIRECTIONS.index((q[0]-p[0],q[1]-p[1]))
    near_edges=[]
    for rr in range(46,60):
        for cc in range(23,32):
            a=rr,cc
            for h2,(dr,dc) in enumerate(DIRECTIONS[1:3],1):
                b=rr+dr,cc+dc
                if belief[a]!=0 or belief[b]!=0 or graph.edges[a+(h2,)]:
                    continue
                if graph.labels[0][a]==graph.labels[0][b]:
                    continue
                receipt=builder.raw.certify(world.xy(a),world.xy(b),.20)
                near_edges.append({"start":a,"end":b,"proof":receipt,"start_component":int(graph.labels[0][a]),"end_component":int(graph.labels[0][b])})
    hypothetical=CertifiedGraph(graph.safe.copy(),graph.edges.copy(),graph.added.copy())
    hypothetical.safe[p]=hypothetical.safe[q]=True
    hypothetical.edges[p+(h,)]=hypothetical.edges[q+((h+2)%4,)]=True
    result={"scope":"read-only fixed archived sensor probe; exactly one unsupported edge inserted only in hypothetical copy, never scored or executed",
            "paid":PAID,"actual":describe_labels(graph,view_pose,wanted),
            "hypothetically_forcing_unsupported_exit":describe_labels(hypothetical,view_pose,wanted),
            "exit_environment_map_states":[int(belief[p]),int(belief[q])],
            "other_known_edge_cut_certification_reasons":near_edges}
    (OUT/"p38_graph_cut.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
