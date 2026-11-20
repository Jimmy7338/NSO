#!/usr/bin/env python3
"""Acquire every original candidate route once per frozen physical layout.

Fixed-route branch experiment, not a closed-loop autonomous exploration trial.
No policy score or hidden reference chooses an executed branch.
"""
import argparse,copy,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import open3d as o3d
import scene
HERE=Path(__file__).resolve().parent
CANDIDATES=HERE/'candidate_attempt_002'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def resource(p):return dict(path=str(p),sha256=sha(p))
def mesh(mapper,path):
    native=mapper.mesh();assert o3d.io.write_triangle_mesh(str(path),native)
    return dict(**resource(path),triangles=len(native.triangles),vertices=len(native.vertices))
def coverage(belief,reachable):
    n=int(reachable.sum());known=(belief!=-1)&reachable;free=(belief==0)&reachable
    return dict(reachable_cells=n,known_cells=int(known.sum()),correct_free_cells=int(free.sum()),
        coverage_fraction=float(known.sum()/n),correct_free_coverage_fraction=float(free.sum()/n))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--layout',choices=tuple(scene.SPEC['layouts']),required=True);a=parser.parse_args()
    scene.verify();assert json.loads((CANDIDATES/'result.json').read_text())['status']=='completed_native_candidates_both_temperatures_all_six_cues'
    out=HERE/'branches_attempt_001'/a.layout;out.mkdir(parents=True,exist_ok=False);start=time.monotonic();done=[]
    from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
    inputs=[Path(__file__),HERE/'evaluate_branches.py',HERE/'branch_protocol.json',HERE/'scene_freeze.json',CANDIDATES/'result.json',CANDIDATES/a.layout/'fixed_candidates.json']
    hashes={str(p):sha(p)for p in inputs};write(out/'freeze.json',dict(status='frozen_before_branch_acquisition',files_sha256=hashes,layout=a.layout,policy_or_hardware=False))
    try:
        original=scene.MechanismWorld(a.layout);frame=original.sense();scan=original.scan()
        initial=ObservedRuntimeMapperV10(original.shape,original.config);initial.update(frame,scan)
        before=mesh(initial,out/'prefix_native_mesh.ply');initial_coverage=coverage(initial.belief,original.reachable)
        routes=json.loads((CANDIDATES/a.layout/'fixed_candidates.json').read_text());stages=[]
        for route in routes:
            cid=int(route['candidate_id']);folder=out/f'candidate_{cid:03d}';folder.mkdir()
            world=copy.copy(original);mapper=ObservedRuntimeMapperV10(world.shape,world.config);mapper.update(frame,scan)
            clone=mesh(mapper,folder/'prefix_check.ply');assert clone['sha256']==before['sha256'];(folder/'prefix_check.ply').unlink()
            assert list(route['outbound_states'][0])==[*world.position,world.heading]
            steps=[];reason='completed_fixed_outbound'
            for action,expected in zip(route['outbound_actions'],route['outbound_states'][1:]):
                newframe,collision,done_flag=world.step(action);newscan=world.scan();mapper.update(newframe,newscan)
                position=[*world.position,world.heading]
                steps.append(dict(action=action,pose=position,expected_pose=list(expected),collision=bool(collision),
                    depth_sha256=hashlib.sha256(newframe.depth_m.tobytes()).hexdigest(),pose_matches_expected=position==list(expected)))
                if collision or position!=list(expected):reason='fixed_outbound_failure_retained';break
            after=mesh(mapper,folder/'native_mesh.ply')
            execution=dict(status=reason,candidate_id=cid,planned_actions=len(route['outbound_actions']),completed_actions=len(steps),
                endpoint_reached=[*world.position,world.heading]==list(route['outbound_states'][-1]),endpoint_pose=[*world.position,world.heading],
                collisions=sum(x['collision']for x in steps),path_length_m=world.moves*world.config.resolution_m,
                initial_coverage=initial_coverage,final_coverage=coverage(mapper.belief,world.reachable),steps=steps,
                prefix_mesh_byte_identical=True,source_geometry_supplied_to_mapper=False,predicted_geometry_fused=False,
                complete_closed_loop_trial=False,physical_layout=a.layout)
            write(folder/'execution.json',execution);stages.append(dict(candidate_id=cid,before_mesh=before,after_mesh=after,execution=resource(folder/'execution.json')))
            done.append(cid);write(out/'manifest.json',dict(layout=a.layout,stages=stages,all_candidates_complete=len(done)==len(routes),
                conditions_reuse_same_branch_outcomes=True,independent_policy_replicates=False))
            print(json.dumps(dict(layout=a.layout,candidate=cid,actions=len(steps),endpoint_reached=execution['endpoint_reached'])),flush=True)
        assert original.step_count==0 and original.moves==0
        assert hashes=={p:sha(p)for p in hashes};scene.verify()
        write(out/'result.json',dict(status='completed_all_native_candidate_branches',layout=a.layout,candidates=len(done),
            actual_branch_actions=sum(len(json.loads((out/f'candidate_{c:03d}'/'execution.json').read_text())['steps'])for c in done),
            elapsed_s=time.monotonic()-start,hardware=False,full_exploration=False))
    except BaseException as error:
        write(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),completed_candidates=done,elapsed_s=time.monotonic()-start));raise
if __name__=='__main__':main()
