#!/usr/bin/env python3
"""Exercise only saved paid prefixes; no World, TSDF or quality evaluation."""
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_external_routes_v39 import ExperimentV39,prior


def main():
    from nso.cpu_four_modules_v35 import ObservationV35
    from nso.decision_replay_v13 import load_packet
    from env.information_pixel_v34 import sensor_counts_v34
    from nso.research_evidence_v31 import freeze,write,seal
    prior.cpu_contract()
    root=ROOT/'audit_results/v39_saved_prefix_preflight_r1_20260920'
    root.mkdir()
    rows=[]; inputs={}; start=time.monotonic()
    config=prior.read(prior.OUTPUT/'config.json'); harness=ExperimentV39()
    for old_index in (0,2,4,6):
        old_case=config['physical_cases'][old_index]; acq=config['parents'][old_case['parent']]
        for method in ('SWAP','VISTA'):
            controller,models=harness.create_controller(dict(parent=old_case['parent'],method=method),acq)
            for paid in range(19):
                path=prior.OUTPUT/f'case{old_index:02d}'/'packets'/f'{paid:03d}.npz'
                inputs[str(path.relative_to(ROOT))]=prior.sha(path)
                packet=load_packet(path); pose=prior.observed_pose(packet,acq['translation'])
                obs=ObservationV35(frame_id=f'opaque-{paid}',step=paid,pose=pose,action=packet.action,
                    depth=packet.frame.depth_m,rgb=packet.frame.color_rgb,ranges=packet.scan.ranges_m,
                    collision=packet.collision)
                controller.accept(obs); receipt=harness.after_observation(controller,obs,packet)
                action=controller.next_action()
                if paid<18 and action!=acq['common_prefix_actions'][paid]:
                    raise AssertionError('paid prefix changed')
            plan=controller.plans[-1]['planning']
            if plan['posterior_used_to_choose_action'] is not False:
                raise AssertionError('inspired policy mislabels posterior use')
            if not plan['candidates']: raise AssertionError('no feasible external candidates')
            rows.append(dict(parent=old_case['parent'],saved_hypothesis=old_case['hypothesis'],
                method=method,action=action,ledger=receipt,planning=plan,
                registered_class=controller.planner.registered_class))
    if any(sensor_counts_v34().values()): raise AssertionError('preflight constructed sensor')
    freeze(root,[Path(__file__),ROOT/'nso/external_planners_v39.py',ROOT/'nso/observed_geometry_v39.py'],
        input_sha256=inputs,worlds=0,sensor_queries=0,mapper_constructions=0,quality_evaluations=0)
    write(root,root/'result.json',dict(status='passed',saved_prefixes=8,paid_packet_reads=152,
        worlds=0,mapper_constructions=0,quality_evaluations=0,rows=rows,seconds=time.monotonic()-start))
    seal(root)
    print(json.dumps(dict(status='passed',rows=[{k:r[k] for k in ('parent','saved_hypothesis','method','action','registered_class')}
        for r in rows],worlds=0)),flush=True)


if __name__=='__main__': main()
