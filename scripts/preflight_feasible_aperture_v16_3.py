#!/usr/bin/env python3
"""Restore all recorded arrivals, check shared pools, freeze all new roles."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
from copy import deepcopy
import gzip,json
from pathlib import Path
import sys,zipfile
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from env.virtual3d_inspection_v4 import InspectionConfigV4
from nso.cpu_four_modules_v16_3 import viewpoint_components_v16_3
from nso.staged_approach_runtime_v16 import StagedApproachRuntimeV16
from nso.cpu_sensor_contract_v10 import GridTransform,digest,json_value
from nso.decision_replay_v13 import load_packet,decision_state
from nso.axis_history_view_v16 import AxisHistoryViewV16
from nso.hierarchical_options_v16_3 import generate_options
from nso.semantic_opportunities_v14 import observed_descriptors,route_instance_features
from scripts.preflight_axis_backend_v16 import LabelView
from scripts.collect_semantic_gain_v13_history import sha,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    m=json.loads((a.source/'manifest.json').read_text());protocol=m['protocol']
    if m['status']!='complete':raise ValueError('complete physical probes required')
    for n,v in json.loads((a.source/'artifact_hashes.json').read_text()).items():
        if sha(a.source/n)!=v:raise ValueError('input changed')
    for n,v in m['source_sha256'].items():
        if sha(ROOT/n)!=v:raise ValueError('frozen source changed')
    files=set(m['source_sha256'])|{str(Path(__file__).relative_to(ROOT)),
        'nso/cpu_four_modules_v16_3.py','nso/hierarchical_options_v16_3.py','configs/virtual3d/feasible_aperture_v16_3.json'}
    frozen={n:sha(ROOT/n) for n in sorted(files)}
    a.output.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='running',source_sha256=frozen,input_root=str(a.source),
        input_inventory_sha256=sha(a.source/'artifact_hashes.json'),new_physical_outcomes=0,training_allowed=False)
    write(a.output/'manifest.json',manifest)
    with zipfile.ZipFile(a.output/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for n in frozen:z.write(ROOT/n,n)
    try:
        hp=json.loads((ROOT/protocol['history']/'manifest.json').read_text())['protocol']
        doc=json.loads((ROOT/hp['scene_protocol']).read_text());c=next(x for x in doc['contexts'] if x['id']==hp['context'])
        config=InspectionConfigV4(**{**doc['shared_conditions'],**{k:v for k,v in c.items() if k not in ('id','seed')}})
        hf=ROOT/protocol['history']/f"structure_{protocol['structure_seed']}";bounds=json.loads((hf/'all_decisions.json').read_text())[0]['bounds']
        shape=(bounds[1],bounds[3]);rows=[];declarations=[]
        for first in protocol['candidates']:
            folder=a.source/f"candidate_{first['candidate_id']}"
            old=json.loads(gzip.decompress((folder/'deterministic.json.gz').read_bytes()))
            arrival=protocol['action_id']+old['outcome']['outbound_actions']
            args=SimpleNamespace(nso_backend='cpu_v10',eval=True,train_global=False,use_open_vocab_semantic=True,
                use_topo_graph=True,use_rpn_uq=True,use_igcr=True,cpu_score_mode='G',cpu_disable_feedback=False,
                cpu_max_candidates=5,cpu_coverage_slots=4,cpu_planner_revision='v10_3_1',
                cpu_semantic_source_schema='inspection_v4',cpu_measured_novelty_floor=.25,cpu_viewpoint_start_action=arrival)
            comp=viewpoint_components_v16_3(args,shape);runtime=StagedApproachRuntimeV16(comp,1,shape)
            initial=load_packet(hf/'packets/0000.npz')
            runtime.start_sensor_episode(0,config=config,transform=GridTransform(shape,config.resolution_m),packets=[initial],
                total_budget=protocol['total_task_budget'],return_anchor=(*initial.position,initial.heading))
            for aid in range(1,arrival+1):
                if aid==protocol['action_id']+1:runtime.install_candidate(first,protocol['candidate_pool_sha256'])
                packet=load_packet((hf if aid<=protocol['action_id'] else folder)/f'packets/{aid:04d}.npz')
                phase='external_prefix' if aid<=protocol['action_id'] else 'outbound'
                if runtime.authorize_probe_action(packet.action,phase=phase) is None:raise ValueError('recorded action denied')
                runtime.observe(0,aid,None,None,None,sensor_packet=packet)
            b=comp._cpu_backend;s=b.scenes[0];mapper=s['mapper'];state=decision_state(runtime)
            for key in ('map_arrays','mesh'):
                if state['evidence'][key]!=old['arrival_state']['evidence'][key]:raise ValueError('arrival map differs')
            if s['ledger'].snapshot()!=old['arrival_state']['evidence']['modules']['feedback']:
                raise ValueError('arrival feedback differs')
            b.select_target(0);selected=deepcopy(s['last_selection']);routes=selected['candidates']
            base=old['arrival_pool']['candidates']
            if routes[:len(base)]!=base:raise ValueError('base candidates changed')
            attempted=s['gain'].attempted_camera_mask(mapper,s['ledger'].planning_camera_poses())
            for zero in (False,True):
                pool,_=generate_options(LabelView(AxisHistoryViewV16(mapper,s['axis_frames']),zero),packet.position,packet.heading,
                    s['ledger'].remaining_budget,s['return_anchor'],max_candidates=5+5*len(s['assets']),
                    coverage_strategy='total_diverse',coverage_slots=4,attempted_camera_mask=attempted)
                if digest(pool)!=digest(routes):raise ValueError('labels changed shared pool')
            assets=s['assets']
            try:
                for intervention in ('swap','zero'):
                    s['assets']=deepcopy(assets)
                    for asset in s['assets']:
                        if intervention=='swap':asset['class_vote']=-asset['class_vote']
                        else:asset['marked_points']=0
                    scores,_=b._scores_v10_1(s,routes,attempted)
                    if any(scores[x]!=selected['scores'][x] for x in ('N','G','M')):raise ValueError('geometry score changed')
                    if scores['S']!=selected['scores']['X' if intervention=='swap' else 'G']:raise ValueError('semantic score contract')
            finally:s['assets']=assets
            assets,descriptors=observed_descriptors(runtime,routes)
            features=route_instance_features(mapper,routes,assets,descriptors)
            zero=route_instance_features(mapper,routes,assets,descriptors,confidence_scale=0.)
            for f,z in zip(features,zero):
                if len(f['semantic'])!=26 or len(f['geometry'])!=26 or f['geometry']!=z['geometry'] or z['semantic'][18:]!=[0.]*8:
                    raise ValueError('capacity or confidence contract')
            # Independent local footprint stencil and exact discrete actions.
            radius=config.robot_radius_m/config.resolution_m+2.**.5/2.;bound=int(np.ceil(radius))
            for route in routes:
                if route['cost']!=len(route['actions']) or route['cost']>s['ledger'].remaining_budget:raise ValueError('budget')
                if route['outbound_cost']+route['return_cost']!=route['cost']:raise ValueError('unreserved return')
                if route['states'][0]!=[*packet.position,packet.heading] or route['states'][-1]!=list(s['return_anchor']):raise ValueError('anchors')
                for (r,c,h),action,second in zip(route['states'],route['actions'],route['states'][1:]):
                    dr,dc=((-1,0),(0,1),(1,0),(0,-1))[h] if action=='forward' else (0,0)
                    if second!=[r+dr,c+dc,(h+(1 if action=='right' else -1 if action=='left' else 0))%4]:raise ValueError('primitive')
                for r,c,_ in route['states']:
                    for dr in range(-bound,bound+1):
                        for dc in range(-bound,bound+1):
                            if dr*dr+dc*dc>radius*radius:continue
                            if not (0<=r+dr<shape[0] and 0<=c+dc<shape[1] and mapper.belief[r+dr,c+dc]==0):raise ValueError('footprint')
            added=[r for r in routes if r['group'].endswith('_aperture_view')]
            for route in added:declarations.append(dict(first_candidate=first,arrival_action_id=arrival,
                arrival_pool_sha256=digest(routes),second_candidate=route))
            row=dict(first_candidate_id=first['candidate_id'],arrival_action_id=arrival,base_candidates=len(base),
                new_candidates=len(added),candidate_pool_sha256=digest(routes),selection=selected,features=features,
                arrival_state=state,map_mesh_feedback_exact=True,label_controls_passed=True,score_controls_passed=True,
                independent_route_checks_passed=True,executed_second_views=False)
            rows.append(row);write(a.output/f"after_{first['candidate_id']}.json",row)
        for n,v in frozen.items():
            if sha(ROOT/n)!=v:raise ValueError('source changed while running')
        write(a.output/'summary.json',dict(status='passed',arrival_states=len(rows),
            preserved_candidates=sum(x['base_candidates'] for x in rows),new_candidates=len(declarations),new_physical_outcomes=0,
            semantic_efficacy_proven=False,training_allowed=False))
        write(a.output/'second_view_declarations.json',dict(status='declared_not_executed',selection_rule='all new aperture views at all three arrivals',
            source=str(a.source),source_protocol=protocol,declarations=declarations))
        manifest['status']='complete'
    except Exception as error:manifest.update(status='failed',error=repr(error));raise
    finally:
        write(a.output/'manifest.json',manifest)
        write(a.output/'artifact_hashes.json',{x.name:sha(x) for x in sorted(a.output.iterdir()) if x.is_file() and x.name!='artifact_hashes.json'})


if __name__=='__main__':main()
