#!/usr/bin/env python3
"""Frozen single-intervention outcomes with per-branch subprocess replay."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace
import zipfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from env.canonical_rgbd_v15 import CanonicalRGBDInspectionWorldV15
from env.virtual3d_inspection_v4 import InspectionConfigV4
from nso.candidate_intervention_v15 import CandidateInterventionRuntimeV15
from nso.components import NSO_Components
from nso.cpu_sensor_contract_v10 import GridTransform, digest
from nso.decision_replay_v13 import load_packet, save_packet, decision_state
from scripts.collect_semantic_gain_v13_history import packet, sha, write
from utils.reconstruction_metrics import ReconstructionEvaluator
from utils.counterfactual_surface_visibility import reference_visible


def seal(folder):
    write(folder / 'artifact_hashes.json', {str(p.relative_to(folder)): sha(p)
          for p in sorted(folder.rglob('*')) if p.is_file() and p.name != 'artifact_hashes.json'})


def verify_sources(manifest):
    for name, expected in manifest['source_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f'changed source: {name}')


def verify_inputs(protocol):
    for name, expected in protocol['input_inventory_sha256'].items():
        folder = ROOT / name
        if sha(folder / 'artifact_hashes.json') != expected:
            raise ValueError('input inventory differs')
        for name, expected in json.loads((folder / 'artifact_hashes.json').read_text()).items():
            if sha(folder / name) != expected:
                raise ValueError(f'changed evidence: {folder / name}')


def branch_name(d):
    return f"structure_{d['structure_seed']}_step_{d['action_id']}_candidate_{d['candidate_id']}"


def run_branch(root, protocol, declaration, replay=False):
    seed, step = declaration['structure_seed'], declaration['action_id']
    folder = root / branch_name(declaration)
    if not replay:
        folder.mkdir(); (folder / 'packets').mkdir()
    history = ROOT / protocol['histories']
    hp = json.loads((history / 'manifest.json').read_text())['protocol']
    scene = json.loads((ROOT / hp['scene_protocol']).read_text())
    declared = next(c for c in scene['contexts'] if c['id'] == protocol['parent'])
    settings = {**scene['shared_conditions'], **{k:v for k,v in declared.items() if k not in ('id','seed')}}
    world = CanonicalRGBDInspectionWorldV15(InspectionConfigV4(**settings), seed=seed, semantic_condition='aligned')
    hf = history / f'structure_{seed}'
    checkpoint = next(c for c in json.loads((hf / 'all_decisions.json').read_text()) if c['action_id'] == step)
    certificate = next(c for s in protocol['states'] if s['action_id'] == step
                       for c in s['certificates'] if c['structure_seed'] == seed)
    if checkpoint['candidate_sha256'] != certificate['candidate_pool_sha256']:
        raise ValueError('declared pool differs')
    config = dict(parent=protocol['parent'])
    first = packet(world, config, None)
    assert first.sha256() == load_packet(hf / 'packets/0000.npz').sha256()
    args = SimpleNamespace(nso_backend='cpu_v10',eval=True,train_global=False,
        use_open_vocab_semantic=True,use_topo_graph=True,use_rpn_uq=True,use_igcr=True,
        cpu_score_mode='G',cpu_disable_feedback=False,cpu_max_candidates=5,cpu_coverage_slots=4,
        cpu_planner_revision='v10_3_1',cpu_measured_novelty_floor=.25,run_id='v13-history-smoke')
    comp = NSO_Components(args); comp.initialize('cpu',1,*world.shape,*world.shape)
    runtime = CandidateInterventionRuntimeV15(comp,1,world.shape)
    transform = GridTransform(tuple(world.shape),world.config.resolution_m)
    runtime.start_sensor_episode(0,config=world.config,transform=transform,packets=[first],
        total_budget=protocol['total_task_budget'],return_anchor=(*first.position,first.heading))
    runtime.configure_intervention(checkpoint,declaration['candidate_id'])
    ep = protocol['evaluation']
    evaluators = {s: ReconstructionEvaluator(world,count=ep['reference_count'],seed=s) for s in ep['reference_seeds']}
    evaluator = evaluators[ep['reference_seeds'][0]]
    weight = float(world.mesh.get_surface_area()) / ep['reference_count']
    visible = reference_visible(evaluator.reference,first.frame,evaluator.truth,
                                first.frame.world_from_camera,world.config.max_depth_m)
    def evaluate():
        mapper = runtime.states[0]['mapper']
        cov = float(np.count_nonzero((mapper.belief != -1) & world.reachable) / world.reachable.sum())
        return {str(s): dict(coverage_2d=cov,**e.evaluate(mapper.mesh(),cov,thresholds=ep['thresholds_m']))
                for s,e in evaluators.items()}
    def mesh_arrays():
        mesh = runtime.states[0]['mapper'].mesh()
        return {k:np.asarray(getattr(mesh,k)) for k in ('vertices','triangles','vertex_colors')}
    expected_actions = json.loads((folder/'actions.json').read_text()) if replay else None
    actions=[]; before=None; prefix_visible=None; forced=[]; reached=False; started=perf_counter()
    while not runtime.states[0]['closed']:
        action=runtime.next_local_action(0)
        if runtime.intervention_receipt is not None and before is None:
            if world.step_count != step: raise ValueError('intervention frame differs')
            before=evaluate(); prefix_visible=visible.copy()
            if not replay: np.savez_compressed(folder/'prefix_mesh.npz',**mesh_arrays())
        if action is None: break
        active=runtime.states[0]['option']
        receipt=runtime.intervention_receipt
        is_forced=bool(receipt and active and active['option_id']==receipt['option']['option_id'])
        frame,collision,done=world.step(action)
        observed=packet(world,config,action,frame,collision,done).validate(transform,world.config)
        if world.step_count <= step:
            assert observed.sha256()==load_packet(hf/f'packets/{world.step_count:04d}.npz').sha256(), 'prefix diverged'
        elif replay:
            assert observed.sha256()==load_packet(folder/f'packets/{world.step_count:04d}.npz').sha256(), 'replay packet differs'
        else: save_packet(folder/f'packets/{world.step_count:04d}.npz',observed)
        runtime.observe(0,world.step_count,None,None,None,sensor_packet=observed)
        visible |= reference_visible(evaluator.reference,frame,evaluator.truth,frame.world_from_camera,world.config.max_depth_m)
        if is_forced:
            forced.append(action)
            reached |= [*world.position,world.heading]==receipt['option']['pose']
        row=dict(action_id=world.step_count,action=action,pose=[*world.position,world.heading],
                 collision=bool(collision),packet_sha256=observed.sha256(),forced_option=is_forced)
        if replay: assert row==expected_actions[len(actions)],'replay action differs'
        actions.append(row)
        if not replay: write(folder/'actions.json',actions)
        if world.step_count%32==0:
            print(f'{"REPLAY" if replay else "COLLECT"} {folder.name} paid={world.step_count}',flush=True)
    if before is None: raise ValueError('intervention never executed')
    terminal=runtime.sensor_episode_summary(0)
    result=dict(**declaration,parent=protocol['parent'],before=before,after=evaluate(),
        total_paid_actions=world.step_count,continuation_paid_actions=world.step_count-step,
        total_budget=protocol['total_task_budget'],collisions=world.collisions,
        termination=terminal['termination'],first_target_reached=bool(reached),
        forced_outbound_complete=forced==receipt['option']['outbound_actions'],forced_actions=forced,
        new_visible_surface_m2=float(np.count_nonzero(visible & ~prefix_visible))*weight,
        continuation_path_distance_m=sum(r['action']=='forward' for r in actions[step:])*world.config.resolution_m,
        total_path_distance_m=sum(r['action']=='forward' for r in actions)*world.config.resolution_m,
        action_time_s=world.step_count*world.config.action_duration_s,
        intervention=receipt,global_decisions=len(runtime.decision_snapshots),
        recovery_denials=[r for r in runtime.audit if r['event']=='v14_route_denied'],
        final_state_sha256=decision_state(runtime)['sha256'],
        modules_called=sorted({c['module'] for c in comp._cpu_backend.calls}),
        future_rewards_used_by_planner=False,semantic_efficacy_proven=False)
    if replay:
        assert len(actions)==len(expected_actions)
        assert digest(result)==digest(json.loads((folder/'result.json').read_text())), 'result replay differs'
        assert runtime.decision_snapshots==json.loads((folder/'all_decisions.json').read_text())
        with np.load(folder/'final_mesh.npz',allow_pickle=False) as stored:
            for name,value in mesh_arrays().items():np.testing.assert_array_equal(value,stored[name])
        write(folder/'verification.json',dict(status='passed',physical_actions=world.step_count,
            sensor_actions_decisions_metrics_and_final_mesh_exact=True))
    else:
        write(folder/'result.json',result);write(folder/'terminal.json',terminal)
        write(folder/'all_decisions.json',runtime.decision_snapshots)
        write(folder/'runtime_audit.json',runtime.audit);write(folder/'module_calls.json',comp._cpu_backend.calls)
        write(folder/'timing.json',dict(execution_and_evaluation_wall_time_s=perf_counter()-started))
        np.savez_compressed(folder/'final_mesh.npz',**mesh_arrays())
        np.savez_compressed(folder/'visibility.npz',before=prefix_visible,after=visible,
                            reference=evaluator.reference,classes=evaluator.classes,sample_weight_m2=weight)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--replay-index',type=int);p.add_argument('--all',action='store_true');a=p.parse_args()
    if a.replay_index is not None:
        m=json.loads((a.output/'manifest.json').read_text());verify_sources(m)
        run_branch(a.output,m['protocol'],m['declarations'][a.replay_index],True);return
    pp=ROOT/'configs/virtual3d/semantic_v15_observed_group_outcome_pilot.json'
    protocol=json.loads(pp.read_text());verify_inputs(protocol)
    hm=json.loads((ROOT/protocol['histories']/'manifest.json').read_text())
    frozen=dict(hm['source_sha256'])
    for f in [pp,Path(__file__).resolve(),ROOT/'nso/candidate_intervention_v15.py']:
        frozen[str(f.relative_to(ROOT))]=sha(f)
    declarations=protocol['smoke_subset'] if not a.all else [
        dict(structure_seed=c['structure_seed'],action_id=s['action_id'],candidate_id=cid)
        for s in protocol['states'] for c in s['certificates'] for cid in c['candidate_ids']]
    m=dict(status='running',protocol=protocol,source_sha256=frozen,declarations=declarations,
           created_utc=datetime.now(timezone.utc).isoformat(),training_allowed=False)
    verify_sources(m);a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'manifest.json',m)
    with zipfile.ZipFile(a.output/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name in frozen:z.write(ROOT/name,name)
    rows=[]
    try:
        for index,d in enumerate(declarations):
            r=run_branch(a.output,protocol,d)
            subprocess.run([sys.executable,str(Path(__file__)),'--output',str(a.output),
                            '--replay-index',str(index)],check=True)
            rows.append(r);write(a.output/'partial.json',rows)
            print(json.dumps(dict(branch=branch_name(d),replayed=True,paid=r['total_paid_actions'],
                target_reached=r['first_target_reached'],joint=r['after']['2026']['joint_05cm'])),flush=True)
        verify_sources(m);verify_inputs(protocol)
        write(a.output/'summary.json',dict(status='complete',outcomes=len(rows),replayed=len(rows),
            collisions=sum(r['collisions'] for r in rows),
            failed=sum(r['termination']['failed'] for r in rows),
            returns=sum(r['termination']['returned_to_anchor'] for r in rows),
            budget_violations=sum(r['total_paid_actions']>r['total_budget'] for r in rows),
            first_targets_reached=sum(r['first_target_reached'] for r in rows),
            parent_groups=1,training_allowed=False,semantic_efficacy_proven=False))
        m['status']='complete'
    except Exception as error:m.update(status='failed',error=repr(error));raise
    finally:write(a.output/'manifest.json',m);seal(a.output)


if __name__=='__main__':main()
