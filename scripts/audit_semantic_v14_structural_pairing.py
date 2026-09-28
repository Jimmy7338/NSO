#!/usr/bin/env python3
"""Test exact observable-geometry pairing before any candidate branch labels."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from env.virtual3d_inspection_v4 import InspectionConfigV4, InspectionWorldV4
from nso.components import NSO_Components
from nso.cpu_sensor_contract_v10 import GridTransform, digest
from nso.decision_capture_v14 import DecisionCaptureRuntimeV14
from nso.decision_replay_v13 import load_packet, save_packet
from scripts.collect_semantic_gain_v13_history import packet, sha, write


def geometry_digest(p):
    frame = replace(p.frame, color_rgb=np.zeros_like(p.frame.color_rgb), semantic=np.zeros_like(p.frame.semantic))
    return replace(p, frame=frame).sha256()


def start(config, world, first):
    args = SimpleNamespace(nso_backend='cpu_v10', eval=True, train_global=False,
        use_open_vocab_semantic=True, use_topo_graph=True, use_rpn_uq=True, use_igcr=True,
        cpu_score_mode='G', cpu_disable_feedback=False, cpu_max_candidates=5,
        cpu_coverage_slots=4, cpu_planner_revision='v10_3_1', cpu_measured_novelty_floor=.25,
        run_id='v13-history-smoke')
    comp = NSO_Components(args)
    comp.initialize('cpu', 1, *world.shape, *world.shape)
    runtime = DecisionCaptureRuntimeV14(comp, 1, world.shape)
    runtime.start_sensor_episode(0, config=world.config,
        transform=GridTransform(tuple(world.shape), world.config.resolution_m), packets=[first],
        total_budget=config['total_budget'], return_anchor=(*first.position, first.heading))
    return runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    pp = Path('configs/virtual3d/semantic_v14_structural_pairing_pilot.json')
    protocol = json.loads(pp.read_text())
    source = Path(protocol['source_history'])
    decision_source = Path(protocol['source_decisions'])
    for root in (source, decision_source):
        m = json.loads((root/'manifest.json').read_text())
        if m['status'] != 'complete':
            raise ValueError(f'incomplete source: {root}')
        for name, expected in m['source_sha256'].items():
            assert sha(Path(name)) == expected, name
        for name, expected in json.loads((root/'artifact_hashes.json').read_text()).items():
            assert sha(root/name) == expected, name
    sm = json.loads((source/'manifest.json').read_text())
    scene = json.loads(Path(sm['history_protocol']['scene_protocol']).read_text())
    declared = next(c for c in scene['contexts'] if c['id'] == protocol['layout_context'])
    assert declared['seed'] == protocol['baseline_structure_seed']
    settings = {**scene['shared_conditions'], **{k:v for k,v in declared.items() if k not in ('id','seed')}}
    baseline_decisions = {r['action_id']:r for r in json.loads((decision_source/'all_decisions.json').read_text())}
    baseline = [load_packet(source/f'packets/{i:04d}.npz') for i in range(protocol['max_prefix_paid_actions']+1)]
    config = dict(parent=protocol['layout_context'], total_budget=protocol['total_budget'])
    output.mkdir(parents=True, exist_ok=False)
    names = sorted({str(pp), str(Path(__file__).relative_to(ROOT)), *sm['source_sha256'],
                    'nso/decision_capture_v14.py'})
    frozen = {name:sha(Path(name)) for name in names}
    manifest = dict(status='running', protocol=protocol, source_sha256=frozen,
        created_utc=datetime.now(timezone.utc).isoformat(),
        input_inventories={str(root):sha(root/'artifact_hashes.json') for root in (source,decision_source)})
    write(output/'manifest.json',manifest)
    with zipfile.ZipFile(output/'sources.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(name,name)
    variants = []
    try:
        for seed in protocol['alternative_structure_seeds']:
            folder = output/f'structure_{seed}'
            folder.mkdir(); (folder/'packets').mkdir()
            world = InspectionWorldV4(InspectionConfigV4(**settings),seed=seed,semantic_condition='aligned')
            initial = packet(world,config,None)
            runtime = start(config,world,initial)
            rows, decisions = [], []
            current = initial
            mismatch = None
            for i in range(protocol['max_prefix_paid_actions']+1):
                save_packet(folder/f'packets/{i:04d}.npz',current)
                equal = geometry_digest(current) == geometry_digest(baseline[i])
                rows.append(dict(action_id=i, geometry_equal=equal,
                    current_geometry_sha256=geometry_digest(current),baseline_geometry_sha256=geometry_digest(baseline[i]),
                    semantic_equal=digest(current.frame.semantic)==digest(baseline[i].frame.semantic)))
                if not equal:
                    mismatch=dict(action_id=i,reason='depth_scan_pose_prefix_differs')
                    break
                count = len(runtime.decision_snapshots)
                action = runtime.next_local_action(0)
                if len(runtime.decision_snapshots)>count:
                    snap = runtime.decision_snapshots[-1]
                    if i in protocol['decision_steps']:
                        old = baseline_decisions[i]
                        pool = snap['candidate_sha256']==old['candidate_sha256']
                        geom = [f['geometry'] for f in snap['features']]==[f['geometry'] for f in old['features']]
                        votes = [[a['class_vote'] for a in d['observed_assets']] for d in snap['descriptors']]
                        oldvotes = [[a['class_vote'] for a in d['observed_assets']] for d in old['descriptors']]
                        decisions.append(dict(action_id=i,pool_equal=pool,geometry_features_equal=geom,
                            observed_votes_differ=votes!=oldvotes,eligible_semantic_pair=pool and geom and votes!=oldvotes))
                        write(folder/f'decision_{i:03d}.json',snap)
                if i == protocol['max_prefix_paid_actions']:
                    break
                if action != baseline[i+1].action:
                    mismatch=dict(action_id=i,reason='geometry_policy_action_differs')
                    break
                frame, collision, done = world.step(action)
                current = packet(world,config,action,frame,collision,done)
                if collision or done:
                    mismatch=dict(action_id=i+1,reason='unexpected_prefix_terminal')
                    save_packet(folder/f'packets/{i+1:04d}.npz',current)
                    break
                runtime.observe(0,world.step_count,None,None,None,sensor_packet=current)
            recorded = {d['action_id'] for d in decisions}
            for step in protocol['decision_steps']:
                if step not in recorded:
                    decisions.append(dict(action_id=step,eligible_semantic_pair=False,
                        reason='prefix_mismatch_before_decision' if mismatch else 'no_shared_natural_boundary'))
            decisions.sort(key=lambda d:d['action_id'])
            write(folder/'prefix_checks.json',rows)
            result=dict(structure_seed=seed,first_mismatch=mismatch,decisions=decisions,
                        paid_prefix_actions=world.step_count,collisions=world.collisions,
                        prefix_only_not_complete_episode=True)
            variants.append(result)
            write(folder/'result.json',result);write(output/'partial.json',variants)
            print(json.dumps(result),flush=True)
        write(output/'summary.json',dict(status='complete',variants=variants,parent_groups=1,
            eligible_semantic_pairs=sum(d['eligible_semantic_pair'] for r in variants for d in r['decisions']),
            future_candidate_outcomes=0,training_allowed=False,semantic_efficacy_proven=False))
        for name, expected in frozen.items():
            assert sha(Path(name))==expected,name
        manifest['status']='complete'
    except Exception as error:
        manifest.update(status='failed',error=repr(error));raise
    finally:
        write(output/'manifest.json',manifest)
        write(output/'artifact_hashes.json',{str(p.relative_to(output)):sha(p)
            for p in sorted(output.rglob('*')) if p.is_file() and p.name!='artifact_hashes.json'})


if __name__=='__main__':
    main()
