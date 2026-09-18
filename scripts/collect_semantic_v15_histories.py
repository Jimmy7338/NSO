#!/usr/bin/env python3
"""Fresh canonical-sensor geometry histories; full per-variant physical replay."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from env.canonical_rgbd_v15 import CanonicalRGBDInspectionWorldV15
from nso.components import NSO_Components
from nso.decision_capture_v14 import DecisionCaptureRuntimeV14
from scripts import collect_semantic_v14_cold_start as driver
from scripts.collect_semantic_gain_v13_history import sha,write


def run_seed(root,protocol,seed,replay=False):
    folder=root/f'structure_{seed}'
    if not replay:folder.mkdir()
    history=dict(scene_protocol=protocol['scene_protocol'],context=protocol['context'],
                 total_budget=protocol['total_budget'],semantic_condition='aligned',score_mode='G',
                 planner_revision='v10_3_1',coverage_slots=4)
    instances=[]
    def world_factory(config,**kwargs):
        return CanonicalRGBDInspectionWorldV15(config,seed=seed,semantic_condition=kwargs['semantic_condition'])
    def start(config,world,transform,first):
        args=SimpleNamespace(nso_backend='cpu_v10',eval=True,train_global=False,
            use_open_vocab_semantic=True,use_topo_graph=True,use_rpn_uq=True,use_igcr=True,
            cpu_score_mode='G',cpu_disable_feedback=False,cpu_max_candidates=config['candidate_cap'],
            cpu_coverage_slots=4,cpu_planner_revision='v10_3_1',cpu_measured_novelty_floor=.25,run_id='v13-history-smoke')
        comp=NSO_Components(args);comp.initialize('cpu',1,*world.shape,*world.shape)
        runtime=DecisionCaptureRuntimeV14(comp,1,world.shape)
        runtime.start_sensor_episode(0,config=world.config,transform=transform,packets=[first],
            total_budget=config['total_budget'],return_anchor=(*first.position,first.heading))
        instances.append(runtime)
        return runtime
    original_world,original_start=driver.InspectionWorldV4,driver.start
    driver.InspectionWorldV4,driver.start=world_factory,start
    try:terminal=driver.execute(folder,history,replay=replay)
    finally:driver.InspectionWorldV4,driver.start=original_world,original_start
    runtime=instances[0]
    if replay:
        assert runtime.decision_snapshots==json.loads((folder/'all_decisions.json').read_text())
        write(folder/'verification.json',dict(status='passed',physical_actions=terminal['paid_actions'],
            global_decisions=len(runtime.decision_snapshots),all_sensor_bytes_and_states_exact=True))
    else:
        write(folder/'all_decisions.json',runtime.decision_snapshots)
        write(folder/'runtime_audit.json',runtime.audit)
        write(folder/'module_calls.json',runtime.components._cpu_backend.calls)
    return dict(structure_seed=seed,paid_actions=terminal['paid_actions'],global_decisions=len(runtime.decision_snapshots),
                candidate_descriptors=sum(len(r['candidates']) for r in runtime.decision_snapshots),
                collisions=terminal['collisions'],returned=terminal['returned_to_initial_pose'],
                terminal_reason=terminal['termination']['reason'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--replay-seed',type=int);args=p.parse_args();root=args.output
    if args.replay_seed is not None:
        m=json.loads((root/'manifest.json').read_text())
        for name,expected in m['source_sha256'].items():assert sha(Path(name))==expected,name
        assert args.replay_seed in m['protocol']['structure_seeds']
        run_seed(root,m['protocol'],args.replay_seed,True);return
    root.mkdir(parents=True,exist_ok=False)
    pp=Path('configs/virtual3d/semantic_v15_canonical_histories.json');protocol=json.loads(pp.read_text())
    names=sorted({str(pp),str(Path(__file__).relative_to(ROOT)),protocol['scene_protocol'],
        'scripts/collect_semantic_v14_cold_start.py','scripts/collect_semantic_gain_v13_history.py',
        'tests/virtual3d/test_canonical_rgbd_v15.py',
        *[str(p) for base in ('env','nso','utils') for p in Path(base).rglob('*.py')]})
    frozen={name:sha(Path(name)) for name in names}
    m=dict(status='running',protocol=protocol,source_sha256=frozen,
           created_utc=datetime.now(timezone.utc).isoformat(),old_trajectory_reuse=False,training_allowed=False)
    write(root/'manifest.json',m)
    with zipfile.ZipFile(root/'sources.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in names:z.write(name,name)
    rows=[]
    try:
        for seed in protocol['structure_seeds']:
            row=run_seed(root,protocol,seed)
            subprocess.run([sys.executable,str(Path(__file__)),'--output',str(root),'--replay-seed',str(seed)],check=True)
            row['independent_physical_replay_passed']=True;rows.append(row)
            write(root/'partial.json',rows);print(json.dumps(row),flush=True)
        for name,expected in frozen.items():assert sha(Path(name))==expected,name
        write(root/'summary.json',dict(status='complete',variants=rows,parent_groups=1,
            new_candidate_outcomes=0,semantic_efficacy_proven=False,training_allowed=False))
        m['status']='complete'
    except Exception as error:m.update(status='failed',error=repr(error));raise
    finally:
        write(root/'manifest.json',m)
        write(root/'artifact_hashes.json',{str(p.relative_to(root)):sha(p)
            for p in sorted(root.rglob('*')) if p.is_file() and p.name!='artifact_hashes.json'})


if __name__=='__main__':main()
