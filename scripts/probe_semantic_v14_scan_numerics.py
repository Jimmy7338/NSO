#!/usr/bin/env python3
"""Post-hoc renderer diagnostic at fixed poses; not a new policy trajectory."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from env.canonical_box_scan_v14 import CanonicalScanInspectionWorldV14
from env.virtual3d_inspection_v4 import InspectionConfigV4, InspectionWorldV4
from nso.decision_replay_v13 import load_packet
from scripts.collect_semantic_gain_v13_history import sha,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads(Path('configs/virtual3d/semantic_v14_structural_pairing_pilot.json').read_text())
    source=Path(protocol['source_history']);m=json.loads((source/'manifest.json').read_text())
    scene_path=Path(m['history_protocol']['scene_protocol']);scene=json.loads(scene_path.read_text())
    declared=next(c for c in scene['contexts'] if c['id']==protocol['layout_context'])
    settings={**scene['shared_conditions'],**{k:v for k,v in declared.items() if k not in ('id','seed')}}
    files=[source/f'packets/{i:04d}.npz' for i in range(27)]
    inventory=json.loads((source/'artifact_hashes.json').read_text())
    for path in files:assert sha(path)==inventory[str(path.relative_to(source))]
    packets=[load_packet(path) for path in files]
    seeds=[protocol['baseline_structure_seed'],*protocol['alternative_structure_seeds']]
    names=sorted({str(Path(__file__).relative_to(ROOT)),str(scene_path),
        'configs/virtual3d/semantic_v14_structural_pairing_pilot.json',
        *[str(p) for base in ('nso','env','utils') for p in Path(base).rglob('*.py')]})
    frozen={name:sha(Path(name)) for name in names}
    write(out/'manifest.json',dict(status='running',source_sha256=frozen,post_hoc=True,
        fixed_observation_poses_only=True,source_packets={str(p):sha(p) for p in files}))
    with zipfile.ZipFile(out/'sources.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in names:z.write(name,name)
    frames={};rows=[]
    for seed in seeds:
        world=CanonicalScanInspectionWorldV14(InspectionConfigV4(**settings),seed=seed,semantic_condition='aligned')
        frames[seed]=[]
        for packet in packets:
            world.position,world.heading,world.step_count=packet.position,packet.heading,packet.action_id
            frame=world.sense();canonical=world.scan().ranges_m
            original=InspectionWorldV4.scan(world).ranges_m
            frames[seed].append((frame.depth_m.copy(),frame.semantic.copy(),canonical.copy(),original.copy()))
        write(out/f'seed_{seed}_renderer_bound.json',dict(
            max_original_canonical_range_difference_m=max(float(np.max(np.abs(r[2]-r[3]))) for r in frames[seed])))
    baseline=frames[seeds[0]]
    for seed in seeds[1:]:
        details=[]
        for i,(a,b) in enumerate(zip(baseline,frames[seed])):
            details.append(dict(action_id=i,depth_equal=np.array_equal(a[0],b[0]),
                semantic_equal=np.array_equal(a[1],b[1]),canonical_scan_equal=np.array_equal(a[2],b[2]),
                original_scan_equal=np.array_equal(a[3],b[3]),
                max_depth_difference_m=float(np.max(np.abs(a[0]-b[0]))),
                max_canonical_scan_difference_m=float(np.max(np.abs(a[2]-b[2]))),
                max_original_scan_difference_m=float(np.max(np.abs(a[3]-b[3])))))
        rows.append(dict(seed=seed,
            first_original_geometry_difference=next((r['action_id'] for r in details if not (r['depth_equal'] and r['original_scan_equal'])),None),
            first_canonical_geometry_difference=next((r['action_id'] for r in details if not (r['depth_equal'] and r['canonical_scan_equal'])),None),
            first_semantic_difference=next((r['action_id'] for r in details if not r['semantic_equal']),None),details=details))
    for name,expected in frozen.items():assert sha(Path(name))==expected,name
    write(out/'result.json',dict(status='complete',rows=rows,post_hoc=True,
        canonical_renderer_unit_tests_passed=4,raw_pairing_failure_retained=True,
        new_canonical_policy_history_collected=False,eligible_training_pairs_certified=False,
        boundary='Fixed-pose rendering probe only. New canonical scan changes observations; old states and rewards cannot be reused as new-renderer trajectories.'))
    manifest=json.loads((out/'manifest.json').read_text());manifest['status']='complete';write(out/'manifest.json',manifest)
    write(out/'artifact_hashes.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print(json.dumps([{k:v for k,v in row.items() if k!='details'} for row in rows]),flush=True)


if __name__=='__main__':main()
