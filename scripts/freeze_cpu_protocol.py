#!/usr/bin/env python3
"""Choose unused geometry without reading performance; freeze prospective CPU suite."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils.cpu_protocol import geometry_hash,digest_json,file_hash,verify_protocol,runtime_versions
from env.grid_layouts import generate_layout,LAYOUTS


def freeze():
    config_path=ROOT/'configs/cpu/formal_v1.json'
    manifest_path=ROOT/'configs/cpu/formal_protocol_v1.json'
    if config_path.exists() or manifest_path.exists():raise FileExistsError('protocol already frozen')
    source=json.loads((ROOT/'configs/cpu/mechanism_ablation.json').read_text())
    seen=set();sources=[]
    for path in sorted((ROOT/'eval_results').glob('cpu_*/*/observations.npz')):
        with np.load(path) as d:value=geometry_hash(d['occupancy'])
        seen.add(value);sources.append(dict(path=str(path.relative_to(ROOT)),geometry_hash=value))
    chosen=[];maps={};selection=[];used=set(seen)
    for layout in LAYOUTS:
        found=[]
        for seed in range(1000,1100):
            world=generate_layout(layout,128,seed,5);value=geometry_hash(world)
            if value in used:continue
            used.add(value);found.append(seed);chosen.append(dict(layout=layout,seed=seed))
            maps[f'{layout}_seed{seed}']=value
            if len(found)==2:break
        if not found:raise ValueError(f'no unused geometry for {layout}')
        selection.append(dict(layout=layout,seeds=found,requested_maps=2,selected_maps=len(found)))
    source.update(purpose='prospectively frozen synthetic CPU comparison; unseen geometry modulo rotations/reflections',
                  maps=chosen,start_seeds=[301,302],
                  methods=['nearest_frontier','gain_per_cost','gain_control','gain_semantic','gain_structure',
                           'gain_semantic_structure','gain_semantic_structure_rpn','gain_semantic_structure_budget_soft'],
                  frozen_protocol='configs/cpu/formal_protocol_v1.json')
    names=set()
    for directory in ('env','nso','utils'):
        names.update(str(p.relative_to(ROOT)) for p in (ROOT/directory).rglob('*.py'))
    names.update(['scripts/eval_cpu.py','scripts/analyze_cpu_results.py','scripts/analyze_cpu_mechanisms.py',
                  'scripts/analyze_cpu_formal.py','scripts/freeze_cpu_protocol.py',
                  'requirements-cpu.lock.txt',source['rpn_checkpoint']])
    manifest=dict(schema='frozen_cpu_protocol_v1',created_at=datetime.now(timezone.utc).isoformat(),
        runtime=runtime_versions(),
        config_sha256=digest_json(source),files_sha256={n:file_hash(ROOT/n) for n in sorted(names)},
        maps=maps,excluded_geometry_hashes=sorted(seen),exclusion_sources=sources,selection=selection,
        selection_rule='first up to two unused D4-canonical geometries per layout in seed range [1000,1100); no outcome-based selection',
        expected_episodes=len(chosen)*2*8,primary_metric='coverage_auc',
        primary_contrast=['gain_control','gain_semantic_structure'],
        secondary_metrics=['coverage_ratio','revisit_ratio','path_length_m','collisions'],
        statistics='average starts within each map, then equal map weights; 10000 paired map bootstrap resamples, seed 20260910, percentile 95% intervals',
        limitations=['same procedural layout families; unique geometry does not imply independent real scenes',
                     'one unused loop geometry; unequal layout representation',
                     'secondary comparisons exploratory without multiplicity correction',
                     'synthetic salience has no task value calibration; area coverage remains primary',
                     'perfect odometry, no RGB-D or learned SLAM'],
        change_policy='retain all outcomes; changes require new version and new held-out geometry, never silently replace this protocol')
    config_path.write_text(json.dumps(source,indent=2)+'\n')
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    verify_protocol(source,ROOT)
    print(json.dumps(dict(maps=len(maps),episodes=manifest['expected_episodes'],selection=selection),indent=2))

if __name__=='__main__':freeze()
