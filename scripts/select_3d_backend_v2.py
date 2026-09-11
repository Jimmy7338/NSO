#!/usr/bin/env python3
"""Independent paired factorial backend study before planner development."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from env.virtual3d import VirtualConfig
from utils.inspection_benchmark import InspectionWorld,TargetEvaluator
from nso.mapping3d_v2 import QualityMapperV2
from utils.cpu_protocol import file_hash


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    protocol=dict(seeds=[115,116,117,118],geometries=['box','shelf'],noise=['iid_01','iid_03'],
        pose_translation_sigma_m=[0.,.01],voxel_m=[.03,.06],truncation_m=[.12,.24],
        selection='highest mean target F1@2cm, provided mean update latency <100ms',
        scope='backend selection only, not planner confirmation')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    files=['nso/mapping3d_v2.py','utils/inspection_benchmark.py','nso/mapping3d.py','scripts/select_3d_backend_v2.py']
    (output/'sources_sha256.json').write_text(json.dumps({f:file_hash(ROOT/f) for f in files},indent=2)+'\n')
    records=[]
    for seed in protocol['seeds']:
        for geometry in protocol['geometries']:
            world=InspectionWorld(VirtualConfig(),seed,geometry)
            evaluator=TargetEvaluator(world,24000,60000)
            for noise in protocol['noise']:
                clean=[]
                for i,(radius,angle) in enumerate([(1.4,-90)]*4+[(1.4,-45)]*4+[(1.4,90)]*4+[(.9,-90)]*4):
                    pose,_=world.pose(radius,angle);clean.append(world.capture(pose,i,noise))
                for pose_sigma in protocol['pose_translation_sigma_m']:
                    frames=[]
                    for i,frame in enumerate(clean):
                        pose=frame.world_from_camera.copy()
                        pose[:3,3]+=np.random.default_rng(np.random.SeedSequence([seed,i,882])).normal(0,pose_sigma,3)
                        frames.append(replace(frame,world_from_camera=pose).validate())
                    for voxel in protocol['voxel_m']:
                        for truncation in protocol['truncation_m']:
                            mapper=QualityMapperV2(world.shape,VirtualConfig(voxel_m=voxel),truncation)
                            started=time.perf_counter()
                            for frame in frames:mapper.update(frame)
                            latency=(time.perf_counter()-started)/len(frames)*1000
                            metrics=evaluator.evaluate(mapper.mesh())
                            records.append(dict(seed=seed,geometry=geometry,noise=noise,pose_sigma_m=pose_sigma,
                                voxel_m=voxel,truncation_m=truncation,update_ms=latency,**metrics))
            print(seed,geometry,len(records),flush=True)
    summaries=[]
    for v in protocol['voxel_m']:
        for t in protocol['truncation_m']:
            subset=[r for r in records if r['voxel_m']==v and r['truncation_m']==t]
            summaries.append(dict(voxel_m=v,truncation_m=t,**{k:float(np.mean([r[k] for r in subset]))
                for k in ('f1_02cm','f1_05cm','surface_error_m','update_ms')}))
    eligible=[r for r in summaries if r['update_ms']<100]
    selected=max(eligible,key=lambda r:r['f1_02cm'])
    (output/'measurements.json').write_text(json.dumps(records,indent=2)+'\n')
    (output/'selection.json').write_text(json.dumps(dict(protocol=protocol,summary=summaries,selected=selected),indent=2)+'\n')
    print(json.dumps(summaries));print('SELECTED',selected)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
