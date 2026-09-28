#!/usr/bin/env python3
"""New sensor-only model on already executed development routes; no new data."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1'); os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from env.virtual3d_inspection_v4 import InspectionConfigV4
from utils.cpu_protocol import file_hash
from utils.rgbd_contract import RGBDFrame,PlanarScan
from scripts.eval_counterfactual_views import remap,grid_config,summarize,write_json
from nso.counterfactual_view_scoring import score_routes,CounterfactualScoreConfig


def run(source,output):
    from nso.semantic_completion_v5 import ObjectCompletionModel as ObjectCompletionModelV5
    from scripts.analyze_counterfactual_views import analyze
    source_meta=json.loads((source/'metadata.json').read_text())
    verification=json.loads((source/'verification.json').read_text())
    assert source_meta['status']=='complete' and verification['status']=='passed_full'
    original=json.loads((source/'summary.json').read_text())
    original_hashes=json.loads((source/'artifact_hashes.json').read_text())
    # The map/sensor model is held fixed while the completion model is replaced.
    dependencies=['env/virtual3d.py','env/virtual3d_v2.py','env/virtual3d_inspection_v4.py',
                  'nso/mapping3d.py','nso/mapping3d_v2.py','nso/camera_mapping_v2.py','nso/semantic_completion_v3.py']
    for name in dependencies:assert file_hash(ROOT/name)==source_meta['source_sha256'][name],name
    output.mkdir(parents=True,exist_ok=False);started=time.time()
    names=sorted({str(p.relative_to(ROOT)) for folder in ('env','nso','utils') for p in (ROOT/folder).glob('*.py')} |
                 {'scripts/rescore_counterfactual_v5.py','scripts/eval_counterfactual_views.py',
                  'scripts/analyze_counterfactual_views.py','requirements-3d.lock.txt'})
    hashes={name:file_hash(ROOT/name) for name in names}
    with zipfile.ZipFile(output/'sources.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in names:archive.write(ROOT/name,name)
    meta={'status':'running','scope':'in-sample V5 model rescore of fixed executed routes; no new navigation or independent evidence',
          'source':str(source.resolve()),'source_sha256':hashes,'new_physical_branches':0,
          'unavailable_histories':source_meta.get('unavailable_histories',[]),'source_verification_sha256':file_hash(source/'verification.json')}
    write_json(output/'metadata.json',meta)
    config=json.loads((source/'config.json').read_text());config['scope']=meta['scope'];write_json(output/'config.json',config)
    variants={'v3_rate':(None,'rate'),'v3_auc':(None,'horizon_auc'),
              'v5_rate':(ObjectCompletionModelV5,'rate'),'v5_auc':(ObjectCompletionModelV5,'horizon_auc')}
    records={name:[] for name in variants};input_hashes={};old_reproduced=0
    for name in variants:
        target=output/name;target.mkdir()
        write_json(target/'config.json',{**config,'model_scoring_variant':name})
    def check_input(path):
        actual=file_hash(path);key=str(path.relative_to(source))
        assert original_hashes.get(key)==actual,f'original artifact changed: {key}'
        input_hashes[str(path.resolve())]=actual
    for name in ('summary.json','config.json','metadata.json'):
        check_input(source/name)
    for history in original['histories']:
        fixture=source/history['fixture'];entry=json.loads((fixture/'fixture.json').read_text());c=InspectionConfigV4(**entry['environment'])
        index=history['history_step'];folder=source/history['predictions'];hdir=folder.parent
        for path in [fixture/'fixture.json',fixture/'prefix/records.json',folder,
                     hdir/'candidates.json',hdir/'outcomes.json']:
            check_input(path)
        for i in range(index+1):
            check_input(fixture/'prefix/frames'/f'{i:04d}.npz')
            check_input(fixture/'prefix/scans'/f'{i:04d}.npz')
        raw_rows=json.loads((fixture/'prefix/records.json').read_text())[:index+1]
        frames=[RGBDFrame.load(fixture/'prefix/frames'/f'{i:04d}.npz') for i in range(index+1)]
        scans=[PlanarScan.load(fixture/'prefix/scans'/f'{i:04d}.npz') for i in range(index+1)]
        shape=(round(c.height_m/c.resolution_m),round(c.width_m/c.resolution_m))
        mappers={mode:remap(frames,scans,raw_rows,c,shape,mode) for mode in ('aligned','shuffled','absent')}
        routes=json.loads((hdir/'candidates.json').read_text())
        old=json.loads(folder.read_text());score_config=CounterfactualScoreConfig(grid_config(c),c)
        baseline=score_routes(mappers,routes,score_config)
        assert baseline['candidates']==old['candidates'] and baseline['selected']==old['selected'],'old scores failed exact reproduction'
        old_reproduced+=1
        for name,(factory,objective) in variants.items():
            target=output/name;out=target/history['fixture']/hdir.name;out.mkdir(parents=True)
            prediction=baseline if name=='v3_rate' else score_routes(mappers,routes,
                        replace(score_config,score_objective=objective),model_factory=factory)
            write_json(out/'predictions.json',prediction)
            # Only small outcome tables are copied; all physical data stay shared.
            for filename in ('candidates.json','outcomes.json'):
                shutil.copyfile(hdir/filename,out/filename)
            rows={**history,'predictions':str((out/'predictions.json').relative_to(target)),
                  'outcomes':str((out/'outcomes.json').relative_to(target)),'shared_physical_source':str(hdir.resolve())}
            records[name].append(rows)
        print('rescored',history['fixture'],index,flush=True)
        del mappers
    assert all(file_hash(ROOT/name)==sha for name,sha in hashes.items()),'sources changed during model rescore'
    for name,rows in records.items():
        target=output/name;summarize(target,rows)
        write_json(target/'metadata.json',{**meta,'status':'complete','model_scoring_variant':name,
                   'histories':len(rows),'branches':sum(h['candidate_count'] for h in rows),
                   'source_archive':'../sources.zip','new_physical_branches':0})
        analyze(target)
    meta.update(status='complete',histories=len(original['histories']),branches=sum(h['candidate_count'] for h in original['histories']),
                variants=list(variants),
                old_prediction_histories_exactly_reproduced=old_reproduced,elapsed_s=time.time()-started)
    write_json(output/'metadata.json',meta);write_json(output/'input_hashes.json',input_hashes)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.source,a.output)
