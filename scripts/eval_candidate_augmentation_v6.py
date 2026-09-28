#!/usr/bin/env python3
"""Execute only geometry-diversity additions to the original sealed route pool.

Prefix raw data are reused by immutable hardlinks. Original 12-route outcomes
remain untouched; downstream union-pool comparisons must combine both sources.
The independent archived-source physical replayer supports this output schema.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils.cpu_protocol import file_hash,digest_json
from utils.rgbd_contract import RGBDFrame,PlanarScan
from scripts import eval_counterfactual_views as executor
from nso.counterfactual_candidates_v6 import augment_candidate_routes


def read(path):return json.loads(path.read_text())


def task(config,entry,output,source):
    original_candidate_function=executor.candidate_routes
    original_prefix_function=executor.generate_prefix
    fixture=[None];input_hashes={};manifest=read(source/'artifact_hashes.json')
    def checked(path):
        relative=str(path.relative_to(source));sha=file_hash(path)
        assert manifest[relative]==sha,relative
        input_hashes[relative]=sha;return path
    def reuse_prefix(world,checkpoints,out):
        key=f"{entry.get('group','main')}_{entry['seed']}_{digest_json(asdict(world.config))[:12]}"
        fixture[0]=source/key;old=fixture[0]/'prefix'
        assert read(checked(fixture[0]/'fixture.json'))['environment']==asdict(world.config)
        out.mkdir();(out/'frames').mkdir();(out/'scans').mkdir()
        rows=read(checked(old/'records.json'))
        for kind in ('frames','scans'):
            for index in range(len(rows)):
                path=checked(old/kind/f'{index:04d}.npz');os.link(path,out/kind/path.name)
        for name in ('records.json','source_hashes.json'):
            shutil.copyfile(checked(old/name),out/name)
        frames=[RGBDFrame.load(out/'frames'/f'{i:04d}.npz') for i in range(len(rows))]
        scans=[PlanarScan.load(out/'scans'/f'{i:04d}.npz') for i in range(len(rows))]
        return frames,scans,rows
    def extra_candidates(mapper,obs,max_actions=48,max_candidates=12):
        original,audit=original_candidate_function(mapper,obs,max_actions,max_candidates)
        expected=read(checked(fixture[0]/f'history_{obs.step:04d}'/'candidates.json'))
        assert json.loads(json.dumps(original))==expected,'original routes no longer reproduce'
        extras,augmentation=augment_candidate_routes(mapper,obs,original,max_actions=max_actions,max_extra=4)
        audit.update(augmentation=augmentation,original_candidate_count=len(original),executed_pool='extra candidates only')
        return extras,audit
    executor.generate_prefix=reuse_prefix;executor.candidate_routes=extra_candidates
    try:
        records=executor.fixture_task(config,entry,output)
        key=fixture[0].name
        # Independently frozen old reference must be identical, not merely similar.
        import numpy as np
        with np.load(checked(fixture[0]/'reference.npz')) as a,np.load(output/key/'reference.npz') as b:
            assert a.files==b.files
            for name in a.files:np.testing.assert_array_equal(a[name],b[name])
        assert all(file_hash(source/name)==sha for name,sha in input_hashes.items())
        executor.write_json(output/key/'original_inputs.json',input_hashes)
        return records
    finally:
        executor.generate_prefix=original_prefix_function;executor.candidate_routes=original_candidate_function


def run(source,output):
    metadata=read(source/'metadata.json');assert metadata['status']=='complete'
    assert read(source/'verification.json')['status']=='passed_full'
    output.mkdir(parents=True,exist_ok=False);started=time.time()
    config=read(source/'config.json');config['scope']='development candidate sufficiency additions; original pools/outcomes preserved; 751/752 only; not independent confirmation'
    config['candidate_augmentation']={'max_extra':4,'selection':'fixed label-agnostic observed-object/azimuth diversity',
        'primary_comparison':'union with original 12 candidate routes; extra-only summary is not a planner comparison',
        'original_run':str(source.resolve()),'expected_upper_bound_new_branches':24}
    names=sorted(set(metadata['source_sha256'])|{'nso/counterfactual_candidates_v6.py','scripts/eval_candidate_augmentation_v6.py'})
    hashes={name:file_hash(ROOT/name) for name in names}
    with zipfile.ZipFile(output/'sources.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in names:z.write(ROOT/name,name)
    current={'status':'running','config_sha256':digest_json(config),'source_sha256':hashes,
        'scope':config['scope'],'started_unix':started,'replayed':False,'original_run':str(source.resolve()),
        'original_verification_sha256':file_hash(source/'verification.json'),'prefix_storage':'hardlinks to verified immutable originals'}
    executor.write_json(output/'metadata.json',current);executor.write_json(output/'config.json',config)
    records=[]
    try:
        with ProcessPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(task,config,entry,output,source) for entry in config['fixtures']]
            for future in as_completed(futures):records.extend(future.result())
        assert all(file_hash(ROOT/name)==sha for name,sha in hashes.items()),'source changed during execution'
        executor.summarize(output,records)
        unavailable=[read(path)|{'path':str(path.relative_to(output))} for path in output.glob('*/history_*/status.json')]
        current.update(status='complete',histories=len(records),branches=sum(h['candidate_count'] for h in records),
                       unavailable_histories=unavailable,elapsed_s=time.time()-started)
    except Exception as exc:
        current.update(status='failed',error=repr(exc));executor.write_json(output/'metadata.json',current);raise
    executor.write_json(output/'metadata.json',current)
    executor.write_json(output/'artifact_hashes.json',{str(p.relative_to(output)):file_hash(p)
        for p in output.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run(args.source,args.output)
