#!/usr/bin/env python3
"""Orchestrate isolated observation-only banks; no temperature or outcome tuning."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
PARENT=Path('/home/ouc/nso_robot/analyses/dual_template_uncertainty_20261120')
PYTHON=Path('/home/ouc/nso_robot/envs/science312/bin/python')
SOURCE=Path('/home/ouc/nso_robot/analyses/reproduction_20261104/simulation_publication_stage/experiments/unknown_exploration_20261104/source')
INSTANCE=Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029/nso_bridge/nso_bridge/instance_support.py')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def resource(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def read(path):return json.loads(Path(path).read_text())
def write(path,data):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temp.replace(path)


def frozen_paths():
    parent=PARENT/'calibration/inference_freeze.json'
    paths=[Path(__file__),HERE/'infer_observations.py',HERE/'isolated_io.py',HERE/'test_isolation.py',
        HERE/'inference_protocol.json',parent,PARENT/'audit/calibration_attempt_001/selected_scale.json',
        ROOT/'generator/freeze.json',ROOT/'generator/dataset_manifest.json',ROOT/'generator/counterfactual_sensor_receipt.json']
    paths += [Path(p)for p in read(parent)['files_sha256']]
    paths += [Path(p)for p in read(ROOT/'generator/freeze.json')['files_sha256']]
    paths += list(SOURCE.rglob('*.py'))+[INSTANCE]
    for row in read(ROOT/'generator/dataset_manifest.json')['cases']+read(ROOT/'generator/counterfactual_sensor_receipt.json')['fixtures']:
        paths += [Path(row['observation_manifest']['path']),Path(row['truth']['path'])]
        for frame in read(row['observation_manifest']['path'])['frames']:
            paths += [Path(frame['frame']['path']),Path(frame['owner']['path'])]
    return sorted(set(paths))


def freeze():
    assert not (HERE/'inference_freeze.json').exists()
    assert read(HERE/'inference_protocol.json')['status']=='frozen_before_model_inference'
    assert read(PARENT/'audit/calibration_attempt_001/selected_scale.json')['sigma_m']==.01
    for receipt in (PARENT/'calibration/inference_freeze.json',ROOT/'generator/freeze.json'):
        for name,digest in read(receipt)['files_sha256'].items():
            assert sha(Path(name))==digest,name
    write(HERE/'inference_freeze.json',dict(schema='B9_isolated_inference_freeze_v1',created_unix_s=time.time(),
        files_sha256={str(p):sha(p)for p in frozen_paths()},temperature_selected=False,sigma_m=.01,
        interval_correction_applied=False,policy_or_branch_actions=0))


def verify():
    freeze_data=read(HERE/'inference_freeze.json')
    assert {str(p):sha(p)for p in frozen_paths()}==freeze_data['files_sha256']


def run(workers):
    verify();dataset=read(ROOT/'generator/dataset_manifest.json')
    if (HERE/'inference_index.json').exists():raise FileExistsError('Retain original batch; explicit new attempt is required')
    jobs=[dict(case_id=row['case_id'],observation_manifest=row['observation_manifest'])for row in dataset['cases']]
    fixture=read(ROOT/'generator/counterfactual_sensor_receipt.json')
    for i,row in enumerate(fixture['fixtures']):jobs.append(dict(case_id=f'counterfactual_{i}',observation_manifest=row['observation_manifest'],counterfactual=True))
    output=HERE/'inference_attempt_001';output.mkdir(exist_ok=False)
    state=dict(status='running',freeze=resource(HERE/'inference_freeze.json'),cases=[],counterfactual=[],pending_case_ids=[row['case_id']for row in jobs],failures=[])
    write(HERE/'inference_index.json',state);start=time.monotonic()
    def execute(row):
        dest=output/row['case_id'];log=output/(row['case_id']+'.log')
        args=[str(PYTHON),str(HERE/'infer_observations.py'),'--observations',row['observation_manifest']['path'],'--output',str(dest)]
        with log.open('w')as stream:
            process=subprocess.run(args,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
        item=dict(case_id=row['case_id'],status='completed'if process.returncode==0 else 'failed',returncode=process.returncode,stdout=resource(log))
        if process.returncode==0:
            item.update(prediction=resource(dest/'prediction.json'),execution=resource(dest/'result.json'))
        elif (dest/'failure.json').exists():item['failure']=resource(dest/'failure.json')
        return row,item
    with ThreadPoolExecutor(max_workers=workers)as executor:
        futures=[executor.submit(execute,row)for row in jobs]
        for future in as_completed(futures):
            row,item=future.result();target='counterfactual'if row.get('counterfactual')else'cases'
            state[target].append(item);state[target].sort(key=lambda x:x['case_id']);state['pending_case_ids'].remove(row['case_id'])
            if item['status']!='completed':state['failures'].append(item['case_id'])
            state['elapsed_s']=time.monotonic()-start;write(HERE/'inference_index.json',state)
            completed=len(state['cases'])
            if completed and completed%24==0:print(json.dumps(dict(primary_cases_complete=completed,total=192,failures=len(state['failures']))),flush=True)
    verify();state['status']='completed_isolated_banks'if not state['failures']else'completed_with_failures';state['elapsed_s']=time.monotonic()-start
    write(HERE/'inference_index.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--workers',type=int,default=3);args=p.parse_args()
    assert 1<=args.workers<=3
    if args.freeze:freeze()
    if args.run:run(args.workers)
