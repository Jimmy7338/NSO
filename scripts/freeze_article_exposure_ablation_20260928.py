#!/usr/bin/env python3
"""Freeze only the final twelve declared common exposure ablations; no World.

Ground's complete terminal cohort must already exist. Failures remain paired;
neither an output directory nor a new version resets the cumulative cap of 24.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from env.development_sensor_v41 import runtime_counts_v41
from nso.article_experiment_ground_v2 import validate_protocol as validate_ground_protocol
from nso.article_experiment_exposure_v3 import (
    SCHEMA, runtime_metadata, source_hashes, validate_protocol, validate_pairing,
)

GROUND_PATH='configs/virtual3d/article_ground_ablation_v2_20260928.json'
GROUND_SHA='6621c39667e15a91f010d52c81f5a357049a447b6379246d07141953f631338a'
DECLARATION_PATH='docs/research/ARTICLE_EXPOSURE_ABLATION_PREDECLARATION_20260928.md'
DECLARATION_SHA='a008017abce88bcdc176f750c7df6ba87259c66375ac62255a7f4f9aeb295bc9'
PREDICTOR_PATH='nso/view_quality_article_exposure_v3.py'
PREDICTOR_SHA='0843b90ddf7c71a01a16cd629d355ffd272a653049a4d0cd32d17a75a86126cf'
TERMINAL=frozenset(('controller_stop','budget_exhausted','wall_time_limit',
    'controller_blocked','artifact_limit','episode_error','prediction_save_failed',
    'stopped_without_confirmed_return','attempt_failed'))


def read(path):return json.loads(Path(path).read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)


def contained(root,relative):
    path=Path(relative)
    require(bool(path.parts) and not path.is_absolute() and '..' not in path.parts,'contained relative path required')
    result=(root/path).resolve()
    require(result.is_relative_to(root.resolve()),'path escapes repository')
    return result


def verify_terminal_cohort(root,ground,ground_pin):
    output=contained(root,ground['output_relative_path'])
    ledger=read(output/'start_ledger.json')
    require(ledger['schema']=='article.start_ledger.v1' and ledger['phase']=='ablation'
        and ledger['protocol_sha256']==ground_pin and ledger['slots']==ground['slots'],
        'Ground terminal ledger/protocol binding differs')
    entries=ledger['entries']
    require(len(entries)==len(ground['slots']) and {r['run_id'] for r in entries}==set(ground['slots']),
        'all twelve Ground attempts must already be represented')
    require(all(r['status'] in TERMINAL for r in entries),'unfinished Ground attempt; no exposure freeze yet')
    registry=read(output.parent/'execution_registry.json')
    require(registry['schema']=='article.execution_registry.v1'
        and registry['phase_caps']=={'development':12,'main':96,'ablation':24},'shared global registry contract differs')
    records=[r for r in registry['entries'] if r['phase']=='ablation']
    require(len(records)==12,'last twelve ablation capacity must be wholly available')
    require({(r['ledger'],r['run_id'],r['protocol_sha256']) for r in records}
        =={(str((output/'start_ledger.json').resolve()),name,ground_pin) for name in ground['slots']},
        'global Ground reservations do not match terminal cohort')
    return dict(ledger=str((output/'start_ledger.json').relative_to(root)),
        ledger_sha256=sha(output/'start_ledger.json'),retained_statuses={r['run_id']:r['status'] for r in entries},
        registry_sha256=sha(output.parent/'execution_registry.json'),all_failures_retained=True)


def build_protocol(*,root=ROOT):
    root=Path(root).resolve();old_path=contained(root,GROUND_PATH)
    require(sha(old_path)==GROUND_SHA,'predeclared Ground baseline protocol changed')
    require(sha(contained(root,DECLARATION_PATH))==DECLARATION_SHA,'exposure predeclaration changed')
    require(sha(contained(root,PREDICTOR_PATH))==PREDICTOR_SHA,'independently reviewed exposure predictor changed')
    old=validate_ground_protocol(read(old_path))
    expected={(f'ART1_{f}_DEV',m,160,92801) for f in ('AISLE','CELL','LOOP') for m in ('G','B','S','NBV')}
    require(old['phase']=='ablation' and len(old['slots'])==12 and
        {(s['scene_id'],s['method'],s['budget'],s['noise_seed']) for s in old['slots'].values()}==expected,
        'exact three-development-layout, four-method baseline required')
    terminal=verify_terminal_cohort(root,old,GROUND_SHA)
    require(old['numerical_runtime']==runtime_metadata(),'same numerical build/path fingerprint required')
    old_archive=contained(root,old['source_archive'])
    require(sha(old_archive)==old['source_archive_sha256'],'original Ground source archive changed')
    with zipfile.ZipFile(old_archive) as stream:
        require(len(stream.namelist())==len(set(stream.namelist())) and set(stream.namelist())==set(old['source_sha256']),
            'exact original Ground archive closure required')
        for name,pin in old['source_sha256'].items():
            require(sha(contained(root,name))==pin and hashlib.sha256(stream.read(name)).hexdigest()==pin,
                'original Ground source changed: '+name)
    pins=source_hashes()
    for name,pin in old['source_sha256'].items():
        if name in pins:require(pins[name]==pin,'shared frozen source changed: '+name)
    require(sha(contained(root,old['asset_root'])/'manifest.json')==old['asset_manifest_sha256'],'same scene asset manifest required')
    for entry in old['references'].values():
        require(sha(contained(root,entry['root'])/'manifest.json')==entry['manifest_sha256'],'same frozen reference manifest required')
    result=deepcopy(old)
    for key in ('common_baseline_evaluation_plan','formal_followup'):
        result.pop(key,None)
    result.update(schema=SCHEMA,phase='ablation',
        output_relative_path='audit_results/article_stage_20260928/exposure_ablation_v3',
        source_sha256=pins,source_archive='audit_results/article_stage_20260928/execution_sources_exposure_v3.zip',
        paired_baseline_protocol=dict(path=GROUND_PATH,sha256=GROUND_SHA),
        scientific_predeclaration=dict(path=DECLARATION_PATH,sha256=DECLARATION_SHA),
        scope='Post-diagnosis common same-center paid-exposure ablation on all three development layouts; not held-out confirmation.',
        slots={'exposure_'+name.removeprefix('ground_'):dict(slot,paired_baseline_run_id=name)
            for name,slot in old['slots'].items()},
        predecessor_terminal_cohort=terminal,
        cumulative_attempt_rule='development12 + ground ablation12 + exposure ablation12; all failures consume original shared caps',
        generation=dict(script='scripts/freeze_article_exposure_ablation_20260928.py',
            script_sha256=sha(__file__),new_worlds=0,new_surface_evaluations=0))
    result.pop('source_archive_sha256',None)
    result['controller']['same_center_exposure_dedup']=True
    validate_protocol(result)
    validate_pairing(result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'configs/virtual3d/article_exposure_ablation_v3_20260928.json')
    args=parser.parse_args();path=args.output.resolve()
    require(path.is_relative_to(ROOT),'configuration must remain inside repository')
    before=runtime_counts_v41();protocol=build_protocol();archive=ROOT/protocol['source_archive']
    if path.exists() or archive.exists():raise FileExistsError('frozen configuration/archive cannot be replaced')
    pins=protocol['source_sha256']
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as stream:
        for name,pin in sorted(pins.items()):
            content=(ROOT/name).read_bytes();require(hashlib.sha256(content).hexdigest()==pin,'source changed before archive')
            info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o100644<<16;stream.writestr(info,content)
    protocol['source_archive_sha256']=sha(archive)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as output:json.dump(protocol,output,sort_keys=True,separators=(',',':'),allow_nan=False);output.write('\n')
    require(runtime_counts_v41()==before,'configuration freezing created a World')
    require(source_hashes()==pins,'sources changed during freezing')
    print(json.dumps(dict(protocol=str(path.relative_to(ROOT)),protocol_sha256=sha(path),
        source_files=len(pins),source_archive_sha256=sha(archive),slots=len(protocol['slots']),new_worlds=0)))


if __name__=='__main__':main()
