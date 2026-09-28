#!/usr/bin/env python3
"""Generate the preselected 48-slot main protocol; never start a World or run.

Six unseen layouts x budgets 120/160 x noise realization 92811 x four methods.
The second realization 92812 is explicitly deferred. A frozen configuration
is not a scientific release: the root must review the declared gate before
calling a runner. No failed attempt can be refunded by this generator.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import zipfile
import hashlib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from nso.article_experiment_ground_v2 import (
    validate_protocol, source_hashes, runtime_metadata, EVALUATION,
)
from env.development_sensor_v41 import runtime_counts_v41

LAYOUTS=tuple('ART1_'+family+'_'+split for split in ('T0','T1') for family in ('AISLE','CELL','LOOP'))
METHODS=('G','B','S','NBV')
NOISE_SEED=92811


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)


def contained(root,relative):
    path=Path(relative)
    require(bool(path.parts) and not path.is_absolute() and '..' not in path.parts,'contained relative path required')
    result=(root/path).resolve()
    require(result.is_relative_to(root.resolve()),'path escapes repository')
    return result


def declared_slots():
    """Complete paired blocks, rotating methods; all six layouts in each pass."""
    slots={};schedule=[]
    for sweep in range(2):
        for index,scene in enumerate(LAYOUTS):
            budget=120 if (index+sweep)%2==0 else 160
            block=len(schedule)
            methods=METHODS[block%4:]+METHODS[:block%4]
            row=dict(block=block,scene_id=scene,budget=budget,noise_seed=NOISE_SEED,run_ids=[])
            for position,method in enumerate(methods):
                run_id=f'main48_{block:02d}_{position}_{scene[5:]}_{method}_b{budget}_n{NOISE_SEED}'
                slots[run_id]=dict(scene_id=scene,method=method,budget=budget,noise_seed=NOISE_SEED)
                row['run_ids'].append(run_id)
            schedule.append(row)
    return slots,schedule


def verify_reference(root,path,scene,asset_pin):
    record=read(path/'reference.json')
    require(record['scene_id']==scene and record['asset_manifest_sha256']==asset_pin,'common reference/scene binding')
    require(record['evaluation']==EVALUATION and record['route_independent'] is True,'frozen route-independent reference settings')
    require(record['reference_uses_semantic_weights'] is False,'unweighted common reference required')
    require(record['surface']['target_instances']==list(range(4)),'all-four-instance reference denominator required')
    manifest=read(path/'manifest.json')
    for name,row in manifest.items():
        target=contained(path,name)
        require(target.is_file() and target.stat().st_size==row['bytes'] and sha(target)==row['sha256'],
                'reference artifact differs: '+scene+'/'+name)
    return dict(root=str(path.relative_to(root)),manifest_sha256=sha(path/'manifest.json'))


def build_protocol(ground_protocol_path,*,root=ROOT,reference_root=None,
                   output_relative_path='audit_results/article_stage_20260928/main_ground_v2_48_n92811'):
    root=Path(root).resolve();path=Path(ground_protocol_path).resolve()
    require(path.is_relative_to(root),'ground protocol must be inside repository')
    ground=validate_protocol(read(path))
    require(ground['phase']=='ablation' and len(ground['slots'])==12,'frozen twelve-slot ground ablation predecessor required')
    require(ground['source_sha256']==source_hashes(),'ground execution sources changed')
    require(len(ground['source_sha256'])==62,'same sixty-two-source execution closure required')
    require(ground.get('numerical_runtime')==runtime_metadata(),'same complete numerical runtime fingerprint required')
    archive=contained(root,ground['source_archive'])
    require(sha(archive)==ground['source_archive_sha256'],'ground source archive changed')
    with zipfile.ZipFile(archive) as stream:
        require(len(stream.namelist())==len(set(stream.namelist())) and
                set(stream.namelist())==set(ground['source_sha256']),'exact ground archive closure required')
        for name,pin in ground['source_sha256'].items():
            require(hashlib.sha256(stream.read(name)).hexdigest()==pin,'archived source differs: '+name)
    asset=contained(root,ground['asset_root'])
    require(sha(asset/'manifest.json')==ground['asset_manifest_sha256'],'same asset manifest pin required')
    if reference_root is None:
        parents={str(Path(entry['root']).parent) for entry in ground['references'].values()}
        require(len(parents)==1,'one common existing reference parent required')
        references=contained(root,next(iter(parents)))
    else:
        references=contained(root,reference_root)
    slots,schedule=declared_slots()
    result=deepcopy(ground)
    for name in ('paired_baseline_protocol','formal_followup','common_baseline_evaluation_plan',
                 'scientific_predeclaration'):
        result.pop(name,None)
    result.update(phase='main',slots=slots,output_relative_path=output_relative_path,
        references={scene:verify_reference(root,references/scene,scene,ground['asset_manifest_sha256']) for scene in LAYOUTS},
        maximum_phase_bytes=48*ground['maximum_episode_bytes']+1024**2,
        scope='Conditional fixed 48-slot unseen-layout main matrix; no automatic second-noise expansion.',
        frozen_before_world_creation=True,
        predecessor_ground_protocol=dict(path=str(path.relative_to(root)),sha256=sha(path)),
        matrix=dict(independent_layouts=list(LAYOUTS),methods=list(METHODS),paid_budgets=[120,160],
            noise_realizations=[NOISE_SEED],declared_slots=48,second_noise_seed=92812,
            second_noise_slots=48,second_noise_explicitly_deferred=True,
            selection_basis='Time and storage before Ground outcomes; preserve budget contrast, six layouts and four methods.'),
        administrative_schedule=dict(ordered_complete_four_method_blocks=schedule,
            lexical_run_ids_encode_schedule=True,maximum_workers=2,partial_blocks_must_be_reported=True),
        scientific_release_gate=dict(required_before_any_world=True,
            technical_pipeline_and_common_metric_validation_required=True,
            actionable_shared_semantic_versus_fixed_bayes_mechanism_required=True,
            positive_endpoint_difference_required=False,automatic_release=False,
            decision_owner='root review before first main reservation',
            declaration_is_not_evidence_of_passage=True,
            instructions='Confirm the completed Ground causal study and technical review; retain all failures. Use a new main reviewer version.'),
        generation=dict(script='scripts/freeze_article_ground_main_20260928.py',script_sha256=sha(__file__),
            new_worlds=0,new_sensor_queries=0,new_policy_runs=0,new_surface_evaluations=0))
    validate_protocol(result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ground-protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='new protocol JSON inside configs/virtual3d')
    parser.add_argument('--reference-root',help='existing contained reference directory; inferred by default')
    parser.add_argument('--output-relative-path',default='audit_results/article_stage_20260928/main_ground_v2_48_n92811')
    args=parser.parse_args()
    target=args.output.resolve()
    require(target.parent==ROOT/'configs/virtual3d' and target.suffix=='.json','new configs/virtual3d JSON required')
    require(not target.exists(),'existing protocol cannot be overwritten')
    before=runtime_counts_v41()
    result=build_protocol(args.ground_protocol,reference_root=args.reference_root,
                          output_relative_path=args.output_relative_path)
    require(runtime_counts_v41()==before,'configuration generation must not create a World')
    with target.open('xb') as stream:
        stream.write((json.dumps(result,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode())
    print(json.dumps(dict(status='configuration_frozen_not_executed',path=str(target),sha256=sha(target),
        slots=len(result['slots']),source_files=len(result['source_sha256']),new_worlds=0,
        scientific_release_required=True,second_noise_automatically_authorized=False),ensure_ascii=False))


if __name__=='__main__':main()
