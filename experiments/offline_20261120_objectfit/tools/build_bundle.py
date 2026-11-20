"""Compact B5/O8, B6 and B7 scientific package; no upload or source mutations."""
from pathlib import Path
import argparse,csv,hashlib,json,shutil,zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PARTS=ROOT.parent/'dual_template_parts_20261120'
DEPTH=ROOT.parent/'dual_template_depth_20261120'
PAYLOAD=HERE/'payload'
PREFIX='experiments/offline_20261120_objectfit'
TRANSIENT={'heartbeat.json','evaluation_progress.json','progress.json','draft_inventory.json','batch_execution.json'}
ARCHIVES=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,data):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def copy(source,target):target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);assert sha(source)==sha(target)
def archive(name,files,source_root=ROOT):
    path=PAYLOAD/'archives'/name;path.parent.mkdir(parents=True,exist_ok=True);members=[]
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(set(files)):
            data=p.read_bytes();rel=p.relative_to(source_root).as_posix()
            info=zipfile.ZipInfo(rel,(2026,11,20,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
            z.writestr(info,data,compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
            members.append(dict(path=rel,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        for m in members:assert hashlib.sha256(z.read(m['path'])).hexdigest()==m['sha256']
    ARCHIVES.append(dict(path=path.relative_to(PAYLOAD).as_posix(),source_study=source_root.name,bytes=path.stat().st_size,sha256=sha(path),members=members))

def full_csv(study_root,target):
    rows=[]
    for p in sorted((study_root/'audit').glob('shelf_*/result.json')):
        for stage in json.loads(p.read_text())['stages']:
            actual=stage['actual_fixed_reference_increment']
            for mode,pred in stage['prediction_groups'].items():
                main=pred['primary_predicted_point_match'];parts=main['partition']
                row={k:stage[k] for k in ('stage_id','layout','state_action','variant','candidate_id')}
                row.update(model=mode,predicted_weight_m2=main['total_weighted_template_area_m2'],
                    global_source_supported_weight_m2=main['source_supported_weighted_template_area_m2'],
                    global_support_fraction=main['source_supported_weight_fraction'],
                    global_realized_supported_weight_m2=parts['true_previously_unmatched_now_matched']['weighted_template_area_m2'],
                    actual_object_gross_m2=sum(actual[k]['gross_new_correct_surface_area_estimate_m2'] for k in ('open31','closed118')),
                    actual_object_lost_m2=sum(actual[k]['lost_correct_surface_area_estimate_m2'] for k in ('open31','closed118')),
                    actual_object_net_m2=sum(actual[k]['net_correct_surface_area_estimate_m2'] for k in ('open31','closed118')),
                    actual_global_net_m2=actual['all']['net_correct_surface_area_estimate_m2'],
                    execution_status=stage['execution_receipt']['status'])
                attribution=stage['observed_object_attribution'][mode]['current']['all_observed_object_tracks']
                for object_name in ('open31','closed118'):
                    item=attribution[object_name]
                    row[object_name+'_supported_weight_m2']=item['source_supported_weighted_template_area_m2']
                    row[object_name+'_realized_weight_m2']=item['partition']['true_previously_unmatched_now_matched']['weighted_template_area_m2']
                rows.append(row)
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return len(rows)

def frozen_method(study):
    frozen=json.loads((study/'method/validation.json').read_text())
    for name,expected in frozen['files'].items():assert sha(study/'method'/name)==expected,(study,name)
    return dict(validation_sha256=sha(study/'method/validation.json'),files=frozen['files'])

def copy_sources(study,target):
    for name in ('protocol.json','run.py'):
        if (study/name).exists():copy(study/name,target/name)
    for folder in ('method','replay','audit'):
        for source in (study/folder).glob('*'):
            if source.is_file() and source.suffix in ('.py','.json','.csv','.md','.log') and source.name not in TRANSIENT:
                copy(source,target/source.relative_to(study))
    # Nested independent evaluators are part of the scientific code, not only
    # top-level scripts. Their results are also archived below.
    for source in (study/'audit').rglob('*.py'):
        if '__pycache__' not in source.parts:copy(source,target/source.relative_to(study))

def fixed_study(study,target,archive_prefix,refs,support):
    summary=json.loads((study/'audit/summary.json').read_text())
    assert summary['completed_states']==6 and not summary['pending']
    copy_sources(study,target)
    predictions=[]
    for folder in sorted((study/'replay').glob('shelf_*_attempt_*')):
        if not folder.is_dir():continue
        for p in folder.iterdir():
            if p.suffix=='.json' or p.name.endswith('_predictions.npz'):predictions.append(p)
            if p.suffix=='.npz' and not p.name.endswith('_predictions.npz'):
                support.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),published=False))
        for stage in json.loads((folder/'manifest.json').read_text())['stages']:
            for key in ('before_mesh','after_mesh','execution_receipt'):
                item=stage[key];path=Path(item['path']);assert sha(path)==item['sha256']
                refs[item['path']]=dict(role=key,path=item['path'],sha256=item['sha256'],bytes=path.stat().st_size,
                    source_study='dual_template_geometry_20261120',reused=True,published_again=False,
                    public_availability='prior_receipt_archive' if key=='execution_receipt' else 'local_only_hash_published')
    archive(archive_prefix+'three_stage_predictions.zip',predictions,study)
    archive(archive_prefix+'three_stage_evaluations.zip',list((study/'audit').glob('shelf_*/result.json')),study)
    return full_csv(study,target/'audit/full_stage_results.csv')

def policy_results():
    root=ROOT/'rollouts/object_support'
    summary=json.loads((ROOT/'audit/object_support_rollouts/summary.json').read_text())
    assert summary['status']=='completed_independent_eight_run_evaluation' and not summary['pending'] and not summary['failures']
    assert sum(len(arms) for arms in summary['rows'].values())==8
    for arms in summary['rows'].values():
        for row in arms.values():
            assert row['free_correct_fraction']==1.0 and row['collisions']==0
    for source in root.rglob('*.py'):
        if '__pycache__' not in source.parts:copy(source,PAYLOAD/source.relative_to(ROOT))
    # Split by layout for manageable archive sizes, while retaining every
    # candidate score, action outcome, audit, failure and receipt in each run.
    metadata=[p for p in root.iterdir() if p.is_file() and p.suffix in ('.json','.log','.txt','.md') and p.name not in TRANSIENT]
    archive('O8_policy_protocol_and_failures.zip',metadata)
    for layout in ('shelf_left','shelf_right'):
        files=[p for p in (root/'layouts'/layout).rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.csv','.log','.txt','.md') and p.name not in TRANSIENT]
        archive('O8_policy_'+layout+'_compact_results.zip',files)
    audit=ROOT/'audit/object_support_rollouts'
    files=[p for p in audit.rglob('*') if p.is_file() and p.suffix in ('.json','.csv','.log') and p.name not in TRANSIENT]
    archive('O8_policy_independent_evaluation.zip',files)
    # Expose short final decisions and first divergence directly as well.
    for name in ('summary.json','results.csv','left_first_divergence.json','claim_evidence_assessment.json','evaluation_batch_receipt.json'):
        copy(audit/name,PAYLOAD/'audit/object_support_rollouts'/name)
    records=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),published=False) for p in root.rglob('*') if p.is_file() and p.suffix in ('.ply','.npz')]
    write(PAYLOAD/'LOCAL_POLICY_GEOMETRY_MANIFEST.json',dict(files=records,note='Native meshes, voxel evidence and belief arrays retained locally; compact decision/execution records are published.'))
    return dict(completed_runs=8,methods=['G','S','S_no_fit','S_no_measured_consumption'],related_layouts=2,
        independent_floorplans=1,semantic_efficiency_superiority_supported=False,full_correct_free_coverage_all_runs=True,
        shared_initial_coverage_fraction=0.4260614934114202,stochastic_repeats=0)

def build(final=False):
    ARCHIVES.clear()
    closure={'status':'draft_not_final'}
    if final:
        closure=json.loads((HERE/'finalization.json').read_text())
        assert closure['status']=='ready_for_package_finalization'
        for path in closure['required_completed_json']:
            p=Path(path);assert p.exists();json.loads(p.read_text())
    freezes={study.name:frozen_method(study) for study in (ROOT,PARTS,DEPTH)}
    # This path contains only this builder's generated payload. Rebuild so stale
    # draft/progress files cannot survive into the completed package.
    assert PAYLOAD==HERE/'payload' and PAYLOAD.parent==HERE
    if PAYLOAD.exists():shutil.rmtree(PAYLOAD)
    PAYLOAD.mkdir()
    refs={};support=[]
    rows=fixed_study(ROOT,PAYLOAD,'B5_',refs,support)
    parts_rows=fixed_study(PARTS,PAYLOAD/'parts','B6_',refs,support)
    observed_receipts=[]
    for folder in sorted((ROOT/'replay/observed_prefixes').glob('shelf_*')):
        if not folder.is_dir():continue
        for p in folder.iterdir():
            if p.suffix=='.json':observed_receipts.append(p)
            elif p.suffix=='.npz':support.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),published=False))
    archive('observed_prefix_receipts.zip',observed_receipts)
    write(PAYLOAD/'REUSED_BRANCH_REFERENCES.json',dict(reused_candidate_branches=66,new_candidate_acquisitions=0,
        B5_and_B6_share_the_same_66_branches=True,prior_repository_prefix='experiments/offline_20261120',
        note='Earlier public package contains compact execution receipts and hashes. Native/source meshes are retained locally and are NOT included in that public package.',files=list(refs.values())))
    write(PAYLOAD/'LOCAL_OBSERVED_SUPPORT_MANIFEST.json',dict(files=support,
        note='Full observed-support arrays retained locally; figure includes only its small selected-track plot input.'))
    for source in (HERE/'figure').iterdir():
        if source.is_file() and source.suffix in ('.py','.json','.md','.pdf','.svg'):
            copy(source,PAYLOAD/'figure'/source.name)
    for name in ('build_bundle.py','derive_fit_diagnosis.py','fit_failure_diagnosis.json'):
        copy(HERE/name,PAYLOAD/'tools'/name)
    # B7 is a pure diagnostic using B6-AB geometry. Preserve the complete curves,
    # immutable-input parity checks and independent audit, without new actions.
    depth_summary=json.loads((DEPTH/'audit/summary.json').read_text())
    assert depth_summary['completed_states']==6 and depth_summary['object_states']==12
    assert depth_summary['model_updates_applied']==0 and depth_summary['new_candidate_actions']==0
    assert depth_summary['all_frozen_inputs_intact'] and depth_summary['all_B6_models_and_acquired_mapper_unchanged']
    copy_sources(DEPTH,PAYLOAD/'depth')
    profile_files=[p for p in (DEPTH/'replay').glob('shelf_*_attempt_*/*.json')]
    archive('B7_complete_depth_profiles.zip',profile_files,DEPTH)
    policy=policy_results() if final else dict(status='awaiting_finalization')
    write(PAYLOAD/'FROZEN_METHODS.json',freezes)
    write(PAYLOAD/'ARCHIVE_INDEX.json',dict(archives=ARCHIVES,total_members=sum(len(x['members']) for x in ARCHIVES)))
    scope=dict(status='complete' if final else 'draft_not_final',fixed_states=6,reused_candidate_branches=66,
        new_fixed_branch_acquisitions=0,B5_stages=3,B6_stages=3,models_per_stage=2,
        B5_full_stage_csv_rows=rows,B6_full_stage_csv_rows=parts_rows,B7_profiled_states=6,B7_object_states=12,
        B7_new_actions=0,B7_model_updates=depth_summary['model_updates_applied'],
        independent_floorplans=1,related_layouts=2,hardware=False,GT_to_actor=False,
        G_uses_shared_observed_instance_support=True,policy=policy,closure=closure)
    write(PAYLOAD/'EVIDENCE_SCOPE.json',scope)
    (PAYLOAD/'README.md').write_text('''# Observed-object geometry, parts and depth diagnostics (B5–B7, 20 November 2026)

This compact package preserves the completed method-development evidence,
including null comparisons, regressions and corrected/invalid diagnostics.
B5 comprises O (acquired object tracks, inherited shape), O_full (all acquired
track points, inherited shape), and OF (partial outside-plane fit). Unknown-track
fragments retain generic measured-geometry and coverage rewards but do not
generate complete furniture hypotheses. G and S share acquired ideal object
support; G uses a neutral class prior and is not a semantics-free baseline.

The parts/ directory contains B6: qualified outside-plane orientation (A),
absolute observed/retained layer heights (B), and their combination (AB).
The depth/ directory contains B7: fixed-ray, class-neutral depth-profile
identifiability diagnostics. B7 applies no parameter changes, rescoring or
policy execution. Four eligible profiles favor smaller depths, but their
predeclared minimum-plus-one heuristic bands retain the original 0.65 m prior;
eight other object states lack a credible single-side anchor. This does not
show that the observations contain no depth information.

Six fixed prefixes from two related furniture permutations of one controlled
floorplan reuse exactly the same previous 66 candidate outcomes in B5 and B6.
These stages acquired no new candidate branches. The earlier public
experiments/offline_20261120 package contains compact execution receipts and
hashes; native reconstruction meshes and source meshes are NOT included there.
Native meshes remain locally retained. Regeneration requires the earlier
source/protocol/B2 packages, available licensed assets and the pinned
Python/Open3D environment. REUSED_BRANCH_REFERENCES.json states these boundaries.
Repeated stage/G/S evaluations are dependent measurements, not new trials.

Read audit/summary.json, parts/audit/summary.json and depth/audit/summary.json
for every fixed state. Each three-stage study includes all 396 candidate-model
rows in audit/full_stage_results.csv, complete predictions and independent
evaluator outputs in archives/. ARCHIVE_INDEX.json records every member hash.
Source truth is confined to evaluation, never shape fitting or actor input.

Eight O-policy runs use the declared G/S/two-ablation matrix in the two related
layouts, with no stochastic repeats. All start from the same acquired prelude
(42.606% initial coverage), rather than from an empty map. Their complete
compact decision/execution logs, initial self-check
failure and correction, independent evaluation, and first left-layout G/S
divergence are preserved. All eight complete correct-free coverage; this is
closed-loop feasibility in the controlled simulator. S does not improve left
layout coverage efficiency and ties G on the right. OF and B6 were not promoted
to policy runs; B7 retains the previous parameters. These results do not establish
semantic efficiency superiority, unseen-surface accuracy, or real-robot effects.

The figure was preselected at left/right prefix 24 and retains both tracked
objects, including failed placement. Points are acquired simulated measurements;
wireframes are predicted hypothesis envelopes, not complete reconstructions.
SVG/PDF, editable plot source, caption and small plot input are included.

Full observed-support arrays and native meshes remain local with hashes. Raw
RGB-D, licensed source assets, GT distance arrays, credentials and project-private
priority files are excluded. This package is not a standalone simulator and
cannot reproduce source-dependent evaluation without those dependencies.
No robot motion or upload occurs here.
''')
    # Verify frozen methods again after packaging; no hidden method changes.
    for study in (ROOT,PARTS,DEPTH):assert freezes[study.name]==frozen_method(study)
    if not final:
        write(HERE/'draft_inventory.json',scope);return scope
    write(PAYLOAD/'PUBLICATION_INDEX.json',dict(files=[dict(path=p.relative_to(PAYLOAD).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(PAYLOAD.rglob('*')) if p.is_file() and p.name!='PUBLICATION_INDEX.json']))
    files=[dict(local_path=str(p),repository_relative_path=p.relative_to(PAYLOAD).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(PAYLOAD.rglob('*')) if p.is_file()]
    manifest=dict(schema='nso.github_payload_manifest.v1',payload_root=str(PAYLOAD),suggested_repository_prefix=PREFIX,
        files=files,file_count=len(files),bytes=sum(x['bytes'] for x in files),archives=ARCHIVES,remote_upload_performed=False)
    write(HERE/'content_manifest.json',manifest)
    return {k:v for k,v in manifest.items() if k not in ('files','archives')}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');args=parser.parse_args()
    print(json.dumps(build(args.final),indent=2))
