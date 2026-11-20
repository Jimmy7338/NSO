"""Compact evidence packaging; never uploads and never modifies frozen inputs."""
import argparse,csv,hashlib,json,shutil,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOTS={'v1':HERE.parent,'v2':HERE.parents[1]/'dual_template_instance_20261120'}
PAYLOAD=HERE/'payload'
PREFIX='experiments/offline_20261120'
ARCHIVES=[]
TRANSIENT_NAMES={'heartbeat.json','evaluation_progress.json','progress.json','draft_inventory.json'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def copy(p,target):target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);assert sha(p)==sha(target)
def archive(path,root,files):
    path.parent.mkdir(parents=True,exist_ok=True)
    members=[]
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9)as z:
        for p in sorted(files):
            rel=p.relative_to(root).as_posix();data=p.read_bytes()
            info=zipfile.ZipInfo(rel,(2026,11,20,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
            z.writestr(info,data,compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
            members.append(dict(path=rel,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    with zipfile.ZipFile(path)as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(members)
        for m in members:assert hashlib.sha256(z.read(m['path'])).hexdigest()==m['sha256']
    ARCHIVES.append(dict(path=path.relative_to(PAYLOAD).as_posix(),bytes=path.stat().st_size,sha256=sha(path),members=members))

def stage_csv(root,target):
    rows=[]
    for p in sorted((root/'audit').glob('shelf_*/result.json')):
        d=json.loads(p.read_text())
        for stage in d['stages']:
            actual=stage['actual_fixed_reference_increment']
            for mode,entry in stage['prediction_groups'].items():
                primary=entry['primary_predicted_point_match'];partition=primary['partition']
                row={k:stage[k]for k in ['stage_id','layout','state_action','variant','candidate_id']}
                row.update(model=mode,predicted_weighted_template_area_m2=primary['total_weighted_template_area_m2'],
                    supported_weight_m2=primary['source_supported_weighted_template_area_m2'],supported_fraction=primary['source_supported_weight_fraction'],
                    realized_supported_weight_m2=partition['true_previously_unmatched_now_matched']['weighted_template_area_m2'],
                    false_geometry_weight_m2=partition['false_geometry']['weighted_template_area_m2'],
                    actual_global_net_m2=actual['all']['net_correct_surface_area_estimate_m2'],
                    actual_object_net_m2=sum(actual[k]['net_correct_surface_area_estimate_m2']for k in ['open31','closed118']),
                    actual_object_gross_m2=sum(actual[k]['gross_new_correct_surface_area_estimate_m2']for k in ['open31','closed118']),
                    actual_object_lost_m2=sum(actual[k]['lost_correct_surface_area_estimate_m2']for k in ['open31','closed118']),
                    execution_status=stage['execution_receipt']['status'])
                rows.append(row)
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('w',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return len(rows)

def build(final=False):
    PAYLOAD.mkdir(exist_ok=True)
    state_counts={};csv_counts={}
    for name,root in ROOTS.items():
        for p in [root/'protocol.json',root/'run.py']:
            copy(p,PAYLOAD/name/p.relative_to(root))
        for directory in [root/'method',root/'audit',root/'replay']:
            for p in directory.glob('*'):
                if p.is_file()and p.suffix in {'.py','.json','.csv','.md','.log'}and p.name not in TRANSIENT_NAMES:
                    copy(p,PAYLOAD/name/p.relative_to(root))
        for p in (root/'audit').rglob('*.py'):
            if '__pycache__'not in p.parts:copy(p,PAYLOAD/name/p.relative_to(root))
        replay_files=[p for p in (root/'replay').rglob('*')if p.is_file()and '_attempt_' in p.as_posix()and p.suffix in {'.json','.npz','.log'}and '__pycache__'not in p.parts]
        assert all('distances'not in p.name and (p.suffix!='.npz' or p.name.endswith('_predictions.npz'))for p in replay_files)
        archive(PAYLOAD/'archives'/f'{name}_fixed_replay_predictions.zip',root,replay_files)
        evaluation_files=[p for p in (root/'audit').rglob('*')if p.is_file()and p.parent!=root/'audit'and p.suffix in {'.json','.csv','.log'}and '__pycache__'not in p.parts and 'sampling'not in p.relative_to(root/'audit').as_posix()and p.name not in TRANSIENT_NAMES]
        archive(PAYLOAD/'archives'/f'{name}_complete_stage_evaluations.zip',root,evaluation_files)
        csv_counts[name]=stage_csv(root,PAYLOAD/name/'audit/full_stage_results.csv')
        state_counts[name]=len(list((root/'replay').glob('*_attempt_001/result.json')))
    for name in ['derive_model_decomposition.py','model_decomposition.json','model_decomposition.csv','build_bundle.py']:
        copy(HERE/name,PAYLOAD/'tools'/name)
    rollouts=ROOTS['v1']/'rollouts/sampling'
    finished=list(rollouts.glob('layouts/*/runs/*/attempt_*/result.json'))
    # The complete runner writes summary.json rather than result.json on some versions.
    attempts=list(rollouts.glob('layouts/*/runs/*/attempt_*/attempt.json'))
    statuses=[]
    for p in attempts:
        statuses.append({'attempt_path':str(p.parent),'attempt':json.loads(p.read_text()),'result_files':[x.name for x in p.parent.glob('*.json')]})
    if final:
        final_summary=ROOTS['v1']/'audit/sampling_rollouts/summary.json'
        if not final_summary.exists():raise RuntimeError('Completed sampling evaluation required')
        evaluated=json.loads(final_summary.read_text())
        if not evaluated.get('status','').startswith('complete') or evaluated.get('pending'):raise RuntimeError('Sampling evaluator still incomplete')
        if len(attempts)!=6:raise RuntimeError(f'Expected six sampling attempts; got {len(attempts)}')
        for p in attempts:
            if not (p.parent/'result.json').exists():raise RuntimeError(f'Unfinished sampling result: {p.parent}')
        for layout in ['shelf_left','shelf_right']:
            files=[p for p in rollouts.rglob('*')if p.is_file()and p.suffix in {'.json','.jsonl','.csv','.log','.txt'}and p.name not in TRANSIENT_NAMES and (layout in p.parts or layout in p.name or p==rollouts/'protocol.json')]
            archive(PAYLOAD/'archives'/f'sampling_policy_{layout}.zip',ROOTS['v1'],files)
        audit=ROOTS['v1']/'audit'
        sampling_results=[p for p in audit.rglob('*')if p.is_file()and 'sampling' in p.relative_to(audit).as_posix()and p.suffix in {'.json','.csv','.log'}and p.name not in TRANSIENT_NAMES]
        if sampling_results:archive(PAYLOAD/'archives'/'sampling_policy_evaluation.zip',ROOTS['v1'],sampling_results)
    mesh_records=[]
    for p in sorted((ROOTS['v1']/'replay').rglob('*.ply')):
        mesh_records.append(dict(study='v1_fixed_replay',path=str(p),relative_path=p.relative_to(ROOTS['v1']).as_posix(),bytes=p.stat().st_size,sha256=sha(p),retained_local_only=True))
    if final:
        for p in sorted(rollouts.rglob('*.ply')):
            mesh_records.append(dict(study='sampling_policy',path=str(p),relative_path=p.relative_to(ROOTS['v1']).as_posix(),bytes=p.stat().st_size,sha256=sha(p),retained_local_only=True))
    write(PAYLOAD/'LOCAL_NATIVE_MESH_MANIFEST.json',{'no_mesh_files_published':True,'files':mesh_records})
    write(PAYLOAD/'ARCHIVE_INDEX.json',{'archives':ARCHIVES,'total_members':sum(len(a['members'])for a in ARCHIVES)})
    metadata=dict(status='complete'if final else 'draft_sampling_runs_pending',source_studies={k:str(v)for k,v in ROOTS.items()},fixed_prefixes=6,candidate_branches=66,new_v2_candidate_acquisitions=0,independent_floorplans=1,related_object_permutations=2,diagnostic_states=state_counts,full_csv_rows=csv_counts,completed_sampling_result_files=len(finished),declared_sampling_runs=6,all_revised_geometry_instance_variants_promoted=False,model_Z_calibrated=False,hardware_experiments=False)
    write(PAYLOAD/'EVIDENCE_SCOPE.json',metadata)
    readme='''# NSO fixed-observation model diagnostics and sampling control (20 November 2026)

This compact evidence package retains the negative and mixed outcomes of two explicitly versioned template-model diagnostics. Six fixed prefixes from two object permutations of one controlled floorplan share 66 acquired candidate branches. **v2 reuses those same 66 branch outcomes; it does not provide another 66 acquisitions.** These branches are development diagnostics, not 66 independent scenes or autonomous runs. Ideal visible labels and exact simulator poses remain declared conditions.

`v1` records full-basis integration, finite observed-envelope estimation and free-space pseudo-likelihood. `v2` records acquired-instance pooling and heuristic absolute free-consistency attenuation. Neither geometry nor pooled-instance revision was promoted to the complete-policy comparison on these mixed/negative diagnostics. The separate six-run policy matrix uses the full sampling basis to compare G, S and removal of measured-surface reward consumption while retaining verified-free rejection and B2 visibility. Its final results must be read from the included sampling evaluation, not inferred from the fixed-branch scores.

`unmodeled` mass is heuristic residual accounting, not a fitted or calibrated third class. Predicted quadrature weights are not independently measured physical surface area. Stage tables retain both unsupported and realized weights; evaluator increments retain gross gains and lost matches. Six related prefixes, repeated states and G/S evaluations do not create independent sample counts.

## Contents

- `v1/`, `v2/`: frozen model/protocol/replay/evaluation source, tests, independent reviews, full six-state summaries and aggregate CSVs.
- `v1/audit/full_stage_results.csv`, `v2/audit/full_stage_results.csv`: every candidate, stage and G/S row, including ties and negative increments.
- `archives/`: complete compact replay receipts, actual candidate executions, all predicted points/weights, full evaluation JSON and completed sampling-policy logs.
- `ARCHIVE_INDEX.json`: byte counts and SHA-256 of every archived member.
- `LOCAL_NATIVE_MESH_MANIFEST.json`: hashes of original reconstructed meshes retained locally, without publishing the meshes.
- `tools/model_decomposition.*`: attribution of predicted weight to acquired instance groups; this is prediction accounting, not additional accuracy evidence.

## Dependencies and replay boundary

The frozen scripts retain their original absolute-path receipts. Reproduction also requires the earlier November 19 source/protocol and B2 packages, compatible pinned Python/Open3D dependencies and separately obtained licensed scene assets. This archive is not a standalone one-command simulator. Asset truth is confined to the renderer/evaluator and never supplied to the models. Raw RGB-D streams, source assets, native reconstructed meshes, ground-truth distance arrays, credentials and `PROJECT_PRIORITIES.json` are intentionally excluded.

The manuscript and editable figures are published separately with the task's paper manifest. Physical robot experiments remain suspended. No robot command is contained in this packaging operation.

## Integrity

Use `PUBLICATION_INDEX.json` for payload hashes and `ARCHIVE_INDEX.json` for archived member hashes. Extract the archives into a separate review directory; original frozen input files should remain unchanged. Initial attempt receipts retain their original start-time status; final result and completed evaluator summary determine completion. Transient heartbeat/progress files are omitted, while genuine failure or correction records are retained. Every ZIP was reopened and each member's SHA-256 checked before packaging completion.
'''
    (PAYLOAD/'README.md').write_text(readme)
    if not final:
        write(HERE/'draft_inventory.json',metadata)
        return metadata
    index_files=[p for p in sorted(PAYLOAD.rglob('*'))if p.is_file()and p.name!='PUBLICATION_INDEX.json']
    write(PAYLOAD/'PUBLICATION_INDEX.json',{'schema':'nso.offline20261120.payload.v1','files':[dict(path=p.relative_to(PAYLOAD).as_posix(),bytes=p.stat().st_size,sha256=sha(p))for p in index_files]})
    files=[dict(local_path=str(p),repository_relative_path=p.relative_to(PAYLOAD).as_posix(),bytes=p.stat().st_size,sha256=sha(p))for p in sorted(PAYLOAD.rglob('*'))if p.is_file()]
    manifest=dict(schema='nso.github_payload_manifest.v1',payload_root=str(PAYLOAD),suggested_repository_prefix=PREFIX,files=files,file_count=len(files),bytes=sum(f['bytes']for f in files),remote_upload_performed=False,manuscript_pending_root_addition=True,archives=ARCHIVES)
    write(HERE/'content_manifest.json',manifest)
    return {k:v for k,v in manifest.items()if k not in ['files','archives']}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');args=parser.parse_args()
    print(json.dumps(build(args.final),indent=2))
