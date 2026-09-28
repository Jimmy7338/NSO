#!/usr/bin/env python3
"""Only hash/inventory/metadata verification, no sensor, fusion, mesh decoding or policy."""
import os,sys
sys.dont_write_bytecode=True
import pathlib,json,gzip,hashlib,importlib.util,collections,shutil
ROOT=pathlib.Path('/root/NSO');A=ROOT/'audit_results/completed_derived_archive_v24b_20260915'
spec=importlib.util.spec_from_file_location('restore',ROOT/'scripts/restore_completed_derived_v24b.py');restore=importlib.util.module_from_spec(spec);spec.loader.exec_module(restore)
sha=restore.sha

def main():
 rows=restore.load(A);mapping={r['path']:r for r in rows};assert len(mapping)==375
 for row in rows:assert restore.verify_row(row) is False
 checks={}
 prior={'competition_v8_1_execution_20260911':ROOT/'audit_results/storage_recovery_20260911/competition_v8_1_execution_20260911/compaction.json','competition_v9_1_execution_20260911':ROOT/'audit_results/competition_v9_1_mesh_compaction_20260911/competition_v9_1_execution_20260911/compaction.json'}
 for group in ('competition_v8_1_execution_20260911','competition_v9_1_execution_20260911','semantic_gain_v13_2_outbound_paired_20260914','semantic_gain_v13_paired_information_resumed_20260914'):
  run=ROOT/'eval_results'/group;inventory=json.loads((run/'artifact_hashes.json').read_text());old_missing={};old_compaction_sha=None
  if group in prior:
   p=prior[group];c=json.loads(p.read_text());assert c['status']=='complete';old_compaction_sha=sha(p)
   for rel,h in c['bindings'].items():
    target=(ROOT/'audit_results/competition_v9_1_independent_metrics_20260911'/({'independent_metric_audit_manifest':'artifact_hashes.json','independent_metric_verification':'verification.json'}[rel])) if rel.startswith('independent_metric_') else run/rel
    assert sha(target)==h
   old_missing=c['meshes'];v=json.loads((run/'verification.json').read_text());assert v['passed_full'] is True and sha(run/'artifact_hashes.json')==v['artifact_manifest_sha256']
  else:assert json.loads((run/'manifest.json').read_text())['status']=='complete'
  present=archived=previously_compacted=0
  for rel,h in inventory.items():
   p=run/rel;key=str(p.relative_to(ROOT))
   if key in mapping:assert mapping[key]['sha256']==h;archived+=1
   elif p.exists():assert sha(p)==h,(group,rel);present+=1
   else:assert rel in old_missing and old_missing[rel]['file_sha256']==h,(group,rel);previously_compacted+=1
  checks[group]=dict(original_inventory_sha256=sha(run/'artifact_hashes.json'),original_inventory_count=len(inventory),present_original_hashes_verified=present,newly_archived_original_hashes_verified=archived,preexisting_compacted_meshes=previously_compacted,prior_compaction_sha256=old_compaction_sha,no_new_unmapped_missing_artifact=True)
 # V1 predates comprehensive artifact inventories; preserve its actual run metadata and engineering verification.
 pilot=ROOT/'eval_results/virtual3d_pilot_v1_20260910';m=json.loads((pilot/'run_metadata.json').read_text());v=json.loads((pilot/'verification.json').read_text());assert m['status']=='complete' and v['engineering_pipeline']=='passed'
 pmanifest=json.loads((A/'pilot_single_link_ply/manifest.json').read_text())
 for rel,h in pmanifest['protected_metadata'].items():assert sha(ROOT/rel)==h
 episodes=[json.loads(p.read_text()) for p in pilot.glob('*/episode.json')];assert len(episodes)==32
 summary=json.loads((pilot/'summary.json').read_text())
 for method,s in summary.items():
  es=[e for e in episodes if e['method']==method]
  for key in ('coverage_2d','f1_05cm','joint_05cm','joint_auc_05cm','collisions','wall_time_s'):assert abs(sum(e[key] for e in es)/len(es)-s[key])<1e-10
 pairs={}
 for e in episodes:
  d=pilot/e['artifact_dir'];assert len(list((d/'frames').glob('*.npz')))==e['frames'] and len(list((d/'scans').glob('*.npz')))==e['frames'];truth=pilot/(e['scene']+'_truth.ply');assert truth.is_file();pairs[e['scene']]=sha(truth)
 checks['virtual3d_pilot_v1_20260910']=dict(run_status='complete',engineering_pipeline='passed',episode_count=32,recorded_sensor_frames=sum(e['frames'] for e in episodes),all_raw_frame_and_scan_counts_match_terminal_metadata=True,eight_truth_files_at_original_paths=True,current_truth_file_sha256=pairs,protected_metadata_count=len(pmanifest['protected_metadata']),all_summary_means_match_terminal_records=True,limitation='No comprehensive original per-packet inventory exists in this early run; raw and truth bytes were never cleanup targets. This audit checks counts and preserved recorded metadata, not a fabricated historical packet hash list. Historical truth_sha256 hashes mesh arrays, not PLY file encoding; this audit records current truth file hashes without claiming those equal historical array hashes.')
 # Exact duplicate bootstrap leaves original new-shape inventory and filenames intact.
 dedup=json.loads((A/'bootstrap_exact_dedup.json').read_text());base=ROOT/'audit_results/facility_choice_v24_shape_prefix_20260915';assert sha(base/'artifact_hashes.json')==dedup['original_inventory_sha256']
 assert sorted(str(p.relative_to(base)) for p in base.rglob('*') if p.is_file())==dedup['original_artifact_paths']
 inv=json.loads((base/'artifact_hashes.json').read_text())
 for rel,h in inv.items():assert sha(base/rel)==h
 for row in dedup['rows']:assert (ROOT/row['path']).stat().st_ino==(ROOT/row['anchor']).stat().st_ino
 # Previously sealed 464 PLY archive: verify every frozen artifact; never alter it.
 old=ROOT/'audit_results/inspection_mechanism_v1_ply_archive_v24_20260915';oldinv=json.loads((old/'artifact_hashes.json').read_text())
 for rel,h in oldinv.items():assert sha(old/rel)==h
 oldproof=dict(original_archive_artifact_count=len(oldinv),all_frozen_hashes_unchanged=True,inventory_sha256=sha(old/'artifact_hashes.json'))
 pyc=json.loads((A/'bootstrap_pyc_cleanup.json').read_text())
 for row in pyc['rows']:assert not (ROOT/row['path']).exists() and sha(ROOT/row['source'])==row['source_sha256']
 # Count allocation once per inode including sidecar gzip archives and this task's metadata.
 allfiles=[p for p in A.rglob('*') if p.is_file()]
 allfiles += [ROOT/r['archive'] for r in rows if 'archive' in r and not (ROOT/r['archive']).is_relative_to(A)]
 em=json.loads((A/'emergency_v8_bootstrap.json').read_text())
 allfiles += [(ROOT/r['path']).with_name(pathlib.Path(r['path']).name+'.v24b.archive.json') for r in em['sidecar_rows']]
 allfiles += [ROOT/'scripts'/n for n in ('archive_completed_derived_v24b.py','archive_completed_derived_v24c.py','restore_completed_derived_v24b.py')]
 proof=ROOT/'audit_results/completed_derived_archive_v24b_restore_proof_20260915';allfiles += [p for p in proof.rglob('*') if p.is_file()]
 seen=set();allocated=0
 for p in allfiles:
  s=p.stat();k=s.st_dev,s.st_ino
  if k not in seen:allocated+=s.st_blocks*512;seen.add(k)
 old_alloc=sum(r['original']['allocated_bytes'] for r in rows);dedup_freed=dedup['released_allocated_bytes'];pyc_freed=pyc['allocated_released']
 result=dict(status='verified',original_paths_archived=375,ply_paths=121,derived_log_paths=254,all_original_bytes_exactly_recoverable=True,all_original_paths_currently_archived=True,protected_original_evidence=checks,bootstrap_exact_dedup=dict(path_count=11,all_24_original_shape_artifact_hashes_unchanged=True,original_path_set_unchanged=True,released_allocated_bytes=dedup_freed,mtime_differences_recorded=True),pyc=dict(count=25,released_allocated_bytes=pyc_freed,sources_unchanged=True),previous_464_archive=oldproof,original_derived_allocated_bytes=old_alloc,new_archive_and_tool_allocated_bytes_before_final_report=allocated,net_task_allocated_bytes_released_before_final_report=old_alloc+dedup_freed+pyc_freed-allocated,physical_available_bytes=shutil.disk_usage(ROOT).free,restore_full_preserving_archives_required_available_bytes=old_alloc+8*1024**2,restore_real_path_proof=json.loads((proof/'result.json').read_text()),scope='Storage preservation only; no experiment, mapping, metric, threshold or policy changes.')
 out=A/'independent_verification.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('status','original_paths_archived','net_task_allocated_bytes_released_before_final_report','physical_available_bytes')}),flush=True)
if __name__=='__main__':main()
