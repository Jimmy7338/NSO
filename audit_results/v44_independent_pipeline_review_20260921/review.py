"""Read only sealed files and NumPy arrays; no generator, backend or replay call."""
from collections import deque
import gzip,hashlib,json,math,re
from pathlib import Path
import numpy as np
ROOT=Path('/root/NSO');BASE=ROOT/'audit_results/v44_evidence_pipeline_20260921'
OUT=ROOT/'audit_results/v44_independent_pipeline_review_20260921'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(a):
 a=np.ascontiguousarray(a);return hashlib.sha256(f'{a.dtype.str}:{a.shape}:'.encode()+a.tobytes()).hexdigest()
def assert_inventory(base,records):
 for name,row in records.items():
  p=base/name;assert p.is_file() and not p.is_symlink();assert p.stat().st_size==row['bytes'];assert sha(p)==row['sha256']
r=read(BASE/'result.json');artifacts=read(BASE/'artifact_sha256.json')
assert_inventory(BASE,artifacts)
assert set(artifacts)=={str(p.relative_to(BASE)) for p in BASE.rglob('*') if p.is_file()}-{'artifact_sha256.json'}
for name,digest in r['source_sha256'].items():assert sha(ROOT/name)==sha(BASE/'source'/name)==digest
seals=[];combined={}
for row in r['old_seals']:
 p=ROOT/row['manifest'];assert sha(p)==row['sha256'];m=read(p)['files'];assert len(m)==row['entries_checked'];assert_inventory(ROOT,m)
 combined.update(m);seals.append(dict(manifest=row['manifest'],entries=len(m),sha256=sha(p)))
references=[]
for declaration in r['references']:
 asset=declaration['asset_id'];folder=BASE/'references'/asset;record=read(folder/'reference.json');surface=record['surface'];desc=record['coverage'];manifest=read(folder/'manifest.json')
 assert sha(folder/'manifest.json')==declaration['manifest_sha256'];assert_inventory(folder,manifest['files'])
 assert manifest['asset_id']==record['asset_id']==asset
 assert record['generator_source_sha256']==r['source_sha256']['nso/offline_evaluation_v44.py']
 assert record['surface_evaluator_sha256']==r['source_sha256']['nso/surface_evaluation_v40.py']
 for path,digest in record['input_sha256'].items():assert digest==sha(ROOT/path)==combined[path]['sha256']
 metadata=read(ROOT/f'audit_results/v40_p1_development_geometry_20260920/{asset}/evaluation_private/instances.json')
 workspace=read(ROOT/f'audit_results/v40_p1_development_geometry_20260920/{asset}/public_workspace.json')
 assert metadata['public_workspace']==workspace
 targets=sorted(x['instance_id'] for x in metadata['private_instances'])
 assert targets==record['target_instance_inventory']==surface['target_instances']
 with np.load(folder/'coverage_domain.npz',allow_pickle=False) as data:domain=data['domain'].copy()
 assert domain.dtype==np.bool_ and list(domain.shape)==desc['shape'];assert int(domain.sum())==desc['denominator_cells'];assert ah(domain)==desc['denominator_mask_sha256']
 assert math.isclose(domain.sum()*desc['resolution_m']**2,desc['denominator_area_m2'],abs_tol=1e-12)
 assert desc['resolution_m']==.1 and desc['robot_radius_m']==.2
 bounds=np.asarray(workspace['bounds_xy_m']);inner=np.asarray(workspace['room_inner_bounds_xy_m']);yx=np.argwhere(domain)
 centers=bounds[0]+(yx[:,::-1]+.5)*.1
 assert np.all((centers>inner[0]+.2)&(centers<inner[1]-.2))
 rectangles=[]
 for row in metadata['private_instances']:
  box=np.asarray(row['world_aabb_m']);rectangles.append([box[0,0]-.2,box[1,0]+.2,box[0,1]-.2,box[1,1]+.2])
 for box in metadata['background_boxes']:
  if box[5]>.05 and box[4]<=.9:rectangles.append([box[0]-.2,box[1]+.2,box[2]-.2,box[3]+.2])
 for x0,x1,y0,y1 in rectangles:assert not np.any((centers[:,0]>=x0)&(centers[:,0]<=x1)&(centers[:,1]>=y0)&(centers[:,1]<=y1))
 start=np.floor((np.asarray(workspace['start_position_world_m'])[:2]-bounds[0])/.1).astype(int)[::-1];start=tuple(start)
 assert domain[start];seen={start};todo=deque([start])
 while todo:
  y,x=todo.popleft()
  for dy,dx in [(0,1),(0,-1),(1,0),(-1,0)]:
   pos=(y+dy,x+dx)
   if 0<=pos[0]<domain.shape[0] and 0<=pos[1]<domain.shape[1] and domain[pos] and pos not in seen:seen.add(pos);todo.append(pos)
 assert len(seen)==int(domain.sum())
 with np.load(folder/'surface.npz',allow_pickle=False) as data:arrays={name:data[name].copy() for name in data.files}
 assert {name:ah(a) for name,a in arrays.items()}==surface['arrays']
 assert sorted(np.unique(arrays['point_instance_id']).tolist())==targets
 assert len(arrays['points'])==surface['observable_sample_count']==declaration['observable_samples']
 assert np.all(arrays['area_weights']>0) and math.isclose(float(arrays['area_weights'].sum()),surface['observable_area_m2_estimate'],abs_tol=1e-12)
 views=read(folder/'candidate_views.json');summaries=[]
 for view in views['candidate_views']:
  summary={k:view[k] for k in ['view_id','width','height','near_m','far_m']}
  summary.update(intrinsic=ah(np.asarray(view['intrinsic'],float)),world_from_camera=ah(np.asarray(view['world_from_camera'],float)));summaries.append(summary)
 assert summaries==surface['candidate_views'];assert len(summaries)==4*views['reachable_cells']
 content={k:surface[k] for k in ['arrays','candidate_views','sample_spacing_m','seed','full_target_sample_count','depth_convention']}
 assert hashlib.sha256(json.dumps(content,sort_keys=True,separators=(',',':')).encode()).hexdigest()==surface['fingerprint']
 assert surface['route_independent'] and not views['selection_uses_prediction_or_planner_trajectory']
 assert not desc['uses_detected_instances'] and not desc['uses_planner_trajectory'] and not record['prediction_selection_or_trajectory_used']
 assert declaration['empty_prediction_sanity']==dict(C_nav=0.,Q=0.,J_nav=0.,all_task_instances_in_denominator=len(targets),study_result=False)
 if asset=='DEV_A_00':assert surface['fingerprint']==read(ROOT/'audit_results/v40_p1_surface_integration_20260920/reference_manifest.json')['fingerprint']
 references.append(dict(asset_id=asset,target_instances=len(targets),target_ids=targets,coverage_cells=int(domain.sum()),coverage_area_m2=desc['denominator_area_m2'],observable_surface_samples=len(arrays['points']),observable_area_m2_estimate=float(arrays['area_weights'].sum()),candidate_views=len(summaries),reference_fingerprint=surface['fingerprint'],manifest_sha256=sha(folder/'manifest.json'),input_hashes_match_original_seals=True,domain_mask_interior_obstacle_exclusion_connectivity_checked=True,regenerated_reference=False))
episode=BASE/'finite_initial_packet_episode';em=read(episode/'artifact_manifest.json');assert_inventory(episode,em['files'])
for p,d in em['source_sha256'].items():assert sha(ROOT/p)==sha(episode/'source'/p)==d
started=read(episode/'started.json');endpoint=read(episode/'result.json');mapper=read(episode/'prediction/mapper.json');runtime=read(episode/'runtime.json')
assert started['finite_fixture'] and runtime['finite_fixture'] and not runtime['world_created']
assert endpoint['status']=='controller_stop' and endpoint['executed_paid_actions']==endpoint['submitted_paid_actions']==0
assert mapper['frames']==mapper['tsdf_integration_count']==endpoint['mapper_frames']==endpoint['mapper_tsdf_integrations']==1
assert endpoint['quality_and_coverage_scores'] is None
replay=read(BASE/'independent_saved_replay.json');stdout=read(BASE/'replay_stdout.txt');assert replay==stdout==r['fresh_process_saved_replay']
assert replay['source_manifest_sha256']==sha(episode/'artifact_manifest.json') and replay['status']=='verified' and replay['frames_verified']==1 and replay['prediction_verified']
assert replay['source_is_finite_fixture'] and replay['externally_pinned_manifest'] and not replay['eligible_study_episode']
assert replay['new_worlds']==replay['new_sensor_queries']==replay['physical_actions']==0
rejection=read(BASE/'fixture_evaluation_rejection.json');assert rejection['rejected'] and rejection['reason']=='finite fixture cannot be scored as a development episode'
pre=read(BASE/'actual_entrypoint_preflight.json');assert pre['status']=='blocked_before_world_creation' and not pre['start_slot_reserved'] and not pre['asset_read']
assert pre['resource']['required_free_bytes']==max(10*1024**3,2*pre['resource']['expected_batch_peak_bytes']+2*1024**3)
assert not any(pre['runtime'].values()) and not any(r['runtime_counts'].values())
ledger=ROOT/read(ROOT/'configs/virtual3d/v43_runtime_protocol_20260921.json')['ledger_relative_path'];assert not ledger.exists()
log=(BASE/'unittest.txt').read_text();count=int(re.search(r'Ran (\d+) tests',log).group(1));assert count==37 and log.rstrip().endswith('OK')
assert read(BASE/'test_result.json')['returncode']==0
payload=sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file());assert payload<=16*1024**2
result=dict(status='pass',scope='independent read-only SHA/array/source/count audit; no freeze, raycast, sensor, World, controller or TSDF execution',pipeline_root=str(BASE.relative_to(ROOT)),pipeline_result_sha256=sha(BASE/'result.json'),pipeline_inventory_sha256=sha(BASE/'artifact_sha256.json'),artifact_entries_verified=len(artifacts),current_and_archived_source_entries_verified=len(r['source_sha256']),old_seals=seals,old_seal_entries_verified=sum(x['entries'] for x in seals),references=references,recorded_tests_passed=count,finite_episode=dict(archive_source_entries_checked=len(em['source_sha256']),manifest_entries_checked=len(em['files']),saved_frames=1,recorded_main_tsdf_integrations=1,recorded_fresh_process_recomputation_tsdf_integrations=1,physical_actions=0,actual_development=False,fresh_process_stdout_matches_saved_receipt=True),fixture_development_evaluation_rejected=True,canonical_start_ledger_absent=True,persistent_preflight=pre['resource'],pipeline_total_bytes=payload,this_review_new_worlds=0,this_review_new_sensor_packets=0,this_review_new_tsdf_integrations=0,semantic_performance_claim_added=False,limitations=['Surface denominator is fixed sparse V40 observable-area quadrature, not all physical surfaces.','Coverage denominator is conservative navigable center-cell area quadrature, not exact free-floor area.','The singleton initial-grant fixture tests serialization/recomputation; no autonomous study trajectory has run.','Recorded unit test fixture integrations are distinct from the two main/recomputation integrations.'])
(OUT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['status','artifact_entries_verified','current_and_archived_source_entries_verified','old_seal_entries_verified','recorded_tests_passed','pipeline_total_bytes']},indent=2))
