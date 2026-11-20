"""Compact B8 publication payload. Never changes frozen sources or uploads.

Draft builds include only completed evidence; final requires an explicit
finalization receipt after the policy matrix is complete. Final manuscript
source/PDF are published separately; only completed experimental tables enter.
"""
from pathlib import Path
import argparse,csv,hashlib,json,shutil,zipfile
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;PAYLOAD=HERE/'payload'
PREFIX='experiments/offline_20261120_uncertainty'
TRANSIENT={'heartbeat.json','progress.json','evaluation_progress.json','batch_execution.json','draft_inventory.json'}
FORBIDDEN_SUFFIX={'.ply','.obj','.glb','.gltf','.png','.jpg','.jpeg','.mp4','.bag','.svo','.db'}
ARCHIVES=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def copy(p,target):
 before=sha(p);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);assert sha(p)==sha(target)==before

def archive(name,files):
 target=PAYLOAD/'archives'/name;target.parent.mkdir(parents=True,exist_ok=True);members=[]
 with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=9)as z:
  for p in sorted(set(files)):
   assert p.suffix not in FORBIDDEN_SUFFIX and p.name not in TRANSIENT
   data=p.read_bytes();rel=p.relative_to(ROOT).as_posix();h=hashlib.sha256(data).hexdigest()
   info=zipfile.ZipInfo(rel,(2026,11,20,0,0,0));info.external_attr=0o100644<<16;info.compress_type=zipfile.ZIP_DEFLATED
   z.writestr(info,data,compress_type=zipfile.ZIP_DEFLATED,compresslevel=9);assert sha(p)==h
   members.append(dict(path=rel,bytes=len(data),sha256=h))
 with zipfile.ZipFile(target)as z:
  assert z.testzip()is None
  for row in members:assert hashlib.sha256(z.read(row['path'])).hexdigest()==row['sha256']
 ARCHIVES.append(dict(path=target.relative_to(PAYLOAD).as_posix(),bytes=target.stat().st_size,sha256=sha(target),members=members))

def verify_freezes():
 records={}
 for rel in ('generator/freeze.json','calibration/inference_freeze.json','replay/freeze.json','audit/validation.json'):
  path=ROOT/rel;value=json.loads(path.read_text());records[rel]=sha(path)
  for p,h in value['files_sha256'].items():
   source=Path(p)if Path(p).is_absolute()else path.parent/p
   assert sha(source)==h,source
 path=ROOT/'method/validation.json';value=json.loads(path.read_text());records['method/validation.json']=sha(path)
 for name,h in value['files'].items():assert sha(path.parent/name)==h,name
 return records

def completed_inference():
 root=ROOT/'calibration/inference_attempt_001';files=[];cases=0
 for folder in sorted(root.iterdir()):
  if not folder.is_dir():continue
  if (folder/'result.json').exists():
   result=json.loads((folder/'result.json').read_text());assert result['status']=='completed_observation_only_particle_bank'
   for key in ('prediction','observed_inputs','observation_manifest'):
    ref=result[key];assert sha(ref['path'])==ref['sha256']
   files += list(folder.glob('*.json'));cases+=1
  elif (folder/'failure.json').exists():files += list(folder.glob('*.json'))
 # Main 96 condition cases plus 2 identical-sensor/different-hidden-depth fixtures.
 assert cases==98,cases
 files += [p for p in root.glob('*.log')]
 archive('calibration_all_96_cases_and_2_counterfactual_banks.zip',files)
 manifests=list((ROOT/'observations').glob('*/manifest.json'))+list((ROOT/'counterfactual').glob('*/manifest.json'))
 assert len(manifests)==98
 archive('sanitized_sensor_manifests_only.zip',manifests)
 # Small procedural truth specs are authorized reproducibility inputs for the
 # independent generator/evaluator. Their actor-access prohibition is explicit.
 truth=list((ROOT/'evaluator_private').glob('*.json'));assert len(truth)==26
 archive('procedural_geometry_truth_evaluator_only.zip',truth)
 return cases

def fixed_replay():
 refs={};count=0;states=0;local=[]
 for folder in sorted((ROOT/'replay').glob('shelf_*_attempt_*')):
  if not folder.is_dir():continue
  if not (folder/'result.json').exists():
   if (folder/'failure.json').exists():archive(folder.name+'_failed_attempt.zip',list(folder.glob('*.json')))
   continue
  result=json.loads((folder/'result.json').read_text());assert result['status']=='completed_five_arm_fixed_prefix_reused_branches'
  assert result['newly_executed_candidate_actions']==0 and result['native_mesh_unchanged_after_prediction']
  manifest=json.loads((folder/'manifest.json').read_text());assert sha(folder/'manifest.json')==result['manifest']['sha256']
  assert len(manifest['stages'])==5*result['candidates'];count+=len(manifest['stages']);states+=1
  files=[]
  for p in folder.iterdir():
   if p.suffix=='.json' or p.name.endswith('_predictions.npz'):files.append(p)
   elif p.suffix=='.npz':local.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),published=False))
  archive(folder.name+'_five_arm_predictions.zip',files)
  for row in manifest['stages']:
   for role in ('before_mesh','after_mesh','execution_receipt'):
    ref=row[role];assert sha(ref['path'])==ref['sha256']
    refs[ref['path']]=dict(role=role,**ref,published_again=False,
     availability='prior_public_compact_receipt'if role=='execution_receipt'else'local_native_mesh_only_hash_published')
 assert states==6 and count==330
 write(PAYLOAD/'REUSED_66_BRANCH_REFERENCES.json',dict(files=list(refs.values()),reused_branches=66,
  dependent_candidate_arm_rows=count,new_candidate_actions=0,earlier_repository_prefix='experiments/offline_20261120',
  note='Earlier public packages contain compact execution receipts and mesh hashes, not native/source meshes. Regeneration requires available licensed assets and the frozen simulator dependencies.'))
 write(PAYLOAD/'LOCAL_OBSERVED_SUPPORT_MANIFEST.json',dict(files=local,published=False))
 return states,count

def completed_replay_evaluations(final):
 root=ROOT/'audit/particle_replay';files=[];count=0
 if root.exists():
  for p in root.rglob('result.json'):
   row=json.loads(p.read_text())
   if row.get('status')=='completed_five_arm_particle_fixed_branch_evaluation':files.append(p);count+=len(row['stages'])
  if count==330:
   for name in ('summary.json','results.csv','readiness.json','readiness_assessment.json'):
    if (root/name).exists():copy(root/name,PAYLOAD/'audit/particle_replay'/name)
  for p in root.rglob('failure.json'):files.append(p)
 for p in files:archive('fixed_replay_'+p.parent.name+'_'+p.stem+'.zip',[p])
 if final:assert count==330,'All five arms and six states must be independently evaluated before final package'
 return count

def full_stage_csv(final):
 rows=[]
 for p in sorted((ROOT/'audit/particle_replay').rglob('result.json')):
  value=json.loads(p.read_text())
  if value.get('status')!='completed_five_arm_particle_fixed_branch_evaluation':continue
  for stage in value['stages']:
   actual=stage['actual_fixed_reference_increment']
   for name,pred in stage['prediction_groups'].items():
    main=pred['primary_predicted_point_match'];part=main['partition']
    row={k:stage[k]for k in ('stage_id','layout','state_action','variant','candidate_id')}
    row.update(group=name,predicted_weight_m2=main['total_weighted_template_area_m2'],
     supported_weight_m2=main['source_supported_weighted_template_area_m2'],support_fraction=main['source_supported_weight_fraction'],
     realized_supported_weight_m2=part['true_previously_unmatched_now_matched']['weighted_template_area_m2'],
     actual_object_gross_m2=sum(actual[k]['gross_new_correct_surface_area_estimate_m2']for k in ('open31','closed118')),
     actual_object_lost_m2=sum(actual[k]['lost_correct_surface_area_estimate_m2']for k in ('open31','closed118')),
     actual_object_net_m2=sum(actual[k]['net_correct_surface_area_estimate_m2']for k in ('open31','closed118')),
     execution_status=stage['execution_receipt']['status'])
    attributed=stage['observed_object_attribution'][name]['current']['all_observed_object_tracks']
    for obj in ('open31','closed118'):
     row[obj+'_supported_weight_m2']=attributed[obj]['source_supported_weighted_template_area_m2']
     row[obj+'_realized_weight_m2']=attributed[obj]['partition']['true_previously_unmatched_now_matched']['weighted_template_area_m2']
    rows.append(row)
 if rows:
  target=PAYLOAD/'audit/particle_replay/full_330_stage_results.csv';target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('w',newline='')as f:
   writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 if final:assert len(rows)==330
 return len(rows)

def policy_results(closure):
 files=[];local=[];completed=[]
 root=ROOT/'rollouts'
 if not root.exists():return dict(status='not_started',completed_runs=0)
 summary=json.loads((ROOT/'audit/policy_rollouts/summary.json').read_text())
 assert summary['status']=='completed_independent_ten_run_evaluation'
 assert not summary['pending'] and not summary['failures']
 assert sum(len(row)for row in summary['rows'].values())==10
 freeze=ROOT/'audit/policy_rollouts/freeze.json'
 for path,h in json.loads(freeze.read_text())['files_sha256'].items():assert sha(path)==h,path
 for path in root.glob('joint_depth/layouts/*/prepared/freeze.json'):
  for source,h in json.loads(path.read_text())['code_sha256'].items():assert sha(source)==h,source
 for path in root.glob('joint_depth/layouts/*/runs/*/attempt_*/artifact_manifest.json'):
  record=json.loads(path.read_text());assert record['status']=='completed_execution'
  for name,h in record['files_sha256'].items():assert sha(path.parent/name)==h,path.parent/name
  completed.append(str(path.parent))
 assert len(completed)==10
 # Only finalization can include policy outputs, after required summaries and
 # expected run count have been explicitly checked by the parent.
 for p in root.rglob('*'):
  if not p.is_file()or '__pycache__'in p.parts or p.name in TRANSIENT:continue
  if p.suffix=='.py':copy(p,PAYLOAD/p.relative_to(ROOT))
  elif p.suffix in ('.json','.jsonl','.csv','.log','.txt','.md'):files.append(p)
  elif p.suffix in ('.npz','.ply'):local.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),published=False))
 # Keep each complete run in a small independently verifiable archive.
 for layout in ('shelf_left','shelf_right'):
  selected=[p for p in files if layout in p.parts]
  for arm in ('G_fixed','S_fixed','G_uncertainty','S_uncertainty','S_mean_collapse'):
   run=[p for p in selected if 'runs'in p.parts and arm in p.parts]
   assert run
   archive('policy_'+layout+'_'+arm+'_compact_records.zip',run)
  setup=[p for p in selected if 'runs'not in p.parts]
  if setup:archive('policy_'+layout+'_setup_and_selfcheck.zip',setup)
 other=[p for p in files if not {'shelf_left','shelf_right'}&set(p.parts)]
 if other:archive('policy_protocols_and_other_records.zip',other)
 write(PAYLOAD/'LOCAL_POLICY_GEOMETRY_MANIFEST.json',dict(files=local,published=False))
 for rel in closure.get('policy_evidence_directories',[]):
  directory=ROOT/rel;assert directory.is_dir()
  for p in directory.rglob('*.py'):copy(p,PAYLOAD/p.relative_to(ROOT))
  selected=[p for p in directory.rglob('*')if p.is_file()and p.suffix in('.json','.csv','.log','.md')and p.name not in TRANSIENT]
  if selected:archive('policy_independent_'+directory.name+'.zip',selected)
 return closure['policy_outcome']

def completed_tables():
 # These receipts were declared final by the paper agent. Their original
 # generation-status strings remain historical provenance, not package status.
 receipts=[]
 for name in ('table_receipt.json','policy_table_receipt.json'):
  path=ROOT/'paper_update'/name;row=json.loads(path.read_text())
  for ref in row.get('scientific_sources',row.get('sources',[])):
   assert sha(ref['path'])==ref['sha256'],ref['path']
  for ref in [row['build_script']]+row['outputs']:
   p=Path(ref['path']);assert sha(p)==ref['sha256'],p
   copy(p,PAYLOAD/p.relative_to(ROOT))
  copy(path,PAYLOAD/path.relative_to(ROOT));receipts.append(dict(path=name,sha256=sha(path)))
 write(PAYLOAD/'paper_update/TABLES_SCOPE.json',dict(status='completed_tables_from_frozen_experimental_results',receipts=receipts,
  manuscript_source_and_PDF_published_separately=True,temporary_drafts_included=False))

def build(final=False):
 ARCHIVES.clear();freezes=verify_freezes();closure=dict(status='draft_policy_and_manuscript_pending')
 if final:
  closure=json.loads((HERE/'finalization.json').read_text());assert closure['status']=='ready_for_package_finalization'
  for path,h in closure['required_completed_json_sha256'].items():assert sha(path)==h;json.loads(Path(path).read_text())
 assert PAYLOAD==HERE/'payload'
 if PAYLOAD.exists():shutil.rmtree(PAYLOAD)
 PAYLOAD.mkdir()
 for rel in ('DESIGN.md','ARCHITECTURE_DELTA.json','policy_design.json'):copy(ROOT/rel,PAYLOAD/rel)
 for folder in ('method','generator','calibration','replay','audit'):
  for p in (ROOT/folder).glob('*'):
   if p.is_file()and p.suffix in('.py','.json','.csv','.md','.log')and p.name not in TRANSIENT and p.name!='DESIGN_DRAFT.md':copy(p,PAYLOAD/p.relative_to(ROOT))
 # These calibration stages completed before any old-prefix scores were read.
 for rel in ('audit/calibration_attempt_001','audit/evaluation_attempt_001'):
  directory=ROOT/rel;archive(directory.name+'_full_results.zip',list(directory.glob('*.json')))
 cases=completed_inference();states,rows=fixed_replay();evaluated=completed_replay_evaluations(final);csv_rows=full_stage_csv(final)
 for p in (ROOT/'figure').glob('*'):
  if p.is_file()and p.suffix in('.py','.json','.csv','.md','.pdf','.svg'):copy(p,PAYLOAD/'figure'/p.name)
 if final:
  completed_tables()
  policy=policy_results(closure)
 else:policy=dict(status='not_included_until_complete_finalization')
 local_raw=[]
 for folder in ('observations','counterfactual'):
  for p in (ROOT/folder).rglob('*.npz'):local_raw.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),published=False))
 write(PAYLOAD/'LOCAL_SENSOR_ARRAY_MANIFEST.json',dict(files=local_raw,raw_RGBD_published=False,
  note='Sensor manifests and procedural generator/truth specifications are published; raw RGB-D/owner arrays remain local.'))
 write(PAYLOAD/'ARCHIVE_INDEX.json',dict(archives=ARCHIVES,total_members=sum(len(a['members'])for a in ARCHIVES)))
 write(PAYLOAD/'EVIDENCE_SCOPE.json',dict(status='complete'if final else'draft_policy_and_manuscript_pending',hardware=False,
  calibration_object_units=24,train=12,validation=6,heldout=6,conditions_per_object=4,main_calibration_cases=96,
  counterfactual_sensor_twins=2,completed_inference_outputs=cases,sigma_m=.01,nominal_interval_calibration_established=False,
  fixed_states=states,reused_candidate_branches=66,candidate_arm_rows=rows,independently_evaluated_rows=evaluated,
  full_stage_csv_rows=csv_rows,new_candidate_actions=0,policy=policy,closure=closure))
 copy(Path(__file__),PAYLOAD/'tools/build_bundle.py')
 (PAYLOAD/'README.md').write_text('''# B8 conditional dimension uncertainty — complete development evidence

B8 replaces an unresolved fixed depth with a class-neutral depth/placement
particle bank. G/S share acquired instance support, particle geometry and
surface/free-space losses; only their structural class prior differs. Marginal
depth beliefs can consequently differ. Width, orientation, height and parts
remain conditional B6 AB estimates. This is not a full shape posterior or a
new navigation algorithm. DESIGN.md, ARCHITECTURE_DELTA.json and method/ state
all changes; earlier methods and their negative findings remain unchanged.

Independent procedural calibration contains 24 geometry units split 12/6/6,
with four observation conditions per unit (96 dependent condition cases), plus
two identical-sensor counterfactual fixtures. The shared sigma=0.01 m was chosen
by training-only CRPS and frozen. Every case, energy table, fallback, training
score, validation and held-out result is retained. The declared informative-view
CRPS progression criterion passed; 95% intervals covered only 9/12 validation
and 8/12 held-out informative cases. Nominal probability calibration is therefore
not established. Validation/held-out outcomes did not trigger retuning.

The small evaluator_private JSON archive contains the independently generated
procedural truth/part specifications needed to reproduce calibration. It is
explicitly evaluator/generator-only input. Isolated inference opened only
sanitized sensor manifests and acquired measurements; open-file receipts and
counterfactual checks preserve that boundary. Procedural geometry generation
code is included; third-party furniture assets are not included.

Five arms (G_fixed, S_fixed, G_uncertainty, S_uncertainty, S_mean_collapse) are
scored on six previously inspected prefixes, reusing exactly 66 old acquired
candidate outcomes. The 330 arm/candidate rows are dependent measurements, not
330 acquisitions or independent environments. All matrices, object/particle
IDs, input hashes and independent full evaluations are retained. Mean-collapse
uses the S posterior's per-structure means and masses without refitting. Fixed
controls retain the same position integration and calibrated loss, and must not
be relabeled as historical O. All ten new policy runs and independent evaluations
are complete: both furniture permutations attained full correct-free coverage
with zero simulated collisions. In the left layout, extent updating first reached
full coverage at 88 instead of 97 actions and raised final global F1 by 0.3773
percentage points, while shelf recall decreased and cabinet recall increased.
The right layout tied exactly. G/S trajectories and native meshes tied within
both matched representations, and S mean-collapse reproduced full uncertainty.
These results do not establish semantic planning benefit or superiority of
planning with the full distribution. Timing records preserve the substantial
particle evaluation cost and shared-compute caveat. There is one floorplan
family, two inspected furniture permutations and no new random repetitions.
Frozen readiness receipts retain their historical pre-run status; final run
status and claim boundaries are in EVIDENCE_SCOPE.json and independent results.

Raw RGB-D/owner arrays, native/source meshes and large TSDF/GT distance arrays
remain local with hash manifests. Earlier public evidence packages contain
compact branch execution receipts and hashes, not the native or licensed source
meshes. Regenerating the old exploration evaluation requires those available
licensed assets, previous frozen source packages and the pinned numerical
runtime. This payload is not a standalone simulator. Reused branch references
do not imply that unpublished meshes are downloadable from GitHub.

Editable scientific SVG/PDF, plotting source and CSV inputs are included; figures
are computed evidence, not generated experimental imagery. All failures, ties
and negative effects are retained. This builder performs no robot motion, no
new fitting and no remote upload.
''')
 assert verify_freezes()==freezes
 for p in PAYLOAD.rglob('*'):
  if p.is_file():assert p.suffix not in FORBIDDEN_SUFFIX,p
 if not final:
  summary=dict(files=sum(p.is_file()for p in PAYLOAD.rglob('*')),bytes=sum(p.stat().st_size for p in PAYLOAD.rglob('*')if p.is_file()),evaluated_rows=evaluated,finalized=False)
  write(HERE/'draft_inventory.json',summary);return summary
 write(PAYLOAD/'PUBLICATION_INDEX.json',dict(files=[dict(path=p.relative_to(PAYLOAD).as_posix(),bytes=p.stat().st_size,sha256=sha(p))for p in sorted(PAYLOAD.rglob('*'))if p.is_file()and p.name!='PUBLICATION_INDEX.json']))
 files=[dict(local_path=str(p),repository_relative_path=p.relative_to(PAYLOAD).as_posix(),bytes=p.stat().st_size,sha256=sha(p))for p in sorted(PAYLOAD.rglob('*'))if p.is_file()]
 manifest=dict(schema='nso.github_payload_manifest.v1',payload_root=str(PAYLOAD),suggested_repository_prefix=PREFIX,
  files=files,file_count=len(files),bytes=sum(f['bytes']for f in files),archives=ARCHIVES,remote_upload_performed=False)
 write(HERE/'content_manifest.json',manifest);return {k:v for k,v in manifest.items()if k not in('files','archives')}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');args=p.parse_args();print(json.dumps(build(args.final),indent=2))
