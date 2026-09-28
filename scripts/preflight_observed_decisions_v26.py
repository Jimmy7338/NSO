#!/usr/bin/env python3
"""V26 interface-only decisions from one saved, GT-designed paid history.

Only public world_config/shape fields are taken from case_00/result.json; this
file IS opened, but its evaluation results, assignment and route are not inputs.
The original packets 0..234 are fused once into one offline mapper. This DOES
rebuild TSDF from existing packets. No new world, physical action/sensor packet,
quality evaluation or closed-loop trajectory is performed. G receives geometry
only; S receives the same geometry plus cumulative actually visible RGB cues.
The past trajectory is scripted: these decisions cannot establish policy gains.
"""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['NUMEXPR_NUM_THREADS']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse
from dataclasses import asdict,is_dataclass
import hashlib
import io
import json
from pathlib import Path
import shutil
import signal
import sys
from time import perf_counter
import traceback
from types import SimpleNamespace
import zipfile
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
SOURCE=ROOT/'audit_results/facility_choice_v25r1_paid_p00_20260915'
CASE=SOURCE/'case_00'
OUT=ROOT/'audit_results/observed_decisions_v26_20260916'
CHECKPOINTS=(0,24,111,234)
MODES=('G','S')
MAX_CANDIDATES=12
MAX_PATCHES=256
CAP=2*1024*1024
RESERVE=64*1024*1024
FAILURE_RESERVE=32*1024
SECONDS=600
CONFIG_FIELDS=('resolution_m','robot_radius_m','camera_height_m','width_px','height_px',
 'fov_deg','max_depth_m','depth_sigma_m','dropout','action_duration_s','voxel_m',
 'laser_height_m','laser_rays','width_m','height_m','truncation_m','pose_noise_m',
 'stereo_model','stereo_reference_fx_px','stereo_baseline_m')
COUNTERS={'new_world_instances':0,'forbidden_world_constructor_calls':0,
 'forbidden_step_calls':0,'forbidden_sense_calls':0,'forbidden_scan_calls':0,
 'new_physical_tasks':0,'new_physical_actions':0,'new_sensor_packets':0,
 'new_quality_evaluations':0,'saved_packets_loaded':0,'offline_mapper_updates':0,
 'offline_tsdf_integrations_from_saved_packets':0,'semantic_memory_updates':0,
 'geometry_snapshots':0,'decision_calls':0}

class CheckFailure(ValueError):pass
class LimitStop(BaseException):pass

def require(condition,message):
 if not condition:raise CheckFailure(message)
def interrupt(signum,frame):raise LimitStop('signal '+str(signum))
def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def jsonable(value):
 if is_dataclass(value):return jsonable(asdict(value))
 if isinstance(value,dict):return {str(k):jsonable(v) for k,v in value.items()}
 if isinstance(value,(tuple,list)):return [jsonable(x) for x in value]
 if hasattr(value,'tolist'):return jsonable(value.tolist())
 if isinstance(value,(str,int,float,bool)) or value is None:return value
 raise TypeError('Unserializable evidence: '+str(type(value)))
def digest(value):return hashlib.sha256(json.dumps(jsonable(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def capacity(amount=0,emergency=False):
 ancestor=OUT if OUT.exists() else OUT.parent
 used=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) if OUT.exists() else 0
 require(shutil.disk_usage(ancestor).free-amount-4096>=RESERVE,'64 MiB free reserve')
 require(used+amount+4096<=(CAP if emergency else CAP-FAILURE_RESERVE),'2 MiB output hard limit')
def write(name,value,emergency=False):
 blob=(json.dumps(jsonable(value),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode()
 capacity(len(blob),emergency);path=OUT/name;tmp=OUT/(name+'.tmp')
 with tmp.open('xb') as stream:stream.write(blob);stream.flush();os.fsync(stream.fileno())
 os.replace(tmp,path)
def verify(mapping):
 for name,expected in mapping.items():require(sha(ROOT/name)==expected,'Source/input hash mismatch: '+name)

def public_inputs():
 """Whitelisted config/provenance, never evaluation data or route objects."""
 source_inventory=read(SOURCE/'artifact_hashes.json');case_inventory=read(CASE/'artifact_hashes.json')
 require(source_inventory.get('case_00/artifact_hashes.json')==sha(CASE/'artifact_hashes.json'),'Case inventory is not sealed by the original root inventory')
 inputs={}
 def checked(path):
  relative=str(path.relative_to(SOURCE));case_relative=str(path.relative_to(CASE)) if path.is_relative_to(CASE) else None
  expected=case_inventory.get(case_relative) if case_relative else source_inventory.get(relative)
  require(expected is not None and sha(path)==expected,'Sealed input mismatch: '+relative)
  if relative in source_inventory:require(source_inventory[relative]==expected,'Root/case inventory disagree')
  inputs[str(path.relative_to(ROOT))]=expected
  return read(path)
 # Do not pass any other root/case fields to mapper, adapter or planner.
 manifest=checked(SOURCE/'manifest.json')
 require(manifest['status']=='complete','Original batch incomplete')
 budget=manifest['config']['total_budget'];require(type(budget) is int and budget==400,'Expected public 400-action budget')
 frozen_source_sha=dict(manifest['source_sha256'])
 versions=manifest.get('versions',{})
 del manifest
 record=checked(CASE/'result.json')
 public_config={key:record['world_config'][key] for key in CONFIG_FIELDS}
 shape=tuple(record['shape']);del record
 require(shape==(50,80),'Expected declared public P00 task shape')
 require(public_config['resolution_m']==.2 and public_config['width_m']==16. and public_config['height_m']==10.,'Public map contract')
 replay=checked(CASE/'verification.json')
 require(replay['status']=='passed' and replay['independent_process'],'Original case independent replay missing')
 replay_summary={k:replay[k] for k in ('status','independent_process','physical_process_id','replay_process_id','actual_physical_replay_actions','saved_packets_verified')}
 del replay
 for action_id in range(CHECKPOINTS[-1]+1):
  name=f'packets/{action_id:04d}.npz';p=CASE/name
  require(name in case_inventory and sha(p)==case_inventory[name],'Saved packet SHA '+name)
  root_name=str(p.relative_to(SOURCE));require(source_inventory.get(root_name)==case_inventory[name],'Packet root inventory mismatch')
  inputs[str(p.relative_to(ROOT))]=case_inventory[name]
 for p in (SOURCE/'artifact_hashes.json',CASE/'artifact_hashes.json'):inputs[str(p.relative_to(ROOT))]=sha(p)
 verify(frozen_source_sha)
 return dict(inputs=inputs,public_config=public_config,shape=shape,total_budget=budget,
  frozen_source_sha=frozen_source_sha,recorded_versions=versions,original_replay=replay_summary,
  configuration_access='Opened case_00/result.json; used only public world_config whitelist and shape; all evaluation/assignment/route fields discarded',
  world_config_whitelist=list(CONFIG_FIELDS),route_catalog_fields_accessed_or_forwarded=False)


def source_snapshot():
 paths={Path(__file__).resolve()}
 for module in tuple(sys.modules.values()):
  filename=getattr(module,'__file__',None)
  if filename:
   path=Path(filename).resolve()
   if path.suffix=='.py' and path.is_relative_to(ROOT) and not any(part.startswith('.venv') for part in path.relative_to(ROOT).parts):paths.add(path)
 test=ROOT/'tests/virtual3d/test_observed_state_v26.py'
 if test.is_file():paths.add(test)
 return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}


def forbid_world_calls():
 """Tripwires for already imported project world classes; no source edits."""
 guarded=[];seen=set()
 def denied(kind):
  def reject(*args,**kwargs):
   COUNTERS[kind]+=1;raise CheckFailure('Physical world use forbidden: '+kind)
  return reject
 for module in tuple(sys.modules.values()):
  if not getattr(module,'__name__','').startswith('env.'):continue
  for value in tuple(vars(module).values()):
   if not isinstance(value,type) or id(value) in seen or not value.__module__.startswith('env.'):continue
   if 'World' not in value.__name__:continue
   seen.add(id(value));guarded.append(value.__module__+'.'+value.__name__)
   value.__init__=denied('forbidden_world_constructor_calls')
   for name in ('step','sense','scan'):
    if hasattr(value,name):setattr(value,name,denied('forbidden_'+name+'_calls'))
 return sorted(guarded)


def mapper_identity(mapper):
 """Cheap full observed-state checks; no mesh extraction or GT evaluation."""
 arrays={}
 for name in ('belief','visible','camera_seen'):
  value=getattr(mapper,name,None)
  if value is not None:arrays[name]=digest(value)
 return dict(frames=int(mapper.frames),map_arrays=arrays,
  quality_without_labels=digest([{k:v for k,v in row.items() if k!='label'} for row in mapper.quality.values()]),
  keyframe_count=len(mapper.keyframes))


def check_history(previous,packet,transform,config):
 packet.validate(transform,config)
 require(not packet.done and not packet.collision,'Checkpoint history contains terminal/collision packet')
 if previous is None:
  require(packet.action_id==0 and packet.action is None,'Frame zero required')
 else:
  require((packet.scene_id,packet.episode_id)==(previous.scene_id,previous.episode_id),'History identity changed')
  require(packet.action_id==previous.action_id+1 and packet.frame_id!=previous.frame_id,'Nonconsecutive packet')
  require(packet.frame.timestamp_s>previous.frame.timestamp_s,'Nonmonotonic timestamp')
  r,c=previous.position;h=previous.heading
  if packet.action=='forward':
   dr,dc=((-1,0),(0,1),(1,0),(0,-1))[h];r+=dr;c+=dc
  elif packet.action=='left':h=(h-1)%4
  elif packet.action=='right':h=(h+1)%4
  else:raise CheckFailure('Undeclared saved paid action')
  require((*packet.position,packet.heading)==(r,c,h),'Saved odometry/action mismatch')


def candidate_identity(candidate):
 """Actual pose/path/cost identity, independent of ranking and source labels."""
 keys=('pose','states','actions','outbound_states','outbound_actions','return_states',
       'return_actions','outbound_cost','return_cost','cost')
 return {key:jsonable(candidate[key]) for key in keys if key in candidate}


def check_candidate_sources(answer,geometry,cues,mode):
 cue_by_id={cue.cue_id:cue for cue in cues}
 patch_keys={tuple(p.key) for p in geometry.patches}
 counts={}
 for candidate in answer['candidates']:
  require(candidate.get('sources'),'Candidate must retain measured proposal provenance')
  for source in candidate['sources']:
   kind=source['kind'];counts[kind]=counts.get(kind,0)+1
   if kind in ('observed_frontier','observed_surface_deficit'):
    require(source['action_id']==geometry.action_id,'Geometry source not current snapshot')
    require(source['geometry_sha256']==geometry.geometry_sha256,'Candidate geometry source hash')
    if kind=='observed_surface_deficit':require(tuple(source['patch_key']) in patch_keys,'Unknown measured patch source')
   elif kind=='visible_class_direction_hypothesis':
    require(mode!='G','G received semantic candidate provenance')
    require(source['cue_id'] in cue_by_id,'Candidate cue not actually observed')
    cue=cue_by_id[source['cue_id']]
    require(source['cue_action_id']==cue.action_id<=geometry.action_id,'Future or mismatched semantic cue')
    require(source['class_id']==cue.class_id and source['cue_source']==cue.source,'Semantic source lineage mismatch')
   else:raise CheckFailure('Undeclared candidate source kind '+str(kind))
 audit=answer['audit']
 require(audit['geometry_sha256']==geometry.geometry_sha256,'Planner audit geometry mismatch')
 require(audit['candidate_limit']==MAX_CANDIDATES and audit['candidate_count']==len(answer['candidates']),'Planner cap/audit mismatch')
 require(audit['route_catalogue_used'] is False and audit['evaluation_truth_used'] is False,'Planner reports forbidden truth input')
 require(sum(str(c['group']).startswith('coverage_') for c in answer['candidates'])<=4,'Coverage candidate cap exceeded')
 require(sum(not str(c['group']).startswith('coverage_') for c in answer['candidates'])<=MAX_CANDIDATES-4,'Noncoverage candidate cap exceeded')
 for field in ('geometric_proposal_queries','semantic_proposal_queries','scored_unique_poses','planning_seconds'):
  require(field in audit and audit[field]>=0,'Missing prefilter/computation audit: '+field)
 if mode=='G':require(audit['semantic_proposal_queries']==audit['semantic_cue_count']==0,'G consumed semantic cues')
 return counts


def execute(progress):
 require((ROOT/'nso/observed_state_v26.py').is_file() and (ROOT/'nso/observed_planner_v26.py').is_file(),'V26 API not ready')
 public=public_inputs();progress['inputs']=public['inputs'];progress['old_sources']=public['frozen_source_sha']
 import numpy as np
 from nso.decision_replay_v13 import load_packet
 from nso.cpu_sensor_contract_v10 import GridTransform
 from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
 from nso.observed_state_v26 import geometry_state_v26,VisibleSemanticMemoryV26
 from nso.observed_planner_v26 import ObservedPlannerV26
 forbidden_modules=('utils.facility_outline_v23','utils.facility_metrics_v19')
 require(not any(name in sys.modules for name in forbidden_modules),'Evaluation module unexpectedly imported')
 guarded=forbid_world_calls()
 sources=source_snapshot();progress['new_sources']=sources
 capacity(CAP-128*1024);OUT.mkdir();progress['created_output']=True
 manifest=dict(status='running',source_sha256=sources,original_source_sha256=public['frozen_source_sha'],
  input_sha256=public['inputs'],public_configuration=public['public_config'],shape=public['shape'],total_budget=public['total_budget'],
  configuration_access=public['configuration_access'],world_config_whitelist=public['world_config_whitelist'],
  root_manifest_opened_for_public_budget_and_source_provenance=True,route_catalog_fields_accessed_or_forwarded=False,
  original_replay=public['original_replay'],checkpoints=list(CHECKPOINTS),modes=list(MODES),max_candidates=MAX_CANDIDATES,max_patches=MAX_PATCHES,
  max_seconds=SECONDS,output_cap_bytes=CAP,free_reserve_bytes=RESERVE,guarded_world_classes=guarded,
  evidence_scope='Existing GT-designed trajectory offline interface diagnostic; no autonomous policy execution or efficacy comparison',
  mapper_rebuilds_tsdf_from_saved_packets=True,no_claim_of_zero_TSDF_processing=True,
  label_RGB_invariance_scope='Dedicated observed_state_v26 adapter tests; this run does not repeat whole-history recoloring or relabeling',
  tested_adapter_source='tests/virtual3d/test_observed_state_v26.py',adapter_test_execution_not_inferred_from_file_presence=True,
  template_mode_run=False)
 write('manifest.json',manifest)
 archive=io.BytesIO()
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for name in sources:z.writestr(name,(ROOT/name).read_bytes())
 blob=archive.getvalue();capacity(len(blob))
 with (OUT/'sources.zip').open('xb') as stream:stream.write(blob);stream.flush();os.fsync(stream.fileno())
 config=SimpleNamespace(**public['public_config'])
 mapper=ObservedRuntimeMapperV10(public['shape'],config,truncation_m=config.truncation_m)
 memory=VisibleSemanticMemoryV26();transform=GridTransform(public['shape'],config.resolution_m)
 previous=None;anchor=None;identities=set();checkpoints=[];paid_history=[];mapping_seconds=0.;memory_seconds=0.
 for action_id in range(CHECKPOINTS[-1]+1):
  progress.update(phase='offline_saved_packet_mapping',action_id=action_id)
  packet=load_packet(CASE/f'packets/{action_id:04d}.npz');COUNTERS['saved_packets_loaded']+=1
  require(packet.action_id==action_id and packet.frame_id not in identities,'Wrong/duplicate saved packet')
  check_history(previous,packet,transform,config);identities.add(packet.frame_id)
  if anchor is None:anchor=(*packet.position,packet.heading)
  before=int(mapper.frames);started=perf_counter();mapper.update(packet.frame,packet.scan);mapping_seconds+=perf_counter()-started
  require(int(mapper.frames)==before+1==action_id+1,'Mapper must update once per saved packet')
  COUNTERS['offline_mapper_updates']+=1;COUNTERS['offline_tsdf_integrations_from_saved_packets']+=1
  started=perf_counter();cues=memory.update(packet);memory_seconds+=perf_counter()-started;COUNTERS['semantic_memory_updates']+=1
  paid_history.append(dict(action_id=action_id,frame_id=packet.frame_id,packet_sha256=public['inputs'][str((CASE/f'packets/{action_id:04d}.npz').relative_to(ROOT))],mapper_frames=mapper.frames))
  if action_id in CHECKPOINTS:
   progress['phase']='read_only_snapshot_decisions'
   identity_before=mapper_identity(mapper);started=perf_counter()
   geometry=geometry_state_v26(mapper,packet,anchor,public['total_budget']-action_id,max_patches=MAX_PATCHES)
   adapter_seconds=perf_counter()-started;COUNTERS['geometry_snapshots']+=1
   geometry_hash=digest(geometry);cue_hash=digest(cues)
   write(f'action_{action_id:04d}_geometry.json',dict(action_id=action_id,sha256=geometry_hash,geometry=geometry))
   rows=[];pool_maps={}
   for mode in MODES:
    planner=ObservedPlannerV26(mode=mode,max_candidates=MAX_CANDIDATES)
    started=perf_counter()
    # Deliberately omit the cues argument entirely for the geometry baseline.
    answer=planner.plan(geometry) if mode=='G' else planner.plan(geometry,cues=cues)
    seconds=perf_counter()-started;COUNTERS['decision_calls']+=1
    require(isinstance(answer,dict) and all(k in answer for k in ('candidates','selected','audit')),'Planner schema mismatch')
    require(len(answer['candidates'])<=MAX_CANDIDATES,'Unequal/expanded candidate budget')
    require(digest(geometry)==geometry_hash and digest(cues)==cue_hash,'Decision mutated shared observed inputs')
    source_counts=check_candidate_sources(answer,geometry,cues if mode!='G' else (),mode)
    candidates=sorted([candidate_identity(c) for c in answer['candidates']],key=lambda c:tuple(c['pose']))
    require(len({tuple(c['pose']) for c in candidates})==len(candidates),'Duplicate final candidate pose')
    pool_maps[mode]={tuple(c['pose']):c for c in candidates}
    pool_hash=digest(candidates)
    row=dict(mode=mode,action_id=action_id,geometry_sha256=geometry_hash,cues_argument_supplied=(mode!='G'),
     supplied_cue_count=0 if mode=='G' else len(cues),candidates_count=len(answer['candidates']),
     candidate_geometry_sha256=pool_hash,candidate_source_counts=source_counts,planning_seconds=seconds,fresh_planner_no_counterfactual_execution_feedback=True,
     result=answer)
    write(f'action_{action_id:04d}_{mode}.json',row);rows.append(row)
   poses_G=set(pool_maps['G']);poses_S=set(pool_maps['S']);common=poses_G&poses_S
   common_paths_equal=all(pool_maps['G'][pose]==pool_maps['S'][pose] for pose in common)
   require(common_paths_equal,'Same candidate pose has different geometric path/cost under G/S')
   comparison=dict(common_poses=sorted(common),G_only_poses=sorted(poses_G-poses_S),S_only_poses=sorted(poses_S-poses_G),
    pose_sets_equal=poses_G==poses_S,common_pose_paths_equal=common_paths_equal,
    equality_required=False,reason='V26 S may propose semantic directions and rerank; different pose sets are a measured mechanism, not an interface error')
   identity_after=mapper_identity(mapper);require(identity_before==identity_after,'Read-only adapter/planning altered mapper state')
   summary=dict(action_id=action_id,paid_historical_actions=action_id,remaining_budget=public['total_budget']-action_id,
    geometry_sha256=geometry_hash,cue_sha256=cue_hash,visible_cues=cues,
    adapter_seconds=adapter_seconds,geometry_input_same_for_G_and_S=True,candidate_pool_comparison=comparison,mapper_state_unchanged=True,
    mapper_identity=identity_after,decisions=[{k:v for k,v in row.items() if k!='result'}|dict(selected=row['result']['selected'],audit=row['result']['audit']) for row in rows])
   checkpoints.append(summary);progress['completed_checkpoint_actions']=[x['action_id'] for x in checkpoints]
   print(json.dumps(dict(progress='checkpoint_complete',action_id=action_id,cues=len(cues),candidates={r['mode']:r['candidates_count'] for r in rows},same_pose_set=comparison['pose_sets_equal'])),flush=True)
   write('partial_result.json',dict(status='partial',checkpoints=checkpoints,counters=COUNTERS))
  if action_id%50==0 or action_id==CHECKPOINTS[-1]:
   print(json.dumps(dict(progress='offline_saved_packets',action_id=action_id,frames=mapper.frames,decisions=COUNTERS['decision_calls'],mapping_seconds=mapping_seconds)),flush=True)
  previous=packet
 require(COUNTERS['saved_packets_loaded']==COUNTERS['offline_mapper_updates']==235,'Unexpected replay count')
 require(COUNTERS['decision_calls']==8 and COUNTERS['geometry_snapshots']==4,'Unexpected decision scope')
 require(all(COUNTERS[k]==0 for k in ('forbidden_world_constructor_calls','forbidden_step_calls','forbidden_sense_calls','forbidden_scan_calls')),'Forbidden physical call attempted')
 verify(public['inputs']);verify(public['frozen_source_sha']);verify(sources)
 result=dict(status='complete_offline_interface_check',checkpoints=checkpoints,counters=COUNTERS,
  all_shared_geometry_and_candidate_provenance_checks_passed=True,public_configuration_access=public['configuration_access'],
  all_inputs_and_sources_unchanged=True,original_paid_history_actions_used=234,new_paid_actions=0,
  mapping_seconds=mapping_seconds,visible_semantic_memory_seconds=memory_seconds,
  autonomous_trajectory_executed=False,semantic_effectiveness_evaluated=False,quality_evaluations=0,
  whole_history_label_RGB_invariance_replay_performed=False,
  adapter_invariance_requires_separate_test_receipt=True,
  warnings=['Past poses were chosen by a GT-designed fixed route, not this planner.',
    'G/S outputs share geometry at isolated checkpoints; no candidate is executed.',
    'Existing packets rebuild a TSDF offline; there is no new physical data or scene.',
    'Natural recognition, complete-system advantage and closed-loop safety are not demonstrated.'])
 write('paid_history_replay.json',dict(saved_packets=paid_history,public_anchor_from_packet0=anchor,counters=COUNTERS))
 write('result.json',result);manifest['status']='complete';write('manifest.json',manifest)
 print(json.dumps({'status':result['status'],'counters':COUNTERS,'output':str(OUT)}),flush=True)

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');args=parser.parse_args()
 if not args.run:parser.error('--run required after V26 APIs and tests are ready')
 require(not OUT.exists(),'Existing result/failure directory cannot be overwritten')
 if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
 started=perf_counter();progress={'created_output':False,'phase':'preflight'};code=0
 for s in (signal.SIGINT,signal.SIGTERM,signal.SIGALRM):signal.signal(s,interrupt)
 signal.setitimer(signal.ITIMER_REAL,SECONDS)
 try:execute(progress)
 except BaseException as error:
  signal.setitimer(signal.ITIMER_REAL,0);code=124 if isinstance(error,LimitStop) else 1
  failure=dict(status='bounded_incomplete' if code==124 else 'failed',error=repr(error),traceback=traceback.format_exc(),
   counters=COUNTERS,progress={k:v for k,v in progress.items() if k not in ('inputs','old_sources','new_sources')},
   elapsed_s=perf_counter()-started,process_id=os.getpid(),no_automatic_repeat=True)
  try:
   for key in ('inputs','old_sources','new_sources'):verify(progress.get(key,{}))
   failure['inputs_and_sources_unchanged']=True
  except BaseException as e:failure['inputs_and_sources_unchanged']=False;failure['hash_error']=repr(e)
  print(json.dumps(failure,ensure_ascii=False),file=sys.stderr,flush=True)
  if progress['created_output']:
   try:
    for tmp in OUT.glob('*.tmp'):tmp.rename(tmp.with_name('interrupted_'+tmp.name))
    write('failure.json',failure,True);m=read(OUT/'manifest.json');m['status']=failure['status'];write('manifest.json',m,True)
   except BaseException as e:print('Could not preserve receipt within reserve: '+repr(e),file=sys.stderr)
 finally:
  signal.setitimer(signal.ITIMER_REAL,0)
  if progress['created_output']:
   try:
    write('timing.json',dict(elapsed_s=perf_counter()-started,exit_code=code,process_id=os.getpid(),
     cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None),code!=0)
    inv={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='artifact_hashes.json'}
    write('artifact_hashes.json',inv,code!=0)
   except BaseException as e:print('Final sealing error: '+repr(e),file=sys.stderr);code=code or 1
 raise SystemExit(code)
if __name__=='__main__':main()
