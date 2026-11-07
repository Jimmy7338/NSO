#!/usr/bin/env python3
"""Run cached-semantic 41-frame physical history through the original offline S engine."""
import argparse,copy,hashlib,importlib.metadata,json,math,os,pathlib,sys,time,traceback
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
P=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(P/'code'))
from nso_bridge.joint_engine import JointSession
p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=pathlib.Path);p.add_argument('--prefix-audit-module',required=True,type=pathlib.Path,help='Read-only audit immediately after observe action 20');p.add_argument('--self-test',action='store_true',help='Run negative comparison tests only');p.add_argument('--post-audit-module',type=pathlib.Path,help='Optional separate frozen-state audit called only after exact replay matches');p.add_argument('--atol',type=float,default=1e-8);p.add_argument('--rtol',type=float,default=1e-8);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
rows=[json.loads(x) for x in (P/'decisions.jsonl').read_text().splitlines()];manifest=json.loads((P/'frames_manifest.json').read_text());frames={x['action_id']:x for x in manifest['frames']};assert len(frames)==41 and set(frames)==set(range(41));
for f in frames.values():
 assert hashlib.sha256((P/f['file']).read_bytes()).hexdigest()==f['sha256']
config=json.loads((P/'source/joint_configuration.json').read_text());recorded_init=next(x['request']['params'] for x in rows if x['request']['method']=='init');assert config==recorded_init,'recorded init differs from configuration snapshot';assert config['configuration']['score_mode']=='S' and config['configuration']['total_budget']==40 and config['configuration'].get('asset_grouping_profile') is None
config['execution_allowed']=False;config.pop('semantic_configuration',None)
diffs=[];counts={'observe':0,'next_action':0,'global_choice':0,'next_action_returns_matched':0,'paid_actions_matched':0,'candidate_factors_compared':0};actual_actions=[];actual_selections=[];last_action=None;started=time.monotonic()
def compare(label,expected,actual):
 def walk(path,x,y):
  if isinstance(x,bool) or isinstance(y,bool):
   if x!=y or type(x)!=type(y):diffs.append({'field':path,'expected':x,'actual':y})
  elif isinstance(x,int) and isinstance(y,int):
   if x!=y:diffs.append({'field':path,'expected':x,'actual':y})
  elif isinstance(x,(int,float)) and isinstance(y,(int,float)):
   if type(x)!=type(y):
    diffs.append({'field':path,'expected_type':type(x).__name__,'actual_type':type(y).__name__});return
   exact_time=any(k in path.lower() for k in ('stamp','timestamp','time_ns'))
   if (x!=y if exact_time else not math.isclose(x,y,rel_tol=a.rtol,abs_tol=a.atol)):diffs.append({'field':path,'expected':x,'actual':y})
  elif isinstance(x,dict) and isinstance(y,dict):
   if x.keys()!=y.keys():diffs.append({'field':path,'expected_keys':sorted(x),'actual_keys':sorted(y)})
   for k in x.keys()&y.keys():walk(path+'.'+k,x[k],y[k])
  elif isinstance(x,list) and isinstance(y,list):
   if len(x)!=len(y):diffs.append({'field':path,'expected_length':len(x),'actual_length':len(y)})
   for i,(u,v) in enumerate(zip(x,y)):walk(f'{path}[{i}]',u,v)
  elif x!=y:diffs.append({'field':path,'expected':x,'actual':y})
 walk(label,expected,actual)
def put_result(status,error=None):
 output={'schema':'nso.cached_semantic_S_policy_replay.v1','status':status,'scope':'offline downstream physical-input replay, cached semantic arrays replace frontend model; no robot, ROS or inference','tolerances':{'atol':a.atol,'rtol':a.rtol},'counts':counts,'actual_actions':actual_actions,'actual_global_choices':actual_selections,'recorded_init_equals_source_configuration':True,'comparison_scope':{'exact_frontend_substitution':['recorded real_scene_structural_v1 to replay no_real_semantics','recorded frontend_enabled true to replay false','recorded inference receipt to replay null; cached labels preserved'],'selection':'all keys, candidate paths, scores, score audit, candidate audit and hashes','integers_and_timestamps':'exact','other_numeric':'declared atol/rtol','event_comparison':'complete runtime event objects including guard assessments'},'source_sha256':{str(x.relative_to(P)):hashlib.sha256(x.read_bytes()).hexdigest() for x in [pathlib.Path(__file__).resolve(),P/'decisions.jsonl',P/'frames_manifest.json',P/'source/joint_configuration.json',P/'source/source_manifest.json']},'versions':{'python':sys.version.split()[0],**{name:importlib.metadata.version(name) for name in ('numpy','scipy','open3d')}},'elapsed_s':time.monotonic()-started,'difference_count':len(diffs),'differences':diffs[:100],'error':error}
 (out/'replay_result.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n');print(json.dumps({'status':status,**counts,'difference_count':len(diffs),'error':error}),flush=True)
def normalized_summary(summary,original):
 value=copy.deepcopy(summary);variant=value['system_variant'];diag=value['semantic_diagnostics'];expected_variant=('exploration_joint_v1+real_scene_structural_v1+sensor_range_matched_v1' if original else 'exploration_joint_v1+no_real_semantics+sensor_range_matched_v1');assert variant==expected_variant,('unexpected system variant',variant);assert diag['frontend_enabled'] is original
 if original:
  assert isinstance(diag['receipt'],dict) and diag['receipt'].get('model_execution_performed') is True
 else:assert diag['receipt'] is None
 value['system_variant']='exploration_joint_v1+cached_real_scene_structural_v1+sensor_range_matched_v1';diag['frontend_enabled']='cached_semantics';diag['receipt']='cached_receipt_not_reexecuted';return value
def normalized_result(result,original):
 value=copy.deepcopy(result);value['summary']=normalized_summary(value['summary'],original);return value
if a.self_test:
 for label,x,y in [('large_integer',1794058115711594857,1794058115711594858),('timestamp',1794058115.7115948,1794058116.7115948),('frame_id','frame-a','frame-b'),('candidate',[1,2,3],[1,2,4]),('score',1.0,1.01),('numeric_type',1,1.0)]:
  before=len(diffs);compare(label,x,y);assert len(diffs)>before,label
 assert len(diffs)==6;diffs.clear();put_result('validator_self_test_passed');sys.exit(0)
try:
 session=JointSession(**config)
 for record in rows:
  method=record['request']['method'];expected=record['response'].get('result');assert record['response'].get('ok')
  if method=='observe':
   params=dict(record['request']['params']);aid=params['action_id'];f=frames[aid];assert pathlib.Path(params['path']).name==f['source_packet_name'];params['path']=str(P/f['file'])
   if aid>0 and params.get('action')!=last_action:raise RuntimeError(f'replay action diverged at {aid}: engine={last_action} historical={params.get("action")}')
   result=session.observe(**params);counts['observe']+=1
   compare(f'observe[{aid}].complete_result',normalized_result(expected,True),normalized_result(result,False))
   for key in ('accepted','duplicate','value','parts','role','state'):
    if key in expected or key in result:compare(f'observe[{aid}].{key}',expected.get(key),result.get(key))
   compare(f'observe[{aid}].summary',normalized_summary(expected['summary'],True),normalized_summary(result['summary'],False))
   er=expected['summary']['runtime'];rr=result['summary']['runtime']
   for key in ('closed','mapper_frames','unique_packet_count','arrived_count','termination'):compare(f'observe[{aid}].runtime.{key}',er.get(key),rr.get(key))
   for key in ('feedback','observed_gain_calibration','conditional_gain_residual','return_anchor','plans','graph_version','semantic_version'):compare(f'observe[{aid}].modules.{key}',er['modules'].get(key),rr['modules'].get(key))
   compare(f'observe[{aid}].semantic_quality_labels',expected['summary']['semantic_diagnostics']['quality_labels'],result['summary']['semantic_diagnostics']['quality_labels'])
   compare(f'observe[{aid}].measured_assets',expected['summary']['semantic_diagnostics']['measured_assets'],result['summary']['semantic_diagnostics']['measured_assets'])
   if aid==20:
    if diffs:raise RuntimeError('prefix20 audit skipped because policy replay diverged')
    import importlib.util
    spec=importlib.util.spec_from_file_location('nso_prefix20_semantic_audit',str(a.prefix_audit_module.resolve()));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.audit(session,out/'prefix20_budget40_candidate_audit.json')
    print('prefix20 remaining40 read-only candidate audit completed before next_action',flush=True)
  elif method=='next_action':
   result=session.next_action();idx=counts['next_action'];counts['next_action']+=1;last_action=result.get('action');actual_actions.append(last_action)
   compare(f'next[{idx}].complete_result',normalized_result(expected,True),normalized_result(result,False))
   compare(f'next[{idx}].action',expected.get('action'),last_action)
   if expected.get('action')==last_action:
    counts['next_action_returns_matched']+=1
    if last_action is not None:counts['paid_actions_matched']+=1
   for key in ('pending','selected'):
    compare(f'next[{idx}].{key}',expected.get(key),result.get(key))
   es=expected.get('selection');rs=result.get('selection')
   if (es is None)!=(rs is None):compare(f'next[{idx}].selection',es,rs)
   elif es is not None:
    compare(f'next[{idx}].selection',es,rs)
    counts['candidate_factors_compared']+=len((rs or {}).get('candidates') or [])
   compare(f'next[{idx}].summary',normalized_summary(expected['summary'],True),normalized_summary(result['summary'],False))
   ee=[e.get('event') for e in expected.get('runtime_events',[])];re=[e.get('event') for e in result.get('runtime_events',[])];compare(f'next[{idx}].runtime_events',ee,re)
   if 'global_choice' in re:
    counts['global_choice']+=1;sel=result.get('selected') or {};actual_selections.append({'selection_call_id':sel.get('selection_call_id'),'selected_map_version':sel.get('selected_map_version'),'candidate_id':sel.get('candidate_id'),'plan_number':(result.get('selection') or {}).get('plan_number')})
   compare(f'next[{idx}].termination',(expected['summary']['runtime'] or {}).get('termination'),(result['summary']['runtime'] or {}).get('termination'))
   for key in ('feedback','observed_gain_calibration','conditional_gain_residual','return_anchor','plans','graph_version','semantic_version'):
    compare(f'next[{idx}].modules.{key}',expected['summary']['runtime']['modules'].get(key),result['summary']['runtime']['modules'].get(key))
   print(f'next {idx}: {last_action} global={counts["global_choice"]} diffs={len(diffs)}',flush=True)
   if diffs:put_result('diverged');sys.exit(1)
  elif method in ('init','shutdown'):continue
  else:raise RuntimeError('unexpected recorded method '+str(method))
 assert counts['observe']==41 and counts['next_action']==41 and counts['global_choice']==12 and counts['next_action_returns_matched']==41 and counts['paid_actions_matched']==40
 put_result('matched' if not diffs else 'diverged')
 if a.post_audit_module is not None and not diffs:
  import importlib.util
  try:
   spec=importlib.util.spec_from_file_location('nso_frozen_semantic_audit',str(a.post_audit_module.resolve()));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.audit(session,out/'frozen_semantic_availability.json')
   print('optional frozen-state audit completed separately',flush=True)
  except Exception as error:
   (out/'post_audit_error.json').write_text(json.dumps({'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc(limit=6)},indent=2)+'\n');print('optional audit failed; replay result retained',flush=True);sys.exit(2)
except Exception as e:
 put_result('error',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc(limit=4)});raise
