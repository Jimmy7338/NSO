"""Evaluate a declared short stage and freeze its table; never launch actors."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time,traceback
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def verify_evaluator():
 f=read(HERE/'freeze.json')
 for p,h in f['files_sha256'].items():assert sha(p)==h,p
def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['short_first','short_second'],required=True);p.add_argument('--batch',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False);verify_evaluator();begin=time.time();expected=['chain_a','cycle_a']if a.phase=='short_first'else['chain_b','cycle_b'];seen={};env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 while True:
  status=read(a.batch/'status.json');assert status['phase']==a.phase
  jobs=status['completed']+status['running']+status['pending'];assert len(jobs)==4
  assert {(x['layout'],x['prefix'])for x in jobs}=={(l,s)for l in expected for s in ('p1','p2')}
  for job in jobs:
   actor=Path(job['output']);key=f"{job['layout']}/{job['prefix']}";manifest=actor/'manifest.json'
   if key in seen:continue
   if manifest.exists():
    out=HERE/'short_results'/actor.name
    if not (out/'result.json').exists():
     if out.exists():out=a.output/'failed_retry_evaluations'/actor.name
     run=subprocess.run([sys.executable,str(HERE/'evaluate_state.py'),'--manifest',str(manifest),'--output',str(out)],env=env,text=True,capture_output=True)
     if run.returncode:
      seen[key]=dict(status='evaluator_failed',actor_output=str(actor),output=str(out),returncode=run.returncode,stderr=run.stderr);continue
    d=read(out/'result.json')
    for path,h in d['input_sha256'].items():assert sha(path)==h,path
    seen[key]=dict(status='evaluated',actor_output=str(actor),result=str(out/'result.json'),sha256=sha(out/'result.json'),branch_status={str(cid):v['status']for cid,v in d['actual_by_candidate'].items()})
   elif job.get('status')in ('process_completed','process_failed'):
    f=actor/'failure.json';seen[key]=dict(status='actor_failed_without_complete_manifest',actor_output=str(actor),exit_code=job.get('exit_code'),failure=read(f)if f.exists()else None)
  write(a.output/'status.json',dict(status='running',phase=a.phase,states=seen,elapsed_s=time.time()-begin,new_actor_actions=0))
  if status['status']=='finished'and len(seen)==4:break
  time.sleep(10)
 results=[v['result']for v in seen.values()if v['status']=='evaluated'];table=a.output/'table'
 if results:
  run=subprocess.run([sys.executable,str(HERE/'build_short_table.py'),'--state-results',*results,'--layouts',*expected,'--output',str(table)],env=env,text=True,capture_output=True);assert run.returncode==0,run.stderr
 completed=sum(v=='completed'for s in seen.values()for v in s.get('branch_status',{}).values());attempted=sum(len(s.get('branch_status',{}))for s in seen.values());ready=len(results)==4 and completed==12
 receipt=dict(status='first_table_frozen_ready_for_next_declared_phase'if ready else'stage_evaluation_frozen_with_incomplete_or_failed_evidence',phase=a.phase,completed_actual_branches=completed,represented_actual_candidates=attempted,expected_actual_branches=12,all_four_states_evaluated=len(results)==4,states=seen,table=str(table)if results else None,release_ready=ready,new_actor_actions=0,no_outcome_sign_selection=True,independent_layouts=2,source_sha256=sha(__file__),evaluator_freeze_sha256=sha(HERE/'freeze.json'),preparation_addendum_sha256=sha(HERE/'preparation_v2_freeze.json'),execution_freeze_sha256=sha(ROOT/'execution/freeze.json'),batch_status_sha256=sha(a.batch/'status.json'),created_unix_s=time.time())
 files=list(table.glob('*'))if table.exists()else[];receipt['table_files_sha256']={str(f):sha(f)for f in files if f.is_file()};write(a.output/'release_receipt.json',receipt);write(a.output/'status.json',receipt);print(json.dumps(receipt))
if __name__=='__main__':main()
