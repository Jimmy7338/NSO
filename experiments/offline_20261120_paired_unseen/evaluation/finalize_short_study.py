"""Freeze all prescribed short evidence after both stages, regardless of sign."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--first',type=Path,required=True);p.add_argument('--second',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--wait',action='store_true');a=p.parse_args()
 while not(a.first.exists()and a.second.exists()):
  if not a.wait:raise FileNotFoundError('Both phase release receipts are required')
  time.sleep(10)
 first,second=read(a.first),read(a.second);assert first['phase']=='short_first'and second['phase']=='short_second'
 refs=[];phasehash={}
 for path,d in [(a.first,first),(a.second,second)]:
  phasehash[str(path.resolve())]=sha(path)
  for f,h in d['table_files_sha256'].items():assert sha(f)==h,f
  refs.extend(Path(x['result'])for x in d['states'].values()if x['status']=='evaluated')
 assert len(refs)==8,'All eight prescribed states must be represented before final table'
 a.output.mkdir(parents=True,exist_ok=False);env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 def run(script,args):
  r=subprocess.run([sys.executable,str(script),*map(str,args)],env=env,text=True,capture_output=True);assert r.returncode==0,r.stderr;print(r.stdout.strip(),flush=True)
 run(HERE/'build_short_table.py',['--state-results',*refs,'--layouts','chain_a','cycle_a','chain_b','cycle_b','--output',a.output/'table'])
 run(HERE/'selection_decomposition.py',['--state-results',*refs,'--output',a.output/'decomposition'])
 run(ROOT/'paper_update/build_short_results.py',['--summary',a.output/'table/summary.json','--output',a.output/'paper_short'])
 run(ROOT/'paper_update/build_short_figures.py',['--summary',a.output/'table/summary.json','--output',a.output/'figures'])
 summary=read(a.output/'table/summary.json');paper=read(a.output/'paper_short/PAPER_SHORT_RECEIPT.json');completed=first['completed_actual_branches']+second['completed_actual_branches']
 sources=[Path(__file__),HERE/'build_short_table.py',HERE/'selection_decomposition.py',ROOT/'paper_update/build_short_results.py',ROOT/'paper_update/build_short_figures.py']
 receipt=dict(status='all_24_short_branches_and_paper_tables_frozen',created_unix_s=time.time(),independent_layouts=4,room_topologies=2,dependent_states=8,unique_actual_candidates=summary['unique_actual_branches'],completed_actual_branches=completed,incomplete_or_failed_branches=paper['failed_or_incomplete_branches'],old_new_changed_states=paper['changed_states'],actual_area_wins_losses_ties=[paper['actual_gain_state_wins'],paper['actual_gain_state_losses'],paper['actual_gain_state_ties']],release_ready_for_declared_full_runs=first['release_ready']and second['release_ready'],no_outcome_sign_selection=True,actor_actions=0,phase_receipts_sha256=phasehash,table_summary=summary['layout_rows'],old_MAE_m2=paper['old_MAE_m2'],new_MAE_m2=paper['new_MAE_m2'],source_sha256={str(f.resolve()):sha(f)for f in sources},files_sha256={str(f.resolve()):sha(f)for f in a.output.rglob('*')if f.is_file()})
 (a.output/'RELEASE_FULL_RUNS.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
