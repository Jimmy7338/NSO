"""Detached, explicit B11 study phases; children survive tool-call lifetimes."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
LAYOUT_ORDER=('chain_a','cycle_a','chain_b','cycle_b')
def write(p,x):
 temp=p.with_suffix('.tmp');temp.write_text(json.dumps(x,indent=2)+'\n');temp.replace(p)
def attempt(parent):
 parent.mkdir(parents=True,exist_ok=True)
 for i in range(1,10000):
  p=parent/f'attempt_{i:03d}'
  try:p.mkdir();return p
  except FileExistsError:pass
 raise RuntimeError('attempt indices exhausted')
def jobs(phase):
 if phase=='full':return [dict(layout=l,arm=a,command=[sys.executable,str(HERE/'run.py'),'run','--layout',l,'--arm',a])for l in LAYOUT_ORDER for a in ('S_visibility','S_acquisition')]
 layouts=LAYOUT_ORDER[:2]if phase=='short_first'else LAYOUT_ORDER[2:]
 result=[]
 for l in layouts:
  for p in ('p1','p2'):
   stem=ROOT/'short'/f'{l}_{p}';i=1
   while Path(str(stem)+f'_attempt_{i:03d}').exists():i+=1
   out=Path(str(stem)+f'_attempt_{i:03d}')
   result.append(dict(layout=l,prefix=p,output=str(out),command=[sys.executable,str(HERE/'short_branches.py'),'--layout',l,'--prefix',p,'--output',str(out)]))
 return result

def supervise(phase,folder):
 from short_branches import verify
 verify();pending=jobs(phase);running=[];completed=[];started=time.time()
 env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1')
 while pending or running:
  while pending and len(running)<3:
   row=pending.pop(0);log=folder/(row['layout']+'_'+row.get('prefix',row.get('arm'))+'.log');f=log.open('wb')
   p=subprocess.Popen(row['command'],stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True);row.update(pid=p.pid,log=str(log),started_unix_s=time.time(),status='running');running.append((p,f,row))
  for p,f,row in list(running):
   code=p.poll()
   if code is not None:
    f.close();row.update(exit_code=code,finished_unix_s=time.time(),status='process_completed'if code==0 else'process_failed');completed.append(row);running.remove((p,f,row))
  write(folder/'status.json',dict(phase=phase,status='running'if pending or running else'finished',supervisor_pid=os.getpid(),started_unix_s=started,elapsed_s=time.time()-started,completed=completed,running=[r for _,_,r in running],pending=pending,max_concurrent_runs=3,numerical_library_thread_limit=1,interrupted_attempts_never_counted_complete=True))
  if running:time.sleep(.5)

def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=('launch','supervise'));p.add_argument('--phase',required=True,choices=('short_first','short_second','full'));p.add_argument('--folder',type=Path);a=p.parse_args()
 if a.command=='supervise':supervise(a.phase,a.folder);return
 folder=attempt(HERE/'batches'/a.phase);log=(folder/'supervisor.log').open('wb')
 child=subprocess.Popen([sys.executable,str(Path(__file__)),'supervise','--phase',a.phase,'--folder',str(folder)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close()
 receipt=dict(status='detached_supervisor_started',phase=a.phase,supervisor_pid=child.pid,folder=str(folder));write(folder/'launch.json',receipt);print(json.dumps(receipt))
if __name__=='__main__':main()
