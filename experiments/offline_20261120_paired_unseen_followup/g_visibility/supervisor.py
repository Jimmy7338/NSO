"""Detached four-run extension; append attempts, never select a best rerun."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
HERE=Path(__file__).resolve().parent
LAYOUTS=('chain_a','cycle_a','chain_b','cycle_b')
def write(path,data):
    t=path.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2)+'\n');t.replace(path)
def supervise(folder,limit):
    import run
    run.verify()
    pending=[dict(layout=l,arm='G_visibility')for l in LAYOUTS];running=[];completed=[];started=time.time()
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    while pending or running:
        while pending and len(running)<limit:
            row=pending.pop(0);log=folder/(row['layout']+'.log');f=log.open('wb')
            command=[sys.executable,str(HERE/'run.py'),'run','--layout',row['layout']]
            process=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            row.update(pid=process.pid,command=command,log=str(log),started_unix_s=time.time(),status='running')
            running.append((process,f,row))
        for process,f,row in list(running):
            code=process.poll()
            if code is not None:
                f.close();row.update(exit_code=code,finished_unix_s=time.time(),status='process_completed'if code==0 else'process_failed')
                completed.append(row);running.remove((process,f,row))
        write(folder/'status.json',dict(status='running'if pending or running else'finished',supervisor_pid=os.getpid(),
            started_unix_s=started,elapsed_s=time.time()-started,running=[r for _,_,r in running],pending=pending,
            completed=completed,max_concurrent_runs=limit,numerical_library_thread_limit=1,failed_attempts_retained=True))
        if running:time.sleep(.5)
def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('launch','supervise'));p.add_argument('--folder',type=Path);p.add_argument('--parallel',type=int,default=3,choices=(1,2,3));a=p.parse_args()
    if a.command=='supervise':supervise(a.folder,a.parallel);return
    parent=HERE/'batches';parent.mkdir(exist_ok=True)
    for i in range(1,10000):
        folder=parent/f'attempt_{i:03d}'
        try:folder.mkdir();break
        except FileExistsError:continue
    log=(folder/'supervisor.log').open('wb')
    child=subprocess.Popen([sys.executable,str(Path(__file__)),'supervise','--folder',str(folder),'--parallel',str(a.parallel)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close()
    receipt=dict(status='detached_supervisor_started',supervisor_pid=child.pid,folder=str(folder),max_concurrent_runs=a.parallel)
    write(folder/'launch.json',receipt);print(json.dumps(receipt))
if __name__=='__main__':main()
