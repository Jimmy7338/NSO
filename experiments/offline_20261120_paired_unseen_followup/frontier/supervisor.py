"""Finite predeclared batch with at most two low-thread frontier actors."""
from pathlib import Path
import json, os, subprocess, sys, time
HERE=Path(__file__).resolve().parent
LAYOUTS=['chain_a','cycle_a','chain_b','cycle_b']

def main():
    out=HERE/'batch_attempt_001';out.mkdir(exist_ok=False)
    pending=LAYOUTS.copy();active={};completed=[];failures=[]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    while pending or active:
        while pending and len(active)<2:
            layout=pending.pop(0);log=open(out/(layout+'.log'),'w')
            p=subprocess.Popen([sys.executable,str(HERE/'run.py'),'run','--layout',layout],
                stdout=log,stderr=subprocess.STDOUT,env=env)
            active[layout]=(p,log)
        for layout,(p,log) in list(active.items()):
            code=p.poll()
            if code is not None:
                log.close();del active[layout]
                (completed if code==0 else failures).append(dict(layout=layout,exit_code=code))
        status=dict(status='running' if pending or active else 'completed_batch',
            active={k:v[0].pid for k,v in active.items()},pending=pending,completed=completed,
            failures=failures,automatic_retries=False,concurrency=2)
        temp=out/'status.tmp';temp.write_text(json.dumps(status,indent=2));temp.replace(out/'status.json')
        if pending or active:time.sleep(2)
    print(json.dumps(status),flush=True)
    return bool(failures)

if __name__=='__main__':raise SystemExit(main())
