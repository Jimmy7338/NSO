#!/usr/bin/env python3
"""Run only the six declared reused-prefix prediction jobs, three at a time."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
HERE=Path(__file__).resolve().parent
TASKS=[('shelf_left',24),('shelf_left',54),('shelf_left',109),('shelf_right',24),('shelf_right',53),('shelf_right',110)]
PYTHON='/home/ouc/nso_robot/envs/science312/bin/python'

def run(pair):
 layout,paid=pair;out=HERE/f'{layout}_{paid}_attempt_001';log=HERE/(out.name+'.log')
 assert not out.exists() and not log.exists(),'Never overwrite an attempt'
 command=[PYTHON,str(HERE/'run_particle_predictions.py'),'--layout',layout,'--state',str(paid),'--output',str(out)]
 start=time.monotonic();env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
 with log.open('x')as handle:code=subprocess.call(command,stdout=handle,stderr=subprocess.STDOUT,env=env)
 row=dict(layout=layout,state=paid,returncode=code,output=str(out),log=str(log),elapsed_s=time.monotonic()-start)
 if (out/'result.json').exists():row['result_sha256']=hashlib.sha256((out/'result.json').read_bytes()).hexdigest()
 return row

def main():
 records=[];start=time.monotonic()
 with ThreadPoolExecutor(max_workers=3)as executor:
  for future in as_completed([executor.submit(run,pair)for pair in TASKS]):
   row=future.result();records.append(row)
   receipt=dict(schema='B8_six_prefix_prediction_execution_v1',tasks=TASKS,completed=records,pending=len(TASKS)-len(records),new_candidate_actions=0,hardware=False,elapsed_s=time.monotonic()-start)
   target=HERE/'batch_execution.json';temp=target.with_suffix('.tmp');temp.write_text(json.dumps(receipt,indent=2)+'\n');temp.replace(target)
   print(json.dumps(row),flush=True)
 return 0 if all(r['returncode']==0 for r in records) else 1
if __name__=='__main__':sys.exit(main())
