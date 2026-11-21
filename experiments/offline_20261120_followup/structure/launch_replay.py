import concurrent.futures, json, os, subprocess, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
OUT=HERE/'replay_attempt_001'
OUT.mkdir(exist_ok=False)
jobs=[(l,n) for l,steps in [('shelf_left',(24,54,109)),('shelf_right',(24,53,110))] for n in steps]
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
def run(job):
    layout,state=job;name=f'{layout}_{state}';start=time.monotonic()
    cmd=[sys.executable,str(HERE/'replay.py'),'--layout',layout,'--state',str(state),
         '--temperature-config',str(HERE/'audit/calibration_attempt_001/selected_temperature.json'),
         '--output',str(OUT/name)]
    with (OUT/(name+'.log')).open('w') as stream:
        process=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,env=env)
    return dict(layout=layout,state=state,returncode=process.returncode,elapsed_s=time.monotonic()-start)
rows=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    for future in concurrent.futures.as_completed([pool.submit(run,job) for job in jobs]):
        row=future.result();rows.append(row);print(json.dumps(row),flush=True)
        (OUT/'batch.json').write_text(json.dumps(dict(completed=len(rows),total=len(jobs),rows=rows),indent=2)+'\n')
sys.exit(int(any(row['returncode'] for row in rows)))
