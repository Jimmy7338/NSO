import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import ast, pathlib,json,hashlib,time,tracemalloc,statistics,platform,sys
import numpy as np
from reference_function import conservative_belief as reference
from candidate import conservative_belief as candidate
D=pathlib.Path(__file__).resolve().parent
P=pathlib.Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029/runtime/joint_observation.py')
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
source=P.read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='conservative_belief');fn=ast.get_source_segment(source,node)
namespace={'np':np};exec(fn,namespace);production=namespace['conservative_belief']
refnode=next(n for n in ast.parse((D/'reference_function.py').read_text()).body if isinstance(n,ast.FunctionDef))
assert ast.dump(node,include_attributes=False)==ast.dump(refnode,include_attributes=False)
rows=[]
roots=['matched_system_pilot_20261108_B1_C','nov06_timing_S40_20261108_02']
for episode in roots:
 for p in sorted((pathlib.Path('/home/ouc/nso_robot/reports/nso-hybrid-20261108')/episode).rglob('map-*.npz')):
  with np.load(p,allow_pickle=False) as z:
   a=z['occupancy'];res=float(z['resolution_m']);m=json.loads(str(z['receipt_json']))
  T=np.asarray(m['planning_from_grid']);shape=tuple(m['joint_grid']['shape']);args=(a,res,T,shape,m['joint_grid']['resolution_m'])
  times={};outs={}
  for name,func in [('production',production),('candidate',candidate)]:
   begin=time.perf_counter();outs[name]=func(*args);times[name]=time.perf_counter()-begin
  saved=p.with_name(p.name.replace('map-','joint-',1))
  with np.load(saved,allow_pickle=False) as z:saved_output=z['joint_belief_grid']
  equal=np.array_equal(outs['production'],outs['candidate']) and np.array_equal(outs['production'],saved_output)
  row=dict(path=str(p),input_sha256=sha(p),source_shape=list(a.shape),source_dtype=str(a.dtype),resolution_m=res,output_shape=list(shape),output_dtype=str(outs['candidate'].dtype),exact_equal_production_candidate_saved=bool(equal),output_sha256=hashlib.sha256(outs['production'].tobytes()).hexdigest(),timing_s=times,counts={str(v):int((outs['production']==v).sum()) for v in [-1,0,1]})
  rows.append(row);print(json.dumps(dict(case=len(rows),equal=equal,times=times)),flush=True)
  assert equal and outs['candidate'].dtype==outs['production'].dtype==saved_output.dtype==np.int8
  if len(rows)==1:representative=args
memory={};repeat={}
for name,func in [('production',production),('candidate',candidate)]:
 vals=[]
 for i in range(5):
  begin=time.perf_counter();func(*representative);vals.append(time.perf_counter()-begin)
 repeat[name]=dict(samples_s=vals,median_s=statistics.median(vals))
 tracemalloc.start();func(*representative);current,peak=tracemalloc.get_traced_memory();tracemalloc.stop();memory[name]=dict(tracemalloc_peak_bytes=peak,note='Includes NumPy allocations tracked by Python allocator; not process RSS; timing excludes tracing')
result=dict(status='offline_real_saved_input_validation_and_benchmark_only_not_deployed',all_exact_equal=all(r['exact_equal_production_candidate_saved'] for r in rows),cases=rows,source_hashes={str(p):sha(p) for p in [P,D/'reference_function.py',D/'candidate.py',D/'offline_benchmark.py']},production_function_sha256=hashlib.sha256(fn.encode()).hexdigest(),production_ast_matches_preserved_reference=True,production_unchanged_after_benchmark=sha(P)==sha(P),environment=dict(python=sys.version,numpy=np.__version__,platform=platform.platform(),thread_env={k:os.environ[k] for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']}),representative_repeat=repeat,representative_memory=memory)
result['aggregate']={k:dict(mean_s=statistics.mean(r['timing_s'][k] for r in rows),median_s=statistics.median(r['timing_s'][k] for r in rows)) for k in ['production','candidate']}
(D/'offline_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['aggregate','representative_repeat','representative_memory']}),flush=True)
