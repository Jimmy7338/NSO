"""Offline compression comparison on recorded packets; no ROS or production writes."""
import hashlib,json,pathlib,tempfile,time,zipfile
import numpy as np
import sys
sys.path.insert(0,"/home/ouc/nso_robot/workspaces/nso_hybrid_20261029/runtime")
from numpy_archive import savez_fast
P=pathlib.Path('/home/ouc/nso_robot/reports/nso-hybrid-20261108/nov06_timing_S40_20261108_02')
OUT=pathlib.Path(__file__).parent/('production_python38_receipt.json' if sys.version_info[:2] == (3,8) else 'receipt.json')
def sha(data):return hashlib.sha256(data).hexdigest()
def summary(a):return {'dtype':a.dtype.str,'shape':list(a.shape),'array_sha256':sha(a.tobytes(order='C'))}
def stats(v):return {'count':len(v),'sum_s':sum(v),'mean_s':float(np.mean(v)),'median_s':float(np.median(v)),'min_s':min(v),'max_s':max(v)}
events=[json.loads(s) for s in (P/'events.jsonl').read_text().splitlines()]; obs=[x for x in events if x.get('event')=='NSO_joint_observation_accepted'];assert len(obs)==41
keys=sorted({k for x in obs for k in x['capture_timings']}); timings={k:stats([x['capture_timings'][k] for x in obs if k in x['capture_timings']]) for k in keys}
rpcs={}
for x in events:
 if x.get('event')=='NSO_joint_rpc' and 'rpc_elapsed_s' in x:rpcs.setdefault(x['method'],[]).append(x['rpc_elapsed_s'])
rpcs={k:stats(v) for k,v in rpcs.items()}
results=[]
with tempfile.TemporaryDirectory(prefix='nso_npz_compression_') as tmp:
 for i in (0,20,40):
  joint=pathlib.Path(obs[i]['path']); stamp=joint.stem[len('joint-'):]; raw=P/'observed'/f'{stamp}.npz'
  for kind,source in [('reference',raw),('joint',joint)]:
   with np.load(source,allow_pickle=False) as z: arrays={k:z[k] for k in z.files}
   expected={k:summary(v) for k,v in arrays.items()}; row={'action_id':i,'kind':kind,'source':str(source),'source_sha256':sha(source.read_bytes()),'keys':expected,'methods':{}}
   for method in ('numpy_default','zip_deflate_level1'):
    target=pathlib.Path(tmp)/f'{i}_{kind}_{method}.npz';t=time.perf_counter()
    if method=='numpy_default':np.savez_compressed(target,**arrays)
    else:
     savez_fast(target,**arrays)
    elapsed=time.perf_counter()-t
    with np.load(target,allow_pickle=False) as z:
     assert set(z.files)==set(arrays), (i,kind,method)
     got={k:summary(z[k]) for k in z.files}
    assert got==expected,(i,kind,method)
    row['methods'][method]={'seconds':elapsed,'bytes':target.stat().st_size,'sha256':sha(target.read_bytes()),'all_keys_dtype_shape_and_raw_array_sha_equal':True}
   results.append(row)
receipt={'scope':'single offline sequential run on three fixed recorded frames, six packets; compression and file write timing excludes reading/decompression/array hashing; not live latency prediction','thread_limits':'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1','numpy_version':np.__version__,'events_sha256':sha((P/'events.jsonl').read_bytes()),'capture_timings_41':timings,'rpc_timings':rpcs,'packets':results,'limits':['One run per method and packet; filesystem cache and current host load affect timings.','Same raw array bytes including NaN bit patterns; NPZ archive bytes need not match.','No production changes; no ROS or robot.']};OUT.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'capture':{k:timings[k] for k in ['reference_npz_compress_write_s','joint_input_npz_compress_write_s']},'rpcs':rpcs,'bench':[{'id':x['action_id'],'kind':x['kind'],'methods':x['methods']} for x in results]}))
