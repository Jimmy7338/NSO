"""Rerun unchanged frozen validation after NumPy scalar JSON failure.

Only final serialization is changed; source, protocol, objects, method and
all arrays remain fixed. The retry is not an independent replicate.
"""
from pathlib import Path
import importlib.util,json,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 Path(p).write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda a:a.item()if isinstance(a,np.generic)else a.tolist())+'\n')
def main():
 out=HERE/'serialization_retry_002';out.mkdir(exist_ok=False)
 write(HERE/'failure_attempt_001.json',dict(status='all12computations_completed_final_JSON_serialization_failed',error='numpy.int64 summary count not JSON serializable',source_sha256=sha(HERE/'validate.py'),protocol_sha256=sha(HERE/'protocol.json'),prior_completed=12,method_or_parameters_changed=False))
 protocol=json.loads((HERE/'protocol.json').read_text())
 for p,h in protocol['code_sha256'].items():assert sha(p)==h
 (out/'protocol.json').write_bytes((HERE/'protocol.json').read_bytes())
 write(out/'retry_freeze.json',dict(status='frozen_serialization_only_retry',wrapper_sha256=sha(Path(__file__)),protocol_sha256=sha(HERE/'protocol.json'),original_failure_sha256=sha(HERE/'failure_attempt_001.json'),prior_predictions_sha256={p.name:sha(p)for p in HERE.glob('object_*_predictions.npz')},extra_independent_samples=0))
 spec=importlib.util.spec_from_file_location('fixed_validation_retry',HERE/'validate.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
 v.HERE=out;v.write=write;v.run()
 parity=[]
 for p in HERE.glob('object_*_predictions.npz'):
  a=np.load(p,allow_pickle=False);b=np.load(out/p.name,allow_pickle=False);assert a.files==b.files
  for k in a.files:np.testing.assert_array_equal(a[k],b[k])
  parity.append(p.name)
 write(out/'retry_receipt.json',dict(status='serialization_retry_completed_exact_original_arrays',arrays_exactly_identical=parity,objects=12,routes=36,extra_independent_samples=0,result_sha256=sha(out/'result.json')))
if __name__=='__main__':main()
