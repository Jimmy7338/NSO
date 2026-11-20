"""Validate completed artifact after stdout-only NumPy JSON error; no rerun."""
from pathlib import Path
import json,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 out=HERE/'serialization_retry_002';result=json.loads((out/'result.json').read_text());assert result['objects_completed']==12 and result['routes_evaluated']==36
 protocol=json.loads((HERE/'protocol.json').read_text())
 for p,h in protocol['code_sha256'].items():assert sha(p)==h,p
 frozen=json.loads((out/'retry_freeze.json').read_text());assert sha(HERE/'serialize_retry.py')==frozen['wrapper_sha256']
 parity=[]
 for name,h in frozen['prior_predictions_sha256'].items():
  assert sha(HERE/name)==h
  a=np.load(HERE/name,allow_pickle=False);b=np.load(out/name,allow_pickle=False);assert a.files==b.files
  for k in a.files:np.testing.assert_array_equal(a[k],b[k])
  parity.append(name)
 receipt=dict(status='completed_artifact_verified_after_stdout_serialization_error',objects=12,routes=36,
  arrays_exactly_identical_to_original_attempt=parity,source_protocol_and_method_unchanged=True,extra_independent_samples=0,
  original_failure='Final result JSON write failed on numpy.int64 after all12 computations.',
  retry_terminal_error='Result JSON successfully written with scalar conversion; subsequent stdout json.dumps still raised numpy.int64. No third run was performed.',
  result_sha256=sha(out/'result.json'),protocol_sha256=sha(out/'protocol.json'),finalizer_sha256=sha(Path(__file__)))
 (out/'retry_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(dict(result_sha256=receipt['result_sha256'],summary=result['summary'],receipt_sha256=sha(out/'retry_receipt.json'))))
if __name__=='__main__':main()
