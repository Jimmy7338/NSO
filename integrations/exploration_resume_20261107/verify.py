#!/usr/bin/env python3
"""Verify packaged canonical arrays and recorded scalar diagnostic offline."""
import hashlib,json,pathlib,subprocess,sys,numpy as np
P=pathlib.Path(__file__).resolve().parent;h=lambda b:hashlib.sha256(b).hexdigest();m=json.loads((P/'frames_manifest.json').read_text());assert len(m['frames'])==41
for expected_action,f in enumerate(m['frames']):
 assert f['action_id']==expected_action;path=P/f['file'];assert h(path.read_bytes())==f['sha256'];assert h((P/f['receipt']).read_bytes())==f['source_receipt_sha256'];rec=json.loads((P/f['receipt']).read_text());assert rec['source_file_sha256']==f['source_packet_sha256']
 with np.load(path,allow_pickle=False) as z:
  assert set(z.files)==set(f['arrays'])
  for k,a in f['arrays'].items():
   v=z[k];assert list(v.shape)==a['shape'] and str(v.dtype)==a['dtype'] and h(np.ascontiguousarray(v).tobytes())==a['sha256'],(f['file'],k)
  assert str(z['frame_id'])==f['frame_id'] and h(np.ascontiguousarray(z['semantic']).tobytes())==rec['canonical_semantic_sha256']
actual=json.loads(subprocess.check_output([sys.executable,str(P/'audit.py'),str(P/'decisions.jsonl')],text=True));expected=json.loads((P/'semantic_aperture_audit.json').read_text());assert actual==expected
selection=json.loads(subprocess.check_output([sys.executable,str(P/'selection_audit.py'),str(P/'decisions.jsonl')],text=True));assert selection==json.loads((P/'selection_audit.json').read_text())
print(json.dumps({'global_choices_verified':selection['global_choice_events'],'frames_verified':len(m['frames']),'semantic_aperture_scalar_audit':'matched','scope':'array integrity and recorded scalar calculations; no inference or policy replay'}))
