#!/usr/bin/env python3
"""Offline derived-input integrity check and measured-history map replay."""
import argparse, hashlib, json, os, pathlib, sys, tempfile
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np
P=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(P/'code'))
from nso_bridge.joint_engine import JointSession
h=lambda b:hashlib.sha256(b).hexdigest()
def check():
 m=json.loads((P/'manifest.json').read_text())
 for f in m['frames']:
  p=P/f['file']; assert h(p.read_bytes())==f['sha256']
  with np.load(p,allow_pickle=False) as z:
   assert set(z.files)==set(f['arrays'])
   for k,x in f['arrays'].items():
    a=z[k]; assert str(a.dtype)==x['dtype'] and list(a.shape)==x['shape'] and h(np.ascontiguousarray(a).tobytes())==x['sha256'],(f['file'],k)
 return m

def replay(m,source=None):
 base=json.loads((P/'reference/configuration.json').read_text());base['execution_allowed']=False;base.pop('semantic_configuration',None)
 with tempfile.TemporaryDirectory() as td:
  desc=[]
  for f in m['frames']:
   p=pathlib.Path(source)/pathlib.Path(f['file']).name if source else P/f['file']; sha=h(p.read_bytes()); rec=pathlib.Path(td)/(str(f['action_id'])+'.json');rec.write_text(json.dumps({'copied_packet_sha256':sha}));desc.append({'path':str(p),'sha256':sha,'annotation_receipt':str(rec),'annotation_receipt_sha256':h(rec.read_bytes()),'registration':{'kind':'same_continuous_odom','externally_verified':True}})
  base['shared_observed_prefix']=desc
  s=JointSession(**base); obs=s.observe(str(P/'reference/live0.npz'),0); decision=s.next_action()
  arrays={k:{'shape':list(v.shape),'dtype':str(v.dtype),'sha256':h(np.ascontiguousarray(v).tobytes())} for k,v in vars(s.mapper).items() if isinstance(v,np.ndarray)}
  return {'map_arrays':arrays,'state':obs['state'],'next_action':decision}
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--compare-original',help='directory containing original converted frame_*.npz');a=ap.parse_args();m=check();r=replay(m)
 if a.compare_original:
  
  for f in m['frames']:
   with np.load(pathlib.Path(a.compare_original)/pathlib.Path(f['file']).name,allow_pickle=False) as z:
    for k,x in f['arrays'].items():
     v=z[k]; assert str(v.dtype)==x['dtype'] and list(v.shape)==x['shape'] and h(np.ascontiguousarray(v).tobytes())==x['sha256'], (f['file'],k)
  orig=replay(m,a.compare_original);assert r==orig,'map, state or decision differs from original converted inputs'
 expected=P/'reference/replay_expected.json'
 assert expected.exists(),'missing bundled expected result'
 assert r==json.loads(expected.read_text()),'map, state or decision differs from bundled expected result'
 print(json.dumps({'frames_verified':len(m['frames']),'map_arrays_verified':len(r['map_arrays']),'original_comparison':bool(a.compare_original),'state':r['state'],'next_action':r['next_action'].get('action')}))
