"""Read-only original audit; tiny synthetic fixtures intercept EVERY unlink."""
from pathlib import Path
import contextlib,copy,hashlib,importlib.util,io,json,subprocess,sys,tempfile
from unittest.mock import patch
import numpy as np
ROOT=Path('/root/NSO'); OUT=Path(__file__).resolve().parent
SOURCE=ROOT/'scripts/compact_response_v7_meshes.py'
spec=importlib.util.spec_from_file_location('compaction_under_review',SOURCE)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
initial_source_sha=mod.sha(SOURCE)

def blob(extra=False):
 b=io.BytesIO();a={'vertices':np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]),'triangles':np.array([[0,1,2]],np.int32)}
 if extra:a['vertex_colors']=np.ones((3,3))
 np.savez_compressed(b,**a);return b.getvalue()

def write_json(p,data):p.write_text(json.dumps(data,sort_keys=True)+'\n')

def fixture(folder,case):
 folder.mkdir()
 normal=blob();protected=['storage_shelves/prefix/frames/0020.npz','storage_shelves/prefix/scans/0020.npz',
 'storage_shelves/prefix_mesh.npz','storage_shelves/reference.npz','storage_shelves/candidate_000/arrival_mesh.npz',
 'storage_shelves/candidate_000/final_mesh.npz']
 metadata={'status':'complete','failures':0,'source_sha256':{'environment.py':'a'*64}}
 config={'storage':{'mesh_removal_only_after_independent_replay':True,'mesh_fast_inspection_route':0}}
 if case=='running_metadata':metadata['status']='running'
 if case=='failed_branch':metadata['failures']=1
 if case=='policy_disallows_removal':config['storage']['mesh_removal_only_after_independent_replay']=False
 if case=='wrong_keep_route':config['storage']['mesh_fast_inspection_route']=1
 write_json(folder/'metadata.json',metadata);write_json(folder/'config.json',config)
 (folder/'sources.zip').write_bytes(b'FAKE_SOURCE_CONTAINER_ONLY_FOR_PREFLIGHT_TEST')
 files=sorted(mod.REMOVABLE)+protected
 intended={}
 for i,name in enumerate(files):
  path=folder/name;path.parent.mkdir(parents=True,exist_ok=True)
  value=blob(extra=True) if case=='extra_mesh_array' and name==sorted(mod.REMOVABLE)[0] else normal
  intended[name]=hashlib.sha256(value).hexdigest()
  if not (case=='missing_protected_raw' and name==protected[0]):path.write_bytes(value)
 for name in ['metadata.json','config.json','sources.zip']:intended[name]=mod.sha(folder/name)
 write_json(folder/'artifact_hashes.json',intended)
 verification={'status':'passed_full','passed_full':True,'partial':False,'max_branches':None,
  'branches_checked':12,'branches_total':12,'raw_hashes_rechecked_after_replay':True,
  'artifact_manifest_sha256':mod.sha(folder/'artifact_hashes.json'),
  'source_archive_sha256':mod.sha(folder/'sources.zip'),'source_sha256':metadata['source_sha256']}
 changes={'failed_verification':{'status':'failed','passed_full':False},
 'partial_verification':{'partial':True},'partial_cap':{'max_branches':2},
 'eleven_checked':{'branches_checked':11},'raw_not_rechecked':{'raw_hashes_rechecked_after_replay':False},
 'wrong_original_manifest':{'artifact_manifest_sha256':'0'*64},
 'wrong_source_archive':{'source_archive_sha256':'0'*64},
 'wrong_source_inventory':{'source_sha256':{'environment.py':'b'*64}}}
 verification.update(changes.get(case,{}));write_json(folder/'verification.json',verification)
 return protected

class InterceptedUnlink(RuntimeError):pass
rows=[]
cases=['running_metadata','failed_branch','failed_verification','partial_verification','partial_cap','eleven_checked',
 'raw_not_rechecked','wrong_original_manifest','wrong_source_archive','wrong_source_inventory',
 'policy_disallows_removal','wrong_keep_route','extra_mesh_array','missing_protected_raw','valid_until_first_unlink']
with tempfile.TemporaryDirectory(prefix='response-v7-compaction-readonly-review-') as temporary:
 for case in cases:
  folder=Path(temporary)/case;protected=fixture(folder,case)
  before={str(p.relative_to(folder)):mod.sha(p) for p in folder.rglob('*') if p.is_file()}
  attempts=[]
  def intercept(path,*args,**kwargs):
   attempts.append(str(path.relative_to(folder)))
   raise InterceptedUnlink('intercepted before any actual filesystem unlink')
  exception=None
  try:
   with patch.object(Path,'unlink',intercept),contextlib.redirect_stdout(io.StringIO()):mod.compact(folder)
  except Exception as e:exception=type(e).__name__+': '+str(e)
  unchanged=all((folder/name).exists() and mod.sha(folder/name)==value for name,value in before.items())
  if case=='valid_until_first_unlink':
   assert len(attempts)==1 and exception.startswith('InterceptedUnlink')
   record=json.loads((folder/'compaction_manifest.json').read_text())
   assert record['status']=='planned' and set(record['meshes'])==mod.REMOVABLE and record['deleted']==[]
   try:mod.validate_run_assets(folder)
   except AssertionError:partial_rejected=True
   else:partial_rejected=False
   assert partial_rejected
   durable=all(set(r['arrays'])=={'vertices','triangles'} for r in record['meshes'].values())
  else:
   assert exception is not None and not attempts and not (folder/'compaction_manifest.json').exists(),(case,exception,attempts)
   partial_rejected=None;durable=None
  assert unchanged
  rows.append(dict(case=case,status='passed',exception=exception,unlink_attempts_intercepted=attempts,
   actual_unlinks=0,all_original_fixture_files_unchanged=unchanged,planned_record_rejected=partial_rejected,
   all_twenty_array_identities_saved_before_first_unlink=durable))
 # Optimized Python is rejected before argument processing and before any data access.
 process=subprocess.run([sys.executable,'-O',str(SOURCE),'--run',str(Path(temporary)/'nonexistent')],text=True,capture_output=True)
 assert process.returncode!=0 and 'require enabled assertions' in process.stderr
 rows.append(dict(case='python_optimized_mode',status='passed',returncode=process.returncode,
                  guard_error='require enabled assertions',actual_unlinks=0))
assert mod.sha(SOURCE)==initial_source_sha
report={'status':'passed_all_preflight_checks','compactor_sha256':initial_source_sha,'cases':rows,
 'original_run_access':'no compact() call on original run; no original outcome values read',
 'actual_mesh_files_removed':0,'synthetic_scope':'tiny in-memory-created mesh fixtures; all unlink calls intercepted before filesystem action',
 'post_compaction_replay_not_tested':True}
(OUT/'preflight_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'cases':len(rows),'actual_mesh_files_removed':0,'source_sha256':initial_source_sha}))
