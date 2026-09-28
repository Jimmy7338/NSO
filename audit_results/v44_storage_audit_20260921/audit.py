"""Read-only V44 storage audit. Writes only this audit's receipt; no cleanup."""
from pathlib import Path
import hashlib,json,os,subprocess,sys
ROOT=Path('/root/NSO');sys.path.insert(0,str(ROOT))
from env.development_sensor_v41 import storage_report_v41
import numpy,scipy,open3d,matplotlib,shapely
out=ROOT/'audit_results/v44_storage_audit_20260921'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def rec(path,reason,source=None):
 s=path.stat();d=dict(path=str(path),bytes=s.st_size,allocated_bytes=s.st_blocks*512,hard_links=s.st_nlink,sha256=sha(path),reason=reason)
 if source is not None:d.update(source=str(source),source_sha256=sha(source))
 return d
candidates=[];excluded=[]
for scope in ['nso','env','scripts','tests']:
 for p in sorted((ROOT/scope).rglob('*.pyc')):
  if p.is_symlink() or not p.is_file():continue
  src=p.parent.parent/(p.name.split('.cpython-')[0]+'.py')
  if not src.is_file():excluded.append(dict(path=str(p),reason='source not established; retain'));continue
  if subprocess.run(['git','ls-files','--error-unmatch',str(p.relative_to(ROOT))],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:
   excluded.append(dict(path=str(p),reason='tracked file; retain'));continue
  candidates.append(rec(p,'untracked regenerable CPython bytecode; corresponding source exists',src))
for name in ['nso-v38-qualitative-mpl','nso-v40-mpl','nso-v42-mpl','nso-v43-mpl','nso_v34_mpl','nso_v35_matplotlib','nso_v36_matplotlib','nso_v38_mpl','nso_v39_mpl']:
 p=Path('/tmp')/name/'fontlist-v330.json'
 if p.is_file():
  try:d=json.loads(p.read_text())
  except (ValueError,OSError):continue
  if '_version' in d and 'ttflist' in d:candidates.append(rec(p,'NSO-named Matplotlib generated font cache; regenerable; no experimental observations'))
link=ROOT/'.venv-3d/lib/python3.12/site-packages'; old=json.loads((ROOT/'audit_results/v39_storage_20260920/python_runtime_to_ram.json').read_text())
result=dict(scope='read-only audit; no deletion, installation, download, World creation, or experimental data changes',date='2026-09-21',resource=storage_report_v41(out,402653184),
 sandbox_visibility_warning='ordinary sandbox /dev/shm is a different empty mount; host escalated read confirms live dependency target',
 dependencies=dict(venv_link=str(link),link_target=os.readlink(link),host_target_exists=link.exists(),versions={k:v.__version__ for k,v in [('numpy',numpy),('scipy',scipy),('open3d',open3d),('matplotlib',matplotlib),('shapely',shapely)]},restore_needed=False,
 historical_recovery_inputs=['requirements-3d-v23.txt','requirements-3d.txt','requirements-cpu.txt','requirements-3d.lock.txt'],
 existing_persistent_cpu_alternative='.venv-cpu (numpy 1.26.4/scipy 1.11.4/matplotlib 3.8.4; not a full Open3D replacement)',
 historical_ram_move_receipt='audit_results/v39_storage_20260920/python_runtime_to_ram.json',historical_ram_move_verified_files=len(old['verified_files']),historical_ram_move_payload_bytes=sum(x['bytes'] for x in old['verified_files'].values()),local_wheels_found_in_project_tmp_rootcache=0),
 candidates=candidates,candidate_file_count=len(candidates),candidate_logical_bytes=sum(x['bytes'] for x in candidates),candidate_allocated_single_link_bytes=sum(x['allocated_bytes'] for x in candidates if x['hard_links']==1),
 excluded_candidates=excluded,excluded_scopes={'.git':'repository history, no pruning authorized by this audit','eval_results':'original or sealed experiment evidence; not cache','audit_results':'original audit/evidence/archive material; not cache','third_party':'baseline source dependencies, not disposable build directories','/tmp non-NSO trees':'scanner/OTA/field-trace project data outside this task','/root/.cache':'only small font/other-project state; no large pip wheel cache'},
 gate_formula='max(10 * 1024**3, 2 * expected_batch_peak_bytes + 2 * 1024**3)',gate_source=dict(path='env/development_sensor_v41.py',sha256=sha(ROOT/'env/development_sensor_v41.py'),start_line=357),runtime_protocol=dict(path='configs/virtual3d/v43_runtime_protocol_20260921.json',sha256=sha(ROOT/'configs/virtual3d/v43_runtime_protocol_20260921.json')))
result['shortfall_before_cleanup_bytes']=max(0,result['resource']['required_free_bytes']-result['resource']['free_bytes'])
result['shortfall_after_candidate_file_blocks_bytes']=max(0,result['shortfall_before_cleanup_bytes']-result['candidate_allocated_single_link_bytes'])
result['candidate_cleanup_sufficient_to_pass_gate']=result['shortfall_after_candidate_file_blocks_bytes']==0
(out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['resource','candidate_file_count','candidate_logical_bytes','candidate_allocated_single_link_bytes','shortfall_before_cleanup_bytes','candidate_cleanup_sufficient_to_pass_gate']},indent=2))
