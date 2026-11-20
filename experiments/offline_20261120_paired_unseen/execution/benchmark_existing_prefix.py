"""One preselected old L54 route, performance only; no new-layout ranking."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
import sys,json,hashlib,time,importlib.util
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1]
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
a=load('B11_performance_old_prefix',BASE/'dual_template_gain_alignment_20261120/execution_audit/prefix_adapter.py')
t=time.monotonic();c=a.rebuild_observed_prefix('shelf_left',54)
m=load('B11_performance_bank',BASE/'dual_template_uncertainty_20261120/method/model.py')
r=load('B11_performance_reweight',BASE/'dual_template_followup_20261120/structure/replay.py')
from full_posterior_forecast import FullPosteriorForecast
bank=m.build_particle_bank(c.mapper);models={}
for arm in ('G','S'):
 models[arm]=m.FullTemplateModel(c.mapper,semantic=arm=='S',sigma_m=.01,extent_mode='uncertainty',bank=bank);r.reweight(models[arm],16.,None)
e=FullPosteriorForecast(c.mapper,models,c.acquired_frames)
route=c.routes[0];start=time.monotonic();result=e.acquisition(route['outbound_states'][1:]);elapsed=time.monotonic()-start
receipt=e.assert_unchanged();receipt.update(status='completed_existing_prefix_performance_only',new_layouts_read=False,candidate_id=int(route['candidate_id']),candidate_selection='original first candidate before any new scoring; no outcome comparison',outbound_frames=len(route['outbound_states'])-1,forecast_seconds=elapsed,total_seconds=time.monotonic()-t,result_integrity_sha256=hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest(),positive_terms=len(result['terms']),forecast_gain_not_exported_for_selection=True,engine_sha256=sha(HERE/'full_posterior_forecast.py'),script_sha256=sha(Path(__file__)))
(HERE/'performance_existing_prefix.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
