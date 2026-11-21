#!/usr/bin/env python3
"""Independent frozen-output checks. No actor or additional acquisition."""
import hashlib,json
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parents[1]; D=B/'geometry_audit/forecast_development'; H=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def resource(p):return dict(path=str(p),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def main():
 protocol=read(D/'protocol.json')
 for p,h in protocol['code_sha256'].items():assert sha(p)==h,p
 results={}
 for split in ('development','independent_test'):
  root=D/(split+'_attempt_001'); data=read(root/'result.json');assert data['protocol_sha256']==sha(D/'protocol.json')
  assert len(data['objects'])==12 and data['independent_objects']==12 and data['dependent_route_rows']==36
  assert data['not_fitted_geometry'] and data['new_unknown_exploration_policy_actions']==0
  flat=[]
  for obj in data['objects']:
   with np.load(root/(obj['object_id']+'_predictions.npz'),allow_pickle=False)as z:
    for row in obj['candidates']:
     name=row['route']; f=row['forecast']; assert row['exact_predicted_and_actual_acquisition_mesh']
     assert abs(z[name+'_forecast_gain'].sum()-f['predicted_gross_new_m2'])<1e-12
     assert abs(z[name+'_forecast_loss'].sum()-f['predicted_lost_m2'])<1e-12
     assert abs(row['actual_source_gross_new_m2']-row['actual_source_lost_m2']-row['actual_source_net_new_m2'])<1e-12
     for label,key in [('fine_original',name+'_matched_fine_original'),('fine_native',name+'_matched_fine_native'),('forward',name+'_forecast_gain')]:
      value=float(z[key].sum()); err=value-row['actual_source_gross_new_m2']
      assert abs(err-row[label+'_signed_error_m2'])<1e-12
      assert abs(abs(err)-row[label+'_absolute_error_m2'])<1e-12
     flat.append(row)
  for label in ('fine_original','fine_native','forward'):
   for metric,field in [('mean_absolute_error_m2','_absolute_error_m2'),('mean_signed_error_m2','_signed_error_m2')]:
    assert abs(np.mean([x[label+field]for x in flat])-data['summary'][label][metric])<1e-12
  diff=np.array([x['forward_absolute_error_m2']-x['fine_original_absolute_error_m2']for x in flat])
  results[split]=dict(source=resource(root/'result.json'),objects=12,dependent_routes=36,summary=data['summary'],by_route=data['by_route'],forward_vs_fine_original=dict(improved=int((diff < -1e-12).sum()),worse=int((diff>1e-12).sum()),tie=int((abs(diff)<=1e-12).sum())),all_exact_acquisition_mesh_checks=True)
 fitted=B/'execution_audit/acquisition_forecast';p=read(fitted/'protocol.json')
 for path,h in p['files_sha256'].items():assert sha(path)==h,path
 write(H/'independent_C_review.json',dict(status='passed_output_arithmetic_source_binding_and_readonly_code_review',known_geometry=results,fitted_wrapper_code_review=dict(status='no_blocking_issue_before_predictions',protocol=resource(fitted/'protocol.json'),wrapper=resource(fitted/'run_fitted.py'),renderer=resource(fitted/'render_observed_free.py'),checks=['All11 original routes/four mutually exclusive conditional-mean hypotheses; no candidate selection before forecasting.','55 original acquired frames replayed;21 original history keyframes; measured meshes unchanged.','G/S same geometry and per-hypothesis forecast, only class-mass difference.','Only non-strict-free own area enters forecast; no repeated10cm measured consumption.','No source reference or actual future depth passed into forecasting; independent comparison follows frozen predictions.','Actual fitted renderer inclusive.15 boundary preserved; toy range distinction disclosed.']),limits=['Known-geometry future exactly equals generator: verifies acquisition/fusion operator with independent area quadrature, not fitted-shape quality or autonomous exploration.','fine_native replaces original10cm history with5cm native consumption on samefine basis; it is not additive native-mask A.','12 objects per split,36 dependent routes; old12 reused for development; all negative rows retained.','Fitted wrapper substitutes per-structure conditional mean for within-shape particle integration and omits unseen unmodeled occluders.'],sources=dict(protocol=resource(D/'protocol.json'),core=resource(D/'acquisition_forecast.py'),basis=resource(D/'surface_basis.py'),wrapper=resource(D/'run.py'),script=resource(__file__))))
 print(json.dumps({'status':'passed','receipt':resource(H/'independent_C_review.json'),'outcomes':{k:v['forward_vs_fine_original']for k,v in results.items()}}))
if __name__=='__main__':main()
