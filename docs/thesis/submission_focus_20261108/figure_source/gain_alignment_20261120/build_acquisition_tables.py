from pathlib import Path
import json,hashlib,csv
from build_gain_tables import table
P=Path(__file__).resolve().parent; B=P.parent; inputs={}
def read(p):inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
F=B/'geometry_audit/forecast_development'
data={k:read(F/(k+'_attempt_001/result.json'))for k in ('development','independent_test')}
fitted=read(B/'policy_audit/fitted_C_attempt_001/summary.json'); allrows=[]
for split,s in data.items():
 for obj in s['objects']:
  for r in obj['candidates']:
   for mode in ('fine_original','fine_native','forward'):
    allrows.append(dict(split=split,object_id=obj['object_id'],family=obj['family'],route=r['route'],mode=mode,actual_gross_m2=r['actual_source_gross_new_m2'],actual_lost_m2=r['actual_source_lost_m2'],actual_net_m2=r['actual_source_net_new_m2'],absolute_error_m2=r[mode+'_absolute_error_m2'],signed_error_m2=r[mode+'_signed_error_m2']))
with(P/'acquisition_fixture_values.csv').open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
rows=[]
for mode,label in [('fine_original','Visibility'),('forward','Acquisition forecast')]:
 s=data['independent_test']['summary'][mode];rows.append(['Known geometry',label,f"{s['mean_absolute_error_m2']:.4f}",'--','--','--'])
for mode,label in [('original_T16','Original distribution'),('fine_original','Mean + fine visibility'),('fine_native','Mean + native visibility'),('forward','Mean + acquisition forecast')]:
 g=fitted['groups'][mode+'_G'];s=fitted['groups'][mode+'_S'];rows.append(['Fitted geometry',label,f"{g['MAE_vs_object_gross_m2']:.4f}",f"{s['MAE_vs_object_gross_m2']:.4f}",f"{g['template_rho_vs_object_gross_m2']:.3f}",f"{s['template_rho_vs_object_gross_m2']:.3f}"])
table(P/'acquisition_main_table.tex','Prediction versus actual object gross-new area at 5\\,cm. Known-geometry rows average three dependent routes on each of twelve new objects; their single error has no G/S prior. Fitted rows use all eleven original candidates at one retained prefix. The last three share conditional-mean shapes and a 0.06\\,m quadrature; the original distribution also differs in geometric integration and basis. MAE is in m$^2$; $\\rho$ is candidate Spearman correlation. No complete policy uses the acquisition forecast.','tab:acquisition-alignment','llrrrr','Geometry & Estimator & MAE (G) & MAE (S) & $\\rho_G$ & $\\rho_S$',rows)
rows=[]
for split,label in [('development','Development'),('independent_test','New test')]:
 for mode,title in [('fine_original','Visibility / old history'),('fine_native','Visibility / native replacement'),('forward','Acquisition forecast')]:
  s=data[split]; rows.append([label,title,f"{s['summary'][mode]['mean_absolute_error_m2']:.4f}",*[f"{s['by_route'][r][mode]['mean_absolute_error_m2']:.4f}"for r in ('revisit','side','around')],f"{s['summary'][mode]['mean_signed_error_m2']:+.4f}"])
table(P/'acquisition_fixture_table.tex','Matched 0.06\\,m surface-basis probes: twelve objects per split, six closed and six open, with three dependent routes each. Units are m$^2$. Native replacement substitutes a 5\\,cm native-mesh consumption rule for the old 10\\,cm measured-point rule; it is not the additive mask used in the complete-policy ablation. All predictors use the same known geometry and source evaluation.','tab:acquisition-fixtures','llrrrrr','Split & Estimator & All MAE & Revisit & Side & Around & Bias',rows)
rows=[]
for mode,label in [('original_T16','Distribution'),('fine_original','Mean / old history'),('fine_native','Mean / native replacement'),('forward','Mean / acquisition')]:
 for arm in ('G','S'):
  r=fitted['groups'][mode+'_'+arm];a=r['selected_actual'];rows.append([label,arm,f"{r['MAE_vs_object_gross_m2']:.4f}",f"{r['MAE_vs_object_net_m2']:.4f}",f"{r['template_rho_vs_object_gross_m2']:.3f}",str(r['selected_candidate']),f"{a['object_net_m2']:.4f}",f"{a['global_net_m2']:.3f}",str(a['outbound_actions'])])
table(P/'acquisition_fitted_table.tex','All eleven reused actual branch outcomes at left-prefix action 54. Gross prediction error and its mismatch to actual net area are both retained; the latter does not redefine the score. Selected object/global net area is measured independently. All mean/fine variants select candidate 8; selection already changes before replacing visibility with acquisition forecasting.','tab:acquisition-fitted','llrrrrrrr','Estimator & Prior & Gross MAE & Net mismatch & $\\rho$ & Choice & Obj. net & Global net & Actions',rows)
fitted_table=P/'acquisition_fitted_table.tex'
fitted_table.write_text(fitted_table.read_text().replace('\\centering\\small', '\\centering\\small\\setlength{\\tabcolsep}{3pt}'))
(P/'acquisition_table_sources.json').write_text(json.dumps(dict(status='all_completed_results_table_generation',inputs=inputs,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
