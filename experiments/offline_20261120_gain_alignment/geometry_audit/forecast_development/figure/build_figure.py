"""All 36 predeclared independent-test routes; no result-based selection."""
from pathlib import Path
import hashlib,json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
path=ROOT/'independent_test_attempt_001/result.json';result=json.loads(path.read_text());assert result['independent_objects']==12 and result['dependent_route_rows']==36
rows=[]
for obj in result['objects']:
 for r in obj['candidates']:
  rows.append(dict(object_id=obj['object_id'],family=obj['family'],route=r['route'],actual_gross_m2=r['actual_source_gross_new_m2'],actual_lost_m2=r['actual_source_lost_m2'],actual_net_m2=r['actual_source_net_new_m2'],old_fine_prediction_m2=r['matched_fine_original_visibility_m2'],native_fine_prediction_m2=r['matched_fine_native_visibility_m2'],forecast_gross_m2=r['forecast']['predicted_gross_new_m2'],forecast_lost_m2=r['forecast']['predicted_lost_m2'],forecast_net_m2=r['forecast']['predicted_net_new_m2'],old_absolute_error_m2=r['fine_original_absolute_error_m2'],forecast_absolute_error_m2=r['forward_absolute_error_m2'],forecast_wall_s=r['forecast']['elapsed_s']))
def csvwrite(name,rr):
 with(HERE/name).open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
csvwrite('all_test_routes.csv',rows)
objects=[]
for o in result['objects']:
 rr=[r for r in rows if r['object_id']==o['object_id']];objects.append(dict(object_id=o['object_id'],family=o['family'],old_mae_m2=float(np.mean([r['old_absolute_error_m2']for r in rr])),forecast_mae_m2=float(np.mean([r['forecast_absolute_error_m2']for r in rr]))))
csvwrite('paired_object_mae.csv',objects)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8.5,'axes.titlesize':9,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(6.95,3.35),sharex=True,sharey=True)
colors={'revisit':'#009E73','side':'#0072B2','around':'#D55E00'};markers={'revisit':'s','side':'o','around':'^'}
maximum=max(max(r['actual_gross_m2'],r['old_fine_prediction_m2'],r['forecast_gross_m2'])for r in rows);lim=np.ceil(maximum*1.05*10)/10
for ax,key,label in zip(axes,['old_fine_prediction_m2','forecast_gross_m2'],['(a) Point visibility, matched fine basis','(b) Acquisition forward estimate']):
 ax.plot([0,lim],[0,lim],color='#777777',ls='--',lw=.85,zorder=0)
 for route in ['revisit','side','around']:
  rr=[r for r in rows if r['route']==route];ax.scatter([r['actual_gross_m2']for r in rr],[r[key]for r in rr],c=colors[route],marker=markers[route],s=25,alpha=.82,linewidth=.35,edgecolor='white',label={'revisit':'Revisit','side':'Side view','around':'Around object'}[route],clip_on=False)
 mae=np.mean([abs(r[key]-r['actual_gross_m2'])for r in rows]);ax.text(.04,.94,f'MAE = {mae:.4f} m²',transform=ax.transAxes,va='top',fontsize=8.5)
 ax.set_title(label,pad=8);ax.set_xlabel('Actual newly reconstructed surface (m²)');ax.set_xlim(-.035,lim);ax.set_ylim(-.035,lim);ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.16,lw=.5);ax.set_axisbelow(True)
axes[0].set_ylabel('Predicted newly reconstructed surface (m²)')
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.005));fig.subplots_adjust(left=.09,right=.985,bottom=.22,top=.88,wspace=.20)
for suffix in ['pdf','svg','png']:fig.savefig(HERE/('known_geometry_acquisition.'+suffix),dpi=220)
caption='Known-geometry acquisition consistency on 12 newly generated objects (6 closed boxes and 6 open shelves), each with three predeclared routes. The 36 route measurements are dependent within object. Both estimators use the same hypothesis-owned 0.06 m external-visible quadrature and preserve observed free-space rejection. The forward estimate replays acquired past depth and hypothesis-rendered future depth into a separate TSDF, then counts source-independent template surface samples newly within 5 cm of the predicted mesh. Actual gross newly reconstructed area uses independent source-surface Monte Carlo samples. Every route is included; the 12 revisit measurements overlap at the origin in panel (b). MAE falls from 0.1532 to 0.0143 m², with one route having slightly greater error. The hypothesis equals the generating geometry in this controlled probe: matching predicted/actual meshes tests acquisition consistency, not geometry inference, semantic efficiency, or unknown-environment policy performance. Native-only replacement on the same basis has MAE 0.2228 m² (all values and paired object errors are supplied in CSV).'
(HERE/'CAPTION.md').write_text(caption+'\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
summary=dict(status='complete_all_predeclared_test_routes_figure',source=dict(path=str(path),sha256=sha(path)),objects=12,dependent_routes=36,source_rows_unchanged=True,old_mae_m2=float(np.mean([r['old_absolute_error_m2']for r in rows])),forward_mae_m2=float(np.mean([r['forecast_absolute_error_m2']for r in rows])),rows_improved=sum(r['forecast_absolute_error_m2']<r['old_absolute_error_m2']-1e-12 for r in rows),rows_worsened=sum(r['forecast_absolute_error_m2']>r['old_absolute_error_m2']+1e-12 for r in rows),rows_tied=sum(abs(r['forecast_absolute_error_m2']-r['old_absolute_error_m2'])<=1e-12 for r in rows),worsened_rows=[r for r in rows if r['forecast_absolute_error_m2']>r['old_absolute_error_m2']+1e-12],claims=dict(acquisition_oracle_consistency=True,fitted_geometry_validated=False,unknown_exploration_efficiency_validated=False),artifacts={p.name:dict(bytes=p.stat().st_size,sha256=sha(p))for p in HERE.iterdir()if p.is_file()and p.name!='validation.json'})
(HERE/'validation.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:summary[k]for k in ['status','objects','dependent_routes','old_mae_m2','forward_mae_m2','rows_improved','rows_worsened','rows_tied']}))
