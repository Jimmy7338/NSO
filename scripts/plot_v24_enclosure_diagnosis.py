#!/usr/bin/env python3
"""Static evaluator-only visualization of an already frozen diagnostic."""
import hashlib
import itertools
import json
from pathlib import Path
import os
os.environ['MPLCONFIGDIR']='/dev/shm/nso_v24_diag_matplotlib'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
root=Path(__file__).resolve().parents[1]
source=root/'audit_results/v24_enclosure_free_ray_diagnosis_20260915/result.json'
result=json.loads(source.read_text())
output=root/'audit_results/v24_enclosure_diagnosis_figures_20260915'
if output.exists():
    raise ValueError('Preserve previous figures; use a fresh output')
output.mkdir()
fig,axes=plt.subplots(1,2,figsize=(10.4,4.6),layout='constrained')
for ax,row in zip(axes,result['case00']):
    diagnoses=row['diagnostics']; truth=diagnoses['gt_body_exact']
    tlo,thi=np.asarray(truth['lo']),np.asarray(truth['hi'])
    origin=(tlo[:2]+thi[:2])/2
    for key,label,color in [('gt_body_exact','Actual body (evaluation only)','#2468a0'),
                            ('fitted','V24 inferred box: rejected','#d35c32')]:
        box=diagnoses[key];lo,hi=np.asarray(box['lo']),np.asarray(box['hi']);basis=np.asarray(box['basis'])
        points=np.asarray([[lo[0],lo[1],lo[2]],[hi[0],lo[1],lo[2]],
                           [hi[0],hi[1],lo[2]],[lo[0],hi[1],lo[2]]])@basis.T
        points=points[:,:2]-origin
        ax.add_patch(Polygon(points,closed=True,edgecolor=color,facecolor=color,alpha=.15))
        points=np.vstack([points,points[0]])
        ax.plot(points[:,0],points[:,1],color=color,lw=2,label=label)
    ax.set_title('Observed instance '+str(row['slot']))
    ax.set_aspect('equal');ax.autoscale_view();ax.margins(.25)
    ax.set_xlabel('World x relative to body centre (m)');ax.set_ylabel('World y relative to body centre (m)')
    ax.grid(alpha=.2)
    fit=diagnoses['fitted']['ray_test']['fraction']*100
    gt=truth['ray_test']['fraction']*100
    ax.text(.03,.97,f'Free-ray conflicts: fitted {fit:.2f}%\nExact-body control {gt:.3f}%',
            transform=ax.transAxes,va='top',fontsize=10,bbox=dict(facecolor='white',edgecolor='none',alpha=.85))
axes[0].legend(loc='lower center',fontsize=8)
fig.suptitle('Measured depth rejects an overexpanded fitted envelope',fontsize=14)
fig.savefig(output/'top_view.png',dpi=170)
plt.close(fig)
(output/'provenance.json').write_text(json.dumps(dict(input=str(source.relative_to(root)),
    input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    evaluation_control_only=True,new_model_result=False,new_sensor_frames=0),indent=2)+'\n')
(output/'artifact_hashes.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(output.iterdir())},indent=2)+'\n')
print(output/'top_view.png')
