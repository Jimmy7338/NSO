#!/usr/bin/env python3
"""Read four sealed meshes only; show all vertices, no new outline/Q or fusion."""
import hashlib
import io
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/dev/shm/nso_v25_figure_mpl')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
CASE=ROOT/'audit_results/facility_choice_v25r1_paid_p00_20260915/case_00'
CAP=2*1024*1024
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
sealed=read(CASE/'artifact_hashes.json')
sources={}
def checked(name):
    p=CASE/name
    assert sha(p)==sealed[name],name
    sources[str(p.relative_to(ROOT))]=sha(p)
    return p
r=read(checked('result.json'))
v=read(checked('verification.json'))
assert r['status']=='complete' and r['assignment']=='A_complex_B_simple' and r['arm']=='A'
assert r['paid_actions']==354 and r['stages']['prefix']['action_id']==234
assert v['status']=='passed' and v['independent_process']
assert v['all_raw_and_observed_mesh_arrays_equal'] and v['all_frozen_metrics_equal']
assert r['inference_disabled'] and not r['autonomous_planner']
meshes={};counts={}
for stage in ('prefix','final'):
    for slot in (0,1):
        rel=f'meshes/{stage}_slot_{slot}_observed_mesh.npz'
        with np.load(checked(rel),allow_pickle=False) as arrays:
            vv=arrays['vertices'].copy();tt=arrays['triangles']
            assert np.isfinite(vv).all()
            meshes[stage,slot]=vv
            counts[f'{stage}_slot_{slot}']={'vertices':len(vv),'triangles':len(tt),
                'minimum_xyz_m':vv.min(0).tolist(),'maximum_xyz_m':vv.max(0).tolist()}
# Fixed physical units and the same limits for both stages. No point is cropped.
limits=[((2.8,7.2),(4.4,8.5),(-.2,2.7)),((9.0,13.4),(4.4,8.5),(-.2,2.7))]
projections=[('XY plan',0,1),('XZ elevation',0,2),('YZ elevation',1,2)]
colors={'prefix':'#226bb4','final':'#d26818'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
fig,axs=plt.subplots(2,3,figsize=(12.6,8.7),gridspec_kw={'height_ratios':[1,1]})
fig.subplots_adjust(left=.065,right=.985,bottom=.13,top=.865,wspace=.28,hspace=.30)
fig.suptitle('Case 00: saved observed-mesh projections before and after A acquisition',fontsize=15,weight='bold',y=.976)
fig.text(.5,.937,'All saved vertices; identical before/after scales; no filtering, inferred surfaces or new outline evaluation',ha='center',fontsize=10,color='#3d4f5d')
for slot in (0,1):
    for j,(name,d1,d2) in enumerate(projections):
        ax=axs[slot,j]
        for stage in ('prefix','final'):
            vv=meshes[stage,slot]
            assert (vv[:,d1]>=limits[slot][d1][0]).all() and (vv[:,d1]<=limits[slot][d1][1]).all()
            assert (vv[:,d2]>=limits[slot][d2][0]).all() and (vv[:,d2]<=limits[slot][d2][1]).all()
            ax.scatter(vv[:,d1],vv[:,d2],s=.62,linewidths=0,c=colors[stage],alpha=.43,rasterized=True)
        ax.set(xlim=limits[slot][d1],ylim=limits[slot][d2],xlabel='xyz'[d1]+' (m)',ylabel='xyz'[d2]+' (m)')
        ax.set_aspect('equal',adjustable='box');ax.grid(color='#d7dde1',lw=.5,alpha=.8);ax.set_axisbelow(True)
        label='Slot 0 / A side (complex)' if slot==0 else 'Slot 1 / B side (simple)'
        ax.set_title(label+'\n'+name,fontsize=10.2,pad=7)
        ax.spines[['top','right']].set_visible(False)
fig.legend(handles=[Line2D([],[],ls='',marker='o',ms=5,color=colors['prefix'],label='After common prefix: action 234'),
                    Line2D([],[],ls='',marker='o',ms=5,color=colors['final'],label='After A acquisition + return: action 354')],
           loc='center',bbox_to_anchor=(.5,.083),ncol=2,frameon=False,fontsize=10)
fig.text(.5,.033,'Single completed fixed-route case only. Point projections are not the evaluator\'s filled exterior outlines and do not establish a failure cause.',ha='center',fontsize=8.7,color='#334854')
blobs={}
for ext in ('png','pdf'):
    stream=io.BytesIO();fig.savefig(stream,format=ext,dpi=145,facecolor='white',metadata={'Title':'Case 00 observed mesh vertex projections'})
    blobs[ext]=stream.getvalue()
plt.close(fig)
other=sum(p.stat().st_size for p in OUT.glob('p00_experiment_design.*') if p.suffix in ('.png','.pdf'))
assert other+sum(map(len,blobs.values()))<CAP
sv=os.statvfs(OUT)
assert sv.f_bavail*sv.f_frsize-sum(map(len,blobs.values()))>=32*1024*1024
for ext,blob in blobs.items():(OUT/f'case00_observed_projections.{ext}').write_bytes(blob)
receipt={'status':'complete_read_only_vertex_projection','source_sha256':sources,'case':str(CASE.relative_to(ROOT)),
    'case_status':r['status'],'case_replay_status':v['status'],'other_cases_read':False,
    'new_world_instances':0,'physical_actions':0,'sensor_calls':0,'tsdf_fusions':0,'new_quality_evaluations':0,
    'plotted':'Every observed-mesh vertex in each orthogonal projection; no subsampling, clipping, triangle union or inferred geometry',
    'counts_and_extents':counts,'axis_limits':limits,'axis_units':'metres, equal aspect; unchanged between stages',
    'existing_scores_not_recomputed':{stage:{str(item['id']):item['projections']['xy']['iou'] for item in r['stages'][stage]['main_observed']['instances']} for stage in ('prefix','final')},
    'limitations':['Full observed-slot mesh is shown; evaluator scores use its already frozen task windows.',
        'Mesh extends beyond the cabinet body; plotted points alone do not diagnose why.',
        'This case cannot establish the four-case matrix or semantic policy effectiveness.'],
    'figure_bytes':{ext:len(b) for ext,b in blobs.items()},'combined_all_figure_bytes':other+sum(map(len,blobs.values()))}
(OUT/'case00_read_only_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'status':receipt['status'],'combined_all_figure_bytes':receipt['combined_all_figure_bytes'],'physical_actions':0}))
