"""Reproducible measured calibration/decision diagnostics; no generated data."""
from pathlib import Path
import csv, hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
B8=ROOT.parent/'dual_template_uncertainty_20261120'
source=ROOT/'structure/audit/test_attempt_001/test_results.json'
decision=ROOT/'structure/replay_attempt_001/shelf_left_54/result.json'
data=json.loads(source.read_text());run=json.loads(decision.read_text())
objects=sorted({r['object_id'] for r in data['rows']});assert len(objects)==24
rows=[];means={}
for mode in ('G','S'):
    for variant in ('T1','T16'):
        values=[]
        for oid in objects:
            source_mode='G' if mode=='G' else 'S_correct_category'
            source_variant='T1' if variant=='T1' else 'selected'
            subset=[r for r in data['rows'] if r['object_id']==oid and r['mode']==source_mode and r['variant']==source_variant]
            assert len(subset)==4
            nll=float(np.mean([r['class_metrics']['nll'] for r in subset]));values.append(nll)
            rows.append(dict(kind='independent_object_NLL',mode=mode,variant=variant,object_or_candidate=oid,value=nll))
        means[mode,variant]=np.asarray(values)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,axes=plt.subplots(1,2,figsize=(10.8,3.7),gridspec_kw={'width_ratios':[1,1.2]})
ax=axes[0];colors={'T1':'#0072B2','T16':'#D55E00'}
for mode,xbase in [('G',0),('S',2.6)]:
    jitter=np.linspace(-.13,.13,len(objects))
    for i,oid in enumerate(objects):
        ax.plot([xbase+jitter[i],xbase+1+jitter[i]],[means[mode,'T1'][i],means[mode,'T16'][i]],color='#9B9B9B',alpha=.45,lw=.65,zorder=1)
    for variant,dx in [('T1',0),('T16',1)]:
        values=means[mode,variant];ax.scatter(xbase+dx+jitter,values,s=15,color=colors[variant],alpha=.8,zorder=2)
        mean=float(values.mean());ax.plot([xbase+dx-.27,xbase+dx+.27],[mean,mean],color='black',lw=1.7,zorder=3)
        ax.annotate(f'{mean:.3f}',(xbase+dx,mean),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
ax.set_xticks([0,1,2.6,3.6],['G\nT = 1','G\nT = 16','S*\nT = 1','S*\nT = 16'])
ax.set_ylabel('Mean NLL per object (nats; lower is better)')
ax.set_ylim(bottom=-.7);ax.grid(axis='y',alpha=.18)
ax.text(0,1.04,'(a) Independent test: 24 objects',transform=ax.transAxes,weight='bold')

ax=axes[1]
for variant,offset in [('T1',-.1),('T16',.1)]:
    for mode in ('G','S'):
        if variant=='T1':
            p=B8/'replay/shelf_left_54_attempt_001'/f'{mode}_uncertainty_predictions.json'
            item=json.loads(p.read_text());score={int(r['candidate_id']):r['scores'][mode] for r in item['routes']}
        else:score={r['candidate_id']:r['score'] for r in run['arms'][mode]['rows']}
        ids=sorted(score);values=[score[c] for c in ids];assert len(ids)==11
        ax.scatter(np.asarray(ids)+offset,values,color=colors[variant],marker='o' if mode=='G'else'^',
                   facecolors='none' if mode=='G'else colors[variant],s=32,label=f'{mode}, T = {1 if variant=="T1" else 16}')
        for cid,val in zip(ids,values):rows.append(dict(kind='fixed_candidate_total_score',mode=mode,variant=variant,object_or_candidate=cid,value=val))
ax.set_xticks(range(11));ax.set_xlabel('Original candidate ID');ax.set_ylabel('Coverage + surface planning score')
ax.grid(axis='y',alpha=.18);ax.legend(loc='lower right',ncol=2,fontsize=8,frameon=False)
ax.text(0,1.04,'(b) Fixed replay: left layout, action 54',transform=ax.transAxes,weight='bold')
fig.tight_layout(pad=1.5,w_pad=2.5)
for suffix in ('pdf','svg','png'):fig.savefig(HERE/f'structure_calibration_decision.{suffix}',dpi=220,bbox_inches='tight')
plt.close(fig)
with (HERE/'plot_data.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
caption=('Structural calibration and a fixed decision diagnostic. (a) Each point is the mean of four dependent conditions for one of 24 independent test objects; gray lines pair the same object, black bars are means. S* receives a supplied correct 9:1 category prior, not an evaluated detector. (b) All 11 original candidates at one repeatedly inspected development state are shown, with unchanged common scores. T=1 selects candidate8 for both G/S; T=16 selects8 for G and0 for S. This single-state divergence is not a claim of exploration benefit; complete paired rollouts and adverse outcomes are reported separately. T was selected using separate24 calibration objects. No uncertainty bars or independent-run claims are inferred from repeated views.')
(HERE/'CAPTION.txt').write_text(caption+'\n')
sources=[source,decision,B8/'replay/shelf_left_54_attempt_001/G_uncertainty_predictions.json',B8/'replay/shelf_left_54_attempt_001/S_uncertainty_predictions.json']
(HERE/'validation.json').write_text(json.dumps({'status':'generated_from_complete_frozen_records','independent_test_objects':24,'conditions_per_object':4,'paired_object_values':96,'candidate_scores':44,'sources_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
print(json.dumps({'status':'complete','rows':len(rows),'directory':str(HERE)}))
