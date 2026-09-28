#!/usr/bin/env python3
"""All declared actual second views: physical changes, no policy ranking."""
import argparse,csv,gzip,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,sort_keys=True,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    m=json.loads((a.source/'manifest.json').read_text())
    if m['status']!='complete':raise ValueError('complete probes required')
    for n,v in json.loads((a.source/'artifact_hashes.json').read_text()).items():
        if sha(a.source/n)!=v:raise ValueError('input changed')
    a.output.mkdir(parents=True,exist_ok=False);records=[];plots=[];cases=[]
    keys=('coverage_2d','precision_05cm','recall_05cm','f1_05cm','joint_05cm','complex_recall_05cm',
          'simple_recall_05cm','surface_error_mean_m','surface_error_p95_m')
    for d in m['protocol']['declarations']:
        first,second=d['first_candidate']['candidate_id'],d['second_candidate']['candidate_id']
        folder=a.source/f'first_{first}_second_{second}';r=json.loads((folder/'result.json').read_text())
        with np.load(folder/'visibility.npz',allow_pickle=False) as v:
            novel=v['arrival']&~v['first_arrival'];weight=float(v['sample_weight_m2'])
            complex_area=float(np.count_nonzero(novel&(v['classes']==3)))*weight
            simple_area=float(np.count_nonzero(novel&(v['classes']==2)))*weight
        calls=json.loads(gzip.decompress((folder/'module_calls.json.gz').read_bytes()))
        observed=next(x['outputs'] for x in calls if x['module']=='OV-SDF' and x['action_id']==d['arrival_action_id'])
        target=observed['measured_assets'][d['second_candidate']['asset_index']]
        cases.append(dict(first_candidate_id=first,second_candidate_id=second,reached=r['target_reached'],
            target_observed_class_vote=target['class_vote'],target_marked_points=target['marked_points'],
            observed_semantic_convention='open negative, closed positive; unmarked is not known closed',
            returned=r['termination']['returned_to_anchor'],second_view_actions=r['second_view_actions'],
            total_paid_actions=r['total_paid_actions'],new_support_keys=r['second_view_new_support_keys'],
            new_direction_bits=r['second_view_new_directions_on_initial_support'],
            new_visible_surface_m2=r['second_view_new_visible_surface_m2'],
            new_visible_complex_surface_m2=complex_area,new_visible_simple_surface_m2=simple_area,
            depth_supported_initial_points=r['actual_second_arrival_depth_support_on_initial_face'],
            new_entry_deep_candidate_count=r['post_arrival_back_candidates']))
        for seed in m['protocol']['evaluation']['reference_seeds']:
            before,after=r['first_arrival'][str(seed)],r['arrival'][str(seed)]
            row=dict(first_candidate_id=first,second_candidate_id=second,reference_seed=seed,
                actual_second_actions=r['second_view_actions'],total_paid_actions=r['total_paid_actions'])
            for k in keys:
                row['before_'+k]=before[k];row['after_'+k]=after[k]
                row['change_'+k]=None if before[k] is None or after[k] is None else after[k]-before[k]
            records.append(row)
        plots.append((f'{first} → {second}',[x for x in records if x['first_candidate_id']==first and x['second_candidate_id']==second]))
    with (a.output/'physical_changes.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    write(a.output/'result.json',dict(status='complete',cases=cases,physical_probes=len(cases),
        reference_metric_rows=len(records),independent_parent_layouts=1,semantic_efficacy_proven=False,
        full_budget_policy_comparison=False,training_allowed=False,
        input_inventory_sha256=sha(a.source/'artifact_hashes.json'),source_sha256=sha(Path(__file__))))
    fig,ax=plt.subplots(figsize=(10,4.6),layout='constrained');x=np.arange(len(plots));width=.25
    for offset,key,label,color in [(-width,'coverage_2d','2D coverage','#44759f'),(0,'f1_05cm','Global F1@5cm','#df932c'),
                                   (width,'complex_recall_05cm','Complex-surface recall','#44805c')]:
        vals=np.array([[r['change_'+key]*100 for r in rows] for _,rows in plots])
        center=vals[:,0];err=np.array([center-vals.min(axis=1),vals.max(axis=1)-center])
        ax.bar(x+offset,center,width,color=color,label=label,yerr=err,capsize=2)
    ax.set_xticks(x,[n for n,_ in plots]);ax.set_xlabel('Declared first approach → actual second view')
    ax.set_ylabel('Change during second view (percentage points)');ax.axhline(0,color='#888',lw=.8)
    ax.legend(loc='upper left');ax.set_title('Second-view observations; unequal costs, not a policy comparison\nBars: reference 2026; whiskers: range across three reference samplings')
    fig.savefig(a.output/'second_view_changes.png',dpi=160,bbox_inches='tight')
    fig.savefig(a.output/'second_view_changes.svg',bbox_inches='tight')
    write(a.output/'artifact_hashes.json',{x.name:sha(x) for x in sorted(a.output.iterdir()) if x.is_file()})


if __name__=='__main__':main()
