#!/usr/bin/env python3
"""Algebraic gain audit and complete outcome table; no new model or trials."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def main():
    summary=read(HERE/'audit_attempt_001/summary.json');table=[];details=[];maxerror=0.
    for row in summary['conditions']:
        layout,T,cue=row['layout'],row['temperature'],row['cue'];stem=f'T{T}_{cue}_predictions'
        receipt=read(HERE/'candidate_attempt_002'/layout/(stem+'.json'))
        with np.load(HERE/'candidate_attempt_002'/layout/(stem+'.npz'),allow_pickle=False)as a:
            np.testing.assert_array_equal(a['G_points'],a['S_points'])
            np.testing.assert_array_equal(a['G_object_ids'],a['S_object_ids'])
            for i,c in enumerate(receipt['candidates']):
                reconstructed=0.;parts=[]
                for oi,(g,s)in enumerate(zip(receipt['model_audits']['G']['objects'],receipt['model_audits']['S']['objects'])):
                    mask=a['G_object_ids']==oi;pG=np.array(g['shape_probability_mass']);pS=np.array(s['shape_probability_mass'])
                    conditional=[]
                    for h in (0,1):
                        which=mask&(a['G_shape_ids']==h)
                        v=float(a['G_route_union_gain_matrix'][i,which].sum()/pG[h])
                        np.testing.assert_allclose(v,a['S_route_union_gain_matrix'][i,which].sum()/pS[h],rtol=1e-9,atol=1e-10)
                        conditional.append(v)
                    delta=float((pS[1]-pG[1])*(conditional[1]-conditional[0]));reconstructed+=delta
                    parts.append(dict(support_track=g['support_track'],G_shelf_p=float(pG[1]),S_shelf_p=float(pS[1]),
                        conditional_box_gain_m2=conditional[0],conditional_shelf_gain_m2=conditional[1],predicted_semantic_gain_delta=delta))
                actual=c['S_total']-c['G_total'];error=abs(actual-reconstructed);maxerror=max(maxerror,error)
                assert error<1e-10
                details.append(dict(layout=layout,temperature=T,cue=cue,candidate_id=c['candidate_id'],
                    actual_score_difference=actual,reconstructed_difference=reconstructed,components=parts))
        by={c['candidate_id']:c for c in receipt['candidates']};g=by[receipt['G_winner']];s=by[receipt['S_winner']]
        margin=g['G_total']-s['G_total'];pressure=(s['S_total']-s['G_total'])-(g['S_total']-g['G_total'])
        out=dict(layout=layout,temperature=T,cue=cue,G_candidate=g['candidate_id'],S_candidate=s['candidate_id'],
            G_margin_over_eventual_S=margin,semantic_advantage_vs_G_winner=pressure,
            advantage_margin_ratio=pressure/margin if margin>1e-12 else None)
        for name in('G','S'):
            actual=row['selected'][name]
            for k in('object_net_area_m2','object_new_area_m2','global_net_area_m2','completed_actions','path_length_m','correct_free_coverage_increment','endpoint_reached','collisions'):
                out[name+'_'+k]=actual[k]
        table.append(out)
    with(HERE/'audit_attempt_001/all_conditions.csv').open('w',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    write(HERE/'audit_attempt_001/semantic_transfer_audit.json',dict(status='passed_all_12_conditions_48_candidate_scores',
        maximum_decomposition_error=maxerror,formula='sum_o(delta class mass * conditional route-gain contrast), same geometry and history within each layout',rows=details))
    p=HERE/'candidate_attempt_002';a=read(p/'shelf_left/T1_correct_predictions.json');b=read(p/'shelf_right/T1_correct_predictions.json')
    with np.load(p/'shelf_left/T1_correct_predictions.npz')as x,np.load(p/'shelf_right/T1_correct_predictions.npz')as y:
        parity=dict(common_score_max_absolute_difference=max(abs(c['common']-d['common'])for c,d in zip(a['candidates'],b['candidates'])),
            G_point_max_coordinate_difference_m=float(abs(x['G_points']-y['G_points']).max()),
            G_pre_history_weight_max_difference=float(abs(x['G_base_weights']-y['G_base_weights']).max()),
            G_history_active_mask_differing_points=int(np.count_nonzero((x['G_weights']>0)!=(y['G_weights']>0))),
            G_template_points=len(x['G_points']),G_max_route_gain_difference=max(abs(c['G_gain']-d['G_gain'])for c,d in zip(a['candidates'],b['candidates'])),
            raw_sensor_data_edited=False,interpretation='Cross-layout scenes are physically paired but not an exact pixel counterfactual. Tiny coordinate differences can cross original hard history thresholds. All within-layout cue comparisons share exactly one bank.')
    write(HERE/'audit_attempt_001/cross_layout_numeric_sensitivity.json',parity)
    assessment=dict(status='semantic_mechanism_effect_with_explicit_tradeoffs_not_global_superiority',actual_branches=8,actual_actions=114,
        endpoints_reached=8,collisions=0,conditions=12,independent_environment_families=1,
        supported=['Original generated targets can change solely through acquired category prior on a common within-layout bank',
            'Neutral cue exactly recovers G; exchanged cue changes effects in the paired control',
            'T16 correct-category right layout gains1.3564704m2 object net surface, reduces2 actions and improves correct-free coverage increment by3.3277870 percentage points'],
        limitations=['That same right comparison loses24.12175647m2 global net recovered surface; not Pareto improvement',
            'T16 correct-category left layout ties G',
            'T1 correct-category left layout loses1.2205876m2 object surface and needs2 more actions while global recovered surface increases',
            'All four generated endpoints are on the right side; symmetric physical construction did not create a symmetric native candidate pool',
            'Eight fixed outbound acquisitions are reused for12conditions; not12 independent policy experiments',
            'Only one cold-start decision; no full unknown-exploration efficiency or real-detector claim',
            'v1 insufficient vertical object support andv2 strict float32 byte-equality failures retained',
            'Cross-layout inputs differ by at most2 float32 ULP and48 history active points; causal comparison is within layout'],
        no_further_scene_weight_or_threshold_tuning=True,original_B8_files_unchanged=True)
    write(HERE/'audit_attempt_001/claim_evidence_assessment.json',assessment)
    for name in('candidate_extraction_freeze_attempt002.json','scene_freeze.json'):
        for path,digest in read(HERE/name)['files_sha256'].items():assert sha(path)==digest,path
    for layout in('shelf_left','shelf_right'):
        for path,digest in read(HERE/'branches_attempt_001'/layout/'freeze.json')['files_sha256'].items():assert sha(path)==digest,path
    files=list(HERE.rglob('*.json'))+list(HERE.rglob('*.py'))+list(HERE.rglob('*.csv'))
    files=[p for p in files if p.name!='validation.json']
    write(HERE/'validation.json',dict(status='completed_all_native_branches_conditions_and_gain_identity',
        files_sha256={str(p):sha(p)for p in sorted(files)},all_frozen_sources_unchanged=True,
        no_full_exploration_trial=True,max_gain_identity_error=maxerror))
    print(json.dumps(dict(status=assessment['status'],maxerror=maxerror,numeric_parity=parity)),flush=True)
if __name__=='__main__':main()
