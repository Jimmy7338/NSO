#!/usr/bin/env python3
"""Independent immutable-receipt and curve audit; no actors or source geometry."""
import csv, hashlib, json, math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
STATES={'shelf_left':[24,54,109],'shelf_right':[24,53,110]}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def verify_resource(r):
    assert sha(r['path'])==r['sha256'],r['path']
    return read(r['path'])
def main():
    frozen=read(ROOT/'replay/freeze.json')
    assert all(sha(p)==s for p,s in frozen['files_sha256'].items()),'frozen input changed'
    states=[]; rows=[]
    for layout,paid_values in STATES.items():
        for paid in paid_values:
            folder=ROOT/'replay'/f'{layout}_{paid}_attempt_001'
            result=read(folder/'result.json')
            assert result['status']=='completed_fixed_prefix_observed_depth_profile'
            assert result['layout']==layout and result['state_action']==paid
            assert result['new_candidate_actions']==result['new_candidate_scores']==0
            assert result['parameter_update_applied'] is False and result['hardware'] is False
            profile=verify_resource(result['profile']);parity=verify_resource(result['baseline_parity'])
            verify_resource(result['observed_input_hashes']);verify_resource(result['freeze'])
            assert parity['exact_frozen_B6_AB_arrays'] and parity['profile_did_not_mutate_mapper_or_model']
            assert parity['G_S_geometry_shared'] and parity['no_new_candidate_scores_computed']
            assert profile['constants']['heuristic_nll_band']==1
            assert profile['constants']['distance_clip_m']==.15
            assert profile['constants']['gibbs_scale_m']==.06
            state=dict(layout=layout,paid_actions=paid,result_path=str(folder/'result.json'),result_sha256=sha(folder/'result.json'),objects=[])
            for obj in profile['objects']:
                row=dict(layout=layout,paid_actions=paid,support_track=obj['support_track'],status=obj['status'],current_depth_m=obj['current_depth_m'],anchor_low_m=obj['anchor_low_m'],anchor_high_m=obj['anchor_high_m'],observed_span_m=obj['observed_span_m'],adopted=False,proposed_depth_m=obj['current_depth_m'],reason=obj['status'],fixed_ray_count=0)
                if obj['status']=='completed_acquired_depth_profile':
                    ds=obj['depths_m'];surface=obj['surface_mean_clipped_distance_squared_m2'];free=obj['free_mean_clipped_depth_gap_squared_m2'];curve=obj['neutral_mixture_nll'];selection=obj['selection']
                    assert len(ds)==len(surface)==len(free)==len(curve) and ds==sorted(set(ds))
                    assert obj['shape_mixture_prior']==[.5,.5] and obj['width_height_orientation_and_layers_unchanged']
                    assert obj['fixed_ray_audit']['denominator_fixed_across_depth_and_hypotheses']
                    assert all(0<=x<=.15**2+1e-12 for r in surface+free for x in r)
                    recomputed=[]
                    for ss,ff in zip(surface,free):
                        logs=[-(s+f)/(2*.06**2) for s,f in zip(ss,ff)]
                        m=max(logs);recomputed.append(-(m+math.log(sum(math.exp(x-m) for x in logs)))+math.log(2))
                    assert max(abs(a-b) for a,b in zip(recomputed,curve))<1e-12
                    minimum=min(curve);pi=ds.index(obj['current_depth_m']);plausible=[i for i,j in enumerate(curve) if j<=minimum+1]
                    assert [ds[i] for i in plausible]==selection['plausible_grid_depths_m']
                    assert selection['statistical_confidence_interval'] is False and selection['exact_hidden_depth_identified'] is False
                    chosen=ds.index(selection['point_estimate_m']);minimum_indices=[i for i,j in enumerate(curve) if abs(j-minimum)<=1e-12]
                    connected=all(b==a+1 for a,b in zip(plausible,plausible[1:]))
                    if selection['adopted']:
                        assert pi not in plausible and connected and plausible[-1]<len(ds)-1
                        assert any(a>b+1e-12 for a,b in zip(free[pi],free[chosen]))
                    else:assert chosen==pi
                    row.update(adopted=selection['adopted'],proposed_depth_m=selection['point_estimate_m'],reason=selection['reason'],fixed_ray_count=obj['fixed_ray_audit']['ray_count'],ray_sha256=obj['fixed_ray_audit']['rays_sha256'],depth_sha256=obj['fixed_ray_audit']['depths_sha256'],curve_min_nll=minimum,curve_max_nll=max(curve),curve_range_nll=max(curve)-minimum,prior_minus_min_nll=curve[pi]-minimum,minimizer_depths_m=[ds[i] for i in minimum_indices],plausible_grid_interval_m=selection['plausible_grid_interval_m'],plausible_depth_count=len(plausible),total_depth_count=len(ds),prior_in_band=pi in plausible,band_connected=connected,artificial_upper_in_band=selection['artificial_upper_boundary_in_band'],prior_surface_m2=surface[pi],prior_free_m2=free[pi],minimizer_surface_m2=surface[minimum_indices[0]],minimizer_free_m2=free[minimum_indices[0]],max_strict_free_hits=max(ev['strict_free_hits'] for pair in obj['per_shape_evidence'] for ev in pair),max_measured_vetoes=max(ev['measured_surface_vetoes'] for pair in obj['per_shape_evidence'] for ev in pair))
                rows.append(row);state['objects'].append(row)
            states.append(state)
    profiled=[r for r in rows if r['status']=='completed_acquired_depth_profile'];adopted=[r for r in rows if r['adopted']]
    assessment=dict(schema='B7_independent_depth_profile_summary_v1',frozen_input_sha256=sha(ROOT/'replay/freeze.json'),completed_states=len(states),object_states=len(rows),profiled_object_states=len(profiled),retained_without_profile=len(rows)-len(profiled),proposed_depth_updates=len(adopted),model_updates_applied=0,new_candidate_scores=0,new_candidate_actions=0,all_frozen_inputs_intact=True,all_B6_models_and_acquired_mapper_unchanged=True,all_curve_energy_and_band_checks_passed=True,states=states,interpretation='Existing six dependent development prefixes; no new physical or closed-loop evidence. Depth profile is heuristic compatibility, not a calibrated posterior or hidden-geometry accuracy.',next_step=('No updated dimension identified: preserve current model and all curves; no 66-branch rescoring justified.' if not adopted else 'Only reported proposals identified: any model update requires separate specification, freeze and full fixed-branch evaluation. No update has been applied.'))
    path=ROOT/'audit/summary.json';path.write_text(json.dumps(assessment,indent=2,allow_nan=False)+'\n')
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (ROOT/'audit/results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:v for k,v in assessment.items() if k!='states'},indent=2));print('summary_sha256',sha(path))
if __name__=='__main__':main()
