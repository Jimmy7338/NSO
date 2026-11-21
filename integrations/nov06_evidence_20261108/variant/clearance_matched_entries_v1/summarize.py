#!/usr/bin/env python3
"""Summarize all fixed factorial cells, including unchanged/negative outcomes."""
import collections, hashlib, json, pathlib
P=pathlib.Path(__file__).resolve().parent
load=lambda name:json.loads((P/name).read_text())
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def signature(r):
    return json.dumps({k:r[k] for k in ('group','asset_index','pose','cost','outbound_actions','return_actions')},sort_keys=True)

def brief(cell):
    routes=cell['candidates'];scores=cell['scores'];audit=cell['candidate_audit']
    ranking={mode:sorted(range(len(routes)),key=lambda i:(-scores[mode][i],routes[i]['cost'],routes[i]['candidate_id'])) for mode in ('G','S','X')}
    return dict(asset_count=cell['assets_count'],candidate_count=len(routes),choices=cell['choices'],
        ranking=ranking,G_S_scores_exact_equal=scores['G']==scores['S'],G_S_ranking_equal=ranking['G']==ranking['S'],
        nonzero_aperture_factors=sum(x!=0 for r in cell['score_audit'] for x in r['incremental_aperture']),
        nonzero_S_prior_factors=sum(r['prior_potentials']['S']!=0 for r in cell['score_audit']),
        entry_roles_selected=[r['group'] for r in routes if r['group'].startswith('asset_')],
        entry_rejections=dict(collections.Counter(r['reason'] for r in audit['target_audit'])),
        offsets=cell['offsets'])

data=load('full01_result.json');old=json.loads((P.parent/'scoring_probe_full01/result.json').read_text());states=[]
assert len(data['states'])==4
for row,previous in zip(data['states'],old['states']):
    assert row['action_id']==previous['action_id']
    cells=row['cells'];base=cells['native_original'];inst=cells['instance_original']
    for key,source in [('native_original','baseline'),('instance_original','instance_variant')]:
        # Existing immutable probe corroborates the unchanged two original cells.
        for field in ('candidates','candidate_audit','scores','score_audit','choices','assets_count'):
            assert cells[key][field]==previous[source][field],(row['action_id'],key,field)
    comparisons={}
    for group in ('native','instance'):
        original=cells[group+'_original'];scaled=cells[group+'_scaled']
        for field in ('belief_sha256','safe_sha256','camera_seen_sha256','attempted_camera_sha256'):
            assert original['candidate_audit'][field]==scaled['candidate_audit'][field]
        before={signature(r):r for r in original['candidates']};after={signature(r):r for r in scaled['candidates']}
        comparisons[group]=dict(candidate_set_exact_unchanged=before.keys()==after.keys(),
            candidate_list_exact_unchanged=original['candidates']==scaled['candidates'],
            G_S_X_scores_exact_unchanged=original['scores']==scaled['scores'],
            added=[after[k] for k in sorted(after.keys()-before.keys())],
            removed=[before[k] for k in sorted(before.keys()-after.keys())],
            choices_before=original['choices'],choices_after=scaled['choices'])
    states.append(dict(action_id=row['action_id'],remaining_budget=row['remaining_budget'],
        cells={k:brief(v) for k,v in cells.items()},scaling_effects=comparisons))
wide=load('widewall_result.json');widebrief=[]
for row in wide['cases']:
    widebrief.append(dict(radius_m=row['radius_m'],generators={k:{v:dict(
        offsets=c['offsets'],target_audit=c['audit']['target_audit'],candidate_count=len(c['routes']))
        for v,c in g.items()} for k,g in row['generators'].items()}))
freeze=load('execution_freeze.json')
for r in freeze['files']+freeze['original_source_hashes']:assert sha(r['path'])==r['sha256']
prior=json.loads((P.parent/'physical_entry_scale_audit/MANIFEST.json').read_text())
assert all(sha(P.parent/'physical_entry_scale_audit'/r['path'])==r['sha256'] for r in prior['files'])
result=dict(schema='nso.clearance_matched_entries_offline_factorial_summary.v1',status='completed_offline_review_only_not_deployed',
    profile='clearance_matched_entries_v1',formula='entry=.4+max(0,R-.2), deep=.8+max(0,R-.2)',
    justification='Retain original .2/.6m target standoff excess over circle for R>=.2; exact original at R=.2. No optimization or new scientific contribution.',
    full01_states=states,widewall=widebrief,
    conclusion=dict(widewall_physical_projection_failure_removed_in_fixture=True,
        full01_semantic_opportunity_restored=False,
        full01_all_original_scaled_candidate_lists_identical=all(v['candidate_list_exact_unchanged'] for r in states for v in r['scaling_effects'].values()),
        full01_all_original_scaled_G_S_X_scores_identical=all(v['G_S_X_scores_exact_unchanged'] for r in states for v in r['scaling_effects'].values()),
        full01_all_16_cells_zero_incremental_aperture_and_semantic_prior=all(v['nonzero_aperture_factors']==0 and v['nonzero_S_prior_factors']==0 for r in states for v in r['cells'].values()),
        performance_benefit_established=False),
    validation=dict(original_and_instance_original_match_previous_frozen_probe=True,
        all_recorded_replay_next_actions_match=True,actual_global_states=4,factorial_cells=16,
        original_sources_and_prior_seven_file_audit_unchanged=True,
        original_tolerance_circle_scores_return_budget_and_other_guards_preserved=True),
    independent_units='One historical episode, four dependent checkpoints plus one synthetic geometric fixture. No variant rollout or independent physical comparison.',
    limits=['Changed candidates are architecture changes, not original NSO equivalence.',
        'Aperture is hypothetical FOV support, not an occlusion or real visibility certificate.',
        'G/S/X use the same mask-dependent partition; G is not a semantics-free perception baseline.',
        'Historical full01 retained original legacy TF/ground repair. New mount calibration was not substituted.',
        'No future observations under variant decisions, task benefit, coverage efficiency, accuracy or safety claim.',
        'Full01 worker returned but supervisor finalization was incomplete; this diagnostic does not repair its missing supervisor result.'],
    robot_execution=False,production_modified=False,deployed=False,published=False)
(P/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps([dict(paid=r['action_id'],cells={k:dict(candidates=v['candidate_count'],assets=v['asset_count'],apertures=v['nonzero_aperture_factors'],S_prior=v['nonzero_S_prior_factors'],G_S_equal=v['G_S_scores_exact_equal']) for k,v in r['cells'].items()}) for r in states],indent=2))
