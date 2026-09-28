#!/usr/bin/env python3
"""Replay old paid observations through the new four-module representation.

Selections are diagnostics at the old checkpoints, not a new policy rollout.
No world or future candidate reward is constructed or read.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from env.virtual3d_inspection_v4 import InspectionConfigV4
from nso.cpu_four_modules_v17 import visibility_components_v17 as axis_components_v16
from nso.axis_history_view_v16 import AxisHistoryViewV16
from nso.cpu_sensor_contract_v10 import digest, json_value
from nso.decision_replay_v13 import array_hash, load_packet
from nso.hierarchical_options_v16_3 import generate_options
from nso.hierarchical_options_v16_3 import generate_options as base_options
from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
from nso.semantic_opportunities_v14 import candidate_capacity
from nso.semantic_opportunities_v17 import observed_descriptors, route_instance_features
from scripts.audit_semantic_v15_feedback_observability import sha, write, require


class LabelView:
    """Read-only label intervention; arrays and the underlying mapper are shared."""
    def __init__(self, mapper, zero=False): self.mapper, self.zero = mapper, zero
    def __getattr__(self, name): return getattr(self.mapper, name)
    def quality_evidence(self, *args, **kwargs):
        q = self.mapper.quality_evidence(*args, **kwargs)
        if q is None: return None
        labels = q['label']
        changed = np.zeros_like(labels) if self.zero else np.where(np.isin(labels, [2, 3]), 5 - labels, labels)
        return {**q, 'label': changed}


def check_checkpoint(comp, mapper, packet, original):
    b = comp._cpu_backend; state = b.scenes[0]
    comp.args.cpu_max_candidates = 5 + 5 * len(state['assets'])
    b.update_topo(0); b.select_target(0)
    selected = deepcopy(state['last_selection']); candidates = selected['candidates']
    for geom, asset in zip(selected['candidate_audit']['measured_assets'], state['assets']):
        for key in geom: require(geom[key] == json_value(asset[key]), 'candidate/scorer geometry differs')
    require(selected['candidate_audit']['observed_asset_count'] == len(state['assets']), 'asset count differs')
    attempted = state['gain'].attempted_camera_mask(mapper, state['ledger'].planning_camera_poses())
    for zero in (False, True):
        pool, _ = generate_options(LabelView(AxisHistoryViewV16(mapper, state['axis_frames']), zero), packet.position, packet.heading,
            state['ledger'].remaining_budget, state['return_anchor'],
            max_candidates=comp.args.cpu_max_candidates, coverage_strategy='total_diverse',
            coverage_slots=4, attempted_camera_mask=attempted)
        require(digest(pool) == digest(candidates), 'label intervention changed candidate pool')
    base, _ = base_options(AxisHistoryViewV16(mapper, state['axis_frames']), packet.position, packet.heading,
        state['ledger'].remaining_budget, state['return_anchor'],
        max_candidates=5+5*len(state['assets']), coverage_strategy='total_diverse',
        coverage_slots=4, attempted_camera_mask=attempted)
    require(digest(base) == digest(candidates[:len(base)]), 'V16 base candidates changed')
    for route in candidates:
        states = route['states']
        require(states[0] == [*packet.position, packet.heading], 'wrong route start')
        require(states[-1] == list(state['return_anchor']), 'wrong return anchor')
        require(route['cost'] == len(route['actions']) <= state['ledger'].remaining_budget, 'paid budget differs')
        require(states[route['outbound_cost']] == route['pose'], 'wrong staged arrival')
        require(len(states) == len(route['actions']) + 1, 'state/action count differs')
        require(route['outbound_cost'] + route['return_cost'] == route['cost'], 'return budget unreserved')
        for first, action, second in zip(states, route['actions'], states[1:]):
            r, c, h = first
            expected = [r, c, (h + (-1 if action == 'left' else 1)) % 4]
            if action == 'forward':
                dr, dc = ((-1, 0), (0, 1), (1, 0), (0, -1))[h]
                expected = [r + dr, c + dc, h]
            require(action in ('forward', 'left', 'right') and second == expected, 'nonphysical primitive')
        # Independent local footprint stencil, without the production inflation helper.
        radius = mapper.config.robot_radius_m / mapper.config.resolution_m + 2.**.5 / 2.
        bound = int(np.ceil(radius))
        for r, c, _ in states:
            for dr in range(-bound, bound + 1):
                for dc in range(-bound, bound + 1):
                    if dr * dr + dc * dc > radius * radius: continue
                    require(0 <= r+dr < mapper.shape[0] and 0 <= c+dc < mapper.shape[1]
                            and mapper.belief[r+dr, c+dc] == 0, 'route footprint is not known safe')
    staged = [r for r in selected['candidate_audit']['target_audit']
              if '_corner_' in r['role'] and r['reason'] == 'selected']
    require(len(staged) == sum('_corner_' in c['group'] for c in candidates), 'staged audit count differs')
    for row in staged:
        require(row['nearest_error_m'] <= row['error_limit_m'] and row['measured_grid_visible_fraction'] > 0.,
                'staged target lacks observable geometric support')
    original_assets = state['assets']
    try:
        for intervention in ('swapped', 'zero_confidence'):
            state['assets'] = deepcopy(original_assets)
            for asset in state['assets']:
                if intervention == 'swapped': asset['class_vote'] = -asset['class_vote']
                else: asset['marked_points'] = 0
            scores, _ = b._scores_v10_1(state, candidates, attempted) if candidates else (selected['scores'], [])
            for mode in ('N', 'G', 'M'):
                require(scores[mode] == selected['scores'][mode], 'semantic intervention changed geometry score')
            require(scores['S'] == selected['scores']['X' if intervention == 'swapped' else 'G'],
                    'fixed semantic swap/fallback contract differs')
    finally: state['assets'] = original_assets
    runtime_view = SimpleNamespace(components=comp)
    assets, descriptors = observed_descriptors(runtime_view, candidates)
    features = route_instance_features(mapper, candidates, assets, descriptors)
    zero_features = route_instance_features(mapper, candidates, assets, descriptors, confidence_scale=0.)
    for row, zero in zip(features, zero_features):
        require(row['geometry'] == zero['geometry'], 'zero confidence changed feature geometry')
        require(len(row['geometry']) == len(row['semantic']) == 26, 'capacity differs')
        require(zero['semantic'][18:] == [0.] * 8 and zero['residual_confidence'] == 0., 'nonzero missing semantics')
    arrays = {name: array_hash(value) for name, value in vars(mapper).items() if isinstance(value, np.ndarray)}
    require(arrays == original['state_before']['evidence']['map_arrays'], 'map history differs')
    mesh = mapper.mesh()
    hashes = {name: array_hash(np.asarray(getattr(mesh, name))) for name in ('vertices', 'triangles', 'vertex_colors')}
    require(hashes == original['state_before']['evidence']['mesh'], 'TSDF history differs')
    return json_value(dict(action_id=packet.action_id, old_candidate_count=len(original['candidates']),
        candidate_count=len(candidates), candidates=candidates, features=features, descriptors=descriptors,
        candidate_sha256=digest(candidates), score_audit=selected['score_audit'], scores=selected['scores'],
        selected_candidate=None if selected['selected'] is None else selected['selected']['candidate_id'],
        candidate_audit=selected['candidate_audit'], shared_axis_revision=b.summary(0)['shared_axis_revision'],
        canonical_observed_assets=[{k: v for k, v in asset.items() if k not in ('points', 'bits')}
                                   for asset in assets],
        inspection_candidates=sum(c['asset_index'] is not None for c in candidates),
        corner_candidates=len(staged), base_candidates_preserved=len(base),
        apertures_corrected=sum(row['incremental_aperture']!=row['legacy_unmasked_audit']['incremental_aperture'] for row in selected['score_audit']),
        route_safety_checked=True, actual_staged_visibility_verified=False,
        label_pool_controls_passed=2, score_controls_passed=2, map_mesh_exact=True,
        executed_new_policy=False, future_rewards_read=False))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    h = json.loads((a.source / 'manifest.json').read_text())
    require(h['status'] == 'complete', 'complete input histories required')
    for n, expected in json.loads((a.source / 'artifact_hashes.json').read_text()).items():
        require(sha(a.source / n) == expected, 'input artifact changed')
    for n, expected in h['source_sha256'].items(): require(sha(ROOT / n) == expected, 'old dependency changed')
    a.output.mkdir(parents=True, exist_ok=False)
    names = set(h['source_sha256']) | {str(Path(__file__).relative_to(ROOT)),
        'nso/cpu_four_modules_v16.py', 'nso/hierarchical_options_v16.py', 'nso/observed_asset_axes_v16.py',
        'nso/semantic_taxonomy_v15.py', 'nso/axis_history_view_v16.py',
        'nso/cpu_four_modules_v16_2.py', 'nso/hierarchical_options_v16_2.py', 'nso/staged_corner_views_v16.py',
        'configs/virtual3d/staged_corner_v16_2_preflight.json', 'tests/virtual3d/test_staged_corner_v16.py', 'scripts/audit_semantic_v15_feedback_observability.py'}
    names |= {'nso/cpu_four_modules_v16_3.py','nso/hierarchical_options_v16_3.py',
        'nso/cpu_four_modules_v17.py','nso/visibility_corrected_scores_v17.py',
        'nso/observed_region_evidence_v17.py','nso/observed_continuation_v17.py',
        'nso/semantic_opportunities_v17.py'}
    frozen = {n: sha(ROOT / n) for n in sorted(names)}
    manifest = dict(status='running', source_sha256=frozen, input_root=str(a.source),
        input_inventory_sha256=sha(a.source / 'artifact_hashes.json'), new_policy_rollout=False,
        training_allowed=False, parent_layouts=1, inspection_opportunity_not_assumed=True)
    write(a.output / 'manifest.json', manifest)
    with zipfile.ZipFile(a.output / 'sources.zip', 'x', zipfile.ZIP_DEFLATED) as z:
        for n in sorted(names): z.write(ROOT / n, n)
    try:
        doc = json.loads((ROOT / h['protocol']['scene_protocol']).read_text())
        c = next(x for x in doc['contexts'] if x['id'] == h['protocol']['context'])
        config = InspectionConfigV4(**{**doc['shared_conditions'], **{k: v for k, v in c.items() if k not in ('id', 'seed')}})
        summaries = []
        for seed in h['protocol']['structure_seeds']:
            folder = a.source / f'structure_{seed}'; decisions = json.loads((folder / 'all_decisions.json').read_text())
            by_action = {}
            for d in decisions:
                if d['action_id']==26 or (d['action_id']==53 and seed in (27101,27104)):
                    by_action.setdefault(d['action_id'], []).append(d)
            bounds = decisions[0]['bounds']; shape = (bounds[1], bounds[3])
            args = SimpleNamespace(nso_backend='cpu_v10', eval=True, train_global=False,
                use_open_vocab_semantic=True, use_topo_graph=True, use_rpn_uq=True, use_igcr=True,
                cpu_score_mode='G', cpu_disable_feedback=False, cpu_max_candidates=12, cpu_coverage_slots=4,
                cpu_planner_revision='v10_3_1', cpu_semantic_source_schema='inspection_v4', cpu_measured_novelty_floor=.25)
            comp = axis_components_v16(args, shape); b = comp._cpu_backend
            mapper = ObservedRuntimeMapperV10(shape, config); records = []
            for aid, path in enumerate(sorted((folder / 'packets').glob('*.npz'))):
                if aid>max(by_action):break
                packet = load_packet(path); require(packet.action_id == aid, 'packet sequence differs')
                if aid: require(b.assess_action(0, packet.action)['allowed'], 'recorded paid action no longer safe')
                mapper.update(packet.frame, packet.scan)
                if aid == 0:
                    b.start_scene(0, mapper=mapper, packet=packet, prefix_packets=[packet],
                        total_budget=h['protocol']['total_budget'], paid_prefix_actions=0,
                        return_anchor=(*packet.position, packet.heading))
                else: b.bind_packet(0, packet)
                b.update_semantic(0)
                if aid: b.compute_reward(0)
                for old in by_action.get(aid, []):
                    records.append(check_checkpoint(comp, mapper, packet, old))
                if aid and aid % 64 == 0: print(f'structure={seed} observed_actions={aid}', flush=True)
            write(a.output / f'structure_{seed}_decisions.json', records)
            summary = dict(structure_seed=seed, observed_actions=packet.action_id, decisions=len(records),
                old_candidates=sum(r['old_candidate_count'] for r in records),
                corrected_candidates=sum(r['candidate_count'] for r in records),
                states_with_inspection_candidates=sum(r['inspection_candidates'] > 0 for r in records),
                inspection_candidates=sum(r['inspection_candidates'] for r in records),
                corner_candidates=sum(r['corner_candidates'] for r in records),
                states_with_corner_candidates=sum(r['corner_candidates'] > 0 for r in records),
                module_interfaces=sorted({x['module'] for x in b.calls}))
            summaries.append(summary); write(a.output / 'partial.json', summaries)
            print(json.dumps(summary), flush=True)
        pairing=[]
        for action_id,seeds in ((26,[27101,27102,27103,27104]),(53,[27101,27104])):
            selected_rows=[]
            for seed in seeds:
                records=json.loads((a.output/f'structure_{seed}_decisions.json').read_text())
                matching=[r for r in records if r['action_id']==action_id]
                require(len(matching)==1,'expected one frozen matching checkpoint')
                selected_rows.append(matching[0])
            first=selected_rows[0]
            require(all(r['candidate_sha256']==first['candidate_sha256'] for r in selected_rows),'shared geometry pool pairing failed')
            require(all([x['geometry'] for x in r['features']]==[x['geometry'] for x in first['features']] for r in selected_rows),'geometry feature pairing failed')
            require(all(r['scores']['G']==first['scores']['G'] for r in selected_rows),'geometry score pairing failed')
            groups={}
            for seed,row in zip(seeds,selected_rows):
                key=digest([x['semantic'] for x in row['features']]);groups.setdefault(key,[]).append(seed)
            pairing.append(dict(action_id=action_id,structures=seeds,observed_semantic_groups=list(groups.values()),
                candidates=first['candidate_count'],shared_pool_geometry_features_scores_exact=True))
        write(a.output/'paired_states.json',pairing)
        for n, expected in frozen.items(): require(sha(ROOT / n) == expected, 'source changed during preflight')
        write(a.output / 'result.json', dict(status='passed', summaries=summaries,
            new_policy_rollout=False, new_candidate_outcomes=0, semantic_efficacy_proven=False,
            candidate_geometry_and_scores_integrated=True, base_candidates_preserved=True,
            staged_candidate_physical_execution_verified=False, training_allowed=False))
        manifest['status'] = 'complete'
    except Exception as error:
        manifest.update(status='failed', error=repr(error)); raise
    finally:
        write(a.output / 'manifest.json', manifest)
        write(a.output / 'artifact_hashes.json', {q.name: sha(q) for q in sorted(a.output.iterdir())
            if q.is_file() and q.name != 'artifact_hashes.json'})


if __name__ == '__main__': main()
