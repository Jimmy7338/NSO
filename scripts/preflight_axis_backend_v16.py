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
from nso.cpu_four_modules_v16 import axis_components_v16
from nso.axis_history_view_v16 import AxisHistoryViewV16
from nso.cpu_sensor_contract_v10 import digest, json_value
from nso.decision_replay_v13 import array_hash, load_packet
from nso.hierarchical_options_v16 import generate_options
from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
from nso.semantic_opportunities_v14 import candidate_capacity, observed_descriptors, route_instance_features
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
    comp.args.cpu_max_candidates = candidate_capacity(len(state['assets']))
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
        'nso/semantic_taxonomy_v15.py', 'nso/axis_history_view_v16.py', 'scripts/audit_semantic_v15_feedback_observability.py'}
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
            for d in decisions: by_action.setdefault(d['action_id'], []).append(d)
            bounds = decisions[0]['bounds']; shape = (bounds[1], bounds[3])
            args = SimpleNamespace(nso_backend='cpu_v10', eval=True, train_global=False,
                use_open_vocab_semantic=True, use_topo_graph=True, use_rpn_uq=True, use_igcr=True,
                cpu_score_mode='G', cpu_disable_feedback=False, cpu_max_candidates=12, cpu_coverage_slots=4,
                cpu_planner_revision='v10_3_1', cpu_semantic_source_schema='inspection_v4', cpu_measured_novelty_floor=.25)
            comp = axis_components_v16(args, shape); b = comp._cpu_backend
            mapper = ObservedRuntimeMapperV10(shape, config); records = []
            for aid, path in enumerate(sorted((folder / 'packets').glob('*.npz'))):
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
            summary = dict(structure_seed=seed, observed_actions=aid, decisions=len(records),
                old_candidates=sum(r['old_candidate_count'] for r in records),
                corrected_candidates=sum(r['candidate_count'] for r in records),
                states_with_inspection_candidates=sum(r['inspection_candidates'] > 0 for r in records),
                inspection_candidates=sum(r['inspection_candidates'] for r in records),
                module_interfaces=sorted({x['module'] for x in b.calls}))
            summaries.append(summary); write(a.output / 'partial.json', summaries)
            print(json.dumps(summary), flush=True)
        for n, expected in frozen.items(): require(sha(ROOT / n) == expected, 'source changed during preflight')
        write(a.output / 'result.json', dict(status='passed', summaries=summaries,
            new_policy_rollout=False, new_candidate_outcomes=0, semantic_efficacy_proven=False,
            candidate_geometry_and_scores_integrated=True, training_allowed=False))
        manifest['status'] = 'complete'
    except Exception as error:
        manifest.update(status='failed', error=repr(error)); raise
    finally:
        write(a.output / 'manifest.json', manifest)
        write(a.output / 'artifact_hashes.json', {q.name: sha(q) for q in sorted(a.output.iterdir())
            if q.is_file() and q.name != 'artifact_hashes.json'})


if __name__ == '__main__': main()
