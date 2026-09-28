#!/usr/bin/env python3
"""Rescore G candidates after exactly five old paid N observations, no world."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
import argparse
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.verify_facility_v21_n_equivalence import public_configuration, initialize_runtime, sha, read
from nso.decision_replay_v13 import load_packet, array_hash
from nso.cpu_sensor_contract_v10 import json_value, digest


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(json_value(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def run(source, output):
    if any((output / name).exists() for name in ('result.json', 'provenance.json', 'sources.zip')):
        raise FileExistsError('Refusing to overwrite a previous warmup5 audit')
    manifest = read(source / 'manifest.json')
    if manifest['status'] != 'complete' or manifest['mode'] != 'N':
        raise ValueError('Complete V20 N history required')
    inventory = read(source / 'artifact_hashes.json')
    protocol_path = ROOT / 'configs/virtual3d/facility_joint_budget_v21_probe.json'
    protocol = read(protocol_path)
    sources = {str(path.relative_to(ROOT)): sha(path)
        for folder in ('env', 'nso', 'utils') for path in sorted((ROOT / folder).glob('*.py'))}
    for path in (Path(__file__).resolve(), protocol_path, ROOT / 'scripts/verify_facility_v21_n_equivalence.py'):
        sources[str(path.relative_to(ROOT))] = sha(path)
    with zipfile.ZipFile(output / 'sources.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for name in sources:
            archive.write(ROOT / name, name)
    inputs = {'manifest.json': sha(source / 'manifest.json'),
              'artifact_hashes.json': sha(source / 'artifact_hashes.json')}

    def saved_packet(case, step):
        if not 0 <= step <= 5:
            raise ValueError('Only initial through paid action5 may be loaded')
        name = f'case_{case["index"]:02d}/packets/{step:04d}.npz'
        path = source / name
        if sha(path) != inventory[name]:
            raise ValueError('Old sensor packet changed')
        inputs[name] = sha(path)
        return load_packet(path)

    summaries, details = [], {}
    for parent in ('D19-P00', 'D19-P01'):
        case = next(c for c in manifest['cases'] if c['parent'] == parent and c['assignment'] == 'A_complex_B_simple')
        reference_name = f'references/{parent}_{case["assignment"]}_2026.npz'
        reference = Path(manifest['reference_source_root']) / reference_name
        if sha(reference) != manifest['reference_sha256'][reference_name]:
            raise ValueError('Old public configuration source changed')
        public, config = public_configuration(reference)
        shape = (round(config.height_m / config.resolution_m), round(config.width_m / config.resolution_m))
        first = saved_packet(case, 0)
        if first.sha256() != case['initial_packet_sha256']:
            raise ValueError('Initial packet differs')
        runtime = initialize_runtime(first, config, shape, case, protocol)
        backend = runtime.components._cpu_backend
        receipts = []
        for step in range(1, 6):
            # Propose before opening the next recorded sensor packet.
            proposed = runtime.next_local_action(0)
            measured = saved_packet(case, step)
            if proposed != measured.action or measured.action_id != step:
                raise ValueError('Independent runtime differs from the old N prefix')
            runtime.observe(0, step, None, None, None, sensor_packet=measured)
            receipts.append(dict(action_id=step, proposed_action=proposed, matched_saved_action=True,
                observed_pose=[*measured.position, measured.heading], packet_sha256=measured.sha256()))
        state = runtime.states[0]
        if state['pending'] is not None or state['packet'].action_id != 5 or state['closed']:
            raise ValueError('Expected complete paid observation boundary at step5')
        scene = backend.scenes[0]
        mapper = state['mapper']
        before_belief = array_hash(mapper.belief)
        before_frames = mapper.frames
        before_ledger = scene['ledger'].remaining_budget
        before_coverage = digest(scene['coverage_v21'].snapshot())
        runtime.args.cpu_score_mode = 'G'
        backend.args.cpu_score_mode = 'G'
        backend.select_target(0)
        selection = scene['last_selection']
        if (array_hash(mapper.belief) != before_belief or mapper.frames != before_frames
                or scene['ledger'].remaining_budget != before_ledger
                or digest(scene['coverage_v21'].snapshot()) != before_coverage):
            raise ValueError('Candidate rescore changed paid/map/coverage state')
        assets = []
        for index, asset in enumerate(scene['assets']):
            low, high = np.asarray(asset['observed_low']), np.asarray(asset['observed_high'])
            assets.append(dict(asset_index=index, marked_points=asset['marked_points'],
                class_vote=asset['class_vote'], observed_low=low, observed_high=high,
                observed_center=(low + high) / 2.))
        marked = sorted((a for a in assets if a['marked_points'] > 0), key=lambda a: a['observed_center'][0])
        if len(marked) != 2:
            raise ValueError('A/B association requires exactly two observed marked assets')
        roles = {marked[0]['asset_index']: 'A_observed_left_marked',
                 marked[1]['asset_index']: 'B_observed_right_marked'}
        candidates = []
        selected_id = None if selection['selected'] is None else selection['selected']['candidate_id']
        for route, score in zip(selection['candidates'], selection['score_audit']):
            index = route['candidate_id']
            budget = score['v21_coverage']['coverage_budget']
            terms = score['v19_task_proxy']
            quality = terms['observed_direction_term'] + terms['observed_precision_term']
            asset_index = route.get('asset_index')
            candidates.append(dict(candidate_id=index, group=route['group'], pose=route['pose'],
                asset_index=asset_index, observed_marked_role=roles.get(asset_index),
                association_uses_hidden_geometry=False,
                outbound_cost=route['outbound_cost'], return_cost=route['return_cost'], cost=route['cost'],
                N_score=selection['scores']['N'][index], G_score=selection['scores']['G'][index],
                direction_quality_term=terms['observed_direction_term'],
                precision_quality_term=terms['observed_precision_term'], total_quality_term=quality,
                budget_allowed=budget['allowed'], budget_reason=budget['reason'], coverage_budget=budget,
                participates_in_G_competition=bool(budget['allowed'] and selection['scores']['G'][index] > 0),
                selected_in_diagnostic=index == selected_id, actual_candidate_executed=False))
        groups = {role: [c for c in candidates if c['observed_marked_role'] == role] for role in roles.values()}
        row = dict(parent=parent, case_index=case['index'], assignment=case['assignment'],
            consumed_packet_ids=list(range(6)), matched_paid_actions=5, prefix_receipts=receipts,
            remaining_budget=before_ledger, belief_unchanged_by_rescore=True,
            paid_coverage_state_unchanged_by_rescore=True, coverage_state=selection['coverage_state'],
            observed_assets=assets, association_rule='A/B are observed left/right marked clusters; no world bounds are read',
            candidate_count=len(candidates), selected_candidate_id=selected_id,
            effective_objective=selection['effective_objective'], coverage_pressure=selection['coverage_pressure'],
            marked_role_summary={role: dict(candidate_count=len(rows),
                budget_admitted=sum(c['budget_allowed'] for c in rows),
                G_admitted=sum(c['participates_in_G_competition'] for c in rows),
                positive_quality_candidates=sum(c['total_quality_term'] > 0 for c in rows)) for role, rows in groups.items()},
            candidates=candidates, public_configuration_sha256=digest(public),
            public_configuration_source=str(reference), reference_file_sha256=sha(reference))
        summaries.append(row)
        details[parent] = selection
        print(parent, json.dumps(json_value(row['marked_role_summary']), sort_keys=True), flush=True)
    for name, wanted in sources.items():
        if sha(ROOT / name) != wanted:
            raise ValueError('Source changed during bounded audit: ' + name)
    write_new(output / 'result.json', dict(status='complete', cases=summaries,
        task='G candidate-only rescore after exactly five old paid N observations',
        saved_paid_observations_consumed=10, new_physical_actions=0, new_sensor_frames=0,
        candidate_installed_in_runtime=False, candidate_action_executed=False,
        future_sensor_packet_loaded=False, evaluator_or_world_instantiated=False,
        public_cache_access='metadata.signature_payload.config only; no reference geometry arrays',
        semantic_information_gain_proven=False, final_task_feasibility_proven=False))
    write_new(output / 'full_selections.json', details)
    write_new(output / 'provenance.json', dict(source_root=str(source), inputs_sha256=inputs,
        source_sha256=sources, excluded_old_data=['case result metrics', 'packets after action5', 'reference geometry arrays'],
        current_running_physical_batch_untouched=True))
    write_new(output / 'artifact_hashes.json', {p.name: sha(p) for p in sorted(output.iterdir())
        if p.is_file() and p.name != 'artifact_hashes.json'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    run(args.source.resolve(), output)
